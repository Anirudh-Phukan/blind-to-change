#!/usr/bin/env python3
"""Gather tenure end events from Wikidata.

Queries the Wikidata SPARQL endpoint for every position held by a person whose
end date falls in a given window, where the person has an English Wikipedia
article, and writes the raw result rows to a JSONL file. This is the first
step of the tenures pipeline.

Usage:
    python scripts/get_tenure_events.py 01-02-2026 30-06-2026 --output tenure_events.jsonl
"""
import argparse
import json
from datetime import datetime, timedelta

import requests

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# Wikidata identifiers used below:
#   P31 = instance of, Q5 = human, P39 = position held,
#   P580 = start time and P582 = end time (qualifiers of a P39 statement),
#   P569 = date of birth, P106 = occupation, P27 = country of citizenship.
#
# Each P39 statement is one tenure, so a person can hold several. The OPTIONAL
# occupation and citizenship joins also yield one row per (tenure, occupation,
# citizenship) combination, so a tenure can appear on several lines of the
# output. Downstream scripts deduplicate them.
QUERY_TEMPLATE = """SELECT ?person ?personLabel ?birthDate ?position ?positionLabel ?startDate ?endDate ?occupationLabel ?citizenshipLabel ?sitelinks WHERE {{
  ?person wdt:P31 wd:Q5 .
  ?person wikibase:sitelinks ?sitelinks .

  ?person p:P39 ?positionStatement .
  ?positionStatement ps:P39 ?position .
  ?positionStatement pq:P582 ?endDate .
  hint:Prior hint:rangeSafe true .

  FILTER(?endDate >= "{start_date}"^^xsd:dateTime
      && ?endDate < "{end_date_exclusive}"^^xsd:dateTime)

  # Require English Wikipedia article for the person
  ?article schema:about ?person .
  ?article schema:isPartOf <https://en.wikipedia.org/> .

  OPTIONAL {{ ?positionStatement pq:P580 ?startDate . }}
  OPTIONAL {{ ?person wdt:P569 ?birthDate . }}
  OPTIONAL {{ ?person wdt:P106 ?occupation . }}
  OPTIONAL {{ ?person wdt:P27 ?citizenship . }}

  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
ORDER BY DESC(?sitelinks)"""


def to_iso_date(date_str):
    """Convert a DD-MM-YYYY date to the YYYY-MM-DD form SPARQL expects."""
    return datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y-%m-%d")


def fetch_day(day):
    """Return one SPARQL result row per match, for tenures that ended on the given day.

    The day is YYYY-MM-DD. The window runs from its midnight UTC to the next
    midnight, exclusive, so day-precision end dates fall in exactly one day.
    """
    next_day = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    query = QUERY_TEMPLATE.format(start_date=day, end_date_exclusive=next_day)
    response = requests.get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        headers={"User-Agent": "WikidataEventBot/1.0"},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["results"]["bindings"]


def fetch_tenure_events(start_date, end_date):
    """Return the rows of all tenures that ended in [start_date, end_date], day by day.

    One query per day keeps each query small enough for the endpoint's time
    limit. Rows are in day order, and by sitelinks within each day.
    """
    day = datetime.strptime(start_date, "%Y-%m-%d")
    final = datetime.strptime(end_date, "%Y-%m-%d")
    if day > final:
        raise ValueError("start_date must be on or before end_date")

    events = []
    while day <= final:
        print(f"Querying {day:%Y-%m-%d}...")
        events.extend(fetch_day(day.strftime("%Y-%m-%d")))
        day += timedelta(days=1)
    return events


def main():
    parser = argparse.ArgumentParser(description="Extract tenure end events from Wikidata")
    parser.add_argument("start_date", help="Start date (DD-MM-YYYY)")
    parser.add_argument("end_date", help="End date, inclusive (DD-MM-YYYY)")
    parser.add_argument(
        "--output",
        default="tenure_events.jsonl",
        help="Output JSONL path (default: tenure_events.jsonl)",
    )
    args = parser.parse_args()

    start_date = to_iso_date(args.start_date)
    end_date = to_iso_date(args.end_date)

    print(f"Querying events from {start_date} to {end_date}...")
    events = fetch_tenure_events(start_date, end_date)

    # Each line keeps Wikidata's binding format, e.g.
    # {"person": {"type": "uri", "value": ".../entity/Q42"}, "positionLabel": {...}, ...}
    with open(args.output, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    print(f"Saved {len(events)} events to {args.output}")


if __name__ == "__main__":
    main()
