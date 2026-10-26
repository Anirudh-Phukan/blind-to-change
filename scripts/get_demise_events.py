#!/usr/bin/env python3
"""Gather demise events from Wikidata.

Queries the Wikidata SPARQL endpoint for every person whose date of death falls
in a given window and who has an English Wikipedia article, and writes the raw
result rows to a JSONL file. This is the first step of the demises pipeline.

Usage:
    python scripts/get_demise_events.py 01-02-2026 30-06-2026 --output demise_events.jsonl
"""
import argparse
import json
from datetime import datetime

import requests

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# Wikidata identifiers used below:
#   P31 = instance of, Q5 = human, P570 = date of death,
#   P569 = date of birth, P106 = occupation, P27 = country of citizenship.
#
# The OPTIONAL occupation and citizenship joins yield one row per
# (person, occupation, citizenship) combination, so a person can appear on
# several lines of the output. Downstream scripts deduplicate by Wikidata ID.
QUERY_TEMPLATE = """SELECT ?person ?personLabel ?birthDate ?deathDate ?occupationLabel ?citizenshipLabel ?sitelinks WHERE {{
  ?person wdt:P31 wd:Q5 .
  ?person wdt:P570 ?deathDate .
  ?person wikibase:sitelinks ?sitelinks .

  FILTER(?deathDate >= "{start_date}"^^xsd:dateTime
      && ?deathDate <= "{end_date}"^^xsd:dateTime)

  # Require English Wikipedia article
  ?article schema:about ?person .
  ?article schema:isPartOf <https://en.wikipedia.org/> .

  OPTIONAL {{ ?person wdt:P569 ?birthDate . }}
  OPTIONAL {{ ?person wdt:P106 ?occupation . }}
  OPTIONAL {{ ?person wdt:P27 ?citizenship . }}

  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
}}
ORDER BY DESC(?sitelinks)"""


def to_iso_date(date_str):
    """Convert a DD-MM-YYYY date to the YYYY-MM-DD form SPARQL expects."""
    return datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y-%m-%d")


def fetch_demise_events(start_date, end_date):
    """Return one SPARQL result row per match, for deaths in [start_date, end_date].

    Both dates are YYYY-MM-DD and are read as midnight UTC. Wikidata stores
    day-precision dates at midnight too, so the end date is inclusive.
    """
    query = QUERY_TEMPLATE.format(start_date=start_date, end_date=end_date)
    response = requests.get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        headers={"User-Agent": "WikidataEventBot/1.0"},
    )
    response.raise_for_status()
    return response.json()["results"]["bindings"]


def main():
    parser = argparse.ArgumentParser(description="Extract demise events from Wikidata")
    parser.add_argument("start_date", help="Start date (DD-MM-YYYY)")
    parser.add_argument("end_date", help="End date, inclusive (DD-MM-YYYY)")
    parser.add_argument(
        "--output",
        default="demise_events.jsonl",
        help="Output JSONL path (default: demise_events.jsonl)",
    )
    args = parser.parse_args()

    start_date = to_iso_date(args.start_date)
    end_date = to_iso_date(args.end_date)

    print(f"Querying events from {start_date} to {end_date}...")
    events = fetch_demise_events(start_date, end_date)

    # Each line keeps Wikidata's binding format, e.g.
    # {"person": {"type": "uri", "value": ".../entity/Q42"}, "personLabel": {...}, ...}
    with open(args.output, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    print(f"Saved {len(events)} events to {args.output}")


if __name__ == "__main__":
    main()
