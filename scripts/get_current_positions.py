#!/usr/bin/env python3
"""Fetch the positions each person in a tenure events file holds on a given date.

Reads the output of get_tenure_events.py and, for every person in it, queries
Wikidata for the "position held" statements that are active on the --as-of
date (default: today), with their start and end dates and every label of each
position. A tenure whose person again holds the same position (e.g. after a
re-election) can then be filtered out.

Usage:
    python scripts/get_current_positions.py --input tenure_events.jsonl --output current_positions.jsonl

The output has one line per person, including people who currently hold no
position.
"""
import argparse
import json
import re
from datetime import date

from utils.wiki_utils import get_json, qid_from_uri

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "CurrentPositionFetcher/1.0 (Educational/Research Purpose)"}
TIMEOUT = 60
# People per SPARQL query and positions per label request. 50 is the most
# Wikidata's wbgetentities accepts.
BATCH_SIZE = 50

# Wikidata identifiers used below:
#   P39 = position held, P580 = start time and P582 = end time (qualifiers).
#
# A statement is active on the date if it has started by the end of that day
# (or has no start date) and has not ended before the day began (or has no end
# date). Deprecated statements, which Wikidata marks as wrong, are left out.
QUERY_TEMPLATE = """SELECT DISTINCT ?person ?statement ?position ?startDate ?endDate WHERE {{
  VALUES ?person {{ {people} }}
  ?person p:P39 ?statement.
  ?statement ps:P39 ?position;
             wikibase:rank ?rank.
  FILTER(?rank != wikibase:DeprecatedRank)
  OPTIONAL {{ ?statement pq:P580 ?startDate. }}
  OPTIONAL {{ ?statement pq:P582 ?endDate. }}
  FILTER(!BOUND(?startDate) || ?startDate <= "{day}T23:59:59Z"^^xsd:dateTime)
  FILTER(!BOUND(?endDate) || ?endDate >= "{day}T00:00:00Z"^^xsd:dateTime)
}}"""


def binding_value(row, key):
    """Return the value of one field of a SPARQL result row, or None if absent."""
    return row.get(key, {}).get("value")


def qid_number(qid):
    """Return the number of a Wikidata ID ("Q42" -> 42), to sort IDs numerically."""
    return int(qid[1:])


def batches(items):
    """Yield successive BATCH_SIZE-long slices of a list."""
    for start in range(0, len(items), BATCH_SIZE):
        yield items[start : start + BATCH_SIZE]


def load_people(events_file):
    """Return {wikidata_id: name} for the people in an events file, in order of first appearance."""
    people = {}
    with open(events_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = qid_from_uri(binding_value(row, "person"))
            people.setdefault(qid, binding_value(row, "personLabel"))
    return people


def fetch_current_statements(person_qids, as_of):
    """Return {person_qid: [statement, ...]} for the position statements active on as_of.

    Each statement is a dict with the statement URI, the position's ID, and its
    start and end timestamps (None if missing). A person's statements are
    sorted by position ID, then start date, end date and statement URI.
    """
    statements = {qid: [] for qid in person_qids}
    for i, batch in enumerate(batches(person_qids), 1):
        print(f"Fetching current positions, batch {i} ({len(batch)} people)")
        query = QUERY_TEMPLATE.format(
            people=" ".join(f"wd:{qid}" for qid in batch), day=as_of.isoformat()
        )
        data = get_json(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        for row in data["results"]["bindings"]:
            position_qid = qid_from_uri(binding_value(row, "position"))
            # An "unknown value" position comes back as a blank-node URI, not an item.
            if not re.fullmatch(r"Q\d+", position_qid):
                continue
            statements[qid_from_uri(binding_value(row, "person"))].append(
                {
                    "statement": binding_value(row, "statement"),
                    "position_qid": position_qid,
                    "start_date": binding_value(row, "startDate"),
                    "end_date": binding_value(row, "endDate"),
                }
            )

    for person_statements in statements.values():
        person_statements.sort(
            key=lambda s: (
                qid_number(s["position_qid"]),
                s["start_date"] or "",
                s["end_date"] or "",
                s["statement"],
            )
        )
    return statements


def fetch_labels(qids):
    """Return {qid: {language: label}} with every label of each Wikidata item.

    All languages are kept, so that a position without an English label can
    still be recognised.
    """
    labels = {}
    for i, batch in enumerate(batches(qids), 1):
        print(f"Fetching position labels, batch {i} ({len(batch)} positions)")
        data = get_json(
            WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "format": "json",
                "ids": "|".join(batch),
                "props": "labels",
            },
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        for qid in batch:
            entity_labels = data["entities"].get(qid, {}).get("labels", {})
            labels[qid] = {
                language: entity_labels[language]["value"] for language in sorted(entity_labels)
            }
    return labels


def main():
    parser = argparse.ArgumentParser(
        description="Fetch the positions held on a date by the people in a tenure events file"
    )
    parser.add_argument(
        "--input",
        default="tenure_events.jsonl",
        help="Events JSONL from get_tenure_events.py (default: tenure_events.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="current_positions.jsonl",
        help="Output JSONL path (default: current_positions.jsonl)",
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date.today(),
        help="Date (YYYY-MM-DD) on which a position must be held (default: today)",
    )
    args = parser.parse_args()

    people = load_people(args.input)
    print(f"Found {len(people)} unique persons")

    statements = fetch_current_statements(list(people), args.as_of)
    position_qids = sorted(
        {s["position_qid"] for person_statements in statements.values() for s in person_statements},
        key=qid_number,
    )
    print(f"Found {len(position_qids)} unique current positions")
    labels = fetch_labels(position_qids)

    with open(args.output, "w", encoding="utf-8") as f:
        for qid, name in people.items():
            current_qids = sorted({s["position_qid"] for s in statements[qid]}, key=qid_number)
            record = {
                "wikidata_id": qid,
                "name": name,
                "as_of": args.as_of.isoformat(),
                "current_position_qids": current_qids,
                "current_position_labels": {pos: labels[pos] for pos in current_qids},
                "current_position_statements": statements[qid],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    holding_none = sum(not person_statements for person_statements in statements.values())
    print(f"\nDone! Results for {len(people)} persons saved to {args.output}")
    print(f"Persons holding no position on {args.as_of}: {holding_none}")


if __name__ == "__main__":
    main()
