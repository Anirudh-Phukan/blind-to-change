#!/usr/bin/env python3
"""Drop tenure events whose person holds the position again, or that lack English labels.

Reads the output of get_tenure_events.py and of get_current_positions.py and
removes every tenure for which:
  - the position has no English label (its label is just a Wikidata ID such as
    "Q123"), so probes about it cannot be written;
  - the person has no English label, for the same reason;
  - the person holds the same position again, i.e. has a current statement for
    it that started after the tenure did (typically a re-election);
  - the person holds the next term of the position, i.e. a current position
    whose English label is the same once ordinals are removed (e.g. "Member of
    the 6th Scottish Parliament" -> "Member of the 7th Scottish Parliament").

A tenure that has ended but been resumed does not change who should be
recommended, so it cannot be used as an update. The remaining rows are written
unchanged, in the format of get_tenure_events.py.

Usage:
    python scripts/filter_tenure_events.py --input tenure_events.jsonl \
        --current-positions current_positions.jsonl --output tenure_events_filtered.jsonl
"""
import argparse
import json
import re
from collections import Counter

from utils.wiki_utils import qid_from_uri

QID_PATTERN = re.compile(r"Q\d+")
# Ordinal numbers such as "6th" or "59th", which number the terms of an office.
ORDINAL_PATTERN = re.compile(r"\b\d+(st|nd|rd|th)\b")


def binding_value(row, key):
    """Return the value of one field of a SPARQL result row, or None if absent."""
    return row.get(key, {}).get("value")


def to_date(value):
    """Cut a Wikidata timestamp such as "2026-03-01T00:00:00Z" down to its date."""
    return value.split("T", 1)[0] if value else None


def without_ordinals(label):
    """Return a label lowercased, with ordinal numbers removed and spaces collapsed."""
    return " ".join(ORDINAL_PATTERN.sub(" ", label.casefold()).split())


def load_current_positions(path):
    """Return {wikidata_id: record} from the output of get_current_positions.py."""
    with open(path, "r", encoding="utf-8") as f:
        return {record["wikidata_id"]: record for record in map(json.loads, f)}


def drop_reason(row, current_positions):
    """Return why a tenure event should be dropped, or None to keep it."""
    person_label = binding_value(row, "personLabel")
    position_label = binding_value(row, "positionLabel")
    if QID_PATTERN.fullmatch(position_label):
        return "position has no English label"
    if QID_PATTERN.fullmatch(person_label):
        return "person has no English label"

    person = current_positions.get(qid_from_uri(binding_value(row, "person")))
    if not person:
        return None
    position_qid = qid_from_uri(binding_value(row, "position"))
    start_date = to_date(binding_value(row, "startDate"))

    # A current statement for the same position that started after this tenure
    # is a new term. One that started earlier, or has no start date, is usually
    # an older entry nobody closed, so it is not counted. Without a start date
    # for the tenure, the two cannot be told apart.
    if start_date and any(
        statement["position_qid"] == position_qid
        and (to_date(statement["start_date"]) or "") > start_date
        for statement in person["current_position_statements"]
    ):
        return "holds the same position again"

    target = without_ordinals(position_label)
    for qid, labels in person["current_position_labels"].items():
        if qid != position_qid and without_ordinals(labels.get("en", "")) == target:
            return "holds the next term of the position"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Drop tenure events whose person holds the position again"
    )
    parser.add_argument(
        "--input",
        default="tenure_events.jsonl",
        help="Events JSONL from get_tenure_events.py (default: tenure_events.jsonl)",
    )
    parser.add_argument(
        "--current-positions",
        default="current_positions.jsonl",
        help="Output of get_current_positions.py for the same events "
        "(default: current_positions.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="tenure_events_filtered.jsonl",
        help="Output JSONL path (default: tenure_events_filtered.jsonl)",
    )
    args = parser.parse_args()

    current_positions = load_current_positions(args.current_positions)

    # The events file has several rows per tenure (see get_tenure_events.py);
    # every row of a tenure gets the same verdict, and counts are per tenure.
    verdicts = {}
    kept_rows = 0
    with open(args.input, "r", encoding="utf-8") as f_in, open(
        args.output, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            row = json.loads(line)
            tenure = (
                binding_value(row, "person"),
                binding_value(row, "position"),
                binding_value(row, "startDate"),
                binding_value(row, "endDate"),
            )
            if tenure not in verdicts:
                verdicts[tenure] = drop_reason(row, current_positions)
            if verdicts[tenure] is None:
                f_out.write(line if line.endswith("\n") else line + "\n")
                kept_rows += 1

    reasons = Counter(verdicts.values())
    kept = reasons.pop(None, 0)
    print(f"Read {len(verdicts)} unique tenures")
    for reason, count in reasons.most_common():
        print(f"  Dropped {count}: {reason}")
    print(f"\nDone! Kept {kept} tenures ({kept_rows} rows) in {args.output}")


if __name__ == "__main__":
    main()
