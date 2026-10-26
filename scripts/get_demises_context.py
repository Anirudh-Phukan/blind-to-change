#!/usr/bin/env python3
"""Fetch the Wikipedia biography of every person in a demise events file.

Reads the output of get_demise_events.py, looks up each person's English
Wikipedia article through their Wikidata ID, and writes the article's plain
text and word count to a JSONL file. The biography is the context from which
the probes about that person are later generated.

Usage:
    python scripts/get_demises_context.py --input demise_events.jsonl --output demises_context.jsonl

The run makes two requests per person at about one second each. If it is
interrupted, rerun the same command with --resume to continue where it stopped.
"""
import argparse
import json
from pathlib import Path

from utils.wiki_utils import (
    get_wikidata_entity,
    get_wikipedia_page,
    get_word_count,
    load_processed_ids,
    qid_from_uri,
)

HEADERS = {"User-Agent": "WikipediaBioFetcher/1.0 (Educational/Research Purpose)"}


def load_unique_persons(events_file):
    """Return {wikidata_id: name} for the people in an events file.

    The events file has several rows per person (see get_demise_events.py), so
    this collapses them to one entry each.
    """
    persons = {}
    with open(events_file, "r", encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            uri = event.get("person", {}).get("value")
            name = event.get("personLabel", {}).get("value")
            if uri and name:
                persons[qid_from_uri(uri)] = name
    return persons


def get_wikipedia_title(qid):
    """Return the title of a Wikidata item's English Wikipedia article, or None."""
    try:
        entity = get_wikidata_entity(qid, headers=HEADERS)
        return entity.get("sitelinks", {}).get("enwiki", {}).get("title")
    except Exception as e:
        print(f"  Error getting Wikipedia title for {qid}: {e}")
        return None


def get_wiki_bio(qid, name):
    """Return the full plain text of a person's English Wikipedia article, or None."""
    title = get_wikipedia_title(qid)
    if not title:
        return None

    try:
        page = get_wikipedia_page(title, headers=HEADERS)
        return page["extract"] if page else None
    except Exception as e:
        print(f"  Error fetching bio for {name} ({qid}): {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Wikipedia biographies for the people in a demise events file"
    )
    parser.add_argument(
        "--input",
        default="demise_events.jsonl",
        help="Events JSONL from get_demise_events.py (default: demise_events.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="demises_context.jsonl",
        help="Output JSONL path (default: demises_context.jsonl)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output file, skipping people already in it",
    )
    args = parser.parse_args()
    output_file = Path(args.output)

    persons = load_unique_persons(args.input)
    print(f"Found {len(persons)} unique persons")

    processed_qids = set()
    if args.resume:
        processed_qids = load_processed_ids(output_file)
        print(f"Found {len(processed_qids)} already-processed persons in {output_file}")

    # Process people in alphabetical order of name.
    pending = [
        (qid, name)
        for qid, name in sorted(persons.items(), key=lambda item: item[1])
        if qid not in processed_qids
    ]
    print(f"Processing {len(pending)} remaining persons")
    if not pending:
        print(f"\nDone! No new results to write to {output_file}")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a" if args.resume else "w", encoding="utf-8") as f:
        for i, (qid, name) in enumerate(pending, 1):
            print(f"[{i}/{len(pending)}] Fetching bio for: {name} ({qid})")
            bio = get_wiki_bio(qid, name)

            # People without a retrievable article are left out of the output,
            # so a later --resume run tries them again.
            if not bio:
                print("  ✗ No Wikipedia page found")
                continue

            word_count = get_word_count(bio)
            record = {
                "wikidata_id": qid,
                "name": name,
                "wikipedia_bio": bio,
                "wikipedia_bio_word_count": word_count,
            }
            f.write(json.dumps(record) + "\n")
            print(f"  ✓ Success ({word_count} words)")

    print(f"\nDone! Results saved to {output_file}")


if __name__ == "__main__":
    main()
