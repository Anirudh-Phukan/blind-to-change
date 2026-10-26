#!/usr/bin/env python3
"""Fetch person and position context for every tenure in a tenure events file.

Reads the output of get_tenure_events.py, collapses its rows into one record
per tenure, and adds for each tenure the person's English Wikipedia article
and a description of the position: its Wikidata description and the plain
text of its English Wikipedia article. The probes about that tenure are later
generated from this context.

Usage:
    python scripts/get_tenures_context.py --input tenure_events.jsonl --output tenures_context.jsonl

The run makes four or more requests per tenure at about one second each. If it
is interrupted, rerun the same command with --resume to continue where it
stopped.
"""
import argparse
import json
import re
from pathlib import Path

from utils.wiki_utils import (
    WIKIPEDIA_API,
    get_json,
    get_wikidata_entity,
    get_wikipedia_page,
    get_word_count,
    load_processed_ids,
    qid_from_uri,
)

HEADERS = {"User-Agent": "PositionContextFetcher/1.0 (Educational/Research Purpose)"}
TIMEOUT = 20
# Number of Wikipedia search results tried when a position has no article of its own.
SEARCH_RESULT_LIMIT = 5


def binding_value(row, key):
    """Return the value of one field of a SPARQL result row, or None if absent."""
    return row.get(key, {}).get("value")


def to_date(value):
    """Cut a Wikidata timestamp such as "2026-03-01T00:00:00Z" down to its date."""
    return value.split("T", 1)[0] if value else None


def load_tenures(events_file):
    """Return one record per tenure in an events file, in order of first appearance.

    A tenure is identified by person, position, start date and end date. The
    events file has one row per (tenure, occupation, citizenship) combination
    (see get_tenure_events.py), so the occupations and citizenships of a
    tenure's rows are gathered into sorted lists.
    """
    tenures = {}
    with open(events_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            row = json.loads(line)
            person_uri = binding_value(row, "person")
            position_uri = binding_value(row, "position")
            name = binding_value(row, "personLabel")
            position_label = binding_value(row, "positionLabel")
            start_date = to_date(binding_value(row, "startDate"))
            end_date = to_date(binding_value(row, "endDate"))
            if not all([person_uri, position_uri, name, position_label, end_date]):
                print(f"Line {line_num}: missing required fields, skipping")
                continue

            person_qid = qid_from_uri(person_uri)
            position_qid = qid_from_uri(position_uri)
            fact_id = f"{person_qid}|{position_qid}|{start_date or ''}|{end_date}"
            record = tenures.setdefault(
                fact_id,
                {
                    "fact_id": fact_id,
                    "fact_type": "position_ended",
                    "wikidata_id": person_qid,
                    "name": name,
                    "position_id": position_qid,
                    "position_label": position_label,
                    "start_date": start_date,
                    "end_date": end_date,
                    "birth_date": to_date(binding_value(row, "birthDate")),
                    "sitelinks": int(binding_value(row, "sitelinks") or 0),
                    "occupation_labels": set(),
                    "citizenship_labels": set(),
                },
            )
            occupation = binding_value(row, "occupationLabel")
            citizenship = binding_value(row, "citizenshipLabel")
            if occupation:
                record["occupation_labels"].add(occupation)
            if citizenship:
                record["citizenship_labels"].add(citizenship)

    for record in tenures.values():
        record["occupation_labels"] = sorted(record["occupation_labels"])
        record["citizenship_labels"] = sorted(record["citizenship_labels"])
    return list(tenures.values())


def get_entity_metadata(qid):
    """Return the English Wikipedia title and English description of a Wikidata item."""
    entity = get_wikidata_entity(qid, headers=HEADERS, timeout=TIMEOUT)
    return {
        "title": entity.get("sitelinks", {}).get("enwiki", {}).get("title"),
        "description": entity.get("descriptions", {}).get("en", {}).get("value"),
    }


def get_page(title):
    """Return {"title", "extract"} of an English Wikipedia article, following redirects."""
    if not title:
        return None
    return get_wikipedia_page(title, headers=HEADERS, follow_redirects=True, timeout=TIMEOUT)


def normalize_title(text):
    """Lowercase a title and replace underscores and punctuation with spaces."""
    text = text.replace("_", " ").casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def search_wikipedia(query):
    """Return the titles of the top English Wikipedia search results for a query."""
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": SEARCH_RESULT_LIMIT,
        "srprop": "snippet",
    }
    data = get_json(WIKIPEDIA_API, params=params, headers=HEADERS, timeout=TIMEOUT)
    return [item["title"] for item in data.get("query", {}).get("search", []) if item.get("title")]


def resolve_position_context(position_label, position_wikipedia_title):
    """Find the English Wikipedia article that describes a position.

    Tries, in order: the article linked from the position's Wikidata item, an
    article whose title equals the position label, and the first search result
    for the label that has an article. The source field records which of the
    three ("wikidata_enwiki", "exact_title", "search_fallback") was used, or is
    None if none was found.
    """
    page = get_page(position_wikipedia_title)
    if page:
        return {"title": page["title"], "context": page["extract"], "source": "wikidata_enwiki"}

    # A label that is just a Wikidata ID (e.g. "Q123") means the position has
    # no English label, so there is nothing to search for.
    if position_label and not re.fullmatch(r"Q\d+", position_label.strip()):
        page = get_page(position_label)
        if page and normalize_title(page["title"]) == normalize_title(position_label):
            return {"title": page["title"], "context": page["extract"], "source": "exact_title"}

        for title in search_wikipedia(position_label):
            page = get_page(title)
            if page:
                return {"title": page["title"], "context": page["extract"], "source": "search_fallback"}

    return {"title": position_wikipedia_title, "context": None, "source": None}


def build_fact_text(record):
    """Return the one-sentence statement of the tenure's end that the probes test."""
    return (
        f"{record['name']} stopped holding the position "
        f"{record['position_label']} on {record['end_date']}."
    )


def build_relevant_information(record):
    """Return the tenure details and position context as one text block."""
    parts = [
        f"Fact type: {record['fact_type']}",
        f"Ended position: {record['position_label']}",
    ]
    if record.get("position_wikidata_description"):
        parts.append(f"Position description: {record['position_wikidata_description']}")
    if record.get("start_date"):
        parts.append(f"Position start date: {record['start_date']}")
    if record.get("end_date"):
        parts.append(f"Position end date: {record['end_date']}")
    if record.get("occupation_labels"):
        parts.append("Occupations: " + ", ".join(record["occupation_labels"]))
    if record.get("citizenship_labels"):
        parts.append("Citizenships: " + ", ".join(record["citizenship_labels"]))
    if record.get("position_context"):
        parts.append("Position context:\n" + record["position_context"])
    return "\n".join(parts)


def add_context(record):
    """Add the person and position context to a tenure record.

    Returns False, leaving the record incomplete, if the person's Wikipedia
    article cannot be fetched. A missing position article only leaves the
    position context empty.
    """
    name = record["name"]
    try:
        person_meta = get_entity_metadata(record["wikidata_id"])
        person_page = get_page(person_meta["title"])
    except Exception as e:
        print(f"  Error fetching person context for {name} ({record['wikidata_id']}): {e}")
        return False
    if not person_page:
        return False

    try:
        position_meta = get_entity_metadata(record["position_id"])
        position = resolve_position_context(record["position_label"], position_meta["title"])
    except Exception as e:
        print(
            f"  Warning: could not fetch position context for "
            f"{record['position_label']} ({record['position_id']}): {e}"
        )
        position_meta = {"title": None, "description": None}
        position = {"title": None, "context": None, "source": None}

    bio = person_page["extract"]
    record["person_wikipedia_title"] = person_meta["title"]
    record["wikipedia_bio"] = bio
    record["wikipedia_bio_word_count"] = get_word_count(bio)
    record["position_wikipedia_title"] = position["title"] or position_meta["title"]
    record["position_wikidata_description"] = position_meta["description"]
    record["position_context"] = position["context"]
    record["position_context_source"] = position["source"]
    record["position_context_word_count"] = (
        get_word_count(position["context"]) if position["context"] else 0
    )
    record["fact_text"] = build_fact_text(record)
    record["relevant_information"] = build_relevant_information(record)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fetch person and position context for the tenures in a tenure events file"
    )
    parser.add_argument(
        "--input",
        default="tenure_events.jsonl",
        help="Events JSONL from get_tenure_events.py (default: tenure_events.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="tenures_context.jsonl",
        help="Output JSONL path (default: tenures_context.jsonl)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output file, skipping tenures already in it",
    )
    args = parser.parse_args()
    output_file = Path(args.output)

    tenures = load_tenures(args.input)
    print(f"Found {len(tenures)} unique tenures")

    processed_ids = set()
    if args.resume:
        processed_ids = load_processed_ids(output_file, key="fact_id")
        print(f"Found {len(processed_ids)} already-processed tenures in {output_file}")

    pending = [record for record in tenures if record["fact_id"] not in processed_ids]
    print(f"Processing {len(pending)} remaining tenures")
    if not pending:
        print(f"\nDone! No new results to write to {output_file}")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a" if args.resume else "w", encoding="utf-8") as f:
        for i, record in enumerate(pending, 1):
            print(
                f"[{i}/{len(pending)}] Fetching context for: {record['name']} "
                f"({record['wikidata_id']}) -> {record['position_label']} ({record['position_id']})"
            )
            # Tenures whose person has no retrievable article are left out of
            # the output, so a later --resume run tries them again.
            if not add_context(record):
                print("  ✗ No Wikipedia page found for the person")
                continue

            f.write(json.dumps(record) + "\n")
            print(
                f"  ✓ Success (person bio {record['wikipedia_bio_word_count']} words, "
                f"position context {record['position_context_word_count']} words)"
            )

    print(f"\nDone! Results saved to {output_file}")


if __name__ == "__main__":
    main()
