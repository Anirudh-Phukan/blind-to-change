#!/usr/bin/env python3
"""Keep the indirect probes on which an agent relies on the outdated fact.

Reads the judgements of an agent's no-search responses to the candidate
indirect probes (three responses per probe) and keeps a probe only if at least
--min-incorrect-judgements of them were judged "incorrect". The default of 3
keeps a probe only if all three were. Here "incorrect" means that the
response recommended the affected person as if the event had not happened.
Without search that is the expected behaviour: it shows that the agent would
recommend this person from memory, so the probe can later test whether the
agent discovers the event once search is enabled.

The filter is run separately for each agent, and works for both demises and
tenures. The kept probes are written grouped by event, in the same format as
the probe generation output.

Usage:
    python scripts/filter_indirect_probes.py --input answers_gpt-5.4_judgements.jsonl \\
        --output filtered_indirect_probes_gpt-5.4.jsonl
"""
import argparse
import json


def load_retained_probes(judgements_file, min_incorrect, num_judgements_considered=None):
    """Return {(wikidata_id, name, fact_text): [probe, ...]} for the retained probes.

    The input has one line per probe. fact_text is the statement of the ended
    position for tenures and None for demises. It is part of the key because
    one person can have several ended positions, each with its own probes.
    """
    retained = {}
    with open(judgements_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} has malformed JSON, skipping: {e}")
                continue

            missing = [k for k in ("wikidata_id", "name", "question", "judgements") if k not in entry]
            if missing:
                print(f"Warning: Line {line_num} missing fields {missing}, skipping")
                continue

            # With num_judgements_considered=None the slice covers all judgements.
            judgements = entry["judgements"][:num_judgements_considered]
            num_incorrect = sum(judgement == "incorrect" for judgement in judgements)
            if num_incorrect >= min_incorrect:
                key = (entry["wikidata_id"], entry["name"], entry.get("fact_text"))
                retained.setdefault(key, []).append(entry["question"])
    return retained


def write_retained_probes(retained, output_file):
    """Write one record per event, in the format of the probe generation output."""
    with open(output_file, "w", encoding="utf-8") as f:
        for (wikidata_id, name, fact_text), probes in retained.items():
            record = {
                "wikidata_id": wikidata_id,
                "name": name,
                # The judgements file carries only the probe text, so the other
                # per-probe fields of the generation output are not included.
                "questions": [{"question": probe} for probe in probes],
                "success": True,
                "error": None,
            }
            if fact_text is not None:
                record["fact_text"] = fact_text
            f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Keep the indirect probes whose no-search responses were all judged incorrect"
    )
    parser.add_argument(
        "--input", required=True, help="Judgements JSONL of an agent's no-search responses"
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--min-incorrect-judgements",
        type=int,
        default=3,
        help="Number of incorrect judgements a probe needs to be kept (default: 3)",
    )
    parser.add_argument(
        "--num-judgements-considered",
        type=int,
        help="Count only the first N judgements of each probe (default: all)",
    )
    args = parser.parse_args()

    if args.min_incorrect_judgements < 1:
        parser.error("--min-incorrect-judgements must be >= 1")
    if args.num_judgements_considered is not None and args.num_judgements_considered < 1:
        parser.error("--num-judgements-considered must be >= 1")

    print(f"Reading from {args.input}...")
    retained = load_retained_probes(
        args.input, args.min_incorrect_judgements, args.num_judgements_considered
    )

    num_probes = sum(len(probes) for probes in retained.values())
    considered = (
        f" among the first {args.num_judgements_considered} judgements"
        if args.num_judgements_considered is not None
        else ""
    )
    print(
        f"Found {num_probes} questions with at least "
        f"{args.min_incorrect_judgements} incorrect judgements{considered} "
        f"across {len(retained)} people"
    )

    write_retained_probes(retained, args.output)
    print(f"Done! Output written to {args.output}")


if __name__ == "__main__":
    main()
