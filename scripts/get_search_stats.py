#!/usr/bin/env python3
"""Print how much a model searched in the shared harness.

Reads a trace file from generate_answers_shared.py and counts, for every
answer, its search queries (web_search calls) and its search turns (model
calls that issue at least one search; a model call can issue several
searches at once). Prints the minimum, quartiles, mean and maximum of both
counts over all answers, and the share of answers with no search.

Usage:
    python scripts/get_search_stats.py --trace-file shared_answers_gpt-5.4_high_trace.jsonl
"""
import argparse
import json
import statistics


def search_counts(steps):
    """Return (search turns, search queries) of one answer's trace steps."""
    turns = queries = 0
    previous_was_search = False
    for step in steps:
        is_search = step["input"].startswith("web_search(")
        if is_search:
            queries += 1
            # Searches that follow each other were issued by the same model call.
            if not previous_was_search:
                turns += 1
        previous_was_search = is_search
    return turns, queries


def summarize(values):
    """Return min, P25, median, mean, P75 and max of a list of counts."""
    p25, median, p75 = statistics.quantiles(values, n=4, method="inclusive")
    return [min(values), p25, median, statistics.mean(values), p75, max(values)]


def format_number(value):
    """Return an integer without decimals and any other number with two."""
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(description="Print how much a model searched in the shared harness")
    parser.add_argument(
        "--trace-file", required=True, help="Trace JSONL from generate_answers_shared.py"
    )
    args = parser.parse_args()

    with open(args.trace_file, "r", encoding="utf-8") as f:
        counts = [search_counts(json.loads(line)["steps"]) for line in f if line.strip()]
    turns = [t for t, _ in counts]
    queries = [q for _, q in counts]

    print(f"Answers: {len(counts)}")
    print(f"No search: {queries.count(0) / len(queries) * 100:.1f}%")
    print(f"{'':<16}" + "".join(f"{h:>8}" for h in ("Min", "P25", "Median", "Mean", "P75", "Max")))
    for name, values in (("Search turns", turns), ("Search queries", queries)):
        print(f"{name:<16}" + "".join(f"{format_number(v):>8}" for v in summarize(values)))


if __name__ == "__main__":
    main()
