#!/usr/bin/env python3
"""Keep only the people whose Wikipedia biography is long enough.

Reads the output of get_demises_context.py and writes the records whose
biography has at least --min-words words. Short biographies do not give
enough material to write probes from. A summary of the filtering is printed
and also saved to a text file next to the output.

Usage:
    python scripts/filter_demise_events.py --input demises_context.jsonl --min-words 300 \\
        --output demises_context_filtered.jsonl
"""
import argparse
import json
import statistics
from pathlib import Path


def load_entries_with_word_counts(context_file):
    """Return a list of (entry, word_count) for the records in a context file.

    Lines that are not valid JSON, or that have no usable word count, are
    skipped.
    """
    data = []
    with open(context_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                word_count = entry.get("wikipedia_bio_word_count")
                if word_count is None:
                    continue
                data.append((entry, int(word_count)))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return data


def build_stats_text(input_file, output_file, stats_file, min_words, word_counts, num_kept):
    """Return the summary that is printed and written to the stats file.

    The word count statistics describe the input, before filtering.
    """
    total = len(word_counts)
    return "\n".join(
        [
            "=" * 60,
            "Filtering complete",
            "=" * 60,
            f"Input file: {input_file}",
            f"Output file: {output_file}",
            f"Stats file: {stats_file}",
            f"Minimum words required: {min_words}",
            f"Total valid entries: {total}",
            f"Entries kept: {num_kept} ({num_kept / total * 100:.1f}%)",
            f"Entries removed: {total - num_kept}",
            "Word count stats (input):",
            f"  Mean: {sum(word_counts) / total:.1f}",
            f"  Median: {statistics.median(word_counts):.1f}",
            f"  Min: {min(word_counts)}",
            f"  Max: {max(word_counts)}",
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Filter Wikipedia biographies by minimum word count"
    )
    parser.add_argument(
        "--input",
        default="demises_context.jsonl",
        help="Context JSONL from get_demises_context.py (default: demises_context.jsonl)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        required=True,
        help="Minimum number of words a biography needs to be kept",
    )
    parser.add_argument(
        "--output",
        default="demises_context_filtered.jsonl",
        help="Output JSONL path (default: demises_context_filtered.jsonl)",
    )
    parser.add_argument(
        "--stats-file",
        help="Path for the summary text file (default: <output_stem>_filter_stats.txt)",
    )
    args = parser.parse_args()

    if args.min_words < 0:
        raise ValueError("--min-words must be 0 or greater.")

    output_file = args.output
    output_path = Path(output_file)
    stats_file = args.stats_file or str(
        output_path.with_name(f"{output_path.stem}_filter_stats.txt")
    )

    print(f"Loading data from {args.input}...")
    data = load_entries_with_word_counts(args.input)
    if not data:
        raise ValueError("No valid entries with word counts were found in the input file.")

    # Kept entries are written unchanged and in their input order.
    kept = [entry for entry, word_count in data if word_count >= args.min_words]
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in kept:
            f.write(json.dumps(entry) + "\n")

    stats_text = build_stats_text(
        input_file=args.input,
        output_file=output_file,
        stats_file=stats_file,
        min_words=args.min_words,
        word_counts=[word_count for _, word_count in data],
        num_kept=len(kept),
    )
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(stats_text + "\n")
    print("\n" + stats_text)


if __name__ == "__main__":
    main()
