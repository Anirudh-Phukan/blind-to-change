#!/usr/bin/env python3
"""Judge answers to direct probes for demise or tenure events with Gemini.

Reads an answers file from generate_answers_native.py or
generate_answers_shared.py for the direct probes and asks the judge, for
every answer, whether it gives the event:

  demises  "Is <name>, the <description>, alive?": does the answer say that
           the person is dead?
  tenures  "Is <name> currently serving as <position>?": does the answer say
           that the person no longer holds that exact position?

An answer is "correct" only if it clearly says so, and "incorrect" if it says
the opposite, is uncertain or avoids the question. The prompt is
prompts/direct_judge_<event-type>.txt. One line per probe is written to the
output JSONL, in input order.

Usage:
    export GEMINI_API_KEY=...
    python scripts/judge_direct_answers.py --event-type demises \\
        --input direct_answers_gpt-5.4_high.jsonl \\
        --output direct_answers_gpt-5.4_high_judgements.jsonl

If the run is interrupted, rerun the same command with --resume to skip the
probes already in the output file. Without --resume an existing output file
is overwritten. To print the statistics of an existing judgements file:

    python scripts/judge_direct_answers.py --stats-only \\
        --output direct_answers_gpt-5.4_high_judgements.jsonl
"""
import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google import genai

from judge_indirect_answers import (
    PROMPT_DIR,
    TENURE_FIELDS,
    load_entries,
    load_judgements,
    log_statistics,
)
from utils.llm_utils import RateLimiter, call_gemini, load_processed_probes, parse_json_response

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_THINKING_LEVEL = "low"
TEMPERATURE = 0.0

logger = logging.getLogger(__name__)


def parse_judgement(response):
    """Return (judgement, reasoning) from the judge's JSON response.

    Raises ValueError if the response does not have the requested form.
    """
    result = parse_json_response(response)
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    judgement = result.get("judgement")
    if not isinstance(judgement, str) or judgement.strip().lower() not in ("correct", "incorrect"):
        raise ValueError("'judgement' must be 'correct' or 'incorrect'")
    reasoning = result.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise ValueError("'reasoning' must be a string")
    return judgement.strip().lower(), reasoning.strip()


def judge_answer(client, rate_limiter, name, prompt, model, thinking_level):
    """Return (judgement, reasoning) for one answer.

    A response that does not pass parse_judgement is retried like a failed
    API call. If every attempt fails, the result is (None, None) rather than
    an exception, so that one answer cannot stop the run.
    """
    try:
        return call_gemini(
            client, rate_limiter, prompt, model, thinking_level, TEMPERATURE, parse=parse_judgement
        )
    except Exception as e:
        logger.error(f"Failed to judge answer for {name}: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Judge answers to direct probes for demise or tenure events with Gemini"
    )
    parser.add_argument(
        "--event-type",
        choices=["demises", "tenures"],
        help="Selects the judge prompt (required unless --stats-only)",
    )
    parser.add_argument(
        "--input",
        help="Answers JSONL from generate_answers_native.py or generate_answers_shared.py",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--thinking-level",
        choices=["minimal", "low", "medium", "high"],
        default=DEFAULT_THINKING_LEVEL,
        help=f"Gemini thinking level (default: {DEFAULT_THINKING_LEVEL})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of requests to run in parallel (default: 10)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Minimum seconds between the start of two requests (default: 1.0)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip probes already in the output file"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print the statistics of an existing output file without calling the judge",
    )
    args = parser.parse_args()
    output_file = Path(args.output)

    if args.stats_only:
        if not output_file.exists():
            parser.error(f"Output file not found: {output_file}")
        log_statistics(load_judgements(output_file))
        return

    if not args.input:
        parser.error("--input is required")
    if not args.event_type:
        parser.error("--event-type is required")
    if output_file.resolve() == Path(args.input).resolve():
        parser.error("--output must be different from --input")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        parser.error("Set the GEMINI_API_KEY environment variable")
    client = genai.Client(api_key=api_key)
    rate_limiter = RateLimiter(args.rate_limit)

    # The file ends with a newline, which is not part of the prompt.
    prompt_file = PROMPT_DIR / f"direct_judge_{args.event_type}.txt"
    prompt_template = prompt_file.read_text(encoding="utf-8").strip()
    tenures = args.event_type == "tenures"

    processed = load_processed_probes(output_file) if args.resume else set()
    entries = load_entries(args.input, processed, required_fields=("fact_text",) if tenures else ())
    logger.info(f"Event type: {args.event_type}")
    logger.info(f"Model: {args.model} (thinking level: {args.thinking_level})")
    logger.info(f"Probes to judge: {len(entries)}")

    # One request per answer. Every probe's requests are submitted up front
    # and its line is written once all of them have finished.
    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    futures = [
        [
            executor.submit(
                judge_answer,
                client,
                rate_limiter,
                entry["name"],
                # The tenure prompt has no {name}; format() ignores it.
                prompt_template.format(
                    name=entry["name"], question=entry["question"], answer=answer
                ),
                args.model,
                args.thinking_level,
            )
            for answer in entry["answers"]
        ]
        for entry in entries
    ]

    with output_file.open("a" if args.resume else "w", encoding="utf-8") as f:
        try:
            for done, (entry, entry_futures) in enumerate(zip(entries, futures), 1):
                results = [future.result() for future in entry_futures]
                judgements = [judgement for judgement, _ in results]
                record = {
                    "wikidata_id": entry["wikidata_id"],
                    "name": entry["name"],
                    "question": entry["question"],
                }
                if tenures:
                    record.update({k: entry[k] for k in TENURE_FIELDS if k in entry})
                record.update({
                    "answers": entry["answers"],
                    "judgements": judgements,
                    "judgement_reasonings": [reasoning for _, reasoning in results],
                    "incorrect_judgement_count": judgements.count("incorrect"),
                })
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                logger.info(
                    f"[{done}/{len(entries)}] {entry['name']}: "
                    f"correct={judgements.count('correct')}, "
                    f"incorrect={judgements.count('incorrect')}, "
                    f"failed={judgements.count(None)}"
                )
        except KeyboardInterrupt:
            # Drop the requests that have not started; those already running
            # finish in the background but their results are not written.
            executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Interrupted. Progress has been saved; rerun with --resume to continue.")

    # The statistics cover the whole output file, including the probes judged
    # before a resumed run.
    judgement_rows = load_judgements(output_file)
    log_statistics(judgement_rows)
    logger.info(f"Results saved to {output_file}")
    # A non-zero exit code tells a calling script that some judgements are missing.
    if any(judgement is None for row in judgement_rows for judgement in row):
        sys.exit(1)


if __name__ == "__main__":
    main()
