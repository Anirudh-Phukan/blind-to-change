#!/usr/bin/env python3
"""Judge answers to indirect probes with Gemini.

Reads an answers file from generate_answers_native.py and asks the judge, for
every answer, whether it is consistent with the event:

  demises  The fact is "<name> died". An answer is "incorrect" if it
           recommends the person as if they were still alive, and "correct"
           otherwise: it need not say that the person died.
  tenures  The fact is the entry's "fact_text" (e.g. "X stopped being Y").
           An answer is "incorrect" if it treats the person as still holding,
           or as currently authoritative because of, the ended position.

The prompt is prompts/indirect_judge_<event-type>.txt. One line per probe is
written to the output JSONL, in input order.

The pipeline uses this script twice: on the answers without search, whose
judgements are the input of filter_indirect_probes.py, and on the answers with
search, whose judgements are the benchmark's scores.

Usage:
    export GEMINI_API_KEY=...
    python scripts/judge_indirect_answers.py --event-type demises \\
        --input answers_gpt-5.4.jsonl \\
        --output answers_gpt-5.4_judgements.jsonl

If the run is interrupted, rerun the same command with --resume to skip the
probes already in the output file. Without --resume an existing output file
is overwritten. To print the statistics of an existing judgements file:

    python scripts/judge_indirect_answers.py --stats-only \\
        --output answers_gpt-5.4_judgements.jsonl
"""
import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google import genai

from utils.llm_utils import RateLimiter, call_gemini, load_processed_probes, parse_json_response

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# Fields of a tenure entry that are copied to the output.
TENURE_FIELDS = ("fact_text", "fact_id", "fact_type", "position_label")

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_THINKING_LEVEL = "low"
TEMPERATURE = 0.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def load_entries(answers_file, processed, required_fields=()):
    """Return the probes in an answers file that still need to be judged.

    Every entry needs wikidata_id, name and question, plus required_fields.
    """
    entries = []
    with open(answers_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Malformed JSON, skipping: {e}")
                continue
            missing = [k for k in ("wikidata_id", "name", "question", *required_fields) if k not in entry]
            if missing:
                logger.warning(f"Line {line_num}: Missing fields {missing}, skipping")
                continue
            # A probe has no answers if all of its requests failed.
            if not isinstance(entry.get("answers"), list) or not entry["answers"]:
                logger.warning(f"Line {line_num}: Missing or empty 'answers' list, skipping")
                continue
            if (entry["wikidata_id"], entry["question"]) not in processed:
                entries.append(entry)
    return entries


def judge_answer(client, rate_limiter, name, prompt, model, thinking_level):
    """Return (judgement, reasoning) for one answer.

    The judgement is "correct" or "incorrect". A failure (API error or
    unparseable response) gives (None, None) rather than an exception, so that
    one answer cannot stop the run.
    """
    try:
        response = call_gemini(client, rate_limiter, prompt, model, thinking_level, TEMPERATURE)
        result = parse_json_response(response)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object")
        if "judgement" not in result:
            raise ValueError("Missing 'judgement' field in response")
        judgement = result["judgement"].lower().strip()
        if judgement not in ("correct", "incorrect"):
            raise ValueError(f"Invalid judgement value: {judgement}")
        return judgement, result.get("reasoning", "")
    except Exception as e:
        logger.error(f"Failed to judge answer for {name}: {e}")
        return None, None


def load_judgements(judgements_file):
    """Return the list of judgements of every probe in a judgements file."""
    rows = []
    with open(judgements_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                rows.append(json.loads(line.strip())["judgements"])
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Line {line_num}: Could not read judgements, skipping: {e}")
    return rows


def log_statistics(judgement_rows):
    """Log how many answers were judged correct and incorrect."""
    judgements = [judgement for row in judgement_rows for judgement in row]
    correct = judgements.count("correct")
    incorrect = judgements.count("incorrect")
    logger.info(f"Probes judged: {len(judgement_rows)}")
    logger.info(f"Answers judged: {len(judgements)}")
    logger.info(f"Failed answer judgements: {judgements.count(None)}")
    logger.info(f"Correct: {correct}")
    logger.info(f"Incorrect: {incorrect}")
    if correct + incorrect > 0:
        logger.info(f"Accuracy: {correct / (correct + incorrect) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Judge answers to indirect probes with Gemini")
    parser.add_argument(
        "--event-type",
        choices=["demises", "tenures"],
        help="Selects the judge prompt and the fact (required unless --stats-only)",
    )
    parser.add_argument("--input", help="Answers JSONL from generate_answers_native.py")
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
    prompt_file = PROMPT_DIR / f"indirect_judge_{args.event_type}.txt"
    prompt_template = prompt_file.read_text(encoding="utf-8").strip()
    tenures = args.event_type == "tenures"

    processed = load_processed_probes(output_file) if args.resume else set()
    entries = load_entries(args.input, processed, ("fact_text",) if tenures else ())
    logger.info(f"Event type: {args.event_type}")
    logger.info(f"Model: {args.model} (thinking level: {args.thinking_level})")
    logger.info(f"Probes to judge: {len(entries)}")

    # One request per answer. Every probe's requests are submitted up front
    # and its line is written once all of them have finished.
    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    futures = []
    for entry in entries:
        fact = entry["fact_text"].strip() if tenures else f"{entry['name']} died"
        prompts = [
            prompt_template.format(fact=fact, question=entry["question"], answer=answer)
            for answer in entry["answers"]
        ]
        futures.append(
            [
                executor.submit(
                    judge_answer,
                    client,
                    rate_limiter,
                    entry["name"],
                    prompt,
                    args.model,
                    args.thinking_level,
                )
                for prompt in prompts
            ]
        )

    judgement_rows = []
    with output_file.open("a" if args.resume else "w", encoding="utf-8") as f:
        try:
            for entry, entry_futures in zip(entries, futures):
                results = [future.result() for future in entry_futures]
                judgements = [judgement for judgement, _ in results]
                record = {
                    "wikidata_id": entry["wikidata_id"],
                    "name": entry["name"],
                    "question": entry["question"],
                    "answers": entry["answers"],
                    "judgements": judgements,
                    "judgement_reasonings": [reasoning for _, reasoning in results],
                    "incorrect_judgement_count": judgements.count("incorrect"),
                }
                if tenures:
                    record.update({k: entry[k] for k in TENURE_FIELDS if k in entry})
                    record["fact_text"] = entry["fact_text"].strip()
                f.write(json.dumps(record) + "\n")
                f.flush()
                judgement_rows.append(judgements)
                logger.info(
                    f"[{len(judgement_rows)}/{len(entries)}] {entry['name']}: "
                    f"correct={judgements.count('correct')}, "
                    f"incorrect={judgements.count('incorrect')}, "
                    f"failed={judgements.count(None)}"
                )
        except KeyboardInterrupt:
            # Drop the requests that have not started; those already running
            # finish in the background but their results are not written.
            executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Interrupted. Progress has been saved; rerun with --resume to continue.")

    log_statistics(judgement_rows)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
