#!/usr/bin/env python3
"""Generate indirect probes for demise or tenure events with Gemini.

For every event in a context file, asks Gemini for five indirect probes:
natural recommendation requests that never name the person, for which the
person is a plausible answer only if the old fact still holds, i.e. they are
still alive (demises) or still hold the position (tenures). Each result is
appended to the output JSONL as soon as it is ready.

The prompt is prompts/indirect_probe_generation_<event-type>.txt. A demise is
one person; a tenure is one ended position, so a person with several ended
positions gets one line per position.

Usage:
    export GEMINI_API_KEY=...
    python scripts/generate_indirect_probes.py --event-type demises \\
        --input demises_context_filtered.jsonl --output indirect_probes_demises.jsonl
    python scripts/generate_indirect_probes.py --event-type tenures \\
        --input tenures_context.jsonl --output indirect_probes_tenures.jsonl

If the run is interrupted, rerun the same command with --resume to skip the
events already in the output file.
"""
import argparse
import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai

from generate_direct_probes import PROMPT_DIR, event_key, load_processed_keys
from judge_indirect_answers import TENURE_FIELDS
from utils.llm_utils import RateLimiter, call_gemini, parse_json_response

CONSEQUENCE_TYPES_FILE = PROMPT_DIR / "demise_consequence_types.txt"
REQUIRED_FIELDS = {
    "demises": ("wikidata_id", "name", "wikipedia_bio"),
    "tenures": ("wikidata_id", "name", "wikipedia_bio", "fact_id", "fact_text",
                "relevant_information"),
}

DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_THINKING_LEVEL = "high"
TEMPERATURE = 0.2
# Each prompt lists a random subset of the consequence types (the kinds of
# present-day participation a probe can ask for), so that the probes vary
# from person to person. Tenure prompts do not use them.
CONSEQUENCE_TYPES_PER_PROMPT = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_entries(context_file, event_type, processed_keys):
    """Return the events in a context file that still need probes."""
    entries = []
    with open(context_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Malformed JSON, skipping: {e}")
                continue
            missing = [k for k in REQUIRED_FIELDS[event_type] if k not in entry]
            if missing:
                logger.warning(f"Line {line_num}: Missing fields {missing}, skipping")
                continue
            if event_key(entry, event_type) not in processed_keys:
                entries.append(entry)
    return entries


def build_prompt(prompt_template, consequence_types, entry, event_type):
    """Fill the prompt template for one event."""
    if event_type == "tenures":
        return prompt_template.format(
            fact_text=entry["fact_text"],
            entity_context=entry["wikipedia_bio"],
            relevant_information=entry["relevant_information"],
        )
    sampled_types = random.sample(
        consequence_types, min(CONSEQUENCE_TYPES_PER_PROMPT, len(consequence_types))
    )
    return prompt_template.format(
        fact=f"{entry['name']} died",
        context=entry["wikipedia_bio"],
        consequence_types_list="\n".join(f"- {t}" for t in sampled_types),
    )


def parse_probes(response):
    """Parse the model's response into a list of probes."""
    probes = parse_json_response(response)
    if not isinstance(probes, list):
        raise ValueError("Expected list of questions")
    return probes


def generate_probes(client, rate_limiter, entry, prompt, event_type, model, thinking_level):
    """Return the output record for one event.

    A failure (API error or unparseable response) is recorded with
    success=False rather than raised, so that one event cannot stop the run.
    """
    record = {
        "wikidata_id": entry["wikidata_id"],
        "name": entry["name"],
        "questions": [],
        "success": False,
        "error": None,
    }
    if event_type == "tenures":
        record.update({k: entry[k] for k in TENURE_FIELDS if k in entry})
    try:
        record["questions"] = parse_probes(
            call_gemini(client, rate_limiter, prompt, model, thinking_level, TEMPERATURE)
        )
        record["success"] = True
    except Exception as e:
        record["error"] = str(e)
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Generate indirect probes for demise or tenure events with Gemini"
    )
    parser.add_argument(
        "--event-type",
        required=True,
        choices=["demises", "tenures"],
        help="Selects the prompt and how events are identified",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Context JSONL: from filter_demise_events.py for demises, with one line "
        "per ended position for tenures",
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
        "--resume",
        action="store_true",
        help="Skip events already in the output file",
    )
    args = parser.parse_args()
    event_type = args.event_type
    output_file = Path(args.output)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        parser.error("Set the GEMINI_API_KEY environment variable")
    client = genai.Client(api_key=api_key)
    rate_limiter = RateLimiter(args.rate_limit)

    prompt_file = PROMPT_DIR / f"indirect_probe_generation_{event_type}.txt"
    prompt_template = prompt_file.read_text(encoding="utf-8")
    consequence_types = []
    if event_type == "demises":
        consequence_types = [
            line.strip()
            for line in CONSEQUENCE_TYPES_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    processed_keys = load_processed_keys(output_file, event_type) if args.resume else set()
    entries = load_entries(args.input, event_type, processed_keys)
    logger.info(f"Event type: {event_type}")
    logger.info(f"Model: {args.model} (thinking level: {args.thinking_level})")
    logger.info(f"Entries to process: {len(entries)}")

    successful = 0
    done = 0
    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    prompts = [
        build_prompt(prompt_template, consequence_types, entry, event_type) for entry in entries
    ]
    futures = [
        executor.submit(
            generate_probes,
            client,
            rate_limiter,
            entry,
            prompt,
            event_type,
            args.model,
            args.thinking_level,
        )
        for entry, prompt in zip(entries, prompts)
    ]
    # Results are appended in the order they finish, not in input order.
    with output_file.open("a", encoding="utf-8") as f:
        try:
            for future in as_completed(futures):
                record = future.result()
                f.write(json.dumps(record) + "\n")
                f.flush()
                done += 1

                if record["success"]:
                    successful += 1
                    logger.info(
                        f"[{done}/{len(entries)}] ✓ Generated {len(record['questions'])} "
                        f"questions for {record['name']}"
                    )
                else:
                    logger.error(
                        f"[{done}/{len(entries)}] ✗ Failed for {record['name']} "
                        f"({record['wikidata_id']}): {record['error']}"
                    )
        except KeyboardInterrupt:
            # Drop the requests that have not started; those already running
            # finish in the background but their results are not written.
            executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Interrupted. Progress has been saved; rerun with --resume to continue.")

    logger.info(f"Done. Successful: {successful}, Failed: {done - successful}")
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
