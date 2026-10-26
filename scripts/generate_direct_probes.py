#!/usr/bin/env python3
"""Generate direct probes for demise or tenure events with Gemini.

Reads a file of indirect probes and, for every line, asks Gemini for one
direct probe of the event:

  demises  Is <name>, the <short identifying description>, alive?
           The description is taken from the person's Wikipedia biography
           and tells the person apart from others with the same name. The
           input has one line per person.
  tenures  Is <name> currently serving as <position>?
           The position is qualified by an organization, institution or
           jurisdiction when the context makes it clear. The input has one
           line per ended position, so a person who left several positions
           has several lines, each told apart by its "fact_text".

The prompt is prompts/direct_probe_generation_<event-type>.txt. The output
has the same format as the input, with the direct probe in place of the
indirect one, so the answering and judging scripts read both in the same way.
Each result is appended to the output JSONL as soon as it is ready.

Usage:
    export GEMINI_API_KEY=...
    python scripts/generate_direct_probes.py --event-type demises \\
        --input indirect_probes_gpt-5.4.jsonl \\
        --context demises_context_filtered.jsonl --output direct_probes_gpt-5.4.jsonl

If the run is interrupted, rerun the same command with --resume to skip the
lines already in the output file. Without --resume an existing output file
is overwritten.
"""
import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai

from utils.llm_utils import RateLimiter, call_gemini

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

DEFAULT_MODEL = "gemini-3.7-flash"
DEFAULT_THINKING_LEVEL = "low"
TEMPERATURE = 0.2

# Words that would give away the answer to the probe.
ANSWER_REVEALING_PATTERNS = {
    "demises": re.compile(r"\b(?:dead|death|deceased|died|late)\b", re.IGNORECASE),
    "tenures": re.compile(
        r"\b(?:former|formerly|previously|still|not|stopped|ended|ceased|successor|replaced)\b"
        r"|\bno\s+longer\b",
        re.IGNORECASE,
    ),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def event_key(record, event_type):
    """Return the key that identifies one event in the input, context and output.

    A demise is identified by the person; a tenure by the person and the
    ended position, which fact_text names.
    """
    if event_type == "tenures":
        return record["wikidata_id"], record["fact_text"]
    return record["wikidata_id"]


def read_jsonl(path):
    """Return the records of a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_processed_keys(output_file, event_type):
    """Return the event keys already in an output file, to resume a run.

    Malformed lines (e.g. a half-written last line) are reported and skipped.
    """
    processed = set()
    if not output_file.exists():
        return processed
    with output_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                processed.add(event_key(json.loads(line), event_type))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"{output_file}, line {line_num}: skipping malformed line: {e}")
    return processed


def build_prompt(template, entry, context, event_type):
    """Fill the prompt template for one event from its context record."""
    if event_type == "demises":
        return template.format(name=entry["name"], wikipedia_bio=context["wikipedia_bio"])
    # The position's own context (Wikidata description and Wikipedia
    # article) is missing for some positions.
    position_context = (context.get("position_context") or "").strip()
    return template.format(
        name=entry["name"],
        position_label=context["position_label"].strip(),
        wikipedia_bio=context["wikipedia_bio"],
        optional_position_context=(
            f"\nPosition context:\n{position_context}\n" if position_context else ""
        ),
    )


def parse_probe(response, name, event_type):
    """Return the direct probe in a model response.

    Raises ValueError if the response does not follow the requested form or
    gives away the answer.
    """
    text = response.strip()
    # The prompt asks for the bare question, but models sometimes wrap it in
    # a Markdown code fence or in JSON quotation marks.
    lines = text.splitlines()
    if text.startswith("```") and text.endswith("```") and len(lines) >= 3:
        text = "\n".join(lines[1:-1]).strip()
    try:
        decoded = json.loads(text)
        if isinstance(decoded, str):
            text = decoded.strip()
    except json.JSONDecodeError:
        pass
    text = " ".join(text.split())

    if event_type == "demises":
        prefix, suffix = f"Is {name}, the ", ", alive?"
    else:
        prefix, suffix = f"Is {name} currently ", "?"
    if not text.startswith(prefix):
        raise ValueError(f"question must begin with {prefix!r}")
    if not text.endswith(suffix):
        raise ValueError(f"question must end with {suffix!r}")
    if text.count("?") != 1:
        raise ValueError("response must contain exactly one question")
    if ANSWER_REVEALING_PATTERNS[event_type].search(text):
        raise ValueError("question contains answer-revealing language")
    if not text[len(prefix) : -len(suffix)].strip():
        raise ValueError("question has nothing between its fixed beginning and end")
    return text


def generate_probe(client, rate_limiter, entry, prompt, event_type, model, thinking_level):
    """Return the output record for one event.

    A response that does not pass parse_probe is retried like a failed API
    call. If every attempt fails, the failure is recorded with success=False
    rather than raised, so that one event cannot stop the run.
    """
    # Keep every field of the input record and replace its probe.
    record = dict(entry)
    try:
        probe = call_gemini(
            client,
            rate_limiter,
            prompt,
            model,
            thinking_level,
            TEMPERATURE,
            parse=lambda response: parse_probe(response, entry["name"], event_type),
        )
        # The two empty fields keep the format of the indirect probes.
        record["questions"] = [{"question": probe, "scenario_persona": "", "category": ""}]
        record["success"] = True
        record["error"] = None
    except Exception as e:
        record["questions"] = []
        record["success"] = False
        record["error"] = str(e)
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Generate one direct probe per event for demises or tenures with Gemini"
    )
    parser.add_argument(
        "--event-type",
        required=True,
        choices=["demises", "tenures"],
        help="Selects the prompt, the form of the probe and how events are identified",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Indirect probes JSONL with one line per event (per person for demises)",
    )
    parser.add_argument(
        "--context",
        required=True,
        help="Context JSONL: from get_demises_context.py or filter_demise_events.py for "
        "demises; with position_label and position_context for tenures",
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
        "--resume", action="store_true", help="Skip events already in the output file"
    )
    args = parser.parse_args()
    event_type = args.event_type
    output_file = Path(args.output)

    if output_file.resolve() in (Path(args.input).resolve(), Path(args.context).resolve()):
        parser.error("--output must be different from --input and --context")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        parser.error("Set the GEMINI_API_KEY environment variable")
    client = genai.Client(api_key=api_key)
    rate_limiter = RateLimiter(args.rate_limit)

    # The file ends with a newline, which is not part of the prompt.
    prompt_file = PROMPT_DIR / f"direct_probe_generation_{event_type}.txt"
    prompt_template = prompt_file.read_text(encoding="utf-8").strip()

    contexts = {event_key(r, event_type): r for r in read_jsonl(args.context)}
    entries = read_jsonl(args.input)
    # One direct probe is generated per input line, so an event that appears
    # twice would get two.
    keys = [event_key(entry, event_type) for entry in entries]
    if len(set(keys)) != len(keys):
        parser.error(f"{args.input}: an event appears on more than one line")
    missing = [entry["name"] for entry, key in zip(entries, keys) if key not in contexts]
    if missing:
        parser.error(f"{args.context}: no context for {', '.join(missing)}")

    processed = load_processed_keys(output_file, event_type) if args.resume else set()
    entries = [entry for entry in entries if event_key(entry, event_type) not in processed]
    logger.info(f"Event type: {event_type}")
    logger.info(f"Model: {args.model} (thinking level: {args.thinking_level})")
    logger.info(f"Entries to process: {len(entries)}")

    successful = 0
    done = 0
    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    futures = [
        executor.submit(
            generate_probe,
            client,
            rate_limiter,
            entry,
            build_prompt(
                prompt_template, entry, contexts[event_key(entry, event_type)], event_type
            ),
            event_type,
            args.model,
            args.thinking_level,
        )
        for entry in entries
    ]
    # Results are appended in the order they finish, not in input order.
    with output_file.open("a" if args.resume else "w", encoding="utf-8") as f:
        try:
            for future in as_completed(futures):
                record = future.result()
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                done += 1

                if record["success"]:
                    successful += 1
                    logger.info(
                        f"[{done}/{len(entries)}] ✓ {record['questions'][0]['question']}"
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
    # A non-zero exit code tells a calling script that some probes are missing.
    if done > successful:
        sys.exit(1)


if __name__ == "__main__":
    main()
