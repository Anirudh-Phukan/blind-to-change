#!/usr/bin/env python3
"""Classify the failures on indirect probes in the shared harness with Gemini.

Reads the judgements of answers from generate_answers_shared.py (judged by
judge_indirect_answers.py) and the trace of those answers. Every probe whose
answer was judged incorrect is assigned to the first failure mode that it
satisfies, in this order:

  Reasoning error                  The context before the final answer
                                   explicitly states the death or the end of
                                   the position, yet the answer is incorrect.
  Ineffective but explicit search  At least one search query directly checks
                                   the event (e.g. whether the person is
                                   alive, or who currently holds the position).
  Indirect search                  At least one query could reveal the event
                                   indirectly (e.g. the person's recent
                                   activity in the role).
  No/unrelated search              No query could reveal the event, including
                                   answers with no search at all.

The context is the input of the last model call in the trace, and the queries
are the web_search calls in it. The first mode is decided by one Gemini call
(prompts/failure_evidence_<event-type>.txt). If it does not apply and there
are queries, a second call labels each query explicit, indirect or none
(prompts/failure_query_labels_<event-type>.txt; the demise prompt calls the
indirect label "inadequate").

Tenures also need --position-file, the tenure context file with the position
label, description and context of each (wikidata_id, fact_text).

Usage:
    export GEMINI_API_KEY=...
    python scripts/classify_failures.py --event-type demises \\
        --judgements-file shared_answers_gpt-5.4_high_judgements.jsonl \\
        --trace-file shared_answers_gpt-5.4_high_trace.jsonl \\
        --output shared_answers_gpt-5.4_high_failures.jsonl

One line per classified probe is written to the output, in input order. A
probe whose classification fails is left out and logged. Rerun the same
command with --resume to retry those probes and to continue an interrupted
run. Without --resume an existing output file is overwritten.
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

REASONING_ERROR = "Reasoning error"
EXPLICIT_SEARCH = "Ineffective but explicit search"
INDIRECT_SEARCH = "Indirect search"
NO_SEARCH = "No/unrelated search"

# The key of the evidence response and the label of indirect queries.
EVIDENCE_KEY = {"demises": "explicit_target_death", "tenures": "explicit_position_departure"}
INDIRECT_LABEL = {"demises": "inadequate", "tenures": "indirect"}

DEFAULT_MODEL = "gemini-3.7-flash"
DEFAULT_THINKING_LEVEL = "low"
TEMPERATURE = 0.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_queries(context):
    """Return the queries of every web_search(<JSON>) call in a context, in order."""
    decoder = json.JSONDecoder()
    queries = []
    marker = "web_search("
    start = context.find(marker)
    while start >= 0:
        arguments, end = decoder.raw_decode(context, start + len(marker))
        # Older traces have a "queries" list instead of a single "query".
        queries.extend(arguments["queries"] if "queries" in arguments else [arguments["query"]])
        start = context.find(marker, end)
    return queries


def load_positions(position_file):
    """Return the first entry of each (wikidata_id, fact_text) in a tenure context file."""
    positions = {}
    for entry in read_jsonl(position_file):
        positions.setdefault((entry["wikidata_id"], entry["fact_text"]), entry)
    return positions


def load_items(judgements_file, trace_file, positions, processed):
    """Return the incorrect probes that still need to be classified.

    Each item is the judgement row with the context and queries of its trace,
    and for tenures the position fields used by the prompts.
    """
    traces = {}
    for trace in read_jsonl(trace_file):
        if trace["question"] in traces:
            raise ValueError(f"Probe has more than one answer in the trace: {trace['question']}")
        traces[trace["question"]] = trace

    items = []
    for row in read_jsonl(judgements_file):
        if row["judgements"] != ["incorrect"]:
            continue
        if (row["wikidata_id"], row["question"]) in processed:
            continue
        context = traces[row["question"]]["steps"][-1]["input"]
        item = {
            "wikidata_id": row["wikidata_id"],
            "name": row["name"],
            "question": row["question"],
            "context": context,
            "queries": extract_queries(context),
        }
        if positions is not None:
            position = positions[(row["wikidata_id"], row["fact_text"])]
            item.update(
                fact_text=row["fact_text"],
                position_label=position["position_label"],
                position_description=(position.get("position_wikidata_description") or "").strip()
                or "Not provided",
                position_context=(position.get("position_context") or "").strip() or "Not provided",
            )
        items.append(item)
    return items


def parse_evidence(text, key):
    """Return (explicit, evidence) from the evidence response."""
    result = parse_json_response(text)
    if set(result) != {key, "evidence"} or not isinstance(result[key], bool):
        raise ValueError(f"Expected exactly '{key}' (a Boolean) and 'evidence'")
    if result[key] and not (isinstance(result["evidence"], str) and result["evidence"].strip()):
        raise ValueError("A positive decision needs an evidence excerpt")
    return result[key], result["evidence"] if result[key] else None


def parse_labels(text, queries, event_type):
    """Return the labelled queries from the query-label response."""
    result = parse_json_response(text)
    rows = result["queries"]
    if len(rows) != len(queries):
        raise ValueError(f"Expected {len(queries)} labelled queries, got {len(rows)}")
    keys = {"index", "query", "label"} | ({"reason"} if event_type == "tenures" else set())
    labels = ("explicit", INDIRECT_LABEL[event_type], "none")
    for index, (row, query) in enumerate(zip(rows, queries)):
        if set(row) != keys or row["index"] != index or row["query"] != query:
            raise ValueError(f"Query {index} is not repeated verbatim with the expected fields")
        if row["label"] not in labels:
            raise ValueError(f"Query {index} has an invalid label: {row['label']}")
        if event_type == "tenures" and not str(row["reason"]).strip():
            raise ValueError(f"Query {index} has no reason")
    return rows


def classify(client, rate_limiter, item, event_type, prompts, model, thinking_level):
    """Return (failure mode, evidence, labelled queries) for one probe, or None on failure."""
    fields = {k: v for k, v in item.items() if k not in ("wikidata_id", "fact_text", "queries")}
    fields["final_context"] = fields.pop("context")
    try:
        explicit, evidence = call_gemini(
            client,
            rate_limiter,
            prompts["evidence"].format(**fields),
            model,
            thinking_level,
            TEMPERATURE,
            parse=lambda text: parse_evidence(text, EVIDENCE_KEY[event_type]),
        )
        if explicit:
            return REASONING_ERROR, evidence, []
        if not item["queries"]:
            return NO_SEARCH, None, []

        queries_json = json.dumps(
            [{"index": i, "query": q} for i, q in enumerate(item["queries"])], ensure_ascii=False
        )
        rows = call_gemini(
            client,
            rate_limiter,
            prompts["query_labels"].format(queries_json=queries_json, **fields),
            model,
            thinking_level,
            TEMPERATURE,
            parse=lambda text: parse_labels(text, item["queries"], event_type),
        )
    except Exception as e:
        logger.error(f"Failed to classify the failure for {item['name']}: {e}")
        return None

    labels = [row["label"] for row in rows]
    if "explicit" in labels:
        return EXPLICIT_SEARCH, None, rows
    if INDIRECT_LABEL[event_type] in labels:
        return INDIRECT_SEARCH, None, rows
    return NO_SEARCH, None, rows


def main():
    parser = argparse.ArgumentParser(
        description="Classify the failures on indirect probes in the shared harness with Gemini"
    )
    parser.add_argument("--event-type", required=True, choices=["demises", "tenures"])
    parser.add_argument(
        "--judgements-file", required=True, help="Judgements JSONL from judge_indirect_answers.py"
    )
    parser.add_argument(
        "--trace-file", required=True, help="Trace JSONL from generate_answers_shared.py"
    )
    parser.add_argument(
        "--position-file", help="Tenure context JSONL with the position fields (tenures only)"
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
        help="Number of probes to classify in parallel (default: 10)",
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
    args = parser.parse_args()
    output_file = Path(args.output)
    tenures = args.event_type == "tenures"

    if tenures and not args.position_file:
        parser.error("--position-file is required for tenures")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        parser.error("Set the GEMINI_API_KEY environment variable")
    client = genai.Client(api_key=api_key)
    rate_limiter = RateLimiter(args.rate_limit)

    prompts = {
        name: (PROMPT_DIR / f"failure_{name}_{args.event_type}.txt").read_text(encoding="utf-8")
        for name in ("evidence", "query_labels")
    }
    positions = load_positions(args.position_file) if tenures else None
    processed = load_processed_probes(output_file) if args.resume else set()
    items = load_items(args.judgements_file, args.trace_file, positions, processed)
    logger.info(f"Event type: {args.event_type}")
    logger.info(f"Model: {args.model} (thinking level: {args.thinking_level})")
    logger.info(f"Incorrect probes to classify: {len(items)}")

    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    futures = [
        executor.submit(
            classify,
            client,
            rate_limiter,
            item,
            args.event_type,
            prompts,
            args.model,
            args.thinking_level,
        )
        for item in items
    ]

    counts = {}
    failed = 0
    with output_file.open("a" if args.resume else "w", encoding="utf-8") as f:
        try:
            for done, (item, future) in enumerate(zip(items, futures), 1):
                result = future.result()
                if result is None:
                    failed += 1
                    continue
                mode, evidence, rows = result
                record = {k: item[k] for k in ("wikidata_id", "name", "question")}
                if tenures:
                    record.update(fact_text=item["fact_text"], position_label=item["position_label"])
                record.update(
                    failure_mode=mode,
                    evidence=evidence,
                    queries=rows or [{"index": i, "query": q} for i, q in enumerate(item["queries"])],
                )
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                counts[mode] = counts.get(mode, 0) + 1
                logger.info(f"[{done}/{len(items)}] {item['name']}: {mode}")
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Interrupted. Progress has been saved; rerun with --resume to continue.")

    for mode in (REASONING_ERROR, EXPLICIT_SEARCH, INDIRECT_SEARCH, NO_SEARCH):
        logger.info(f"{mode}: {counts.get(mode, 0)}")
    if failed:
        logger.warning(f"Failed: {failed} (rerun with --resume to retry)")
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
