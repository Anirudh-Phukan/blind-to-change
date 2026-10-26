#!/usr/bin/env python3
"""Answer probes with a model in the shared harness (Inspect AI with Exa search).

Reads a probe file and gives every probe to the model together with a system
prompt and a single web_search tool backed by Exa. This is the shared harness
of the paper: every model gets the same prompt, the same tool and the same
search backend. The model writes its own queries and may search as often as
it wants before answering.

Usage:
    export OPENAI_API_KEY=... EXA_API_KEY=...
    python scripts/generate_answers_shared.py --provider openai --model gpt-5.4 \\
        --effort high --knowledge-cutoff 2025-08-31 \\
        --input filtered_indirect_probes_gpt-5.4.jsonl \\
        --output shared_answers_gpt-5.4_high.jsonl \\
        --trace-output shared_answers_gpt-5.4_high_trace.jsonl

--knowledge-cutoff is the model's knowledge cutoff as reported by its
provider. It is stated in the system prompt. The model's API key is read from
OPENAI_API_KEY or GEMINI_API_KEY and the Exa key from EXA_API_KEY.

Probes are run in batches, and the answers of a batch are appended to the
output when it finishes. A probe is left out of the output if the model gave
no answer or one of its searches failed, and the script then exits with
status 1. Rerun the same command with --resume to retry those probes and to
continue an interrupted run. Without --resume an existing output file is
overwritten.

The trace file has one line per answer with its steps in order: every model
call (its input, reasoning summary and output) and every search (the query
and the results that the model saw).
"""
import argparse
import asyncio
import json
import logging
import os
from datetime import date
from pathlib import Path

from exa_py import Exa
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import resolve_sample_attachments
from inspect_ai.model import ContentReasoning, ContentText
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import ToolError, tool

from utils.llm_utils import load_probes, load_processed_probes

SYSTEM_MESSAGE = (
    "Current date: {current_date}. "
    "Model knowledge cutoff: {knowledge_cutoff}. "
    "Interpret 'current', 'currently', 'now', 'today', 'recently', "
    "'this year', 'latest', and similar terms relative to the current date. "
    "If a fact may have changed after the model knowledge cutoff, or is otherwise "
    "time-sensitive, call the web_search tool before answering."
)
TEMPERATURE = 0.2
EXA_NUM_RESULTS = 5
# Inspect truncates a tool's output if it is longer than this many bytes.
MAX_TOOL_OUTPUT = 65536
MAX_RETRIES = 3

# "inspect_prefix" is the name of the provider in Inspect's model names. The
# effort values are the provider's own. The paper's low and high settings are
# none and high for GPT-5.4, and low and high for Gemini-3.1-Pro.
PROVIDERS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-5.4",
        "inspect_prefix": "openai",
        "effort_choices": ("none", "low", "medium", "high", "xhigh"),
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "default_model": "gemini-3.1-pro-preview",
        "inspect_prefix": "google",
        "effort_choices": ("minimal", "low", "medium", "high"),
    },
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def to_json(value):
    """Return a compact JSON string, as given to the model and stored in the trace."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def search_exa(query):
    """Return the Exa results for a query as the JSON string that the model sees."""
    response = Exa(os.environ["EXA_API_KEY"]).search(
        query, num_results=EXA_NUM_RESULTS, type="auto", contents={"highlights": True}
    )
    results = [
        {
            "title": str(result.title or ""),
            "url": str(result.url or ""),
            "publishedDate": str(result.published_date or ""),
            "highlights": [str(highlight) for highlight in result.highlights or []],
        }
        for result in response.results
    ]
    return to_json({"results": results})


@tool(name="web_search")
def web_search():
    # Inspect gives the docstring below to the model as the tool's description.
    async def execute(query: str) -> str:
        """
        Search the web for current information.

        Args:
            query: Search query.

        Returns:
            A JSON string with source results containing title, url, publishedDate, and highlights.
        """
        try:
            # The Exa client blocks, so it runs in a thread.
            return await asyncio.to_thread(search_exa, query)
        except Exception as e:
            raise ToolError(f"Exa search failed: {e}") from e

    return execute


def content_text(content):
    """Return the text of a message's content, without its reasoning parts."""
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content if isinstance(part, ContentText) and part.text)


def format_tool_call(function, arguments):
    return f"{function}({to_json(arguments)})"


def message_text(message):
    """Return a message's text followed by the tool calls it makes, one per line."""
    parts = [content_text(message.content)]
    for tool_call in getattr(message, "tool_calls", None) or []:
        parts.append(format_tool_call(tool_call.function, tool_call.arguments))
    return "\n".join(part for part in parts if part)


def reasoning_summary(message):
    """Return the reasoning summary of a model response, or None if it has none."""
    if isinstance(message.content, str):
        return None
    for part in message.content:
        if isinstance(part, ContentReasoning):
            text = (part.summary or "").strip() or (part.reasoning or "").strip()
            if text:
                return text
    return None


def trace_steps(sample):
    """Return the model calls and searches of one answer, in order."""
    # Inspect stores long strings once and refers to them elsewhere. This
    # puts them back in place.
    sample = resolve_sample_attachments(sample, "full")
    steps = []
    for event in sample.events:
        if event.event == "model":
            response = event.output.choices[0].message if event.output.choices else None
            steps.append(
                {
                    "input": "\n\n".join(filter(None, map(message_text, event.input))),
                    "reasoning_summary": reasoning_summary(response) if response else None,
                    "output": message_text(response) if response else "",
                }
            )
        elif event.event == "tool" and event.function == "web_search":
            steps.append(
                {
                    "input": format_tool_call(event.function, event.arguments),
                    "reasoning_summary": None,
                    "output": event.result,
                    "error": vars(event.error) if event.error else None,
                }
            )
    return steps


def has_failed_search(steps):
    """Return True if a search returned nothing, which is what a failed search does."""
    return any("error" in step and not str(step["output"] or "").strip() for step in steps)


def run_batch(batch, model, system_message, args, log_dir):
    """Run one batch of (sample_id, probe) pairs and return {sample_id: [sample, ...]}.

    A sample is one answer of the model to one probe.
    """
    task = Task(
        dataset=MemoryDataset(
            [Sample(id=sample_id, input=probe[2]) for sample_id, probe in batch]
        ),
        solver=[use_tools(web_search(), tool_choice="auto"), generate()],
    )
    logs = inspect_eval(
        task,
        model=model,
        system_message=system_message,
        temperature=TEMPERATURE,
        reasoning_effort=args.effort,
        max_tool_output=MAX_TOOL_OUTPUT,
        epochs=args.num_samples,
        max_connections=args.concurrency,
        max_retries=MAX_RETRIES,
        # A probe that fails must not stop the others in its batch.
        fail_on_error=False,
        continue_on_fail=True,
        score=False,
        log_dir=str(log_dir),
        log_format="eval",
        log_samples=True,
        log_model_api=True,
        display="plain",
        metadata={"answer_output": args.output},
    )
    samples = {}
    for sample in sorted(logs[0].samples or [], key=lambda s: s.epoch):
        samples.setdefault(str(sample.id), []).append(sample)
    return samples


def write_records(path, records):
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Answer probes with a model in the shared harness (Inspect AI with Exa search)"
    )
    parser.add_argument("--input", required=True, help="Probes JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--provider", choices=sorted(PROVIDERS), required=True, help="API provider"
    )
    parser.add_argument(
        "--model", help="Model name (default: gpt-5.4 for openai, gemini-3.1-pro-preview for gemini)"
    )
    parser.add_argument(
        "--knowledge-cutoff",
        required=True,
        help="The model's knowledge cutoff as YYYY-MM-DD, stated in the system prompt",
    )
    parser.add_argument(
        "--effort",
        help="Reasoning effort for openai (none, low, medium, high, xhigh) or thinking "
        "level for gemini (minimal, low, medium, high) (default: the provider's default)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1, help="Number of answers per probe (default: 1)"
    )
    parser.add_argument(
        "--trace-output", help="Also write the steps of every answer to this JSONL"
    )
    parser.add_argument(
        "--inspect-log-dir",
        help="Directory for Inspect's own logs (default: inspect_logs next to the output)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of model requests to run in parallel (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of probes to run before their answers are saved (default: 5)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip probes already in the output file"
    )
    args = parser.parse_args()

    provider = PROVIDERS[args.provider]
    model = f"{provider['inspect_prefix']}/{args.model or provider['default_model']}"
    if args.effort is not None and args.effort not in provider["effort_choices"]:
        parser.error(
            f"--effort for {args.provider} must be one of {', '.join(provider['effort_choices'])}"
        )
    try:
        date.fromisoformat(args.knowledge_cutoff)
    except ValueError:
        parser.error("--knowledge-cutoff must be a date in the form YYYY-MM-DD")
    for name in ("num_samples", "concurrency", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    for env_var in (provider["api_key_env"], "EXA_API_KEY"):
        if not os.environ.get(env_var):
            parser.error(f"Set the {env_var} environment variable")
    # Inspect reads the Gemini key from GOOGLE_API_KEY.
    if args.provider == "gemini":
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    output_file = Path(args.output)
    log_dir = Path(args.inspect_log_dir or output_file.parent / "inspect_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    system_message = SYSTEM_MESSAGE.format(
        current_date=date.today().isoformat(), knowledge_cutoff=args.knowledge_cutoff
    )

    # A probe's sample ID is its position in the input file, so that it stays
    # the same when a run is resumed.
    processed = load_processed_probes(output_file) if args.resume else set()
    probes = [
        (str(index), probe)
        for index, probe in enumerate(load_probes(args.input), 1)
        if (probe[0], probe[2]) not in processed
    ]
    if not args.resume:
        for path in filter(None, (args.output, args.trace_output)):
            open(path, "w", encoding="utf-8").close()

    logger.info(f"Model: {model}, effort: {args.effort or 'provider default'}")
    logger.info(f"System message: {system_message}")
    logger.info(f"Probes to answer: {len(probes)} ({args.num_samples} answers each)")

    successful = 0
    failed = 0
    batches = [probes[i : i + args.batch_size] for i in range(0, len(probes), args.batch_size)]
    try:
        for batch_num, batch in enumerate(batches, 1):
            samples = run_batch(batch, model, system_message, args, log_dir)
            records = []
            trace_records = []
            for sample_id, (wikidata_id, name, probe, fact_text) in batch:
                probe_samples = samples.get(sample_id, [])
                steps = [trace_steps(sample) for sample in probe_samples]
                # A sample that ended in an error can still hold a partial
                # answer, which is not used.
                answers = [
                    sample.output.completion.strip()
                    for sample in probe_samples
                    if not sample.error and sample.output.completion.strip()
                ]
                if not answers or any(map(has_failed_search, steps)):
                    logger.warning(f"No answer or a failed search for {name}: {probe[:60]}")
                    continue

                record = {
                    "wikidata_id": wikidata_id,
                    "name": name,
                    "question": probe,
                    "answers": answers,
                }
                if fact_text is not None:
                    record["fact_text"] = fact_text
                records.append(record)
                for sample, sample_steps in zip(probe_samples, steps):
                    trace_records.append(
                        {
                            "sample_id": sample_id,
                            "epoch": sample.epoch,
                            "question": probe,
                            "steps": sample_steps,
                        }
                    )

            write_records(output_file, records)
            if args.trace_output:
                write_records(args.trace_output, trace_records)
            successful += len(records)
            failed += len(batch) - len(records)
            logger.info(
                f"Batch {batch_num}/{len(batches)}: saved {len(records)}, failed {len(batch) - len(records)}"
            )
            if not records:
                # Every probe failing usually means that the Exa or model key
                # has stopped working, so further batches would fail too.
                logger.error("Every probe in the batch failed; stopping.")
                break
    except KeyboardInterrupt:
        logger.info("Interrupted. Finished batches have been saved; rerun with --resume to continue.")

    logger.info(f"Done. Successful: {successful}, Failed: {failed}")
    logger.info(f"Results saved to {output_file}")
    if failed:
        logger.error(f"{failed} probes failed and were left out; rerun with --resume to retry them.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
