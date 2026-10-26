#!/usr/bin/env python3
"""Answer probes with a model through its provider's own API (OpenAI or Gemini).

Reads a probe file (the output of the probe generation or of
filter_indirect_probes.py) and sends every probe to the model as it is, with
no system prompt. Each probe is answered --num-samples times. With
--enable-web-search the model can use the provider's own search tool, which
is the native harness of the paper. One line per probe is appended to the
output JSONL as soon as all of its answers are ready.

The pipeline uses this script twice:

    # Answers without search, three per probe, used to filter the probes.
    python scripts/generate_answers_native.py --provider openai --model gpt-5.4 \\
        --effort none --num-samples 3 --input indirect_probes_demises.jsonl \\
        --output answers_gpt-5.4.jsonl --trace-output answers_gpt-5.4_trace.jsonl

    # Answers with the provider's search tool, one per probe.
    python scripts/generate_answers_native.py --provider gemini \\
        --model gemini-3.1-pro-preview --effort high --enable-web-search \\
        --input filtered_indirect_probes_gemini-3.1-pro-preview.jsonl \\
        --output search_answers_gemini-3.1-pro-preview_high.jsonl \\
        --trace-output search_answers_gemini-3.1-pro-preview_high_trace.jsonl

The API key is read from OPENAI_API_KEY or GEMINI_API_KEY. If the run is
interrupted, rerun the same command with --resume to skip the probes already
in the output file.

To evaluate a model from another provider, add a request function next to
request_openai and request_gemini and register it in PROVIDERS.
"""
import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from utils.llm_utils import RateLimiter, call_with_retry, load_probes, load_processed_probes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def request_openai(client, model, probe, effort, enable_web_search, trace):
    """Send one probe to the OpenAI Responses API.

    Returns (answer, request, response), the last two in JSON form for the
    trace file.
    """
    request = {"model": model, "input": probe}
    if enable_web_search:
        request["tools"] = [{"type": "web_search"}]
    reasoning = {}
    if effort is not None:
        reasoning["effort"] = effort
    if trace:
        # Also return what the trace should record: reasoning summaries, and
        # the sources and results of every search.
        reasoning["summary"] = "auto"
        request["include"] = [
            "web_search_call.action.sources",
            "web_search_call.results",
            "reasoning.encrypted_content",
        ]
    if reasoning:
        request["reasoning"] = reasoning

    response = client.responses.create(**request)
    return (response.output_text or "").strip(), request, response.model_dump(mode="json")


def request_gemini(client, model, probe, effort, enable_web_search, trace):
    """Send one probe to the Gemini API.

    Returns (answer, request, response), the last two in JSON form for the
    trace file.
    """
    from google.genai import types

    config = types.GenerateContentConfig()
    if enable_web_search:
        config.tools = [types.Tool(google_search=types.GoogleSearch())]
    if effort is not None or trace:
        # include_thoughts returns summaries of the model's thinking, which
        # the trace records. They are not part of response.text.
        config.thinking_config = types.ThinkingConfig(
            thinking_level=effort, include_thoughts=True if trace else None
        )

    response = client.models.generate_content(model=model, contents=probe, config=config)
    request = {"model": model, "contents": probe, "config": config.model_dump(mode="json")}
    return (response.text or "").strip(), request, response.model_dump(mode="json")


def make_openai_client(api_key):
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def make_gemini_client(api_key):
    from google import genai

    return genai.Client(api_key=api_key)


# The effort values are the provider's own. The paper's low and high settings
# are none and high for GPT-5.4, and low and high for Gemini-3.1-Pro.
PROVIDERS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-5.4",
        "effort_choices": ("none", "low", "medium", "high", "xhigh"),
        "make_client": make_openai_client,
        "request": request_openai,
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "default_model": "gemini-3.1-pro-preview",
        "effort_choices": ("minimal", "low", "medium", "high"),
        "make_client": make_gemini_client,
        "request": request_gemini,
    },
}


def answer_probe_once(send_request, rate_limiter, api_name):
    """Return one sample for a probe: its answer, request, response and error.

    A failed call is recorded in "error" rather than raised, so that one probe
    cannot stop the run.
    """
    sample = {"request": None, "answer_text": "", "response": None, "error": None}
    try:
        answer, request, response = call_with_retry(send_request, rate_limiter, api_name)
        sample.update(answer_text=answer, request=request, response=response)
    except Exception as e:
        sample["error"] = str(e)
    return sample


def write_record(f, record):
    f.write(json.dumps(record) + "\n")
    f.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Answer probes with a model through its provider's own API"
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
        "--effort",
        help="Reasoning effort for openai (none, low, medium, high, xhigh) or thinking "
        "level for gemini (minimal, low, medium, high) (default: the provider's default)",
    )
    parser.add_argument(
        "--enable-web-search",
        action="store_true",
        help="Let the model use the provider's web search tool",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1, help="Number of answers per probe (default: 1)"
    )
    parser.add_argument(
        "--trace-output",
        help="Also write every request and raw response to this JSONL, one line per answer. "
        "This also asks the provider to return reasoning summaries.",
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
    args = parser.parse_args()

    provider = PROVIDERS[args.provider]
    model = args.model or provider["default_model"]
    if args.effort is not None and args.effort not in provider["effort_choices"]:
        parser.error(
            f"--effort for {args.provider} must be one of {', '.join(provider['effort_choices'])}"
        )
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    api_key = os.environ.get(provider["api_key_env"])
    if not api_key:
        parser.error(f"Set the {provider['api_key_env']} environment variable")

    output_file = Path(args.output)
    answered = load_processed_probes(output_file) if args.resume else set()
    probes = [p for p in load_probes(args.input) if (p[0], p[2]) not in answered]
    logger.info(f"Provider: {args.provider}, model: {model}")
    logger.info(f"Effort: {args.effort or 'provider default'}")
    logger.info(f"Web search: {'enabled' if args.enable_web_search else 'disabled'}")
    logger.info(f"Probes to answer: {len(probes)} ({args.num_samples} answers each)")

    client = provider["make_client"](api_key)
    rate_limiter = RateLimiter(args.rate_limit)
    trace = args.trace_output is not None

    # One request per answer, so every probe has num_samples requests. A
    # probe's line is written once all of its requests have finished.
    executor = ThreadPoolExecutor(max_workers=args.concurrency)
    futures = {}
    for probe_idx, (_, _, probe, _) in enumerate(probes):
        send_request = partial(
            provider["request"], client, model, probe, args.effort, args.enable_web_search, trace
        )
        for sample_idx in range(args.num_samples):
            future = executor.submit(answer_probe_once, send_request, rate_limiter, args.provider)
            futures[future] = (probe_idx, sample_idx)

    samples = [[None] * args.num_samples for _ in probes]
    num_pending = [args.num_samples] * len(probes)
    successful = 0
    done = 0
    trace_f = open(args.trace_output, "a", encoding="utf-8") if trace else None
    # Probes are appended in the order they finish, not in input order.
    with output_file.open("a", encoding="utf-8") as f:
        try:
            for future in as_completed(futures):
                probe_idx, sample_idx = futures.pop(future)
                samples[probe_idx][sample_idx] = future.result()
                num_pending[probe_idx] -= 1
                if num_pending[probe_idx] > 0:
                    continue

                wikidata_id, name, probe, fact_text = probes[probe_idx]
                probe_samples = samples[probe_idx]
                errors = [
                    f"sample {i}: {sample['error']}"
                    for i, sample in enumerate(probe_samples, 1)
                    if sample["error"]
                ]
                # A probe is written even if some of its requests failed, with
                # only the answers that were received.
                record = {
                    "wikidata_id": wikidata_id,
                    "name": name,
                    "question": probe,
                    "answers": [s["answer_text"] for s in probe_samples if not s["error"]],
                }
                if fact_text is not None:
                    record["fact_text"] = fact_text
                write_record(f, record)

                if trace:
                    for i, sample in enumerate(probe_samples, 1):
                        write_record(
                            trace_f,
                            {
                                "provider": args.provider,
                                "model": model,
                                "wikidata_id": wikidata_id,
                                "name": name,
                                "question": probe,
                                "sample_index": i,
                                **sample,
                            },
                        )

                samples[probe_idx] = None  # Free the responses, which can be large.
                done += 1
                if errors:
                    logger.error(f"[{done}/{len(probes)}] ✗ Failed for {name}: {'; '.join(errors)}")
                else:
                    successful += 1
                    logger.info(f"[{done}/{len(probes)}] ✓ Generated answer for {name}")
        except KeyboardInterrupt:
            # Drop the requests that have not started; those already running
            # finish in the background but their results are not written.
            executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Interrupted. Progress has been saved; rerun with --resume to continue.")
        finally:
            if trace_f:
                trace_f.close()

    logger.info(f"Done. Successful: {successful}, Failed: {done - successful}")
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
