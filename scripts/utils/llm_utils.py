"""Helpers shared by the scripts that call LLM APIs."""
import json
import logging
import random
import threading
import time

MAX_ATTEMPTS = 3

logger = logging.getLogger(__name__)


class RateLimiter:
    """Space out the start of requests across all worker threads."""

    def __init__(self, min_interval_seconds):
        self.min_interval_seconds = min_interval_seconds
        self.last_call_time = 0.0
        self.lock = threading.Lock()

    def wait(self):
        """Block until at least min_interval_seconds after the previous request."""
        with self.lock:
            elapsed = time.monotonic() - self.last_call_time
            if elapsed < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - elapsed)
            self.last_call_time = time.monotonic()


def call_with_retry(send_request, rate_limiter, api_name):
    """Return send_request(), retrying failed calls.

    Authentication errors are raised at once, since retrying cannot fix them.
    """
    for attempt in range(MAX_ATTEMPTS):
        try:
            rate_limiter.wait()
            return send_request()
        except Exception as e:
            error = str(e).lower()
            is_auth_error = any(s in error for s in ("authentication", "unauthorized", "401"))
            if is_auth_error or attempt == MAX_ATTEMPTS - 1:
                raise
            # Wait 1s, then 2s, plus a little jitter so that concurrent
            # requests do not all retry at the same moment.
            wait = 2 ** attempt + random.uniform(0.0, 0.25)
            logger.warning(
                f"{api_name} call failed (attempt {attempt + 1}/{MAX_ATTEMPTS}): {e}. "
                f"Retrying in {wait:.2f}s"
            )
            time.sleep(wait)


def call_gemini(client, rate_limiter, prompt, model, thinking_level, temperature, parse=None):
    """Return Gemini's text response to a prompt, retrying failed calls.

    If parse is given, parse(text) is returned instead. A response that makes
    parse raise an exception is retried like a failed call.
    """
    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=temperature,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
    )

    def send_request():
        response = client.models.generate_content(model=model, contents=prompt, config=config)
        text = (response.text or "").strip()
        return parse(text) if parse else text

    return call_with_retry(send_request, rate_limiter, "Gemini")


def parse_json_response(response):
    """Parse a model's JSON response.

    The prompts ask for bare JSON, but models sometimes wrap it in a Markdown
    code fence, which is removed first.
    """
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    return json.loads(response.strip())


def load_probes(probes_file):
    """Return every probe in a probe file as (wikidata_id, name, probe, fact_text).

    The input has one line per event with its probes under "questions".
    fact_text is the statement of the ended position for tenures and None for
    demises. Events whose probe generation failed are skipped.
    """
    probes = []
    with open(probes_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Malformed JSON, skipping: {e}")
                continue
            missing = [k for k in ("wikidata_id", "name", "questions") if k not in entry]
            if missing:
                logger.warning(f"Line {line_num}: Missing fields {missing}, skipping")
                continue
            if not entry.get("success", True):
                continue
            if not isinstance(entry["questions"], list):
                logger.warning(f"Line {line_num}: 'questions' is not a list, skipping")
                continue

            for question in entry["questions"]:
                probe = question.get("question") if isinstance(question, dict) else None
                if not probe:
                    logger.warning(f"Line {line_num}: Empty or invalid question, skipping")
                    continue
                probes.append((entry["wikidata_id"], entry["name"], probe, entry.get("fact_text")))
    return probes


def load_processed_probes(output_file):
    """Return the (wikidata_id, probe) pairs already in an output file."""
    processed = set()
    if not output_file.exists():
        return processed
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if "wikidata_id" in entry and "question" in entry:
                processed.add((entry["wikidata_id"], entry["question"]))
    return processed
