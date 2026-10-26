"""Helpers shared by the scripts that read from Wikidata and Wikipedia."""
import json
import time

import requests

# Pause after every successful request, to stay well inside Wikimedia's limits.
SUCCESS_SLEEP_SECONDS = 1.0
# First wait after an HTTP 429; doubles on every further 429 for the same request.
RATE_LIMIT_SLEEP_SECONDS = 10.0
MAX_RATE_LIMIT_RETRIES = 6

WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


def get_json(url, *, headers, params=None, timeout=10):
    """GET a URL and return the parsed JSON, backing off on HTTP 429.

    A 429 response is retried after the server's Retry-After delay if it gives
    one, otherwise after an exponentially growing wait. Any other HTTP error
    raises immediately.
    """
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        if response.status_code == 429 and attempt < MAX_RATE_LIMIT_RETRIES:
            retry_after = response.headers.get("Retry-After")
            sleep_seconds = (
                float(retry_after)
                if retry_after and retry_after.isdigit()
                else RATE_LIMIT_SLEEP_SECONDS * (2 ** attempt)
            )
            print(
                f"  Rate limited by {url}; sleeping {sleep_seconds} seconds "
                f"before retry {attempt + 1}/{MAX_RATE_LIMIT_RETRIES}"
            )
            time.sleep(sleep_seconds)
            continue

        response.raise_for_status()
        data = response.json()
        time.sleep(SUCCESS_SLEEP_SECONDS)
        return data


def get_wikidata_entity(qid, *, headers, timeout=10):
    """Return a Wikidata item's entity record (labels, descriptions, sitelinks, ...).

    If the item has been merged into another, Wikidata redirects to that item
    and returns it under the new ID, so the single returned entity is used
    rather than looking up qid.
    """
    data = get_json(WIKIDATA_ENTITY_URL.format(qid=qid), headers=headers, timeout=timeout)
    return next(iter(data["entities"].values()))


def get_wikipedia_page(title, *, headers, follow_redirects=False, timeout=10):
    """Return {"title": ..., "extract": ...} for an English Wikipedia article, or None.

    The extract is the article's full plain text. None means the page does not
    exist or has no text. With follow_redirects, a redirect title resolves to
    its target article, and the target's title is returned.
    """
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,  # plain text rather than HTML
        "exsectionformat": "plain",  # section headings without wiki markup
    }
    if follow_redirects:
        params["redirects"] = 1
    data = get_json(WIKIPEDIA_API, params=params, headers=headers, timeout=timeout)
    # A single-title query returns exactly one page, keyed by its page ID
    # ("-1" if the title does not exist).
    page_id, page = next(iter(data["query"]["pages"].items()))
    if page_id == "-1" or not page.get("extract"):
        return None
    return {"title": page.get("title", title), "extract": page["extract"]}


def qid_from_uri(uri):
    """Return the Wikidata ID from an entity URI such as ".../entity/Q59215"."""
    return uri.split("/")[-1]


def get_word_count(text):
    """Return the number of whitespace-separated words in text."""
    return len(text.split())


def load_processed_ids(output_file, key="wikidata_id"):
    """Return the set of `key` values already written to a JSONL output file.

    Used to resume an interrupted run. A missing file gives an empty set, and
    malformed lines (e.g. a half-written last line) are reported and skipped.
    """
    processed = set()
    if not output_file.exists():
        return processed

    with output_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: malformed JSON in {output_file} line {line_num}: {e}")
                continue
            if record.get(key):
                processed.add(record[key])
    return processed
