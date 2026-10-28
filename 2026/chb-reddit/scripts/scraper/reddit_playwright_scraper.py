#!/usr/bin/env python3
"""
Reddit Student AI Scraper (Playwright) — bypasses Reddit's HTTP blocking by
using a real browser (Chromium) that solves www.reddit.com's JS challenge.

Collects posts and comments from subreddits about AI experiences.

Strategy (default — monthly collection):
  1. Splits the --start-date → --end-date range into monthly windows.
  2. For each month, fetches on-topic posts via search.json + timestamp filter (cloudsearch).
  3. Fetches comments ({permalink}.json) only for the posts found.
  4. Aggregates everything into a single *_raw.jsonl, deduplicates, applies tiered filtering, and exports.

Strategy (--legacy-listing): paginates new.json in a single pass (useful for short ranges).

Usage:
  python reddit_playwright_scraper.py
  python reddit_playwright_scraper.py --start-date 2025-07-01 --end-date 2026-06-30 \
      --subreddits Teachers professors --output corpus/completed
  python reddit_playwright_scraper.py --legacy-listing --start-date 2026-05-01 \
      --end-date 2026-06-30 --output corpus/teste

  # Reprocess an existing raw JSONL with improved cleaning (no re-scrape):
  python reddit_playwright_scraper.py --refilter corpus/year_raw.jsonl \
      --core-only --output corpus/year_clean

  # Clean corpus for IRaMuTeQ (no bots, monthly period, posts only):
  python reddit_playwright_scraper.py --refilter corpus/year_raw.jsonl \
      --core-only --posts-only --iramutec-period month --output corpus/year_posts
"""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
import urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    from langdetect import detect as _langdetect_detect, LangDetectException
    from langdetect import DetectorFactory as _DetectorFactory
    _DetectorFactory.seed = 42  # deterministic results
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------

DEFAULT_SUBREDDITS = [
    "Teachers",
    "professors",
]

DEFAULT_DAYS = 90

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def sleep_with_jitter(base: float) -> None:
    time.sleep(base + random.uniform(0.3, 1.0))


def clean_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split()).replace("\x00", "")


# Compiled once, for stripping textual noise.
_RE_MD_LINK  = re.compile(r'\[([^\]]*)\]\([^)]*\)')    # [label](url) → label
_RE_URL      = re.compile(r'https?://\S+|www\.\S+')
_RE_MD_FMT   = re.compile(r'[*_~`#>|\\]')              # markdown markers
_RE_MULTISPC = re.compile(r' {2,}')
_RE_URL_RAW  = re.compile(r'https?://')                 # URL detector (for metadata)


def clean_for_analysis(text: str) -> str:
    """Removes URLs, markdown, and other noise before textual analysis.

    Sequence:
      1. Decodes HTML entities (&amp; etc.)
      2. Converts markdown links [label](url) → label
      3. Removes bare URLs
      4. Removes markdown markers (* _ ~ ` # > | \\)
      5. Normalizes whitespace
    """
    import html as _html
    text = _html.unescape(text)
    text = _RE_MD_LINK.sub(r'\1', text)
    text = _RE_URL.sub(' ', text)
    text = _RE_MD_FMT.sub(' ', text)
    text = text.replace('\x00', '')
    return _RE_MULTISPC.sub(' ', ' '.join(text.split())).strip()


def parse_date(s: str, end_of_day: bool = False) -> float:
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    if end_of_day:
        dt += timedelta(days=1) - timedelta(seconds=1)
    return dt.timestamp()


def default_start_date() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=DEFAULT_DAYS)).strftime("%Y-%m-%d")


def default_end_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def word_count(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Bots and boilerplate
# ---------------------------------------------------------------------------

BOT_AUTHORS: frozenset = frozenset({
    "AutoModerator", "RemindMeBot", "sneakpeek_bot", "transcribot",
    "RepostSleuthBot", "ModMailBot", "reddit", "BotDefense",
    "Remindmebot", "WikiSummarizerBot", "TotesMessenger",
})

_NOISE_PATTERNS: list = [
    re.compile(r, re.IGNORECASE) for r in [
        r"your post/comment was removed",
        r"I will be messaging you\b",
        r"defaulted to one day",
        r"wolframalpha\.com/input",
        r"^I am a bot",
        r"this action was performed automatically",
        r"please contact the moderators of this subreddit",
        r"\bRemindMe\b.*?\b(hour|day|week|month|year)s?\b",
    ]
]


def is_bot_author(author: str) -> bool:
    """Returns True if the author is a known bot or has a 'bot' suffix."""
    if not author or author in ("[deleted]", "[removed]"):
        return False
    return author in BOT_AUTHORS or author.lower().endswith("bot")


def iter_month_windows(start_date: str, end_date: str):
    """Generates monthly windows (label, start_ts, end_ts, date_start, date_end).

    Order: from the oldest month to the most recent. The first and last
    month respect the exact CLI dates.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    overall_start_ts = parse_date(start_date, end_of_day=False)
    overall_end_ts = parse_date(end_date, end_of_day=True)

    cursor = start_dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while cursor <= end_dt:
        label = cursor.strftime("%Y-%m")
        if cursor.month == 12:
            next_month = cursor.replace(year=cursor.year + 1, month=1, day=1)
        else:
            next_month = cursor.replace(month=cursor.month + 1, day=1)
        month_end_dt = next_month - timedelta(seconds=1)

        window_start_ts = max(cursor.timestamp(), overall_start_ts)
        window_end_ts = min(month_end_dt.timestamp(), overall_end_ts)

        if window_start_ts <= window_end_ts:
            date_start = datetime.fromtimestamp(
                window_start_ts, timezone.utc
            ).strftime("%Y-%m-%d")
            date_end = datetime.fromtimestamp(
                window_end_ts, timezone.utc
            ).strftime("%Y-%m-%d")
            yield label, window_start_ts, window_end_ts, date_start, date_end

        cursor = next_month


# Terms used in the monthly searches (plain search, no OAuth).
# Each search is run separately and the results are deduplicated.
MONTHLY_SEARCH_TERMS = [
    # Generic AI terms
    "ChatGPT", "AI", "artificial intelligence", "generative AI",
    "LLM", "chatbot",
    # AI models and platforms
    "Claude", "OpenAI", "Gemini", "Copilot", "Perplexity", "DeepSeek",
    "NotebookLM",
    # Detection and academic-writing tools
    "Turnitin", "GPTZero", "Grammarly", "Quillbot",
]


# ---------------------------------------------------------------------------
# Playwright-based scraper (www.reddit.com)
# ---------------------------------------------------------------------------

class RedditPlaywrightScraper:
    BASE = "https://www.reddit.com"

    def __init__(self, page, delay: float, start_ts: float, end_ts: float,
                 scrolls: int, max_retries: int = 4, cooldown: float = 12.0):
        self.page = page
        self.delay = delay
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.scrolls = scrolls
        self.max_retries = max_retries
        self.cooldown = cooldown
        self._warmed = False

    # --- downloads JSON from a URL using the browser context (with the challenge cookie) ---

    # Statuses that indicate a rate limit/instability and justify a new backoff+retry.
    RETRY_STATUSES = {400, 403, 429, 500, 502, 503, 504}

    def _fetch_json(self, url: str):
        """Downloads JSON; on a rate-limit/instability status, waits progressively and retries."""
        last_status = None
        for attempt in range(self.max_retries + 1):
            resp = self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
            status = resp.status if resp else None
            last_status = status
            if status == 200:
                txt = self.page.inner_text("body")
                return json.loads(txt)
            if status in self.RETRY_STATUSES and attempt < self.max_retries:
                # exponential backoff with jitter: cooldown, 2x, 4x, ...
                wait = self.cooldown * (2 ** attempt) + random.uniform(0.5, 2.0)
                log(f"  HTTP {status} (rate limit/instability) — waiting {wait:.0f}s "
                    f"and retrying ({attempt + 1}/{self.max_retries})", "WARN")
                time.sleep(wait)
                continue
            break
        raise RuntimeError(f"HTTP {last_status}")

    def _warmup(self, subreddit: str) -> bool:
        """The first navigation solves the JS challenge and sets the session cookie.

        Resilient to network instability: retries with backoff and returns
        True/False instead of raising an exception (which would abort the
        whole collection run).
        """
        url = f"{self.BASE}/r/{subreddit}/"
        for attempt in range(self.max_retries + 1):
            try:
                self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
                self.page.wait_for_timeout(2500)
                self._warmed = True
                return True
            except Exception as e:
                if attempt < self.max_retries:
                    wait = self.cooldown * (2 ** attempt) + random.uniform(0.5, 2.0)
                    log(f"  Warmup failed for r/{subreddit} ({e}) — waiting {wait:.0f}s "
                        f"and retrying ({attempt + 1}/{self.max_retries})", "WARN")
                    time.sleep(wait)
                    continue
                log(f"  Warmup for r/{subreddit} failed after {self.max_retries} attempts: {e}", "ERROR")
                return False
        return False

    # --- Listing: collects posts via new.json (in bulk) ---

    def list_posts(self, subreddit: str) -> list[dict]:
        """Collects posts by paginating ``new.json`` until past ``start_ts``.

        Ordered from most recent to oldest; stops when a post older than
        ``start_ts`` is found. If ``self.scrolls > 0``, applies a safety cap
        on the number of pages (~100 items each). Returns only posts within
        the ``[start_ts, end_ts]`` range.

        Stores the permalink's raw path in ``_permalink_path`` so the caller
        only fetches comments for the posts that are worth it.
        """
        posts: list[dict] = []
        seen: set[str] = set()
        after: str | None = None
        page_idx = 0
        stop_reason = "unknown"

        while True:
            if self.scrolls > 0 and page_idx >= self.scrolls:
                stop_reason = f"{self.scrolls}-page cap"
                log(f"  r/{subreddit}: {stop_reason} — raise --scrolls if history is missing",
                    "WARN")
                break

            url = f"{self.BASE}/r/{subreddit}/new.json?limit=100"
            if after:
                url += f"&after={urllib.parse.quote(after)}"
            try:
                data = self._fetch_json(url)
            except Exception as e:
                stop_reason = f"error on page {page_idx + 1}"
                log(f"  Error listing r/{subreddit} (page {page_idx + 1}): {e}", "ERROR")
                break

            page_idx += 1
            listing = data.get("data", {}) if isinstance(data, dict) else {}
            children = listing.get("children", [])
            if not children:
                stop_reason = "no more results"
                break

            past_start = False
            for ch in children:
                if ch.get("kind") != "t3":
                    continue
                d = ch.get("data", {})
                created = d.get("created_utc", 0)
                if created < self.start_ts:
                    past_start = True
                    break
                if created > self.end_ts:
                    continue
                pid = d.get("id", "")
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                parsed = self._parse_post(d)
                parsed["_permalink_path"] = d.get("permalink", "")
                posts.append(parsed)

            if past_start:
                stop_reason = "reached --start-date"
                break

            after = listing.get("after")
            if not after:
                stop_reason = "Reddit ran out of pagination"
                break

            if page_idx % 10 == 0:
                log(f"  r/{subreddit}: page {page_idx}, {len(posts)} posts in range…")

            sleep_with_jitter(self.delay)

        log(f"  r/{subreddit} → {len(posts)} posts in range "
            f"({page_idx} page(s), stopped: {stop_reason})")
        return posts

    # --- Monthly search: on-topic posts via search.json + timestamp ---

    def _ingest_search_children(
        self,
        children: list,
        seen: set[str],
        posts: list[dict],
        start_ts: float,
        end_ts: float,
    ) -> None:
        for ch in children:
            if ch.get("kind") != "t3":
                continue
            d = ch.get("data", {})
            created = d.get("created_utc", 0)
            if not (start_ts <= created <= end_ts):
                continue
            pid = d.get("id", "")
            if not pid or pid in seen:
                continue
            parsed = self._parse_post(d)
            if not mentions_ai(parsed):
                continue
            seen.add(pid)
            parsed["_permalink_path"] = d.get("permalink", "")
            posts.append(parsed)

    def _paginate_search_plain(
        self,
        subreddit: str,
        term: str,
        start_ts: float,
        end_ts: float,
    ) -> tuple[list[dict], str]:
        """Paginates search.json with a plain search (no OAuth, no cloudsearch).

        Sorts by ``new`` and stops paginating once the oldest item on the
        page predates ``start_ts`` — efficient for any historical window.
        """
        posts: list[dict] = []
        seen: set[str] = set()
        after: str | None = None
        page_idx = 0
        max_pages = self.scrolls if self.scrolls > 0 else 20
        stop_reason = "unknown"

        while True:
            if page_idx >= max_pages:
                stop_reason = f"{max_pages}-page cap"
                break

            params = (
                f"q={urllib.parse.quote(term)}"
                f"&restrict_sr=1&sort=new&t=all&limit=100"
            )
            if after:
                params += f"&after={urllib.parse.quote(after)}"
            url = f"{self.BASE}/r/{subreddit}/search.json?{params}"

            try:
                data = self._fetch_json(url)
            except Exception as e:
                stop_reason = f"error on page {page_idx + 1}"
                log(f"  Error searching '{term}' r/{subreddit} (p.{page_idx + 1}): {e}", "ERROR")
                break

            page_idx += 1
            listing = data.get("data", {}) if isinstance(data, dict) else {}
            children = listing.get("children", [])
            if not children:
                stop_reason = "no more results"
                break

            # Stop paginating once the oldest item on the page predates the window.
            page_oldest_ts = min(
                (
                    ch.get("data", {}).get("created_utc", float("inf"))
                    for ch in children
                    if ch.get("kind") == "t3"
                ),
                default=float("inf"),
            )

            self._ingest_search_children(children, seen, posts, start_ts, end_ts)

            if page_oldest_ts < start_ts:
                stop_reason = "reached --start-date"
                break

            after = listing.get("after")
            if not after:
                stop_reason = "reached the end of the search"
                break

            if page_idx % 5 == 0:
                log(f"  '{term}' r/{subreddit}: page {page_idx}, {len(posts)} posts…")

            sleep_with_jitter(self.delay)

        return posts, stop_reason

    def search_posts_month(
        self,
        subreddit: str,
        start_ts: float,
        end_ts: float,
    ) -> list[dict]:
        """Collects on-topic posts within a monthly window via search.json (plain, no OAuth).

        Runs one search per entry in MONTHLY_SEARCH_TERMS, deduplicating by
        ID. Does not use cloudsearch (requires OAuth and has a 512-char
        query limit).
        """
        merged: list[dict] = []
        seen: set[str] = set()

        for term in MONTHLY_SEARCH_TERMS:
            try:
                batch, reason = self._paginate_search_plain(
                    subreddit, term, start_ts, end_ts,
                )
            except Exception as e:
                log(f"  Error searching '{term}': {e}", "WARN")
                batch = []
                reason = "error"

            new_posts = [p for p in batch if p["id"] not in seen]
            for p in new_posts:
                seen.add(p["id"])
            merged.extend(new_posts)
            if new_posts:
                log(f"  '{term}': +{len(new_posts)} new posts ({reason})")

        log(f"  r/{subreddit} → {len(merged)} on-topic posts in the window")
        return merged

    # --- Comments for a post (.json) ---

    def fetch_comments(self, permalink_path: str, limit: int,
                       post_id: str = "") -> list[dict]:
        """Downloads only the comments for a post (the post itself came from the listing)."""
        url = f"{self.BASE}{permalink_path.rstrip('/')}.json?limit={limit}&sort=new&depth=3"
        try:
            data = self._fetch_json(url)
            if not isinstance(data, list) or len(data) < 2:
                return []
            comment_children = data[1].get("data", {}).get("children", [])
            comments = self._flatten_comments(comment_children)
            for c in comments:
                c["post_id"] = post_id
            return comments
        except Exception as e:
            log(f"  Error fetching comments {permalink_path}: {e}", "WARN")
            return []
        finally:
            sleep_with_jitter(self.delay)

    # --- Parsers (same as the original scraper) ---

    def _parse_post(self, d: dict) -> dict:
        author = d.get("author", "[deleted]")
        title  = clean_text(d.get("title", ""))
        text   = clean_text(d.get("selftext", ""))
        raw    = f"{title} {text}"
        return {
            "source": "reddit",
            "type": "post",
            "id": d.get("id", ""),
            "post_id": "",
            "subreddit": d.get("subreddit", ""),
            "author": author,
            "title": title,
            "text": text,
            "score": d.get("score", 0),
            "num_comments": d.get("num_comments", 0),
            "permalink": f"https://reddit.com{d.get('permalink', '')}",
            "created_utc": d.get("created_utc", 0),
            "created_date": datetime.fromtimestamp(
                d.get("created_utc", 0), timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "word_count": word_count(raw),
            "has_url": bool(_RE_URL_RAW.search(raw)),
            "is_bot": is_bot_author(author),
        }

    def _parse_comment(self, d: dict) -> dict:
        author = d.get("author", "[deleted]")
        text   = clean_text(d.get("body", ""))
        return {
            "source": "reddit",
            "type": "comment",
            "id": d.get("id", ""),
            "post_id": "",
            "subreddit": d.get("subreddit", ""),
            "author": author,
            "title": "",
            "text": text,
            "score": d.get("score", 0),
            "num_comments": 0,
            "permalink": f"https://reddit.com{d.get('permalink', '')}",
            "created_utc": d.get("created_utc", 0),
            "created_date": datetime.fromtimestamp(
                d.get("created_utc", 0), timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "word_count": word_count(text),
            "has_url": bool(_RE_URL_RAW.search(text)),
            "is_bot": is_bot_author(author),
        }

    def _flatten_comments(self, children: list, depth: int = 0) -> list[dict]:
        results = []
        if depth > 4:
            return results
        for child in children:
            if child.get("kind") != "t1":
                continue
            d = child.get("data", {})
            body = d.get("body", "")
            if body and body not in ("[deleted]", "[removed]") and len(body) > 20:
                results.append(self._parse_comment(d))
            replies = d.get("replies", {})
            if isinstance(replies, dict):
                nested = replies.get("data", {}).get("children", [])
                results.extend(self._flatten_comments(nested, depth + 1))
        return results

    def is_in_range(self, item: dict) -> bool:
        ts = item.get("created_utc", 0)
        return self.start_ts <= ts <= self.end_ts


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def is_english(text: str, min_chars: int = 40) -> bool:
    """Returns True if the text is detected as English.

    Texts with fewer than ``min_chars`` characters are accepted without
    verification, since there is not enough signal for reliable detection.
    If the library fails, the item is kept (conservative policy).
    Requires langdetect to be installed (``pip install langdetect``).
    """
    if not _LANGDETECT_AVAILABLE:
        return True
    text = text.strip()
    if len(text) < min_chars:
        return True
    try:
        return _langdetect_detect(text) == "en"
    except LangDetectException:
        return True


# ---------------------------------------------------------------------------
# Relevance filter (English)
# ---------------------------------------------------------------------------

# Relevance terms — aligned with Table 1 of the paper.
AI_PATTERNS = [
    # Generic concepts (Table 1 — Generic concepts)
    r"artificial intelligence", r"generative ai", r"\bllms?\b",
    r"\bchatbot\w*\b",
    # Tools and platforms (Table 1 — Tools)
    r"chat\s?gpt", r"\bopenai\b", r"\bgemini\b", r"\bcopilot\b",
    r"\bperplexity\b", r"\bnotebook\s?lm\b", r"\bgptzero\b",
    r"\bturnitin\b", r"\bgrammarly\b", r"\bquillbot\b",
    r"\bclaude\b", r"\bdeepseek\b",
]

BARE_AI = re.compile(r"\b(AI|ai)\b")


def is_noise_item(item: dict) -> bool:
    """Returns True for bot items or items with moderation/reminder boilerplate."""
    if item.get("is_bot") or is_bot_author(item.get("author", "")):
        return True
    text = item.get("text", "") + " " + item.get("title", "")
    return any(p.search(text) for p in _NOISE_PATTERNS)


def _count_matches(patterns: list[str], text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text))


def item_text(item: dict) -> str:
    return (item.get("text", "") + " " + item.get("title", "")).strip()


def mentions_ai(item: dict) -> bool:
    """Light gate deciding whether it's worth fetching a post's comments.

    Since fetching comments costs ~--delay per post (the bottleneck), we
    only do this for threads that mention AI in the title/selftext (bare AI
    or a named tool). The finer filter is applied per comment afterward.
    """
    text = item_text(item)
    return bool(BARE_AI.search(text)) or _count_matches(AI_PATTERNS, text.lower()) > 0


def mentions_strong_ai(item: dict) -> bool:
    """AI mention via a named term (not counting the bare word 'AI' alone)."""
    return _count_matches(AI_PATTERNS, item_text(item).lower()) > 0


def is_valid_item(item: dict, min_words: int) -> bool:
    text = item_text(item)
    if word_count(text) < min_words:
        return False
    if "[deleted]" in text or "[removed]" in text:
        return False
    return True


def is_relevant(item: dict, min_words: int = 40) -> bool:
    """Compatibility: equivalent to the ``core`` tier."""
    return is_valid_item(item, min_words) and mentions_ai(item)


def _strong_post_ids(items: list[dict]) -> set[str]:
    return {
        item["id"] for item in items
        if item.get("type") == "post" and mentions_strong_ai(item)
    }


def dedup_by_text(items: list[dict]) -> list[dict]:
    """Removes exact-text duplicates (repeated bot/mod boilerplate).

    Keeps the first occurrence of each normalized title+text combination.
    """
    seen: set[str] = set()
    result: list[dict] = []
    for item in items:
        key = (item.get("title", "") + "\x00" + item.get("text", "")).strip()
        h = hashlib.md5(key.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(item)
    n_removed = len(items) - len(result)
    if n_removed:
        log(f"Text deduplication: {n_removed} duplicates removed")
    return result


def classify_tier(
    item: dict,
    strong_post_ids: set[str],
    min_words: int,
    min_words_comment: int | None = None,
) -> str | None:
    """Classifies relevance into tiers.

    - ``core``: explicit AI mention + minimum word count.
    - ``thread``: a comment in a thread whose post has a strong AI term.

    ``min_words_comment`` applies a different threshold for comments; if
    None, ``min_words`` is used for everything.
    """
    mw = (
        min_words_comment
        if min_words_comment is not None and item.get("type") == "comment"
        else min_words
    )
    if not is_valid_item(item, mw):
        return None
    if mentions_ai(item):
        return "core"
    if (
        item.get("type") == "comment"
        and item.get("post_id") in strong_post_ids
    ):
        return "thread"
    return None


def apply_tiered_filter(
    items: list[dict],
    min_words: int,
    *,
    include_thread: bool = True,
    exclude_bots: bool = True,
    posts_only: bool = False,
    min_words_comment: int | None = None,
    english_only: bool = True,
) -> list[dict]:
    """Filters items and annotates each one with ``relevance_tier``.

    Additional parameters:
    - ``exclude_bots``: removes AutoModerator/RemindMeBot items and mod boilerplate.
    - ``posts_only``: discards comments, keeps only posts.
    - ``min_words_comment``: comment-specific word-count threshold.
    - ``english_only``: removes items not detected as English (requires langdetect).
    """
    before_noise = len(items)
    if exclude_bots:
        items = [i for i in items if not is_noise_item(i)]
        n_noise = before_noise - len(items)
        if n_noise:
            log(f"Bots/boilerplate removed: {n_noise} items")

    if english_only:
        if _LANGDETECT_AVAILABLE:
            before_lang = len(items)
            items = [i for i in items if is_english(item_text(i))]
            n_lang = before_lang - len(items)
            if n_lang:
                log(f"Language filter: {n_lang} non-English items removed")
        else:
            log("WARNING: --english-only is active but langdetect is not installed "
                "(pip install langdetect). Language filter skipped.", "WARN")

    items = dedup_by_text(items)

    if posts_only:
        items = [i for i in items if i.get("type") == "post"]
        log(f"Posts-only mode: {len(items)} posts kept")

    strong_posts = _strong_post_ids(items)
    allowed = {"core", "thread"} if include_thread else {"core"}
    filtered: list[dict] = []
    counts: dict[str, int] = {"core": 0, "thread": 0}

    for item in items:
        tier = classify_tier(item, strong_posts, min_words, min_words_comment)
        if tier not in allowed:
            continue
        out = dict(item)
        out["relevance_tier"] = tier
        filtered.append(out)
        counts[tier] += 1

    log(
        f"Tiered filter: {len(filtered)} items "
        f"(core={counts['core']}, thread={counts['thread']})"
    )
    return filtered


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "source", "type", "id", "post_id", "subreddit", "author",
    "title", "text", "score", "num_comments",
    "permalink", "created_date", "relevance_tier",
    "word_count", "has_url", "is_bot",
]


def save_csv(items: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(items)
    log(f"CSV saved: {path} ({len(items)} items)")


def save_jsonl(items: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            row = {k: item[k] for k in CSV_FIELDS if k in item}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"JSONL saved: {path} ({len(items)} items)")


def _clean_for_iramutec(text: str) -> str:
    """Cleans text for the IRaMuTeQ format.

    Applies ``clean_for_analysis`` (removes URLs, markdown, HTML entities)
    and ensures the result is a single line with no null characters.
    """
    text = clean_for_analysis(text)
    return " ".join(text.split()).strip()


def _iramutec_var(value: str) -> str:
    """Sanitizes an IRaMuTeQ starred-variable value.

    IRaMuTeQ's parser splits the '****' line on whitespace and uses the
    text before the first '_' as the variable name; spaces, '/', '-', and
    accents corrupt the parsing. We keep only [A-Za-z0-9], turning slashes
    into '_' and stripping everything else.
    """
    if not value:
        return ""
    value = value.replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_]", "", value)


def _periodo_from_date(
    created_date: str,
    created_utc: float = 0,
    granularity: str = "month",
) -> str:
    """Derives the period from the item's date.

    ``granularity``:
    - ``"day"``   → YYYYMMDD (e.g., 20260627) — 364 modalities/year, avoid in FCA
    - ``"month"`` → YYYYMM   (e.g., 202606)   — 12 modalities/year (default)
    - ``"none"``  → empty string (does not include *period_ in the header)
    """
    if granularity == "none":
        return ""
    length = 6 if granularity == "month" else 8
    if created_date:
        digits = re.sub(r"\D", "", created_date)
        if len(digits) >= length:
            return digits[:length]
    if created_utc:
        fmt = "%Y%m" if granularity == "month" else "%Y%m%d"
        return datetime.fromtimestamp(created_utc, timezone.utc).strftime(fmt)
    return ""


def save_iramutec(
    items: list[dict],
    path: str,
    *,
    period_granularity: str = "month",
    include_id: bool = False,
) -> None:
    """Exports the corpus in IRaMuTeQ (.txt) format.

    Each item becomes a document with a starred-variable header followed
    by the cleaned text (no URLs, markdown, or null characters).

    Parameters:
    - ``period_granularity``: ``"month"`` (YYYYMM, default), ``"day"`` (YYYYMMDD)
      or ``"none"`` (omits *period_).
    - ``include_id``: if True, includes ``*id_<id>`` in the header. Off by
      default since unique IDs clutter the FCA with no analytical value.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            title = item.get("title", "").strip()
            body = item.get("text", "").strip()
            if item["type"] == "post" and title:
                raw_text = f"{title}. {body}" if body else title
            else:
                raw_text = body
            text = _clean_for_iramutec(raw_text)
            if not text:
                continue

            parts = ["****"]
            if include_id:
                parts.append(f"*id_{item['id']}")
            period = _periodo_from_date(
                item.get("created_date", ""),
                item.get("created_utc", 0),
                granularity=period_granularity,
            )
            if period:
                parts.append(f"*period_{period}")
            parts.append(f"*r_{_iramutec_var(item.get('subreddit', ''))}")
            parts.append(f"*type_{item.get('type', '')}")
            tier = item.get("relevance_tier", "")
            if tier:
                parts.append(f"*tier_{tier}")

            f.write(" ".join(parts) + "\n")
            f.write(f"{text}\n\n")
            count += 1
    log(f"IRaMuTeQ corpus saved: {path} ({count} texts, period={period_granularity})")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def finalize_exports(raw_items: list[dict], base: str, args: argparse.Namespace) -> None:
    """Sorts, applies the tiered filter, and writes CSV/JSONL/IRaMuTeQ."""
    post_dates: dict[str, float] = {}
    for item in raw_items:
        if item["type"] == "post":
            post_dates[item["id"]] = item.get("created_utc", 0)

    def _thread_sort_key(item: dict) -> tuple:
        if item["type"] == "post":
            thread_ts = item.get("created_utc", 0)
            order = 0
        else:
            thread_ts = post_dates.get(item.get("post_id", ""), 0)
            order = 1
        return (-thread_ts, order, -item.get("created_utc", 0))

    raw_items.sort(key=_thread_sort_key)

    all_items = apply_tiered_filter(
        raw_items,
        args.min_words,
        include_thread=not args.core_only,
        exclude_bots=args.exclude_bots,
        posts_only=getattr(args, "posts_only", False),
        min_words_comment=getattr(args, "min_words_comment", None),
        english_only=getattr(args, "english_only", True),
    )
    all_items.sort(key=_thread_sort_key)

    save_csv(all_items, f"{base}.csv")
    save_jsonl(all_items, f"{base}.jsonl")
    save_iramutec(
        all_items,
        f"{base}_iramutec.txt",
        period_granularity=getattr(args, "iramutec_period", "month"),
        include_id=getattr(args, "iramutec_include_id", False),
    )

    log("\n--- PREVIEW (5 most recent) ---")
    for i, item in enumerate(all_items[:5], 1):
        tier = item.get("relevance_tier", "?")
        print(
            f"\n[{i}] {item['type'].upper()} | tier={tier} | "
            f"r/{item['subreddit']} | {item['created_date']} | score={item['score']}"
        )
        print(f"    Author: {item['author']}")
        if item["title"]:
            print(f"    Title: {item['title'][:100]}")
        print(f"    Text: {item['text'][:300]}...")
        print(f"    Link: {item['permalink']}")


def collect_posts_and_comments(
    scraper: RedditPlaywrightScraper,
    posts: list[dict],
    args: argparse.Namespace,
    seen_ids: set[str],
    record,
    raw_count: list[int],
    *,
    posts_on_topic: bool = False,
) -> int:
    """Records posts/comments; returns how many new items were added."""
    before = raw_count[0]

    if args.mode == "posts-then-comments":
        subreddit_posts = []
        for post in posts:
            if post["id"] not in seen_ids:
                seen_ids.add(post["id"])
                record(post)
                subreddit_posts.append(post)

        relevant = (
            subreddit_posts if posts_on_topic
            else [p for p in subreddit_posts if is_relevant(p, args.min_words)]
        )
        log(f"  {len(relevant)}/{len(subreddit_posts)} posts — fetching comments")
        for post in relevant:
            comments = scraper.fetch_comments(
                post["_permalink_path"], limit=args.limit,
                post_id=post["id"],
            )
            for c in comments:
                if c["id"] not in seen_ids and scraper.is_in_range(c):
                    seen_ids.add(c["id"])
                    record(c)
    else:
        for post in posts:
            in_range = scraper.is_in_range(post)

            if (
                args.mode in ("posts", "both")
                and post["id"] not in seen_ids
                and in_range
            ):
                seen_ids.add(post["id"])
                record(post)

            fetch_comments = (
                args.mode in ("comments", "both")
                and in_range
                and (
                    posts_on_topic
                    or args.fetch_all_comments
                    or mentions_ai(post)
                )
            )
            if fetch_comments:
                comments = scraper.fetch_comments(
                    post["_permalink_path"], limit=args.limit,
                    post_id=post["id"],
                )
                for c in comments:
                    if c["id"] not in seen_ids and scraper.is_in_range(c):
                        seen_ids.add(c["id"])
                        record(c)

    return raw_count[0] - before


def run_legacy_collection(
    scraper: RedditPlaywrightScraper,
    args: argparse.Namespace,
    seen_ids: set[str],
    record,
    raw_count: list[int],
) -> None:
    """Collects in a single pass via new.json (previous behavior)."""
    for subreddit in args.subreddits:
        if not scraper._warmup(subreddit):
            log(f"  Skipping r/{subreddit} (warmup failed).", "WARN")
            continue
        log(f"→ r/{subreddit}")
        posts = scraper.list_posts(subreddit)
        n_new = collect_posts_and_comments(
            scraper, posts, args, seen_ids, record, raw_count,
        )
        log(f"  r/{subreddit}: +{n_new} items | cumulative: {raw_count[0]}")


def run_monthly_collection(
    scraper: RedditPlaywrightScraper,
    args: argparse.Namespace,
    seen_ids: set[str],
    record,
    raw_count: list[int],
) -> None:
    """Collects month by month via search.json, aggregating into a single file."""
    windows = list(iter_month_windows(args.start_date, args.end_date))
    first_label = windows[0][0] if windows else "?"
    last_label = windows[-1][0] if windows else "?"
    log(
        f"Monthly collection: {len(windows)} windows ({first_label} → {last_label}) "
        f"| subreddits: {', '.join(args.subreddits)}"
    )

    warmed: set[str] = set()

    for month_label, month_start_ts, month_end_ts, date_start, date_end in windows:
        scraper.start_ts = month_start_ts
        scraper.end_ts = month_end_ts
        month_before = raw_count[0]

        for subreddit in args.subreddits:
            if subreddit not in warmed:
                if not scraper._warmup(subreddit):
                    log(f"  Skipping r/{subreddit} (warmup failed).", "WARN")
                    continue
                warmed.add(subreddit)

            log(
                f">>> Collecting month {month_label} | r/{subreddit} "
                f"({date_start} → {date_end})"
            )
            posts = scraper.search_posts_month(subreddit, month_start_ts, month_end_ts)
            n_new = collect_posts_and_comments(
                scraper, posts, args, seen_ids, record, raw_count,
                posts_on_topic=True,
            )
            log(
                f"    Month {month_label} r/{subreddit}: +{n_new} items "
                f"| raw total: {raw_count[0]}"
            )

        month_new = raw_count[0] - month_before
        log(f"--- Month {month_label} complete: +{month_new} items | cumulative: {raw_count[0]} ---")


def run(args: argparse.Namespace) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.output or f"reddit_ai_students_{ts}"

    if args.refilter:
        raw_path = str(args.refilter)
        log(f"Reprocessing raw corpus: {raw_path}")
        raw_items = load_jsonl(raw_path)
        log(f"Loaded {len(raw_items)} raw items")
        finalize_exports(raw_items, base, args)
        return

    raw_items: list[dict] = []   # everything collected (in range, no relevance filter)
    seen_ids: set[str] = set()
    raw_path = f"{base}_raw.jsonl"

    log(f"Range: {args.start_date} → {args.end_date}")
    log(f"Subreddits: {args.subreddits}")
    _listing_mode = "legacy (new.json)" if args.legacy_listing else "monthly (search.json)"
    _comment_strategy = (
        "all in range" if args.fetch_all_comments
        else "only relevant posts (is_relevant)" if args.mode == "posts-then-comments"
        else "on-topic posts (mentions_ai or monthly search)"
    )
    _scrolls_label = "unlimited" if args.scrolls == 0 else str(args.scrolls)
    log(f"Listing: {_listing_mode}")
    log(f"Mode: {args.mode} | Limit per post: {args.limit} | Pages: {_scrolls_label} "
        f"| Comments: {_comment_strategy}")
    log(f"Writing raw items incrementally to: {raw_path}")
    log("=" * 60)

    Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
    raw_file = open(raw_path, "w", encoding="utf-8")
    raw_count = [0]

    def record(item: dict) -> None:
        """Accumulates and writes the raw item immediately (partial results on disk)."""
        raw_items.append(item)
        raw_count[0] += 1
        row = {k: item[k] for k in CSV_FIELDS if k in item}
        raw_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        raw_file.flush()

    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not args.headful)
            ctx = browser.new_context(user_agent=USER_AGENT, locale="en-US")
            page = ctx.new_page()
            page.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
            )

            scraper = RedditPlaywrightScraper(
                page, delay=args.delay,
                start_ts=args.start_ts, end_ts=args.end_ts,
                scrolls=args.scrolls,
                max_retries=args.max_retries, cooldown=args.cooldown,
            )

            try:
                if args.legacy_listing:
                    run_legacy_collection(
                        scraper, args, seen_ids, record, raw_count,
                    )
                else:
                    run_monthly_collection(
                        scraper, args, seen_ids, record, raw_count,
                    )
            except KeyboardInterrupt:
                log("Interrupted — finishing up with what was already collected.", "WARN")
            finally:
                browser.close()
    finally:
        raw_file.close()

    log("=" * 60)
    log(f"Total raw items collected: {len(raw_items)} in range (unique)")
    log(f"Raw items written to: {raw_path}")

    if not raw_items:
        log("No items found. Check the date range or the subreddits.", "WARN")
        return

    finalize_exports(raw_items, base, args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collects Reddit discourse about AI (via Playwright)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--start-date", type=str, default=default_start_date(),
                        dest="start_date",
                        help=f"Start date YYYY-MM-DD (default: {DEFAULT_DAYS} days ago)")
    parser.add_argument("--end-date", type=str, default=default_end_date(),
                        dest="end_date",
                        help="End date YYYY-MM-DD (default: today, end of day UTC)")
    parser.add_argument("--subreddits", nargs="+", default=DEFAULT_SUBREDDITS,
                        help="Subreddits to collect")
    parser.add_argument("--mode",
                        choices=["posts", "comments", "both", "posts-then-comments"],
                        default="both",
                        help="What to collect (default: both). "
                             "'posts-then-comments': collects all posts via new.json, "
                             "applies the heuristic filter (is_relevant), and only then fetches "
                             "comments for the posts that passed — better precision, "
                             "without wasting fetches on irrelevant posts.")
    parser.add_argument("--limit", type=int, default=50,
                        help="Maximum comments per post (default: 50)")
    parser.add_argument("--scrolls", type=int, default=0,
                        help="Optional cap on listing pages (new.json, ~100 items/page). "
                             "0 = paginate until --start-date (default). Use N>0 as a safety limit.")
    parser.add_argument("--fetch-all-comments", action="store_true", dest="fetch_all_comments",
                        help="Fetches comments for ALL posts in range, ignoring the "
                             "on-topic pre-filter (slower, more comprehensive)")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Pause between requests in seconds (default: 3.0)")
    parser.add_argument("--max-retries", type=int, default=4, dest="max_retries",
                        help="Extra attempts on receiving HTTP 429 (default: 4)")
    parser.add_argument("--cooldown", type=float, default=20.0,
                        help="Base backoff wait (s) on receiving 429 (default: 20)")
    parser.add_argument("--min-words", type=int, default=40, dest="min_words",
                        help="Minimum word count in the text (default: 40)")
    parser.add_argument("--min-words-comment", type=int, default=None,
                        dest="min_words_comment",
                        help="Minimum word count for comments (default: same as --min-words). "
                             "Useful to require more content only for comments, e.g.: --min-words-comment 60")
    parser.add_argument("--core-only", action="store_true", dest="core_only",
                        help="Exports only the core tier (explicit AI mention); "
                             "excludes comments inherited from the thread")
    parser.add_argument("--posts-only", action="store_true", dest="posts_only",
                        help="Exports only posts (discards all comments). "
                             "Produces a smaller, more cohesive corpus for IRaMuTeQ.")
    parser.add_argument("--exclude-bots", action="store_true", default=True,
                        dest="exclude_bots",
                        help="Removes bot items and moderation boilerplate (default: on)")
    parser.add_argument("--no-exclude-bots", action="store_false",
                        dest="exclude_bots",
                        help="Keeps bot items and moderation messages in the corpus")
    parser.add_argument("--english-only", action="store_true", default=True,
                        dest="english_only",
                        help="Removes items not detected as English via langdetect (default: on)")
    parser.add_argument("--no-english-only", action="store_false",
                        dest="english_only",
                        help="Keeps items in any language in the corpus")
    parser.add_argument("--iramutec-period",
                        choices=["day", "month", "none"], default="month",
                        dest="iramutec_period",
                        help="Granularity of the *period_ variable in IRaMuTeQ: "
                             "'month' (YYYYMM, default), 'day' (YYYYMMDD), or "
                             "'none' (omits *period_, avoids an FCA with too many modalities)")
    parser.add_argument("--iramutec-include-id", action="store_true",
                        dest="iramutec_include_id",
                        help="Includes *id_ in the IRaMuTeQ header (off by default — "
                             "unique IDs have no analytical value in the FCA)")
    parser.add_argument("--refilter", type=Path, default=None,
                        help="Reprocesses an existing raw JSONL without a new collection run")
    parser.add_argument("--legacy-listing", action="store_true", dest="legacy_listing",
                        help="Uses new.json in a single pass (short ranges); "
                             "default is monthly collection via search.json")
    parser.add_argument("--headful", action="store_true",
                        help="Opens the browser visibly (useful for debugging)")
    parser.add_argument("--output", type=str, default=None,
                        help="Base name of the output files (no extension)")
    args = parser.parse_args()

    try:
        args.start_ts = parse_date(args.start_date, end_of_day=False)
        args.end_ts = parse_date(args.end_date, end_of_day=True)
    except ValueError:
        parser.error("Invalid dates — use the YYYY-MM-DD format")

    if args.start_ts > args.end_ts:
        parser.error("--start-date must be earlier than or equal to --end-date")

    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        log("Interrupted.", "WARN")
        sys.exit(0)
