"""
╔══════════════════════════════════════════════════════════════════╗
║         EchoChamberX – Bias & Polarization Detection System      ║
║              Web Scraping & Dataset Creation Module              ║
╚══════════════════════════════════════════════════════════════════╝

Module: echochamberx_scraper.py
Purpose: Collect real-world textual data (news articles and Reddit
         discussions) for a given topic and build a structured
         dataset ready for NLP / bias-detection pipelines.

Dependencies:
    pip install requests beautifulsoup4 pandas lxml
    pip install praw          # optional – Reddit API
    pip install selenium      # optional – dynamic pages

Fixes applied (v1.1):
    [FIX 1] Windows-safe timestamp in filenames  →  use %Y%m%d_%H%M%S
            instead of .isoformat() which produces colons (:) that
            Windows forbids in filenames → OSError [Errno 22].
    [FIX 2] RSS XML parser fallback chain        →  try "lxml-xml",
            then "lxml", then "html.parser" so the module works even
            when lxml is not installed (no more "Couldn't find a tree
            builder with features: xml" warning).
"""

# ─────────────────────────────────────────────────────────────────
# Standard & third-party imports
# ─────────────────────────────────────────────────────────────────
import re
import time
import random
import json
import logging
from datetime import datetime, timezone
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# Logging setup  (INFO level → console)
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("EchoChamberX")

# ─────────────────────────────────────────────────────────────────
# Global configuration
# ─────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

REQUEST_TIMEOUT  = 10
MAX_ARTICLES     = 25
MAX_REDDIT_POSTS = 25
DATASET_LIMIT    = 50
MIN_TEXT_LENGTH  = 80


# ─────────────────────────────────────────────────────────────────
# ✅ FIX 1 HELPER – Windows-safe timestamp
# ─────────────────────────────────────────────────────────────────
def _safe_timestamp() -> str:
    """
    Return a filesystem-safe timestamp string.

    WHY THIS EXISTS:
        datetime.isoformat() → "2026-03-23T17:49:11.093292+00:00"
        This contains colons (:) which are ILLEGAL in Windows filenames.
        OSError [Errno 22] is raised when pandas tries to open such a path.

    SOLUTION:
        Use strftime with %Y%m%d_%H%M%S → "20260323_174911"
        No colons, no dots, no special characters — safe on all OS.

    Returns:
        str: e.g. "20260323_174911"
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _readable_timestamp() -> str:
    """
    Return a human-readable ISO-like timestamp for storing IN the dataset
    (inside a CSV/JSON cell, not as a filename — colons are fine here).

    Returns:
        str: e.g. "2026-03-23T17:49:11Z"
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─────────────────────────────────────────────────────────────────
# ✅ FIX 2 HELPER – RSS XML parser with fallback chain
# ─────────────────────────────────────────────────────────────────
def _parse_xml(content: bytes) -> BeautifulSoup:
    """
    Parse XML/RSS content with a safe parser fallback chain.

    WHY THIS EXISTS:
        BeautifulSoup(content, "xml") requires lxml to be installed.
        If lxml is missing it raises:
            "Couldn't find a tree builder with the features you
             requested: xml. Do you need to install a parser library?"
        This caused the RSS scraper to fail silently and fall through
        to slower fallbacks.

    SOLUTION:
        Try parsers in order of preference:
          1. "lxml-xml"    – best for RSS/XML (requires lxml)
          2. "lxml"        – general lxml parser (also requires lxml)
          3. "html.parser" – built into Python stdlib, always available
                            (slightly less accurate for XML but works)

    Args:
        content (bytes): Raw response bytes from requests.

    Returns:
        BeautifulSoup: Parsed document object.
    """
    for parser in ("lxml-xml", "lxml", "html.parser"):
        try:
            soup = BeautifulSoup(content, parser)
            log.debug("XML parsed with parser: %s", parser)
            return soup
        except Exception:
            continue

    # Should never reach here — html.parser is always available
    raise RuntimeError("No suitable XML/HTML parser found.")


# ══════════════════════════════════════════════════════════════════
# 1.  NEWS SCRAPING
# ══════════════════════════════════════════════════════════════════

def fetch_news_articles(topic: str) -> list[dict]:
    """
    Scrape Google News RSS feed for a given topic.

    Uses the Google News RSS endpoint (no API key required).
    Falls back to Google News HTML → Bing News if RSS fails.

    Args:
        topic (str): Search query, e.g. "elections" or "protests".

    Returns:
        list[dict]: Each dict has keys → title, link, source.
    """
    articles = []
    encoded  = quote_plus(topic)

    # ── Primary: Google News RSS ──────────────────────────────────
    rss_url = (
        f"https://news.google.com/rss/search"
        f"?q={encoded}&hl=en-US&gl=US&ceid=US:en"
    )
    log.info("Fetching Google News RSS for topic: '%s'", topic)

    try:
        resp = requests.get(rss_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        # ✅ FIX 2: use safe parser chain instead of hardcoded "xml"
        soup  = _parse_xml(resp.content)
        items = soup.find_all("item")

        for item in items[:MAX_ARTICLES]:
            title  = item.find("title")
            link   = item.find("link")
            source = item.find("source")

            title_text  = title.get_text(strip=True)  if title  else "N/A"
            link_text   = link.get_text(strip=True)   if link   else ""
            source_text = source.get_text(strip=True) if source else "Google News"

            if not link_text:
                continue

            articles.append({
                "title":  title_text,
                "link":   link_text,
                "source": source_text,
            })

        log.info("  → Found %d articles via RSS.", len(articles))

    except Exception as exc:
        log.warning("RSS fetch failed (%s). Trying HTML fallback…", exc)

    # ── Fallback 1: Google News HTML ─────────────────────────────
    if not articles:
        articles = _fetch_news_html_fallback(topic, encoded)

    # ── Fallback 2: Bing News HTML ────────────────────────────────
    if not articles:
        articles = _fetch_bing_news(topic, encoded)

    return articles


def _fetch_news_html_fallback(topic: str, encoded: str) -> list[dict]:
    """Secondary scraper: parse Google News /search HTML page."""
    articles = []
    url = f"https://news.google.com/search?q={encoded}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup    = BeautifulSoup(resp.text, "html.parser")
        anchors = soup.select("a[href*='/articles/']")

        for a in anchors[:MAX_ARTICLES]:
            href  = a.get("href", "")
            title = a.get_text(strip=True)
            if not href or not title:
                continue
            full_url = urljoin("https://news.google.com/", href.lstrip("."))
            articles.append({"title": title, "link": full_url, "source": "Google News"})

        log.info("  → HTML fallback found %d articles.", len(articles))

    except Exception as exc:
        log.warning("HTML fallback failed: %s", exc)

    return articles


def _fetch_bing_news(topic: str, encoded: str) -> list[dict]:
    """Tertiary fallback: scrape Bing News search results."""
    articles = []
    url = f"https://www.bing.com/news/search?q={encoded}&FORM=HDRSC6"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup  = BeautifulSoup(resp.text, "html.parser")
        cards = soup.select("div.news-card, a.title")

        for card in cards[:MAX_ARTICLES]:
            a     = card if card.name == "a" else card.find("a", class_="title")
            if not a:
                continue
            title = a.get_text(strip=True)
            href  = a.get("href", "")
            if href and title:
                articles.append({"title": title, "link": href, "source": "Bing News"})

        log.info("  → Bing News found %d articles.", len(articles))

    except Exception as exc:
        log.warning("Bing News fallback failed: %s", exc)

    return articles


# ══════════════════════════════════════════════════════════════════
# 2.  ARTICLE PARSING
# ══════════════════════════════════════════════════════════════════

def parse_article(url: str) -> str:
    """
    Visit a news article URL and extract the main body text.

    Strategy:
      1. Try <article> tag first (semantic HTML5).
      2. Fall back to the tag with the most <p> children.
      3. Last resort: all <p> tags site-wide.

    Args:
        url (str): Full URL of the article.

    Returns:
        str: Concatenated paragraph text, or empty string on failure.
    """
    time.sleep(random.uniform(1.0, 3.0))

    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        log.debug("Timeout: %s", url)
        return ""
    except requests.exceptions.RequestException as exc:
        log.debug("Request error (%s): %s", type(exc).__name__, url)
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "iframe"]):
        tag.decompose()

    # Strategy 1: <article> tag
    article_tag = soup.find("article")
    if article_tag:
        text = " ".join(" ".join(p.stripped_strings)
                        for p in article_tag.find_all("p"))
        if len(text) >= MIN_TEXT_LENGTH:
            return text

    # Strategy 2: container with most <p> children
    best_container, best_count = None, 0
    for div in soup.find_all(["div", "section", "main"]):
        count = len(div.find_all("p", recursive=False))
        if count > best_count:
            best_count, best_container = count, div

    if best_container:
        text = " ".join(" ".join(p.stripped_strings)
                        for p in best_container.find_all("p"))
        if len(text) >= MIN_TEXT_LENGTH:
            return text

    # Strategy 3: all <p> tags
    return " ".join(" ".join(p.stripped_strings) for p in soup.find_all("p"))

# ══════════════════════════════════════════════════════════════════
# 3.  TEXT CLEANING
# ══════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Normalise raw scraped text for NLP consumption.

    Args:
        text (str): Raw text string.

    Returns:
        str: Cleaned text string.
    """
    if not text:
        return ""

    text = re.sub(r"<[^>]+>", " ", text)

    for entity, char in {
        "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
        "&#39;": "'", "&nbsp;": " ", "&ndash;": "–", "&mdash;": "—",
    }.items():
        text = text.replace(entity, char)

    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^\x20-\x7E\u00A0-\uFFFF]", "", text)
    text = re.sub(r"([^\w\s])\1{2,}", r"\1", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


# ══════════════════════════════════════════════════════════════════
# 4.  REDDIT SCRAPING
# ══════════════════════════════════════════════════════════════════

def scrape_reddit_posts(topic: str) -> list[dict]:
    """
    Scrape Reddit search results using the public JSON endpoint.
    No API key required.

    Args:
        topic (str): Topic to search for on Reddit.

    Returns:
        list[dict]: Each dict → title, text, subreddit, link.
    """
    posts   = []
    encoded = quote_plus(topic)
    url = (
        f"https://www.reddit.com/search.json"
        f"?q={encoded}&sort=relevance&limit={MAX_REDDIT_POSTS}&type=link"
    )

    log.info("Fetching Reddit posts for topic: '%s'", topic)
    time.sleep(random.uniform(1.0, 2.5))

    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        children = resp.json().get("data", {}).get("children", [])
        log.info("  → Found %d Reddit posts.", len(children))

        for child in children:
            post      = child.get("data", {})
            title     = post.get("title", "").strip()
            selftext  = post.get("selftext", "").strip()
            subreddit = post.get("subreddit_name_prefixed", "r/unknown")
            permalink = post.get("permalink", "")
            full_link = f"https://www.reddit.com{permalink}" if permalink else ""
            content   = selftext if selftext else title

            if not content or len(content) < 10:
                continue

            posts.append({
                "title":     title,
                "text":      content,
                "subreddit": subreddit,
                "link":      full_link,
            })

    except Exception as exc:
        log.warning("Reddit scrape failed: %s", exc)

    return posts


def scrape_reddit_posts_praw(topic: str) -> list[dict]:
    """
    OPTIONAL: Scrape Reddit via the PRAW library (requires API credentials).
    Get free credentials at https://www.reddit.com/prefs/apps
    """
    try:
        import praw
    except ImportError:
        log.warning("praw not installed. Run: pip install praw")
        return []

    reddit = praw.Reddit(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
        user_agent="EchoChamberX/1.0 by u/YOUR_USERNAME",
    )
    posts = []
    try:
        for sub in reddit.subreddit("all").search(
            topic, sort="relevance", limit=MAX_REDDIT_POSTS
        ):
            content = sub.selftext.strip() if sub.selftext else sub.title
            if len(content) < 10:
                continue
            posts.append({
                "title":     sub.title,
                "text":      content,
                "subreddit": f"r/{sub.subreddit.display_name}",
                "link":      f"https://www.reddit.com{sub.permalink}",
            })
    except Exception as exc:
        log.warning("PRAW scrape failed: %s", exc)
    return posts


# ══════════════════════════════════════════════════════════════════
# 5.  BUILD DATASET
# ══════════════════════════════════════════════════════════════════

def build_dataset(topic: str, include_reddit: bool = True) -> list[dict]:
    """
    Orchestrator: gather news + Reddit data, clean, deduplicate,
    and return a unified NLP-ready dataset.

    Schema per entry:
        text      – cleaned body text
        source    – "news" | "reddit"
        title     – headline / post title
        url       – original URL
        timestamp – collection time (ISO format, safe for CSV values)

    Args:
        topic          (str):  Search query.
        include_reddit (bool): Include Reddit posts (default True).

    Returns:
        list[dict]: Final dataset capped at DATASET_LIMIT entries.
    """
    dataset  = []
    # ✅ FIX 1 (applied here too): readable ISO string for CELL values is fine
    now_iso  = _readable_timestamp()   # "2026-03-23T17:49:11Z" – safe inside CSV cells

    log.info("=" * 60)
    log.info("EchoChamberX – Building dataset for topic: '%s'", topic)
    log.info("=" * 60)

    # ── News articles ─────────────────────────────────────────────
    raw_articles = fetch_news_articles(topic)
    log.info("Parsing %d article URLs…", len(raw_articles))

    for i, article in enumerate(raw_articles, 1):
        url   = article.get("link", "")
        title = article.get("title", "N/A")
        if not url:
            continue

        log.info("  [%02d/%02d] Parsing: %s", i, len(raw_articles), title[:60])
        clean = clean_text(parse_article(url))

        if len(clean) < MIN_TEXT_LENGTH:
            log.debug("       Skipped (text too short).")
            continue

        dataset.append({
            "text":      clean,
            "source":    "news",
            "title":     clean_text(title),
            "url":       url,
            "timestamp": now_iso,
        })

    log.info("News entries collected: %d", len(dataset))

    # ── Reddit posts ──────────────────────────────────────────────
    if include_reddit:
        for post in scrape_reddit_posts(topic):
            clean = clean_text(post.get("text", ""))
            if len(clean) < MIN_TEXT_LENGTH:
                continue
            dataset.append({
                "text":      clean,
                "source":    "reddit",
                "title":     clean_text(post.get("title", "N/A")),
                "url":       post.get("link", ""),
                "timestamp": now_iso,
            })
        log.info("Reddit entries collected (total so far): %d", len(dataset))

    # ── Deduplicate ───────────────────────────────────────────────
    seen, unique = set(), []
    for entry in dataset:
        key = entry["title"].lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(entry)

    log.info("After deduplication: %d unique entries.", len(unique))

    random.shuffle(unique)
    final = unique[:DATASET_LIMIT]
    log.info("Final dataset size: %d entries (cap=%d).", len(final), DATASET_LIMIT)
    return final


# ══════════════════════════════════════════════════════════════════
# 6.  SAVE DATASET
# ══════════════════════════════════════════════════════════════════

def save_dataset(
    data: list[dict],
    format: str = "csv",
    filename: str | None = None,
) -> str:
    """
    Persist the dataset to disk as CSV or JSON.

    ✅ FIX 1 – Windows-safe filename:
        OLD (broken on Windows):
            ts = datetime.now(UTC).isoformat()
            # → "2026-03-23T17:49:11.093292+00:00"
            # Colons in filename → OSError [Errno 22] on Windows

        NEW (works everywhere):
            ts = _safe_timestamp()
            # → "20260323_174911"
            # No colons, no special characters — valid on all OS

    Args:
        data     (list[dict]): Dataset from build_dataset().
        format   (str):        "csv" or "json".
        filename (str|None):   Custom filename; auto-generated if None.

    Returns:
        str: Path to the saved file.
    """
    if not data:
        log.warning("save_dataset called with empty data – nothing saved.")
        return ""

    if filename is None:
        # ✅ FIX 1: use _safe_timestamp() — no colons, Windows-compatible
        ts       = _safe_timestamp()                  # e.g. "20260323_174911"
        filename = f"echochamberx_dataset_{ts}.{format.lower()}"

    df = pd.DataFrame(data)

    if format.lower() == "csv":
        df.to_csv(filename, index=False, encoding="utf-8")
        log.info("Dataset saved → %s  (%d rows)", filename, len(df))

    elif format.lower() == "json":
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        log.info("Dataset saved → %s  (%d entries)", filename, len(data))

    else:
        raise ValueError(f"Unsupported format '{format}'. Use 'csv' or 'json'.")

    return filename


# ══════════════════════════════════════════════════════════════════
# 7.  UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════

def dataset_summary(data: list[dict]) -> None:
    """Print a quick statistical summary of the collected dataset."""
    if not data:
        print("Dataset is empty.")
        return

    df        = pd.DataFrame(data)
    text_lens = df["text"].str.len()

    print("\n" + "═" * 55)
    print("  EchoChamberX – Dataset Summary")
    print("═" * 55)
    print(f"  Total entries      : {len(df)}")
    print(f"  Source breakdown   :")
    for src, cnt in df["source"].value_counts().items():
        print(f"      {src:<12}: {cnt}")
    print(f"  Avg text length    : {text_lens.mean():.0f} chars")
    print(f"  Min text length    : {text_lens.min()} chars")
    print(f"  Max text length    : {text_lens.max()} chars")
    print(f"  Entries > 500 chars: {(text_lens > 500).sum()}")
    print("═" * 55 + "\n")


def load_dataset(filepath: str) -> list[dict]:
    """Load a previously saved EchoChamberX dataset back into memory."""
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath, encoding="utf-8").to_dict(orient="records")
    elif filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError("Unsupported file type. Expected .csv or .json")


# ══════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    TOPIC = "protests"   # ← change to any topic

    dataset = build_dataset(TOPIC, include_reddit=True)

    print("\n── Sample entries ──────────────────────────────────────")
    for entry in dataset[:2]:
        print(f"\n[{entry['source'].upper()}] {entry['title']}")
        print(f"  URL  : {entry['url'][:80]}…")
        print(f"  Text : {entry['text'][:200]}…")

    dataset_summary(dataset)

    csv_path  = save_dataset(dataset, format="csv")
    json_path = save_dataset(dataset, format="json")

    print(f"Files saved:\n  CSV  → {csv_path}\n  JSON → {json_path}")

    reloaded = load_dataset(csv_path)
    print(f"\nReloaded {len(reloaded)} entries from CSV – ready for NLP pipeline.")