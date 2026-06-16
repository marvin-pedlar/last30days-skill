"""LinkedIn keyword search via ScrapeCreators API for /last30days.

Uses the ScrapeCreators REST API to search public LinkedIn posts by keyword,
extracting engagement metrics (likes, comments) from professional/B2B posts.

LinkedIn is the strongest signal for B2B, procurement, SaaS, and enterprise
topics, where the real conversation happens on company pages and practitioner
profiles rather than Reddit or X.

Requires SCRAPECREATORS_API_KEY in config. Opt-in source via INCLUDE_SOURCES
(set INCLUDE_SOURCES=linkedin) to keep per-call credit usage explicit.
API docs: https://docs.scrapecreators.com/v1/linkedin/search/posts
"""

import math
import re
from typing import Any, Dict, List, Optional

from . import dates, http, log
from .relevance import token_overlap_relevance as _compute_relevance

SCRAPECREATORS_BASE = "https://api.scrapecreators.com/v1/linkedin"

# Depth configurations: how many results to fetch
DEPTH_CONFIG = {
    "quick":   {"results": 10},
    "default": {"results": 20},
    "deep":    {"results": 40},
}

# Each search/posts page returns ~10 posts; cap pagination so deep mode stays
# within a couple of credits (1 credit per request).
_PER_PAGE = 10
_MAX_PAGES = 4

# Extract the numeric activity id from a LinkedIn post URL so items dedupe and
# carry a stable id even though the API response has no id/urn field.
_ACTIVITY_RE = re.compile(r"activity[-:](\d+)")


def _log(msg: str):
    log.source_log("LinkedIn", msg)


def _extract_core_subject(topic: str) -> str:
    """Extract core subject from a verbose query for LinkedIn search."""
    from .query import extract_core_subject
    _LINKEDIN_NOISE = frozenset({
        'best', 'top', 'good', 'great', 'awesome',
        'latest', 'new', 'news', 'update', 'updates',
        'trending', 'hottest', 'popular', 'viral',
        'practices', 'features', 'recommendations', 'advice',
    })
    return extract_core_subject(topic, noise=_LINKEDIN_NOISE)


def _parse_date(item: Dict[str, Any]) -> Optional[str]:
    """Parse the post timestamp to YYYY-MM-DD.

    LinkedIn search/posts returns ISO 8601 in `datePublished`; fall back to
    other common timestamp keys for resilience to response-shape drift.
    """
    for key in ("datePublished", "published_at", "created_at", "date"):
        val = item.get(key)
        if val is None:
            continue
        dt = dates.parse_date(str(val))
        if dt:
            return dt.strftime("%Y-%m-%d")
    return None


def _post_id(url: str, index: int) -> str:
    match = _ACTIVITY_RE.search(url or "")
    if match:
        return f"LI{match.group(1)}"
    return f"LI{index + 1}"


def _parse_items(raw_items: List[Dict[str, Any]], core_topic: str) -> List[Dict[str, Any]]:
    """Parse raw LinkedIn posts into normalized dicts."""
    items = []
    for i, raw in enumerate(raw_items):
        text = raw.get("description") or raw.get("text") or raw.get("content") or ""
        if isinstance(text, dict):
            text = text.get("text", "")
        text = str(text).strip()

        # Author is a company or person NAME (LinkedIn has no @handle).
        author = raw.get("author") or {}
        if isinstance(author, dict):
            display_name = author.get("name") or ""
            author_url = author.get("url") or ""
        elif isinstance(author, str):
            display_name = author
            author_url = ""
        else:
            display_name = ""
            author_url = ""

        # Engagement metrics
        likes = raw.get("likeCount") or raw.get("like_count") or raw.get("likes") or 0
        comments = raw.get("commentCount") or raw.get("comment_count") or raw.get("comments_count") or 0
        reposts = raw.get("repostCount") or raw.get("repost_count") or 0

        url = raw.get("url") or author_url or ""
        post_id = _post_id(url, i)
        date_str = _parse_date(raw)

        # Relevance: text overlap + position rank + engagement boost (mirrors
        # the microblog sources). LinkedIn engagement scales smaller than X, so
        # the log boost keeps a high-like post from dominating on volume alone.
        rank_score = max(0.3, 1.0 - (i * 0.02))
        engagement_boost = min(0.2, math.log1p(likes + comments) / 40)
        text_relevance = _compute_relevance(core_topic, text)
        relevance = min(1.0, text_relevance * 0.5 + rank_score * 0.3 + engagement_boost + 0.1)

        items.append({
            "id": post_id,
            "handle": display_name,
            "display_name": display_name,
            "text": text,
            "url": url,
            "date": date_str,
            "engagement": {
                "likes": likes,
                "comments": comments,
                "reposts": reposts,
            },
            "relevance": round(relevance, 2),
            "why_relevant": f"LinkedIn: {display_name}: {text[:60]}" if text else f"LinkedIn: {display_name}",
        })
    return items


def search_linkedin(
    topic: str,
    from_date: str,
    to_date: str,
    depth: str = "default",
    token: str = None,
) -> Dict[str, Any]:
    """Search public LinkedIn posts via the ScrapeCreators API.

    Args:
        topic: Search topic
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        depth: 'quick', 'default', or 'deep'
        token: ScrapeCreators API key

    Returns:
        Dict with 'items' list and optional 'error'.
    """
    if not token:
        return {"items": [], "error": "No SCRAPECREATORS_API_KEY configured"}

    config = DEPTH_CONFIG.get(depth, DEPTH_CONFIG["default"])
    target = config["results"]
    core_topic = _extract_core_subject(topic)

    _log(f"Searching for '{core_topic}' (depth={depth}, limit={target})")

    raw_items: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    pages = min(_MAX_PAGES, math.ceil(target / _PER_PAGE))
    try:
        for _ in range(pages):
            params: Dict[str, Any] = {"query": core_topic, "date_posted": "last-month"}
            if cursor:
                params["cursor"] = cursor
            data = http.get(
                f"{SCRAPECREATORS_BASE}/search/posts",
                params=params,
                headers=http.scrapecreators_headers(token),
                timeout=30,
                retries=2,
            )
            page_items = (
                data.get("posts")
                or data.get("items")
                or data.get("data")
                or data.get("results")
                or []
            )
            raw_items.extend(page_items)
            cursor = data.get("cursor")
            if not cursor or not page_items or len(raw_items) >= target:
                break
    except Exception as e:
        _log(f"ScrapeCreators error: {e}")
        # Return whatever we paged in before the error rather than nothing.
        if not raw_items:
            return {"items": [], "error": f"{type(e).__name__}: {e}"}

    raw_items = raw_items[:target]
    items = _parse_items(raw_items, core_topic)

    # Date filter (the API's last-month filter is approximate; enforce the
    # explicit window the same way the other sources do).
    in_range = [i for i in items if i["date"] and from_date <= i["date"] <= to_date]
    out_of_range = len(items) - len(in_range)
    if in_range:
        items = in_range
        if out_of_range:
            _log(f"Filtered {out_of_range} posts outside date range")
    else:
        _log(f"No posts within date range, keeping all {len(items)}")

    # Sort by engagement (likes) descending
    items.sort(key=lambda x: x["engagement"]["likes"], reverse=True)

    _log(f"Found {len(items)} LinkedIn posts")
    return {"items": items}


def parse_linkedin_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse a LinkedIn search response to normalized format.

    Returns:
        List of item dicts ready for normalization.
    """
    return response.get("items", [])
