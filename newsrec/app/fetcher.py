from __future__ import annotations

from datetime import datetime
from typing import List

import feedparser

from .models import Article


def fetch_rss(feed_url: str, limit: int = 50) -> List[Article]:
    parsed = feedparser.parse(feed_url)
    items: List[Article] = []
    for entry in parsed.entries[:limit]:
        aid = entry.get("id") or entry.get("link") or entry.get("title")
        if not aid:
            continue
        content = entry.get("summary") or entry.get("description") or ""
        published = None
        if entry.get("published_parsed"):
            try:
                published = datetime(*entry.published_parsed[:6])
            except Exception:
                published = None
        items.append(
            Article(
                id=str(aid),
                title=str(entry.get("title") or ""),
                url=str(entry.get("link") or ""),
                content=str(content),
                published_at=published,
                language="zh",
                categories=[],
            )
        )
    return items
