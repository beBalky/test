from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from .models import Article, Interaction


DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "newsrec" / "db.sqlite3"
DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


class Database:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path: Path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._ensure_schema()

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    url TEXT,
                    content TEXT NOT NULL,
                    published_at TEXT,
                    language TEXT,
                    categories TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    user_id TEXT NOT NULL,
                    article_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def upsert_articles(self, articles: Iterable[Article]) -> int:
        count = 0
        with self.connection() as conn:
            cur = conn.cursor()
            for article in articles:
                categories_json = json.dumps(article.categories or [], ensure_ascii=False)
                published_at_str = (
                    article.published_at.isoformat() if article.published_at else None
                )
                cur.execute(
                    """
                    INSERT INTO articles (id, title, url, content, published_at, language, categories)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        title=excluded.title,
                        url=excluded.url,
                        content=excluded.content,
                        published_at=excluded.published_at,
                        language=excluded.language,
                        categories=excluded.categories
                    """,
                    (
                        article.id,
                        article.title,
                        article.url,
                        article.content,
                        published_at_str,
                        article.language,
                        categories_json,
                    ),
                )
                count += 1
            conn.commit()
        return count

    def list_articles(self) -> List[Article]:
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM articles")
            rows = cur.fetchall()
        articles: List[Article] = []
        for row in rows:
            published_at = (
                datetime.fromisoformat(row["published_at"]) if row["published_at"] else None
            )
            categories = json.loads(row["categories"]) if row["categories"] else []
            articles.append(
                Article(
                    id=row["id"],
                    title=row["title"],
                    url=row["url"],
                    content=row["content"],
                    published_at=published_at,
                    language=row["language"] or "zh",
                    categories=categories,
                )
            )
        return articles

    def get_article(self, article_id: str) -> Optional[Article]:
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM articles WHERE id=?", (article_id,))
            row = cur.fetchone()
        if not row:
            return None
        published_at = (
            datetime.fromisoformat(row["published_at"]) if row["published_at"] else None
        )
        categories = json.loads(row["categories"]) if row["categories"] else []
        return Article(
            id=row["id"],
            title=row["title"],
            url=row["url"],
            content=row["content"],
            published_at=published_at,
            language=row["language"] or "zh",
            categories=categories,
        )

    def add_interaction(self, interaction: Interaction) -> None:
        timestamp_str = (
            interaction.timestamp.isoformat() if interaction.timestamp else datetime.utcnow().isoformat()
        )
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO interactions (user_id, article_id, event_type, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (interaction.user_id, interaction.article_id, interaction.event_type, timestamp_str),
            )
            conn.commit()

    def list_user_interactions(self, user_id: str) -> List[Interaction]:
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT user_id, article_id, event_type, timestamp FROM interactions WHERE user_id=?",
                (user_id,),
            )
            rows = cur.fetchall()
        interactions: List[Interaction] = []
        for row in rows:
            interactions.append(
                Interaction(
                    user_id=row["user_id"],
                    article_id=row["article_id"],
                    event_type=row["event_type"],
                    timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None,
                )
            )
        return interactions

    def list_trending_articles(self, limit: int = 10) -> List[str]:
        with self.connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id FROM articles
                ORDER BY COALESCE(published_at, '''') DESC, rowid DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [row["id"] for row in rows]
