from __future__ import annotations

from typing import List, Optional

from .models import Article
from .recommender import NewsRecommender
from .storage import Database


class Pipeline:
    def __init__(self, db: Optional[Database] = None) -> None:
        self.db = db or Database()
        self.rec = NewsRecommender()
        self.refresh()

    def refresh(self) -> None:
        articles: List[Article] = self.db.list_articles()
        self.rec.fit(articles)

    def ingest_articles(self, articles: List[Article]) -> int:
        n = self.db.upsert_articles(articles)
        self.refresh()
        return n

    def similar_by_article(self, article_id: str, top_k: int = 10):
        return self.rec.recommend_similar(article_id, top_k=top_k)

    def recommend_for_user(self, user_id: str, top_k: int = 10):
        interactions = self.db.list_user_interactions(user_id)
        seed_ids = [it.article_id for it in interactions][-50:]
        if not seed_ids:
            trending = self.db.list_trending_articles(limit=top_k)
            return [(aid, 0.0) for aid in trending]
        return self.rec.recommend_for_profile(seed_ids, top_k=top_k)
