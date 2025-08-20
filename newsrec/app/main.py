from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException

from .models import Article, Interaction, RecommendItem, RecommendResponse, RSSIngestRequest
from .pipeline import Pipeline
from .storage import Database
from .fetcher import fetch_rss


db = Database()
pipeline = Pipeline(db)
app = FastAPI(title="News Recommendation Service")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/articles")
def post_articles(articles: List[Article]) -> dict:
    inserted = pipeline.ingest_articles(articles)
    return {"ok": True, "count": inserted}


@app.post("/interactions")
def post_interaction(interaction: Interaction) -> dict:
    db.add_interaction(interaction)
    return {"ok": True}


@app.get("/recommend", response_model=RecommendResponse)
def get_recommend(user_id: Optional[str] = None, article_id: Optional[str] = None, top_k: int = 10) -> RecommendResponse:
    if user_id:
        pairs = pipeline.recommend_for_user(user_id=user_id, top_k=top_k)
    elif article_id:
        pairs = pipeline.similar_by_article(article_id=article_id, top_k=top_k)
    else:
        raise HTTPException(status_code=400, detail="user_id or article_id required")
    items = [RecommendItem(article_id=aid, score=score) for aid, score in pairs]
    return RecommendResponse(items=items)


@app.post("/ingest/rss")
def ingest_rss(req: RSSIngestRequest) -> dict:
    articles = fetch_rss(req.feed_url, limit=req.limit)
    count = pipeline.ingest_articles(articles)
    return {"ok": True, "count": count}
