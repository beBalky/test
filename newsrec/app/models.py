from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Article(BaseModel):
    id: str = Field(..., description="Unique article id")
    title: str
    url: Optional[str] = None
    content: str
    published_at: Optional[datetime] = None
    language: str = Field(default="zh", description="ISO language code, e.g., \"zh\" or \"en\"")
    categories: Optional[List[str]] = None


class Interaction(BaseModel):
    user_id: str
    article_id: str
    event_type: str = Field("click", description="click/view/like/bookmark/etc")
    timestamp: Optional[datetime] = None


class RecommendItem(BaseModel):
    article_id: str
    score: float


class RecommendResponse(BaseModel):
    items: List[RecommendItem]


class RSSIngestRequest(BaseModel):
    feed_url: str
    limit: int = 50
