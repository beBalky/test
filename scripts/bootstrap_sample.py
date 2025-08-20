from __future__ import annotations

import json
from pathlib import Path

from newsrec.app.models import Article
from newsrec.app.pipeline import Pipeline


def main() -> None:
    data_path = Path(__file__).resolve().parents[1] / "newsrec" / "data" / "sample_news.jsonl"
    if not data_path.exists():
        raise SystemExit(f"sample data not found: {data_path}")
    pipeline = Pipeline()
    articles = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            articles.append(Article(**obj))
    n = pipeline.ingest_articles(articles)
    print(f"ingested: {n}")

    if len(articles) >= 1:
        aid = articles[0].id
        recs = pipeline.similar_by_article(aid, top_k=5)
        print("similar to", aid, recs)


if __name__ == "__main__":
    main()
