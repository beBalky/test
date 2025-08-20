## 新闻推荐系统

一个可运行的端到端示例：
- SQLite 存储文章与交互
- 基于 TF‑IDF + 余弦相似度的内容召回（支持中文分词 jieba）
- FastAPI 提供在线推荐接口
- 支持 RSS 抓取
- 自带示例数据与引导脚本

### 1. 准备环境
```bash
python -m pip install -r requirements.txt
```

### 2. 载入示例数据并验证
```bash
python scripts/bootstrap_sample.py
```
你会看到示例推荐结果输出。

### 3. 启动服务
```bash
uvicorn newsrec.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. API 快览
- 健康检查: `GET /healthz`
- 写入文章: `POST /articles`（传入 Article 或列表）
- 记录交互: `POST /interactions`
- 获取推荐: `GET /recommend?user_id=...` 或 `GET /recommend?article_id=...`
- RSS 抓取: `POST /ingest/rss`（传入 `feed_url`）

### 5. 目录结构
```
newsrec/
  app/
    main.py
    models.py
    recommender.py
    storage.py
    fetcher.py
    pipeline.py
  data/
    sample_news.jsonl
scripts/
  bootstrap_sample.py
requirements.txt
```

### 6. 备注
- 初次启动会基于数据库内容构建 TF‑IDF 索引；文章有变更时可调用 `/articles` 或通过脚本更新索引。
- 生产环境可接入更强的向量模型或召回/重排链路。
