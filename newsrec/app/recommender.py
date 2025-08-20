from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from math import log, sqrt

from .models import Article

try:
    import jieba  # type: ignore
except Exception:
    jieba = None


@dataclass
class CorpusIndex:
    article_ids: List[str]
    id_to_row: Dict[str, int]


class NewsRecommender:
    def __init__(self) -> None:
        self.corpus_index: Optional[CorpusIndex] = None
        self.cached_articles: Dict[str, Article] = {}
        self.vocabulary: Dict[str, int] = {}
        self.doc_term_freqs: List[Dict[int, int]] = []
        self.idf: Dict[int, float] = {}
        self.doc_norms: List[float] = []

    def _tokenize(self, text: str, language_hint: Optional[str]) -> List[str]:
        if language_hint and language_hint.lower().startswith("zh") and jieba is not None:
            return [t.strip() for t in jieba.lcut(text) if t.strip()]
        return [t for t in text.split() if t]

    def fit(self, articles: List[Article]) -> None:
        if not articles:
            self.corpus_index = None
            self.cached_articles = {}
            self.vocabulary = {}
            self.doc_term_freqs = []
            self.idf = {}
            self.doc_norms = []
            return

        languages = [a.language or "zh" for a in articles]
        lang_hint = max(set(languages), key=languages.count)

        self.cached_articles = {a.id: a for a in articles}
        self.vocabulary = {}
        self.doc_term_freqs = []

        article_ids: List[str] = []
        for a in articles:
            text = f"{a.title}\n{a.content}"
            tokens = self._tokenize(text, lang_hint)
            term_freq: Dict[int, int] = {}
            for tok in tokens:
                if tok not in self.vocabulary:
                    self.vocabulary[tok] = len(self.vocabulary)
                tid = self.vocabulary[tok]
                term_freq[tid] = term_freq.get(tid, 0) + 1
            self.doc_term_freqs.append(term_freq)
            article_ids.append(a.id)

        self.corpus_index = CorpusIndex(article_ids=article_ids, id_to_row={aid: i for i, aid in enumerate(article_ids)})

        doc_count = len(self.doc_term_freqs)
        df: Dict[int, int] = {}
        for tf in self.doc_term_freqs:
            for tid in tf.keys():
                df[tid] = df.get(tid, 0) + 1
        self.idf = {tid: log((1 + doc_count) / (1 + dfc)) + 1.0 for tid, dfc in df.items()}

        self.doc_norms = []
        for tf in self.doc_term_freqs:
            norm_sq = 0.0
            for tid, freq in tf.items():
                weight = (1 + log(freq)) * self.idf.get(tid, 0.0)
                norm_sq += weight * weight
            self.doc_norms.append(sqrt(norm_sq) if norm_sq > 0 else 1.0)

    def _vectorize(self, term_freq: Dict[int, int]) -> Dict[int, float]:
        vec: Dict[int, float] = {}
        for tid, freq in term_freq.items():
            vec[tid] = (1 + log(freq)) * self.idf.get(tid, 0.0)
        return vec

    def _cosine(self, a: Dict[int, float], b: Dict[int, float], b_norm: float) -> float:
        dot = 0.0
        for tid, aval in a.items():
            dot += aval * b.get(tid, 0.0)
        a_norm = sqrt(sum(v*v for v in a.values())) or 1.0
        return dot / (a_norm * (b_norm or 1.0))

    def recommend_similar(self, article_id: str, top_k: int = 10, exclude_self: bool = True) -> List[Tuple[str, float]]:
        if not self.corpus_index:
            return []
        row = self.corpus_index.id_to_row.get(article_id)
        if row is None:
            return []
        query_vec = self._vectorize(self.doc_term_freqs[row])
        results: List[Tuple[str, float]] = []
        for idx, tf in enumerate(self.doc_term_freqs):
            aid = self.corpus_index.article_ids[idx]
            if exclude_self and aid == article_id:
                continue
            score = self._cosine(query_vec, self._vectorize(tf), self.doc_norms[idx])
            results.append((aid, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def recommend_for_profile(self, seed_article_ids: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.corpus_index:
            return []
        agg: Dict[int, int] = {}
        count = 0
        for aid in seed_article_ids:
            row = self.corpus_index.id_to_row.get(aid)
            if row is None:
                continue
            count += 1
            for tid, freq in self.doc_term_freqs[row].items():
                agg[tid] = agg.get(tid, 0) + freq
        if count == 0:
            return []
        mean_tf = {tid: int(round(freq / count)) or 1 for tid, freq in agg.items()}
        query_vec = self._vectorize(mean_tf)
        results: List[Tuple[str, float]] = []
        seed_set = set(seed_article_ids)
        for idx, tf in enumerate(self.doc_term_freqs):
            aid = self.corpus_index.article_ids[idx]
            if aid in seed_set:
                continue
            score = self._cosine(query_vec, self._vectorize(tf), self.doc_norms[idx])
            results.append((aid, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
