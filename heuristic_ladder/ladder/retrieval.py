from __future__ import annotations

import math
import re
from collections import Counter

from .data import Paragraph

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _bm25_stats(docs: list[list[str]]) -> tuple[dict, float]:
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n if n else 0.0
    df: Counter = Counter()
    for d in docs:
        for term in set(d):
            df[term] += 1
    idf = {t: max(1e-6, math.log((n - f + 0.5) / (f + 0.5) + 1.0)) for t, f in df.items()}
    return idf, avgdl


def _bm25_score(q_terms, tf: Counter, dl: float, idf: dict, k1: float, b: float, avgdl: float) -> float:
    s = 0.0
    for term in q_terms:
        if term not in tf:
            continue
        f = tf[term]
        denom = f + k1 * (1 - b + b * dl / (avgdl or 1.0))
        s += idf.get(term, 0.0) * (f * (k1 + 1)) / (denom or 1.0)
    return s


class BM25Retriever:
    def __init__(self, paragraphs: list[Paragraph], k1: float = 1.5, b: float = 0.75):
        self.paragraphs = paragraphs
        self.k1, self.b = k1, b
        docs = [tokenize(f"{p.title} {p.text}") for p in paragraphs]
        self._doc_len = [len(d) for d in docs]
        self._tf = [Counter(d) for d in docs]
        self._idf, self._avgdl = _bm25_stats(docs)

    def search(self, query: str, topk: int = 3, exclude_idx: set | None = None) -> list[Paragraph]:
        exclude_idx = exclude_idx or set()
        q_terms = tokenize(query)
        scored = [
            (_bm25_score(q_terms, self._tf[i], self._doc_len[i], self._idf, self.k1, self.b, self._avgdl), i)
            for i in range(len(self.paragraphs))
            if i not in exclude_idx
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))  # score desc, then index asc (deterministic)
        return [self.paragraphs[i] for _, i in scored[:topk]]


class LexicalScorer:
    def __init__(
        self,
        corpus_texts: list[str] | None = None,
        query: str = "",
        k1: float = 1.5,
        b: float = 0.75,
        *,
        idf: dict | None = None,
        avgdl: float | None = None,
    ):
        self.k1, self.b = k1, b
        self.q_terms = tokenize(query)
        if idf is not None and avgdl is not None:
            self._idf, self._avgdl = idf, avgdl
        else:
            self._idf, self._avgdl = _bm25_stats([tokenize(t) for t in (corpus_texts or [])])

    @staticmethod
    def index_corpus(corpus_texts: list[str]) -> tuple:
        return _bm25_stats([tokenize(t) for t in corpus_texts])

    def score(self, text: str) -> float:
        tf = Counter(tokenize(text))
        return _bm25_score(self.q_terms, tf, sum(tf.values()), self._idf, self.k1, self.b, self._avgdl)


def make_retriever(kind: str, paragraphs: list[Paragraph]):
    kind = (kind or "bm25").lower()
    if kind == "bm25":
        return BM25Retriever(paragraphs)
    if kind == "e5":
        from .dense import E5Retriever
        return E5Retriever(paragraphs)
    raise ValueError(f"unknown retriever kind {kind!r}; choose 'bm25' or 'e5'")


def make_scorer(kind: str, corpus_texts: list[str], query: str):
    kind = (kind or "bm25").lower()
    if kind == "bm25":
        return LexicalScorer(corpus_texts, query)
    if kind == "e5":
        from .dense import E5Scorer
        return E5Scorer(corpus_texts, query)
    raise ValueError(f"unknown scorer kind {kind!r}; choose 'bm25' or 'e5'")


class ScorerFactory:
    """Index the corpus once, then mint cheap per-query scorers that reuse it."""

    def __init__(self, kind: str, corpus_texts: list[str]):
        self.kind = (kind or "bm25").lower()
        if self.kind == "e5":
            self._idf = self._avgdl = None
        elif self.kind == "bm25":
            self._idf, self._avgdl = LexicalScorer.index_corpus(corpus_texts)
        else:
            raise ValueError(f"unknown scorer kind {self.kind!r}; choose 'bm25' or 'e5'")

    def for_query(self, query: str):
        if self.kind == "bm25":
            return LexicalScorer(query=query, idf=self._idf, avgdl=self._avgdl)
        from .dense import E5Scorer
        return E5Scorer([], query)
