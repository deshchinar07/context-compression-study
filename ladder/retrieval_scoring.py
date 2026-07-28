from __future__ import annotations

import functools
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
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.paragraphs[i] for _, i in scored[:topk]]


class LexicalScorer:
    def __init__(
        self,
        passage_texts: list[str] | None = None,
        query: str = "",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1, self.b = k1, b
        self.q_terms = tokenize(query)
        self._idf, self._avgdl = _bm25_stats([tokenize(t) for t in (passage_texts or [])])

    def score(self, text: str) -> float:
        tf = Counter(tokenize(text))
        return _bm25_score(self.q_terms, tf, sum(tf.values()), self._idf, self.k1, self.b, self._avgdl)


E5_MODEL = "intfloat/e5-base-v2"


@functools.lru_cache(maxsize=1)
def _load_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "dense retrieval needs sentence-transformers, which is not installed. "
            "Run `pip install sentence-transformers` or use --retrieval bm25 "
            "(the default, dependency-free path)."
        ) from e
    return SentenceTransformer(E5_MODEL)


def _embed(texts: list[str], prefix: str):
    import numpy as np

    model = _load_model()
    prefixed = [f"{prefix}{t}" for t in texts]
    emb = model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(emb, dtype=np.float32)


class E5Retriever:

    def __init__(self, paragraphs: list[Paragraph]):
        self.paragraphs = paragraphs
        docs = [f"{p.title} {p.text}" for p in paragraphs]

        self._doc_emb = _embed(docs, "passage: ") if docs else None

    def search(
        self,
        query: str,
        topk: int = 3,
        exclude_idx: set | None = None,
    ) -> list[Paragraph]:
        exclude_idx = exclude_idx or set()
        if len(self.paragraphs) == 0:
            return []
        q = _embed([query], "query: ")[0]
        sims = self._doc_emb @ q

        order = sorted(
            (i for i in range(len(self.paragraphs)) if i not in exclude_idx),
            key=lambda i: (-float(sims[i]), i),
        )
        return [self.paragraphs[i] for i in order[:topk]]


class E5Scorer:

    def __init__(self, query: str):
        self._q = _embed([query], "query: ")[0]

    def score(self, text: str) -> float:
        if not text.strip():
            return 0.0
        v = _embed([text], "passage: ")[0]
        return 1.0 + float(v @ self._q)


def make_retriever(kind: str, paragraphs: list[Paragraph]):
    kind = (kind or "bm25").lower()
    if kind == "bm25":
        return BM25Retriever(paragraphs)
    if kind == "e5":
        return E5Retriever(paragraphs)
    raise ValueError(f"unknown retriever kind {kind!r}; choose 'bm25' or 'e5'")


def make_scorer(kind: str, passage_texts: list[str], query: str):
    kind = (kind or "bm25").lower()
    if kind == "bm25":
        return LexicalScorer(passage_texts, query)
    if kind == "e5":
        return E5Scorer(query)
    raise ValueError(f"unknown scorer kind {kind!r}; choose 'bm25' or 'e5'")
