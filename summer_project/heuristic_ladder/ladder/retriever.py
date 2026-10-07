"""Retrieval over the per-example paragraph pool.

A dependency-free Okapi BM25 (k1=1.5, b=0.75) over the candidate paragraphs of a
single Example. The agent issues a query string; we return the top-k paragraphs.
This is deterministic and identical for every rung of the ladder -- the retriever
is *outside* the compression policy, so differences between rungs can never come
from retrieval.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Optional

from .data import Paragraph

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _WORD.findall(text.lower())


class BM25Retriever:
    def __init__(self, paragraphs: List[Paragraph], k1: float = 1.5, b: float = 0.75):
        self.paragraphs = paragraphs
        self.k1 = k1
        self.b = b
        self._docs = [tokenize(f"{p.title} {p.text}") for p in paragraphs]
        self._doc_len = [len(d) for d in self._docs]
        self._avgdl = (sum(self._doc_len) / len(self._docs)) if self._docs else 0.0
        self._tf = [Counter(d) for d in self._docs]
        df: Counter = Counter()
        for d in self._docs:
            for term in set(d):
                df[term] += 1
        n = len(self._docs)
        # BM25 idf with the standard +0.5 smoothing, floored at a small positive.
        self._idf = {
            t: max(1e-6, math.log((n - freq + 0.5) / (freq + 0.5) + 1.0))
            for t, freq in df.items()
        }

    def _score(self, doc_idx: int, q_terms: List[str]) -> float:
        tf = self._tf[doc_idx]
        dl = self._doc_len[doc_idx]
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = self._idf.get(term, 0.0)
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
            score += idf * (freq * (self.k1 + 1)) / (denom or 1.0)
        return score

    def search(
        self,
        query: str,
        topk: int = 3,
        exclude_idx: Optional[set] = None,
    ) -> List[Paragraph]:
        exclude_idx = exclude_idx or set()
        q_terms = tokenize(query)
        scored = [
            (self._score(i, q_terms), i)
            for i in range(len(self.paragraphs))
            if i not in exclude_idx
        ]
        # Stable ordering: score desc, then original index asc (deterministic).
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self.paragraphs[i] for _, i in scored[:topk]]


def make_retriever(kind: str, paragraphs: List[Paragraph]):
    """Factory: 'bm25' (default, sparse, dependency-free) or 'e5' (dense).

    Both returns expose the same ``.search(query, topk, exclude_idx)`` interface,
    so the agent loop is identical regardless of which is chosen.
    """
    kind = (kind or "bm25").lower()
    if kind == "bm25":
        return BM25Retriever(paragraphs)
    if kind == "e5":
        from .dense import E5Retriever
        return E5Retriever(paragraphs)
    raise ValueError(f"unknown retriever kind {kind!r}; choose 'bm25' or 'e5'")
