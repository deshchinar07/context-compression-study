"""Query-relevance scoring for the *selection* decision (H2, H3).

This is the "cheap lexical overlap" scorer from the plan: BM25-style scoring of a
block's text against the task query, using IDF estimated over the example's own
paragraph pool. It is training-free, deterministic, and offline.

It is deliberately kept separate from the retriever so that "what the agent can
find" (retrieval) and "what the policy chooses to keep" (selection) are two
distinct, independently-auditable mechanisms.

An optional embedding-based scorer is provided behind the same interface for the
prompt/representation-sensitivity checks, but lexical is the default so the
heuristic ladder carries *zero* training cost.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional

from .retriever import tokenize


class LexicalScorer:
    """BM25-style relevance of arbitrary text against a fixed query.

    The corpus statistics (IDF, average document length) can either be computed
    from ``corpus_texts`` at construction (the per-pool case) or supplied directly
    via ``idf``/``avgdl`` (the shared-corpus case, where the expensive index is
    built once and reused across many queries -- see ``ScorerFactory``).
    """

    def __init__(
        self,
        corpus_texts: Optional[List[str]] = None,
        query: str = "",
        k1: float = 1.5,
        b: float = 0.75,
        *,
        idf: Optional[dict] = None,
        avgdl: Optional[float] = None,
    ):
        self.k1 = k1
        self.b = b
        self.q_terms = tokenize(query)
        if idf is not None and avgdl is not None:
            self._idf = idf
            self._avgdl = avgdl
        else:
            self._idf, self._avgdl = self.index_corpus(corpus_texts or [])

    @staticmethod
    def index_corpus(corpus_texts: List[str]) -> tuple:
        """Precompute (idf, avgdl) over a corpus once; shareable across queries."""
        docs = [tokenize(t) for t in corpus_texts]
        n = max(1, len(docs))
        avgdl = (sum(len(d) for d in docs) / n) if docs else 1.0
        df: Counter = Counter()
        for d in docs:
            for term in set(d):
                df[term] += 1
        idf = {
            t: max(1e-6, math.log((n - freq + 0.5) / (freq + 0.5) + 1.0))
            for t, freq in df.items()
        }
        return idf, avgdl

    def score(self, text: str) -> float:
        tf = Counter(tokenize(text))
        dl = sum(tf.values())
        s = 0.0
        for term in self.q_terms:
            if term not in tf:
                continue
            idf = self._idf.get(term, 0.0)
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0))
            s += idf * (freq * (self.k1 + 1)) / (denom or 1.0)
        return s


def make_scorer(kind: str, corpus_texts: List[str], query: str):
    """Factory mirroring ``retriever.make_retriever``: 'bm25' lexical or 'e5' dense.

    The selection rungs (H2/H3) only call ``.score(text)``, so either backend is a
    drop-in. Keeping retriever and scorer on the *same* backend is what makes an
    e5-vs-e5 (or bm25-vs-bm25) comparison against the papers internally consistent.
    """
    kind = (kind or "bm25").lower()
    if kind == "bm25":
        return LexicalScorer(corpus_texts, query)
    if kind == "e5":
        from .dense import E5Scorer
        return E5Scorer(corpus_texts, query)
    raise ValueError(f"unknown scorer kind {kind!r}; choose 'bm25' or 'e5'")


class ScorerFactory:
    """Build corpus statistics once, then mint cheap per-query scorers.

    Used by the shared ``corpus`` retrieval scope, where a single large passage
    set backs every example: computing IDF per run would be prohibitive, so we
    index the corpus once here and hand out lightweight per-query scorers that
    reuse it. For the 'e5' backend there is nothing to precompute (the scorer just
    embeds the query), so ``for_query`` builds one directly.
    """

    def __init__(self, kind: str, corpus_texts: List[str]):
        self.kind = (kind or "bm25").lower()
        if self.kind == "bm25":
            self._idf, self._avgdl = LexicalScorer.index_corpus(corpus_texts)
        elif self.kind == "e5":
            self._idf = self._avgdl = None
        else:
            raise ValueError(f"unknown scorer kind {self.kind!r}; choose 'bm25' or 'e5'")

    def for_query(self, query: str):
        if self.kind == "bm25":
            return LexicalScorer(query=query, idf=self._idf, avgdl=self._avgdl)
        from .dense import E5Scorer
        return E5Scorer([], query)
