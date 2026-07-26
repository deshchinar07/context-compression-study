"""Dense (e5) retrieval + relevance scoring, behind the BM25/lexical interfaces.

This exists to align our retrieval *algorithm* with Search-R1, whose default
retriever is ``intfloat/e5-base-v2`` over a FAISS index. Selecting ``e5`` here swaps
the ranking function for a dense one while leaving everything else in the harness
identical, so an e5-vs-e5 comparison against that setup isolates the same policy
question a BM25-vs-BM25 comparison does -- just at their operating point instead of
a sparse one.

Two classes, each a drop-in for its sparse sibling:
  * ``E5Retriever``  <-> ``retrieval.BM25Retriever``   (``.search(query, topk, exclude_idx)``)
  * ``E5Scorer``     <-> ``retrieval.LexicalScorer``  (``.score(text)``)

e5 requires asymmetric prefixes -- queries are embedded as ``"query: ..."`` and
passages as ``"passage: ..."`` -- which we honor so the numbers match the paper's
setup rather than a mis-prefixed approximation. Embeddings are L2-normalized so a
dot product is cosine similarity.

The model is heavy and optional: it is imported lazily and only when ``e5`` is
actually requested, so the default BM25 path keeps the harness dependency-free and
training-free. Install with ``pip install sentence-transformers`` to use it.
"""

from __future__ import annotations

import functools
from typing import List, Optional

import numpy as np

from .data import Paragraph

E5_MODEL = "intfloat/e5-base-v2"


@functools.lru_cache(maxsize=1)
def _load_model():
    """Load e5 once per process. Raises a clear error if the dep is missing."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "dense retrieval needs sentence-transformers, which is not installed. "
            "Run `pip install sentence-transformers` or use --retrieval bm25 "
            "(the default, dependency-free path)."
        ) from e
    return SentenceTransformer(E5_MODEL)


def _embed(texts: List[str], prefix: str) -> np.ndarray:
    """Embed ``texts`` with the required e5 prefix, L2-normalized (cosine-ready)."""
    model = _load_model()
    prefixed = [f"{prefix}{t}" for t in texts]
    emb = model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
    return np.asarray(emb, dtype=np.float32)


class E5Retriever:
    """Dense retriever over one Example's paragraph pool (BM25Retriever drop-in)."""

    def __init__(self, paragraphs: List[Paragraph]):
        self.paragraphs = paragraphs
        docs = [f"{p.title} {p.text}" for p in paragraphs]
        # empty pool -> empty matrix with the right 2nd dim only once we embed.
        self._doc_emb = _embed(docs, "passage: ") if docs else np.zeros((0, 0), np.float32)

    def search(
        self,
        query: str,
        topk: int = 3,
        exclude_idx: Optional[set] = None,
    ) -> List[Paragraph]:
        exclude_idx = exclude_idx or set()
        if len(self.paragraphs) == 0:
            return []
        q = _embed([query], "query: ")[0]
        sims = self._doc_emb @ q  # cosine, both normalized
        # score desc, then original index asc -- identical tie-break to BM25Retriever.
        order = sorted(
            (i for i in range(len(self.paragraphs)) if i not in exclude_idx),
            key=lambda i: (-float(sims[i]), i),
        )
        return [self.paragraphs[i] for i in order[:topk]]


class E5Scorer:
    """Dense query-relevance of arbitrary text (LexicalScorer drop-in).

    Query is embedded once at construction; each ``score`` embeds the candidate
    text. Cosine similarity is shifted into ``[0, 2]`` (``1 + cos``) so scores stay
    non-negative like the BM25 scorer, preserving the ``min``-selects-least-relevant
    semantics the policies rely on.
    """

    def __init__(self, corpus_texts: List[str], query: str, **_ignore):
        # corpus_texts is accepted for interface parity with LexicalScorer (it uses
        # it to estimate IDF); e5 needs no corpus statistics, so it is ignored.
        self._q = _embed([query], "query: ")[0]

    def score(self, text: str) -> float:
        if not text.strip():
            return 0.0
        v = _embed([text], "passage: ")[0]
        return 1.0 + float(v @ self._q)
