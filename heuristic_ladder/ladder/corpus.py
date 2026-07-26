"""Shared retrieval corpus (the ``corpus`` retrieval scope).

The default retrieval scope is ``pool``: the retriever searches only an Example's
own ~10-paragraph bundled pool, and gold labels come straight from the dataset
(``Paragraph.is_supporting``). That keeps the Oracle trivially constructible but
makes retrieval unrealistically easy -- the gold is always present.

The ``corpus`` scope instead searches ONE large, shared passage index. Retrieved
passages are arbitrary corpus documents with no dataset label, so the gold label
is assigned at retrieval time by *title match* against the example's
supporting-fact titles (see ``agent.py``). Retrieval can now genuinely miss the
gold, which is the realistic condition we want; the Oracle becomes
"perfect selection *given what was retrieved*", and gold-retrieval recall is
reported separately so retrieval error and selection error stay distinguishable.

Corpus sources
--------------
* ``union`` (implemented, local, no download): the deduplicated union of every
  paragraph across the dataset split. Thousands of passages -- big enough for
  real recall failures, small enough to index in memory on a laptop with BM25.
  Because gold passages come from the same dataset, their titles match exactly,
  so title-based labeling is exact for this source.
* ``kilt`` (Stage 2, not yet implemented): the provenance-aligned Wikipedia
  knowledge source behind these datasets, served via a FAISS/e5 index. This is
  the literal "20GB index" tier; it needs a GPU/large-RAM box to build. It plugs
  into the same ``CorpusIndex`` interface, so switching to it is a config change
  here, not a rewrite anywhere else.
"""

from __future__ import annotations

from typing import List, Sequence

from .data import Example, Paragraph
from .retrieval import ScorerFactory, make_retriever


def build_union_corpus(examples: Sequence[Example]) -> List[Paragraph]:
    """Deduplicated union of all paragraphs across ``examples`` (keyed by title).

    Each surviving passage is re-indexed with a fresh corpus-level ``idx`` and
    carries no meaningful ``is_supporting`` flag -- corpus passages are unlabeled
    by construction; labeling happens per-example at retrieval time by title.
    """
    seen: set = set()
    passages: List[Paragraph] = []
    for ex in examples:
        for p in ex.paragraphs:
            if p.title in seen:
                continue
            seen.add(p.title)
            passages.append(
                Paragraph(
                    idx=len(passages),
                    title=p.title,
                    text=p.text,
                    is_supporting=False,  # unused for corpus passages; labeled by title later
                    objective_idx=0,
                )
            )
    return passages


def build_corpus(
    source: str,
    examples: Sequence[Example],
    kind: str = "bm25",
    index_dir: Optional[str] = None,
) -> "CorpusIndex":
    """Build a shared :class:`CorpusIndex` for the requested source.

    ``kind`` is the retrieval/scoring backend ('bm25' or 'e5'), matching the
    ``--retrieval`` flag so retriever and selection scorer stay on one backend.
    ``index_dir`` is only used by the 'kilt' source (path to the prebuilt FAISS
    index); ``examples`` is ignored there since the corpus is dataset-independent.
    """
    source = (source or "union").lower()
    if source == "union":
        passages = build_union_corpus(examples)
        return CorpusIndex(passages, kind=kind)
    if source == "kilt":
        # Full-Wikipedia FAISS/e5 index (Stage 2). Lives in ladder/kilt.py so the
        # heavy, optional deps (faiss, sentence-transformers) stay isolated.
        if not index_dir:
            raise ValueError(
                "corpus-source 'kilt' needs --corpus-index-dir pointing at a "
                "prebuilt FAISS index directory (see ladder/kilt.py for the layout)."
            )
        from .kilt import build_kilt_corpus

        return build_kilt_corpus(index_dir, kind=kind)
    raise ValueError(f"unknown corpus source {source!r}; choose 'union' or 'kilt'")


class CorpusIndex:
    """A shared, build-once retriever + selection-scorer over a fixed passage set.

    Constructed ONCE per (dataset, split) and reused across every example and
    every rung, because building a retriever/IDF index over tens of thousands of
    passages per run would dominate runtime. Exposes the same ``.search`` surface
    the agent expects from a per-pool retriever, plus ``.scorer_for(query)`` to
    mint a cheap per-example selection scorer that shares the corpus-level IDF.
    """

    def __init__(
        self,
        passages: List[Paragraph],
        kind: str = "bm25",
        *,
        retriever=None,
        scorer_factory=None,
    ):
        # ``retriever``/``scorer_factory`` can be injected (e.g. a FAISS-backed
        # retriever for the 'kilt' 20GB tier); otherwise they are built in memory
        # over ``passages`` (the 'union' tier). Either way the interface is
        # identical, so the agent/baseline never know which tier they are on.
        self.passages = passages
        self.kind = (kind or "bm25").lower()
        self._retriever = retriever if retriever is not None else make_retriever(self.kind, passages)
        self._scorer_factory = (
            scorer_factory if scorer_factory is not None
            else ScorerFactory(self.kind, [p.text for p in passages])
        )

    def __len__(self) -> int:
        return len(self.passages)

    def search(self, query: str, topk: int = 3, exclude_idx=None) -> List[Paragraph]:
        return self._retriever.search(query, topk=topk, exclude_idx=exclude_idx)

    def scorer_for(self, query: str):
        return self._scorer_factory.for_query(query)
