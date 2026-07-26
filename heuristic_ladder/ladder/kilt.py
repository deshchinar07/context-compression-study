"""Stage-2 corpus source: the full Wikipedia FAISS/e5 index (the "20GB" tier).

This is the drop-in that makes ``--retrieval-scope corpus --corpus-source kilt``
work. It is intentionally decoupled from the rest of the harness: it only has to
produce a :class:`ladder.corpus.CorpusIndex` whose ``.search`` /
``.scorer_for`` behave like every other corpus, so nothing else in the pipeline
changes when you switch to it.

Why it is separate from ``union``: the union corpus is a few tens of thousands of
passages indexed in memory with BM25 -- trivial on a laptop. The full Wikipedia
corpus is ~21M passages; you cannot BM25-scan a Python list per query, and you
cannot build lexical IDF over 21M docs on the fly. So this tier uses a prebuilt
**FAISS** index over **e5** embeddings for retrieval, and an e5 (embedding)
selection scorer that needs no corpus statistics.

BUILDING THE INDEX (done once, off-box, on a GPU/large-RAM machine)
------------------------------------------------------------------
This module *loads and serves* a prebuilt index; it does not build one (building
21M e5 embeddings is a GPU job). Point ``index_dir`` at a directory containing:

    passages.jsonl   one JSON object per line: {"idx": int, "title": str, "text": str}
                     where ``idx`` is the row's FAISS id (0..N-1, in index order)
    index.faiss      a FAISS index of the e5 passage embeddings (inner-product on
                     L2-normalized vectors == cosine), ids aligned to ``idx``

The standard source is the KILT Wikipedia knowledge source (provenance-aligned to
HotpotQA/2Wiki), embedded with ``intfloat/e5-base-v2`` -- the same retriever
Search-R1 / MEM1 use -- so title-match labeling lines up with the datasets'
supporting-fact titles. (Title normalization between KILT and the datasets is the
one trust requirement here; keep it exact.)

This code path is UNTESTED until a real index exists (there is nothing to test it
against on a laptop). It is written to be correct-by-inspection and fails loudly
with actionable messages if the deps or files are missing.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from .data import Paragraph


def _require(module: str, pip_name: Optional[str] = None):
    try:
        return __import__(module)
    except ImportError as e:  # pragma: no cover - depends on optional heavy deps
        raise RuntimeError(
            f"the 'kilt' corpus source needs {module!r}, which is not installed. "
            f"Install it (`pip install {pip_name or module}`) on the machine that "
            "serves the index."
        ) from e


def _load_passages(path: str) -> List[Paragraph]:
    """Load the passage store; row i must have FAISS id i (validated)."""
    passages: List[Paragraph] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            idx = int(d.get("idx", i))
            if idx != len(passages):
                raise ValueError(
                    f"{path}: passage FAISS ids must be dense 0..N-1 in file order; "
                    f"row {len(passages)} declares idx={idx}."
                )
            passages.append(
                Paragraph(
                    idx=idx,
                    title=d["title"],
                    text=d["text"],
                    is_supporting=False,  # labeled by title match per-example later
                    objective_idx=0,
                )
            )
    return passages


class FaissRetriever:
    """Dense retriever over a prebuilt FAISS index (BM25Retriever drop-in).

    Exposes the same ``.search(query, topk, exclude_idx)`` surface the agent and
    MEM1 baseline call, returning :class:`Paragraph` objects mapped from FAISS ids.
    """

    E5_MODEL = "intfloat/e5-base-v2"

    def __init__(self, index_dir: str):
        if not os.path.isdir(index_dir):
            raise FileNotFoundError(
                f"kilt index_dir {index_dir!r} does not exist. Build the index "
                "off-box and point --corpus-index-dir at it (see ladder/kilt.py)."
            )
        faiss = _require("faiss", "faiss-cpu")
        idx_path = os.path.join(index_dir, "index.faiss")
        psg_path = os.path.join(index_dir, "passages.jsonl")
        for p in (idx_path, psg_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"kilt index is missing required file: {p}")
        self._index = faiss.read_index(idx_path)
        self.passages = _load_passages(psg_path)
        if self._index.ntotal != len(self.passages):
            raise ValueError(
                f"FAISS index has {self._index.ntotal} vectors but passages.jsonl has "
                f"{len(self.passages)} rows; they must be 1:1 and id-aligned."
            )
        # Load the query encoder lazily-shared via the dense module's loader so we
        # reuse one e5 instance across retriever and scorer.
        from .dense import _load_model  # noqa: WPS437 - intentional internal reuse

        self._encode = _load_model()

    def _embed_query(self, query: str):
        np = _require("numpy")
        v = self._encode.encode(
            [f"query: {query}"], normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(v, dtype="float32")

    def search(self, query: str, topk: int = 3, exclude_idx: Optional[set] = None):
        exclude_idx = exclude_idx or set()
        # Over-fetch so we can drop already-seen ids and still return topk.
        k = topk + len(exclude_idx) + 1
        q = self._embed_query(query)
        _scores, ids = self._index.search(q, k)
        out: List[Paragraph] = []
        for i in ids[0]:
            i = int(i)
            if i < 0 or i in exclude_idx:
                continue
            out.append(self.passages[i])
            if len(out) >= topk:
                break
        return out


def build_kilt_corpus(index_dir: str, kind: str = "e5"):
    """Return a CorpusIndex backed by the prebuilt FAISS/e5 index.

    ``kind`` is forced to 'e5' for both retriever and selection scorer: lexical
    IDF over 21M passages is infeasible, and e5 matches the Search-R1/MEM1 setup.
    """
    from .corpus import CorpusIndex
    from .retrieval import ScorerFactory

    retriever = FaissRetriever(index_dir)
    # e5 scorer needs no corpus stats (it embeds the query), so this is cheap.
    scorer_factory = ScorerFactory("e5", [])
    return CorpusIndex(
        passages=retriever.passages,
        kind="e5",
        retriever=retriever,
        scorer_factory=scorer_factory,
    )
