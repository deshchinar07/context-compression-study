"""Dataset loading and the multi-objective construction.

Three oracle-enabled multi-hop QA datasets, all of which ship with
supporting-fact / supporting-passage annotations (so the Oracle rung is built
directly from the data, never hand-judged):

  hotpotqa  -> hotpotqa/hotpot_qa            (config "distractor")
  2wiki     -> scholarly-shadows-syndicate/2wikimultihopqa
  musique   -> dgslibisey/MuSiQue

Everything is normalised into the ``Example`` schema below. A per-example
paragraph *pool* (gold + distractor paragraphs) is what the retriever searches,
so the whole study is self-contained -- no 20GB Wikipedia index required -- and
the supporting labels map cleanly onto retrieved paragraphs.

The MEM1 multi-objective construction (``build_multi_objective``) concatenates N
single questions into one task; the union of their pools becomes the passage
set and a paragraph is "supporting" if it supports *any* sub-question. summed-F1
/ summed-EM over the N answers is then directly comparable to MEM1's tables.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

DATASET_IDS = {
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor"),
    "2wiki": ("scholarly-shadows-syndicate/2wikimultihopqa", None),
    "musique": ("dgslibisey/MuSiQue", None),
}

# For these datasets the official test split has no public labels, so the
# validation split is the held-out "test" and a slice of train is the "dev" set
# used for any heuristic hyperparameter choices.
SPLIT_ALIASES = {"dev": "train", "test": "validation", "train": "train", "validation": "validation"}


@dataclass
class Paragraph:
    idx: int
    title: str
    text: str
    is_supporting: bool
    objective_idx: int = 0  # which sub-question this paragraph belongs to


@dataclass
class Example:
    """A (possibly multi-objective) QA task with a self-contained paragraph pool."""

    id: str
    questions: List[str]                 # one entry per objective
    answers: List[List[str]]             # acceptable gold answers, per objective
    paragraphs: List[Paragraph]          # candidate pool (union across objectives)
    dataset: str
    n_objectives: int = 1

    @property
    def supporting_titles(self) -> set:
        return {p.title for p in self.paragraphs if p.is_supporting}


# --------------------------------------------------------------------------- #
# Per-dataset normalisation into single-objective Examples.
# --------------------------------------------------------------------------- #
def _from_hotpot(row: dict) -> Example:
    titles = row["context"]["title"]
    sents = row["context"]["sentences"]
    support = set(row["supporting_facts"]["title"])
    paras = [
        Paragraph(idx=i, title=t, text=" ".join(s).strip(), is_supporting=(t in support))
        for i, (t, s) in enumerate(zip(titles, sents))
    ]
    return Example(
        id=str(row["id"]),
        questions=[row["question"]],
        answers=[[row["answer"]]],
        paragraphs=paras,
        dataset="hotpotqa",
    )


def _from_2wiki(row: dict) -> Example:
    # This mirror stores context/supporting_facts as JSON strings; parse if needed.
    context = json.loads(row["context"]) if isinstance(row["context"], str) else row["context"]
    sf = row["supporting_facts"]
    sf = json.loads(sf) if isinstance(sf, str) else sf
    support = {t for t, _ in sf}
    paras = [
        Paragraph(idx=i, title=t, text=" ".join(s).strip(), is_supporting=(t in support))
        for i, (t, s) in enumerate(context)
    ]
    return Example(
        id=str(row["_id"]),
        questions=[row["question"]],
        answers=[[row["answer"]]],
        paragraphs=paras,
        dataset="2wiki",
    )


def _from_musique(row: dict) -> Example:
    paras = [
        Paragraph(
            idx=i,
            title=p["title"],
            text=p["paragraph_text"].strip(),
            is_supporting=bool(p["is_supporting"]),
        )
        for i, p in enumerate(row["paragraphs"])
    ]
    gold = [row["answer"]] + list(row.get("answer_aliases") or [])
    return Example(
        id=str(row["id"]),
        questions=[row["question"]],
        answers=[gold],
        paragraphs=paras,
        dataset="musique",
    )


_CONVERTERS = {"hotpotqa": _from_hotpot, "2wiki": _from_2wiki, "musique": _from_musique}


# --------------------------------------------------------------------------- #
# Loading + local JSONL cache (download once, extract locally forever after).
# --------------------------------------------------------------------------- #
def _cache_path(cache_dir: str, dataset: str, split: str) -> str:
    return os.path.join(cache_dir, f"{dataset}.{split}.jsonl")


def _serialize(ex: Example) -> str:
    return json.dumps(asdict(ex), ensure_ascii=False)


def _deserialize(line: str) -> Example:
    d = json.loads(line)
    d["paragraphs"] = [Paragraph(**p) for p in d["paragraphs"]]
    return Example(**d)


def load_single(
    dataset: str,
    split: str = "validation",
    cache_dir: str = "data",
    limit: Optional[int] = None,
) -> List[Example]:
    """Load single-objective examples from a local JSONL cache.

    On a cache miss: download from HuggingFace, convert, write
    ``{cache_dir}/{dataset}.{split}.jsonl``, then return. On a hit: read that
    file only (no network). ``split`` accepts aliases in ``SPLIT_ALIASES``.
    """
    if dataset not in DATASET_IDS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {list(DATASET_IDS)}")
    hf_split = SPLIT_ALIASES.get(split, split)
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, dataset, hf_split)

    if os.path.exists(path):
        examples = [_deserialize(l) for l in open(path, encoding="utf-8") if l.strip()]
    else:
        examples = _download(dataset, hf_split)
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(_serialize(ex) + "\n")

    if limit is not None:
        examples = examples[:limit]
    return examples


def _download(dataset: str, hf_split: str) -> List[Example]:
    from datasets import load_dataset

    repo, config = DATASET_IDS[dataset]
    ds = load_dataset(repo, config) if config else load_dataset(repo)
    if hf_split not in ds:
        raise KeyError(f"{dataset} has no split {hf_split!r}; available: {list(ds.keys())}")
    convert = _CONVERTERS[dataset]
    return [convert(row) for row in ds[hf_split]]


# --------------------------------------------------------------------------- #
# MEM1-style multi-objective construction.
# --------------------------------------------------------------------------- #
def build_multi_objective(
    singles: List[Example],
    n_objectives: int,
    seed: int = 0,
) -> List[Example]:
    """Concatenate consecutive single-objective examples into N-objective tasks.

    Deterministic: examples are shuffled once with ``seed`` then chunked, so a
    given (dataset, split, seed, N) always yields the same tasks. Paragraph pools
    are unioned and re-indexed; each paragraph keeps the objective it came from.
    """
    if n_objectives <= 1:
        return singles
    rng = random.Random(seed)
    order = list(range(len(singles)))
    rng.shuffle(order)
    shuffled = [singles[i] for i in order]

    merged: List[Example] = []
    for start in range(0, len(shuffled) - n_objectives + 1, n_objectives):
        group = shuffled[start : start + n_objectives]
        questions: List[str] = []
        answers: List[List[str]] = []
        paragraphs: List[Paragraph] = []
        seen_titles: Dict[str, Paragraph] = {}
        for obj_idx, ex in enumerate(group):
            questions.append(ex.questions[0])
            answers.append(ex.answers[0])
            for p in ex.paragraphs:
                # Union by title; a paragraph is supporting if it supports any objective.
                if p.title in seen_titles:
                    if p.is_supporting:
                        seen_titles[p.title].is_supporting = True
                    continue
                new_p = Paragraph(
                    idx=len(paragraphs),
                    title=p.title,
                    text=p.text,
                    is_supporting=p.is_supporting,
                    objective_idx=obj_idx,
                )
                seen_titles[p.title] = new_p
                paragraphs.append(new_p)
        merged.append(
            Example(
                id="+".join(ex.id for ex in group),
                questions=questions,
                answers=answers,
                paragraphs=paragraphs,
                dataset=group[0].dataset,
                n_objectives=n_objectives,
            )
        )
    return merged


def load_examples(
    dataset: str,
    split: str = "test",
    n_objectives: int = 1,
    limit: Optional[int] = None,
    seed: int = 0,
    cache_dir: str = "data",
) -> List[Example]:
    """Top-level loader: single- or multi-objective, JSONL-cached, deterministic."""
    # For multi-objective we build from the full split then truncate, so ``limit``
    # counts *tasks*, not underlying single questions.
    singles = load_single(dataset, split=split, cache_dir=cache_dir, limit=None)
    if n_objectives <= 1:
        out = singles
    else:
        out = build_multi_objective(singles, n_objectives, seed=seed)
    if limit is not None:
        out = out[:limit]
    return out
