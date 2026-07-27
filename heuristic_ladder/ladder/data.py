
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


SPLIT_ALIASES = {"dev": "train", "test": "validation", "train": "train", "validation": "validation"}


@dataclass
class Paragraph:
    idx: int
    title: str
    text: str
    is_supporting: bool
    objective_idx: int = 0


@dataclass
class Example:

    id: str
    questions: List[str]
    answers: List[List[str]]
    paragraphs: List[Paragraph]
    dataset: str
    n_objectives: int = 1

    @property
    def supporting_titles(self) -> set:
        return {p.title for p in self.paragraphs if p.is_supporting}


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
    if dataset not in DATASET_IDS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {list(DATASET_IDS)}")
    hf_split = SPLIT_ALIASES.get(split, split)
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, dataset, hf_split)

    if os.path.exists(path):
        examples = [_deserialize(line) for line in open(path, encoding="utf-8") if line.strip()]
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


def build_multi_objective(
    singles: List[Example],
    n_objectives: int,
    seed: int = 0,
) -> List[Example]:
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
    
    
    singles = load_single(dataset, split=split, cache_dir=cache_dir, limit=None)
    if n_objectives <= 1:
        out = singles
    else:
        out = build_multi_objective(singles, n_objectives, seed=seed)
    if limit is not None:
        out = out[:limit]
    return out
