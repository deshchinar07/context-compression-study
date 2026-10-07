"""Answer-quality metrics with two clearly separated evaluation protocols.

``mem1_table`` is the primary protocol for headline comparisons against MEM1:

* MEM1's preprocessing (lowercase, replace punctuation with spaces, normalize
  whitespace; articles are *not* removed);
* set-based token overlap for F1;
* strict semicolon splitting; and
* zero for the entire example when the predicted answer count is wrong.

``standard_qa`` is a secondary diagnostic protocol. It uses the conventional
SQuAD/HotpotQA normalization and Counter-based token F1, and pads/truncates
malformed multi-answer output so correctly answered objectives still receive
credit. This helps distinguish reasoning quality from formatting failures.

Never compare a ``standard_qa`` number directly against a MEM1 table.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence


def standard_qa_normalize(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def standard_qa_em(prediction: str, golden_answers) -> int:
    """SQuAD/HotpotQA normalized exact match against any acceptable answer."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    norm_pred = standard_qa_normalize(prediction)
    for gold in golden_answers:
        if standard_qa_normalize(gold) == norm_pred:
            return 1
    return 0


def standard_qa_f1(prediction: str, golden_answers) -> float:
    """SQuAD-style token F1; max over acceptable gold answers."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred_tokens = standard_qa_normalize(prediction).split()
    best = 0.0
    for gold in golden_answers:
        gold_tokens = standard_qa_normalize(gold).split()
        if not pred_tokens and not gold_tokens:
            best = max(best, 1.0)
            continue
        if not pred_tokens or not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def mem1_normalize(s: str) -> str:
    """Match ``MEM1-main/Mem1/inference/eval.py::preprocess_text``."""
    text = s.lower()
    for punct in string.punctuation:
        text = text.replace(punct, " ")
    return re.sub(r"\s+", " ", text).strip()


def mem1_em(prediction: str, golden_answers) -> int:
    """MEM1 normalized exact match against any acceptable answer."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    prediction = mem1_normalize(prediction)
    return int(prediction in [mem1_normalize(gold) for gold in golden_answers])


def mem1_f1(prediction: str, golden_answers) -> float:
    """MEM1's set-based token F1, max over acceptable gold answers."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    pred_tokens = set(mem1_normalize(prediction).split())
    if not pred_tokens:
        return 0.0

    best = 0.0
    for gold in golden_answers:
        gold_tokens = set(mem1_normalize(gold).split())
        if not gold_tokens:
            continue
        common = pred_tokens & gold_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        if precision + recall:
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


@dataclass
class AnswerScore:
    """Per-example scores. For single-objective, the summed_* equal the single value."""

    summed_em: float
    summed_f1: float
    n_objectives: int

    @property
    def mean_em(self) -> float:
        return self.summed_em / self.n_objectives if self.n_objectives else 0.0

    @property
    def mean_f1(self) -> float:
        return self.summed_f1 / self.n_objectives if self.n_objectives else 0.0


@dataclass
class EvaluationScores:
    """Both protocols for one prediction; labels are intentionally explicit."""

    mem1_table: AnswerScore
    standard_qa: AnswerScore


def standard_qa_split(prediction: str, n_expected: int) -> List[str]:
    """Forgiving split: pad/truncate so valid objectives retain diagnostic credit."""
    parts = [p.strip() for p in prediction.split(";")] if prediction else []
    if len(parts) < n_expected:
        parts = parts + [""] * (n_expected - len(parts))
    return parts[:n_expected]


def score_mem1_table(
    prediction: str, gold_per_objective: Sequence[Sequence[str]]
) -> AnswerScore:
    """Exact MEM1 table protocol for answer-content scoring.

    MEM1 splits on semicolons and returns zero for the whole example unless the
    resulting answer count exactly equals the objective count.
    """
    n = len(gold_per_objective)
    predictions = prediction.split(";")
    if len(predictions) != n:
        return AnswerScore(summed_em=0.0, summed_f1=0.0, n_objectives=n)

    summed_em = 0.0
    summed_f1 = 0.0
    for pred, gold in zip(predictions, gold_per_objective):
        summed_em += mem1_em(pred, gold)
        summed_f1 += mem1_f1(pred, gold)
    return AnswerScore(summed_em=summed_em, summed_f1=summed_f1, n_objectives=n)


def score_standard_qa(
    prediction: str, gold_per_objective: Sequence[Sequence[str]]
) -> AnswerScore:
    """Forgiving SQuAD/HotpotQA diagnostic scoring."""
    n = len(gold_per_objective)
    predictions = standard_qa_split(prediction, n)
    summed_em = 0.0
    summed_f1 = 0.0
    for pred, gold in zip(predictions, gold_per_objective):
        summed_em += standard_qa_em(pred, gold)
        summed_f1 += standard_qa_f1(pred, gold)
    return AnswerScore(summed_em=summed_em, summed_f1=summed_f1, n_objectives=n)


def score_prediction(
    prediction: str, gold_per_objective: Sequence[Sequence[str]]
) -> EvaluationScores:
    """Score one prediction under both named protocols.

    ``gold_per_objective`` is a list (one per sub-question) of acceptable gold
    answers. For single-objective this is a length-1 list.
    """
    return EvaluationScores(
        mem1_table=score_mem1_table(prediction, gold_per_objective),
        standard_qa=score_standard_qa(prediction, gold_per_objective),
    )
