"""Answer-quality metrics + result aggregation into the decomposition table.

Two clearly separated evaluation protocols are reported and never conflated:

* ``mem1_table`` is the primary/headline protocol: lowercase, replace punctuation
  with spaces, normalize whitespace (articles are *not* removed); set-based token
  overlap for F1; strict semicolon splitting; zero for the entire example when the
  predicted answer count is wrong. This is the protocol directly comparable to the
  MEM1 paper's reported F1.
* ``standard_qa`` is a secondary diagnostic: conventional SQuAD/HotpotQA
  normalization and Counter-based token F1, padding/truncating malformed
  multi-objective output so correctly answered objectives still receive credit.
  Never compare a ``standard_qa`` number directly against a MEM1 table.

``aggregate`` groups result rows by (dataset, split, n_objectives, budget) and
means them plainly -- no smoothing, weighting, or cherry-picking. ``format_report``
renders the ladder rungs and the four decomposition gaps (H0->H1 timing, H1->H2
selection, H2->H3 selection++, H3->Oracle headroom) and each rung's percent of the
Oracle ceiling. Cost columns are protocol-independent.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence

# --- metrics -------------------------------------------------------------------

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

# --- aggregation --------------------------------------------------------------

LADDER_ORDER = ["H0", "H1", "H2", "H3", "Oracle"]
SCORE_METRICS = [
    "mem1_table_summed_f1",
    "mem1_table_summed_em",
    "mem1_table_mean_f1",
    "mem1_table_mean_em",
    "standard_qa_summed_f1",
    "standard_qa_summed_em",
    "standard_qa_mean_f1",
    "standard_qa_mean_em",
]


def load_rows(path: str) -> List[dict]:
    """Load result rows from one path or several comma-separated paths.

    Multiple files let you combine several runs (e.g. a bm25 run and an e5 run)
    into a single report.
    """
    rows = []
    for one in str(path).split(","):
        one = one.strip()
        if not one:
            continue
        for line in open(one, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "_meta" in obj:
                continue
            rows.append(obj)
    return rows


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(rows: List[dict]) -> Dict[tuple, Dict[str, dict]]:
    """group -> policy -> aggregated metrics."""
    groups: Dict[tuple, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["dataset"], r.get("split", "?"), r["n_objectives"], r["budget"])
        groups[key][r["policy"]].append(r)

    out: Dict[tuple, Dict[str, dict]] = {}
    for key, by_policy in groups.items():
        out[key] = {}
        for pol, rs in by_policy.items():
            missing = [metric for metric in SCORE_METRICS if metric not in rs[0]]
            if missing:
                raise ValueError(
                    "results file predates the dual scoring protocols and cannot "
                    f"support a MEM1-table comparison (missing {missing}). Re-run "
                    "the experiment to produce explicit mem1_table_* and "
                    "standard_qa_* fields."
                )
            metrics = {
                "n": len(rs),
                "peak_tokens": _mean([x["peak_context_tokens"] for x in rs]),
                "infer_tokens": _mean([x["prompt_tokens"] + x["completion_tokens"] for x in rs]),
                "llm_calls": _mean([x["llm_calls"] for x in rs]),
                "compress_triggered": _mean([x["compress_triggered"] for x in rs]),
                "compress_dropped": _mean([x["compress_dropped"] for x in rs]),
                "compress_summarized": _mean([x["compress_summarized"] for x in rs]),
                "supp_kept_frac": _mean(
                    [
                        (x["final_supporting_kept"] / x["final_supporting_total"])
                        if x["final_supporting_total"] else 1.0
                        for x in rs
                    ]
                ),
                # Gold-retrieval recall: did the retriever surface the gold at all
                # across the agent's searches? Separates retrieval miss (gold ranked
                # below top-k) from selection error. Falls back to 1.0 when unknown.
                "gold_recall": _mean(
                    [
                        (x["gold_titles_retrieved"] / x["gold_titles_total"])
                        if x.get("gold_titles_total") else 1.0
                        for x in rs
                    ]
                ),
            }
            for metric in SCORE_METRICS:
                metrics[metric] = _mean([x[metric] for x in rs])
            out[key][pol] = metrics
    return out



def format_report(
    agg: Dict[tuple, Dict[str, dict]],
    metric: str = "mem1_table_mean_f1",
) -> str:
    if metric not in SCORE_METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {SCORE_METRICS}")
    lines: List[str] = []
    for key in sorted(agg.keys()):
        dataset, split, n_obj, budget = key
        by_pol = agg[key]
        present = [p for p in LADDER_ORDER if p in by_pol]
        if not present:
            continue
        lines.append("")
        lines.append(
            f"=== {dataset} | split={split} | N_obj={n_obj} | budget={budget} ==="
        )
        header = f"{'rung':<8}{'n':>5}{'  '}{metric:>9}{'%oracle':>9}{'peakTok':>9}{'inferTok':>9}{'compr':>7}{'summ':>6}{'suppKept':>9}{'recall':>8}"
        lines.append(header)
        for p in present:
            m = by_pol[p]
            oracle_val = by_pol.get("Oracle", {}).get(metric, 0.0)
            pct = (100.0 * m[metric] / oracle_val) if oracle_val else float("nan")
            lines.append(
                f"{p:<8}{m['n']:>5}  {m[metric]:>9.4f}{pct:>9.1f}{m['peak_tokens']:>9.0f}"
                f"{m['infer_tokens']:>9.0f}{m['compress_dropped']:>7.1f}"
                f"{m['compress_summarized']:>6.1f}{m['supp_kept_frac']:>9.2f}"
                f"{m['gold_recall']:>8.2f}"
            )
        # the four decomposition gaps
        def g(a, b):
            if a in by_pol and b in by_pol:
                return by_pol[b][metric] - by_pol[a][metric]
            return float("nan")
        lines.append(
            "  gaps: "
            f"timing(H0->H1)={g('H0','H1'):+.4f}  "
            f"selection(H1->H2)={g('H1','H2'):+.4f}  "
            f"selection++(H2->H3)={g('H2','H3'):+.4f}  "
            f"headroom(H3->Oracle)={g('H3','Oracle'):+.4f}"
        )
    return "\n".join(lines)


def report(path: str, metric: str = "both") -> str:
    aggregated = aggregate(load_rows(path))
    if metric == "both":
        headline = format_report(aggregated, metric="mem1_table_summed_f1")
        diagnostic = format_report(aggregated, metric="standard_qa_mean_f1")
        return (
            "## HEADLINE -- mem1_table_summed_f1 (MEM1 protocol: summed over "
            "objectives, meaned over examples -- the key directly comparable to "
            "MEM1's reported F1)\n"
            f"{headline}\n\n"
            "## DIAGNOSTIC -- standard_qa_mean_f1 "
            "(reasoning quality with forgiving formatting; NEVER compare to "
            "published tables)\n"
            f"{diagnostic}"
        )
    return format_report(aggregated, metric=metric)
