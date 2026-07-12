"""Aggregate a results JSONL into the decomposition table.

Reports both explicitly named protocols per (dataset, n_objectives, budget):
``mem1_table`` is primary for MEM1 headline comparisons; ``standard_qa`` is the
secondary reasoning-quality diagnostic. The selected metric drives the four key
gaps (H0->H1 timing, H1->H2 selection, H2->H3 selection++, H3->Oracle headroom)
and each rung's % of the Oracle ceiling. Cost columns remain protocol-independent.

This module does no smoothing, weighting, or cherry-picking: it is a plain mean
over whatever rows are present. Interpret honestly.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

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

    Multiple files let you aggregate ladder rows and a separately-produced
    measured baseline (e.g. MEM1 via run_baseline_grid) in a single report.
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


def _measured_baselines_by_task(
    agg: Dict[tuple, Dict[str, dict]]
) -> Dict[tuple, Dict[str, dict]]:
    """(dataset, split, n_obj) -> {policy: metrics} for non-ladder MEASURED rows.

    These come from ``run_baseline_grid`` (e.g. MEM1 run in-harness). They carry no
    token budget (budget=-1), so they are keyed without budget and shown against
    every budget group -- they are budget-independent by construction.
    """
    out: Dict[tuple, Dict[str, dict]] = defaultdict(dict)
    for (dataset, split, n_obj, _budget), by_pol in agg.items():
        for pol, metrics in by_pol.items():
            if pol not in LADDER_ORDER:
                out[(dataset, split, n_obj)][pol] = metrics
    return out


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def load_baselines(path: str) -> dict:
    """Load the published-baseline overlay config (JSON).

    Returns a dict with ``caveats`` (dict) and ``baselines`` (list). Enforces the
    research-integrity rules from the config header: every baseline must carry a
    ``metric`` that is one of our own SCORE_METRICS, and any entry whose ``score``
    is null is kept but flagged as UNVERIFIED so it renders as a TODO, never as a
    number.
    """
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    baselines = cfg.get("baselines", [])
    for b in baselines:
        metric = b.get("metric")
        if metric not in SCORE_METRICS:
            raise ValueError(
                f"baseline {b.get('method')!r} declares metric {metric!r}, which is "
                f"not one of our result keys {SCORE_METRICS}. A published number can "
                "only be overlaid against the exact metric it is comparable to."
            )
    return {"caveats": cfg.get("caveats", {}), "baselines": baselines}


def _matching_baselines(baselines: List[dict], dataset, n_obj, metric: str) -> List[dict]:
    """Baselines whose (dataset, n_objectives, metric) match a report group.

    Split and budget are intentionally NOT matched: published numbers rarely share
    our exact split slice or token budget, and forcing a match would silently hide
    the very comparison the plan is built around. The mismatch is surfaced in the
    printed caveats instead.
    """
    out = []
    for b in baselines:
        if b.get("metric") != metric:
            continue
        if str(b.get("dataset")) != str(dataset):
            continue
        if int(b.get("n_objectives", -1)) != int(n_obj):
            continue
        out.append(b)
    return out


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
                "retrieval_scope": rs[0].get("retrieval_scope", "pool"),
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
                # Gold-retrieval recall: did the retriever surface the gold at all?
                # Separates retrieval error from selection error (see corpus.py).
                # Falls back to 1.0 when unknown (older rows / no gold titles).
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


def _format_overlay(matches: List[dict], metric: str) -> List[str]:
    """Render matching published baselines as an explicitly-caveated block.

    Verified entries (non-null score) print their number; unverified entries
    (score is null) print as TODO with their source, so an un-transcribed number
    can never be mistaken for a measured result.
    """
    lines = ["  --- published (learned/RL, NOT a rung; full-corpus, caveated) ---"]
    for b in matches:
        src = b.get("source", {})
        cite = f"{src.get('arxiv', '?')} {src.get('table', 'table ?')}"
        head = f"    {b.get('method', '?')} [{b.get('backbone', '?')} | {b.get('retrieval_corpus', '?')}]"
        if b.get("score") is None:
            lines.append(f"{head}: score=TODO (transcribe from {cite})")
        else:
            lines.append(f"{head}: {metric}={float(b['score']):.4f}  (src {cite})")
    return lines


def format_report(
    agg: Dict[tuple, Dict[str, dict]],
    metric: str = "mem1_table_mean_f1",
    baselines: Optional[List[dict]] = None,
) -> str:
    if metric not in SCORE_METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {SCORE_METRICS}")
    baselines = baselines or []
    measured = _measured_baselines_by_task(agg)
    lines: List[str] = []
    printed_ladder = False
    for key in sorted(agg.keys()):
        dataset, split, n_obj, budget = key
        by_pol = agg[key]
        present = [p for p in LADDER_ORDER if p in by_pol]
        if not present:
            continue  # pure measured-baseline group (budget=-1); shown inline below
        printed_ladder = True
        scope = next(iter(by_pol.values())).get("retrieval_scope", "pool")
        lines.append("")
        lines.append(
            f"=== {dataset} | split={split} | N_obj={n_obj} | budget={budget} "
            f"| scope={scope} ==="
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
        # MEASURED in-harness baselines (e.g. MEM1) -- real numbers, NO caveat,
        # because they ran under this same retriever/harness. Budget-independent.
        for pol, m in sorted(measured.get((dataset, split, n_obj), {}).items()):
            oracle_val = by_pol.get("Oracle", {}).get(metric, 0.0)
            pct = (100.0 * m[metric] / oracle_val) if oracle_val else float("nan")
            lines.append(
                f"  measured (in-harness, no caveat): {pol:<8}"
                f"{metric}={m[metric]:.4f}  %oracle={pct:.1f}  "
                f"peakTok={m['peak_tokens']:.0f}  suppKept={m['supp_kept_frac']:.2f}  "
                f"recall={m['gold_recall']:.2f}"
            )
        # PUBLISHED baselines -- different corpus/backbone, caveated by construction.
        matches = _matching_baselines(baselines, dataset, n_obj, metric)
        if matches:
            lines.extend(_format_overlay(matches, metric))

    # If nothing had ladder rungs (e.g. aggregating only a baseline file), still
    # surface the measured baselines so the run isn't silently empty.
    if not printed_ladder and measured:
        for (dataset, split, n_obj), pols in sorted(measured.items()):
            lines.append("")
            lines.append(f"=== {dataset} | split={split} | N_obj={n_obj} | (measured baselines only) ===")
            for pol, m in sorted(pols.items()):
                lines.append(
                    f"  {pol:<8}{metric}={m[metric]:.4f}  "
                    f"peakTok={m['peak_tokens']:.0f}  suppKept={m['supp_kept_frac']:.2f}  "
                    f"recall={m['gold_recall']:.2f}"
                )
    return "\n".join(lines)


def _format_caveats(caveats: dict) -> str:
    if not caveats:
        return ""
    out = ["## CAVEATS — read before comparing any published number above"]
    for k, v in caveats.items():
        out.append(f"- [{k}] {v}")
    return "\n".join(out)


def report(path: str, metric: str = "both", baselines_path: Optional[str] = None) -> str:
    aggregated = aggregate(load_rows(path))
    overlay = load_baselines(baselines_path) if baselines_path else {"caveats": {}, "baselines": []}
    bl = overlay["baselines"]
    if metric == "both":
        headline = format_report(aggregated, metric="mem1_table_summed_f1", baselines=bl)
        diagnostic = format_report(aggregated, metric="standard_qa_mean_f1", baselines=bl)
        body = (
            "## HEADLINE — mem1_table_summed_f1 (MEM1 protocol: summed over "
            "objectives, meaned over examples — the key directly comparable to "
            "MEM1's reported F1)\n"
            f"{headline}\n\n"
            "## DIAGNOSTIC — standard_qa_mean_f1 "
            "(reasoning quality with forgiving formatting; NEVER compare to "
            "published tables)\n"
            f"{diagnostic}"
        )
    else:
        body = format_report(aggregated, metric=metric, baselines=bl)
    caveats = _format_caveats(overlay["caveats"]) if bl else ""
    return f"{body}\n\n{caveats}" if caveats else body
