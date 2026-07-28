#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

LADDER = ["H0", "H1", "H2", "H3", "Oracle"]
METRIC = "mem1_table_summed_f1"


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or '"_meta"' in line[:20]:
                continue
            rows.append(json.loads(line))
    return rows


def mean_by_policy(
    rows: list[dict],
    *,
    dataset: str | None = None,
    n_objectives: int | None = None,
    budget: int | None = None,
) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if dataset is not None and r.get("dataset") != dataset:
            continue
        if n_objectives is not None and int(r.get("n_objectives", 1)) != n_objectives:
            continue
        if budget is not None and int(r["budget"]) != budget:
            continue
        pol = r["policy"]
        if pol not in LADDER:
            continue
        buckets[pol].append(float(r[METRIC]))
    return {p: (sum(buckets[p]) / len(buckets[p]) if buckets[p] else float("nan")) for p in LADDER}


def mean_recall(rows, **filt) -> float:
    vals = []
    for r in rows:
        if filt.get("dataset") and r.get("dataset") != filt["dataset"]:
            continue
        if filt.get("n_objectives") is not None and int(r.get("n_objectives", 1)) != filt["n_objectives"]:
            continue
        if filt.get("budget") is not None and int(r["budget"]) != filt["budget"]:
            continue
        tot = r.get("gold_titles_total") or 0
        got = r.get("gold_titles_retrieved") or 0
        vals.append(got / tot if tot else 1.0)
    return sum(vals) / len(vals) if vals else float("nan")


def n_examples(rows, **filt) -> int:
    ids = set()
    for r in rows:
        if filt.get("dataset") and r.get("dataset") != filt["dataset"]:
            continue
        if filt.get("n_objectives") is not None and int(r.get("n_objectives", 1)) != filt["n_objectives"]:
            continue
        if filt.get("budget") is not None and int(r["budget"]) != filt["budget"]:
            continue
        ids.add(r["example_id"])
    return len(ids)


def build_runs(results_dir: Path) -> list[dict]:
    # Canonical BM25 grid: results/rerun/ (full re-run under the corrected harness).
    rerun = results_dir / "rerun"
    hotpot_n1 = load_rows(rerun / "rerun_hotpot_n1.jsonl")
    hotpot_n2 = load_rows(rerun / "rerun_hotpot_n2.jsonl")
    hotpot_n8 = load_rows(rerun / "rerun_hotpot_n8.jsonl")
    wiki_n2 = load_rows(rerun / "rerun_2wiki_n2.jsonl")
    wiki_n8 = load_rows(rerun / "rerun_2wiki_n8.jsonl")

    specs = [
        ("Pool Hotpot N=1 B=512", hotpot_n1, dict(dataset="hotpotqa", n_objectives=1, budget=512)),
        ("Pool Hotpot N=1 B=256", hotpot_n1, dict(dataset="hotpotqa", n_objectives=1, budget=256)),
        ("Pool Hotpot N=2 B=512", hotpot_n2, dict(dataset="hotpotqa", n_objectives=2, budget=512)),
        ("Pool Hotpot N=8 B=512", hotpot_n8, dict(dataset="hotpotqa", n_objectives=8, budget=512)),
        ("Pool 2Wiki N=2 B=512", wiki_n2, dict(dataset="2wiki", n_objectives=2, budget=512)),
        ("Pool 2Wiki N=8 B=512", wiki_n8, dict(dataset="2wiki", n_objectives=8, budget=512)),
    ]

    runs = []
    for name, rows, filt in specs:
        means = mean_by_policy(rows, **filt)
        if all(v != v for v in means.values()):
            continue
        runs.append(
            {
                "name": name,
                "n": n_examples(rows, **filt),
                "recall": mean_recall(rows, **filt),
                "f1": means,
                "gaps": {
                    "timing": means["H1"] - means["H0"],
                    "selection": means["H2"] - means["H1"],
                    "selection++": means["H3"] - means["H2"],
                    "headroom": means["Oracle"] - means["H3"],
                },
            }
        )
    return runs


def build_e5_comparison(results_dir: Path) -> list[dict]:
    bm200 = load_rows(results_dir / "pool_hotpot_n200.jsonl")
    bm_mhp = load_rows(results_dir / "pool_multiobj_n100.jsonl")
    bm_m2w = load_rows(results_dir / "pool_multiobj_2wiki.jsonl")
    e5_hp1 = load_rows(results_dir / "pool_hotpot_n100_e5.jsonl")
    e5_mhp = load_rows(results_dir / "pool_multiobj_n100_e5.jsonl")
    e5_m2w = load_rows(results_dir / "pool_multiobj_2wiki_e5.jsonl")

    keep = {r["example_id"] for r in e5_hp1}
    bm200_match = [r for r in bm200 if r["example_id"] in keep]

    out = []
    for name, bm_rows, e5_rows, filt in [
        ("Pool Hotpot N=1", bm200_match, e5_hp1,
         dict(dataset="hotpotqa", n_objectives=1, budget=512)),
        ("Pool Hotpot N=2", bm_mhp, e5_mhp,
         dict(dataset="hotpotqa", n_objectives=2, budget=512)),
        ("Pool 2Wiki N=2", bm_m2w, e5_m2w,
         dict(dataset="2wiki", n_objectives=2, budget=512)),
    ]:
        bm = mean_by_policy(bm_rows, **filt)
        e5 = mean_by_policy(e5_rows, **filt)
        out.append({"name": name, "bm25": bm, "e5": e5})
    return out


def plot_e5_comparison(comp: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []


    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(comp))
    width = 0.35
    for i, pol in enumerate(LADDER):
        xs_bm = [xi + (i - 2.5) * width / 2.5 for xi in x]
        xs_e5 = [xi + (i - 2.5) * width / 2.5 + width for xi in x]
        ax.bar(xs_bm, [c["bm25"][pol] for c in comp], width / 2.5, label=f"{pol} bm25", alpha=0.55)
        ax.bar(xs_e5, [c["e5"][pol] for c in comp], width / 2.5, label=f"{pol} e5")
    ax.set_xticks(list(x))
    ax.set_xticklabels([c["name"] for c in comp], fontsize=9)
    ax.set_ylabel(f"{METRIC} (mean)")
    ax.set_title("e5 vs bm25 — F1 by rung (B=512)")
    ax.legend(ncols=5, fontsize=6, loc="upper left")
    fig.tight_layout()
    p1 = out_dir / "e5_vs_bm25_f1.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)


    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(comp))
    width = 0.35
    bm_sel = [c["bm25"]["H2"] - c["bm25"]["H1"] for c in comp]
    e5_sel = [c["e5"]["H2"] - c["e5"]["H1"] for c in comp]
    ax.bar([xi - width / 2 for xi in x], bm_sel, width, label="bm25 selection (H2−H1)")
    ax.bar([xi + width / 2 for xi in x], e5_sel, width, label="e5 selection (H2−H1)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([c["name"] for c in comp], fontsize=9)
    ax.set_ylabel("Δ F1 (selection gap)")
    ax.set_title("Does semantic selection rescue the selection arrow? (B=512)")
    ax.legend()
    fig.tight_layout()
    p2 = out_dir / "e5_vs_bm25_selection_gap.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)

    return paths


def print_table(runs: list[dict]) -> None:
    header = f"{'run':28} {'n':>4}  " + " ".join(f"{p:>7}" for p in LADDER) + "  recall"
    print(header)
    print("-" * len(header))
    for run in runs:
        f1 = run["f1"]
        cells = " ".join(f"{f1[p]:7.3f}" for p in LADDER)
        print(f"{run['name']:28} {run['n']:4d}  {cells}  {run['recall']:6.2f}")
    print()
    print("Gaps (H0→H1 timing, H1→H2 selection, H2→H3 ++, H3→Oracle headroom)")
    print(f"{'run':28}  {'tim':>7} {'sel':>7} {'sel++':>7} {'head':>7}")
    for run in runs:
        g = run["gaps"]
        print(
            f"{run['name']:28}  {g['timing']:+7.3f} {g['selection']:+7.3f} "
            f"{g['selection++']:+7.3f} {g['headroom']:+7.3f}"
        )


def plot_runs(runs: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []


    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(runs))
    width = 0.15
    for i, pol in enumerate(LADDER):
        xs = [xi + (i - 2) * width for xi in x]
        ys = [run["f1"][pol] for run in runs]
        ax.bar(xs, ys, width, label=pol)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["name"].replace(" ", "\n") for r in runs], fontsize=8)
    ax.set_ylabel(f"{METRIC} (mean)")
    ax.set_title("Heuristic ladder F1 across runs")
    ax.legend(ncols=5, fontsize=8, loc="upper left")
    ax.set_ylim(0, max(max(r["f1"].values()) for r in runs) * 1.15)
    fig.tight_layout()
    p1 = out_dir / "ladder_f1_by_run.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)


    focus = [r for r in runs if "B=512" in r["name"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in focus:
        ax.plot(LADDER, [run["f1"][p] for p in LADDER], marker="o", label=run["name"])
    ax.set_xlabel("Ladder rung")
    ax.set_ylabel(f"{METRIC} (mean)")
    ax.set_title("Ladder curves @ budget=512")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = out_dir / "ladder_curves_b512.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)


    fig, ax = plt.subplots(figsize=(9, 4.5))
    gap_keys = ["timing", "selection", "selection++", "headroom"]
    x = range(len(focus))
    width = 0.2
    for i, gk in enumerate(gap_keys):
        xs = [xi + (i - 1.5) * width for xi in x]
        ys = [run["gaps"][gk] for run in focus]
        ax.bar(xs, ys, width, label=gk)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["name"].replace(" ", "\n") for r in focus], fontsize=8)
    ax.set_ylabel("Δ F1")
    ax.set_title("Decomposition gaps @ budget=512")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p3 = out_dir / "ladder_gaps_b512.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    paths.append(p3)

    return paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot ladder decomposition figures (F1 by rung, curves, gaps) from results JSONL files."
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "figures",
    )
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    runs = build_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no runs found under {args.results_dir}")

    print(f"Metric: {METRIC}\n")
    print_table(runs)

    if not args.no_plot:
        paths = plot_runs(runs, args.out_dir)

        try:
            comp = build_e5_comparison(args.results_dir)
        except FileNotFoundError:
            comp = []
        if comp:
            paths += plot_e5_comparison(comp, args.out_dir)
        print("\nWrote:")
        for p in paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
