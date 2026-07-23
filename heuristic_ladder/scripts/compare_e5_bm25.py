#!/usr/bin/env python3
"""Compare e5 vs bm25 selection on the matched slices.

For N=1 we filter the bm25 n=200 run down to the same 100 example_ids the e5
n=100 run used (both seed=0), so the comparison is apples-to-apples.
"""
from __future__ import annotations

import json
from pathlib import Path

from plot_ladder_results import LADDER, METRIC, load_rows, mean_by_policy

RES = Path(__file__).resolve().parents[1] / "results"


def ids_of(rows):
    return {r["example_id"] for r in rows}


def filter_ids(rows, keep):
    return [r for r in rows if r["example_id"] in keep]


def gaps(m):
    return {
        "timing": m["H1"] - m["H0"],
        "selection": m["H2"] - m["H1"],
        "sel++": m["H3"] - m["H2"],
        "headroom": m["Oracle"] - m["H3"],
    }


def show(name, bm25_rows, e5_rows, filt):
    bm = mean_by_policy(bm25_rows, **filt)
    e5 = mean_by_policy(e5_rows, **filt)
    gb, ge = gaps(bm), gaps(e5)
    print(f"\n== {name} ==")
    print(f"  {'rung':>8}  {'bm25':>7}  {'e5':>7}  {'delta':>7}")
    for p in LADDER:
        d = e5[p] - bm[p]
        print(f"  {p:>8}  {bm[p]:7.3f}  {e5[p]:7.3f}  {d:+7.3f}")
    print(f"  {'gaps':>8}  {'bm25':>7}  {'e5':>7}")
    for k in ("timing", "selection", "sel++", "headroom"):
        print(f"  {k:>8}  {gb[k]:+7.3f}  {ge[k]:+7.3f}")


def main():
    bm200 = load_rows(RES / "pool_hotpot_n200.jsonl")
    bm_mhp = load_rows(RES / "pool_multiobj_n100.jsonl")
    bm_m2w = load_rows(RES / "pool_multiobj_2wiki.jsonl")

    e5_hp1 = load_rows(RES / "pool_hotpot_n100_e5.jsonl")
    e5_mhp = load_rows(RES / "pool_multiobj_n100_e5.jsonl")
    e5_m2w = load_rows(RES / "pool_multiobj_2wiki_e5.jsonl")

    # N=1: filter bm25 n=200 to the same 100 ids as e5
    keep = ids_of(e5_hp1)
    bm200_match = filter_ids(bm200, keep)

    show("Pool Hotpot N=1 B=512 (bm25 filtered to same 100 ids)",
         bm200_match, e5_hp1,
         dict(dataset="hotpotqa", n_objectives=1, budget=512, scope="pool"))

    show("Pool Hotpot N=2 B=512",
         bm_mhp, e5_mhp,
         dict(dataset="hotpotqa", n_objectives=2, budget=512, scope="pool"))

    show("Pool 2Wiki N=2 B=512",
         bm_m2w, e5_m2w,
         dict(dataset="2wiki", n_objectives=2, budget=512, scope="pool"))


if __name__ == "__main__":
    main()
