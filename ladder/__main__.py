from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Optional, Sequence

from .tokenizer import BACKBONE_MODEL

if TYPE_CHECKING:
    from .llm import LLMBackend


def _load_dotenv(path: str = ".env") -> None:
    for candidate in (path, os.path.join("..", path)):
        if os.path.exists(candidate):
            for line in open(candidate, encoding="utf-8"):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k.strip(), v)
            return


def _csv(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def _ints(s: str):
    return [int(x) for x in _csv(s)]


def run_grid(
    datasets: Sequence[str],
    splits: Sequence[str],
    n_objectives_list: Sequence[int],
    budgets: Sequence[int],
    policies: Sequence[str],
    out_path: str,
    limit: Optional[int] = None,
    seed: int = 0,
    backend: Optional["LLMBackend"] = None,
    max_steps: int = 8,
    topk: int = 3,
    summary_max_words: int = 40,
    retrieval: str = "bm25",
    cache_dir: str = "data",
    verbose: bool = True,
) -> str:
    from . import tokenizer
    from .agent import ReActAgent
    from .data import load_examples
    from .llm import LLMBackend
    from .policies import build_policy
    from .report import score_prediction

    backend = backend or LLMBackend()
    agent = ReActAgent(
        backend, max_steps=max_steps, topk=topk,
        retrieval=retrieval,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    meta = {
        "backbone": backend.model,
        "base_url": backend.base_url,
        "temperature": backend.temperature,
        "tokenizer": tokenizer.backend_name(),
        "retrieval": retrieval,
        "topk": topk,
        "max_steps": max_steps,
        "summary_max_words": summary_max_words,
        "seed": seed,
        "primary_metric_protocol": "mem1_table",
        "diagnostic_metric_protocol": "standard_qa",
    }

    n_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for dataset in datasets:
            for split in splits:
                for n_obj in n_objectives_list:
                    examples = load_examples(
                        dataset, split=split, n_objectives=n_obj,
                        limit=limit, seed=seed, cache_dir=cache_dir,
                    )
                    for budget in budgets:
                        for ex in examples:
                            for pol_name in policies:
                                policy = build_policy(pol_name, summary_max_words=summary_max_words)
                                t0 = time.time()
                                res = agent.run(ex, policy, budget, summary_max_words=summary_max_words)
                                score = score_prediction(res.prediction, ex.answers)
                                row = asdict(res)
                                row.update(
                                    {
                                        "split": split,
                                        "mem1_table_summed_em": score.mem1_table.summed_em,
                                        "mem1_table_summed_f1": score.mem1_table.summed_f1,
                                        "mem1_table_mean_em": score.mem1_table.mean_em,
                                        "mem1_table_mean_f1": score.mem1_table.mean_f1,
                                        "standard_qa_summed_em": score.standard_qa.summed_em,
                                        "standard_qa_summed_f1": score.standard_qa.summed_f1,
                                        "standard_qa_mean_em": score.standard_qa.mean_em,
                                        "standard_qa_mean_f1": score.standard_qa.mean_f1,
                                        "uses_gold": policy.uses_gold,
                                        "wall_time_s": round(time.time() - t0, 2),
                                    }
                                )
                                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                                f.flush()
                                n_written += 1
                            if verbose:
                                print(
                                    f"[{dataset}/{split} N={n_obj} B={budget}] "
                                    f"ex={ex.id[:24]} done "
                                    f"(rows so far: {n_written})",
                                    flush=True,
                                )
    if verbose:
        print(f"\nWrote {n_written} rows to {out_path}")
    return out_path


def cmd_prepare_data(args):
    from .data import load_examples

    for dataset in _csv(args.datasets):
        for split in _csv(args.splits):
            for n in _ints(args.n_objectives):
                ex = load_examples(
                    dataset, split=split, n_objectives=n,
                    limit=args.limit, seed=args.seed, cache_dir=args.cache_dir,
                )
                print(f"{dataset}/{split} N={n}: {len(ex)} examples cached in {args.cache_dir}/")


def cmd_run(args):
    _load_dotenv()
    from .llm import LLMBackend

    if args.model != BACKBONE_MODEL:
        raise SystemExit(
            f"--model {args.model!r} does not match the fixed backbone "
            f"{BACKBONE_MODEL!r} that ladder/tokenizer.py counts against. "
            "This study is deliberately single-backbone; if you really want to "
            "benchmark a different model, update BACKBONE_MODEL in "
            "ladder/tokenizer.py so the tokenizer and the API calls stay in sync."
        )

    backend = LLMBackend(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    run_grid(
        datasets=_csv(args.datasets),
        splits=_csv(args.splits),
        n_objectives_list=_ints(args.n_objectives),
        budgets=_ints(args.budgets),
        policies=_csv(args.policies),
        out_path=args.out,
        limit=args.limit,
        seed=args.seed,
        backend=backend,
        max_steps=args.max_steps,
        topk=args.topk,
        summary_max_words=args.summary_max_words,
        retrieval=args.retrieval,
        cache_dir=args.cache_dir,
    )


def cmd_aggregate(args):
    from .report import report

    print(report(args.results, metric=args.metric))


def build_parser():
    p = argparse.ArgumentParser(prog="ladder", description="Heuristic-ladder harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("prepare-data", help="download + cache datasets as local JSONL")
    pd.add_argument("--datasets", default="hotpotqa,2wiki,musique")
    pd.add_argument("--splits", default="test")
    pd.add_argument("--n-objectives", dest="n_objectives", default="1")
    pd.add_argument("--limit", type=int, default=None)
    pd.add_argument("--seed", type=int, default=0)
    pd.add_argument("--cache-dir", default="data")
    pd.set_defaults(func=cmd_prepare_data)

    pr = sub.add_parser("run", help="run the experiment grid")
    pr.add_argument("--datasets", default="hotpotqa")
    pr.add_argument("--splits", default="test")
    pr.add_argument("--n-objectives", dest="n_objectives", default="1")
    pr.add_argument("--budgets", default="1024")
    pr.add_argument("--policies", default="H0,H0p,H1,H2,H3,Oracle")
    pr.add_argument("--limit", type=int, default=20)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", default="results/run.jsonl")
    pr.add_argument("--cache-dir", default="data")
    pr.add_argument("--model", default=BACKBONE_MODEL)
    pr.add_argument("--base-url", default="https://api.deepinfra.com/v1/openai")
    pr.add_argument("--temperature", type=float, default=0.0)
    pr.add_argument("--max-tokens", type=int, default=512)
    pr.add_argument("--max-steps", type=int, default=8)
    pr.add_argument("--topk", type=int, default=3)
    pr.add_argument("--summary-max-words", type=int, default=40)
    pr.add_argument(
        "--retrieval",
        default="bm25",
        choices=["bm25", "e5"],
        help=(
            "Ranking backend shared by retrieval and the selection scorer. 'bm25' "
            "(default) is sparse and dependency-free; 'e5' is the intfloat/e5-base-v2 "
            "dense retriever Search-R1 uses (needs sentence-transformers)."
        ),
    )
    pr.set_defaults(func=cmd_run)

    ag = sub.add_parser("aggregate", help="summarize a results file")
    ag.add_argument("--results", required=True,
                    help="one path, or comma-separated (e.g. run1.jsonl,run2.jsonl)")
    ag.add_argument(
        "--metric",
        default="both",
        choices=[
            "both",
            "mem1_table_mean_f1",
            "mem1_table_mean_em",
            "mem1_table_summed_f1",
            "mem1_table_summed_em",
            "standard_qa_mean_f1",
            "standard_qa_mean_em",
            "standard_qa_summed_f1",
            "standard_qa_summed_em",
        ],
        help=(
            "Default 'both' reports headline mem1_table_summed_f1 first (the key "
            "directly comparable to MEM1's reported F1) and diagnostic "
            "standard_qa_mean_f1 second. standard_qa_* is diagnostic only."
        ),
    )
    ag.set_defaults(func=cmd_aggregate)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
