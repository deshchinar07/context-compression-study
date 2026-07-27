
from __future__ import annotations

import argparse
import os
import sys

from .tokenizer import BACKBONE_MODEL


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
    from .runner import run_grid
    from .tokenizer import BACKBONE_MODEL

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
        prompt_variant=args.prompt_variant,
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
    pr.add_argument("--policies", default="H0,H1,H2,H3,Oracle")
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
    pr.add_argument("--prompt-variant", default="v0")
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
    sys.exit(main())
