"""Experiment runner: sweep policies x budgets x datasets x objective-counts.

For each example we run *every* policy under identical conditions (same backbone,
same decoding, same retriever, same prompt), score the prediction under both the
primary ``mem1_table`` protocol and secondary ``standard_qa`` diagnostic, and
append one JSON row per (example, policy, budget) to a results file. A fresh
policy instance is created per run so its compression counters are per-run.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import List, Optional, Sequence

from . import tokenizer
from .agent import ReActAgent
from .data import Example, load_examples
from .llm import LLMBackend
from .policies import build_policy
from .report import score_prediction


def _get_corpus(
    cache: dict,
    source: str,
    dataset: str,
    split: str,
    retrieval: str,
    index_dir: Optional[str],
    seed: int,
    cache_dir: str,
    verbose: bool,
):
    """Build (or reuse) the shared corpus for a group.

    'union' depends on (dataset, split) so is cached per-group; 'kilt' is one
    global index, cached by (source, index_dir) so it loads only once per run.
    """
    from .corpus import build_corpus

    key = (source, index_dir) if source == "kilt" else (source, dataset, split)
    if key not in cache:
        corpus_examples = (
            [] if source == "kilt"
            else load_examples(dataset, split=split, n_objectives=1,
                               limit=None, seed=seed, cache_dir=cache_dir)
        )
        cache[key] = build_corpus(source, corpus_examples, kind=retrieval, index_dir=index_dir)
        if verbose:
            print(
                f"[{dataset}/{split}] corpus '{source}': "
                f"{len(cache[key])} passages ({retrieval})",
                flush=True,
            )
    return cache[key]


def run_grid(
    datasets: Sequence[str],
    splits: Sequence[str],
    n_objectives_list: Sequence[int],
    budgets: Sequence[int],
    policies: Sequence[str],
    out_path: str,
    limit: Optional[int] = None,
    seed: int = 0,
    backend: Optional[LLMBackend] = None,
    max_steps: int = 8,
    topk: int = 3,
    prompt_variant: str = "v0",
    summary_max_words: int = 40,
    retrieval: str = "bm25",
    retrieval_scope: str = "pool",
    corpus_source: str = "union",
    corpus_index_dir: Optional[str] = None,
    cache_dir: str = "data",
    verbose: bool = True,
) -> str:
    backend = backend or LLMBackend()
    retrieval_scope = (retrieval_scope or "pool").lower()
    agent = ReActAgent(
        backend, max_steps=max_steps, topk=topk,
        prompt_variant=prompt_variant, retrieval=retrieval,
        retrieval_scope=retrieval_scope,
    )
    corpus_cache: dict = {}  # reuse one built corpus across groups (esp. global 'kilt')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    meta = {
        "backbone": backend.model,
        "base_url": backend.base_url,
        "temperature": backend.temperature,
        "tokenizer": tokenizer.backend_name(),
        "prompt_variant": prompt_variant,
        "retrieval": retrieval,
        "retrieval_scope": retrieval_scope,
        "corpus_source": corpus_source if retrieval_scope == "corpus" else None,
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
                # Build the shared corpus ONCE per (dataset, split) from the full
                # split's passage universe (not the limited eval slice), then reuse
                # it across every objective-count, budget, example, and rung.
                if retrieval_scope == "corpus":
                    agent.corpus_index = _get_corpus(
                        corpus_cache, corpus_source, dataset, split,
                        retrieval, corpus_index_dir, seed, cache_dir, verbose,
                    )
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
