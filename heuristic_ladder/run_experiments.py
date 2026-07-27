#!/usr/bin/env python3

from __future__ import annotations


MODE = "run"


DATASETS = ["hotpotqa"]
SPLITS = ["test"]
N_OBJECTIVES = [1]
LIMIT = 20
SEED = 0
CACHE_DIR = "data"


RETRIEVAL = "bm25"


POLICIES = ["H0", "H1", "H2", "H3", "Oracle"]
BUDGETS = [512, 256]
MAX_STEPS = 8
TOPK = 3
PROMPT_VARIANT = "v0"
SUMMARY_MAX_WORDS = 40
RUN_OUT = "results/pool_pilot.jsonl"


MODEL = None
BASE_URL = "https://api.deepinfra.com/v1/openai"
API_KEY_ENV = "DEEPINFRA_API_KEY"
TEMPERATURE = 0.0
MAX_TOKENS = 512


AGG_RESULTS = "results/pool_pilot.jsonl"
AGG_METRIC = "both"


def _backend(model, base_url, api_key_env, temperature, max_tokens=MAX_TOKENS):
    from ladder.llm import LLMBackend
    from ladder.tokenizer import BACKBONE_MODEL

    resolved = model or BACKBONE_MODEL
    if resolved != BACKBONE_MODEL:
        
        
        print(
            f"WARNING: MODEL {resolved!r} != tokenizer BACKBONE_MODEL {BACKBONE_MODEL!r}. "
            "Token budgets are still counted against BACKBONE_MODEL. Only override if "
            "the served model shares that tokenizer."
        )
    return LLMBackend(
        model=resolved, base_url=base_url, api_key_env=api_key_env,
        temperature=temperature, max_tokens=max_tokens,
    )


def main():
    from ladder.cli import _load_dotenv

    _load_dotenv()

    if MODE == "prepare-data":
        from ladder.data import load_examples

        for dataset in DATASETS:
            for split in SPLITS:
                for n in N_OBJECTIVES:
                    ex = load_examples(dataset, split=split, n_objectives=n,
                                       limit=LIMIT, seed=SEED, cache_dir=CACHE_DIR)
                    print(f"{dataset}/{split} N={n}: {len(ex)} examples cached in {CACHE_DIR}/")
        return

    if MODE == "run":
        from ladder.runner import run_grid

        run_grid(
            datasets=DATASETS, splits=SPLITS, n_objectives_list=N_OBJECTIVES,
            budgets=BUDGETS, policies=POLICIES, out_path=RUN_OUT,
            limit=LIMIT, seed=SEED,
            backend=_backend(MODEL, BASE_URL, API_KEY_ENV, TEMPERATURE),
            max_steps=MAX_STEPS, topk=TOPK, prompt_variant=PROMPT_VARIANT,
            summary_max_words=SUMMARY_MAX_WORDS,
            retrieval=RETRIEVAL,
            cache_dir=CACHE_DIR,
        )
        return

    if MODE == "aggregate":
        from ladder.report import report

        print(report(AGG_RESULTS, metric=AGG_METRIC))
        return

    raise SystemExit(
        f"unknown MODE {MODE!r}; choose 'prepare-data', 'run', 'aggregate'."
    )


if __name__ == "__main__":
    raise SystemExit(main())
