#!/usr/bin/env python3
"""Central entry point + config for the heuristic-ladder harness.

Edit the CONFIG block below and run:  ``python run_experiments.py``
Everything the CLI (`python -m ladder ...`) can do is drivable from here; this
file just puts every knob in one place with labels. Set ``MODE`` to pick the
action, tweak the relevant section, run.

--------------------------------------------------------------------------------
THE THREE RETRIEVAL TIERS (the main axis you will change)
--------------------------------------------------------------------------------
  RETRIEVAL_SCOPE="pool"                      -> each example's ~10-paragraph pool.
      Easy retrieval, EXACT Oracle. Use for development + the clean-Oracle
      primary result. Runs anywhere.
  RETRIEVAL_SCOPE="corpus", CORPUS_SOURCE="union"
      -> one shared index = the dedup union of the split's paragraphs (~66k for
      HotpotQA). Real recall failures, Oracle-given-retrieval, recall reported.
      Local, no GPU. Use as the realistic robustness tier on your laptop.
  RETRIEVAL_SCOPE="corpus", CORPUS_SOURCE="kilt", CORPUS_INDEX_DIR=<path>
      -> the full Wikipedia FAISS/e5 index (the "20GB" open-domain tier, matches
      MEM1's setup). Needs a prebuilt index + a GPU/large-RAM box. Use for the
      final open-domain comparison. Build the index off-box (see ladder/kilt.py).

Keep RETRIEVAL / RETRIEVAL_SCOPE / CORPUS_SOURCE IDENTICAL between a ladder run
and any baseline run you compare it to, or the comparison is confounded.

--------------------------------------------------------------------------------
EQUIVALENT CLI COMMANDS (what each is for)
--------------------------------------------------------------------------------
# 0) One-time download + cache as local JSONL (needs network; do this first).
python -m ladder prepare-data --datasets hotpotqa,2wiki,musique --splits test

# 1) POOL pilot -- smoke-test the whole loop + clean-Oracle numbers (dev default).
python -m ladder run --datasets hotpotqa --n-objectives 1 \
  --budgets 512,256 --policies H0,H1,H2,H3,Oracle --limit 20 \
  --out results/pool_pilot.jsonl

# 2) POOL multi-objective sweep -- the core clean-Oracle decomposition figure.
python -m ladder run --datasets hotpotqa,2wiki --n-objectives 2,8,16,32 \
  --budgets 16000,8000,4000 --policies H0,H1,H2,H3,Oracle --limit 200 \
  --out results/pool_multiobj.jsonl

# 3) UNION corpus run -- same grid under realistic local retrieval (no GPU).
#    Checks whether the bounded pool was compressing the rung gaps.
python -m ladder run --datasets hotpotqa --retrieval-scope corpus --corpus-source union \
  --budgets 512,256 --policies H0,H1,H2,H3,Oracle --limit 200 \
  --out results/union_run.jsonl

# 4) KILT / 20GB run -- final open-domain tier (GPU box + prebuilt index).
python -m ladder run --datasets hotpotqa --retrieval-scope corpus --corpus-source kilt \
  --corpus-index-dir /path/to/kilt_index --retrieval e5 \
  --budgets 4000 --policies H0,H1,H2,H3,Oracle --limit 200 \
  --out results/kilt_run.jsonl

# 5) MEM1 baseline in-harness -- MUST match the ladder run's scope to be fair.
#    Requires serving the MEM1 checkpoint (vLLM) on --base-url; see ladder/baselines/mem1.py.
python -m ladder run-baseline --baseline mem1 --datasets hotpotqa --n-objectives 16 \
  --retrieval-scope corpus --corpus-source union \
  --base-url http://localhost:8000/v1 --limit 200 \
  --out results/mem1_union.jsonl

# 6) Aggregate -- decomposition table, % of Oracle, gaps, recall column.
#    Pass several comma-separated files to overlay a measured baseline inline.
python -m ladder aggregate --results results/pool_multiobj.jsonl
python -m ladder aggregate --results results/union_run.jsonl,results/mem1_union.jsonl
python -m ladder aggregate --results results/pool_multiobj.jsonl \
  --baselines configs/published_baselines.json

# Local serving on macOS (no CUDA): serve Qwen via MLX/Ollama/LM Studio and set
#   BASE_URL below to that endpoint (e.g. http://localhost:8000/v1). vLLM is
#   CUDA-only; see the README "Backbone fidelity" note for the caveats.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

# ============================================================================
# CONFIG  ---  edit this block, then run `python run_experiments.py`
# ============================================================================

# Which action to run: "prepare-data" | "run" | "run-baseline" | "aggregate".
MODE = "run"

# --- Data -------------------------------------------------------------------
DATASETS = ["hotpotqa"]          # subset of: hotpotqa, 2wiki, musique
SPLITS = ["test"]                # test = validation split; dev = slice of train
N_OBJECTIVES = [1]               # MEM1-style multi-objective packing (e.g. [2,8,16,32])
LIMIT = 20                       # cap on #tasks (None = all). Counts TASKS, not sub-questions.
SEED = 0
CACHE_DIR = "data"               # local JSONL cache (download once, reuse forever)

# --- Retrieval tier (see the three-tiers note at the top) -------------------
RETRIEVAL = "bm25"               # ranking backend: "bm25" (default) | "e5" (dense)
RETRIEVAL_SCOPE = "pool"         # "pool" | "corpus"
CORPUS_SOURCE = "union"          # if scope=corpus: "union" (local) | "kilt" (20GB)
CORPUS_INDEX_DIR = None          # if source=kilt: path to prebuilt FAISS index dir

# --- Ladder run knobs (MODE="run") ------------------------------------------
POLICIES = ["H0", "H1", "H2", "H3", "Oracle"]
BUDGETS = [512, 256]             # token budgets; must bind or rungs coincide (see README)
MAX_STEPS = 8                    # max ReAct steps per example
TOPK = 3                         # passages returned per search
PROMPT_VARIANT = "v0"            # "v0" | "v1" | "v2" (for prompt-sensitivity checks)
SUMMARY_MAX_WORDS = 40           # H3's summary length (the single dev-set knob)
RUN_OUT = "results/pool_pilot.jsonl"

# --- Backbone / endpoint (the frozen LLM every rung shares) -----------------
# NOTE: MODEL must equal the tokenizer's BACKBONE_MODEL (single-backbone study).
#       Change it in ladder/tokenizer.py if you truly need a different backbone.
MODEL = None                     # None -> use tokenizer.BACKBONE_MODEL
BASE_URL = "https://api.deepinfra.com/v1/openai"   # or a local OpenAI-compatible endpoint
API_KEY_ENV = "DEEPINFRA_API_KEY"                  # env var holding the key (.env is auto-loaded)
TEMPERATURE = 0.0                # 0 = deterministic (the fairness premise)
MAX_TOKENS = 512

# --- Baseline run knobs (MODE="run-baseline") -------------------------------
BASELINE = "mem1"
BASELINE_MODEL = None            # None -> the baseline's own checkpoint (e.g. MEM1 release)
BASELINE_BASE_URL = "http://localhost:8000/v1"     # where the checkpoint is served (vLLM)
MAX_ITERATIONS = 6               # MEM1 constant-memory loop length
BASELINE_OUT = "results/mem1_baseline.jsonl"

# --- Aggregate knobs (MODE="aggregate") -------------------------------------
AGG_RESULTS = "results/pool_pilot.jsonl"   # one path, or comma-separated to overlay
AGG_METRIC = "both"              # "both" | a specific key (see ladder/aggregate.py)
AGG_BASELINES = None             # e.g. "configs/published_baselines.json" for the overlay

# ============================================================================
# END CONFIG  ---  logic below; you normally do not need to edit past here.
# ============================================================================


def _backend(model, base_url, api_key_env, temperature, max_tokens=MAX_TOKENS):
    from ladder.llm import LLMBackend
    from ladder.tokenizer import BACKBONE_MODEL

    resolved = model or BACKBONE_MODEL
    if resolved != BACKBONE_MODEL:
        # Budgets are always counted with BACKBONE_MODEL's tokenizer; a mismatched
        # serving model would make "budget=X tokens" mean something different.
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

    _load_dotenv()  # populate API keys from a .env at repo root (does not overwrite)

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
            retrieval=RETRIEVAL, retrieval_scope=RETRIEVAL_SCOPE,
            corpus_source=CORPUS_SOURCE, corpus_index_dir=CORPUS_INDEX_DIR,
            cache_dir=CACHE_DIR,
        )
        return

    if MODE == "run-baseline":
        from ladder.runner import run_baseline_grid

        if BASELINE == "mem1":
            from ladder.baselines.mem1 import MEM1Baseline, MEM1_CHECKPOINT

            backend = _backend(
                BASELINE_MODEL or MEM1_CHECKPOINT, BASELINE_BASE_URL,
                API_KEY_ENV, TEMPERATURE,
            )
            baseline = MEM1Baseline(
                backend=backend, retrieval=RETRIEVAL, topk=TOPK,
                max_iterations=MAX_ITERATIONS, retrieval_scope=RETRIEVAL_SCOPE,
            )
        else:
            raise SystemExit(f"unknown BASELINE {BASELINE!r}")

        run_baseline_grid(
            baseline, datasets=DATASETS, splits=SPLITS,
            n_objectives_list=N_OBJECTIVES, out_path=BASELINE_OUT,
            limit=LIMIT, seed=SEED,
            corpus_source=CORPUS_SOURCE, corpus_index_dir=CORPUS_INDEX_DIR,
            cache_dir=CACHE_DIR,
        )
        return

    if MODE == "aggregate":
        from ladder.aggregate import report

        print(report(AGG_RESULTS, metric=AGG_METRIC, baselines_path=AGG_BASELINES))
        return

    raise SystemExit(
        f"unknown MODE {MODE!r}; choose 'prepare-data', 'run', 'run-baseline', 'aggregate'."
    )


if __name__ == "__main__":
    raise SystemExit(main())
