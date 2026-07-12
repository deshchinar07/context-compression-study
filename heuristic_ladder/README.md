# Heuristic Ladder: decomposing where learned context-compression gains come from

A controlled harness that answers **"how much of learned context-compression's
benefit comes from *learning*, and where specifically?"** It runs a five-rung
*heuristic ladder* on a single frozen backbone with a fixed retrieval/action
space, so the **only** variable across rungs is the compression policy.

```
H0  -> H1  : does being proactive about the budget help at all?   (TIMING)
H1  -> H2  : does WHAT you drop matter, holding timing fixed?      (SELECTION)
H2  -> H3  : do anchors + summarize-instead-of-delete add more?    (SELECTION++)
H3  -> Oracle : how much headroom is left to perfect selection?    (CEILING)
```

Learned methods (MEM1, BACM-RL, ...) are **not** a rung — they never decomposed
timing vs. selection, so where they land relative to H3 and Oracle is the
*finding*, overlaid at aggregation time from their **published** numbers (this
harness never runs their code). That overlay is comparative by construction, so
it is caveated by construction — see *Comparing to published baselines* below.

## Why you can trust the numbers (fairness invariants)

These are enforced in code, not just intended:

- **One backbone, frozen, temperature 0.** Every rung calls the same
  `Qwen/Qwen2.5-7B-Instruct` with identical decoding. A given context yields a
  deterministic continuation. (`ladder/llm.py`)
- **Identical everything-except-the-policy.** Same retriever (BM25 over the
  example's own paragraph pool), same action space (`<think>/<search>/<answer>`,
  results in `<information>`, per Search-R1), same prompt. (`ladder/agent.py`)
- **Only the Oracle may read gold labels.** `Block.is_supporting` is read
  exclusively by `Oracle`; every other rung asserts nothing about it. The Oracle
  is built directly from the datasets' supporting-fact annotations — never
  hand-judged. (`ladder/policies.py`)
- **Token-accurate budgets with the backbone's own tokenizer.** Budgets are in
  real Qwen tokens; the same counter is used for all rungs. (`ladder/tokenizer.py`)
- **Two scoring protocols are reported and never conflated.** `mem1_table` is
  the primary/headline protocol for comparisons against MEM1: MEM1
  preprocessing, set-based token F1, strict semicolon splitting, and zero for
  the whole example when the answer count is wrong. `standard_qa` is the
  secondary SQuAD/HotpotQA diagnostic: Counter-based token F1 with forgiving
  pad/truncate behavior, which separates reasoning quality from formatting
  failures. (`ladder/metrics.py`)
- **No per-dataset tuning.** The single dev-set knob is H3's summary length.
  Pre-register the grid in `configs/preregistration.yaml` before looking at test.

One deliberate design choice, applied identically to all rungs and documented in
`ladder/policies.py`: **only the evidence store (observation/summary blocks) is
compressible.** The question is structural; the reasoning trace is transient and
regenerated each turn from the evidence, so budget pressure comes purely from
accumulated evidence — which is exactly what the compression literature targets.

## Install

```bash
cd heuristic_ladder
pip install -r requirements.txt
```

You also need an OpenAI-compatible endpoint serving the backbone. The default is
DeepInfra; put your key in a `.env` at the repo root:

```
DEEPINFRA_API_KEY="your_key_here"
```

(To run a **local** frozen Qwen instead — the stronger, no-caveat tier — serve it
with vLLM: `vllm serve Qwen/Qwen2.5-7B-Instruct`, then pass
`--base-url http://localhost:8000/v1`. Same code path.)

### Backbone fidelity: why "no-caveat", and the macOS catch

The two tiers above are not just a convenience choice — they differ in how much
you can trust the backbone, which is the fairness premise of the whole harness:

- **Hosted (DeepInfra) = the caveated tier.** You are trusting a provider's copy
  of `Qwen/Qwen2.5-7B-Instruct`. It may be **quantized**, it may be **updated
  under you** over time (so runs are not bit-reproducible), and it **cannot be
  verified** to be the exact frozen reference weights that Search-R1 / MEM1 /
  BACM-RL ran against. The internal rung-to-rung gaps (H0→H1→H2→H3→Oracle) stay
  clean regardless — they share one backend — but any number you place next to a
  *published* table inherits this backbone-fidelity caveat on top of the
  retrieval-scope one described in *Comparing to published baselines*.
- **Local, full-precision (vLLM/TGI) = the no-caveat tier.** You control the exact
  weights, precision, and decoding, and the run is reproducible. This is the only
  tier where "it is literally the same frozen backbone" is a claim you can stand
  behind.

**The macOS catch:** vLLM (and HF TGI) are **CUDA-only**, so the no-caveat tier is
**not reachable on an Apple-Silicon Mac**. Your local Mac options — MLX
(`mlx_lm.server`), Ollama, LM Studio — all speak the same OpenAI-compatible API
and work fine via `--base-url`, but they serve **quantized** weights by default,
which *reintroduces* a (smaller) fidelity caveat. So on a Mac you get "local and
convenient", not "no-caveat". Notes if you go this route:

- Set a dummy key in `.env` (`DEEPINFRA_API_KEY="local"`) — a key is required even
  locally.
- The client always sends the model id `Qwen/Qwen2.5-7B-Instruct`, so alias your
  local model to that exact name (e.g. `ollama cp qwen2.5:7b-instruct Qwen/Qwen2.5-7B-Instruct`).
- The Qwen **tokenizer** is still fetched from HuggingFace for budget accounting
  regardless of who serves the model.

**Practical recommendation.** Use hosted DeepInfra while learning the harness and
iterating (a 7B at temperature 0 is cents per pilot); reserve a **rented single
CUDA GPU running vLLM** for the final, pristine numbers — it is a one-line
`--base-url` change from everything else.

## Data — download before running

All three datasets ship supporting-fact / supporting-passage labels (so the
Oracle is constructible directly). They are pulled from HuggingFace once and
cached as normalized JSONL under `data/`:

| name       | HuggingFace id                               | labels used            |
|------------|----------------------------------------------|------------------------|
| `hotpotqa` | `hotpotqa/hotpot_qa` (config `distractor`)   | `supporting_facts`     |
| `2wiki`    | `scholarly-shadows-syndicate/2wikimultihopqa`| `supporting_facts`     |
| `musique`  | `dgslibisey/MuSiQue`                          | `paragraphs.is_supporting` |

```bash
# one-time download + cache (needs network; ~seconds each)
python -m ladder prepare-data --datasets hotpotqa,2wiki,musique --splits test
```

Splits: the official test splits have no public labels, so **`test` = the
validation split** and **`dev` = a slice of train** (used only for tuning). This
is handled by `SPLIT_ALIASES` in `ladder/data.py`.

After the JSONL exists, loads are fully offline for data. The backbone's
tokenizer is also fetched once from HuggingFace and cached; after that only
LLM API calls need the network.

## Run

```bash
# a quick pilot: 20 HotpotQA examples, all rungs, two binding budgets
python -m ladder run \
  --datasets hotpotqa --splits test --n-objectives 1 \
  --budgets 512,256 --policies H0,H1,H2,H3,Oracle \
  --limit 20 --out results/pilot.jsonl

# the MEM1-comparable multi-objective sweep (the plan's core figure)
python -m ladder run \
  --datasets hotpotqa,2wiki --splits test --n-objectives 2,8,16,32 \
  --budgets 16000,8000,4000 --policies H0,H1,H2,H3,Oracle \
  --limit 200 --out results/multiobj.jsonl
```

## Aggregate

```bash
# Default: report both, with the MEM1-comparable headline first:
python -m ladder aggregate --results results/pilot.jsonl

# Optional: print only one protocol:
python -m ladder aggregate --results results/pilot.jsonl \
  --metric mem1_table_mean_f1
python -m ladder aggregate --results results/pilot.jsonl \
  --metric standard_qa_mean_f1
```

Prints, per `(dataset, split, n_objectives, budget)`: each rung's score, its
**% of the Oracle ceiling**, peak context tokens, mean inference tokens (for
cost-normalization), compression counts, and the **fraction of supporting
evidence kept** — plus the four decomposition gaps. Result JSONL rows store
both protocols under explicit `mem1_table_*` and `standard_qa_*` keys. Never
compare a `standard_qa_*` number directly against a published MEM1 table.

The `both` headline uses **`mem1_table_summed_f1`** — F1 summed over objectives
then meaned over examples — because that is what MEM1's own `eval.py`
(`compute_score`) reports, so it is the only key directly comparable to their
tables. `mem1_table_mean_f1` divides by `n_objectives` (handy for reading a
per-question rate) and is **not** what MEM1 prints.

## Comparing to published baselines (the overlay)

The plan's headline hypothesis — *do training-free heuristics recover most of
what RL-trained folding buys?* — is inherently a comparison against **other
papers' numbers** (MEM1, BACM-RL/FoldAct, PEEK). Two facts make that comparison
delicate, and the overlay is built to keep you honest about both:

1. **Internal rung-to-rung gaps need no caveat.** `H0→H1→H2→H3→Oracle` all share
   one retriever, backbone, and prompt, so their *relative* differences are
   clean regardless of how our setup compares to any paper.
2. **Any number placed next to a published table is confounded.** Those numbers
   came from full-corpus retrieval (whole-Wikipedia dense index) and often a
   different/RL-trained backbone; ours come from a small bundled gold+distractor
   pool on a frozen Qwen2.5-7B-Instruct. A raw win/loss is therefore *not* yet
   evidence about policy quality. The clean fix (README's no-caveat tier) is to
   re-run the learned checkpoint **inside this harness** against our retriever so
   the confound is applied equally to both sides.

`configs/published_baselines.json` holds the transcribed numbers under a strict
provenance schema: every entry names a source (`arxiv`, `table`, `page`), the
exact metric key it is comparable to, its backbone, and its retrieval corpus.
**Scores start `null`** — you transcribe them from the cited table; nothing is
typed from memory, so no un-sourced number can silently enter a figure. Pass the
file to `aggregate` to overlay matching baselines per group and auto-print the
retrieval-scope / backbone caveats:

```bash
python -m ladder aggregate --results results/multiobj.jsonl \
  --baselines configs/published_baselines.json
```

Entries still `null` render as `score=TODO (transcribe from <arxiv> <table>)`;
verified entries render their number. Baselines are pinned to their comparable
metric, so a MEM1 number never prints under the `standard_qa` diagnostic.

### The no-caveat tier: run the learned baseline in-harness

The clean way to *remove* (not just disclose) the retrieval/backbone confound is
to run the learned checkpoint inside this harness against our own retriever.
`ladder/baselines/mem1.py` does exactly that for MEM1: it reproduces MEM1's
constant-memory inference loop and task prompt verbatim, swapping only the search
tool to call our per-example retriever.

```bash
# 1) serve the released MEM1 checkpoint with a /v1/completions endpoint (vLLM):
vllm serve Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release --port 8000

# 2) run it in-harness over the same examples (match --retrieval to your ladder run):
python -m ladder run-baseline --baseline mem1 \
  --datasets hotpotqa --splits test --n-objectives 16 \
  --base-url http://localhost:8000/v1 --retrieval bm25 \
  --limit 200 --out results/mem1_inharness.jsonl

# 3) aggregate ladder + measured baseline together (comma-separated):
python -m ladder aggregate \
  --results results/multiobj.jsonl,results/mem1_inharness.jsonl \
  --baselines configs/published_baselines.json
```

The measured MEM1 row prints inline as `measured (in-harness, no caveat)` with its
% of the Oracle ceiling — a legitimate apples-to-apples comparison, because both
sides now face the identical retriever. MEM1 has no token budget (its memory is
structurally constant), so it is shown once per task against every budget group.

### Retrieval backend (`--retrieval bm25|e5`)

Both `run` and `run-baseline` take `--retrieval`. `bm25` (default) is sparse and
dependency-free. `e5` uses `intfloat/e5-base-v2` — the dense retriever Search-R1
and MEM1 use by default — for the retriever *and* the H2/H3 selection scorer, so
you can align the ranking algorithm with those papers (`pip install
sentence-transformers`). Whatever you pick, keep it identical between a ladder run
and the in-harness baseline you compare it to.

### Retrieval scope (`--retrieval-scope pool|corpus`)

*What the retriever searches* — orthogonal to the backend above. Both `run` and
`run-baseline` take it; keep it identical across the runs you compare.

- **`pool`** (default): retrieve from each example's own ~10-paragraph bundled
  pool, with gold labels straight from the dataset. Retrieval is easy (the gold is
  always present) and the Oracle is **exact**. This is the clean-Oracle primary
  condition and the fast dev loop.
- **`corpus`**: retrieve from **one shared index** built once per (dataset,
  split). Retrieved passages are unlabeled corpus docs, so gold is assigned by
  **title match** against the example's supporting titles at retrieval time.
  Retrieval can now genuinely miss the gold, so the Oracle becomes *perfect
  selection given what was retrieved*, and **gold-retrieval recall** is reported
  as its own column (in `aggregate`) so retrieval error and selection error stay
  separable. Pick the corpus with `--corpus-source`:
  - **`union`** (default, local, no download): the deduplicated union of the
    split's own paragraphs (e.g. ~66k passages for HotpotQA). Big enough for real
    recall failures, small enough to index in memory with BM25 on a laptop. Gold
    titles match exactly (same dataset), so labeling is exact.
  - **`kilt`** (Stage 2, not yet built): the full Wikipedia FAISS/e5 index — the
    literal open-domain "20GB" tier that matches MEM1's setup. Needs a GPU/large-RAM
    box; it drops into the same `CorpusIndex` interface, so nothing else changes.

Suggested progression: develop and anchor the clean-Oracle result on `pool`,
show it survives realistic retrieval locally on `corpus --corpus-source union`,
then (on a GPU box) run `corpus --corpus-source kilt` **plus** MEM1 in-harness on
the same index for the final open-domain comparison. The rung-to-rung
decomposition is clean at every tier; the corpus tiers only add external realism /
comparability.

```bash
# local realism tier (no GPU): same grid, shared union corpus
python -m ladder run --datasets hotpotqa --retrieval-scope corpus --corpus-source union \
  --budgets 512,256 --policies H0,H1,H2,H3,Oracle --limit 20 --out results/union_pilot.jsonl
```

## Budget guidance (so compression actually binds)

- **Single-objective** HotpotQA/2Wiki have ~10 short paragraphs (~1–2k tokens);
  use small budgets (**256–1024**) or the ladder rungs coincide (no compression
  fires — itself a valid finding).
- **Multi-objective** unions many pools; the plan's **16k→8k→4k** BACM-RL range
  binds here. This is where the timing/selection gaps open up.

## Cost normalization

Every run records real, measured inference tokens (`prompt_tokens +
completion_tokens`) and LLM call counts. Heuristics carry **zero** training cost;
to reproduce the plan's cost-normalized plot, overlay a learned method's
published training compute amortized over N deployment queries against these
measured inference costs. H3's summarizer calls show up honestly in `infer_tokens`.

## Repo layout

```
ladder/
  blocks.py       # Block + Context (the c_1..c_K history and budget B)
  tokenizer.py    # token-accurate budgets (fixed Qwen tokenizer; no fallback)
  data.py         # dataset loaders + MEM1 multi-objective construction
  retriever.py    # dependency-free BM25 + make_retriever(bm25|e5) factory
  dense.py        # optional e5 (intfloat/e5-base-v2) dense retriever + scorer
  scoring.py      # training-free lexical relevance + make_scorer(bm25|e5)
  summarizer.py   # query-focused summarizer (H3 only)
  policies.py     # H0, H1, H2, H3, Oracle  <-- the ladder
  prompts.py      # 3 prompt variants for the sensitivity finding
  agent.py        # the fixed ReAct environment shared by all rungs
  metrics.py      # primary MEM1-table + secondary standard-QA scoring
  runner.py       # sweep policies x budgets x datasets x N (+ run_baseline_grid)
  aggregate.py    # decomposition table + gaps + % of oracle + baseline overlays
  baselines/
    mem1.py       # MEM1 (RL) run in-harness against our retriever (no-caveat tier)
  cli.py          # prepare-data / run / run-baseline / aggregate
tests/            # offline behavioural tests for the rungs and metrics
configs/preregistration.yaml       # the pre-registered grid (planning doc)
configs/published_baselines.json   # transcribed MEM1/BACM-RL/PEEK numbers + caveats
```

## Tests

```bash
python tests/test_policies.py
python tests/test_metrics.py
```

These are offline (no network, no API) and assert the *exact* behavioural
difference between rungs — e.g. that H0 can discard the question, H1 pins it,
H2 drops least-relevant (not oldest), H3 summarizes + anchors recency, and the
Oracle sheds distractors before gold evidence.

## What this becomes

- **If a training-free rung (H3) recovers most of the learned gain** → the field's
  RL machinery buys little once you control for budget-awareness; future work
  should target the specific sub-decision where H3 still trails Oracle.
- **If learned methods clearly beat H3 and approach Oracle** → learning earns its
  keep on the *selection* decision, and the H3→Oracle headroom is the roadmap.

Either way the decomposition + oracle + cost framework is the reusable
contribution.
