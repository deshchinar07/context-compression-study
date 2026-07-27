# Heuristic Ladder

A controlled harness for decomposing *where* the gains of learned context compression come from. It runs a five-rung **heuristic ladder** on a single frozen backbone with a fixed retrieval and action space, so the only variable across rungs is the compression policy. The central question: how much of the benefit that RL-trained folding methods (MEM1, BACM-RL/FoldAct, PEEK) attribute to *learning* is already recoverable by training-free heuristics, once budget-awareness and selection are isolated?

## The ladder

Each rung adds exactly one design decision, so every adjacent gap measures one thing:

| transition | what it isolates |
|---|---|
| H0 → H1 | **Timing** — does acting before the budget overflows help at all? |
| H1 → H2 | **Selection** — does *what* you drop matter, holding timing fixed? |
| H2 → H3 | **Selection++** — do anchoring plus summarize-instead-of-delete add more? |
| H3 → Oracle | **Ceiling** — how much headroom is left to perfect selection? |

Learned methods are not a rung. They never decomposed timing from selection, so where they land relative to H3 and Oracle is the *finding*, not an input. This harness does not execute their code; comparing against their published numbers is a separate, confounded step (different retrieval setup and often a different, RL-trained backbone) and is left to future work. The internal rung-to-rung gaps need no such caveat — they share one retriever, one backbone, and one prompt.

## Fairness invariants

These are enforced in code, not merely intended:

- **One backbone, frozen, temperature 0.** Every rung calls the same `Qwen/Qwen2.5-7B-Instruct` with identical decoding, so a given context yields a deterministic continuation. (`ladder/llm.py`)
- **Identical everything-except-the-policy.** Same retriever (BM25 over the example's own paragraph pool), same action space (a think / search / answer ReAct loop with results returned inside an information block, per Search-R1), same prompt. (`ladder/agent.py`)
- **Only the Oracle may read gold labels.** `Block.is_supporting` is read exclusively by `Oracle`; every other rung is blind to it. The Oracle is built directly from the datasets' supporting-fact annotations, never hand-judged. (`ladder/policies.py`)
- **Token-accurate budgets.** Budgets are measured in real Qwen tokens with the backbone's own tokenizer; the same counter is used for all rungs. (`ladder/tokenizer.py`)
- **Two scoring protocols, reported and never conflated.** `mem1_table` is the headline protocol — it matches MEM1's `eval.py` exactly (MEM1 preprocessing, set-based token F1, strict semicolon splitting, and zero for the whole example when the answer count is wrong). `standard_qa` is the SQuAD/HotpotQA-style diagnostic (Counter-based token F1 with forgiving pad/truncate behavior), which separates reasoning quality from formatting failures. (`ladder/report.py`)
- **No per-dataset tuning.** The only dev-set knob is H3's summary length. The evaluation grid should be fixed before inspecting test-split results.

One design choice, applied identically to all rungs: **only the evidence store (observation and summary blocks) is compressible.** The question is structural and the reasoning trace is transient, regenerated each turn from the evidence, so budget pressure comes purely from accumulated evidence — exactly what the compression literature targets. (`ladder/policies.py`)

## Installation

```bash
pip install -r requirements.txt
```

An OpenAI-compatible endpoint serving the backbone is also required. The default is DeepInfra; put your key in a `.env` at the repository root:

```
DEEPINFRA_API_KEY="your_key_here"
```

To serve the backbone locally instead, run it with vLLM (`vllm serve Qwen/Qwen2.5-7B-Instruct`) and pass `--base-url http://localhost:8000/v1`. The code path is identical.

### Backbone fidelity

**What we did.** All reported results were produced against DeepInfra's hosted copy of `Qwen/Qwen2.5-7B-Instruct` at temperature 0. Because the weights are served by a third party, their precision is not visible to us, and the deployment may change over time, so a run is not bit-reproducible and cannot be certified as the exact reference weights prior work used.

This does not affect the harness's main claim. Every rung in a given run hits the same endpoint with the same decoding settings, so the rung-to-rung gaps — the quantity this study measures — are internally valid regardless of which copy of the model answers. What it does affect is any number placed directly beside a published table, which already carries a separate confound from the differing retrieval setup.

**Future work.** Re-run the final grid against a locally served, full-precision Qwen on a single CUDA GPU via vLLM. That fixes the weights, precision, and decoding under our own control and makes runs reproducible, so absolute scores become directly quotable rather than only internally comparable. It requires no code change — only a different `--base-url`.

## Data

All three datasets ship supporting-fact or supporting-passage labels, so the Oracle is constructible directly. They are pulled from HuggingFace once and cached as normalized JSONL under `data/`:

| name | HuggingFace id | labels used |
|---|---|---|
| `hotpotqa` | `hotpotqa/hotpot_qa` (config `distractor`) | `supporting_facts` |
| `2wiki` | `scholarly-shadows-syndicate/2wikimultihopqa` | `supporting_facts` |
| `musique` | `dgslibisey/MuSiQue` | `paragraphs.is_supporting` |

```bash
python -m ladder prepare-data --datasets hotpotqa,2wiki,musique --splits test
```

Official test splits have no public labels, so **`test` maps to the validation split** and **`dev` maps to a slice of train** (used only for tuning). This is handled by `SPLIT_ALIASES` in `ladder/data.py`. After the JSONL exists, data loads are fully offline; only LLM API calls need the network.

## Running experiments

```bash
# a quick pilot: 20 HotpotQA examples, all rungs, two binding budgets
python -m ladder run \
  --datasets hotpotqa --splits test --n-objectives 1 \
  --budgets 512,256 --policies H0,H1,H2,H3,Oracle \
  --limit 20 --out results/pilot.jsonl

# the MEM1-comparable multi-objective sweep (the core figure)
python -m ladder run \
  --datasets hotpotqa,2wiki --splits test --n-objectives 2,8,16,32 \
  --budgets 16000,8000,4000 --policies H0,H1,H2,H3,Oracle \
  --limit 200 --out results/multiobj.jsonl
```

## Aggregation

```bash
# default: report both protocols, headline first
python -m ladder aggregate --results results/pilot.jsonl

# print a single protocol
python -m ladder aggregate --results results/pilot.jsonl --metric mem1_table_mean_f1
python -m ladder aggregate --results results/pilot.jsonl --metric standard_qa_mean_f1
```

For each `(dataset, split, n_objectives, budget)` group, the report prints every rung's score, its **% of the Oracle ceiling**, peak context tokens, mean inference tokens (for cost normalization), compression counts, the **fraction of supporting evidence kept**, and the four decomposition gaps. Result JSONL rows store both protocols under explicit `mem1_table_*` and `standard_qa_*` keys; never compare a `standard_qa_*` number directly against a published MEM1 table.

The `both` headline uses **`mem1_table_summed_f1`** — F1 summed over objectives then meaned over examples — because that is what MEM1's own `eval.py` reports, so it is the only key directly comparable to their tables. `mem1_table_mean_f1` divides by `n_objectives` (a per-question rate) and is not what MEM1 prints.

## Retrieval backend

`--retrieval` selects the ranking backend, shared by the retriever and the H2/H3 selection scorer:

- `bm25` (default) — sparse, dependency-free.
- `e5` — uses `intfloat/e5-base-v2`, the dense retriever Search-R1 uses by default; requires `pip install sentence-transformers`. Aligns the ranking algorithm with that setup.

Keep the backend identical across any runs you compare.

## Budget guidance

- **Single-objective** HotpotQA/2Wiki have ~10 short paragraphs (~1–2k tokens); use small budgets (**256–1024**) or the rungs coincide because no compression fires — itself a valid finding.
- **Multi-objective** unions many pools; the **16k → 8k → 4k** range binds here, and the timing/selection gaps open up.

## Cost normalization

Every run records measured inference tokens (`prompt_tokens + completion_tokens`) and LLM call counts. Heuristics carry zero training cost. To reproduce the cost-normalized plot, overlay a learned method's published training compute amortized over N deployment queries against these measured inference costs. H3's summarizer calls appear honestly in the inference token totals.

## Comparison to learned methods

The headline hypothesis — *do training-free heuristics recover most of what RL-trained folding buys?* — is ultimately a comparison against learned methods. Two facts keep it honest:

1. **Internal rung-to-rung gaps need no caveat.** H0→H1→H2→H3→Oracle share one retriever, backbone, and prompt, so their relative differences are clean regardless of how this setup compares to any paper.
2. **Any number placed next to a published table is confounded.** Those numbers came from whole-Wikipedia dense retrieval and often a different, RL-trained backbone; ours come from a small bundled gold-plus-distractor pool on a frozen Qwen2.5-7B-Instruct. A raw win/loss is therefore not yet evidence about policy quality.

The clean fix — re-running the learned checkpoint *inside this harness* against the same retriever, so the confound applies equally to both sides — is future work and is not wired into this version. Until then, treat the rung decomposition as the contribution and any cross-paper number as a caveated, separately sourced overlay.

## Repository layout

```
ladder/
  __init__.py            package init; disables tokenizers parallelism
  blocks.py              Block + Context: the interaction history and token-budgeted window
  tokenizer.py           token counting with the frozen Qwen2.5 tokenizer (BACKBONE_MODEL)
  data.py               dataset loaders (HotpotQA, 2Wiki, MuSiQue) + multi-objective construction
  retrieval_scoring.py   BM25 + E5 retrievers and selection scorers (make_retriever / make_scorer)
  policies.py            H0–H3 + Oracle policies and the H3 query-focused summarizer
  llm.py                 frozen-backbone LLM client and the ReAct prompt template
  agent.py               the fixed ReAct environment shared by all rungs
  report.py              mem1_table + standard_qa scoring, aggregation, decomposition report
  __main__.py            CLI (prepare-data / run / aggregate) and the experiment grid runner
tests/
  test_policies.py       offline behavioural tests for the compression rungs
  test_metrics.py        offline tests for the scoring protocols
scripts/
  plot_ladder_results.py plots: F1 by rung, ladder curves, decomposition gaps
  compare_e5_bm25.py      compares E5 vs. BM25 backends on matched examples
requirements.txt
pyproject.toml           ruff configuration
```

The top-level directories `Search-R1-main/`, `MEM1-main/`, `FoldAct-main/`, and `peek-main/` are vendored copies of the reference systems, kept for cross-comparison only; they are not part of the harness and are not required to run it.

## Tests

```bash
python tests/test_policies.py
python tests/test_metrics.py
```

Both are offline (no network, no API). They assert the exact behavioural difference between rungs — e.g. that H0 can discard the question, H1 pins it, H2 drops least-relevant (not oldest), H3 summarizes and anchors recency, and the Oracle sheds distractors before gold evidence.

## Interpretation

- **If a training-free rung (H3) recovers most of the learned gain** → the field's RL machinery buys little once you control for budget-awareness; future work should target the specific sub-decision where H3 still trails the Oracle.
- **If learned methods clearly beat H3 and approach the Oracle** → learning earns its keep on the *selection* decision, and the H3→Oracle headroom is the roadmap.

Either way, the decomposition plus the oracle ceiling plus the cost framework is the reusable contribution.
