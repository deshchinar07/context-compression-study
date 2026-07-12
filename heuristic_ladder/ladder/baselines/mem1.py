"""MEM1 (RL) baseline, run inside our harness against our retriever.

This faithfully reproduces MEM1's *inference* procedure -- the constant-memory
loop from ``MEM1-main/Mem1/inference/data_pipelines.py`` (``Mem1Pipeline`` with
``inference_type='mem1'``) and its prompt assembly from
``.../inference/models.py`` (``VLLMOpenAIClient.make_completion``) -- with exactly
one thing swapped: the search tool calls **our** per-example retriever
(``ladder.retriever`` / ``ladder.dense``) instead of MEM1's full-Wikipedia FAISS
server. Everything else (the task template, the <think> internal-state memory, the
per-turn context reset, the turn hints, top-k, the stop tokens) is preserved.

Why swap only retrieval: it puts MEM1 and the heuristic ladder under the *same*
retrieval condition, which is the only way a win/loss between them reflects the
compression *policy* rather than a difference in search difficulty (see the
retrieval-scope caveat in configs/published_baselines.json).

MEM1 is NOT a compression rung: it does memory consolidation *inside the model*
(rewriting a compact <think> state each turn), so there is no external policy and
no token budget -- its memory is structurally constant. It therefore has its own
runner (``run``) that emits the same ``RunResult`` schema as the ladder so
``metrics``/``aggregate`` treat it uniformly.

Serving the checkpoint (required to actually run this):

    vllm serve Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release --port 8000
    # then point an LLMBackend at it:
    #   LLMBackend(model="Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release",
    #              base_url="http://localhost:8000/v1", api_key_env=...)

The backend must expose an OpenAI-compatible ``/v1/completions`` (raw) endpoint,
because MEM1 continues its own assistant turn; vLLM does. Hosted chat-only
endpoints will not work for the MEM1 continuation trick.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import List, Optional

from .. import tokenizer
from ..agent import RunResult
from ..data import Example
from ..llm import LLMBackend
from ..retriever import make_retriever

# The released MEM1 RL checkpoint (see repo README / HF).
MEM1_CHECKPOINT = "Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release"
# MEM1 assembles prompts with the base Qwen2.5-7B chat template (models.py).
MEM1_CHAT_TEMPLATE_MODEL = "Qwen/Qwen2.5-7B"

# Verbatim from MEM1-main/.../gen_data/data_process/qa_search_test_merge_multi.py
# (``make_prefix``, template_type='base'). {questions} is filled with our Example's
# sub-questions joined by "; " -- exactly how MEM1 composes a multi-objective task.
MEM1_TASK_TEMPLATE = """You will answer multiple complex questions using iterative reasoning, summarization, and web search.

At each step, you will see the questions, a cumulative summary of relevant information, the current search query, and search results (except in the first step, where only the questions are provided). Your task is to:

1. Perform reasoning and update a cumulative, concise summary within <think> ... </think>. This acts as persistent memory and must include all essential information from previous <think> and <information> tags.

2. Then choose one of the following actions:
   - If any question remains unanswered, issue a single query for one question inside <search> ... </search>. The query should consist of keywords or a short phrase. Only search one question at a time.
   - If all questions are answered, provide the final answers—separated by semicolons—within <answer> answer1; answer2; ... </answer>. The answers must be concise, contain only essential words, and avoid any explanations.

Important:
- Always follow this structure after <information> or the initial questions: <think> ... </think><search> ... </search> or <think> ... </think><answer> ... </answer>.
- Do not search multiple queries or questions simultaneously.

Answer the following questions: {questions}
"""

_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@functools.lru_cache(maxsize=2)
def _chat_tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def _close_open_tag(text: str) -> str:
    """Re-append the stop tag the server consumed (mirrors models.py behavior).

    The /v1/completions endpoint does not echo the matched stop string, so if the
    model opened <search>/<answer> without its closer, we add it back so the action
    parser sees a well-formed tag -- identical to MEM1's own post-processing.
    """
    t = text
    if "<search>" in t and "</search>" not in t:
        t += "</search>"
    if "<answer>" in t and "</answer>" not in t:
        t += "</answer>"
    return t


def _passages2string(paragraphs) -> str:
    """Format retrieved paragraphs the way MEM1's batch_search does."""
    out = ""
    for i, p in enumerate(paragraphs):
        out += f"Doc {i + 1}(Title: {p.title}) {p.text}\n"
    return out


@dataclass
class MEM1Baseline:
    backend: LLMBackend
    retrieval: str = "bm25"
    topk: int = 3
    max_iterations: int = 6
    max_tokens: int = 1024
    temperature: float = 0.0  # MEM1 used 0.01; 0.0 keeps our runs reproducible
    chat_template_model: str = MEM1_CHAT_TEMPLATE_MODEL
    name: str = "MEM1"
    # Retrieval scope, matched to the ladder run it is compared against. 'pool':
    # the example's bundled pool (dataset labels). 'corpus': the shared prebuilt
    # index (``corpus_index``), gold assigned by title match. Keep this identical
    # to the ladder run or the comparison reintroduces a retrieval confound.
    retrieval_scope: str = "pool"
    corpus_index: object = None

    def _build_prompt(self, initial_prompt: str, cur_obs: str) -> str:
        """[user: task] + [assistant: cur_obs], chat-templated, trailing im_end stripped
        so the model *continues* the assistant turn (models.py make_completion)."""
        tok = _chat_tokenizer(self.chat_template_model)
        messages = [
            {"role": "user", "content": initial_prompt},
            {"role": "assistant", "content": cur_obs},
        ]
        text = tok.apply_chat_template(messages, tokenize=False)
        end = "<|im_end|>\n"
        if text.endswith(end):
            text = text[: -len(end)]
        return text

    def run(self, example: Example, budget: int = -1, **_ignore) -> RunResult:
        initial_prompt = MEM1_TASK_TEMPLATE.format(
            questions="; ".join(example.questions)
        )
        gold_titles = example.supporting_titles
        if (self.retrieval_scope or "pool").lower() == "corpus":
            if self.corpus_index is None:
                raise RuntimeError(
                    "MEM1Baseline retrieval_scope='corpus' requires a prebuilt "
                    "corpus_index (build it once with ladder.corpus.build_corpus)."
                )
            retriever = self.corpus_index
        else:
            retriever = make_retriever(self.retrieval, example.paragraphs)

        u0 = self.backend.usage
        base_calls, base_pt, base_ct = u0.n_calls, u0.prompt_tokens, u0.completion_tokens

        cur_obs = ""  # MEM1 keeps only the latest (response + information) as memory
        retrieved_idx: set = set()
        retrieved_gold_titles: set = set()
        prediction = ""
        answered = False
        n_searches = 0
        peak = 0
        last_prompt_tokens = 0
        step = 0

        for step in range(self.max_iterations):
            is_last = step == self.max_iterations - 1
            prompt_message = self._build_prompt(initial_prompt, cur_obs)
            last_prompt_tokens = tokenizer.count_tokens(prompt_message)
            peak = max(peak, last_prompt_tokens)

            stop = ["</answer>"] if is_last else ["</search>", "</answer>"]
            out = self.backend.complete_raw(
                prompt_message, stop=stop,
                max_tokens=self.max_tokens, temperature=self.temperature,
            )
            cur_response = _close_open_tag(out.strip())

            ans = _ANSWER_RE.search(cur_response)
            srch = _SEARCH_RE.search(cur_response)
            # act(): answer wins only if it appears; MEM1 checks search first.
            if srch and not is_last:
                n_searches += 1
                query = srch.group(1).strip()
                hits = retriever.search(query, topk=self.topk, exclude_idx=retrieved_idx)
                for p in hits:
                    retrieved_idx.add(p.idx)
                    if p.title in gold_titles:
                        retrieved_gold_titles.add(p.title)
                turns_left = self.max_iterations - step - 1
                if turns_left > 1:
                    hint = f"[HINT]You have {turns_left} turns left.[/HINT]"
                else:
                    hint = (
                        f"[HINT]You have {turns_left} turn left. "
                        "You must answer the question now.[/HINT]"
                    )
                info = f"<information>\n{hint}\n{_passages2string(hits)}\n</information>"
                # constant memory: drop old obs, keep only this turn's response + info
                cur_obs = cur_response + info
                continue
            if ans:
                prediction = ans.group(1).strip()
                answered = True
                break
            # No parseable action -> stop (MEM1 returns None here).
            break

        u1 = self.backend.usage
        # Title-based accounting works under both scopes (corpus passages carry no
        # dataset label; the pool's supporting titles are the gold titles anyway).
        supporting_total = len(gold_titles)
        supporting_kept = len(retrieved_gold_titles)

        return RunResult(
            example_id=example.id,
            dataset=example.dataset,
            policy=self.name,
            budget=budget,
            n_objectives=example.n_objectives,
            prediction=prediction,
            n_steps=step + 1,
            n_searches=n_searches,
            n_stale_searches=0,
            peak_context_tokens=peak,
            final_context_tokens=last_prompt_tokens,
            answered=answered,
            compress_triggered=0,
            compress_dropped=0,
            compress_summarized=0,
            llm_calls=u1.n_calls - base_calls,
            prompt_tokens=u1.prompt_tokens - base_pt,
            completion_tokens=u1.completion_tokens - base_ct,
            final_supporting_kept=supporting_kept,
            final_supporting_total=supporting_total,
            retrieval_scope=(self.retrieval_scope or "pool").lower(),
            gold_titles_total=len(gold_titles),
            gold_titles_retrieved=len(retrieved_gold_titles),
        )
