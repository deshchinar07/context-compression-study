"""The ReAct agent loop.

This is the fixed *environment* shared by every rung of the ladder. Given an
Example, a compression policy, and a token budget, it runs a think/search/answer
loop over the example's own paragraph pool, invoking the policy at the single
decision point where a new observation is about to be appended.

Only the compression policy varies across runs; the backbone, decoding params,
retriever, action space, and prompt are held fixed. That is what makes the
cross-rung comparison legitimate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from . import prompts, tokenizer
from .blocks import Block, Context, OBSERVATION, QUESTION
from .data import Example
from .llm import LLMBackend
from .policies import CompressionPolicy
from .retriever import make_retriever
from .scoring import make_scorer
from .summarizer import Summarizer

_SEARCH_RE = re.compile(r"<search>(.*?)(?:</search>|$)", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def _clean_answer(text: str) -> str:
    """Best-effort answer extraction identical for every rung.

    Prefer an explicit <answer> span; otherwise strip reasoning (<think>) and
    dangling tags and return the remainder. A model that never commits to an
    answer scores as such (matching the Search-R1/MEM1 convention that an
    unanswered task gets no credit) -- but we never let stray reasoning text be
    scored *as if* it were the answer for one rung and not another.
    """
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    cleaned = _THINK_RE.sub("", text)
    for tag in ("<answer>", "</answer>", "<search>", "</search>",
                "<information>", "</information>", "<memory>", "</memory>"):
        cleaned = cleaned.replace(tag, " ")
    return " ".join(cleaned.split()).strip()


@dataclass
class RunResult:
    example_id: str
    dataset: str
    policy: str
    budget: int
    n_objectives: int
    prediction: str
    n_steps: int = 0
    n_searches: int = 0
    n_stale_searches: int = 0
    peak_context_tokens: int = 0
    final_context_tokens: int = 0
    answered: bool = False
    # compression stats
    compress_triggered: int = 0
    compress_dropped: int = 0
    compress_summarized: int = 0
    # inference cost attributable to this run
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # kept for auditing which evidence survived to the final context
    final_supporting_kept: int = 0
    final_supporting_total: int = 0
    # retrieval scope + gold-retrieval recall (separates retrieval error from
    # selection error; under 'corpus' scope the retriever can miss the gold).
    retrieval_scope: str = "pool"
    gold_titles_total: int = 0
    gold_titles_retrieved: int = 0


def _parse_action(text: str):
    """Return ('answer', str) | ('search', str) | (None, None), taking the FIRST
    action tag that appears (the model reasons, then acts once)."""
    s_pos = text.find("<search>")
    a_pos = text.find("<answer>")
    if s_pos == -1 and a_pos == -1:
        return None, None
    if a_pos != -1 and (s_pos == -1 or a_pos < s_pos):
        m = _ANSWER_RE.search(text)
        return "answer", (m.group(1).strip() if m else "")
    m = _SEARCH_RE.search(text)
    return "search", (m.group(1).strip() if m else "")


class ReActAgent:
    def __init__(
        self,
        backend: LLMBackend,
        max_steps: int = 8,
        topk: int = 3,
        prompt_variant: str = "v0",
        retrieval: str = "bm25",
        retrieval_scope: str = "pool",
        corpus_index=None,
    ):
        self.backend = backend
        self.max_steps = max_steps
        self.topk = topk
        self.prompt_variant = prompt_variant
        self.retrieval = retrieval
        # 'pool' (default): per-example bundled pool, dataset gold labels.
        # 'corpus': shared prebuilt index (``corpus_index``), gold assigned by
        # title match at retrieval time. See ladder/corpus.py.
        self.retrieval_scope = (retrieval_scope or "pool").lower()
        self.corpus_index = corpus_index

    def run(
        self,
        example: Example,
        policy: CompressionPolicy,
        budget: int,
        summary_max_words: int = 40,
    ) -> RunResult:
        query = " ; ".join(example.questions)
        instr = prompts.instruction(example.n_objectives, self.prompt_variant, query)

        ctx = Context(budget=budget, blocks=[Block(id=0, role=QUESTION, text=instr, step_idx=0)])
        # Gold titles drive labeling under 'corpus' scope (retrieved passages carry
        # no dataset label) and recall accounting under both scopes.
        gold_titles = example.supporting_titles
        if self.retrieval_scope == "corpus":
            if self.corpus_index is None:
                raise RuntimeError(
                    "retrieval_scope='corpus' requires a prebuilt corpus_index "
                    "(build it once with ladder.corpus.build_corpus and pass it in)."
                )
            retriever = self.corpus_index
            scorer = self.corpus_index.scorer_for(query)
            label_by_title = True
        else:
            retriever = make_retriever(self.retrieval, example.paragraphs)
            scorer = make_scorer(self.retrieval, [p.text for p in example.paragraphs], query)
            label_by_title = False
        summarizer = Summarizer(self.backend, max_words=summary_max_words)

        # snapshot global usage so we can attribute this run's inference cost
        u0 = self.backend.usage
        base_calls, base_pt, base_ct = u0.n_calls, u0.prompt_tokens, u0.completion_tokens

        retrieved_idx: set = set()
        retrieved_gold_titles: set = set()
        peak = ctx.used()
        prediction = ""
        answered = False
        n_searches = 0
        n_stale = 0
        step = 0

        for step in range(1, self.max_steps + 1):
            prompt = ctx.render_prompt() + prompts.CONTINUE_CUE
            out = self.backend.complete(prompt, stop=["</search>", "</answer>", "<information>"])
            kind, payload = _parse_action(out)

            if kind == "answer":
                prediction = payload
                answered = True
                break

            if kind == "search":
                n_searches += 1
                hits = retriever.search(payload, topk=self.topk, exclude_idx=retrieved_idx)
                new_hits = [p for p in hits if p.idx not in retrieved_idx]
                if not new_hits:
                    n_stale += 1
                    if n_stale >= 2:
                        break  # nothing new to find; go answer
                    continue
                for p in new_hits:
                    retrieved_idx.add(p.idx)
                    # Under 'corpus' scope the passage is an unlabeled corpus doc;
                    # its gold status is decided here by title match. Under 'pool'
                    # scope the dataset label is authoritative (identical result,
                    # since a supporting paragraph's title is a gold title).
                    is_supporting = (
                        (p.title in gold_titles) if label_by_title else p.is_supporting
                    )
                    if is_supporting:
                        retrieved_gold_titles.add(p.title)
                    pending = Block(
                        id=ctx.next_id(),
                        role=OBSERVATION,
                        text=p.text,
                        step_idx=step,
                        objective_idx=p.objective_idx,
                        is_supporting=is_supporting,
                        source_title=p.title,
                    )
                    policy.on_append(
                        ctx, pending, scorer=scorer, summarizer=summarizer, query=query
                    )
                    peak = max(peak, ctx.used())
                continue

            # No parseable action: nudge once more, then bail to a forced answer.
            if step >= 2:
                break

        if not answered:
            prompt = ctx.render_prompt() + prompts.FORCE_ANSWER_CUE
            out = self.backend.complete(prompt, stop=["</answer>"], max_tokens=256)
            prediction = _clean_answer(out)

        u1 = self.backend.usage
        # Denominator is the gold-title count so it is meaningful under both scopes
        # (under 'corpus' the bundled pool is not the retrieval space).
        supporting_total = len(gold_titles)
        supporting_kept = sum(
            1 for b in ctx.blocks if b.role == OBSERVATION and b.is_supporting
        )

        return RunResult(
            example_id=example.id,
            dataset=example.dataset,
            policy=policy.name,
            budget=budget,
            n_objectives=example.n_objectives,
            prediction=prediction,
            n_steps=step,
            n_searches=n_searches,
            n_stale_searches=n_stale,
            peak_context_tokens=peak,
            final_context_tokens=ctx.used(),
            answered=answered,
            compress_triggered=policy.stats.triggered,
            compress_dropped=policy.stats.dropped,
            compress_summarized=policy.stats.summarized,
            llm_calls=u1.n_calls - base_calls,
            prompt_tokens=u1.prompt_tokens - base_pt,
            completion_tokens=u1.completion_tokens - base_ct,
            final_supporting_kept=supporting_kept,
            final_supporting_total=supporting_total,
            retrieval_scope=self.retrieval_scope,
            gold_titles_total=len(gold_titles),
            gold_titles_retrieved=len(retrieved_gold_titles),
        )
