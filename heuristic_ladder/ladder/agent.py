from __future__ import annotations

import re
from dataclasses import dataclass

from .blocks import Block, Context, OBSERVATION, QUESTION
from .data import Example
from .llm import (
    LLMBackend,
    COMMIT_CUE,
    CONTINUE_CUE,
    FORCE_ANSWER_CUE,
    FORCE_ANSWER_RETRY_CUE,
    FORCE_ANSWER_SYSTEM,
    instruction,
)
from .policies import Policy, Summarizer
from .retrieval import make_retriever, make_scorer

_SEARCH_RE = re.compile(r"<search>(.*?)(?:</search>|$)", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)


def _clean_answer(text: str) -> str:
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
    compress_triggered: int = 0
    compress_dropped: int = 0
    compress_summarized: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    final_supporting_kept: int = 0
    final_supporting_total: int = 0
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


def _force_answer(backend: LLMBackend, ctx: Context) -> str:
    """Last-resort answer extraction when the ReAct loop never committed.

    Qwen often ignores a soft force cue and burns max_tokens inside <think>.
    Prefilling the assistant turn with ``<answer>`` makes the model continue
    inside the answer span instead of opening a think block.
    """
    out = backend.complete(
        ctx.render_prompt() + FORCE_ANSWER_CUE,
        stop=["</answer>"],
        max_tokens=64,
        system=FORCE_ANSWER_SYSTEM,
        assistant_prefix="<answer>",
    )
    pred = _clean_answer(out)
    if pred:
        return pred
    # Rare: provider rejected prefill or returned empty -- one plain retry.
    out = backend.complete(
        ctx.render_prompt() + FORCE_ANSWER_RETRY_CUE,
        stop=["</answer>"],
        max_tokens=64,
        system=FORCE_ANSWER_SYSTEM,
        assistant_prefix="<answer>",
    )
    return _clean_answer(out)


class ReActAgent:
    def __init__(
        self,
        backend: LLMBackend,
        max_steps: int = 8,
        topk: int = 3,
        prompt_variant: str = "v0",
        retrieval: str = "bm25",
    ):
        self.backend = backend
        self.max_steps = max_steps
        self.topk = topk
        self.prompt_variant = prompt_variant
        self.retrieval = retrieval

    def run(self,example: Example,policy: Policy,budget: int, summary_max_words: int = 40,) -> RunResult:
        query = " ; ".join(example.questions)
        instr = instruction(example.n_objectives, self.prompt_variant, query)

        ctx = Context(budget=budget, blocks=[Block(id=0, role=QUESTION, text=instr, step_idx=0)])
        gold_titles = example.supporting_titles
        retriever = make_retriever(self.retrieval, example.paragraphs)
        scorer = make_scorer(self.retrieval, [p.text for p in example.paragraphs], query)
        summarizer = Summarizer(self.backend, max_words=summary_max_words)

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
        commit_nudge_used = False

        def apply_search(payload: str) -> bool:
            """Handle a <search> action. Return True to continue the loop, False to stop."""
            nonlocal n_searches, n_stale, peak
            n_searches += 1
            hits = retriever.search(payload, topk=self.topk, exclude_idx=retrieved_idx)
            new_hits = [p for p in hits if p.idx not in retrieved_idx]
            if not new_hits:
                n_stale += 1
                return n_stale < 2  # stop after 2 stale searches
            for p in new_hits:
                retrieved_idx.add(p.idx)
                is_supporting = p.is_supporting
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
            return True

        for step in range(1, self.max_steps + 1):
            out = self.backend.complete(
                ctx.render_prompt() + CONTINUE_CUE,
                stop=["</search>", "</answer>", "<information>"],
            )
            kind, payload = _parse_action(out)

            # Think-only / truncated turn: one hard commit nudge that prefills
            # <answer> so the model cannot open a new <think> block.
            if kind is None and not commit_nudge_used:
                commit_nudge_used = True
                out = self.backend.complete(
                    ctx.render_prompt() + COMMIT_CUE,
                    stop=["</answer>", "</search>"],
                    max_tokens=64,
                    assistant_prefix="<answer>",
                )
                kind, payload = _parse_action(out)
                # If it somehow still didn't answer, leave kind as None and fall
                # through to forced answer after the loop.
                if kind is None:
                    # Prefill may have returned bare answer text without tags.
                    bare = (out or "").replace("<answer>", "").replace("</answer>", "").strip()
                    if bare and "<think>" not in bare and "<search>" not in bare:
                        kind, payload = "answer", bare

            if kind == "answer":
                prediction = payload
                answered = True
                break

            if kind == "search":
                if not apply_search(payload):
                    break
                continue

            # Still no parseable action after the commit nudge (or nudge already used).
            break

        if not answered:
            prediction = _force_answer(self.backend, ctx)

        u1 = self.backend.usage
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
            gold_titles_total=len(gold_titles),
            gold_titles_retrieved=len(retrieved_gold_titles),
        )
