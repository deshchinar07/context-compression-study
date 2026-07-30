
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from .blocks import Block, Context, OBSERVATION, SUMMARY
from .retrieval_scoring import LexicalScorer
from .llm import LLMBackend
from . import tokenizer


_PROMPT = """You are compressing an agent's memory. Rewrite the passage below into \
a compact note that preserves ONLY facts that could help answer the question(s). \
Keep names, dates, numbers, and entities. Drop everything irrelevant. \
Write at most {budget_words} words. No preamble.

Question(s): {query}

Passage:
{text}

Compact note:"""


class Summarizer:
    def __init__(self, backend: LLMBackend, max_words: int = 40):
        self.backend = backend
        self.max_words = max_words

    def summarize(self, text: str, query: str, max_words: Optional[int] = None) -> str:
        budget_words = max_words or self.max_words
        prompt = _PROMPT.format(budget_words=budget_words, query=query, text=text)

        out = self.backend.complete(prompt, max_tokens=budget_words * 4)
        return out.strip()


@dataclass
class CompressionStats:
    triggered: int = 0
    dropped: int = 0
    summarized: int = 0


class H0:

    name = "H0"
    uses_gold = False

    def __init__(self):
        self.stats = CompressionStats()

    def on_append(self, ctx: Context, pending: Block, *, scorer: Optional[LexicalScorer] = None, summarizer: Optional[Summarizer] = None, query: str = "") -> None:
        ctx.blocks.append(pending)
        fired = False
        while ctx.used() > ctx.budget and ctx.blocks:
            ctx.blocks.pop(0)
            self.stats.dropped += 1
            fired = True
        self.stats.triggered += int(fired)


class H0p:
    """Reactive truncation like H0, but the question block is pinned."""

    name = "H0p"
    uses_gold = False

    def __init__(self):
        self.stats = CompressionStats()

    def on_append(self, ctx: Context, pending: Block, *, scorer: Optional[LexicalScorer] = None, summarizer: Optional[Summarizer] = None, query: str = "") -> None:
        ctx.blocks.append(pending)
        fired = False
        while ctx.used() > ctx.budget:
            candidates = [b for b in ctx.blocks if b.role in (OBSERVATION, SUMMARY)]
            if not candidates:
                break
            ctx.remove(candidates[0])
            self.stats.dropped += 1
            fired = True
        self.stats.triggered += int(fired)


class H1:

    name = "H1"
    uses_gold = False

    def __init__(self):
        self.stats = CompressionStats()

    def on_append(self, ctx: Context, pending: Block, *, scorer: Optional[LexicalScorer] = None, summarizer: Optional[Summarizer] = None, query: str = "") -> None:
        fired = False
        while ctx.used() + pending.n_tokens > ctx.budget:

            candidates = [b for b in ctx.blocks if b.role in (OBSERVATION, SUMMARY)]
            if not candidates:
                break
            ctx.remove(candidates[0])
            self.stats.dropped += 1
            fired = True
        ctx.blocks.append(pending)
        self.stats.triggered += int(fired)


class H2:

    name = "H2"
    uses_gold = False

    def __init__(self):
        self.stats = CompressionStats()

    def on_append(self, ctx: Context, pending: Block, *, scorer: Optional[LexicalScorer] = None, summarizer: Optional[Summarizer] = None, query: str = "") -> None:
        assert scorer is not None, "H2 requires a relevance scorer"
        fired = False
        while ctx.used() + pending.n_tokens > ctx.budget:
            candidates = [b for b in ctx.blocks if b.role in (OBSERVATION, SUMMARY)]
            if not candidates:
                break

            victim = min(
                candidates,
                key=lambda b: (scorer.score(b.text), ctx.blocks.index(b)),
            )
            ctx.remove(victim)
            self.stats.dropped += 1
            fired = True
        ctx.blocks.append(pending)
        self.stats.triggered += int(fired)


class H3:

    name = "H3"
    uses_gold = False

    def __init__(self, summary_max_words: int = 40):
        self.stats = CompressionStats()
        self.summary_max_words = summary_max_words

    def on_append(
        self,
        ctx: Context,
        pending: Block,
        *,
        scorer: Optional[LexicalScorer] = None,
        summarizer: Optional[Summarizer] = None,
        query: str = "",
    ) -> None:
        assert scorer is not None and summarizer is not None, "H3 needs scorer+summarizer"
        fired = False

        while ctx.used() + pending.n_tokens > ctx.budget:
            anchor = ctx.most_recent_observation()
            candidates = [
                b for b in ctx.blocks
                if b.role in (OBSERVATION, SUMMARY) and b is not anchor
            ]
            if not candidates:
                break
            victim = min(
                candidates,
                key=lambda b: (scorer.score(b.text), ctx.blocks.index(b)),
            )


            if victim.role == OBSERVATION:
                note = summarizer.summarize(victim.text, query, max_words=self.summary_max_words)
                new_tokens = tokenizer.count_tokens(f"<memory>{note}</memory>")
                self.stats.summarized += 1
                if note and new_tokens < victim.n_tokens:
                    victim.role = SUMMARY
                    victim.text = note
                    victim.recount()
                    fired = True
                    continue

            ctx.remove(victim)
            self.stats.dropped += 1
            fired = True
        ctx.blocks.append(pending)
        self.stats.triggered += int(fired)


class Oracle:

    name = "Oracle"
    uses_gold = True

    def __init__(self):
        self.stats = CompressionStats()

    def on_append(
        self,
        ctx: Context,
        pending: Block,
        *,
        scorer: Optional[LexicalScorer] = None,
        summarizer: Optional[Summarizer] = None,
        query: str = "",
    ) -> None:
        fired = False
        while ctx.used() + pending.n_tokens > ctx.budget:
            candidates = [b for b in ctx.blocks if b.role in (OBSERVATION, SUMMARY)]
            if not candidates:
                break
            victim = min(
                candidates,
                key=lambda b: (
                    1.0 if b.is_supporting else 0.0,
                    ctx.blocks.index(b),
                ),
            )
            ctx.remove(victim)
            self.stats.dropped += 1
            fired = True
        ctx.blocks.append(pending)
        self.stats.triggered += int(fired)


LADDER = {"H0": H0, "H0p": H0p, "H1": H1, "H2": H2, "H3": H3, "Oracle": Oracle}

Policy = Union[H0, H0p, H1, H2, H3, Oracle]


def build_policy(name: str, summary_max_words: int = 40) -> Policy:
    if name not in LADDER:
        raise ValueError(f"unknown policy {name!r}; choose from {list(LADDER)}")
    if name == "H3":
        return H3(summary_max_words=summary_max_words)
    return LADDER[name]()
