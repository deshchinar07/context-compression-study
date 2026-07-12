"""The heuristic ladder: H0, H1, H2, H3, and the Oracle ceiling.

Every rung is invoked at the SAME decision point -- the agent is in a ReAct loop
and is about to append a new observation o_t -- and sees the SAME budget signal
(remaining budget r_t = B - |C_t| and pending size |o_t|). The rungs differ only
in the compression policy, so the gap between consecutive rungs isolates exactly
one design decision:

    H0 -> H1 : does being *proactive* about the budget help at all?      (TIMING)
    H1 -> H2 : does *what* you drop matter, holding timing fixed?         (SELECTION)
    H2 -> H3 : do anchors + summarize-instead-of-delete add anything?     (SELECTION++)
    H3 -> Oracle : how much headroom is left to perfect selection?        (CEILING)

Scope of compression (identical for every rung): only OBSERVATION and SUMMARY
blocks -- the evidence store -- are compressible. The QUESTION block carries the
task and is structural; the reasoning trace (think/action) is transient and is
regenerated each turn from the evidence, so it never accumulates. This keeps the
budget pressure coming from *accumulated evidence*, which is what the compression
literature actually compresses, and keeps the decomposition uncontaminated.

Fairness invariants enforced here:
  * No rung except the Oracle may read ``Block.is_supporting`` (the gold label).
  * No rung is tuned per-dataset; the only knobs are H3's summary length (a dev-set
    choice) and the shared relevance scorer.
  * H0 is the only rung allowed to be dumb enough to discard the question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .blocks import Block, Context, OBSERVATION, QUESTION, SUMMARY
from .scoring import LexicalScorer
from .summarizer import Summarizer
from . import tokenizer


@dataclass
class CompressionStats:
    triggered: int = 0     # number of on_append calls in which any compression fired
    dropped: int = 0       # blocks removed
    summarized: int = 0    # summarize LLM calls made (H3 only)


class CompressionPolicy:
    """Base class. Subclasses implement ``on_append``."""

    name: str = "base"
    uses_gold: bool = False

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
        raise NotImplementedError

    # shared helpers -------------------------------------------------------
    @staticmethod
    def _evictable(ctx: Context) -> List[Block]:
        """Evidence blocks only; the question is never touched by H1+ rungs."""
        return [b for b in ctx.blocks if b.role in (OBSERVATION, SUMMARY)]


class H0(CompressionPolicy):
    """The floor. No timing, no selection.

    Appends unconditionally (never checks the budget first), then -- only if that
    overflowed -- hard-truncates oldest blocks with NO protections: even the
    question can be discarded. This is not a real policy; it is the "what if you do
    nothing clever" baseline every other number is measured against.
    """

    name = "H0"

    def on_append(self, ctx, pending, *, scorer=None, summarizer=None, query=""):
        ctx.blocks.append(pending)
        fired = False
        while ctx.used() > ctx.budget and ctx.blocks:
            ctx.blocks.pop(0)  # oldest, any role -- question included
            self.stats.dropped += 1
            fired = True
        self.stats.triggered += int(fired)


class H1(CompressionPolicy):
    """Timing, alone. Proactive budget check + FIFO drop-oldest, question pinned.

    Same dumb selection rule as H0's panic response (drop oldest), but triggered
    *early* so it never actually overflows, and with the question protected. The
    H0->H1 gap isolates the value of being budget-aware at all.
    """

    name = "H1"

    def on_append(self, ctx, pending, *, scorer=None, summarizer=None, query=""):
        fired = False
        while ctx.used() + pending.n_tokens > ctx.budget:
            candidates = self._evictable(ctx)
            if not candidates:
                break  # only the (pinned) question is left; accept rare overflow
            ctx.remove(candidates[0])  # first in list == oldest
            self.stats.dropped += 1
            fired = True
        ctx.blocks.append(pending)
        self.stats.triggered += int(fired)


class H2(CompressionPolicy):
    """Selection gets a brain. Same timing as H1; drop LEAST query-relevant first.

    The H1->H2 gap isolates whether *what* you drop matters, holding timing fixed.
    Relevance is the training-free lexical scorer -- no gold labels.
    """

    name = "H2"

    def on_append(self, ctx, pending, *, scorer=None, summarizer=None, query=""):
        assert scorer is not None, "H2 requires a relevance scorer"
        fired = False
        while ctx.used() + pending.n_tokens > ctx.budget:
            candidates = self._evictable(ctx)
            if not candidates:
                break
            # lowest relevance first; tie-break oldest (earliest in list).
            victim = min(
                candidates,
                key=lambda b: (scorer.score(b.text), ctx.blocks.index(b)),
            )
            ctx.remove(victim)
            self.stats.dropped += 1
            fired = True
        ctx.blocks.append(pending)
        self.stats.triggered += int(fired)


class H3(CompressionPolicy):
    """The fair strong heuristic. H2's ranking + two additions:

      * anchors: never drop the question or the most-recent observation;
      * summarize the least-relevant block instead of deleting it.

    This is the best training-free rung -- the one a skeptical reviewer would ask
    "why didn't you just try this?" about. The H2->H3 gap isolates the value of
    keeping-some-signal (summarize) and anchoring recency.
    """

    name = "H3"

    def __init__(self, summary_max_words: int = 40):
        super().__init__()
        self.summary_max_words = summary_max_words

    def on_append(self, ctx, pending, *, scorer=None, summarizer=None, query=""):
        assert scorer is not None and summarizer is not None, "H3 needs scorer+summarizer"
        fired = False
        # The most-recent observation already in context is an anchor.
        while ctx.used() + pending.n_tokens > ctx.budget:
            anchor = ctx.most_recent_observation()
            candidates = [b for b in self._evictable(ctx) if b is not anchor]
            if not candidates:
                break
            victim = min(
                candidates,
                key=lambda b: (scorer.score(b.text), ctx.blocks.index(b)),
            )
            # Summarize instead of delete -- unless the block is already a summary
            # or summarizing fails to save tokens, in which case drop it.
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
            # Fallback: drop it.
            ctx.remove(victim)
            self.stats.dropped += 1
            fired = True
        ctx.blocks.append(pending)
        self.stats.triggered += int(fired)


class Oracle(CompressionPolicy):
    """Not a real policy -- a ceiling. Perfect selection at H2 timing.

    Identical to H2 (same proactive timing, question pinned), except the relevance
    score is the GOLD supporting-fact label: 1.0 if the block's source paragraph is
    annotated as needed to answer the question, else 0.0. So under budget pressure
    it keeps exactly the supporting evidence and sheds distractors first.

    The H3->Oracle gap is the headroom: whatever remains is what any smarter policy
    (learned or otherwise) could still capture on the SELECTION decision.
    """

    name = "Oracle"
    uses_gold = True

    @staticmethod
    def _gold_score(b: Block) -> float:
        return 1.0 if b.is_supporting else 0.0

    def on_append(self, ctx, pending, *, scorer=None, summarizer=None, query=""):
        fired = False
        while ctx.used() + pending.n_tokens > ctx.budget:
            candidates = self._evictable(ctx)
            if not candidates:
                break
            victim = min(
                candidates,
                key=lambda b: (self._gold_score(b), ctx.blocks.index(b)),
            )
            ctx.remove(victim)
            self.stats.dropped += 1
            fired = True
        ctx.blocks.append(pending)
        self.stats.triggered += int(fired)


LADDER = {"H0": H0, "H1": H1, "H2": H2, "H3": H3, "Oracle": Oracle}


def build_policy(name: str, summary_max_words: int = 40) -> CompressionPolicy:
    if name not in LADDER:
        raise ValueError(f"unknown policy {name!r}; choose from {list(LADDER)}")
    if name == "H3":
        return H3(summary_max_words=summary_max_words)
    return LADDER[name]()
