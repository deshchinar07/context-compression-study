"""Interaction-history blocks and the context window that holds them.

The agent's history is a sequence of blocks c_1 ... c_K (BACM-RL framing). A block
is the atomic unit a compression policy can keep, drop, or summarize. Every rung
of the ladder operates on exactly this structure and sees exactly the same budget
signal (remaining budget r_t = B - |C_t| and pending observation size |o_t|).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import tokenizer

# Roles. Only OBSERVATION and SUMMARY are "evidence" that compression targets;
# QUESTION carries the task/instruction; THOUGHT/ACTION are the running trace.
QUESTION = "question"
THOUGHT = "thought"
ACTION = "action"
OBSERVATION = "observation"
SUMMARY = "summary"


@dataclass
class Block:
    """One segment of agent interaction history."""

    id: int
    role: str
    text: str
    step_idx: int
    objective_idx: int = -1            # which sub-question this belongs to (multi-objective)
    is_supporting: Optional[bool] = None  # GOLD label; ONLY the Oracle may read this
    source_title: Optional[str] = None    # title of the source paragraph (observations)
    n_tokens: int = 0

    def __post_init__(self):
        if self.n_tokens == 0:
            self.n_tokens = tokenizer.count_tokens(self.rendered())

    def rendered(self) -> str:
        """How this block appears inside the prompt the backbone actually sees."""
        if self.role == QUESTION:
            return self.text
        if self.role == THOUGHT:
            return f"<think>{self.text}</think>"
        if self.role == ACTION:
            return f"<search>{self.text}</search>"
        if self.role == OBSERVATION:
            title = self.source_title or ""
            return f"<information>(Title: {title}) {self.text}</information>"
        if self.role == SUMMARY:
            return f"<memory>{self.text}</memory>"
        return self.text

    def recount(self) -> None:
        self.n_tokens = tokenizer.count_tokens(self.rendered())


@dataclass
class Context:
    """Ordered blocks plus a hard token budget B."""

    budget: int
    blocks: List[Block] = field(default_factory=list)

    def used(self) -> int:
        return sum(b.n_tokens for b in self.blocks)

    def remaining(self) -> int:
        """r_t = B - |C_t| (can go negative under H0's reactive overflow)."""
        return self.budget - self.used()

    def render_prompt(self) -> str:
        return "\n".join(b.rendered() for b in self.blocks)

    def next_id(self) -> int:
        return (max((b.id for b in self.blocks), default=-1)) + 1

    # --- helpers used by the policies -------------------------------------
    def observations(self) -> List[Block]:
        return [b for b in self.blocks if b.role in (OBSERVATION, SUMMARY)]

    def remove(self, block: Block) -> None:
        self.blocks.remove(block)

    def most_recent_observation(self) -> Optional[Block]:
        obs = self.observations()
        return obs[-1] if obs else None
