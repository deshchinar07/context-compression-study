from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import tokenizer

QUESTION = "question"
THOUGHT = "thought"
ACTION = "action"
OBSERVATION = "observation"
SUMMARY = "summary"


@dataclass
class Block:

    id: int
    role: str
    text: str
    step_idx: int
    objective_idx: int = -1
    is_supporting: Optional[bool] = None
    source_title: Optional[str] = None
    n_tokens: int = 0

    def __post_init__(self):
        if self.n_tokens == 0:
            self.n_tokens = tokenizer.count_tokens(self.rendered())

    def rendered(self) -> str:
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

    budget: int
    blocks: List[Block] = field(default_factory=list)

    def used(self) -> int:
        return sum(b.n_tokens for b in self.blocks)


    def render_prompt(self) -> str:
        return "\n".join(b.rendered() for b in self.blocks)

    def next_id(self) -> int:
        return (max((b.id for b in self.blocks), default=-1)) + 1

    def observations(self) -> List[Block]:
        return [b for b in self.blocks if b.role in (OBSERVATION, SUMMARY)]

    def remove(self, block: Block) -> None:
        self.blocks.remove(block)

    def most_recent_observation(self) -> Optional[Block]:
        obs = self.observations()
        return obs[-1] if obs else None
