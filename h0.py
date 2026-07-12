"""
h0.py

H0 -- the floor of the heuristic ladder.

Timing:    none. Never checks the budget before appending. Only reacts once
           the hard limit is actually breached.
Selection: none. When overflow happens, drop oldest blocks first, with no
           protections -- even the question block can get discarded.

This isn't meant to be a good policy. It's the baseline every other rung
(H1, H2, H3, Oracle, Learned) gets measured against.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Block:
    """One segment of agent interaction history."""
    id: int
    role: str  # "question" | "thought" | "action" | "observation"
    text: str
    n_tokens: int
    step_idx: int
    objective_idx: int = -1               # which sub-question this belongs to, in a multi-objective trajectory
    is_supporting: Optional[bool] = None  # gold label; populated when the source dataset provides one


@dataclass
class BudgetState:
    B: int                          # hard token budget
    history: List[Block] = field(default_factory=list)

    @property
    def used(self) -> int:
        return sum(b.n_tokens for b in self.history)


def h0_step(budget: BudgetState, pending_obs: Block) -> tuple[BudgetState, bool]:
    """One ReAct-loop hook, H0 behavior only.

    Appends pending_obs unconditionally (no proactive timing check), then
    -- only if that pushed us over budget -- hard-truncates oldest blocks,
    with no pinning of the question or anything else, until it fits.

    Returns the updated budget and whether truncation fired.
    """
    budget.history.append(pending_obs)

    truncated = False
    while budget.used > budget.B and budget.history:
        budget.history.pop(0)
        truncated = True

    return budget, truncated


def make_observation_block(client: "OpenAI", prompt: str, id_: int, step_idx: int) -> Block:
    """Calls the real backbone and wraps the reply as a Block, using the
    API's own reported token count instead of guessing. This is what makes
    an API-backed demo actually test H0's budget math, not just prove the
    endpoint is reachable."""
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    text = response.choices[0].message.content or ""
    n_tokens = response.usage.completion_tokens if response.usage else len(text.split())
    return Block(id=id_, role="observation", text=text, n_tokens=n_tokens, step_idx=step_idx)


if __name__ == "__main__":
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["DEEPINFRA_API_KEY"],
        base_url="https://api.deepinfra.com/v1/openai",
    )

    question_prompts = [
        "In one sentence, name an unrelated 1998 film.",
        "In one sentence, describe rainy weather in a fictional town.",
        "In one sentence, name a fictional movie director.",
        "In one sentence, give fictional box office numbers for a movie's opening weekend.",
        "In one sentence, describe a director's documentary from the same year as their film.",
    ]

    budget = BudgetState(
        B=120,
        history=[Block(id=0, role="question", text="who directed the movie the composer scored", n_tokens=8, step_idx=0)],
    )
    n_truncations = 0
    for i, prompt in enumerate(question_prompts, start=1):
        obs = make_observation_block(client, prompt, id_=i, step_idx=i)
        budget, truncated = h0_step(budget, obs)
        n_truncations += int(truncated)
        print(f"step {i}: +{obs.n_tokens} tok | used={budget.used}/{budget.B} | truncated={truncated}")

    kept_ids = [b.id for b in budget.history]
    print(f"H0 | kept block ids: {kept_ids} | truncations fired: {n_truncations}")
    print(f"note: block 0 (the question) is {'STILL' if 0 in kept_ids else 'NO LONGER'} in history")