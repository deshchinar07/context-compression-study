"""Query-focused summarization, used only by H3.

H3 is the "fair strong heuristic": instead of *deleting* a low-relevance block it
*summarizes* it, conditioned on the task query, so some signal survives. This is
the training-free move a skeptical reviewer would ask about ("why not just try
this?").

The summarizer uses the same frozen backbone as the agent. Its cost is counted in
the same usage meter, so H3's higher inference cost shows up honestly in the
cost-normalized comparison.
"""

from __future__ import annotations

from typing import Optional

from .llm import LLMBackend

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
        # A summary should be short; cap generation so it cannot balloon the budget.
        out = self.backend.complete(prompt, max_tokens=budget_words * 4)
        return out.strip()
