"""Token counting.

Budget accounting has to be token-accurate and, crucially, *identical* across all
rungs of the ladder -- otherwise a budget of "8k tokens" means something different
for H1 than for H3 and the comparison is corrupted.

We use ONE tokenizer for the entire study: the real Qwen2.5-7B-Instruct tokenizer,
because that's the one frozen backbone every rung runs against (the fairness
premise of the whole harness -- see the research plan's backbone-match
requirement). There is no fallback and no runtime override: a second tokenizer
would only ever be needed if you were comparing across *different* backbones,
which this study deliberately does not do. Hardcoding it here means ``llm.py``'s
API calls and this module's budget accounting can never silently drift apart.

If it can't be loaded we raise immediately and say how to fix it, rather than
substituting a different tokenizer's vocabulary without warning.

Pre-fetch ``Qwen/Qwen2.5-7B-Instruct`` via `transformers` once with network
access so the tokenizer is cached locally; every run after that is offline for
tokenization.
"""

from __future__ import annotations
from transformers import AutoTokenizer

import functools

# The one and only backbone for this study. Import this constant wherever a
# model name is needed (e.g. ``llm.LLMBackend``) instead of re-typing the
# string, so there is exactly one place to change it.
BACKBONE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class _Counter:
    """Wraps the loaded tokenizer behind one uniform ``.count()`` call."""

    def __init__(self, encode_fn):
        self._encode = encode_fn

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encode(text))


@functools.lru_cache(maxsize=1)  # load the tokenizer once, reuse for the whole run
def _load_counter() -> _Counter:
    tok = AutoTokenizer.from_pretrained(BACKBONE_MODEL)
    return _Counter(encode_fn=lambda t: tok.encode(t, add_special_tokens=False))


def backend_name() -> str:
    return f"transformers:{BACKBONE_MODEL}"


def count_tokens(text: str) -> int:
    return _load_counter().count(text)
