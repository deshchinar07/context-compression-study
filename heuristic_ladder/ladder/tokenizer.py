from __future__ import annotations
from transformers import AutoTokenizer

import functools

BACKBONE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class _Counter:

    def __init__(self, encode_fn):
        self._encode = encode_fn

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encode(text))


@functools.lru_cache(maxsize=1) 
def _load_counter() -> _Counter:
    tok = AutoTokenizer.from_pretrained(BACKBONE_MODEL)
    return _Counter(encode_fn=lambda t: tok.encode(t, add_special_tokens=False))


def backend_name() -> str:
    return f"transformers:{BACKBONE_MODEL}"


def count_tokens(text: str) -> int:
    return _load_counter().count(text)
