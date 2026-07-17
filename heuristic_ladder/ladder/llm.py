"""Backbone LLM access.

A thin, provider-agnostic wrapper over an OpenAI-compatible chat endpoint. The
default target is DeepInfra serving a *frozen* ``Qwen/Qwen2.5-7B-Instruct`` -- the
shared backbone across Search-R1 / MEM1 / BACM-RL / FoldAct, which is what makes
the heuristic-vs-learned comparison legitimate.

Determinism: temperature is 0 by default, so a given context yields a fixed
continuation. Every rung of the ladder uses the *same* backend instance and the
*same* decoding parameters; only the context they build differs.

Every call's token usage is captured so cost can be reported against real,
measured inference tokens (not estimates).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .tokenizer import BACKBONE_MODEL


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_calls: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.n_calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMBackend:
    model: str = BACKBONE_MODEL
    base_url: str = "https://api.deepinfra.com/v1/openai"
    api_key_env: str = "DEEPINFRA_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 512
    max_retries: int = 5
    timeout: float = 120.0
    usage: Usage = field(default_factory=Usage)
    _client: object = field(default=None, repr=False)

    def __post_init__(self):
        from openai import OpenAI

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"missing API key: set ${self.api_key_env} (e.g. in a .env file)"
            )
        self._client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=self.timeout)

    def complete(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
        assistant_prefix: Optional[str] = None,
    ) -> str:
        """Single chat completion. Retries with exponential backoff on transient errors.

        ``assistant_prefix`` (e.g. ``"<answer>"``) prefills the assistant turn so the
        model continues from that text -- used to force an answer tag when the model
        otherwise burns its budget inside ``<think>``.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        if assistant_prefix:
            messages.append({"role": "assistant", "content": assistant_prefix})

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    stop=stop,
                )
                text = resp.choices[0].message.content or ""
                # Some servers echo the prefill; others return only the continuation.
                if assistant_prefix and not text.startswith(assistant_prefix):
                    text = assistant_prefix + text
                if resp.usage:
                    self.usage.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
                return text
            except Exception as e:  # noqa: BLE001 - provider errors vary; retry then raise
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")

    def complete_raw(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: float = 0.95,
    ) -> str:
        """Raw (non-chat) text completion via the ``/v1/completions`` endpoint.

        Needed by baselines like MEM1 that *continue* a partially-written assistant
        turn (the chat endpoint cannot: it always starts a fresh assistant message).
        The caller is responsible for building the exact prompt string (e.g. via a
        chat template). Usage is tracked on the same counter as ``complete`` so cost
        accounting stays unified.
        """
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=self.temperature if temperature is None else temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    stop=stop,
                    top_p=top_p,
                )
                text = resp.choices[0].text or ""
                if resp.usage:
                    self.usage.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
                return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"raw LLM call failed after {self.max_retries} retries: {last_err}")
