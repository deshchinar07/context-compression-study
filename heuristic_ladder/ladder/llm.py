
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

from .tokenizer import BACKBONE_MODEL


SINGLE = (
    "Answer the question. You must reason inside <think> and </think> first. "
    "If you need facts, search with <search> keywords </search>; the top results "
    "will appear inside <information> and </information>. You may search multiple "
    "times. When you have enough information, give the final answer inside "
    "<answer> and </answer> using only the essential words, e.g. <answer> Beijing "
    "</answer>.\nQuestion: {questions}\n"
)

MULTI = (
    "You will answer multiple questions using iterative reasoning and search. "
    "Reason inside <think> and </think>. To gather facts, issue ONE query at a "
    "time inside <search> and </search>; results appear inside <information> and "
    "</information>. When every question is answered, provide all final answers, "
    "separated by semicolons, inside <answer> answer1; answer2; ... </answer>. "
    "Each answer must be concise -- only the essential words.\n"
    "Answer the following questions: {questions}\n"
)


CONTINUE_CUE = (
    "\nNow produce your next step: a <think>...</think> followed by exactly one "
    "<search>...</search> or <answer>...</answer>."
)


COMMIT_CUE = (
    "\nStop reasoning. Using the information above, give the final short answer now."
)

FORCE_ANSWER_CUE = (
    "\nCRITICAL: Using only the information above, output ONLY the final "
    "answer(s) inside <answer> and </answer>. "
    "Do NOT write <think>, do NOT search, do NOT explain. "
    "Example: <answer>Beijing</answer>"
)

FORCE_ANSWER_RETRY_CUE = (
    "\nYour previous reply was invalid. Reply with exactly one line of the form "
    "<answer>ANSWER</answer> and nothing else."
)

FORCE_ANSWER_SYSTEM = (
    "You extract short factual answers. Reply with only "
    "<answer>...</answer>. Never use <think> or any other tags."
)


def instruction(n_objectives: int, questions: str) -> str:
    template = SINGLE if n_objectives <= 1 else MULTI
    return template.format(questions=questions)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_calls: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.n_calls += 1


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
                
                if assistant_prefix and not text.startswith(assistant_prefix):
                    text = assistant_prefix + text
                if resp.usage:
                    self.usage.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
                return text
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")
