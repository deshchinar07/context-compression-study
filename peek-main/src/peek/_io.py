from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from an LLM response."""
    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    for match in re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE):
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    for blob in _scan_balanced_braces(text):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    return None


def _scan_balanced_braces(text: str) -> list[str]:
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, start = 1, i
        i += 1
        while i < n and depth > 0:
            c = text[i]
            if c == '"':
                i += 1
                while i < n and text[i] != '"':
                    i += 2 if text[i] == "\\" else 1
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        if depth == 0:
            out.append(text[start:i])
    return out
