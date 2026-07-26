"""Heuristic-ladder harness for decomposing context-compression gains.

The package implements the five-rung ladder (H0, H1, H2, H3, Oracle) described in
the research plan, on top of a single frozen backbone and a fixed retrieval /
action space, so that the *only* thing that varies across rungs is the
compression policy.

Nothing in here is tuned to make any rung win. See ``policies.py`` for the exact,
auditable difference between each rung.
"""

import os as _os

# The Qwen tokenizer is used for budget accounting; silence the noisy fork warning.
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__all__ = [
    "blocks",
    "tokenizer",
    "data",
    "retrieval",
    "corpus",
    "kilt",
    "policies",
    "llm",
    "agent",
    "runner",
    "report",
]
