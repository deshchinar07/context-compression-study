"""Priority-based eviction enforcing a hard token budget on the context map.

Items are evicted in ascending order of their accumulated Distiller score
(helpful = +1, harmful/stale = -1, neutral = 0), ties broken by item age
(older IDs evicted first). See §3.2 of the PEEK paper.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from peek.core.context_map import ContextMap
from peek.core.types import ItemTag

_NUMERIC_TAIL = re.compile(r"-(\d+)$")


def update_scores(scores: dict[str, int], tags: dict[str, ItemTag]) -> dict[str, int]:
    out = dict(scores)
    for item_id, tag in tags.items():
        if tag == "helpful":
            out[item_id] = out.get(item_id, 0) + 1
        elif tag in ("harmful", "stale"):
            out[item_id] = out.get(item_id, 0) - 1
        else:
            out.setdefault(item_id, 0)
    return out


def evict(
    cmap: ContextMap,
    scores: dict[str, int],
    token_budget: int,
    token_counter: Callable[[str], int],
) -> ContextMap:
    if token_counter(cmap.text) <= token_budget:
        return cmap

    ordered_ids = [it.id for it in cmap.items()]
    ordered_ids.sort(key=lambda bid: (scores.get(bid, 0), _id_age(bid)))

    removed: set[str] = set()
    for bid in ordered_ids:
        removed.add(bid)
        trial = _strip_items(cmap.text, removed)
        if token_counter(trial) <= token_budget:
            return ContextMap(trial + "\n" if not trial.endswith("\n") else trial)
    return ContextMap(_strip_items(cmap.text, set(ordered_ids)))


def _id_age(item_id: str) -> int:
    m = _NUMERIC_TAIL.search(item_id)
    return int(m.group(1)) if m else 0


def _strip_items(text: str, ids: set[str]) -> str:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "]" in stripped:
            bid = stripped[1 : stripped.index("]")]
            if bid in ids:
                continue
        out.append(line)
    return "\n".join(out)
