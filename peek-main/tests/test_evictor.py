from peek.core.context_map import ContextMap
from peek.core.evictor import evict, update_scores
from peek.core.types import Operation


def char_counter(s: str) -> int:
    return len(s)


def _build_three_items() -> tuple[ContextMap, list[str]]:
    cmap = ContextMap.initial()
    cmap = cmap.apply(
        [
            Operation(type="ADD", section="context_roadmap", content="A"),
            Operation(type="ADD", section="context_roadmap", content="B"),
            Operation(type="ADD", section="context_roadmap", content="C"),
        ]
    )
    return cmap, [it.id for it in cmap.items()]


def test_no_eviction_under_budget():
    cmap, _ = _build_three_items()
    out = evict(cmap, scores={}, token_budget=10_000, token_counter=char_counter)
    assert out.text == cmap.text


def test_lowest_score_evicted_first():
    cmap, ids = _build_three_items()
    # Force eviction of exactly one item.
    target_budget = char_counter(cmap.text) - len("[cr-00001] A") - 1
    scores = {ids[0]: 5, ids[1]: -3, ids[2]: 0}
    out = evict(cmap, scores=scores, token_budget=target_budget, token_counter=char_counter)
    surviving = {it.id for it in out.items()}
    assert ids[1] not in surviving
    assert ids[0] in surviving


def test_age_tiebreak_oldest_first():
    cmap, ids = _build_three_items()
    target_budget = char_counter(cmap.text) - len("[cr-00001] A") - 1
    out = evict(cmap, scores={}, token_budget=target_budget, token_counter=char_counter)
    surviving = {it.id for it in out.items()}
    assert ids[0] not in surviving  # oldest of the equal-score group is removed first


def test_update_scores_sign_convention():
    s = update_scores({}, {"a": "helpful", "b": "harmful", "c": "stale", "d": "neutral"})
    assert s == {"a": 1, "b": -1, "c": -1, "d": 0}
    s = update_scores(s, {"a": "helpful", "b": "helpful"})
    assert s["a"] == 2 and s["b"] == 0
