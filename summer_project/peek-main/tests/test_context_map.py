from peek.core.context_map import ContextMap
from peek.core.types import Operation


def test_initial_map_has_expected_sections():
    cmap = ContextMap.initial()
    headings = [
        line.strip().lstrip("#").strip().lower()
        for line in cmap.text.splitlines()
        if line.strip().startswith("##")
    ]
    assert "context roadmap" in headings
    assert "reusable results" in headings
    assert cmap.items() == []


def test_apply_add_assigns_stable_ids():
    cmap = ContextMap.initial()
    out = cmap.apply(
        [
            Operation(type="ADD", section="context_roadmap", content="Single 38k-char block."),
            Operation(type="ADD", section="context_understanding", content="Corpus of reviews."),
        ]
    )
    items = out.items()
    assert {it.section for it in items} == {"context_roadmap", "context_understanding"}
    ids = [it.id for it in items]
    assert ids[0].startswith("cr-")
    assert ids[1].startswith("cu-")
    assert ids[0] != ids[1]


def test_apply_replace_preserves_id():
    cmap = ContextMap.initial().apply(
        [Operation(type="ADD", section="context_roadmap", content="old")]
    )
    item_id = cmap.items()[0].id
    after = cmap.apply([Operation(type="REPLACE", item_id=item_id, content="new")])
    assert [it.id for it in after.items()] == [item_id]
    assert after.items()[0].content == "new"


def test_apply_delete_removes_item():
    cmap = ContextMap.initial().apply(
        [Operation(type="ADD", section="context_roadmap", content="x")]
    )
    item_id = cmap.items()[0].id
    after = cmap.apply([Operation(type="DELETE", item_id=item_id)])
    assert after.items() == []


def test_next_id_is_monotonic():
    cmap = ContextMap.initial()
    for _ in range(3):
        cmap = cmap.apply([Operation(type="ADD", section="context_roadmap", content="x")])
    ids = [it.id for it in cmap.items()]
    nums = [int(i.split("-")[1]) for i in ids]
    assert nums == sorted(nums) and len(set(nums)) == 3


def test_unknown_section_is_normalized_or_dropped_at_cartographer_level():
    # Apply itself tolerates arbitrary sections (Cartographer is the validator),
    # but section slug fallback should still produce a usable line.
    cmap = ContextMap.initial().apply(
        [Operation(type="ADD", section="custom_section", content="x")]
    )
    assert any("[cust" in line for line in cmap.text.splitlines())
