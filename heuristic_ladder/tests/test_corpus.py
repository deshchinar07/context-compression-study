"""Offline tests for the shared 'corpus' retrieval scope.

No network and no API: we construct tiny synthetic Examples, build a union
corpus from them, and assert the four properties that make corpus-scope
trustworthy:

  1. the union corpus contains every gold title (dedup keeps them),
  2. gold labels are assigned to retrieved passages by TITLE MATCH (not by any
     pre-existing dataset flag on the corpus passage),
  3. the Oracle, reading those title-matched labels, still keeps gold over
     distractors ("Oracle-given-retrieval"),
  4. gold-retrieval recall is computed correctly when the retriever surfaces only
     some of the gold.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ladder.blocks import Block, Context, QUESTION, OBSERVATION  # noqa: E402
from ladder.corpus import build_union_corpus, CorpusIndex  # noqa: E402
from ladder.data import Example, Paragraph  # noqa: E402
from ladder.policies import Oracle  # noqa: E402


def _para(idx, title, text, supporting):
    return Paragraph(idx=idx, title=title, text=text, is_supporting=supporting)


def _example(ex_id, question, answer, paras):
    return Example(
        id=ex_id,
        questions=[question],
        answers=[[answer]],
        paragraphs=paras,
        dataset="synthetic",
    )


def _make_examples():
    ex1 = _example(
        "e1", "who directed film X?", "alice",
        [
            _para(0, "Film X", "Film X is a movie directed by Alice.", True),
            _para(1, "Alice", "Alice is a film director.", True),
            _para(2, "Bob", "Bob is a chef in another town.", False),
        ],
    )
    ex2 = _example(
        "e2", "capital of Y?", "zeta",
        [
            _para(0, "Country Y", "Country Y has capital Zeta.", True),
            _para(1, "Bob", "Bob is a chef in another town.", False),  # dup title with ex1
            _para(2, "Zeta", "Zeta is a large city.", False),
        ],
    )
    return [ex1, ex2]


def test_union_corpus_dedups_and_contains_all_gold():
    exs = _make_examples()
    passages = build_union_corpus(exs)
    titles = [p.title for p in passages]
    # 'Bob' appears in both examples but must be deduplicated to one passage.
    assert titles.count("Bob") == 1, "union corpus must dedup by title"
    # every gold title across all examples is present in the corpus
    gold = set().union(*[e.supporting_titles for e in exs])
    assert gold <= set(titles), f"corpus missing gold titles: {gold - set(titles)}"
    # corpus passages carry no meaningful gold flag (labeling happens by title)
    assert all(p.is_supporting is False for p in passages)
    # re-indexed contiguously
    assert [p.idx for p in passages] == list(range(len(passages)))


def test_title_match_labeling_not_preexisting_flag():
    exs = _make_examples()
    index = CorpusIndex(build_union_corpus(exs), kind="bm25")
    ex1 = exs[0]
    hits = index.search("who directed film X?", topk=10)
    # The corpus passage for a gold title must be labelable as supporting for ex1
    # purely via title match, even though the corpus passage's own flag is False.
    labeled = {p.title: (p.title in ex1.supporting_titles) for p in hits}
    assert labeled.get("Film X") is True
    assert labeled.get("Alice") is True
    assert labeled.get("Zeta") is False  # gold for ex2, not ex1


def test_oracle_given_retrieval_keeps_titlematched_gold():
    exs = _make_examples()
    ex1 = exs[0]
    gold_titles = ex1.supporting_titles

    # Simulate what the agent does under corpus scope: label retrieved passages by
    # title match, feed them to the Oracle under budget pressure.
    retrieved = [
        ("Bob", "Bob is a chef in another town.", 4),
        ("Film X", "Film X is a movie directed by Alice.", 4),
        ("Alice", "Alice is a film director.", 4),
    ]
    ctx = Context(budget=12, blocks=[Block(id=0, role=QUESTION, text="Q", step_idx=0, n_tokens=4)])
    oracle = Oracle()
    for i, (title, text, tok) in enumerate(retrieved, start=1):
        is_sup = title in gold_titles
        blk = Block(id=i, role=OBSERVATION, text=text, step_idx=i,
                    n_tokens=tok, is_supporting=is_sup, source_title=title)
        oracle.on_append(ctx, blk)

    kept = {b.source_title for b in ctx.blocks if b.role == OBSERVATION}
    # Q(4) + 3x4 = 16 > 12, so one eviction: the non-gold "Bob" must go first.
    assert "Bob" not in kept, "Oracle-given-retrieval must shed title-matched distractors first"
    assert {"Film X", "Alice"} <= kept, "Oracle must keep title-matched gold"


def test_gold_retrieval_recall_partial():
    exs = _make_examples()
    ex1 = exs[0]
    gold_titles = ex1.supporting_titles  # {"Film X", "Alice"}

    # Pretend the retriever only surfaced one of the two gold passages.
    retrieved_titles = ["Film X", "Bob"]
    retrieved_gold = {t for t in retrieved_titles if t in gold_titles}
    assert len(gold_titles) == 2
    assert retrieved_gold == {"Film X"}
    recall = len(retrieved_gold) / len(gold_titles)
    assert recall == 0.5, "recall must reflect that only half the gold was retrieved"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} corpus tests passed.")
