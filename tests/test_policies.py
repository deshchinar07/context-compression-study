
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ladder.blocks import Block, Context, QUESTION, OBSERVATION, SUMMARY
from ladder.policies import H0, H1, H2, H3, Oracle


class DummyScorer:
    def __init__(self, mapping):
        self.mapping = mapping

    def score(self, text):
        return self.mapping.get(text, 0.0)


class DummySummarizer:
    def __init__(self, note="note"):
        self.note = note
        self.calls = 0

    def summarize(self, text, query, max_words=None):
        self.calls += 1
        return self.note


def _q(tok=4):
    return Block(id=0, role=QUESTION, text="Q", step_idx=0, n_tokens=tok)


def _obs(id_, text, tok=4, step=1, supporting=None):
    return Block(id=id_, role=OBSERVATION, text=text, step_idx=step,
                 n_tokens=tok, is_supporting=supporting, source_title=text)


def test_h0_can_discard_the_question():
    ctx = Context(budget=10, blocks=[_q()])
    p = H0()
    for i, name in enumerate(["A", "B", "C"], start=1):
        p.on_append(ctx, _obs(i, name))


    roles = [b.role for b in ctx.blocks]
    assert QUESTION not in roles, "H0 must be able to drop the question"
    assert ctx.used() <= ctx.budget


def test_h1_pins_question_and_drops_oldest():
    ctx = Context(budget=8, blocks=[_q()])
    p = H1()
    p.on_append(ctx, _obs(1, "A"))
    p.on_append(ctx, _obs(2, "B"))
    titles = [b.source_title for b in ctx.blocks if b.role == OBSERVATION]
    assert ctx.blocks[0].role == QUESTION, "H1 must pin the question"
    assert titles == ["B"], "H1 drops oldest (FIFO) evidence"
    assert p.stats.dropped == 1


def test_h2_drops_least_relevant_not_oldest():
    scorer = DummyScorer({"hi": 5.0, "lo": 0.0, "mid": 2.0})
    ctx = Context(budget=12, blocks=[_q()])
    p = H2()
    p.on_append(ctx, _obs(1, "hi"), scorer=scorer)
    p.on_append(ctx, _obs(2, "lo"), scorer=scorer)
    p.on_append(ctx, _obs(3, "mid"), scorer=scorer)
    kept = {b.source_title for b in ctx.blocks if b.role == OBSERVATION}
    assert kept == {"hi", "mid"}, f"H2 should keep high-relevance, drop 'lo'; got {kept}"


def test_h3_summarizes_and_anchors_recent():
    scorer = DummyScorer({"A": 0.0, "B": 5.0})
    summ = DummySummarizer(note="tiny")
    ctx = Context(budget=12, blocks=[_q()])
    p = H3()
    p.on_append(ctx, _obs(1, "A"), scorer=scorer, summarizer=summ)
    p.on_append(ctx, _obs(2, "B"), scorer=scorer, summarizer=summ)
    p.on_append(ctx, _obs(3, "C"), scorer=scorer, summarizer=summ)
    assert p.stats.summarized >= 1, "H3 must attempt summarization before deleting"
    assert ctx.blocks[0].role == QUESTION, "H3 must keep the question"

    titles = [b.source_title for b in ctx.blocks if b.role in (OBSERVATION, SUMMARY)]
    assert "B" in titles, "H3 must anchor the most-recent observation"


def test_oracle_keeps_supporting_drops_distractor():
    ctx = Context(budget=12, blocks=[_q()])
    p = Oracle()
    p.on_append(ctx, _obs(1, "supp", supporting=True))
    p.on_append(ctx, _obs(2, "dist", supporting=False))
    p.on_append(ctx, _obs(3, "dist2", supporting=False))
    kept = {b.source_title for b in ctx.blocks if b.role == OBSERVATION}
    assert "supp" in kept, "Oracle must retain gold supporting evidence"
    assert "dist" not in kept, "Oracle should shed distractors first"


def test_oracle_is_the_only_rung_reading_gold():
    assert Oracle.uses_gold is True
    for cls in (H0, H1, H2, H3):
        assert cls.uses_gold is False


def test_supporting_kept_counts_summarized_blocks():
    """Summarized supporting evidence must count as kept (agent.py counter)."""
    import ladder.agent as agent_mod

    src = open(agent_mod.__file__, encoding="utf-8").read()
    assert "b.role in (OBSERVATION, SUMMARY) and b.is_supporting" in src

    ctx = Context(budget=70, blocks=[_q(tok=5)])
    p = H3()
    scorer = DummyScorer({})
    summ = DummySummarizer(note="tiny note")
    for i, title in enumerate(["A", "B", "C"], start=1):
        p.on_append(
            ctx,
            _obs(i, title, tok=25, supporting=True),
            scorer=scorer,
            summarizer=summ,
        )
    assert any(b.role == SUMMARY and b.is_supporting for b in ctx.blocks), (
        "expected a surviving SUMMARY that retains is_supporting"
    )
    counted = sum(
        1 for b in ctx.blocks if b.role in (OBSERVATION, SUMMARY) and b.is_supporting
    )
    obs_only = sum(
        1 for b in ctx.blocks if b.role == OBSERVATION and b.is_supporting
    )
    assert counted > obs_only


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} policy tests passed.")
