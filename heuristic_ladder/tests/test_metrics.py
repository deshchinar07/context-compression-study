"""Offline tests for primary MEM1-table and secondary standard-QA scoring."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ladder.report import (  # noqa: E402
    mem1_em,
    mem1_f1,
    mem1_normalize,
    score_prediction,
    standard_qa_em,
    standard_qa_f1,
    standard_qa_normalize,
)


def test_protocol_normalization_differs_as_expected():
    # Standard QA removes English articles; MEM1's preprocess_text does not.
    assert standard_qa_normalize("The  Beatles!") == "beatles"
    assert mem1_normalize("The  Beatles!") == "the beatles"


def test_protocol_exact_match():
    assert standard_qa_em("the Beijing", ["Beijing"]) == 1
    assert mem1_em("the Beijing", ["Beijing"]) == 0
    assert mem1_em("Beijing!", ["beijing"]) == 1


def test_standard_qa_f1_partial_credit():
    f1 = standard_qa_f1("New York City", ["New York"])
    assert 0.0 < f1 < 1.0
    assert standard_qa_f1("New York", ["New York"]) == 1.0
    assert standard_qa_f1("Paris", ["London"]) == 0.0


def test_mem1_f1_uses_sets_but_standard_qa_uses_counters():
    prediction = "foo foo bar"
    gold = ["foo bar bar"]
    assert mem1_f1(prediction, gold) == 1.0
    assert standard_qa_f1(prediction, gold) == 2 / 3


def test_both_protocols_sum_well_formed_multi_objective_answers():
    gold = [["Beijing"], ["Einstein"]]
    s = score_prediction("Beijing; Einstein", gold)
    assert s.mem1_table.n_objectives == 2
    assert s.mem1_table.summed_em == 2
    assert s.mem1_table.summed_f1 == 2.0
    assert s.mem1_table.mean_em == 1.0
    assert s.standard_qa.summed_em == 2
    assert s.standard_qa.summed_f1 == 2.0


def test_mem1_zeros_whole_example_when_answer_count_is_wrong():
    gold = [["Beijing"], ["Einstein"]]

    too_few = score_prediction("Beijing", gold)
    assert too_few.mem1_table.summed_em == 0.0
    assert too_few.mem1_table.summed_f1 == 0.0
    assert too_few.standard_qa.summed_em == 1.0
    assert too_few.standard_qa.summed_f1 == 1.0

    too_many = score_prediction("Beijing; Einstein; extra", gold)
    assert too_many.mem1_table.summed_em == 0.0
    assert too_many.mem1_table.summed_f1 == 0.0
    assert too_many.standard_qa.summed_em == 2.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} metric tests passed.")
