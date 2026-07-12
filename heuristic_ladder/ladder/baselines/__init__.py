"""Learned/RL baselines run *inside* this harness (not rungs of the ladder).

Each baseline here reuses our retriever, our datasets, and our RunResult/metrics so
its numbers are produced under the *same* retrieval condition and scoring protocol
as the heuristic ladder. That is the "no-caveat tier" from the top-level README:
running the learned checkpoint against our bundled-pool retriever removes the
retrieval-scope confound because it is applied equally to both sides.
"""

from .mem1 import MEM1Baseline, MEM1_TASK_TEMPLATE

__all__ = ["MEM1Baseline", "MEM1_TASK_TEMPLATE"]
