# SPDX-License-Identifier: Apache-2.0
"""A wave words a failed workflow the way a single-world run does: "graph failed: ...".

``gap.execute`` does not raise for a workflow that ended at a failure node; the
failure is on its result, and the fleet's ``_record_result`` copies it into the
world's trace as ``graph_error``. The single-world policy's termination reason
reads that; the fleet's read only an exception raised by the graph thread, so
every aborted workflow in a wave was reported "graph finished" -- 0 of 180
fleet episodes on cable_routing/L1 against 8 of 90 serially, while 16 of those
180 rows carried the failure in their own ``graph_error``.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from robosimstudio_policy_agent.fleet.solution import SolutionGraphBatched

TIER = "fast"


def _state(error=None, graph_error=None):
    trace = {} if graph_error is None else {"graph_error": graph_error}
    return SimpleNamespace(error=error, trace=trace)


class TestTheTerminationReason(unittest.TestCase):
    def test_a_workflow_that_finished(self):
        self.assertEqual(SolutionGraphBatched._termination_reason(_state()), "graph finished")

    def test_a_workflow_that_ended_at_a_failure_node(self):
        error = "PipelineError: workflow terminated at end node 'abort' (failure)"
        self.assertEqual(SolutionGraphBatched._termination_reason(_state(graph_error=error)), f"graph failed: {error}")

    def test_a_thread_that_raised(self):
        reason = SolutionGraphBatched._termination_reason(_state(error=RuntimeError("boom"), graph_error="other"))
        self.assertEqual(reason, "graph failed: boom")


if __name__ == "__main__":
    unittest.main()
