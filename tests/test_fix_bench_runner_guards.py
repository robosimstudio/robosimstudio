# SPDX-License-Identifier: Apache-2.0
"""Documents refused where only one model is built, the pool's fleet kept current, and the seam check's depth."""

import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

TIER = "fast"


class TestDocumentsNeedAWave(unittest.TestCase):
    def test_run_pooled_refuses_documents_on_the_serial_worker(self):
        from robosimstudio.specs import JobSpec
        from robosimstudio_bench.run import pool

        job = JobSpec(name="j", task="s/a", episodes=(0, 2), documents=("s/a", "s/b"))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            pool.run_pooled(job, root=tmp, workers=1, verbose=False)

    def test_runner_eval_refuses_documents(self):
        from robosimstudio.specs import JobSpec
        from robosimstudio_bench.run.runner import RunnerEval

        job = JobSpec(name="j", task="s/a", episodes=(0, 2), documents=("s/a", "s/b"))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            RunnerEval(job, root=tmp, verbose=False)


class _Closeable:
    def __init__(self, name):
        self.name = name
        self.closed = 0
        self.model = SimpleNamespace(device="cpu")
        self.pipeline = SimpleNamespace(captured=True, capture_error="", substeps=1, control_hz=10.0)

    def close(self):
        self.closed += 1


class _FakeRunner:
    """Opens what it is asked for and, like ``run_work``, rebuilds on every call."""

    instances: ClassVar[list] = []

    def __init__(self, job, verbose=False, run_dir=None):
        self.num_worlds = 2
        self.rows: list = []
        self.summary = SimpleNamespace(waves=[])
        self.opened: list = []
        self._fleet = self._policy = None
        _FakeRunner.instances.append(self)

    def open(self, job=None, episodes=None, task=None):
        self.opened.append((episodes, task))
        n = len(self.opened)
        return _Closeable(f"fleet{n}"), _Closeable(f"policy{n}")

    def run_work(self, fleet, policy, work):
        fleet.close()
        policy.close()
        self._fleet, self._policy = self.open(episodes=[episode for _d, episode in work])
        return []


class TestFleetWorkerKeepsWhatIsOpen(unittest.TestCase):
    def test_built_for_the_first_wave_and_rebuilt_fleet_is_the_one_held(self):
        from robosimstudio.specs import JobSpec
        from robosimstudio_bench.run import fleet, pool

        job = JobSpec(name="j", task="s/a", episodes=(0, 2), num_worlds=2)
        _FakeRunner.instances = []
        with (
            mock.patch.object(fleet, "RunnerFleet", _FakeRunner),
            mock.patch.object(fleet, "plan_mixed_waves", lambda work, n: [[(doc, ("ep", i)) for doc, i in work]]),
            mock.patch.object(pool, "job_task", lambda job, name=None: "doc"),
        ):
            worker = pool._FleetWorker()
            worker.run(job, (("", 0), ("", 1)), Path("/nonexistent"), 0, queue.Queue(), 0)
            runner = _FakeRunner.instances[0]
            # Built once, for this unit's first wave rather than for episode 0.
            self.assertEqual(runner.opened[0], ([("ep", 0), ("ep", 1)], "doc"))
            live = runner._fleet, runner._policy
            self.assertIs(worker.fleet, live[0])
            self.assertIs(worker.policy, live[1])
            worker.close()
            self.assertEqual((live[0].closed, live[1].closed), (1, 1))


class TestReachesOut(unittest.TestCase):
    def test_climbing_past_the_root_is_outside_at_any_depth(self):
        from robosimstudio_bench.taskbuild.contract import reaches_out

        self.assertFalse(reaches_out(("t", "assets"), 2, "x"))
        self.assertTrue(reaches_out(("t", "assets"), 3, "x"))
        self.assertTrue(reaches_out(("t", "assets"), 4, "x"))
        self.assertTrue(reaches_out(("t", "assets"), 6, "x"))
        self.assertFalse(reaches_out(("t", "assets"), 4, "_shared.y"))


if __name__ == "__main__":
    unittest.main()
