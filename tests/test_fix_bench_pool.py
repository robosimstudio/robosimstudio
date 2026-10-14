# SPDX-License-Identifier: Apache-2.0
"""The process pool: thread caps put back, every episode linked, batched rates and errors."""

import os
import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

TIER = "fast"

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class _FakeProcess:
    """Records the thread caps a spawned child would inherit, and reports done."""

    seen: ClassVar[list[dict[str, str | None]]] = []

    def __init__(self, *, target, args, kwargs, name):
        self.results = args[4]
        self.index = args[0]

    def start(self):
        _FakeProcess.seen.append({name: os.environ.get(name) for name in _THREAD_VARS})
        self.results.put(("done", self.index, 0, 0.0))

    def is_alive(self):
        return False

    def join(self, timeout=None):
        pass


class _FakeContext:
    Queue = queue.Queue
    Process = _FakeProcess


class TestThreadCapsArePutBack(unittest.TestCase):
    def test_children_see_the_cap_and_the_parent_gets_its_own_back(self):
        from robosimstudio.specs import JobSpec
        from robosimstudio_bench.run import pool

        before = {name: os.environ.get(name) for name in _THREAD_VARS}
        try:
            os.environ["OMP_NUM_THREADS"] = "7"
            os.environ.pop("MKL_NUM_THREADS", None)
            _FakeProcess.seen = []
            with (
                tempfile.TemporaryDirectory() as tmp,
                mock.patch.object(pool.multiprocessing, "get_context", return_value=_FakeContext()),
            ):
                pool.run_pooled(
                    JobSpec(name="j", task="t", episodes=(0, 2)),
                    root=tmp,
                    workers=2,
                    threads_per_worker=1,
                    verbose=False,
                )
            self.assertEqual([seen["OMP_NUM_THREADS"] for seen in _FakeProcess.seen], ["1", "1"])
            self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "7")
            self.assertNotIn("MKL_NUM_THREADS", os.environ)
        finally:
            for name, value in before.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


class TestEveryEpisodeIsLinked(unittest.TestCase):
    def test_one_failed_link_does_not_drop_the_rest(self):
        from robosimstudio.specs import JobSpec
        from robosimstudio_bench.run import pool

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "j"
            for shard in ("00", "01"):
                for episode in ("ep0", "ep1"):
                    folder = root / "shards" / shard / f"{shard}_{episode}"
                    folder.mkdir(parents=True)
                    (folder / "meta.json").write_text("{}")
            real = Path.symlink_to
            calls = []

            def flaky(self, target, target_is_directory=False):
                calls.append(self)
                if len(calls) == 1:
                    raise OSError(28, "No space left (simulated)")
                return real(self, target, target_is_directory)

            result = pool.PoolResult(job=JobSpec(name="j", task="t"), root=root)
            with mock.patch.object(Path, "symlink_to", flaky), mock.patch("builtins.print"):
                pool._link_episodes(result)
            linked = sorted(p.name for p in root.iterdir() if p.is_symlink())
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(linked), 3)


class TestBatchedPooledStepRate(unittest.TestCase):
    def test_overlapping_rows_are_not_summed(self):
        from robosimstudio.specs import JobSpec
        from robosimstudio_bench.run.pool import PoolResult

        # One worker, one wave of four worlds that each stepped 100 in the same 10 s.
        rows = [{"episode_id": i, "episode_length": 100, "wall_s": 10.0} for i in range(4)]
        job = JobSpec(name="j", task="t", num_worlds=4)
        batched = PoolResult(job=job, root=Path("."), rows=rows, workers=1, wave_wall_s=[10.0])
        self.assertAlmostEqual(batched.steps_per_s_stepping, 40.0)
        # A serial job keeps the inherited, per-episode sum.
        serial = PoolResult(job=JobSpec(name="j", task="t"), root=Path("."), rows=rows, workers=1)
        self.assertAlmostEqual(serial.steps_per_s_stepping, 10.0)


def _job(**kw):
    from robosimstudio.specs import JobSpec

    return JobSpec(name="j", task="t", num_worlds=2, episodes=(0, 2), **kw)


class TestABatchedUnitThatRaises(unittest.TestCase):
    def test_partial_rows_are_sent_and_the_rest_are_errors(self):
        from robosimstudio_bench.run import pool

        job = _job()
        sent_row = {"task": "suite/t", "episode_id": 0, "success": True, "episode_length": 5}

        class Runner:
            def __init__(self):
                self.rows = []
                self.summary = SimpleNamespace(waves=[])

            def run_work(self, fleet, policy, pairs):
                self.rows.append(sent_row)
                raise RuntimeError("wave blew up")

        work, results = queue.Queue(), queue.Queue()
        work.put((0, (("", 0), ("", 1))))
        work.put((0, (("", 0), ("", 1))))
        work.put(None)
        built = []

        def prebuilt(self, job_, *a):
            built.append(1)
            self.key = pool._env_key(job_)
            self.runner = Runner()
            self.task, self.job, self.fleet, self.policy = SimpleNamespace(full_name="suite/t"), job_, None, None
            self._docs = {}

        real_run = pool._FleetWorker.run

        def run(self, job_, work_, root, index, results_, job_index):
            if self.key != pool._env_key(job_):
                prebuilt(self, job_)
            return real_run(self, job_, work_, root, index, results_, job_index)

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(pool._FleetWorker, "run", run),
            mock.patch.object(pool, "job_task", return_value=SimpleNamespace(full_name="suite/t")),
        ):
            pool._worker(0, [job], tmp, work, results, device="", stagger=0.0, preload="")
        messages = []
        while not results.empty():
            messages.append(results.get())
        kinds = [m[0] for m in messages]
        self.assertNotIn("error", kinds)
        rows = [m[3] for m in messages if m[0] == "row"]
        # Both units ran (the worker went on after the first raised), each
        # sending its scored row and an error row for the episode it lost.
        self.assertEqual(len(built), 2)
        self.assertEqual(
            sorted((row["episode_id"], row.get("close_kind", "")) for row in rows),
            [(0, ""), (0, ""), (1, "error"), (1, "error")],
        )
        self.assertEqual(messages[-1][:3], ("done", 0, 4))


class TestAWaveWithANonPositiveBudget(unittest.TestCase):
    def test_is_refused_rather_than_closing_nothing(self):
        from robosimstudio.specs import JobSpec
        from robosimstudio_bench.run.fleet import RunnerFleet

        with tempfile.TemporaryDirectory() as tmp:
            runner = RunnerFleet(JobSpec(name="j", task="t", num_worlds=2, max_steps=-5), verbose=False, run_dir=tmp)
            fleet, policy = mock.MagicMock(), mock.MagicMock()
            episode = SimpleNamespace(episode_id=0, seed=0)
            with mock.patch.object(RunnerFleet, "_evaluator", return_value=None):
                with self.assertRaises(ValueError):
                    runner._run_wave(fleet, policy, 0, [(SimpleNamespace(max_steps=50), episode)])
            policy.act.assert_not_called()


if __name__ == "__main__":
    unittest.main()
