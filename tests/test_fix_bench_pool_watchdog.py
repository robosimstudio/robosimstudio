# SPDX-License-Identifier: Apache-2.0
"""The pool's watchdog: a worker that hangs is killed and replaced, its episode re-run, and nothing outlives the pool.

The end-to-end cases run real worker processes -- forked rather than spawned, so
they inherit the fake build below -- and hang them on purpose: while building,
and at the third control step of an episode. A real pool then has to kill the
silent one, run its episode on a replacement, and still account for every
episode, which is the whole claim.
"""

from __future__ import annotations

import importlib.util
import multiprocessing
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from robosimstudio.specs import JobSpec
from robosimstudio_bench.run import pool
from robosimstudio_bench.run.pool import Watchdog

TIER = "fast"

_MARKS = "WATCHDOG_TEST_MARKS"
"""Directory the fake build and policy leave their hang markers in."""


def _hang_once(what: str) -> None:
    """Hang the first process to get here for ``what``; a marker lets every later one through."""
    if what.startswith("always"):
        while True:
            time.sleep(3600.0)
    marker = Path(os.environ[_MARKS]) / what
    try:
        marker.touch(exist_ok=False)
    except FileExistsError:
        return
    while True:
        time.sleep(3600.0)


class _FakeEnv:
    """Just enough of ``Env`` for :func:`~robosimstudio_bench.run.runner.rollout`: ten steps, observed."""

    max_steps = 10

    def __init__(self, job: JobSpec) -> None:
        self.task = SimpleNamespace(full_name=job.task)
        self.episode = SimpleNamespace(episode_id=-1, seed=0)
        self.step_count = 0
        self._observers: list = []

    def reset(self, episode_id: int = 0):
        self.episode = SimpleNamespace(episode_id=int(episode_id), seed=int(episode_id))
        self.step_count = 0
        return {}, {}

    def add_step_observer(self, observer) -> None:
        self._observers.append(observer)

    def remove_step_observer(self, observer) -> None:
        self._observers.remove(observer)

    def step(self, action, mode=None):
        self.step_count += 1
        for observer in list(self._observers):
            observer(None, action, None, 0.0, {})
        done = self.step_count >= 5
        return {}, 1.0, done, False, {"success": done, "close_kind": "success" if done else ""}

    def close(self) -> None:
        pass


class _FakePolicy:
    """Hangs inside its third action of the episode named by its job's ``solution_kwargs`` (``hang_step``)."""

    name = "fake"
    control_mode = ""

    def __init__(self, job: JobSpec, env: _FakeEnv) -> None:
        self.env = env
        self.hang = dict(job.solution_kwargs).get("hang_step")

    def reset(self, obs, info) -> None:
        pass

    def get_action(self, obs):
        if self.hang is not None and self.env.episode.episode_id == self.hang[0] and self.env.step_count == 2:
            _hang_once(self.hang[1])
        return 0

    def on_step(self, obs, reward, info) -> None:
        pass

    def row_extras(self) -> dict:
        return {}

    def close(self) -> None:
        pass


def _fake_build(job: JobSpec, root: Path, index: int, generation: int = 0):
    hang = dict(job.solution_kwargs).get("hang_build")
    if hang is not None:
        _hang_once(hang)
    env = _FakeEnv(job)
    build = {"task": job.task, "device": "cpu", "captured": True, "capture_error": "", "substeps": 1}
    return env, _FakePolicy(job, env), None, {**build, "control_hz": 50.0, "build_s": 0.0}


def _fake_wave(self, job: JobSpec, work, root, index, results, job_index) -> int:
    """``_FleetWorker.run`` for a fake wave: hangs where ``hang_wave`` says, else scores every world."""
    self.mark(pool._BUILDING)
    hang = dict(job.solution_kwargs).get("hang_wave")
    if hang is not None and any(int(ep) == hang[0] for _name, ep in work):
        _hang_once(hang[1])
    self.sent = []
    for _name, ep in work:
        self._tick(None, None, 1)
        row = {"task": job.task, "episode_id": int(ep), "success": True, "episode_length": 1, "close_kind": "success"}
        self.sent.append(row)
        results.put(("row", index, job_index, {**row, "worker": index}))
    return len(self.sent)


def _run(job: JobSpec, watchdog: Watchdog, *, workers: int = 2, collect=None):
    """``run_pooled`` on forked workers that build with :func:`_fake_build`; ``(results, marks dir)``."""
    tmp = tempfile.TemporaryDirectory()
    marks = Path(tmp.name) / "marks"
    marks.mkdir()
    patches = [
        mock.patch.object(pool.multiprocessing, "get_context", return_value=multiprocessing.get_context("fork")),
        mock.patch.object(pool, "_build", _fake_build),
        mock.patch.object(pool._FleetWorker, "run", _fake_wave),
        mock.patch.object(pool, "_POLL_S", 0.1),
        mock.patch.dict(os.environ, {_MARKS: str(marks)}),
    ]
    if collect is not None:
        patches.append(mock.patch.object(pool, "_collect", collect))
    for patch in patches:
        patch.start()
    try:
        out = pool.run_pooled(job, root=tmp.name, workers=workers, stagger=0.0, verbose=False, watchdog=watchdog)
    except BaseException:
        tmp.cleanup()
        raise
    finally:
        for patch in reversed(patches):
            patch.stop()
    return out[job.name], tmp


class TestAHungBuildIsKilledAndItsEpisodeReRun(unittest.TestCase):
    def test_every_episode_is_scored_once(self) -> None:
        job = JobSpec(name="j", task="suite/t", episodes=(0, 4), solution_kwargs=(("hang_build", "build"),))
        result, tmp = _run(job, Watchdog(build_s=1.0, step_s=1.0))
        with tmp:
            self.assertEqual(sorted(row["episode_id"] for row in result.rows), [0, 1, 2, 3])
            self.assertTrue(all(row["close_kind"] == "success" for row in result.rows))
            kills = result.watchdog["kills"]
            self.assertEqual([(kill["phase"], kill["action"]) for kill in kills], [("building", "retried")])
            # The replacement wrote into a shard of its own.
            shards = sorted(p.name for p in (Path(tmp.name) / "j" / "shards").iterdir())
            self.assertIn(f"{kills[0]['worker']:02d}r1", shards)


class TestAHangMidEpisodeIsKilledAndTheEpisodeReRun(unittest.TestCase):
    def test_the_episode_is_rerun_from_its_start(self) -> None:
        job = JobSpec(name="j", task="suite/t", episodes=(0, 3), solution_kwargs=(("hang_step", (1, "step")),))
        result, tmp = _run(job, Watchdog(build_s=5.0, step_s=1.0))
        with tmp:
            rows = {row["episode_id"]: row for row in result.rows}
            self.assertEqual(sorted(rows), [0, 1, 2])
            self.assertEqual(rows[1]["episode_length"], 5)
            self.assertEqual([kill["phase"] for kill in result.watchdog["kills"]], ["stepping"])


class TestAUnitThatHangsEveryTimeBecomesAnErrorRow(unittest.TestCase):
    def test_after_its_retry_the_rest_still_run(self) -> None:
        job = JobSpec(name="j", task="suite/t", episodes=(0, 3), solution_kwargs=(("hang_step", (2, "always")),))
        result, tmp = _run(job, Watchdog(build_s=5.0, step_s=1.0, retries=1))
        with tmp:
            rows = {row["episode_id"]: row for row in result.rows}
            self.assertEqual(sorted(rows), [0, 1, 2])
            self.assertEqual(rows[2]["close_kind"], "watchdog")
            self.assertIn("2 attempt(s)", rows[2]["error"])
            self.assertEqual([kill["action"] for kill in result.watchdog["kills"]], ["retried", "error rows"])
            self.assertEqual(tuple(result.unrun), ())


class TestAHungWaveIsOneUnit(unittest.TestCase):
    def test_a_killed_wave_reruns_every_world_in_it(self) -> None:
        job = JobSpec(
            name="j", task="suite/t", num_worlds=2, episodes=(0, 4), solution_kwargs=(("hang_wave", (2, "w")),)
        )
        result, tmp = _run(job, Watchdog(build_s=1.0, step_s=0.5))
        with tmp:
            self.assertEqual(sorted(row["episode_id"] for row in result.rows), [0, 1, 2, 3])
            (kill,) = result.watchdog["kills"]
            self.assertEqual(sorted(ep for _task, ep in kill["episodes"]), [2, 3])
            # A wave gets the larger budget whatever it is doing.
            self.assertEqual(kill["budget_s"], 1.0)

    def test_a_wave_out_of_retries_is_an_error_row_per_world(self) -> None:
        job = JobSpec(
            name="j", task="suite/t", num_worlds=2, episodes=(0, 4), solution_kwargs=(("hang_wave", (0, "always")),)
        )
        result, tmp = _run(job, Watchdog(build_s=0.5, step_s=0.5, retries=1))
        with tmp:
            rows = {row["episode_id"]: row["close_kind"] for row in result.rows}
            self.assertEqual(rows, {0: "watchdog", 1: "watchdog", 2: "success", 3: "success"})


class TestNothingOutlivesAPoolThatEndsEarly(unittest.TestCase):
    def test_a_hung_worker_is_killed_on_the_way_out(self) -> None:
        def interrupted(procs, *args, **kwargs):
            # The workers are up and one of them is hung building; then Ctrl-C.
            time.sleep(1.0)
            raise KeyboardInterrupt

        job = JobSpec(name="j", task="suite/t", episodes=(0, 2), solution_kwargs=(("hang_build", "always"),))
        started = time.monotonic()
        with self.assertRaises(KeyboardInterrupt):
            _run(job, Watchdog(build_s=0.0, step_s=0.0), collect=interrupted)
        self.assertEqual(multiprocessing.active_children(), [])
        self.assertLess(time.monotonic() - started, 60.0)


class TestBudgets(unittest.TestCase):
    def test_build_step_bus_timeout_retry_and_fleet(self) -> None:
        watchdog = Watchdog(build_s=900.0, step_s=300.0)
        serial = JobSpec(name="j", task="t")
        graph = JobSpec(name="j", task="t", solution_kwargs=(("bus_timeout", 600.0),))
        fleet = JobSpec(name="j", task="t", num_worlds=4)
        self.assertEqual(watchdog.budget(serial, pool._BUILDING, 0), 900.0)
        self.assertEqual(watchdog.budget(serial, pool._BUILDING, 1), 1800.0)
        self.assertEqual(watchdog.budget(serial, pool._STEPPING, 0), 300.0)
        self.assertEqual(watchdog.budget(graph, pool._STEPPING, 0), 720.0)
        self.assertEqual(watchdog.budget(serial, pool._IDLE, 0), 0.0)
        # A wave's resets and rebuilds have no stamp of their own.
        self.assertEqual(watchdog.budget(fleet, pool._STEPPING, 0), 900.0)
        self.assertEqual(Watchdog(build_s=0.0, step_s=300.0).budget(serial, pool._BUILDING, 0), 0.0)


class TestTheEnvironmentAndTheFlags(unittest.TestCase):
    def test_from_env(self) -> None:
        self.assertEqual(Watchdog.from_env(""), Watchdog())
        self.assertFalse(Watchdog.from_env("off").enabled)
        self.assertEqual(Watchdog.from_env("build=120, step=60,retries=2"), Watchdog(120.0, 60.0, 2))
        with self.assertRaisesRegex(ValueError, pool.WATCHDOG_ENV):
            Watchdog.from_env("bulid=120")

    def test_the_test_hang_names_one_worker_and_one_attempt(self) -> None:
        with mock.patch.dict(os.environ, {pool.TEST_HANG_ENV: "build:1"}):
            self.assertEqual(pool._test_hang(1, 0), "build")
            self.assertEqual(pool._test_hang(1, 1), "")
            self.assertEqual(pool._test_hang(0, 0), "")
        with mock.patch.dict(os.environ, {pool.TEST_HANG_ENV: "step:0:always"}):
            self.assertEqual(pool._test_hang(0, 3), "step")

    def test_bench_run_flags(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "bench_run.py"
        spec = importlib.util.spec_from_file_location("bench_run_for_watchdog_test", path)
        bench_run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench_run)
        parser = bench_run.create_parser()

        def parsed(*flags):
            return bench_run.watchdog_from_args(parser.parse_args(["--task", "s/t", *flags]))

        with mock.patch.dict(os.environ, {pool.WATCHDOG_ENV: "step=50"}):
            self.assertIsNone(parsed())
            self.assertEqual(parsed("--watchdog-build", "120"), Watchdog(120.0, 50.0, 1))
        self.assertFalse(parsed("--no-watchdog").enabled)


if __name__ == "__main__":
    unittest.main()
