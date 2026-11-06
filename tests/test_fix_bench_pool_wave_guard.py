# SPDX-License-Identifier: Apache-2.0
"""The wave guard: a wave its worker spoiled is run once more, and a graph that really fails scores what it scored.

The bad wave is a renderer that lost a material's colour: every world of its wave
aborts at the graph's first perception. Everything here goes through a real pool
of forked workers and the real :meth:`_FleetWorker.run`; only the runner is a
stand-in, and it decides from the build it is (the first, the re-run) and the job
what its worlds do.
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from robosimstudio.specs import JobSpec
from robosimstudio_bench.run import fleet as fleet_module
from robosimstudio_bench.run import pool
from robosimstudio_bench.run.pool import Watchdog

TIER = "fast"

_MARKS = "WAVE_GUARD_TEST_MARKS"
"""Where the stand-in counts its builds and attempts across processes."""

_ABORT = "PipelineError: workflow terminated at end node 'abort' (failure); first node error -- perceive: no cable"


def _next(what: str) -> int:
    """How many processes got here before this one, for ``what``."""
    marks = Path(os.environ[_MARKS])
    for count in range(1000):
        try:
            (marks / f"{what}.{count}").touch(exist_ok=False)
        except FileExistsError:
            continue
        return count
    raise AssertionError(what)


def _world(job: JobSpec, ep: int, world: int, *, aborted: bool, steps: int = 85) -> dict:
    row = {
        "task": job.task,
        "episode_id": int(ep),
        "success": not aborted,
        "reward": 0.0 if aborted else 1.0,
        "episode_length": steps if aborted else 900,
        "close_kind": "policy_request" if aborted else "success",
    }
    if aborted:
        row["graph_error"] = _ABORT
    return {**row, "world": world}


class _Runner:
    """``RunnerFleet`` for :meth:`_FleetWorker.run`; the job's ``fault`` says what each build's worlds do."""

    def __init__(self, job: JobSpec, verbose: bool = False, run_dir: Path | None = None) -> None:
        self.num_worlds = job.num_worlds
        self.rows: list[dict] = []
        self.summary = SimpleNamespace(waves=[])
        self.on_wave_tick = None
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.fault = dict(job.solution_kwargs).get("fault", "")
        self.build = _next(f"build-{job.name}")

    def open(self, job: JobSpec, episodes=None, task=None):
        self.job = job
        model = SimpleNamespace(device="cpu")
        pipeline = SimpleNamespace(captured=True, capture_error="", substeps=1, control_hz=60.0)
        return SimpleNamespace(model=model, pipeline=pipeline, close=lambda: None), SimpleNamespace(close=lambda: None)

    def run_work(self, _fleet, _policy, pairs) -> None:
        job, attempt = self.job, self.build
        _next(f"wave-{job.name}")
        fault = self.fault
        ids = [int(ep) for _document, ep in pairs]
        if fault == "rerun_raises" and attempt > 0:
            raise RuntimeError("the re-run's device fell over")
        if fault == "rerun_dies" and attempt > 0:
            os._exit(1)
        if fault == "rerun_hangs" and attempt > 0:
            while True:
                time.sleep(3600.0)
        bad_builds = {
            "first_build": attempt == 0,
            "rerun_raises": attempt == 0,
            "rerun_dies": attempt == 0,
            "rerun_hangs": attempt == 0,
            "recorded": attempt == 0,
            "always": True,
            "flaky": attempt == 0,
        }
        for world, ep in enumerate(ids):
            aborted = bad_builds.get(fault, False)
            steps = 85
            if fault == "late":
                aborted, steps = True, 8040
            elif fault == "one_survives":
                aborted = world > 0
            elif fault == "flaky" and attempt > 0:
                # Most of the worlds abort without any help from the renderer.
                aborted = world % 3 != 0
            row = _world(job, ep, world, aborted=aborted, steps=steps)
            self.rows.append(row)
            with (self.run_dir / "episode_results.jsonl").open("a") as results:
                results.write(json.dumps(row) + "\n")
            if fault == "recorded":
                folder = self.run_dir / f"{ep:04d}"
                folder.mkdir(parents=True)
                (folder / "meta.json").write_text(json.dumps({"attempt": attempt}))
        self.summary.waves.append({"wall_s": 10.0 + attempt})


def _run(jobs, watchdog: Watchdog, *, workers: int = 1):
    """``run_pooled`` on forked workers with the stand-in runner; ``(job name -> result, attempts, tmp)``."""
    tmp = tempfile.TemporaryDirectory()
    marks = Path(tmp.name) / "marks"
    marks.mkdir()
    patches = [
        mock.patch.object(pool.multiprocessing, "get_context", return_value=multiprocessing.get_context("fork")),
        mock.patch.object(fleet_module, "RunnerFleet", _Runner),
        mock.patch.object(fleet_module, "plan_mixed_waves", lambda work, _width: [work]),
        mock.patch.object(pool, "job_task", lambda job, name="": SimpleNamespace(full_name=name or job.task)),
        mock.patch.object(pool, "_POLL_S", 0.05),
        mock.patch.dict(os.environ, {_MARKS: str(marks)}),
    ]
    for patch in patches:
        patch.start()
    try:
        out = pool.run_pooled(jobs, root=tmp.name, workers=workers, stagger=0.0, verbose=False, watchdog=watchdog)
    except BaseException:
        tmp.cleanup()
        raise
    finally:
        for patch in reversed(patches):
            patch.stop()

    def attempts(job: JobSpec) -> int:
        return len(list(marks.glob(f"wave-{job.name}.*")))

    return out, attempts, tmp


def _job(fault: str = "", *, name: str = "j", worlds: int = 8, episodes: int = 16) -> JobSpec:
    return JobSpec(
        name=name, task="suite/t", num_worlds=worlds, episodes=(0, episodes), solution_kwargs=(("fault", fault),)
    )


def _scored(result) -> list[tuple]:
    """What a run's score is made of, world by world."""
    return sorted(
        (
            row["episode_id"],
            row["success"],
            row["reward"],
            row["episode_length"],
            row["close_kind"],
            row.get("graph_error"),
        )
        for row in result.rows
    )


_GUARD = Watchdog(build_s=30.0, step_s=30.0)


class TestAWaveItsWorkerSpoiledIsRunAgain(unittest.TestCase):
    def test_the_rerun_replaces_it_and_every_world_is_scored_once(self) -> None:
        job = _job("first_build")
        out, attempts, tmp = _run(job, _GUARD)
        with tmp:
            result = out["j"]
            self.assertEqual(sorted(row["episode_id"] for row in result.rows), list(range(16)))
            self.assertTrue(all(row["success"] for row in result.rows))
            self.assertEqual(attempts(job), 3)  # two waves, and the spoiled one again

    def test_the_provenance_says_what_happened(self) -> None:
        out, _attempts, tmp = _run(_job("first_build"), _GUARD)
        with tmp:
            result = out["j"]
            (retry,) = result.watchdog["wave_retries"]
            self.assertEqual((retry["outcome"], retry["stood"]), ("cleared", "rerun"))
            self.assertEqual(retry["first"]["aborted_early"], 8)
            self.assertEqual(retry["first"]["steps"], [85, 85])
            self.assertIn("no cable", retry["first"]["error"])
            self.assertEqual(retry["rerun"]["successes"], 8)
            self.assertEqual(sorted(ep for _task, ep in retry["episodes"]), list(range(8)))
            self.assertEqual(result.watchdog["kills"], [])
            meta = json.loads((Path(tmp.name) / "j" / "run_meta.json").read_text())
            self.assertEqual(meta["provenance"]["watchdog"]["wave_retries"][0]["outcome"], "cleared")
            # The first attempt's rows are not lost: they stay in its shard.
            shards = Path(tmp.name) / "j" / "shards"
            self.assertEqual(retry["shard"], "00")
            kept = (shards / "00" / "episode_results.jsonl").read_text().splitlines()
            self.assertEqual(len(kept), 8)
            self.assertTrue(all(json.loads(line)["episode_length"] == 85 for line in kept))
            self.assertTrue((shards / "00r1").is_dir())

    def test_the_discarded_attempts_time_is_not_counted(self) -> None:
        out, _attempts, tmp = _run(_job("first_build"), _GUARD)
        with tmp:
            # 11 s for each wave of the second build, and none of the first's 10 s.
            self.assertEqual(sorted(out["j"].wave_wall_s), [11.0, 11.0])

    def test_the_other_workers_are_left_alone(self) -> None:
        job = _job("first_build", episodes=48)
        out, _attempts, tmp = _run(job, _GUARD, workers=3)
        with tmp:
            result = out["j"]
            self.assertEqual(sorted(row["episode_id"] for row in result.rows), list(range(48)))
            self.assertTrue(all(row["success"] for row in result.rows))
            self.assertEqual(len(result.watchdog["wave_retries"]), 1)

    def test_a_recording_links_the_attempt_that_stood(self) -> None:
        out, _attempts, tmp = _run(_job("recorded", episodes=8), _GUARD)
        with tmp:
            result = out["j"]
            self.assertTrue(all(row["success"] for row in result.rows))
            root = Path(tmp.name) / "j" / "shards"
            for ep in range(8):
                self.assertEqual(json.loads((root / "00r1" / f"{ep:04d}" / "meta.json").read_text())["attempt"], 1)
                self.assertFalse((root / "00" / f"{ep:04d}").exists())
                self.assertTrue((root / "00" / "discarded" / f"{ep:04d}" / "meta.json").exists())


class TestAGraphThatReallyFailsScoresTheSame(unittest.TestCase):
    def test_a_graph_that_aborts_everywhere(self) -> None:
        job = _job("always", episodes=32)
        with_guard, attempts, tmp = _run(job, _GUARD)
        with tmp:
            without, _none, tmp_off = _run(job, Watchdog(build_s=30.0, step_s=30.0, wave_steps=0))
            with tmp_off:
                self.assertEqual(_scored(with_guard["j"]), _scored(without["j"]))
                self.assertFalse(without["j"].watchdog["wave_retries"])
                (retry,) = with_guard["j"].watchdog["wave_retries"]
                self.assertEqual((retry["outcome"], retry["stood"]), ("failed again", "first"))
                # One retry for the job, not one per wave: four waves, one extra.
                self.assertEqual(attempts(job), 5)

    def test_a_graph_that_aborts_most_worlds_keeps_the_first_attempt(self) -> None:
        # The re-run's worlds abort two in three: not a renderer's doing, and its
        # three survivors must not replace the first attempt's none.
        job = _job("flaky", worlds=9, episodes=9)
        out, _attempts, tmp = _run(job, _GUARD)
        with tmp:
            result = out["j"]
            self.assertEqual(sum(row["success"] for row in result.rows), 0)
            (retry,) = result.watchdog["wave_retries"]
            self.assertEqual((retry["outcome"], retry["stood"]), ("failed again", "first"))

    def test_the_job_that_failed_does_not_turn_off_the_next(self) -> None:
        broken, spoiled = _job("always", name="broken"), _job("first_build", name="spoiled")
        out, attempts, tmp = _run([broken, spoiled], _GUARD)
        with tmp:
            self.assertEqual(sum(row["success"] for row in out["broken"].rows), 0)
            self.assertEqual(attempts(broken), 3)
            self.assertEqual(sum(row["success"] for row in out["spoiled"].rows), 16)
            self.assertEqual(out["spoiled"].watchdog["wave_retries"][0]["outcome"], "cleared")


class TestWhatDoesNotQualify(unittest.TestCase):
    def _no_retry(self, job: JobSpec, successes: int) -> None:
        out, attempts, tmp = _run(job, _GUARD)
        with tmp:
            result = out["j"]
            self.assertEqual(result.watchdog["wave_retries"], [])
            self.assertEqual(sum(row["success"] for row in result.rows), successes)
            self.assertEqual(attempts(job), 2)

    def test_a_late_abort_is_the_graphs(self) -> None:
        self._no_retry(_job("late"), 0)

    def test_one_world_getting_through_clears_the_worker(self) -> None:
        self._no_retry(_job("one_survives"), 2)

    def test_a_narrow_wave_is_no_evidence(self) -> None:
        self._no_retry(_job("always", worlds=3, episodes=6), 0)

    def test_turned_off(self) -> None:
        out, attempts, tmp = _run(_job("always"), Watchdog(build_s=30.0, step_s=30.0, wave_steps=0))
        with tmp:
            self.assertEqual(attempts(_job("always")), 2)
            self.assertEqual(out["j"].watchdog["wave_retries"], [])

    def test_off_with_the_watchdog(self) -> None:
        out, attempts, tmp = _run(_job("always"), Watchdog(build_s=0.0, step_s=0.0))
        with tmp:
            self.assertEqual(attempts(_job("always")), 2)
            self.assertEqual(out["j"].watchdog, {"off": True})
            self.assertEqual(sum(row["success"] for row in out["j"].rows), 0)


class TestARerunThatGoesWrongLeavesTheFirstAttempt(unittest.TestCase):
    def _first_stands(self, fault: str, outcome: str, watchdog: Watchdog) -> None:
        job = _job(fault, episodes=8)
        out, _attempts, tmp = _run(job, watchdog)
        with tmp:
            result = out["j"]
            self.assertEqual(sorted(row["episode_id"] for row in result.rows), list(range(8)))
            self.assertTrue(all(row["episode_length"] == 85 for row in result.rows))
            (retry,) = result.watchdog["wave_retries"]
            self.assertEqual((retry["outcome"], retry["stood"]), (outcome, "first"))

    def test_it_raises(self) -> None:
        self._first_stands("rerun_raises", "failed again", _GUARD)

    def test_its_process_dies(self) -> None:
        self._first_stands("rerun_dies", "re-run died", _GUARD)

    def test_it_hangs_for_good(self) -> None:
        self._first_stands("rerun_hangs", "re-run hung", Watchdog(build_s=0.5, step_s=0.5, retries=0))


class TestTheRule(unittest.TestCase):
    def _rows(self, n: int, **fields) -> list[dict]:
        return [{"graph_error": _ABORT, "success": False, "episode_length": 85, **fields} for _ in range(n)]

    def test_every_world_early_and_by_a_graph(self) -> None:
        self.assertTrue(pool._wave_aborted(self._rows(15), 15, 500, 6))
        self.assertFalse(pool._wave_aborted(self._rows(15, episode_length=501), 15, 500, 6))
        self.assertFalse(pool._wave_aborted(self._rows(15, graph_error=""), 15, 500, 6))
        self.assertFalse(pool._wave_aborted(self._rows(15, success=True), 15, 500, 6))
        self.assertFalse(pool._wave_aborted([*self._rows(14), {"success": True, "episode_length": 900}], 15, 500, 6))

    def test_width(self) -> None:
        self.assertFalse(pool._wave_aborted(self._rows(5), 5, 500, 6))
        self.assertFalse(pool._wave_aborted(self._rows(10), 15, 500, 6), "rows missing: the wave raised part way")
        self.assertFalse(pool._wave_aborted([], 0, 500, 6))

    def test_settings(self) -> None:
        self.assertTrue(Watchdog().guards_waves)
        self.assertFalse(Watchdog(wave_steps=0).guards_waves)
        self.assertFalse(Watchdog(build_s=0.0, step_s=0.0).guards_waves)
        parsed = Watchdog.from_env("wave_steps=100,wave_worlds=3")
        self.assertEqual((parsed.wave_steps, parsed.wave_worlds), (100, 3))
        self.assertEqual(Watchdog.from_env("wave_steps=0").guards_waves, False)
        self.assertEqual(Watchdog().provenance()["wave_steps"], Watchdog.wave_steps)
        with self.assertRaises(ValueError):
            Watchdog.from_env("wave_step=100")

    def test_the_bench_flag(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "bench_run.py"
        spec = importlib.util.spec_from_file_location("bench_run_for_wave_guard_test", path)
        bench_run = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench_run)
        args = bench_run.create_parser().parse_args(["--task", "s/t", "--watchdog-wave-steps", "0"])
        self.assertFalse(bench_run.watchdog_from_args(args).guards_waves)


if __name__ == "__main__":
    unittest.main()
