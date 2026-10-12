# SPDX-License-Identifier: Apache-2.0
"""The golden-trace machinery: what it catches, and what it forgives.

A safety net nobody has watched catch something is a safety net nobody should
trust, so most of this file is mutations: take a recorded episode, change one
thing about the evaluator's behaviour, and assert the comparison notices. Each
mutation corresponds to a mistake a real port could make.

The forgiveness half matters just as much. Identical code has been measured
scoring 0.84 and 0.90 on consecutive folds of one towel, so a gate that demands
exact equality everywhere would be red on every run and would be ignored within a
week. The bands are what keep it honest -- and the fact that they collapse to
exact equality when repeats agree is what keeps them from being an excuse.
"""

from __future__ import annotations

import unittest

from robosimstudio.record.trace import (
    MIN_REPEATS,
    TraceRecorder,
    bands_from_repeats,
    compare,
    read_summary,
    summarise,
    write_summary,
)

TIER = "fast"
"""No GPU. Gates every commit."""


class FakeEnv:
    """Just the two attributes the recorder reads."""

    def __init__(self) -> None:
        self.step_count = 0
        self.sim_time = 0.0

    def advance(self, dt: float = 0.1) -> None:
        self.step_count += 1
        self.sim_time += dt


def record(script: list[tuple[bool, float, dict]]) -> TraceRecorder:
    """Drive a recorder through a scripted episode, as ``rollout`` would."""
    recorder = TraceRecorder()
    env = FakeEnv()
    recorder.begin(episode=None, env=env)
    for success, reward, metrics in script:
        env.advance()
        recorder.step(env, None, {}, reward, {"success": success, "reward": reward, "metrics": metrics})
    recorder.end(env, {"close_kind": "success" if script and script[-1][0] else None})
    return recorder


def episode(turns: list[float], succeed_from: int = 99) -> list[tuple[bool, float, dict]]:
    """A wrap-shaped episode: a rising measure and a verdict that follows it."""
    return [
        (i >= succeed_from, min(1.0, t / 0.5), {"turns": t, "released": i >= succeed_from}) for i, t in enumerate(turns)
    ]


class TestTheRecorderIsUnobtrusive(unittest.TestCase):
    """It reads what the step produced and asks the environment for nothing."""

    def test_it_captures_one_row_per_step(self) -> None:
        recorder = record(episode([0.0, 0.2, 0.4, 0.6]))
        self.assertEqual(len(recorder.rows), 4)
        self.assertTrue(recorder.enabled)

    def test_a_broken_metric_disables_rather_than_ending_the_run(self) -> None:
        """Losing a run because the evidence-gathering broke is the wrong trade."""
        recorder = TraceRecorder()
        env = FakeEnv()
        recorder.begin(episode=None, env=env)
        recorder.step(env, None, {}, 0.0, {"success": False, "metrics": {"bad": {"nested": 1}}})
        self.assertFalse(recorder.enabled)
        self.assertIn("not a scalar", recorder.failures[0])

    def test_end_reports_what_it_captured(self) -> None:
        recorder = record(episode([0.1, 0.2]))
        row = recorder.end(FakeEnv(), {"close_kind": "truncated"})
        self.assertEqual(row["trace_rows"], 2)
        self.assertTrue(row["trace_enabled"])


class TestTheSummaryCapturesTheShape(unittest.TestCase):
    def test_it_records_the_metric_surface(self) -> None:
        got = summarise(record(episode([0.1, 0.3])).rows)
        self.assertEqual(got.metric_keys, ("released", "turns"))

    def test_it_separates_numeric_and_boolean_metrics(self) -> None:
        got = summarise(record(episode([0.1, 0.3])).rows)
        self.assertIn("turns", got.numeric)
        self.assertIn("released", got.boolean)

    def test_it_finds_the_first_success_not_just_the_last(self) -> None:
        got = summarise(record(episode([0.1, 0.3, 0.6, 0.7], succeed_from=2)).rows)
        self.assertEqual(got.success_first_step, 3, "steps are 1-based in the env's own counter")
        self.assertTrue(got.success)

    def test_it_records_where_a_measure_peaked(self) -> None:
        """A peak re-expression moves argmax and max and leaves final alone."""
        got = summarise(record(episode([0.1, 0.9, 0.2])).rows)
        self.assertAlmostEqual(got.numeric["turns"]["max"], 0.9)
        self.assertAlmostEqual(got.numeric["turns"]["final"], 0.2)
        self.assertEqual(got.numeric["turns"]["argmax_step"], 2)


class TestItCatchesTheMistakesAPortWouldMake(unittest.TestCase):
    """The mutation suite. Each is a change a re-expressed evaluator could make."""

    def setUp(self) -> None:
        self.want = summarise(record(episode([0.1, 0.3, 0.55, 0.6], succeed_from=2)).rows)

    def test_it_catches_a_dropped_metric(self) -> None:
        """A changed metric surface is the benchmark moving, never a band."""
        thinner = [(s, r, {"turns": m["turns"]}) for s, r, m in episode([0.1, 0.3, 0.55, 0.6], succeed_from=2)]
        diff = compare(summarise(record(thinner).rows), self.want)
        self.assertFalse(diff.ok)
        self.assertTrue(any("metric surface" in h for h in diff.hard))

    def test_it_catches_a_flipped_verdict(self) -> None:
        diff = compare(summarise(record(episode([0.1, 0.2, 0.2, 0.2])).rows), self.want)
        self.assertTrue(any("success" in h for h in diff.hard))

    def test_it_catches_a_latch_that_fires_at_the_wrong_time(self) -> None:
        """The sharpest probe: the terminal verdict is identical and only the
        first-success step moved, which is exactly what a mis-ported ``once`` or
        ``hold_for`` does."""
        late = summarise(record(episode([0.1, 0.3, 0.55, 0.6], succeed_from=3)).rows)
        diff = compare(late, self.want)
        self.assertFalse(diff.ok)
        self.assertTrue(any("success_first_step" in b for b in diff.banded))

    def test_it_catches_a_reward_that_ends_right_and_travels_wrong(self) -> None:
        """Swapping a progress term for a satisfied-share gets the ending right."""
        flat = [(s, 1.0 if s else 0.0, m) for s, _, m in episode([0.1, 0.3, 0.55, 0.6], succeed_from=2)]
        diff = compare(summarise(record(flat).rows), self.want)
        self.assertTrue(any("reward_auc" in b for b in diff.banded))

    def test_it_catches_a_flag_that_flickers(self) -> None:
        """A debounce or hold-duration change leaves the ending identical and
        doubles the transition count."""
        script = episode([0.1, 0.3, 0.55, 0.6], succeed_from=2)
        flickery = [(s, r, {**m, "released": bool(i % 2)}) for i, (s, r, m) in enumerate(script)]
        diff = compare(summarise(record(flickery).rows), self.want)
        self.assertTrue(any("n_transitions" in b for b in diff.banded))

    def test_an_identical_rerun_is_clean(self) -> None:
        """The control. A gate that fails on an unchanged run is worthless."""
        again = summarise(record(episode([0.1, 0.3, 0.55, 0.6], succeed_from=2)).rows)
        diff = compare(again, self.want)
        self.assertTrue(diff.ok, diff.report())


class TestBandsAreMeasuredNotChosen(unittest.TestCase):
    def test_identical_repeats_collapse_to_an_exact_gate(self) -> None:
        """The property that stops bands being an excuse. Where the physics is
        deterministic -- every rigid suite, anything under an idle policy -- the
        measured band has zero width and the gate is exact equality."""
        repeats = [summarise(record(episode([0.1, 0.3, 0.6])).rows) for _ in range(5)]
        bands = bands_from_repeats(repeats)
        for name, (low, high) in bands.items():
            with self.subTest(field=name):
                self.assertEqual((low, high), (0.0, 0.0))

    def test_a_noisy_field_gets_room_and_a_stable_one_does_not(self) -> None:
        """The wrap's own shape: turns vary run to run, the metric surface does not."""
        repeats = [summarise(record(episode([0.1, 0.3, t])).rows) for t in (0.55, 0.74, 0.96)]
        bands = bands_from_repeats(repeats)
        self.assertGreater(bands["numeric.turns.final"][1], 0.0, "a spread field gets room")
        self.assertEqual(bands["episode_length"], (0.0, 0.0), "a stable field stays exact")

    def test_a_band_forgives_drift_inside_it_and_reports_it_anyway(self) -> None:
        """Inside the band is a pass, but a reviewer should still see the move."""
        want = summarise(record(episode([0.1, 0.3, 0.55])).rows)
        got = summarise(record(episode([0.1, 0.3, 0.60])).rows)
        bands = bands_from_repeats([want, summarise(record(episode([0.1, 0.3, 0.96])).rows)])
        diff = compare(got, want, bands)
        self.assertTrue(diff.ok, diff.report())
        self.assertTrue(diff.within, "drift inside a band is reported, not hidden")

    def test_one_run_measures_no_band(self) -> None:
        """Two samples give a range, not a spread; one gives nothing at all."""
        self.assertEqual(bands_from_repeats([summarise(record(episode([0.1])).rows)]), {})

    def test_the_repeat_floor_is_stated(self) -> None:
        """A guard against the mistake this project already made once.

        A variant was recorded twice, agreed exactly, and was reported as
        deterministic. A later pair of runs of the *same* variant disagreed on
        three fields -- GPU-atomic nondeterminism does not have to show up in
        every pair. Two agreeing samples are not evidence of anything, and the
        floor exists so a campaign cannot quietly be run below it again.
        """
        self.assertGreaterEqual(MIN_REPEATS, 5)


class TestGoldensRoundTrip(unittest.TestCase):
    def test_a_summary_survives_the_trip_to_disk(self) -> None:
        import tempfile

        want = summarise(
            record(episode([0.1, 0.3, 0.6], succeed_from=1)).rows, task="cable_route/port_easy", solution="routine"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/wrap.json"
            write_summary(path, want)
            got = read_summary(path)
        self.assertEqual(got.to_dict(), want.to_dict())
        self.assertTrue(compare(got, want).ok)


if __name__ == "__main__":
    unittest.main()
