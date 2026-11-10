# SPDX-License-Identifier: Apache-2.0
"""``robot.stream_dual`` on a fleet: several arms' trajectories, started together, waited on once.

The single-world facade advances every arm's next waypoint in one action vector
(``GapEnvFacade.stream_dual_trajectory``). A fleet world runs motions as intents,
so the same contract is two things here, and both are threading contracts:

* the arms' trajectories are **drained together** -- one lock, one drain, one
  start step -- or two hands carrying one object leave out of step;
* the thread waiting on them stays **parked until the last one finishes** --
  a completion unparks its waiter outright, so without the group the first arm
  to finish would leave the world reading as running while it is still blocked
  on the second, and the barrier would never close again.

Nothing here needs a device: the fleet's loop is played by a thread that drains
and finishes motions the way :class:`~robosimstudio_policy_agent.fleet.stepper.StepService`
does.
"""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.core.bus import BusTimeout
from robosimstudio_policy_agent.fleet import IntentTable, JointTrajectory, MoveToJoints
from robosimstudio_policy_agent.fleet.barrier import ParkTable
from robosimstudio_policy_agent.fleet.world_facade import WorldFacade

TIER = "fast"
"""No GPU. Gates every commit."""

SETTLE = 5.0


def _path(n: int, value: float = 0.0) -> np.ndarray:
    return np.full((n, 6), value, dtype=np.float32) + np.arange(n, dtype=np.float32)[:, None]


def _until(predicate, timeout: float = SETTLE) -> bool:
    deadline = threading.Event()
    for _ in range(int(timeout / 0.002)):
        if predicate():
            return True
        deadline.wait(0.002)
    return bool(predicate())


class TestSubmitGroup(unittest.TestCase):
    def setUp(self) -> None:
        self.table = IntentTable(num_worlds=2, num_arms=2)

    def test_every_member_is_drained_in_one_drain(self):
        self.table.submit_group(
            1, [JointTrajectory(arm=0, waypoints=_path(3)), JointTrajectory(arm=1, waypoints=_path(3))]
        )
        drained = self.table.drain(step=7)
        self.assertEqual(sorted((world, arm) for world, arm, _i, _h in drained), [(1, 0), (1, 1)])
        self.assertEqual(self.table.drain(step=8), [])

    def test_a_group_holds_one_motion_per_slot(self):
        with self.assertRaises(ValueError):
            self.table.submit_group(0, [JointTrajectory(arm=0, waypoints=_path(2))] * 2)

    def test_a_world_outside_the_fleet_is_rejected(self):
        with self.assertRaises(IndexError):
            self.table.submit_group(2, [JointTrajectory(arm=0, waypoints=_path(2))])

    def test_a_member_supersedes_what_its_slot_held(self):
        old = self.table.submit(0, 1, MoveToJoints(arm=1, target=np.zeros(6, dtype=np.float32)))
        self.table.submit_group(0, [JointTrajectory(arm=1, waypoints=_path(2))])
        self.assertTrue(old.done)
        self.assertTrue(old.superseded)


class TestTheGroupKeepsItsWaiterParked(unittest.TestCase):
    """Only the last completion of a group may unpark the thread waiting on it."""

    def _waiter(self, parks, table, submit):
        done = threading.Event()

        def run() -> None:
            with parks.parked(0):
                for handle in submit():
                    handle.wait(SETTLE)
            done.set()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.assertTrue(_until(lambda: bool(parks.parked_mask()[0]) and table._pending), "the waiter parked")
        return thread, done

    def test_the_first_arm_to_finish_does_not_unpark_the_world(self):
        parks = ParkTable(1)
        table = IntentTable(num_worlds=1, num_arms=2, parks=parks)
        intents = [JointTrajectory(arm=0, waypoints=_path(2)), JointTrajectory(arm=1, waypoints=_path(4))]
        thread, done = self._waiter(parks, table, lambda: table.submit_group(0, intents))
        table.drain(step=0)

        table.finish(0, 0, step=3)
        self.assertFalse(done.wait(0.05), "the thread is still waiting on the second arm")
        self.assertTrue(parks.parked_mask()[0], "and the world still reads as parked, so the fleet can step it")

        table.finish(0, 1, step=7)
        self.assertTrue(done.wait(SETTLE))
        thread.join(SETTLE)
        self.assertFalse(parks.parked_mask()[0])

    def test_two_plain_submits_would_unpark_early(self):
        """Why the group exists: the same wait over two ungrouped handles reads as running mid-wait."""
        parks = ParkTable(1)
        table = IntentTable(num_worlds=1, num_arms=2, parks=parks)

        def submit():
            return [
                table.submit(0, 0, JointTrajectory(arm=0, waypoints=_path(2))),
                table.submit(0, 1, JointTrajectory(arm=1, waypoints=_path(4))),
            ]

        thread, done = self._waiter(parks, table, submit)
        table.drain(step=0)
        table.finish(0, 0, step=3)
        self.assertFalse(done.is_set())
        self.assertFalse(parks.parked_mask()[0], "unparked while still blocked: the barrier would deadlock here")
        table.finish(0, 1, step=7)
        thread.join(SETTLE)


class _Fleet:
    """The fleet's loop in miniature: drain what was submitted, then finish it."""

    def __init__(self, table: IntentTable, *, error_arm: int | None = None) -> None:
        self.table = table
        self.batches: list[list[tuple[int, object]]] = []
        self.parked_between: list[bool] = []
        self.error_arm = error_arm
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        step = 0
        while not self._stop.is_set():
            drained = self.table.drain(step=step)
            if not drained:
                threading.Event().wait(0.001)
                continue
            self.batches.append([(arm, intent) for _w, arm, intent, _h in drained])
            for i, (world, arm, intent, _handle) in enumerate(drained):
                step += 2 * len(intent.waypoints)
                error = RuntimeError(f"arm {arm} failed") if arm == self.error_arm else None
                self.table.finish(world, arm, step=step, error=error)
                if i < len(drained) - 1:
                    parks = self.table.parks
                    self.parked_between.append(bool(parks.parked_mask()[world]))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(SETTLE)


class TestWorldFacadeStreamsTogether(unittest.TestCase):
    """``WorldFacade.stream_dual_trajectory`` keeps the single-world contract."""

    def _facade(self, *, max_path: int = 512, error_arm: int | None = None):
        parks = ParkTable(1)
        table = IntentTable(num_worlds=1, num_arms=2, parks=parks)
        fleet = SimpleNamespace(
            max_steps=100,
            segments={"right": SimpleNamespace(max_path=max_path), "left": SimpleNamespace(max_path=max_path)},
        )
        bus = SimpleNamespace(parks=parks, close=lambda: None)
        facade = WorldFacade(0, fleet=fleet, bus=bus, intents=table, arm_names=["right", "left"])
        runner = _Fleet(table, error_arm=error_arm)
        self.addCleanup(runner.close)
        return facade, runner

    def test_both_arms_start_in_one_drain_and_short_tracks_hold_their_end(self):
        facade, fleet = self._facade()
        work, support = _path(5, 0.0), _path(3, 10.0)
        result = facade.stream_dual_trajectory({0: work.tolist(), 1: support.tolist()}, settle_tolerance=0.004)
        self.assertIsNone(result, "returns nothing, like the serial form")
        self.assertEqual(len(fleet.batches), 1, "one drain started both arms")
        got = dict(fleet.batches[0])
        self.assertEqual(sorted(got), [0, 1])
        for intent in got.values():
            self.assertIsInstance(intent, JointTrajectory)
            self.assertEqual(intent.steps_per_waypoint, 2)
            self.assertEqual(len(intent.waypoints), 5, "every walk is the longest one's length")
            self.assertEqual(intent.settle_tolerance, 0.004)
            self.assertEqual(intent.settle_max_steps, 60)
        np.testing.assert_allclose(got[0].waypoints, work)
        np.testing.assert_allclose(got[1].waypoints[:3], support)
        np.testing.assert_allclose(got[1].waypoints[3:], np.repeat(support[-1:], 2, axis=0))
        self.assertEqual(fleet.parked_between, [True], "the world stayed parked between the two completions")

    def test_settle_zero_waits_for_nothing_past_the_last_waypoint(self):
        facade, fleet = self._facade()
        facade.stream_dual_trajectory({0: _path(4).tolist()}, settle_max_steps=0)
        ((arm, intent),) = fleet.batches[0]
        self.assertEqual((arm, intent.settle_max_steps), (0, 0))

    def test_arm_ids_are_clamped_and_a_repeated_arm_keeps_the_later_track(self):
        facade, fleet = self._facade()
        facade.stream_dual_trajectory({1: _path(2, 1.0).tolist(), 7: _path(2, 7.0).tolist(), 0: []})
        ((arm, intent),) = fleet.batches[0]
        self.assertEqual(arm, 1)
        np.testing.assert_allclose(intent.waypoints, _path(2, 7.0))

    def test_no_waypoints_is_no_motion(self):
        facade, fleet = self._facade()
        facade.stream_dual_trajectory({0: [], 1: []})
        self.assertEqual(fleet.batches, [])

    def test_a_track_longer_than_the_driver_holds_is_cut_not_resampled(self):
        facade, fleet = self._facade(max_path=4)
        work = _path(9)
        facade.stream_dual_trajectory({0: work.tolist(), 1: _path(2, 50.0).tolist()}, settle_max_steps=30)
        pieces = [dict(batch) for batch in fleet.batches]
        self.assertEqual([len(p[0].waypoints) for p in pieces], [4, 4, 3], "every piece fits the driver")
        # Consecutive, each starting on the waypoint the last one ended on, and
        # every waypoint walked: nothing dropped, nothing resampled.
        walked = np.concatenate([pieces[0][0].waypoints] + [p[0].waypoints[1:] for p in pieces[1:]])
        np.testing.assert_allclose(walked, work)
        self.assertEqual([p[0].settle_max_steps for p in pieces], [0, 0, 30], "only the end settles")
        for piece in pieces:
            self.assertEqual(sorted(piece), [0, 1], "both arms in every piece")

    def test_an_arm_that_fails_raises_once_the_group_is_done(self):
        facade, fleet = self._facade(error_arm=0)
        with self.assertRaisesRegex(RuntimeError, "arm 0 failed"):
            facade.stream_dual_trajectory({0: _path(3).tolist(), 1: _path(3).tolist()})
        self.assertEqual(len(fleet.batches), 1)
        self.assertFalse(facade.bus.parks.parked_mask()[0])

    def test_a_closed_world_streams_nothing(self):
        facade, fleet = self._facade()
        facade.close()
        with self.assertRaises(BusTimeout):
            facade.stream_dual_trajectory({0: _path(3).tolist()})
        self.assertEqual(fleet.batches, [])


if __name__ == "__main__":
    unittest.main()
