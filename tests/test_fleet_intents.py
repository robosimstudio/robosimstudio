# SPDX-License-Identifier: Apache-2.0
"""The batched GAP surfaces: motions, per-world buses, and who wakes whom.

These are threading contracts, so they are tested as threading contracts -- a
graph thread that blocks on a motion has to be woken by exactly one thing, and
the ways it can be woken (completion, supersession, the world ending, the fleet
closing) are what a wave's correctness rests on. Nothing here needs a device.
"""

from __future__ import annotations

import dataclasses
import threading
import unittest

import numpy as np

from robosimstudio.core.bus import BusTimeout, SimSnapshot
from robosimstudio_policy_agent.fleet import (
    GripperMove,
    IntentHandle,
    IntentTable,
    JointTrajectory,
    MoveToJoints,
    ServiceQueue,
    SnapshotTable,
    StepWaiter,
    WorldBus,
)
from robosimstudio_policy_agent.fleet.world_bus import _Request

TIER = "fast"
"""No GPU. Gates every commit."""


def _move(arm: int = 0, value: float = 0.0) -> MoveToJoints:
    return MoveToJoints(arm=arm, target=np.full(6, value, dtype=np.float32))


class TestIntentHandle(unittest.TestCase):
    """What a node's thread sees when its motion ends."""

    def test_completion_releases_the_waiter(self):
        handle = IntentHandle()
        threading.Timer(0.01, handle.complete, kwargs={"steps": 7}).start()
        handle.wait(timeout=5.0)
        self.assertTrue(handle.done)
        self.assertEqual(handle.steps, 7)

    def test_an_error_is_re_raised_in_the_waiting_thread(self):
        """How a servo that cannot reach its goal becomes a node error."""
        handle = IntentHandle()
        handle.complete(error=RuntimeError("unreachable"))
        with self.assertRaises(RuntimeError):
            handle.wait(timeout=1.0)

    def test_being_superseded_is_not_an_error(self):
        """The non-blocking pose command relies on replacement returning."""
        handle = IntentHandle()
        handle.complete(superseded=True)
        handle.wait(timeout=1.0)
        self.assertTrue(handle.superseded)

    def test_completion_is_idempotent(self):
        handle = IntentHandle()
        handle.complete(steps=3)
        handle.complete(error=RuntimeError("late"), steps=99)
        handle.wait(timeout=1.0)
        self.assertEqual(handle.steps, 3)

    def test_waiting_past_the_deadline_raises(self):
        with self.assertRaises(TimeoutError):
            IntentHandle().wait(timeout=0.01)


class TestIntentTable(unittest.TestCase):
    """Per-(world, arm) slots, and who gets woken when one is replaced."""

    def setUp(self):
        self.table = IntentTable(num_worlds=4, num_arms=2)

    def test_a_submitted_motion_is_drained_once(self):
        handle = self.table.submit(2, 0, _move())
        drained = self.table.drain()
        self.assertEqual(len(drained), 1)
        world, arm, intent, drained_handle = drained[0]
        self.assertEqual((world, arm), (2, 0))
        self.assertIs(drained_handle, handle)
        self.assertIsInstance(intent, MoveToJoints)
        self.assertEqual(self.table.drain(), [])

    def test_two_arms_of_one_world_hold_separate_slots(self):
        """A bimanual graph drives a support arm while the work arm moves."""
        left, right = self.table.submit(0, 0, _move(0)), self.table.submit(0, 1, _move(1))
        self.assertEqual(len(self.table.drain()), 2)
        self.assertFalse(left.done)
        self.assertFalse(right.done)

    def test_resubmitting_supersedes_and_wakes_the_displaced_waiter(self):
        first = self.table.submit(1, 0, _move(value=1.0))
        second = self.table.submit(1, 0, _move(value=2.0))
        first.wait(timeout=1.0)  # returns rather than raising
        self.assertTrue(first.superseded)
        self.assertFalse(second.done)
        (drained,) = self.table.drain()
        self.assertEqual(float(drained[2].target[0]), 2.0)

    def test_draining_makes_a_motion_active_and_finishing_wakes_it(self):
        """Draining *is* starting: there is no window in which the table has
        forgotten a handle that nothing can complete."""
        handle = self.table.submit(3, 0, _move())
        self.table.drain(step=10)
        self.assertEqual(len(self.table.active()), 1)
        self.assertIsInstance(self.table.running(3, 0), MoveToJoints)
        np.testing.assert_array_equal(self.table.busy_worlds(), [False, False, False, True])

        self.table.finish(3, 0, step=25)
        handle.wait(timeout=1.0)
        self.assertEqual(handle.steps, 15)
        self.assertEqual(self.table.active(), [])
        self.assertIsNone(self.table.running(3, 0))

    def test_draining_over_a_running_motion_supersedes_it(self):
        first = self.table.submit(0, 0, _move(value=1.0))
        self.table.drain(step=0)
        self.table.submit(0, 0, _move(value=2.0))
        self.table.drain(step=5)
        first.wait(timeout=1.0)
        self.assertTrue(first.superseded)
        self.assertEqual(len(self.table.active()), 1)

    def test_finishing_with_an_error_reaches_the_waiting_thread(self):
        handle = self.table.submit(0, 0, _move())
        self.table.drain(step=0)
        self.table.finish(0, 0, error=RuntimeError("could not reach"), step=5)
        with self.assertRaises(RuntimeError):
            handle.wait(timeout=1.0)

    def test_cancelling_a_world_releases_its_threads_and_leaves_others(self):
        """An episode ending must not leave a blocked node waiting out its
        deadline -- and must not disturb the worlds still running."""
        running = self.table.submit(1, 0, _move())
        self.table.drain(step=0)
        pending = self.table.submit(1, 1, _move())  # submitted but not yet drained
        other = self.table.submit(2, 0, _move())

        self.table.cancel_world(1, BusTimeout("world finished"))
        for handle in (running, pending):
            with self.assertRaises(BusTimeout):
                handle.wait(timeout=1.0)
        self.assertFalse(other.done)
        self.assertEqual(self.table.active(), [])
        self.assertEqual([d[:2] for d in self.table.drain()], [(2, 0)])

    def test_cancel_all_releases_everything(self):
        handles = [self.table.submit(w, 0, _move()) for w in range(4)]
        self.table.cancel_all(BusTimeout("the run closed"))
        for handle in handles:
            with self.assertRaises(BusTimeout):
                handle.wait(timeout=1.0)

    def test_a_world_outside_the_fleet_is_rejected(self):
        with self.assertRaises(IndexError):
            self.table.submit(4, 0, _move())

    def test_a_blocked_thread_is_woken_by_the_fleet(self):
        """The end-to-end shape: a graph thread parks on a motion, the fleet
        runs it over several steps, and the thread resumes where it left off.

        This is what the design buys -- the node blocks once for a whole
        trajectory rather than once per control step, and the fleet keeps
        stepping every other world while it waits.
        """
        table, woke = self.table, []

        def graph_thread():
            handle = table.submit(0, 0, JointTrajectory(arm=0, waypoints=np.zeros((3, 6), dtype=np.float32)))
            handle.wait(timeout=5.0)
            woke.append(handle.steps)

        thread = threading.Thread(target=graph_thread, daemon=True)
        thread.start()

        # The fleet's loop, in miniature: drain, step a while, finish.
        started = None
        for step in range(200):
            if started is None:
                if table.drain(step=step):
                    started = step
                else:
                    threading.Event().wait(0.002)
            elif step >= started + 24:
                table.finish(0, 0, step=step)
                break
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive(), "the graph thread was never woken")
        self.assertEqual(woke, [24])


class TestServiceQueue(unittest.TestCase):
    """The marshalled path, for the few reads no snapshot carries."""

    def test_a_posted_call_runs_on_the_pumping_thread(self):
        service = ServiceQueue()
        seen: list[int] = []
        result: list = []

        def worker():
            result.append(service.post(0, lambda x: x * 2, (21,), {}, 5.0))

        thread = threading.Thread(target=worker)
        thread.start()
        while not seen:
            if service.pump():
                seen.append(threading.get_ident())
        thread.join(timeout=5.0)
        self.assertEqual(result, [42])
        self.assertEqual(seen, [threading.get_ident()])

    def test_an_exception_crosses_back_to_the_caller(self):
        service = ServiceQueue()
        errors: list = []

        def worker():
            try:
                service.post(0, _boom, (), {}, 5.0)
            except ValueError as exc:
                errors.append(str(exc))

        thread = threading.Thread(target=worker)
        thread.start()
        while not errors:
            service.pump()
        thread.join(timeout=5.0)
        self.assertEqual(errors, ["boom"])

    def test_the_budget_bounds_one_pump(self):
        """A world posting in a tight loop waits its turn rather than starving
        the fleet's step."""
        service = ServiceQueue(budget=2)
        for _ in range(5):
            service._queue.put(_noop_request())
        self.assertEqual(service.pump(), 2)
        self.assertEqual(service.pump(), 2)
        self.assertEqual(service.pump(), 1)

    def test_a_call_on_the_pumping_thread_runs_inline(self):
        """So a tool called from either side works both ways, as the
        single-world bus already guarantees."""
        service = ServiceQueue()
        service.pump()  # claims ownership
        self.assertEqual(service.post(0, lambda: 7, (), {}, 1.0), 7)

    def test_closing_releases_waiters(self):
        service = ServiceQueue()
        errors: list = []

        def worker():
            try:
                service.post(0, lambda: None, (), {}, 5.0)
            except BusTimeout as exc:
                errors.append(str(exc))

        thread = threading.Thread(target=worker)
        thread.start()
        while service._queue.empty():
            pass
        service.close()
        thread.join(timeout=5.0)
        self.assertEqual(len(errors), 1)

    def test_dropping_one_world_leaves_the_others_queued(self):
        service = ServiceQueue()
        service._queue.put(_noop_request(world=1))
        service._queue.put(_noop_request(world=2))
        service.drop_world(1)
        self.assertEqual(service._queue.qsize(), 1)


def _boom():
    raise ValueError("boom")


def _noop_request(world: int = 0):
    return _Request(fn=lambda: None, args=(), kwargs={}, world=world)


class TestStepWaiter(unittest.TestCase):
    """One condition for the fleet, rather than five hundred pollers."""

    def test_a_waiter_wakes_on_the_next_publish(self):
        waiter = StepWaiter()
        seen: list[int] = []

        def worker():
            seen.append(waiter.wait(after=3, timeout=5.0))

        thread = threading.Thread(target=worker)
        thread.start()
        waiter.notify(3)  # not past `after` -- must not wake
        thread.join(timeout=0.05)
        self.assertTrue(thread.is_alive(), "woke on a step that was not later")
        waiter.notify(4)
        thread.join(timeout=5.0)
        self.assertEqual(seen, [4])

    def test_a_step_already_past_returns_at_once(self):
        waiter = StepWaiter()
        waiter.notify(10)
        self.assertEqual(waiter.wait(after=3, timeout=1.0), 10)

    def test_closing_wakes_waiters_with_a_timeout(self):
        waiter = StepWaiter()
        errors: list = []

        def worker():
            try:
                waiter.wait(after=0, timeout=5.0)
            except BusTimeout as exc:
                errors.append(str(exc))

        thread = threading.Thread(target=worker)
        thread.start()
        threading.Timer(0.01, waiter.close).start()
        thread.join(timeout=5.0)
        self.assertEqual(len(errors), 1)

    def test_the_deadline_is_honoured(self):
        with self.assertRaises(BusTimeout):
            StepWaiter().wait(after=0, timeout=0.01)


class TestSnapshotTableAndWorldBus(unittest.TestCase):
    """Worlds stop at the tool boundary: a bus looks like the single-world one."""

    class _Snap:
        step = 12
        sim_time = 0.24

        def __init__(self, worlds: int):
            self._body = np.arange(worlds * 2 * 7, dtype=np.float64).reshape(worlds * 2, 7)
            self.worlds = worlds

        def body(self, world: int) -> np.ndarray:
            return self._body[world * 2 : world * 2 + 2]

    def test_each_world_gets_its_own_slice_in_the_single_world_type(self):
        table = SnapshotTable(3)
        snap = self._Snap(3)
        table.publish(snap, infos=[{"success": w == 1} for w in range(3)])
        for world in range(3):
            view = table.view(world)
            self.assertIsInstance(view, SimSnapshot)
            self.assertEqual(view.step, 12)
            np.testing.assert_array_equal(view.body_q, snap.body(world))
        self.assertTrue(table.view(1).info["success"])
        self.assertFalse(table.view(0).info["success"])

    def test_the_bus_offers_the_single_world_surface(self):
        """Every attribute a shipped suite tool reads off a bus."""
        table = SnapshotTable(2)
        table.publish(self._Snap(2))
        bus = WorldBus(1, snapshots=table, service=ServiceQueue(), waiter=StepWaiter())
        for attribute in ("snapshot", "call", "wait_for_step", "close", "closed"):
            self.assertTrue(hasattr(bus, attribute), f"a suite tool reads bus.{attribute}")
        np.testing.assert_array_equal(bus.snapshot.body_q, self._Snap(2).body(1))
        self.assertFalse(bus.closed)

    def test_a_closed_bus_refuses_rather_than_hanging(self):
        bus = WorldBus(0, snapshots=SnapshotTable(1), service=ServiceQueue(), waiter=StepWaiter())
        bus.close()
        self.assertTrue(bus.closed)
        with self.assertRaises(BusTimeout):
            bus.call(lambda: None)
        with self.assertRaises(BusTimeout):
            bus.wait_for_step(after=0)

    def test_one_world_closing_does_not_close_another(self):
        table, service, waiter = SnapshotTable(2), ServiceQueue(), StepWaiter()
        first = WorldBus(0, snapshots=table, service=service, waiter=waiter)
        second = WorldBus(1, snapshots=table, service=service, waiter=waiter)
        first.close()
        self.assertTrue(first.closed)
        self.assertFalse(second.closed)


class TestIntentShapes(unittest.TestCase):
    """The motions themselves are plans, carrying what the fleet needs."""

    def test_every_intent_is_frozen(self):
        """A plan the fleet is executing must not change under it."""
        for intent in (
            _move(),
            JointTrajectory(arm=0, waypoints=np.zeros((2, 6))),
            GripperMove(arm=0, command=1.0),
        ):
            with self.assertRaises(dataclasses.FrozenInstanceError):
                intent.arm = 1

    def test_a_fire_and_forget_move_is_expressible(self):
        """max_steps=0 installs the motion without the caller waiting, which is
        the non-blocking pose contract the runtime already has."""
        self.assertEqual(MoveToJoints(arm=0, target=np.zeros(6), max_steps=0).max_steps, 0)


if __name__ == "__main__":
    unittest.main()
