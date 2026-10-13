# SPDX-License-Identifier: Apache-2.0
"""What the fleet answers together, because at the barrier the group is complete.

The other half of the barrier (whose park rule is :mod:`tests.test_fleet_threads`). Once every live world is
parked there is nobody left who could still ask, so a request that wants company
does not need a quorum, a deadline or a guess -- it waits for the barrier and is
answered with everyone else's in one call.

What is worth testing about that is almost entirely the failure directions,
since the happy path is one call returning a list: an unreachable target must
fail *its own world* and not the group, an adapter that raises must not take the
fleet loop down with it, and an adapter that answers the wrong number of worlds
must be loud rather than pairing somebody's answer with somebody else's world.
Nothing here needs a device.
"""

from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from robosimstudio.core.bus import BusTimeout
from robosimstudio.core.streams import reading, stream_lock
from robosimstudio_policy_agent.fleet import (
    MoveToJoints,
    ParkTable,
    Rendezvous,
    ServiceQueue,
    StallTimeout,
    StallWatch,
)
from robosimstudio_policy_agent.fleet.ik_backend import IKBackendFleet, solve_group

TIER = "fast"
"""No GPU. Gates every commit."""


def _all(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


def _move(arm: int = 0) -> MoveToJoints:
    return MoveToJoints(arm=arm, target=np.zeros(6, dtype=np.float32))


def _guard_held() -> bool:
    """Whether the calling thread is inside :func:`~robosimstudio.core.streams.reading`.

    The question the deadlock turns on, and it has to be asked of *this* thread:
    the lock is reentrant, so "can I acquire it" is always yes for its owner and
    says nothing. ``_is_owned`` is true only for the owner, which is exactly
    the property under test, and unlike ``_recursion_count`` (Python 3.12) every
    supported Python has it.
    """
    return stream_lock._is_owned()


class _Parked:
    """Worlds parked for the rest of a test, and kept alive while they are.

    ``parks.parked(w).__enter__()`` on its own is a trap: nothing holds the
    context manager, CPython finalises the generator the moment the call
    returns, and the ``finally`` unparks the world again. A test that wants a
    world to stay parked has to keep the handle, which is what this is for.
    """

    def __init__(self) -> None:
        self._held: list = []

    def park(self, parks: ParkTable, world: int) -> None:
        entered = parks.parked(world)
        entered.__enter__()
        self._held.append(entered)


class _OrderingParks(ParkTable):
    """A park table that records whether a world was parked when it was woken.

    The only way to check the unpark-before-wake rule from outside is to ask the
    waking thread what the table said at the moment it set the event, so the
    table is what remembers.
    """

    def __init__(self, num_worlds: int) -> None:
        super().__init__(num_worlds)
        self.unparked: list[int] = []

    def unpark(self, world: int, ident: int | None = None) -> None:
        self.unparked.append(int(world))
        super().unpark(world, ident)


class TestRendezvous(unittest.TestCase):
    """Requests answered together, because at the barrier the group is complete."""

    def _rv(self) -> tuple[Rendezvous, ParkTable]:
        parks = ParkTable(4)
        return Rendezvous(parks=parks), parks

    def test_one_adapter_call_answers_a_whole_group(self):
        rv, _ = self._rv()
        seen: list[list[tuple[int, object]]] = []
        rv.register("ik", lambda asks: (seen.append(asks), [w * 10 for w, _ in asks])[1])
        out: dict[int, object] = {}
        threads = [threading.Thread(target=lambda w=w: out.__setitem__(w, rv.post(w, "ik", w, 5.0))) for w in (0, 1, 2)]
        for t in threads:
            t.start()
        self._await(rv, 3)
        rv.dispatch()
        for t in threads:
            t.join(timeout=5.0)
        self.assertEqual(len(seen), 1, "the group was answered in more than one call")
        self.assertEqual(out, {0: 0, 1: 10, 2: 20})

    def test_kinds_do_not_share_a_call(self):
        rv, _ = self._rv()
        calls: list[str] = []
        for kind in ("ik", "render"):
            rv.register(kind, lambda asks, k=kind: (calls.append(k), [k] * len(asks))[1])
        out: dict[int, object] = {}
        threads = [
            threading.Thread(target=lambda w=w, k=k: out.__setitem__(w, rv.post(w, k, None, 5.0)))
            for w, k in ((0, "ik"), (1, "render"), (2, "ik"))
        ]
        for t in threads:
            t.start()
        self._await(rv, 3)
        self.assertEqual(rv.dispatch(), 3)
        for t in threads:
            t.join(timeout=5.0)
        self.assertEqual(sorted(calls), ["ik", "render"])
        self.assertEqual(out, {0: "ik", 1: "render", 2: "ik"})

    def test_an_exception_in_a_slot_fails_only_that_world(self):
        """One unreachable target is one world's node error, not the group's."""
        rv, _ = self._rv()
        rv.register("ik", lambda asks: [RuntimeError("unreachable") if w == 1 else "ok" for w, _ in asks])
        out: dict[int, object] = {}

        def ask(world: int) -> None:
            try:
                out[world] = rv.post(world, "ik", None, 5.0)
            except RuntimeError as exc:
                out[world] = exc

        threads = [threading.Thread(target=ask, args=(w,)) for w in (0, 1)]
        for t in threads:
            t.start()
        self._await(rv, 2)
        rv.dispatch()
        for t in threads:
            t.join(timeout=5.0)
        self.assertEqual(out[0], "ok")
        self.assertIsInstance(out[1], RuntimeError)

    def test_an_adapter_that_raises_fails_its_own_group_only(self):
        rv, _ = self._rv()
        rv.register("bad", lambda asks: (_ for _ in ()).throw(ValueError("boom")))
        rv.register("good", lambda asks: ["fine"] * len(asks))
        out: dict[int, object] = {}

        def ask(world: int, kind: str) -> None:
            try:
                out[world] = rv.post(world, kind, None, 5.0)
            except ValueError as exc:
                out[world] = exc

        threads = [threading.Thread(target=ask, args=a) for a in ((0, "bad"), (1, "good"))]
        for t in threads:
            t.start()
        self._await(rv, 2)
        rv.dispatch()
        for t in threads:
            t.join(timeout=5.0)
        self.assertIsInstance(out[0], ValueError)
        self.assertEqual(out[1], "fine")

    def test_a_short_answer_fails_the_group_loudly(self):
        """Silently pairing the wrong answer with the wrong world is the bug to avoid."""
        rv, _ = self._rv()
        rv.register("ik", lambda asks: ["only one"])
        out: dict[int, object] = {}

        def ask(world: int) -> None:
            try:
                out[world] = rv.post(world, "ik", None, 5.0)
            except RuntimeError as exc:
                out[world] = exc

        threads = [threading.Thread(target=ask, args=(w,)) for w in (0, 1)]
        for t in threads:
            t.start()
        self._await(rv, 2)
        rv.dispatch()
        for t in threads:
            t.join(timeout=5.0)
        for world in (0, 1):
            self.assertIsInstance(out[world], RuntimeError)
            self.assertIn("1 of 2", str(out[world]))

    def test_an_unknown_kind_refuses_rather_than_parking_forever(self):
        rv, _ = self._rv()
        self.assertFalse(rv.knows("ik"))
        with self.assertRaises(KeyError):
            rv.post(0, "ik", None, 1.0)

    def test_dispatch_unparks_before_it_wakes(self):
        parks = _OrderingParks(2)
        rv = Rendezvous(parks=parks)
        rv.register("ik", lambda asks: ["ok"] * len(asks))
        observed: list[bool] = []

        def ask() -> None:
            with parks.parked(1):
                rv.post(1, "ik", None, 5.0)
            observed.append(parks.parked_mask()[1])

        thread = threading.Thread(target=ask)
        thread.start()
        self._await(rv, 1)
        rv.dispatch()
        thread.join(timeout=5.0)
        self.assertEqual(parks.unparked, [1])
        self.assertEqual(observed, [False])

    def test_dropping_a_world_releases_only_that_world(self):
        rv, _ = self._rv()
        rv.register("ik", lambda asks: ["ok"] * len(asks))
        out: dict[int, object] = {}

        def ask(world: int) -> None:
            try:
                out[world] = rv.post(world, "ik", None, 5.0)
            except BusTimeout as exc:
                out[world] = exc

        threads = [threading.Thread(target=ask, args=(w,)) for w in (0, 1)]
        for t in threads:
            t.start()
        self._await(rv, 2)
        rv.drop_world(0)
        threads[0].join(timeout=5.0)
        self.assertIsInstance(out[0], BusTimeout)
        self.assertTrue(rv.pending(), "world 1's ask went with world 0's")
        rv.dispatch()
        threads[1].join(timeout=5.0)
        self.assertEqual(out[1], "ok")

    def test_a_stall_error_travels_to_the_dropped_world(self):
        rv, _ = self._rv()
        rv.register("ik", lambda asks: ["ok"] * len(asks))
        out: list[BaseException] = []

        def ask() -> None:
            try:
                rv.post(0, "ik", None, 5.0)
            except BaseException as exc:
                out.append(exc)

        thread = threading.Thread(target=ask)
        thread.start()
        self._await(rv, 1)
        rv.drop_world(0, StallTimeout("held the fleet"))
        thread.join(timeout=5.0)
        self.assertIsInstance(out[0], StallTimeout)

    def test_close_releases_everyone_and_reopen_serves_again(self):
        rv, _ = self._rv()
        rv.register("ik", lambda asks: ["ok"] * len(asks))
        out: list[object] = []
        thread = threading.Thread(target=lambda: out.append(self._swallow_post(rv)))
        thread.start()
        self._await(rv, 1)
        rv.close()
        thread.join(timeout=5.0)
        self.assertIsInstance(out[0], BusTimeout)
        with self.assertRaises(BusTimeout):
            rv.post(0, "ik", None, 1.0)
        rv.reopen()
        self.assertFalse(rv.pending())

    @staticmethod
    def _swallow_post(rv: Rendezvous) -> object:
        try:
            return rv.post(0, "ik", None, 5.0)
        except BusTimeout as exc:
            return exc

    @staticmethod
    def _await(rv: Rendezvous, count: int) -> None:
        for _ in range(400):
            with rv._lock:
                if sum(len(g) for g in rv._pending.values()) >= count:
                    return
            time.sleep(0.005)
        raise AssertionError(f"only {count} asks never arrived")


class TestStallWatch(unittest.TestCase):
    """The one failure this design adds, and its deadline."""

    def test_an_armed_watch_is_not_expired(self):
        watch = StallWatch(timeout=5.0)
        watch.arm()
        self.assertFalse(watch.expired)

    def test_it_expires_past_the_deadline(self):
        watch = StallWatch(timeout=0.01)
        watch.arm()
        time.sleep(0.05)
        self.assertTrue(watch.expired)

    def test_progress_restarts_the_clock(self):
        watch = StallWatch(timeout=0.05)
        watch.arm()
        time.sleep(0.03)
        watch.progress()
        time.sleep(0.03)
        self.assertFalse(watch.expired)


class TestTheStreamGuardIsOnTheFleetsSide(unittest.TestCase):
    """Where the device guard is taken, and why the other place is a deadlock.

    :func:`~robosimstudio.core.streams.reading` is one process-wide reentrant
    lock, and reentrancy is per *thread*. A worker that takes it and then hands
    the solve to the fleet leaves the fleet blocking on a lock the worker will
    not release until the fleet returns -- which it cannot.

    This is not hypothetical and it is not cheap: every ``robot.solve_ik`` in a
    wave used to cost the whole bus timeout and then fail its node, which is why
    no non-privileged graph could run batched. The first test measures that the
    hazard is real, the second asserts the backend does not create it.
    """

    def test_holding_the_guard_across_a_marshalled_call_deadlocks(self):
        """The shape to avoid, measured rather than asserted from the docstring."""
        service = ServiceQueue()
        out: list[object] = []

        def on_the_fleet() -> str:
            with reading():  # what a seed read does
                return "solved"

        def worker() -> None:
            try:
                with reading():  # the wrong side
                    out.append(service.post(0, on_the_fleet, (), {}, 0.5))
            except BusTimeout as exc:
                out.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(0.05)
        service.pump()
        thread.join(timeout=5.0)
        self.assertIsInstance(out[0], BusTimeout, "the deadlock this guards against has gone away")

    def test_the_backend_does_not_hold_the_guard_when_it_marshals(self):
        """The rule, asserted where it is easy to break it again."""
        held: list[bool] = []

        class _Bus:
            timeout = 5.0

            def call(self, fn, *args, **kwargs):
                held.append(_guard_held())
                return "marshalled"

            def batchable(self, kind):
                return False

        backend = IKBackendFleet(fleet=None, world=0, bus=_Bus())
        backend.solve_ik({"position": {"x": 0, "y": 0, "z": 0}, "rotation": {"w": 1, "x": 0, "y": 0, "z": 0}})
        backend.plan_linear({}, {})
        self.assertEqual(held, [False, False], "the worker still holds the stream lock across the bus")

    def test_a_solve_joins_the_group_when_one_serves_it(self):
        asked: list[tuple[str, object]] = []

        class _Bus:
            timeout = 5.0

            def batchable(self, kind):
                return kind == IKBackendFleet.SOLVE

            def batched(self, kind, payload, timeout=None):
                asked.append((kind, payload))
                return "grouped"

            def call(self, fn, *args, **kwargs):
                asked.append(("call", fn))
                return "marshalled"

        backend = IKBackendFleet(fleet=None, world=0, bus=_Bus())
        self.assertEqual(backend.solve_ik({}), "grouped")
        # plan_linear has no adapter on this bus, so it takes the single-world
        # path rather than parking on a group nobody will answer.
        self.assertEqual(backend.plan_linear({}, {}), "marshalled")
        self.assertEqual(asked[0][0], IKBackendFleet.SOLVE)
        self.assertEqual(asked[1][0], "call")


def _ask(world: int, fn, *, kind: str = "plan", args: tuple = ()) -> tuple[int, tuple]:
    """One posted solve, in the shape the backend posts it.

    The payload carries the backend and the kind as well as the call, so the
    group's adapter can read what an ask is *about* and answer a whole arm's
    worth in one search. ``kind`` defaults to something that is not
    :attr:`IKBackendFleet.SOLVE`, which is how these fakes stay on the
    one-at-a-time path: there is no fleet behind them to batch against, and
    what this class is testing is the loop's contract.
    """
    return (world, (None, kind, fn, args))


class TestTheSolveGroup(unittest.TestCase):
    """What the wave's solves come back as, one at a time or fused."""

    def test_every_world_gets_its_own_answer_in_order(self):
        asks = [_ask(w, lambda v=w: [float(v)]) for w in (0, 1, 2)]
        self.assertEqual(solve_group(asks), [[0.0], [1.0], [2.0]])

    def test_a_solve_that_finds_nothing_answers_none(self):
        """``None`` is an answer -- an unreachable pose, not a failure to ask."""
        self.assertEqual(solve_group([_ask(0, lambda: None)]), [None])

    def test_one_world_raising_does_not_take_the_group(self):
        def boom() -> None:
            raise RuntimeError("out of reach")

        answers = solve_group([_ask(0, lambda: "ok"), _ask(1, boom), _ask(2, lambda: "ok")])
        self.assertEqual(answers[0], "ok")
        self.assertIsInstance(answers[1], RuntimeError)
        self.assertEqual(answers[2], "ok")

    def test_the_group_releases_the_guard_when_it_is_done(self):
        solve_group([_ask(0, lambda: None)])
        self.assertFalse(_guard_held())

    def test_a_plan_is_never_folded_into_the_pose_solves(self):
        """``plan_linear`` shares this adapter and must not share its batching:
        an ask there is forty chained solves seeded from one another, which is
        why it has its own rendezvous group in the first place."""
        seen: list[str] = []
        asks = [
            _ask(0, lambda *_a: seen.append("solve") or "a", kind=IKBackendFleet.SOLVE, args=({}, 0, None, None)),
            _ask(1, lambda: seen.append("plan") or "b", kind=IKBackendFleet.PLAN_LINEAR),
        ]

        self.assertEqual(solve_group(asks), ["a", "b"])
        self.assertEqual(sorted(seen), ["plan", "solve"], "both ran, one at a time")

    def test_a_group_with_no_fleet_behind_it_still_answers(self):
        """The adapter declines to fuse whatever it cannot set up, and declining
        is answering one at a time -- never failing the group."""
        asks = [
            _ask(w, lambda *_a, v=w: [float(v)], kind=IKBackendFleet.SOLVE, args=({}, 0, None, None)) for w in (0, 1)
        ]

        self.assertEqual(solve_group(asks), [[0.0], [1.0]])


if __name__ == "__main__":
    unittest.main()
