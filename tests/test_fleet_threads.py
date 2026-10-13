# SPDX-License-Identifier: Apache-2.0
"""A world is not one thread, and the barrier has to survive that.

The barrier's rule is "the fleet steps when every live world's script is parked
in a fleet wait". That reads as though a world were a thread, and for a long
time it was implemented as though it were: one park depth per world, cleared by
whoever completed the thing the world was waiting on.

It is not. The graph runtime starts a background observation poller beside the
script, on its own thread, and that poller calls the fleet too -- it asks for a
picture through the same rendezvous the script's own look goes through. With one
depth per world, answering the *poller* cleared the park the *script* was still
sitting inside. The world then read as running for ever: the barrier never
closed, so the fleet never stepped, so the motion the script was waiting on
could never finish. A vision graph deadlocked on its first move at every world
count, and a privileged one -- which starts no poller -- did not, which is why
it took a twelve-world run to find.

So each thread's waits are counted separately and the barrier asks about one of
them: the world's principal, the thread running its graph. This module is that
distinction, from both sides -- a helper must not release the script's park, and
a helper must not hold the clock either, because a poller sleeps on wall time
between polls and the fleet does not drive that clock.

No GPU: two threads, a table, and a rendezvous.
"""

from __future__ import annotations

import threading
import unittest

import numpy as np

from robosimstudio.core.bus import BusTimeout
from robosimstudio_policy_agent.fleet.barrier import Barrier, ParkTable, Rendezvous

TIER = "fast"
"""Threading contract only. Gates every commit."""

SETTLE = 2.0
"""Seconds to allow a helper thread to reach its wait. Generous: the assertion
is about ordering, and a slow box must not turn that into a flake."""


def _in_thread(fn, *args) -> threading.Thread:
    thread = threading.Thread(target=fn, args=args, daemon=True)
    thread.start()
    return thread


class TestAWorldIsNotOneThread(unittest.TestCase):
    """Whose park the barrier is actually about."""

    def test_a_helper_thread_does_not_release_the_principal(self):
        """The deadlock, in miniature.

        The principal parks on a motion; a helper parks and is woken. Before the
        fix the helper's completion cleared the world outright and the principal
        -- still blocked -- read as running for the rest of the wave.
        """
        parks = ParkTable(2)
        started, release = threading.Event(), threading.Event()

        def principal() -> None:
            parks.adopt(0)
            with parks.parked(0):
                started.set()
                release.wait(SETTLE)

        _in_thread(principal)
        self.assertTrue(started.wait(SETTLE))
        self.assertTrue(parks.parked_mask()[0], "the principal is in a fleet wait")

        helper_in = threading.Event()
        helper_ident: list[int] = []

        def helper() -> None:
            helper_ident.append(threading.get_ident())
            with parks.parked(0):
                helper_in.set()
                release.wait(SETTLE)

        _in_thread(helper)
        self.assertTrue(helper_in.wait(SETTLE))

        parks.unpark(0, helper_ident[0])

        self.assertTrue(parks.parked_mask()[0], "the principal's park is its own")
        release.set()

    def test_a_helper_thread_does_not_hold_the_clock(self):
        """The other direction, and the reason the rule is not "every thread".

        A poller spends most of its life asleep on a wall clock the fleet does
        not drive. A barrier that waited for it would wait for ever between
        polls, which is the same deadlock arriving from the opposite side.
        """
        parks = ParkTable(2)
        started = threading.Event()
        release = threading.Event()

        def principal() -> None:
            parks.adopt(0)
            with parks.parked(0):
                started.set()
                release.wait(SETTLE)

        _in_thread(principal)
        self.assertTrue(started.wait(SETTLE))

        # A helper that is awake, running host code, and not parked at all.
        expected = np.array([True, False])
        self.assertTrue(parks.all_parked(expected), "the world is its principal")
        release.set()

    def test_the_principal_is_whoever_says_so(self):
        """Not the first thread to park -- a poller can easily be that."""
        parks = ParkTable(1)
        first_in, release = threading.Event(), threading.Event()

        def poller() -> None:
            with parks.parked(0):
                first_in.set()
                release.wait(SETTLE)

        _in_thread(poller)
        self.assertTrue(first_in.wait(SETTLE))
        self.assertTrue(parks.parked_mask()[0], "with no principal yet, any park counts")

        claimed = threading.Event()

        def script() -> None:
            parks.adopt(0)
            claimed.set()
            release.wait(SETTLE)

        _in_thread(script)
        self.assertTrue(claimed.wait(SETTLE))
        self.assertFalse(parks.parked_mask()[0], "the script is running; the poller's park is not the world's")
        release.set()

    def test_a_wave_forgets_its_principals(self):
        """A stale ident names a thread that has exited -- a world that could
        never read as parked again, which is a deadlock one wave later."""
        parks = ParkTable(1)
        done = threading.Event()
        _in_thread(lambda: (parks.adopt(0), done.set()))
        self.assertTrue(done.wait(SETTLE))

        parks.reset()

        with parks.parked(0):
            self.assertTrue(parks.parked_mask()[0], "any thread again, until the next graph adopts")

    def test_nested_waits_on_the_principal_still_nest(self):
        """The depth is per thread, so it is still a depth."""
        parks = ParkTable(1)
        parks.adopt(0)
        with parks.parked(0):
            with parks.parked(0):
                self.assertTrue(parks.parked_mask()[0])
            self.assertTrue(parks.parked_mask()[0], "the outer wait is still a wait")
        self.assertFalse(parks.parked_mask()[0])

    def test_unparking_without_an_ident_still_clears_the_world(self):
        """What a single-threaded caller means, and what the pre-barrier
        arrangement meant. Kept because it is right when a world has one
        thread, and every caller that knows better now says so."""
        parks = ParkTable(1)
        parks.adopt(0)
        with parks.parked(0):
            parks.unpark(0)
            self.assertFalse(parks.parked_mask()[0])


class TestTheRendezvousWakesTheAskerItAnswered(unittest.TestCase):
    """The path the deadlock actually came down: a look asked for by a poller."""

    def test_answering_a_helpers_look_leaves_the_script_parked(self):
        parks = ParkTable(1)
        rendezvous = Rendezvous(parks=parks)
        rendezvous.register("render", lambda asks: ["a frame"] * len(asks))

        script_in, release = threading.Event(), threading.Event()

        def script() -> None:
            parks.adopt(0)
            with parks.parked(0):
                script_in.set()
                release.wait(SETTLE)

        _in_thread(script)
        self.assertTrue(script_in.wait(SETTLE))

        looked = threading.Event()

        def poller() -> None:
            # What ``WorldFacade.get_observation`` does, on the runtime's own
            # thread: park, post, wait for the group.
            with parks.parked(0):
                rendezvous.post(0, "render", None, SETTLE)
            looked.set()

        _in_thread(poller)
        barrier_seen = np.array([True])
        for _ in range(int(SETTLE / 0.01)):
            if rendezvous.pending():
                break
            threading.Event().wait(0.01)
        self.assertTrue(rendezvous.pending(), "the poller joined the group")

        rendezvous.dispatch()

        self.assertTrue(looked.wait(SETTLE), "the poller got its frame")
        self.assertTrue(parks.all_parked(barrier_seen), "and the script is still parked")
        release.set()


class TestTheBarrierStepsWithAPollerRunning(unittest.TestCase):
    """End to end on the loop itself: the wave must reach a step."""

    def test_the_fleet_gets_to_step(self):
        parks = ParkTable(1)
        rendezvous = Rendezvous(parks=parks)
        rendezvous.register("render", lambda asks: [None] * len(asks))
        barrier = Barrier(parks, expect=lambda: np.array([True]), rendezvous=rendezvous, timeout=SETTLE)

        script_in, release = threading.Event(), threading.Event()

        def script() -> None:
            parks.adopt(0)
            with parks.parked(0):
                script_in.set()
                release.wait(SETTLE * 2)

        _in_thread(script)
        self.assertTrue(script_in.wait(SETTLE))

        frames: list[object] = []
        ended: list[BusTimeout] = []

        def poller() -> None:
            # A poller outlives the barrier that answered it: whatever it asks
            # after quiesce returns is released when the fleet closes the
            # rendezvous, with a BusTimeout, which is how a wave ends its helpers.
            try:
                for _ in range(3):
                    with parks.parked(0):
                        frames.append(rendezvous.post(0, "render", None, SETTLE))
            except BusTimeout as exc:
                ended.append(exc)

        polling = _in_thread(poller)
        for _ in range(int(SETTLE / 0.01)):
            if rendezvous.pending():
                break
            threading.Event().wait(0.01)
        self.assertTrue(rendezvous.pending(), "the poller is waiting on the rendezvous")

        # Before the fix this never returned: the poller's first answer cleared
        # the script's park and the loop waited out its watchdog instead.
        barrier.quiesce()

        self.assertTrue(parks.all_parked(np.array([True])), "quiesce returns with the world parked")
        self.assertGreaterEqual(barrier.answered, 1, "the barrier answered the poller on the way")

        rendezvous.close()
        polling.join(SETTLE)
        self.assertFalse(polling.is_alive(), "closing the rendezvous releases the poller")
        self.assertTrue(frames, "the poller got the frame the barrier answered")
        if ended:
            self.assertIn("closed", str(ended[0]), "the poller was released by the close, not left to time out")
        release.set()


if __name__ == "__main__":
    unittest.main()
