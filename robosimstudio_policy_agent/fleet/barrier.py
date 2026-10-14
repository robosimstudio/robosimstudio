# SPDX-License-Identifier: Apache-2.0
"""When the fleet is allowed to step, and what it answers together when it is not.

The single-world path gives the graph and the simulation strict turns: while a
node script is thinking, **nothing steps**. That is not an implementation detail,
it is the measurement. An object held in a closing hand does not creep while a
perception node runs, and the episode's step budget is not spent on the seconds
a planner took.

The fleet's first implementation gave that up, and said so as if it were a
feature -- ``stepper``'s docstring calls a thinking world "free", because nothing
writes its targets and it holds its last pose exactly. It holds its *targets*.
It does not hold its *state*: the world is in the same model as everyone else, so
it keeps being integrated while its script runs. Two things follow and both are
wrong.

* **Physics advances during host compute.** Contacts settle, a grasped part
  slips, a part balanced past its tip finishes falling. The longer a node thinks,
  the further the world moves -- and how long it thinks depends on the host, so
  the same episode run twice does not land in the same place.
* **The step budget pays for thinking.** ``limit`` counts control steps. A vision
  graph spending four seconds in a segmentation model burns however many steps
  the fleet managed in four seconds, and is truncated that much earlier than the
  serial run it is supposed to be comparable with.

So this module makes the rule explicit: **the fleet steps only when every live
world's graph thread is parked in a fleet wait.** A world running host code holds
the clock for the whole wave. That is the cost of the parity, and it is the
cost the serial path already pays -- the difference is that a wave pays it once
for whichever world is slowest rather than once per episode.

Two things follow from having the rule, and the second is why it is worth having.

:class:`ParkTable` is the rule. A world is parked exactly when its thread is
inside a wait this module knows about, and the discipline is stated twice because
both halves are races if you get them backwards:

* **Park before handing work over.** A thread that submits a motion and *then*
  parks is running, as far as the fleet can tell, in the window between the two
  -- and a motion that completes inside that window leaves the fleet stepping a
  world whose script is about to run. So the park brackets the submit.
* **Unpark before waking.** A thread whose event has been set is runnable, not
  parked, whatever its depth still says. So whoever completes a waiter clears the
  park first and sets the event second -- the same "wake outside the lock"
  discipline :mod:`.intents` already keeps, for the same kind of reason.

:class:`Rendezvous` is what the rule buys. At the moment the barrier closes,
every live world is parked, which means **the set of outstanding requests is
complete and known**. There is no guessing about who else might ask, no quorum,
no timeout heuristic: a request that wants company simply waits for the barrier
and is answered with everyone else's in one call. Inverse kinematics, a render
and a segmentation model are all the same shape of thing here, and none of them
needed a scheduler of their own.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robosimstudio.core.bus import BusTimeout

__all__ = ["Ask", "Barrier", "ParkTable", "Rendezvous", "StallTimeout", "StallWatch", "park_if"]


@contextmanager
def park_if(parks: ParkTable | None, world: int) -> Iterator[None]:
    """Bracket a blocking fleet wait, when there is a table to tell about it.

    Every wait in the fleet is written once and works with or without a barrier:
    ``None`` is the pre-barrier arrangement, which is what the threading-contract
    tests construct and what a single-world bus is. A negative ``world`` is a
    caller that is not a world -- the fleet's own thread, or a test -- and parks
    nothing, because marking a world parked on the fleet's behalf would let the
    barrier close over a thread that is still running.
    """
    if parks is None or int(world) < 0:
        yield
        return
    with parks.parked(int(world)):
        yield


class StallTimeout(BusTimeout):
    """A world held the barrier past its deadline.

    A :class:`~robosimstudio.core.bus.BusTimeout` because that is what the graph
    runtime already routes as a node failure rather than as a crash -- the world
    that stalled fails its node and its episode ends; the wave keeps its other
    episodes, which is the whole reason the watchdog retires a world instead of
    failing the wave.
    """


class ParkTable:
    """Which worlds' graph threads are parked in a fleet wait.

    Args:
        num_worlds: Worlds in the fleet.

    A depth rather than a flag, so a wait nested inside another wait does not
    unpark the outer one on the way out.

    **Per thread, not per world**, which is the part that was learned the hard
    way. A world is not one thread: the graph runtime starts a background
    observation poller beside the script, and that poller calls the fleet too.
    With one depth per world, answering the poller's render cleared the park the
    *script* was still sitting inside -- so the world read as running for ever,
    the barrier never closed, the fleet never stepped, and the motion the script
    was waiting on could never finish. A vision graph deadlocked on its first
    move at every world count; a privileged one, with no poller, did not.

    So each thread's waits are counted separately, and the barrier asks about
    one of them: the world's **principal**, the thread running its graph, named
    by :meth:`adopt`. A helper thread's wait is its own business -- it must
    still be answered, which is why it is tracked at all, but it cannot hold the
    clock and cannot release it. That is not a shortcut: a poller sleeps on wall
    time between polls, so a barrier that waited for it would wait for a clock
    the fleet does not drive.

    With no principal adopted, any thread's park counts -- the pre-barrier
    arrangement, and what the threading-contract tests construct.
    """

    def __init__(self, num_worlds: int) -> None:
        self.num_worlds = int(num_worlds)
        self._cond = threading.Condition()
        self._depth: list[dict[int, int]] = [{} for _ in range(self.num_worlds)]
        self._principal: list[int | None] = [None] * self.num_worlds
        self._generation = 0

    # ------------------------------------------------------------ principals
    def adopt(self, world: int, ident: int | None = None) -> None:
        """Name the thread whose fleet waits are this world's. Called by it.

        From the graph thread itself, before it runs anything, so that the
        runtime's own helper threads -- which only exist once the graph has
        started -- are never mistaken for it.
        """
        world = int(world)
        if not 0 <= world < self.num_worlds:
            return
        with self._cond:
            self._principal[world] = threading.get_ident() if ident is None else int(ident)
            self._generation += 1
            self._cond.notify_all()

    def _is_parked(self, world: int) -> bool:
        """Whether this world's principal is in a fleet wait. Lock held."""
        who = self._principal[world]
        depth = self._depth[world]
        return bool(depth.get(who, 0) > 0) if who is not None else bool(depth)

    # ------------------------------------------------------------ worker side
    @contextmanager
    def parked(self, world: int) -> Iterator[None]:
        """Bracket a blocking fleet wait. Worker thread.

        Entered **before** the work is handed to the fleet, not after. See the
        module docstring: the window between submitting and waiting is a window
        in which a completion can arrive, and a fleet that stepped through it
        would step a world whose script was about to run.
        """
        world = int(world)
        ident = threading.get_ident()
        with self._cond:
            depth = self._depth[world]
            depth[ident] = depth.get(ident, 0) + 1
            self._generation += 1
            self._cond.notify_all()
        try:
            yield
        finally:
            with self._cond:
                depth = self._depth[world]
                left = depth.get(ident, 0) - 1
                if left > 0:
                    depth[ident] = left
                else:
                    depth.pop(ident, None)
                self._generation += 1
                self._cond.notify_all()

    # ------------------------------------------------------------- fleet side
    def unpark(self, world: int, ident: int | None = None) -> None:
        """Take one thread out of the parked set. Fleet thread, before waking it.

        Called by whoever is about to complete the thing that thread is waiting
        on. Between this and the event being set the thread reads as *running*,
        which is what stops the fleet stepping through a wake-up.

        Args:
            world: The world the waiter belongs to.
            ident: **The thread being woken**, which every completion knows
                because it was recorded when the request was made. Passing the
                wrong one, or leaving it out where a helper thread is being
                woken, releases the park the world's script is still inside --
                see the class docstring for what that costs.

                ``None`` clears every thread of the world, which is right only
                when the world has one: it is what a single-threaded caller and
                the pre-barrier arrangement mean.

        The depth is cleared outright rather than decremented, because the
        thread is about to leave *every* wait it is in; the bracket's own
        decrement then clamps rather than going negative.
        """
        world = int(world)
        if not 0 <= world < self.num_worlds:
            return
        with self._cond:
            depth = self._depth[world]
            if not depth:
                return
            if ident is None:
                depth.clear()
            elif depth.pop(int(ident), None) is None:
                return
            self._generation += 1
            self._cond.notify_all()

    def parked_mask(self) -> np.ndarray:
        with self._cond:
            return np.array([self._is_parked(w) for w in range(self.num_worlds)], dtype=bool)

    def holding(self, expected: np.ndarray) -> np.ndarray:
        """Of the worlds ``expected`` to be parked, which are not.

        Args:
            expected: Per world, whether it has a live thread that owes the
                barrier a park. A world whose graph has finished owes nothing --
                it is the caller that knows this, which is why the mask is passed
                in rather than derived here.
        """
        expected = np.asarray(expected, dtype=bool)
        with self._cond:
            depth = np.array([self._is_parked(w) for w in range(self.num_worlds)], dtype=bool)
        return expected & ~depth

    def all_parked(self, expected: np.ndarray) -> bool:
        return not bool(self.holding(expected).any())

    def wait_for_change(self, timeout: float = 0.02) -> None:
        """Block until some world parks or unparks, or ``timeout`` passes.

        A condition rather than a sleep, so a wave whose worlds are quick does
        not pay a fixed latency per step; a timeout rather than an unbounded wait,
        so the caller's own watchdog keeps ticking when nothing is moving at all.
        """
        with self._cond:
            start = self._generation
            self._cond.wait_for(lambda: self._generation != start, timeout=timeout)

    def reset(self) -> None:
        """Forget every park and every principal. What starting a wave does.

        The principals go too: next wave's graph threads are new ones, and a
        stale ident would name a thread that has exited -- a world that could
        never read as parked again.
        """
        with self._cond:
            self._depth = [{} for _ in range(self.num_worlds)]
            self._principal = [None] * self.num_worlds
            self._generation += 1
            self._cond.notify_all()


@dataclass
class Ask:
    """One world's outstanding request to be answered with everyone else's.

    Attributes:
        world: Which world asked.
        kind: The rendezvous group this joins. Requests share a call only when
            they share a kind.
        payload: Whatever the kind's adapter takes.
        result: Filled by the adapter. A ``BaseException`` here is raised in the
            asking world's thread, which is how one world's unreachable target
            fails that world alone rather than the group.
    """

    world: int
    kind: str
    payload: Any
    ident: int = 0
    """The thread that posted it, so :meth:`Rendezvous._settle` unparks that one
    and not whichever other thread of the world happens to be waiting."""
    done: threading.Event = field(default_factory=threading.Event, repr=False)
    result: Any = None
    abandoned: bool = False
    """Set, under the rendezvous lock, by an asker that timed out before its
    answer: :meth:`Rendezvous._settle` then leaves its park alone, because the
    thread has left this wait and may be inside another."""
    answered: bool = False
    """Set, under the same lock, by the settle that unparks it."""

    def complete(self, result: Any) -> None:
        self.result = result
        self.done.set()


class Rendezvous:
    """Requests that would rather be answered together than one at a time.

    A world posts, parks, and is answered when the barrier closes -- at which
    point every live world is parked, so the group is complete by construction.
    That is the whole argument for doing it here rather than with a quorum or a
    deadline: there is nobody left who might still ask.

    An adapter takes ``[(world, payload), ...]`` and returns one result per entry,
    in order. It may return a ``BaseException`` in a slot to fail that world
    alone. An adapter that cannot fuse its inputs is still welcome to loop -- it
    is no worse than the marshalled call it replaces, and it is off the step path
    either way.

    Args:
        parks: The fleet's park table, so a group can be unparked before it is
            woken.
    """

    def __init__(self, *, parks: ParkTable) -> None:
        self.parks = parks
        self._adapters: dict[str, Callable[[list[tuple[int, Any]]], list[Any]]] = {}
        self._lock = threading.Lock()
        self._pending: dict[str, list[Ask]] = {}
        self._closed = False

    # ------------------------------------------------------------ registration
    def register(self, kind: str, adapter: Callable[[list[tuple[int, Any]]], list[Any]]) -> None:
        """Name the call that answers a whole group of ``kind`` at once."""
        self._adapters[str(kind)] = adapter

    def knows(self, kind: str) -> bool:
        """Whether this kind has an adapter.

        Callers check rather than assume: a kind with no adapter belongs on the
        marshalled service queue, and posting it here would park a thread that
        nothing will ever answer.
        """
        return str(kind) in self._adapters

    # ------------------------------------------------------------ worker side
    def post(self, world: int, kind: str, payload: Any, timeout: float) -> Any:
        """Join the next group of ``kind`` and wait for its answer.

        The caller must already be parked -- this is reached from the world's bus,
        which brackets it. Parking here instead would leave the window the module
        docstring describes: the barrier could close, dispatch could answer the
        group, and the answer would arrive before the asker had declared itself.
        """
        kind = str(kind)
        if kind not in self._adapters:
            raise KeyError(f"no rendezvous adapter for {kind!r}")
        ask = Ask(world=int(world), kind=kind, payload=payload, ident=threading.get_ident())
        with self._lock:
            if self._closed:
                raise BusTimeout("the fleet's rendezvous is closed")
            self._pending.setdefault(kind, []).append(ask)
        if not ask.done.wait(timeout):
            # Withdraw before giving up. An ask left pending is answered later,
            # by which time this thread has moved on to another fleet wait --
            # and settling the stale ask unparks *that* wait, so the barrier
            # reads the world as running while it is blocked.
            with self._lock:
                group = self._pending.get(kind, [])
                if any(entry is ask for entry in group):
                    self._pending[kind] = [entry for entry in group if entry is not ask]
                ask.abandoned = not ask.answered
            if ask.abandoned:
                raise BusTimeout(f"{kind} was not answered within {timeout:.0f} s")
            # Settled in the same instant: its unpark was for this very wait,
            # and the wake-up follows it at once.
            ask.done.wait()
        if isinstance(ask.result, BaseException):
            raise ask.result
        return ask.result

    # ------------------------------------------------------------- fleet side
    def pending(self) -> bool:
        with self._lock:
            return any(group for group in self._pending.values())

    def dispatch(self) -> int:
        """Answer every outstanding group. Fleet thread, at the barrier.

        Returns:
            How many asks were answered.

        An adapter that raises fails only its own group, with the exception
        delivered to each of that group's worlds. The alternative -- letting it
        out of here -- would take down the fleet loop for one world's bad target,
        and the graph runtime already knows what to do with a tool that raised.
        """
        with self._lock:
            groups, self._pending = self._pending, {}
        answered = 0
        for kind, asks in groups.items():
            if not asks:
                continue
            adapter = self._adapters.get(kind)
            if adapter is None:
                self._settle(asks, [KeyError(f"no rendezvous adapter for {kind!r}")] * len(asks))
                answered += len(asks)
                continue
            try:
                results = list(adapter([(ask.world, ask.payload) for ask in asks]))
            except BaseException as exc:
                results = [exc] * len(asks)
            if len(results) != len(asks):
                results = [RuntimeError(f"{kind} adapter answered {len(results)} of {len(asks)} worlds")] * len(asks)
            self._settle(asks, results)
            answered += len(asks)
        return answered

    def _settle(self, asks: list[Ask], results: list[Any]) -> None:
        """Unpark every asker, then wake it. Order matters -- see the module docstring.

        An asker that has already given up is woken but not unparked.
        """
        with self._lock:
            live = [ask for ask in asks if not ask.abandoned]
            for ask in live:
                ask.answered = True
        for ask in live:
            self.parks.unpark(ask.world, ask.ident)
        for ask, result in zip(asks, results, strict=False):
            ask.complete(result)

    def drop_world(self, world: int, error: BaseException | None = None) -> None:
        """Release one world's asks, for a world whose episode has ended."""
        world = int(world)
        failure = error or BusTimeout(f"world {world} finished while a request was pending")
        with self._lock:
            dropped: list[Ask] = []
            for kind, asks in list(self._pending.items()):
                keep = [ask for ask in asks if ask.world != world]
                dropped.extend(ask for ask in asks if ask.world == world)
                self._pending[kind] = keep
        self._settle(dropped, [failure] * len(dropped))

    def close(self) -> None:
        """Release every waiter with a timeout. Fleet thread."""
        with self._lock:
            self._closed = True
            groups, self._pending = self._pending, {}
        for asks in groups.values():
            self._settle(asks, [BusTimeout("the fleet's rendezvous closed")] * len(asks))

    def reopen(self) -> None:
        """Serve again after a close. What starting the next wave does."""
        with self._lock:
            self._closed = False
            self._pending = {}


class StallWatch:
    """How long the barrier has been held, and by whom.

    Split out from the loop it guards so the deadline can be tested without a
    fleet: ``arm`` at the top of a barrier wait, ``progress`` whenever anything
    actually happened, and ``expired`` is the question the loop asks.

    Args:
        timeout: Seconds a world may hold the barrier before it is retired. The
            bus timeout by default, because that is already the answer to "how
            long may a graph take without being presumed dead", and a barrier
            that fired sooner would call a healthy but slow perception node a
            hang.
    """

    def __init__(self, timeout: float = 120.0) -> None:
        self.timeout = float(timeout)
        self._deadline = 0.0

    def arm(self) -> None:
        self._deadline = time.monotonic() + self.timeout

    def progress(self) -> None:
        """Something moved, so the clock starts again.

        A rendezvous round, a service call, a world parking: any of them means
        the wave is alive and the deadline is measuring the wrong thing.
        """
        self.arm()

    @property
    def expired(self) -> bool:
        return time.monotonic() > self._deadline

    @property
    def held_s(self) -> float:
        return max(0.0, self.timeout - (self._deadline - time.monotonic()))


class Barrier:
    """Holds the fleet until every live world is parked, answering groups meanwhile.

    The loop the whole design converges on, and it is four lines of decision:

    1. **A marshalled call was serviced.** Something moved; go round again,
       because the world it woke is now running and owes another park.
    2. **Everyone is parked and a group is outstanding.** This is the moment the
       design exists for -- the set of asks is complete, because there is nobody
       left who could still ask. Answer the whole group in one call and go round
       again *without stepping*: answering a question is not time passing.
    3. **Everyone is parked and nothing is outstanding.** The fleet may step.
    4. **Otherwise** some world is running host code, and the fleet waits. The
       wait is not waste -- in a serial run that world's simulation is stopped
       too, for exactly as long. What it costs is the *other* worlds' time, and
       :attr:`held_s` is where that cost is reported rather than hidden.

    Deliberately built from callables rather than from a fleet. The loop is a
    concurrency contract and concurrency contracts are worth testing as such --
    ``expect`` and ``on_stall`` are the only two things it needs to know about
    worlds, and neither needs a device to answer.

    Args:
        parks: Who is parked.
        rendezvous: Groups waiting to be answered, or ``None`` for none.
        service: The marshalled-call queue to pump, or ``None`` for none.
        expect: ``() -> np.ndarray``, per world whether it owes a park before the
            fleet may step. A finished graph owes nothing.
        on_stall: ``(worlds) -> None``, called with whichever worlds held the
            barrier past the deadline. Retiring them is the caller's business;
            this only stops asking about them, which is what lets the fleet step
            again when a thread will never wake.
        watch: The deadline. Defaults to one built from ``timeout``.
        poll_s: Ceiling on how long a turn waits on the park table. A ceiling
            and not a cadence: the table is a condition, so a world parking wakes
            the fleet at once and this only bounds the watchdog's own ticking.
    """

    def __init__(
        self,
        parks: ParkTable,
        *,
        expect: Callable[[], np.ndarray],
        rendezvous: Rendezvous | None = None,
        service: Any = None,
        on_stall: Callable[[list[int]], None] | None = None,
        watch: StallWatch | None = None,
        timeout: float = 120.0,
        poll_s: float = 0.02,
    ) -> None:
        self.parks = parks
        self.rendezvous = rendezvous
        self.service = service
        self.expect = expect
        self.on_stall = on_stall
        self.watch = watch or StallWatch(timeout)
        self.poll_s = float(poll_s)
        self.held_s = 0.0
        self.rounds = 0
        self.answered = 0
        self.stalls = 0

    def quiesce(self) -> None:
        """Return when every live world is parked and no group is outstanding."""
        started = time.perf_counter()
        self.watch.arm()
        while True:
            if self.service is not None and self.service.pump():
                self.watch.progress()
                continue
            expected = self.expect()
            if self.parks.all_parked(expected):
                if self.rendezvous is None or not self.rendezvous.pending():
                    break
                self.answered += self.rendezvous.dispatch()
                self.rounds += 1
                self.watch.progress()
                continue
            if self.watch.expired:
                held = [int(w) for w in np.flatnonzero(self.parks.holding(expected))]
                self.stalls += len(held)
                if self.on_stall is not None:
                    self.on_stall(held)
                self.watch.arm()
                continue
            self.parks.wait_for_change(timeout=self.poll_s)
        self.held_s += time.perf_counter() - started

    def stats(self) -> dict[str, float]:
        """What the barrier cost, for the run's provenance.

        ``held_s`` is wall time the fleet could not step because some world was
        running host code. Read beside ``FleetSummary.utilization`` rather than
        instead of it: utilization says how much of the fleet was working, this
        says how much of the wave was nobody working at all.
        """
        return {
            "held_s": round(self.held_s, 3),
            "rendezvous_rounds": int(self.rounds),
            "batched_calls": int(self.answered),
            "stalled_worlds": int(self.stalls),
        }
