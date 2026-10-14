# SPDX-License-Identifier: Apache-2.0
"""One world's view of the simulation, in the shape the single-world bus has.

This is what makes worlds invisible above the tool interface. A suite tool is
constructed with ``(registry, env, bus)`` and reads ``bus.snapshot``,
``bus.call`` and ``bus.wait_for_step``; a :class:`WorldBus` offers exactly those,
bound to one world, so **every shipped tool runs against a fleet unchanged**.
That matters beyond convenience: the graph plus its scripts and its tool
interface is the artifact that also runs on a real robot, so anything a batched
run needs scripts to know is something the sim-to-real boundary has to carry too.

The three surfaces differ in what they cost, and the difference is the design:

* ``snapshot`` -- free. A slice of the fleet's last batched read, at most the
  fleet's cadence behind. Nearly every read a graph does lands here.
* ``wait_for_step`` -- free until it blocks. A world waiting for steps to pass
  parks on a condition the fleet notifies once per publish, rather than each
  world polling its own clock.
* ``call`` -- expensive, and rare on purpose. Marshals onto the fleet's thread
  the way the single-world bus does, for the few reads no snapshot carries
  (contact closure, particle state). Kept so every existing tool works on day
  one, not because anything should stay here.

A closed bus releases its waiters with :class:`~robosimstudio.core.bus.BusTimeout`,
which is the error the graph runtime already routes as a node failure -- so a
world whose episode ends under a blocked node fails that node rather than hanging
the wave.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any

from robosimstudio.core.bus import BusTimeout, SimSnapshot

from .barrier import park_if

__all__ = ["ServiceQueue", "StepWaiter", "WorldBus"]


@dataclass
class _Request:
    fn: Any
    args: tuple
    kwargs: dict
    world: int = 0
    ident: int = 0
    """The thread that posted it. See :class:`~.barrier.ParkTable`: a world can
    have more than one thread in a fleet wait, and unparking the wrong one
    releases a park its owner is still inside."""
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None
    state: str = "queued"
    """``queued``, ``claimed`` by whoever services or releases it, then
    ``answered``; or ``abandoned`` by a poster that timed out before the
    answer. Moved under :attr:`ServiceQueue._claims`, so exactly one side
    wins: an abandoned request is never run if still queued, and never unparks
    its thread, which has left this wait and may be inside another."""


class ServiceQueue:
    """The fleet's one queue of marshalled calls, from every world.

    Shared rather than per world so the fleet pumps once per step regardless of
    how many worlds are asking, and so a budget bounds the whole fleet's host
    work rather than each world's.

    Args:
        budget: Most requests to service per pump. A world issuing them in a
            tight loop cannot starve the fleet; it merely waits its turn.
    """

    def __init__(self, *, budget: int = 32, parks: Any = None) -> None:
        self.budget = int(budget)
        self.parks = parks
        """The fleet's park table, or ``None``. See :class:`.barrier.ParkTable`."""
        self._queue: queue.Queue[_Request] = queue.Queue()
        self._closed = threading.Event()
        self._claims = threading.Lock()
        self.owner_thread: int | None = None

    def post(self, world: int, fn, args: tuple, kwargs: dict, timeout: float) -> Any:
        """Run ``fn`` on the fleet's thread and return its result. Worker side.

        The park brackets the *queueing* as well as the wait, and not because it
        is tidier: a request serviced between the put and the wait would leave a
        window in which the fleet reads this world as running and the world is
        in fact about to block. See :mod:`.barrier`.

        A caller already on the pumping thread runs inline and parks nothing --
        it is the fleet, not a world, and marking a world parked on its behalf
        would let the barrier close over a thread that is still running.
        """
        if self._closed.is_set():
            raise BusTimeout("the fleet's service queue is closed")
        if self.owner_thread == threading.get_ident():
            return fn(*args, **kwargs)
        request = _Request(fn=fn, args=args, kwargs=kwargs, world=int(world), ident=threading.get_ident())
        with park_if(self.parks, world):
            self._queue.put(request)
            if not request.done.wait(timeout):
                # Withdraw before giving up. A request left queued would run
                # after its caller had been told it failed, and its unpark would
                # clear whatever wait this thread is in by then -- a world read
                # as running while it is blocked.
                with self._claims:
                    abandoned = request.state != "answered"
                    if abandoned:
                        request.state = "abandoned"
                if abandoned:
                    raise BusTimeout(f"{getattr(fn, '__name__', fn)} was not serviced within {timeout:.0f} s")
                # Answered in the same instant: its unpark was for this very
                # wait, and the wake-up follows it at once.
                request.done.wait()
        if request.error is not None:
            raise request.error
        return request.result

    def pump(self, budget: int | None = None) -> int:
        """Service pending requests. Fleet thread, between steps."""
        self.owner_thread = threading.get_ident()
        limit = self.budget if budget is None else int(budget)
        serviced = 0
        while serviced < limit:
            try:
                request = self._queue.get_nowait()
            except queue.Empty:
                break
            if not self._claim(request):
                continue
            try:
                request.result = request.fn(*request.args, **request.kwargs)
            except BaseException as exc:
                request.error = exc
            finally:
                self._answer(request)
                serviced += 1
        return serviced

    def close(self) -> None:
        """Release every waiter with a timeout. Fleet thread."""
        self._closed.set()
        while True:
            try:
                request = self._queue.get_nowait()
            except queue.Empty:
                break
            if not self._claim(request):
                continue
            request.error = BusTimeout("the fleet's service queue closed")
            self._answer(request)

    def reopen(self) -> None:
        """Serve again after a close. What starting the next wave does.

        The counterpart every other closable thing here has -- :meth:`StepWaiter
        .reopen`, :meth:`Rendezvous.reopen` -- and it was missing, which is only
        invisible while nothing tears a wave down and starts another against the
        same queue. A driver that does gets a queue that refuses every call, and
        the refusal looks exactly like the wave having ended.
        """
        self._closed.clear()

    def drop_world(self, world: int) -> None:
        """Release one world's waiters, for a world whose episode has ended."""
        held: list[_Request] = []
        while True:
            try:
                request = self._queue.get_nowait()
            except queue.Empty:
                break
            if request.world == int(world):
                if not self._claim(request):
                    continue
                request.error = BusTimeout(f"world {world} finished while a call was pending")
                self._answer(request)
            else:
                held.append(request)
        for request in held:
            self._queue.put(request)

    def _claim(self, request: _Request) -> bool:
        """Take a request to service or release, unless its poster withdrew it."""
        with self._claims:
            if request.state != "queued":
                return False
            request.state = "claimed"
            return True

    def _answer(self, request: _Request) -> None:
        """Unpark a claimed request's thread and wake it, unless it gave up waiting."""
        with self._claims:
            answered = request.state == "claimed"
            if answered:
                request.state = "answered"
        if answered:
            self._unpark(request)
        request.done.set()

    def _unpark(self, request: _Request) -> None:
        if self.parks is not None:
            self.parks.unpark(int(request.world), request.ident)

    @property
    def closed(self) -> bool:
        return self._closed.is_set()


class StepWaiter:
    """Parks worlds waiting for the fleet's clock to advance.

    One condition for the whole fleet: the fleet notifies once per publish and
    every waiting world re-checks, rather than each world polling. The
    single-world bus polls at a millisecond because there is one waiter; five
    hundred pollers would be five hundred threads waking constantly to read the
    same integer.
    """

    def __init__(self, parks: Any = None) -> None:
        self._cond = threading.Condition()
        self._step = 0
        self._closed = False
        self.parks = parks
        """The fleet's park table, or ``None``. See :class:`.barrier.ParkTable`."""
        self._waiting: dict[tuple[int, int], int] = {}
        """Per ``(world, thread)`` waiting here, the step it is waiting past.

        Kept because :meth:`notify` has to unpark them *before* it wakes them,
        and a shared condition does not say who is on it. Keyed by thread as
        well as world, as every other table in this stack is: a world is not
        one thread, and a helper's wait overwriting the principal's entry left
        the principal parked in the table after it had been woken -- a world
        reading as parked while its script ran, which is the one thing the
        barrier exists to rule out."""

    def notify(self, step: int) -> None:
        with self._cond:
            self._step = int(step)
            woken = [(world, ident) for (world, ident), after in self._waiting.items() if self._step > after]
        # Outside the lock, and before the notify: a world whose deadline this
        # step satisfies is about to run, so it must stop reading as parked
        # first. See :mod:`.barrier`.
        self._unpark(woken)
        with self._cond:
            self._cond.notify_all()

    def wait(self, *, after: int, timeout: float, world: int = -1) -> int:
        """Block until the fleet publishes a step later than ``after``.

        Args:
            after: The step already seen.
            timeout: Seconds to wait.
            world: Which world is waiting, so the barrier can account for it.
                ``-1`` is a caller with no world -- a test, or the fleet itself.
        """
        with park_if(self.parks, world):
            with self._cond:
                if self._closed:
                    raise BusTimeout("the fleet closed while waiting for a step")
                if world >= 0:
                    # The waiting thread as well as its world: a world can have
                    # more than one thread in a fleet wait, and waking the wrong
                    # one releases a park its owner is still inside. See
                    # :class:`~.barrier.ParkTable`.
                    self._waiting[(int(world), threading.get_ident())] = int(after)
                try:
                    if not self._cond.wait_for(lambda: self._closed or self._step > int(after), timeout=timeout):
                        raise BusTimeout(f"no step past {after} within the deadline")
                    if self._closed:
                        raise BusTimeout("the fleet closed while waiting for a step")
                    return self._step
                finally:
                    if world >= 0:
                        self._waiting.pop((int(world), threading.get_ident()), None)

    def close(self) -> None:
        """Wake every parked world at once, with a timeout.

        This is the *only* thing that releases a thread already inside
        :meth:`wait` -- one world's bus closing does not, because the condition
        is shared by the whole fleet. Tearing a wave down therefore has to close
        the waiter *before* joining any graph thread: otherwise each join waits
        out its own timeout for a thread that cannot wake until the waiter's own
        deadline expires, and a 32-world teardown takes minutes instead of no
        time at all.
        """
        with self._cond:
            self._closed = True
            waiting = list(self._waiting.items())
        self._unpark([(world, ident) for (world, ident), _after in waiting])
        with self._cond:
            self._cond.notify_all()

    def _unpark(self, waiters: list[tuple[int, int]]) -> None:
        if self.parks is None:
            return
        for world, ident in waiters:
            self.parks.unpark(int(world), ident)

    def reopen(self) -> None:
        """Serve again after a close. What starting the next wave does."""
        with self._cond:
            self._closed = False
            self._step = 0
            self._waiting = {}


class WorldBus:
    """The single-world bus surface, bound to one world of a fleet.

    Implements :class:`~robosimstudio.core.bus.Bus`, which is the contract a
    graph tool actually depends on -- and implements it *without* ``pump`` or
    ``publish``, because those are the owner side and here the fleet is the
    owner. A view onto a shared service queue rather than a transport of its
    own, which is the whole difference between this and ``SimBus``.

    Args:
        world: Which world this bus speaks for. Bound here and nowhere above, so
            nothing a script or a suite tool touches mentions it.
        snapshots: The fleet's snapshot table.
        service: The fleet's shared service queue.
        waiter: The fleet's step waiter.
        timeout: How long a worker waits [s]. Longer than the single-world
            default by default, because a fleet's step is slower and a wave has
            many threads sharing the interpreter -- the failure to avoid is
            calling a healthy but loaded fleet dead.
    """

    def __init__(
        self,
        world: int,
        *,
        snapshots,
        service: ServiceQueue,
        waiter: StepWaiter,
        timeout: float = 120.0,
        parks: Any = None,
        rendezvous: Any = None,
    ) -> None:
        self.world = int(world)
        self.timeout = float(timeout)
        self._snapshots = snapshots
        self._service = service
        self._waiter = waiter
        self._closed = threading.Event()
        self.parks = parks
        self.rendezvous = rendezvous
        """The fleet's rendezvous, or ``None``.

        A fourth cost tier above the three this class's docstring names, and the
        cheapest of them per world: a request that joins a group is answered in
        one call for the whole wave. It is reached through :meth:`batched`
        rather than through :meth:`call`, because the two differ in *when* they
        are serviced -- a marshalled call is pumped between steps, a batched one
        waits for the barrier, which is what makes the group complete."""

    # ------------------------------------------------------------ worker side
    def call(self, fn, *args, **kwargs) -> Any:
        """Run ``fn`` on the fleet's thread and return its result."""
        if self._closed.is_set():
            raise BusTimeout(f"world {self.world}'s bus is closed")
        return self._service.post(self.world, fn, args, kwargs, self.timeout)

    @property
    def snapshot(self) -> SimSnapshot:
        """This world's most recent published state. Any thread."""
        return self._snapshots.view(self.world)

    def wait_for_step(self, *, after: int, timeout: float | None = None) -> SimSnapshot:
        """Block until the fleet publishes a snapshot later than ``after``."""
        if self._closed.is_set():
            raise BusTimeout(f"world {self.world}'s bus is closed")
        self._waiter.wait(after=after, timeout=self.timeout if timeout is None else timeout, world=self.world)
        return self.snapshot

    def batched(self, kind: str, payload: Any, timeout: float | None = None) -> Any:
        """Join the next group of ``kind`` and return this world's answer.

        Raises:
            KeyError: When no adapter serves this kind, which is a caller bug --
                check :meth:`batchable` first and fall back to :meth:`call`.
                Refusing rather than quietly marshalling, because a kind nobody
                batches is a kind whose cost nobody has looked at.
        """
        if self._closed.is_set():
            raise BusTimeout(f"world {self.world}'s bus is closed")
        if self.rendezvous is None:
            raise KeyError(f"world {self.world}'s bus has no rendezvous for {kind!r}")
        with park_if(self.parks, self.world):
            return self.rendezvous.post(self.world, kind, payload, self.timeout if timeout is None else timeout)

    def batchable(self, kind: str) -> bool:
        """Whether ``kind`` has an adapter, so :meth:`batched` will be answered."""
        return self.rendezvous is not None and self.rendezvous.knows(kind)

    # ------------------------------------------------------------- fleet side
    def close(self) -> None:
        """Stop serving this world and release whatever it had pending."""
        self._closed.set()
        self._service.drop_world(self.world)
        if self.rendezvous is not None:
            self.rendezvous.drop_world(self.world)

    @property
    def closed(self) -> bool:
        return self._closed.is_set()
