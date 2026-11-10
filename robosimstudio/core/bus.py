# SPDX-License-Identifier: Apache-2.0
"""Marshalling simulator access onto the thread that owns it.

The problem: a graph runtime runs node bodies on worker threads, and Warp has a
single writer. A node that calls ``sim.get_object_pose`` from a worker while the
main loop is mid-step is not a race in theory -- it is a wrong answer or a crash.

The solution is not a lock. A lock held across a Warp launch stalls the
simulation for as long as a node takes to think, which for a node that calls a
vision model is seconds. Instead: workers **post requests and wait**; the owner
thread **pumps** the queue once per control step, between steps, where it is
safe to touch anything; and a snapshot of the observable state is published at
the same point so read-only queries are answered from the snapshot without
waiting at all.

A watchdog bounds the wait. A worker whose request is never pumped -- because the
owner died, or the episode ended -- gets a timeout rather than hanging the graph
forever, and the runtime turns that into a node error its own error handling can
route.

**Lockstep.** By default the owner steps whatever the workers are doing, so how
many control steps pass while a node thinks depends on how fast the host is.
An owner that wants the simulation to stand still while the graph runs host
code asks the bus instead: :meth:`SimBus.await_parked` blocks until the graph's
own thread -- its *principal*, named by :meth:`SimBus.adopt` -- is **parked**,
which it is exactly while it has a request queued or is blocked in
:meth:`SimBus.wait_for_step` on a step not yet published. The discipline is the
fleet's (:mod:`robosimstudio_policy_agent.fleet.barrier`), and both halves hold
here by construction because "parked" is *derived* from the queue and the
published step under one lock rather than kept as a flag beside them: a request
stops counting the instant a pump takes it, and a wait the instant the step it
waits for is published -- before either worker is woken. Any other thread's
requests (gap's observation poller, say) are answered meanwhile, but its waits
neither hold the clock nor release it.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["Bus", "BusClosed", "BusTimeout", "SimBus", "SimSnapshot"]

_log = logging.getLogger(__name__)

_AWAIT_CEILING_S = 0.25
"""Longest a single condition wait in :meth:`SimBus.await_parked` lasts before it
looks again. A ceiling, not a cadence: every park, publish and close notifies."""


class BusTimeout(RuntimeError):
    """A request was not serviced before its deadline."""


class BusClosed(BusTimeout):
    """The episode ended while this request was in flight.

    A subclass rather than a sibling, so every existing ``except BusTimeout``
    still catches it -- but a distinct type, because the two say opposite things
    about what went wrong. A timeout means something was too *slow*: a solve that
    overran, a pump that stalled, a step that never came. A close means the
    episode simply *finished* underneath a node that was still working, which is
    routine at the end of a successful run and is not a fault of the node.

    Worth separating because the runtime cannot tell the difference on its own.
    A node interrupted this way reports a failure and routes to ``on_error``, so
    a healthy episode leaves a node error in its trace -- typically when the
    evaluator declares the task done while a node is still driving a withdrawal.
    Anything reading traces to find real failures has to be able to filter these
    out, and matching on a message string is not a way to do that.

    ``terminal`` says the same thing to the graph runtime, which reads the flag
    off the exception type rather than importing this module. A closed bus is
    not a step that failed and can be retried -- the world is gone -- so the
    runtime re-raises instead of taking the subgraph's ``on_error`` exit. Left
    routable, a graph with a recovery loop answers "the episode ended" by
    calling another tool, which raises this again, and goes round until the
    runtime's super-step cap stops it -- thousands of identical node failures
    from one episode that had in fact finished.
    """

    terminal = True


@dataclass
class SimSnapshot:
    """The observable state as of one control step.

    Published by the owner thread and read by workers without synchronisation:
    it is replaced wholesale rather than mutated, so a worker either sees the
    previous one or the next one and never a half-written mixture.

    Attributes:
        step: Control steps since reset.
        sim_time: Simulated seconds since reset [s].
        obs: The observation dict that step produced.
        body_q: Body transforms, ``[body_count, 7]``.
        info: The step's info, including ``success`` and the metrics.
        frames: The step's rendered camera frames, if any.
    """

    step: int = 0
    sim_time: float = 0.0
    obs: dict[str, Any] = field(default_factory=dict)
    body_q: Any = None
    info: dict[str, Any] = field(default_factory=dict)
    frames: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class _Request:
    fn: Any
    args: tuple
    kwargs: dict
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None
    state: str = "queued"
    """``queued``, then ``running`` once a pump takes it, or ``cancelled`` once
    its worker has given up -- a cancelled request is dropped unrun."""
    ident: int = 0
    """The thread that posted it, so a lockstep owner can tell the graph's own
    request from a helper thread's."""


@runtime_checkable
class Bus(Protocol):
    """What a graph tool needs of a simulator, whatever is on the other side.

    Two things implement this and they are not variations of one another:
    :class:`SimBus` *owns* its transport -- a queue, a snapshot, an owner thread
    it pumps -- while a fleet's ``WorldBus`` is a **view** onto a service queue
    shared by every world, bound to one of them. Neither is the batched twin of
    the other; they are two answers to "how does a worker reach the simulator".

    What is one thing is the surface above, which is why it is written down
    here rather than left to be inferred from whichever implementation somebody
    read first. Note what is *absent*: ``pump`` and ``publish`` are the **owner**
    side and belong to whoever drives the loop -- ``SimBus`` has them because it
    is also its own owner, and a fleet's world bus does not because the fleet
    is. A tool that reached for either would be a tool that had confused
    consuming the simulator with running it.
    """

    @property
    def snapshot(self) -> SimSnapshot:
        """The most recent published state. Safe from any thread."""
        ...

    @property
    def closed(self) -> bool:
        """Whether the episode this bus served has ended."""
        ...

    def call(self, fn, *args, **kwargs) -> Any:
        """Run ``fn`` where the simulator lives and return its result."""
        ...

    def wait_for_step(self, *, after: int, timeout: float | None = None) -> SimSnapshot:
        """Block until a snapshot later than ``after`` is published."""
        ...

    def close(self) -> None:
        """Stop serving. Idempotent."""
        ...


class SimBus:
    """Mediates every simulator access between graph workers and the owner.

    Args:
        timeout: How long a worker waits for its request [s]. Long enough that
            a slow control step is not an error, short enough that a dead owner
            is noticed.
    """

    def __init__(self, *, timeout: float = 30.0) -> None:
        self.timeout = float(timeout)
        self._queue: queue.Queue[_Request] = queue.Queue()
        self._snapshot = SimSnapshot()
        self._closed = threading.Event()
        self._lock = threading.Lock()
        """Makes "closed?" and "enqueue" one step against :meth:`close`, and
        "take" and "cancel" one step between :meth:`pump` and a timed-out worker."""
        self._changed = threading.Condition(self._lock)
        """Notified on every park, publish and close -- what :meth:`wait_for_step`
        and :meth:`await_parked` block on instead of polling."""
        self.owner_thread: int | None = None
        """Set on the first :meth:`pump`, so a re-entrant call can be detected."""
        # Lockstep bookkeeping, all under ``_lock``. Kept whatever the owner's
        # clock: a free-running owner simply never asks.
        self._principal: int | None = None
        self._departed = False
        self._queued: dict[int, int] = {}
        """Per posting thread, its requests no pump has taken yet."""
        self._waiting: dict[int, int] = {}
        """Per thread blocked in :meth:`wait_for_step`, the step it waits to see passed."""
        self._lockstep = False
        """Set once an owner has asked :meth:`await_parked`: this bus's clock is lockstep."""
        self._warned_helper_wait = False

    # ------------------------------------------------------------ worker side
    def call(self, fn, *args, **kwargs) -> Any:
        """Run ``fn`` on the owner thread and return its result.

        Called from a worker. Blocks until the owner pumps, or raises
        :class:`BusTimeout` -- and a request that times out before a pump takes
        it is withdrawn, so it never runs later behind the caller's back. One
        already running when the deadline passes finishes, and its result is
        discarded. Called *on* the owner thread it runs inline, so a tool that
        is sometimes called from either side works both ways.
        """
        return self._call(self.timeout, False, fn, args, kwargs)

    def call_starting_within(self, timeout: float, fn, *args, **kwargs) -> Any:
        """Run ``fn`` on the owner thread if a pump takes it within ``timeout`` seconds, and return its result.

        For a caller that stops waiting then: one not taken in time is
        withdrawn and :class:`BusTimeout` raised, so it never runs; one taken in
        time runs to its end, however long that is, and its result is
        returned -- never a timeout for something that ran.
        """
        return self._call(timeout, True, fn, args, kwargs)

    def _call(self, timeout: float, finish_started: bool, fn, args: tuple, kwargs: dict) -> Any:
        if self.owner_thread == threading.get_ident():
            if self._closed.is_set():
                raise BusClosed("the simulation bus closed: the episode ended")
            return fn(*args, **kwargs)
        request = _Request(fn=fn, args=args, kwargs=kwargs, ident=threading.get_ident())
        # Queued and counted under one lock: from this instant the poster reads
        # as parked to a lockstep owner, and it was never running-with-work-
        # handed-over in between (the fleet's "park before handing work over").
        with self._changed:
            if self._closed.is_set():
                raise BusClosed("the simulation bus closed: the episode ended")
            self._queue.put(request)
            self._queued[request.ident] = self._queued.get(request.ident, 0) + 1
            self._changed.notify_all()
        if not request.done.wait(timeout):
            with self._lock:
                withdrawn = request.state == "queued"
                if withdrawn:
                    request.state = "cancelled"
                    self._unqueue(request.ident)
                serviced = request.done.is_set()  # finished between the wait and the lock
            if finish_started and not withdrawn:
                request.done.wait()  # taken in time: it runs to its end
            elif not serviced:
                raise BusTimeout(f"{getattr(fn, '__name__', fn)} was not serviced within {timeout:.0f} s")
        if request.error is not None:
            raise request.error
        return request.result

    @property
    def snapshot(self) -> SimSnapshot:
        """The most recent published state. Safe to read from any thread."""
        return self._snapshot

    def wait_for_step(self, *, after: int, timeout: float | None = None) -> SimSnapshot:
        """Block until a snapshot later than ``after`` is published.

        What a node that has just commanded an action waits on, rather than
        sleeping for a guessed interval. A condition the owner notifies on every
        publish, not a poll: a graph thread waiting out a settle costs no CPU.

        While it waits, the calling thread is *parked* -- under a lockstep owner
        this wait is what lets the clock advance (see :meth:`await_parked`).
        """
        limit = self.timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + limit
        ident = threading.get_ident()
        with self._changed:
            if self._snapshot.step > after:
                return self._snapshot
            self._waiting[ident] = int(after)
            self._changed.notify_all()
            if self._lockstep and self._principal not in (None, ident) and not self._warned_helper_wait:
                # Only the principal's waits release a lockstep clock (as only a
                # world's principal's release the fleet's barrier), so a wait on
                # any other thread -- a node gap runs on a pool thread in a
                # parallel super-step, a streaming node -- gets its step only
                # when the graph's own thread next parks. Said once, because it
                # is otherwise a silent stall that ends at this wait's timeout.
                self._warned_helper_wait = True
                _log.warning(
                    "lockstep: thread %r waits for a step but is not the graph's own thread; only that "
                    "thread's parks advance the clock, so this wait is answered when it next parks "
                    "(parallel or streaming nodes that wait on the simulator are not lockstep-safe)",
                    threading.current_thread().name,
                )
            try:
                while self._snapshot.step <= after:
                    if self._closed.is_set():
                        raise BusClosed("the simulation bus closed while waiting for a step: the episode ended")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise BusTimeout(f"no step past {after} within the deadline")
                    self._changed.wait(remaining)
            finally:
                self._waiting.pop(ident, None)
            return self._snapshot

    # --------------------------------------------------------- lockstep, graph
    def adopt(self, ident: int | None = None) -> None:
        """Name the thread whose parks hold and release a lockstep clock. Called by it.

        From the graph thread itself, before it runs anything, as the fleet's
        :meth:`~robosimstudio_policy_agent.fleet.barrier.ParkTable.adopt` is: the
        runtime's own helper threads only exist once the graph has started, so
        none of them can be mistaken for it. Harmless under a free-running owner,
        which never asks.
        """
        with self._changed:
            self._principal = threading.get_ident() if ident is None else int(ident)
            self._departed = False
            self._changed.notify_all()

    def depart(self) -> None:
        """Say the principal has finished and will never park again. Called by it, last."""
        with self._changed:
            self._departed = True
            self._changed.notify_all()

    # ------------------------------------------------------------- owner side
    def pump(self, *, budget: int = 64) -> int:
        """Service pending requests. Called by the owner, between steps.

        Args:
            budget: Most requests to service in one pump, so a worker issuing
                them in a tight loop cannot starve the simulation.

        Returns:
            How many were serviced. Requests their workers already gave up on
            are dropped unrun and not counted.

        Raises:
            KeyboardInterrupt, SystemExit: Raised by a request's ``fn``, they
                propagate here on the owner thread rather than being handed to
                the worker. The bus closes first, so that worker and every
                queued one is released with :class:`BusClosed`.
        """
        self.owner_thread = threading.get_ident()
        serviced = 0
        while serviced < budget:
            with self._lock:
                request = self._take()
            if request is None:
                break
            self._service(request)
            serviced += 1
        return serviced

    def parked(self) -> str:
        """What the principal is doing, as :meth:`await_parked` reports it, without blocking."""
        with self._lock:
            return self._parked_state()

    def await_parked(self, timeout: float) -> str:
        """Block until the principal is parked. The lockstep owner's half. Owner thread.

        Requests from any *other* thread are serviced while this waits -- they
        were posted by a parked thread, and servicing one advances nothing, so a
        helper such as an observation poller is never starved by a graph that is
        thinking. The principal's own request is left queued for the caller,
        which decides what it costs in steps (:meth:`service_principal`).

        Args:
            timeout: Wall-clock seconds to wait for the principal to park.

        Returns:
            ``"request"`` -- the principal has a request queued; ``"wait"`` -- it
            is blocked in :meth:`wait_for_step` on a step not yet published;
            ``"departed"`` -- it has finished (:meth:`depart`); ``"closed"`` --
            the bus closed; ``""`` -- it is still running host code at the
            deadline, which is the caller's watchdog to act on.
        """
        self.owner_thread = threading.get_ident()
        self._lockstep = True
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            with self._changed:
                state = self._parked_state()
                if state:
                    return state
                # The principal has nothing queued, or the state would say so:
                # whatever is queued is a helper's.
                request = self._take()
                if request is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return ""
                    self._changed.wait(min(remaining, _AWAIT_CEILING_S))
                    continue
            self._service(request)

    def service_principal(self) -> bool:
        """Service queued requests up to and including the principal's one. Owner thread.

        Exactly one of the principal's, so the steps a lockstep turn costs do not
        depend on how quickly the graph posts its next request. Anything a
        helper queued ahead of it is serviced first, in order.

        Returns:
            Whether a principal request was serviced.
        """
        self.owner_thread = threading.get_ident()
        while True:
            with self._lock:
                request = self._take()
            if request is None:
                return False
            self._service(request)
            if request.ident == self._principal:
                return True

    def publish(self, snapshot: SimSnapshot) -> None:
        """Publish the state of the step just taken. Owner thread.

        Wakes every :meth:`wait_for_step` it satisfies -- and, being set under
        the lock those waits are judged under, un-parks them in the same instant.
        """
        with self._changed:
            self._snapshot = snapshot
            self._changed.notify_all()

    def close(self) -> None:
        """Stop serving: every queued request, and every later call, raises :class:`BusClosed`.

        Owner thread. Idempotent. A request a pump is already running finishes
        and returns normally.
        """
        with self._changed:
            self._closed.set()
            while True:
                try:
                    request = self._queue.get_nowait()
                except queue.Empty:
                    break
                request.error = BusClosed("the simulation bus closed: the episode ended")
                request.done.set()
            self._queued.clear()
            self._changed.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    # ---------------------------------------------------------------- helpers
    def _take(self) -> _Request | None:
        """The next live queued request, now ``running``. Lock held.

        Taking it is what un-parks its poster -- before it runs, so before its
        worker can wake ("unpark before waking").
        """
        while True:
            try:
                request = self._queue.get_nowait()
            except queue.Empty:
                return None
            if request.state == "cancelled":
                continue
            request.state = "running"
            self._unqueue(request.ident)
            return request

    def _service(self, request: _Request) -> None:
        """Run a taken request on this, the owner, thread, and wake its worker."""
        try:
            request.result = request.fn(*request.args, **request.kwargs)
        except Exception as exc:
            request.error = exc
        except BaseException:
            request.error = BusClosed("the simulation bus closed: the owner thread was interrupted")
            self.close()
            raise
        finally:
            request.done.set()

    def _unqueue(self, ident: int) -> None:
        """One fewer request queued by ``ident``. Lock held."""
        left = self._queued.get(ident, 0) - 1
        if left > 0:
            self._queued[ident] = left
        else:
            self._queued.pop(ident, None)

    def _parked_state(self) -> str:
        """The principal's state as :meth:`await_parked` reports it. Lock held."""
        if self._closed.is_set():
            return "closed"
        if self._departed:
            return "departed"
        who = self._principal
        if who is None:
            return ""
        if self._queued.get(who, 0) > 0:
            return "request"
        after = self._waiting.get(who)
        if after is not None and self._snapshot.step <= after:
            return "wait"
        return ""
