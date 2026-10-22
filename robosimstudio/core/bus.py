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
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["Bus", "BusClosed", "BusTimeout", "SimBus", "SimSnapshot"]


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
        self.owner_thread: int | None = None
        """Set on the first :meth:`pump`, so a re-entrant call can be detected."""

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
        request = _Request(fn=fn, args=args, kwargs=kwargs)
        with self._lock:
            if self._closed.is_set():
                raise BusClosed("the simulation bus closed: the episode ended")
            self._queue.put(request)
        if not request.done.wait(timeout):
            with self._lock:
                withdrawn = request.state == "queued"
                if withdrawn:
                    request.state = "cancelled"
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
        sleeping for a guessed interval.
        """
        deadline = time.perf_counter() + (self.timeout if timeout is None else timeout)
        while self._snapshot.step <= after:
            if self._closed.is_set():
                raise BusClosed("the simulation bus closed while waiting for a step: the episode ended")
            if time.perf_counter() > deadline:
                raise BusTimeout(f"no step past {after} within the deadline")
            time.sleep(0.001)
        return self._snapshot

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
                try:
                    request = self._queue.get_nowait()
                except queue.Empty:
                    break
                if request.state == "cancelled":
                    continue
                request.state = "running"
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
                serviced += 1
        return serviced

    def publish(self, snapshot: SimSnapshot) -> None:
        """Publish the state of the step just taken. Owner thread."""
        self._snapshot = snapshot

    def close(self) -> None:
        """Stop serving: every queued request, and every later call, raises :class:`BusClosed`.

        Owner thread. Idempotent. A request a pump is already running finishes
        and returns normally.
        """
        with self._lock:
            self._closed.set()
            while True:
                try:
                    request = self._queue.get_nowait()
                except queue.Empty:
                    break
                request.error = BusClosed("the simulation bus closed: the episode ended")
                request.done.set()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()
