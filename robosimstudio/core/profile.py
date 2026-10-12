# SPDX-License-Identifier: Apache-2.0
"""Where a stepping loop's time goes, measured rather than guessed.

A loop that steps a solver and then draws, records and photographs what it
stepped has two clocks in it, and they do not add up the way they look like they
should. The GPU's work is asynchronous, so the host runs ahead of it -- until
the first ``.numpy()``, which is a synchronous device-to-host copy on the null
stream and drains the whole queue. Everything after that point is paid for
serially. So the loop's period is *GPU time plus host time*, not the larger of
the two, and a millisecond taken off the host side is a millisecond gained.

That also means a naive wall-clock profile lies about *where* the time went: the
first phase to touch a device array is charged for every kernel the step
launched. This module therefore carries three instruments rather than one,
because each is blind to what the others see:

* **Wall clock** (:class:`Profiler`) -- ``perf_counter`` around named phases.
  Perturbs nothing. Tells you the period and its breakdown, with the caveat
  above about which phase gets charged for the stall.
* **Device events** (:class:`DeviceClock`) -- a pair of CUDA events around the
  work, harvested one iteration later with ``synchronize=False`` so no
  synchronisation is added. Tells you how much of the period the device was
  actually busy, which is the number that says whether a loop is device-bound
  at all.
* **Readback tracing** (:class:`SyncTracer`) -- counts and times the
  device-to-host copies themselves, attributed to the phase that made them.
  This is the one that turns "something in here synchronises a lot" into a list
  of call sites.

:func:`device_pass` is a fourth, coarser view: warp's own kernel and memcpy
timing over a fixed window, which needs no cooperation from the code being
measured but synchronises at the window edges.

**What this deliberately does not use** is ``wp.ScopedTimer(synchronize=True)``.
It synchronises on entry and on exit, which in a loop whose problem is
synchronisation would be the instrument creating the disease it came to
diagnose.

Phases are named as dotted paths by their call sites -- ``"draw.garments.fuzz"``
-- rather than discovered from a runtime stack. That keeps a span to two
``perf_counter`` calls and a couple of attribute writes, and it lets the report
subtract children from parents without the caller having to nest anything. The
cost of a disabled span is one attribute lookup and two no-op method calls;
:data:`NULL` is the disabled profiler every caller holds by default, so code
that is not being profiled pays essentially nothing and needs no ``if``.

A span is **not reentrant**: one live timing record exists per name, so a phase
must not appear inside itself. Loops name their phases per subject
(``"drain.arm:left"``) instead of recursing.
"""

from __future__ import annotations

import json
import time
from collections import deque
from contextlib import contextmanager
from typing import Any

import numpy as np
import warp as wp

__all__ = [
    "NULL",
    "DeviceClock",
    "PhaseStats",
    "Profiler",
    "SyncTracer",
]

MODES = ("wall", "device", "syncs")
"""What can be asked for. ``wall`` is implied by any of the others."""

_RING = 4096
"""Samples kept per phase for the percentiles. About five minutes at 15 Hz."""

_SMOOTHING = 0.3
"""EMA weight on the newest sample, matching the live render budget's."""


class PhaseStats:
    """What one named phase cost, summarised two ways.

    A mean hides exactly the thing worth finding. The first frames of a browser
    viewer cost tens of times what the hundredth does, a re-capture stalls one
    step in a thousand, and a stutter is a p95 event by definition -- all of
    which average away to nothing. So this keeps a full ring of recent samples
    for order statistics *and* an exponential average for the live readout,
    which wants a number that moves rather than one that is correct.

    Attributes:
        calls: Times the phase was entered since the last :meth:`reset`.
        total: Seconds spent in it.
        peak: The worst single sample, in seconds.
        ema: Smoothed seconds per call, for a readout that updates live.
        reads: Device-to-host copies attributed to it, when tracing is on.
        read_bytes: Bytes those copies moved.
    """

    __slots__ = ("_ring", "calls", "ema", "peak", "read_bytes", "reads", "total")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.total = 0.0
        self.peak = 0.0
        self.ema = 0.0
        self.reads = 0
        self.read_bytes = 0
        self._ring: deque[float] = deque(maxlen=_RING)

    def record(self, seconds: float) -> None:
        self.calls += 1
        self.total += seconds
        if seconds > self.peak:
            self.peak = seconds
        self.ema = seconds if self.calls == 1 else (1.0 - _SMOOTHING) * self.ema + _SMOOTHING * seconds
        self._ring.append(seconds)

    def quantiles(self) -> tuple[float, float]:
        """Median and p95 of the retained samples, in seconds. ``(0, 0)`` if empty."""
        if not self._ring:
            return 0.0, 0.0
        samples = np.fromiter(self._ring, dtype=np.float64, count=len(self._ring))
        return float(np.median(samples)), float(np.percentile(samples, 95))

    def as_dict(self) -> dict[str, Any]:
        p50, p95 = self.quantiles()
        return {
            "calls": self.calls,
            "total_s": self.total,
            "mean_s": self.total / self.calls if self.calls else 0.0,
            "p50_s": p50,
            "p95_s": p95,
            "peak_s": self.peak,
            "reads": self.reads,
            "read_bytes": self.read_bytes,
        }


class _Span:
    """One phase's timing record, reused across entries.

    Allocated once per name and handed back by :meth:`Profiler.phase`, so a
    profiled loop allocates nothing per iteration. This is why a span cannot
    nest inside itself -- there is one ``_t0`` to hold the entry time.
    """

    __slots__ = ("_outer", "_profiler", "_t0", "name", "stats")

    def __init__(self, profiler: Profiler, name: str) -> None:
        self._profiler = profiler
        self.name = name
        self.stats = PhaseStats()
        self._t0 = 0.0
        self._outer: _Span | None = None

    def __enter__(self) -> _Span:
        # Save the enclosing span rather than clearing on exit: a nested phase
        # that reset the profiler's open span to None would leave every readback
        # its parent made afterwards attributed to nobody.
        self._outer = self._profiler._open
        self._profiler._open = self
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stats.record(time.perf_counter() - self._t0)
        self._profiler._open = self._outer
        self._outer = None


class _NullSpan:
    """The disabled span: entering and leaving it does nothing at all."""

    __slots__ = ()

    def __enter__(self) -> _NullSpan:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class Profiler:
    """Named phases, their costs, and a report.

    Phase names are dotted paths supplied by the caller. The report derives the
    tree from the names, so a parent that also has children prints an
    ``(unattributed)`` remainder -- the part of it that no child claimed. That
    remainder is the honest way to show what has not been instrumented, and a
    large one means the breakdown should not be trusted yet.

    Args:
        modes: Which instruments to arm. Any non-empty set implies wall timing.
        device: Device for the event clock; ``None`` takes warp's current one.

    Attributes:
        steps: Control steps counted, for the per-step columns.
        phases: Name to :class:`PhaseStats`.
        notes: Free-form key/value context to carry into the report, for things
            like the scene census and the graph-capture counters.
    """

    enabled = True

    def __init__(self, modes: frozenset[str] | set[str] | None = None, *, device: Any = None) -> None:
        self.modes = frozenset(modes or ("wall",))
        self.steps = 0
        self.phases: dict[str, PhaseStats] = {}
        self.notes: dict[str, Any] = {}
        self._spans: dict[str, _Span] = {}
        self._open: _Span | None = None
        self._t0 = time.perf_counter()
        self.device = DeviceClock(device=device) if "device" in self.modes else None
        self.syncs = SyncTracer(self) if "syncs" in self.modes else None

    # ------------------------------------------------------------------ timing
    def phase(self, name: str):
        """A context manager timing ``name``. Reused, so not reentrant."""
        span = self._spans.get(name)
        if span is None:
            span = _Span(self, name)
            self._spans[name] = span
            self.phases[name] = span.stats
        return span

    def mark_step(self) -> None:
        """One control step happened. Drives the per-step columns."""
        self.steps += 1

    def note(self, **kwargs: Any) -> None:
        """Record context to print beside the numbers."""
        self.notes.update(kwargs)

    def attribute_read(self, nbytes: int, seconds: float) -> None:
        """Charge one device-to-host copy to whichever phase is open."""
        span = self._open
        stats = span.stats if span is not None else self.phases.setdefault("(outside any phase)", PhaseStats())
        stats.reads += 1
        stats.read_bytes += nbytes
        _ = seconds

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        """Arm the instruments that have to hook something."""
        if self.syncs is not None:
            self.syncs.install()

    def stop(self) -> None:
        """Disarm. Safe to call twice; leaves the numbers intact."""
        if self.syncs is not None:
            self.syncs.remove()

    def reset(self) -> None:
        """Forget every sample. For discarding a warm-up window."""
        for stats in self.phases.values():
            stats.reset()
        self.steps = 0
        self._t0 = time.perf_counter()
        if self.device is not None:
            self.device.reset()

    @property
    def wall(self) -> float:
        """Seconds since construction or the last :meth:`reset`."""
        return time.perf_counter() - self._t0

    # ---------------------------------------------------------------- reporting
    def _unattributed(self) -> dict[str, float]:
        """Per parent phase, the seconds its named children did not account for."""
        out: dict[str, float] = {}
        for name, stats in self.phases.items():
            children = sum(
                child.total
                for other, child in self.phases.items()
                if other.startswith(f"{name}.") and "." not in other[len(name) + 1 :]
            )
            if children:
                out[name] = stats.total - children
        return out

    def as_dict(self) -> dict[str, Any]:
        """Everything measured, as plain data for a JSON sidecar."""
        return {
            "schema": 1,
            "modes": sorted(self.modes),
            "wall_s": self.wall,
            "steps": self.steps,
            "phases": {name: stats.as_dict() for name, stats in sorted(self.phases.items())},
            "unattributed_s": self._unattributed(),
            "device": self.device.as_dict() if self.device is not None else None,
            "notes": self.notes,
        }

    def write(self, path: str) -> None:
        with open(path, "w") as handle:
            json.dump(self.as_dict(), handle, indent=2, default=str)

    def report(self) -> str:
        """The table, ordered as the loop runs and indented by name depth."""
        wall = max(self.wall, 1e-9)
        steps = max(self.steps, 1)
        unattributed = self._unattributed()
        lines = [
            f"[profile] wall={wall:.1f} s  steps={self.steps}  "
            f"({self.steps / wall:.1f} steps/s)  modes={','.join(sorted(self.modes))}",
            "",
            f"  {'phase':<34}{'calls/step':>11}{'ms/call':>10}{'ms/step':>10}{'p95 ms':>9}{'% wall':>8}{'D2H':>7}{'MB':>8}",
        ]
        for name in sorted(self.phases):
            stats = self.phases[name]
            if not stats.calls:
                continue
            _, p95 = stats.quantiles()
            depth = name.count(".")
            label = "  " * depth + name.rsplit(".", 1)[-1]
            lines.append(
                f"  {label:<34}{stats.calls / steps:>11.2f}"
                f"{1e3 * stats.total / stats.calls:>10.2f}"
                f"{1e3 * stats.total / steps:>10.2f}"
                f"{1e3 * p95:>9.2f}"
                f"{100.0 * stats.total / wall:>8.1f}"
                f"{stats.reads:>7}"
                f"{stats.read_bytes / 2**20:>8.2f}"
            )
            left = unattributed.get(name)
            if left is not None and abs(left) > 1e-6:
                gap = "  " * (depth + 1) + "(unattributed)"
                lines.append(
                    f"  {gap:<34}{'':>11}{'':>10}{1e3 * left / steps:>10.2f}{'':>9}{100.0 * left / wall:>8.1f}"
                )
        if self.device is not None:
            busy = self.device.busy_seconds
            lines += [
                "",
                f"  device  busy {1e3 * busy / steps:.2f} ms/step   utilisation {100.0 * busy / wall:.1f}%"
                f"   samples {self.device.samples}",
            ]
        if self.notes:
            lines += ["", "  " + "  ".join(f"{k}={v}" for k, v in sorted(self.notes.items()))]
        return "\n".join(lines)


class _NullProfiler(Profiler):
    """The profiler every caller holds until someone asks for a real one.

    Every method is a no-op and :meth:`phase` hands back a shared do-nothing
    span, so instrumented code needs no ``if`` around its ``with`` blocks and
    costs an attribute lookup plus two empty calls when nobody is measuring.
    """

    enabled = False
    _NULL_SPAN = _NullSpan()

    def __init__(self) -> None:
        super().__init__(frozenset())

    def phase(self, name: str) -> _NullSpan:
        return self._NULL_SPAN

    def mark_step(self) -> None:
        return None

    def note(self, **kwargs: Any) -> None:
        return None

    def attribute_read(self, nbytes: int, seconds: float) -> None:
        return None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


NULL = _NullProfiler()
"""The disabled profiler. Default for every instrumented object."""


class DeviceClock:
    """How long the device was busy, without making anyone wait to find out.

    A pair of CUDA events is recorded around the work. Reading the elapsed time
    between them would normally block until the second one has happened, which
    is exactly the synchronisation this whole module exists to avoid adding --
    so the pairs go into a ring and are read back *later*, with
    ``synchronize=False``, by which time the loop has moved on and the events
    are long since complete. The cost at the call site is two event records,
    around two microseconds.

    The ring is deliberately small. If a harvest ever finds an event that has
    not completed, warp raises rather than blocking, and that sample is dropped
    -- a missing sample is a much better failure than a stalled loop.

    Args:
        device: Which device to record on; ``None`` takes warp's current one.
        depth: Event pairs in flight before the oldest is harvested.
    """

    def __init__(self, *, device: Any = None, depth: int = 4) -> None:
        self.device = device
        self.depth = max(2, int(depth))
        self._free: list[tuple[wp.Event, wp.Event]] = []
        self._pending: deque[tuple[wp.Event, wp.Event]] = deque()
        self.busy_seconds = 0.0
        self.samples = 0
        self.dropped = 0

    def reset(self) -> None:
        self.busy_seconds = 0.0
        self.samples = 0
        self.dropped = 0

    def _pair(self) -> tuple[wp.Event, wp.Event]:
        if self._free:
            return self._free.pop()
        return (
            wp.Event(device=self.device, enable_timing=True),
            wp.Event(device=self.device, enable_timing=True),
        )

    @contextmanager
    def measure(self):
        """Record events around a block of device work."""
        pair = self._pair()
        # On this timer's OWN stream. ``wp.record_event`` records on warp's
        # current device's stream, which is only this timer's device when the
        # two happen to agree -- see the note in
        # :meth:`~robosimstudio.core.readback.Readback.enqueue`, where the
        # same convenience killed every worker of a second-GPU run. ``device``
        # here defaults to None, meaning "warp's current one", and
        # ``wp.get_stream(None)`` is exactly what the convenience already does,
        # so this is a no-op on the default path and correct off it.
        stream = wp.get_stream(self.device)
        stream.record_event(pair[0])
        try:
            yield
        finally:
            stream.record_event(pair[1])
            self._pending.append(pair)
            if len(self._pending) > self.depth:
                self._harvest_one()

    def _harvest_one(self) -> None:
        pair = self._pending.popleft()
        try:
            self.busy_seconds += 1e-3 * wp.get_event_elapsed_time(pair[0], pair[1], synchronize=False)
            self.samples += 1
        except Exception:
            self.dropped += 1
        self._free.append(pair)

    def drain(self) -> None:
        """Harvest everything still in flight. For the end of a run."""
        while self._pending:
            self._harvest_one()

    def as_dict(self) -> dict[str, Any]:
        return {
            "busy_s": self.busy_seconds,
            "samples": self.samples,
            "dropped": self.dropped,
        }


class SyncTracer:
    """Counts the device-to-host copies, and says which phase paid for each.

    ``array.numpy()`` is where a stepping loop actually stalls, and neither a
    wall clock nor a device event can point at the caller: the wall clock
    charges whichever phase happened to touch a device array first, and the
    events only know about the device. So this wraps the method itself for the
    duration of a profiling run and attributes each copy to whatever phase is
    open at the time.

    It reaches into warp's internals to do it, which is a deliberate and
    contained bet: nothing on disk is modified, the wrapper is installed only
    while profiling and removed in :meth:`remove`, and if a warp upgrade moves
    the method then :attr:`available` comes back False and every other
    measurement in this module still works. An instrument that degrades to
    silence is fine; one that breaks the thing it measures is not.
    """

    def __init__(self, profiler: Profiler) -> None:
        self.profiler = profiler
        self.available = False
        self.count = 0
        self.nbytes = 0
        self.seconds = 0.0
        self._target: Any = None
        self._original: Any = None

    def install(self) -> None:
        if self._original is not None:
            return
        try:
            from warp._src.types import array as _array
        except Exception:
            self.available = False
            return

        original = _array.numpy
        tracer = self

        def numpy(self):
            t0 = time.perf_counter()
            out = original(self)
            elapsed = time.perf_counter() - t0
            # For a contiguous array the innermost stride is the element size,
            # which is all we need and avoids reaching for another private name.
            strides = getattr(self, "strides", None)
            nbytes = int(getattr(self, "size", 0)) * int(strides[-1]) if strides else 0
            tracer.count += 1
            tracer.nbytes += nbytes
            tracer.seconds += elapsed
            tracer.profiler.attribute_read(nbytes, elapsed)
            return out

        _array.numpy = numpy
        self._target = _array
        self._original = original
        self.available = True

    def remove(self) -> None:
        if self._original is None:
            return
        self._target.numpy = self._original
        self._original = None
        self._target = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "count": self.count,
            "bytes": self.nbytes,
            "seconds": self.seconds,
        }


def parse_modes(spec: str) -> Profiler:
    """Build a profiler from a comma-separated mode string. Empty gives :data:`NULL`."""
    modes = {token.strip() for token in spec.split(",") if token.strip()}
    if not modes:
        return NULL
    unknown = modes - set(MODES)
    if unknown:
        raise ValueError(f"unknown profile mode(s) {sorted(unknown)}; known: {list(MODES)}")
    return Profiler(modes | {"wall"})
