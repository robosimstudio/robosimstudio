# SPDX-License-Identifier: Apache-2.0
"""Whose turn it is on a GPU several workers share: physics together, a render alone.

MPS lets a pool's workers run compute at once, and for physics that pays: two
processes stepping 15 worlds each lose about a quarter of their rate to one
another and get 1.6 times the work done. A render beside physics is a bad trade
for both. The tracer and another worker's compute take the GPU in time slices,
so each pass of the one waits out slices of the other. Measured under MPS, one
worker rendering and another stepping idle physics, 15 worlds each:

=====================  =========  =================  =================================
                       A render   Beside the other   What one render costs the stepper
=====================  =========  =================  =================================
Alone                  143 ms     --                 --
Together               --         340 ms             128 ms of its stepping
A render at a time     --         165 ms, with wait  138 ms of its stepping
=====================  =========  =================  =================================

So taking turns halves the render and costs the other worker no more than sharing
did. On a whole ``cable_routing/L2`` cell, 15 worlds on each of 2 workers, with real
contact on the other side, a render went from 700 ms to 240 ms, and rendering from
11 % of a wave's wall time to 4 %. A step waits 3 ms for its turn, on average.

Physics still overlaps physics: every stepper holds the turn shared, and only a
render holds it alone. The worker waiting is at most one physics step behind,
because the turn is held per step. Nothing is simulated differently. A step takes
the same inputs and only starts a moment later on the wall clock, and under the
lockstep clock the simulation cannot see the wall clock.

Two ``flock`` files per device: ``gate`` is a turnstile, so a queue of steppers
always sharing ``use`` cannot starve a render waiting to have it alone. A process
that dies holding the turn gives it back with its file descriptors. Waiting polls
rather than blocks in the kernel, because a blocking ``flock`` cannot time out, and
a worker wedged inside a render must not wedge the others: past :data:`PATIENCE_S`
a waiter stops taking turns and goes back to sharing the GPU the way it always did,
which is slower and no less correct.

A worker takes turns when its pool put a directory in :data:`TURN_ENV`
(:func:`~robosimstudio_bench.run.pool.run_pooled` does, for two workers or more).
Off is ``ROBOSIMSTUDIO_GPU_TURN=off``.

No Warp here: a pool's parent imports this and must leave CUDA uninitialised.
"""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

__all__ = ["OFF", "PATIENCE_S", "TURN_ENV", "GpuTurn", "gpu_turn"]

TURN_ENV = "ROBOSIMSTUDIO_GPU_TURN"
"""The directory a pool's workers take turns in, or :data:`OFF`."""

OFF = "off"

PATIENCE_S = 60.0
"""How long a worker waits for its turn before it stops taking turns."""

POLL_S = 0.0005
"""Between tries for the turn: well under a physics step, which is what it adds to."""


class GpuTurn:
    """One process's end of one device's turn.

    Args:
        directory: Where this pool's lock files live. The same for every worker.
        device: The device the turn is for, in Warp's spelling.
        patience: Seconds to wait for a turn before giving up on taking them.
    """

    def __init__(self, directory: str, device: str = "", *, patience: float = PATIENCE_S) -> None:
        key = str(device).replace(":", "_") or "device"
        self._gate = os.open(os.path.join(directory, f"{key}.gate"), os.O_CREAT | os.O_RDWR, 0o600)
        self._use = os.open(os.path.join(directory, f"{key}.use"), os.O_CREAT | os.O_RDWR, 0o600)
        self.patience = float(patience)
        self.given_up = False
        """Set by a wait that ran out of patience. From then on this process just shares the GPU."""

    def _take(self, fd: int, mode: int) -> bool:
        deadline = time.monotonic() + self.patience
        while True:
            try:
                fcntl.flock(fd, mode | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                if time.monotonic() > deadline:
                    self.given_up = True
                    return False
                time.sleep(POLL_S)

    @contextmanager
    def stepping(self) -> Iterator[None]:
        """Run physics, alongside any other worker's physics and no render."""
        if self.given_up or not self._take(self._gate, fcntl.LOCK_SH):
            yield
            return
        fcntl.flock(self._gate, fcntl.LOCK_UN)
        held = self._take(self._use, fcntl.LOCK_SH)
        try:
            yield
        finally:
            if held:
                fcntl.flock(self._use, fcntl.LOCK_UN)

    @contextmanager
    def rendering(self) -> Iterator[None]:
        """Have the GPU alone: closes the gate on new steps, then waits out the ones running."""
        if self.given_up or not self._take(self._gate, fcntl.LOCK_EX):
            yield
            return
        held = self._take(self._use, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if held:
                fcntl.flock(self._use, fcntl.LOCK_UN)
            fcntl.flock(self._gate, fcntl.LOCK_UN)

    def close(self) -> None:
        os.close(self._gate)
        os.close(self._use)


def gpu_turn(device: str = "") -> GpuTurn | None:
    """This process's turn on ``device``, or ``None`` when its pool did not arrange one."""
    directory = os.environ.get(TURN_ENV, "").strip()
    if not directory or directory.lower() == OFF:
        return None
    try:
        return GpuTurn(directory, device)
    except OSError as exc:
        # A speed-up, so never a reason not to run.
        print(f"[gpu-turn] not taking turns ({type(exc).__name__}: {exc})")
        return None
