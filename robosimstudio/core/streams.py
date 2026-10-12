# SPDX-License-Identifier: Apache-2.0
"""One device, one current stream, many threads -- and the stream is a global.

Warp keeps the current CUDA stream on the *device object*, process-wide rather
than per thread, and ``wp.ScopedStream`` restores whatever it found on the way
in. That is safe in one thread and unsound in several. ``array.numpy()`` uses
it on every call::

    with warp.ScopedStream(self.device.null_stream):   # warp/_src/types.py
        a = self.to("cpu")

so two threads reading at once interleave save and restore::

    T1 enter: saved = real, current = null
    T2 enter: saved = null, current = null
    T1 exit : current = real
    T2 exit : current = null      <- and nothing ever puts it back

``null_stream`` is the CUDA default stream, which cannot be graph-captured. From
that moment **every** CUDA graph capture in the process fails with "Warp error:
unknown stream", including captures for scenes compiled long afterwards. It does
not heal: closing whatever was reading does not restore the stream.

This is reachable here because a fleet runs one script thread per world, and a
script reads state through the same tool surface a single-world graph uses --
sixteen worlds means sixteen threads calling ``.numpy()`` against one device.
Measured: eight reader threads and one array poison a process in under a second.

So two things live here, and both are needed:

* :func:`reading` serializes *our* readers, which is where the interleaving
  comes from, so they stop creating the condition.
* :func:`capturing` repairs the current stream before a capture and takes the
  same lock for its duration, so a capture cannot be poisoned by a reader and a
  process that was already poisoned recovers instead of staying broken.

The lock is one process-wide reentrant lock rather than one per device: it
guards Warp's global, not ours, and reentrancy matters because a capture body
may legitimately read.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any

import warp as wp

__all__ = ["capturable", "capturing", "reading", "stream_lock"]

stream_lock = threading.RLock()
"""Held across anything that moves a device's current stream. See the module docstring."""


def capturable(device: Any) -> bool:
    """Make ``device`` capture-ready, and say whether it had to be repaired.

    A device whose current stream is the CUDA default stream cannot begin a
    graph capture. Warp encodes that stream as ``cuda_stream = None``, which is
    also exactly what a lost race leaves behind, so the test and the repair are
    the same one line: give the device a real stream again.
    """
    if device is None or not getattr(device, "is_cuda", False):
        return False
    if device.stream.cuda_stream is not None:
        return False
    device.set_stream(wp.Stream(device))
    return True


@contextmanager
def reading():
    """Serialize a device-to-host read against every other one, and against capture.

    Wrap ``.numpy()`` calls made off the main thread in this. It is the cheap
    half of the fix: the copy itself is unchanged and still asynchronous with
    respect to other work, and all this prevents is two ``ScopedStream`` blocks
    overlapping -- which is the only thing that loses the current stream.
    """
    with stream_lock:
        yield


@contextmanager
def capturing(device: Any = None, **kwargs: Any):
    """``wp.ScopedCapture``, with the current stream guaranteed and held.

    Repairs the device first, so a process that already lost its stream to a
    race captures rather than failing forever, and holds the lock for the whole
    capture, so a concurrent reader cannot move the stream out from under it.
    """
    resolved = wp.get_device(device) if device is not None else wp.get_device()
    with stream_lock:
        capturable(resolved)
        with wp.ScopedCapture(device=device, **kwargs) as capture:
            yield capture
