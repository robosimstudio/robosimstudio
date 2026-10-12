# SPDX-License-Identifier: Apache-2.0
"""The current-stream guard, without a device.

Warp keeps a CUDA device's current stream on the device object, process-wide,
and ``array.numpy()`` moves it for the duration of the copy. Several threads
reading at once therefore lose it: the last one out restores whatever it found,
which may be the CUDA default stream another reader had installed. Nothing puts
it back, and the default stream cannot be graph-captured -- so from then on every
capture in the process fails with "unknown stream", including captures for
scenes compiled much later.

That is reachable here because a grid runs one script thread per world. What is
tested below is the guard, not Warp: that the repair fires on exactly the state a
lost race leaves behind and on nothing else, and that the lock a reader takes is
the lock a capture takes.
"""

from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from robosimstudio.core import streams

TIER = "fast"
"""No GPU. Gates every commit."""


class _FakeStream:
    def __init__(self, handle):
        self.cuda_stream = handle


class _FakeDevice:
    def __init__(self, *, is_cuda=True, handle=0xABC):
        self.is_cuda = is_cuda
        self.stream = _FakeStream(handle)
        self.set_streams = []

    def set_stream(self, stream):
        self.set_streams.append(stream)
        self.stream = stream


class TestCapturable(unittest.TestCase):
    def test_a_device_on_the_default_stream_is_repaired(self):
        """``cuda_stream = None`` is Warp's CUDA default stream -- and uncapturable."""
        device = _FakeDevice(handle=None)
        with mock.patch.object(streams.wp, "Stream", lambda d: _FakeStream(0x123)):
            self.assertTrue(streams.capturable(device))
        self.assertEqual(device.stream.cuda_stream, 0x123)

    def test_a_healthy_device_is_left_alone(self):
        device = _FakeDevice(handle=0xABC)
        with mock.patch.object(streams.wp, "Stream", lambda d: _FakeStream(0x123)):
            self.assertFalse(streams.capturable(device))
        self.assertEqual(device.set_streams, [])
        self.assertEqual(device.stream.cuda_stream, 0xABC)

    def test_a_cpu_device_has_no_stream_to_repair(self):
        device = _FakeDevice(is_cuda=False, handle=None)
        self.assertFalse(streams.capturable(device))
        self.assertEqual(device.set_streams, [])

    def test_no_device_is_not_an_error(self):
        self.assertFalse(streams.capturable(None))

    def test_something_without_is_cuda_is_left_alone(self):
        """Duck-typed on purpose: a stub env in a test must not be repaired."""
        self.assertFalse(streams.capturable(SimpleNamespace()))


class TestSerialization(unittest.TestCase):
    def test_a_reader_and_a_capture_take_the_same_lock(self):
        """Different locks would serialize readers against each other and nothing else."""
        self.assertIsInstance(streams.stream_lock, type(threading.RLock()))
        with streams.reading():
            # Reentrant, because a capture body may legitimately read.
            with streams.reading():
                pass

    def test_a_reader_excludes_another_thread(self):
        order = []
        released = threading.Event()

        def late():
            with streams.reading():
                order.append("second")

        with streams.reading():
            thread = threading.Thread(target=late)
            thread.start()
            thread.join(timeout=0.2)
            order.append("first")
            released.set()
        thread.join(timeout=5)
        self.assertEqual(order, ["first", "second"])


if __name__ == "__main__":
    unittest.main()
