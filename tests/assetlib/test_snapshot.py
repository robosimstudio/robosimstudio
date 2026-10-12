# SPDX-License-Identifier: Apache-2.0
"""What ``Snapshot.freeze`` actually reaches.

``freeze`` exists so a measure that mutates its input fails where it did it. It
guarded ``body_q``, ``body_qd``, ``settled_body_q`` and ``joint_target_q`` -- the
four arrays a *rigid* measure reads -- and nothing a deformable one is handed. A
cloth, sponge or centerline handler is given the probe's arrays, and those
outlive the measure, because the environment re-uses them across steps.

The ``base is None`` guard is what makes covering them safe rather than merely
strict, and it is the guard that was already there: a probe fed from the batched
read-back holds a **view** of a live buffer, and freezing that would break the
next step's write. ``update_arrays`` reassigns rather than writing in place, so a
frozen array is replaced rather than mutated.
"""

from __future__ import annotations

import unittest
from typing import ClassVar

import numpy as np

from robosimstudio.assetlib.semantics.snapshot import Snapshot

TIER = "fast"
"""Arrays and flags. No scene, no model, no device."""


class _Probe:
    def __init__(self, positions, velocities):
        self.positions = positions
        self.velocities = velocities


class _Ctx:
    body_q = np.zeros((2, 7))
    body_qd = np.zeros((2, 6))
    settled_body_q = np.zeros((2, 7))
    joint_target_q = np.zeros(2)
    contacts = None
    step = 0
    sim_time = 0.0
    extras: ClassVar[dict] = {}
    handles = None


class TestFreezeReachesTheDeformableHalf(unittest.TestCase):
    def test_an_owning_probe_array_is_frozen(self) -> None:
        """A fixture's or a description's own array -- what freeze is for."""
        ctx = _Ctx()
        ctx.particles = {"towel": _Probe(np.zeros((4, 3)), np.zeros((4, 3)))}
        Snapshot(ctx).freeze()
        self.assertFalse(ctx.particles["towel"].positions.flags.writeable)
        self.assertFalse(ctx.particles["towel"].velocities.flags.writeable)

    def test_a_view_of_a_live_buffer_is_left_alone(self) -> None:
        """The hot path. Freezing here would break the next step's write."""
        buffer = np.zeros(12)
        ctx = _Ctx()
        ctx.particles = {"towel": _Probe(buffer.reshape(-1, 3), buffer.reshape(-1, 3))}
        Snapshot(ctx).freeze()
        self.assertTrue(buffer.flags.writeable, "a read-back buffer a probe merely views must stay writable")

    def test_the_rigid_four_are_still_frozen(self) -> None:
        """The half that already worked, so widening it cannot narrow it."""
        ctx = _Ctx()
        ctx.particles = {}
        snap = Snapshot(ctx).freeze()
        self.assertFalse(snap.body_q.flags.writeable)
        self.assertFalse(snap.body_qd.flags.writeable)


if __name__ == "__main__":
    unittest.main()
