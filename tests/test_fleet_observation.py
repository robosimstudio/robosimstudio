# SPDX-License-Identifier: Apache-2.0
"""What a world hands the graph runtime when it looks.

A picture is not an observation. A perception node finds something in pixels and
then has to say where that is in the world, which takes a camera matrix and a
camera pose; a block carrying only ``rgb`` and ``depth`` gets all the way
through the expensive models and fails at the unprojection -- which is a
diagnosis several tool calls and one VLM bill away from the cause. The batched
formatter shipped exactly that block, so every vision graph died at its first
perception node with ``KeyError: 'intrinsics'``.

The other half is per world, and is the reason this is not just a missing key.
The rig's specs carry **template** parent bodies, and a wave's snapshot carries
each world's rows rebased to template indices -- so the pose has to be resolved
against *this world's* snapshot. Resolving it once and broadcasting is a rig
that photographs one robot and labels the picture with sixteen episode ids.

No GPU: a fake snapshot, a real camera spec, and the dictionary that comes out.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio.core.cameras import CameraSpec
from robosimstudio_policy_agent.contract import ALLOWED_KEYS
from robosimstudio_policy_agent.fleet.obs_format import format_runtime_obs

TIER = "fast"
"""A dict in, a dict out. Gates every commit."""

CALIBRATION = ("intrinsics", "pose")
"""What a frame needs beyond its pixels to locate anything.

Named here rather than inline because the single-world facade sets exactly
these beside ``images``/``depth_data``, and the two formatters answering the
same runtime must agree on them."""


class _Snapshot:
    """One world's published view, as :class:`~.snapshots.SimSnapshot` carries it."""

    def __init__(self, frames, body_q=None):
        self.frames = frames
        self.body_q = body_q
        self.obs = {"proprio": {}, "instruction": "wipe it"}


def _camera(name: str, parent_body: int = -1) -> CameraSpec:
    return CameraSpec(name=name, width=64, height=48, parent_body=parent_body, eye=(0.0, -1.0, 0.5))


def _frame() -> dict:
    return {"rgb": np.zeros((48, 64, 3), dtype=np.uint8), "depth": np.ones((48, 64), dtype=np.float32)}


class TestACameraBlockCarriesItsCalibration(unittest.TestCase):
    """The keys a perception node reads after the models have run."""

    def test_a_block_carries_intrinsics_and_a_pose(self):
        out = format_runtime_obs(
            _Snapshot({"overhead": _frame()}, body_q=np.zeros((4, 7))),
            arm_count=1,
            cameras=[_camera("overhead")],
        )

        block = out["overhead"]
        for key in CALIBRATION:
            self.assertIn(key, block, f"a frame without {key} cannot locate anything")
        self.assertEqual((3, 3), np.asarray(block["intrinsics"]).shape)
        self.assertEqual((7,), np.asarray(block["pose"]).shape, "xyz + wxyz, the runtime's order")

    def test_the_pixels_are_still_there_both_ways(self):
        """``images`` is what the runtime's own camera builder reads;
        ``depth_data`` is the wire convention. Setting one is silent."""
        out = format_runtime_obs(
            _Snapshot({"overhead": _frame()}, body_q=np.zeros((4, 7))),
            arm_count=1,
            cameras=[_camera("overhead")],
        )

        block = out["overhead"]
        self.assertIsNotNone(block["images"]["rgb"])
        self.assertIsNotNone(block["images"]["depth"])
        self.assertIsNotNone(block["depth_data"])

    def test_every_key_the_block_carries_is_one_the_bridge_speaks(self):
        """The contract lists what may cross. A block inventing a key is a block
        no real rig could have produced, and the graph would be reading
        something the hardware will never send."""
        out = format_runtime_obs(
            _Snapshot({"overhead": _frame()}, body_q=np.zeros((4, 7))),
            arm_count=1,
            cameras=[_camera("overhead")],
        )

        block = out["overhead"]
        for key in (*block, *block["images"]):
            self.assertIn(key, ALLOWED_KEYS, f"{key!r} is not a key the bridge speaks")

    def test_a_camera_nobody_declared_still_hands_over_its_pixels(self):
        """A frame with no matching spec is a rig mismatch, not a crash: the
        pixels are still worth having, and the missing calibration is the
        caller's to notice."""
        out = format_runtime_obs(_Snapshot({"mystery": _frame()}), arm_count=1, cameras=[_camera("overhead")])

        self.assertIn("images", out["mystery"])
        for key in CALIBRATION:
            self.assertNotIn(key, out["mystery"])

    def test_no_rig_formats_the_picture_alone(self):
        """What a caller with no cameras means. It must not raise."""
        out = format_runtime_obs(_Snapshot({"overhead": _frame()}), arm_count=1)

        self.assertIn("images", out["overhead"])
        self.assertNotIn("intrinsics", out["overhead"])


class TestAnEyeInHandCameraIsThisWorldsWrist(unittest.TestCase):
    """The per-world half, and the bug that broadcasting would hide."""

    def test_the_pose_follows_the_snapshot_it_was_given(self):
        """Two worlds' snapshots, one spec, two different answers.

        The spec's ``parent_body`` is a template index and each snapshot's rows
        are that world's rebased to template indices, so this is the whole of
        what makes a wrist camera per world -- and if it ever resolved against
        the fleet's own arrays instead, both worlds would answer with world
        zero's wrist and nothing about the picture would look wrong.
        """
        spec = _camera("wrist", parent_body=2)
        near = np.zeros((4, 7))
        near[2] = [0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        far = np.zeros((4, 7))
        far[2] = [0.90, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        first = format_runtime_obs(_Snapshot({"wrist": _frame()}, body_q=near), arm_count=1, cameras=[spec])
        second = format_runtime_obs(_Snapshot({"wrist": _frame()}, body_q=far), arm_count=1, cameras=[spec])

        gap = float(second["wrist"]["pose"][0] - first["wrist"]["pose"][0])
        self.assertAlmostEqual(0.80, gap, places=9, msg="each world's camera rides its own arm")

    def test_a_world_fixed_camera_ignores_the_body_rows(self):
        """Every world is superimposed at the origin, so a camera bolted to the
        room does see the same place in all of them."""
        spec = _camera("overhead")
        moved = np.zeros((4, 7))
        moved[2] = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        still = format_runtime_obs(
            _Snapshot({"overhead": _frame()}, body_q=np.zeros((4, 7))), arm_count=1, cameras=[spec]
        )
        other = format_runtime_obs(_Snapshot({"overhead": _frame()}, body_q=moved), arm_count=1, cameras=[spec])

        np.testing.assert_allclose(still["overhead"]["pose"], other["overhead"]["pose"], atol=0)


if __name__ == "__main__":
    unittest.main()
