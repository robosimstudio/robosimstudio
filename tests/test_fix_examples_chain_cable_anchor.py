# SPDX-License-Identifier: Apache-2.0
"""Three examples held to what they say: the chain damps its own links only, the
cable's capsules are spaced above twice their radius, and a rod's anchor latches
with the hand it posed, and only when the rod permits a latch.

No GPU: the chain's kernel runs on Warp's CPU device, the cable is read off its
source, and the anchor runs against stand-ins for the arm, its solver and the
pipeline.
"""

from __future__ import annotations

import ast
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests.fixtures.examples import EXAMPLES, _load

TIER = "fast"
"""Warp on its CPU device, a source file, and stand-ins: no GPU."""


class TestTheChainDampsOnlyItsLinks(unittest.TestCase):
    """``04_extending/01_new_kind``: the drag hook belongs to the chain, not the robot beside it."""

    def test_other_bodies_keep_their_velocity(self) -> None:
        import warp as wp

        module = _load("04_extending/01_new_kind.py")
        qd = np.ones((6, 6), dtype=np.float32)
        state = SimpleNamespace(body_qd=wp.array(qd, dtype=wp.spatial_vector, device="cpu"))
        chain = SimpleNamespace(bodies=[2, 4], spec=SimpleNamespace(damping=0.5))
        module.MyChain._drag(chain, state, 0.001)
        got = state.body_qd.numpy()
        np.testing.assert_allclose(got[[2, 4]], 0.5, err_msg="the chain's links are damped")
        np.testing.assert_allclose(got[[0, 1, 3, 5]], 1.0, err_msg="a body that is not the chain's was damped")


class TestTheCableKeepsItsOwnSpacingRule(unittest.TestCase):
    """``10_rope/00_cable``: "their spacing must stay above twice the radius"."""

    def test_spacing_is_above_twice_the_radius(self) -> None:
        tree = ast.parse((EXAMPLES / "10_rope/00_cable.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        values = {
            n.targets[0].id: ast.literal_eval(n.value)
            for n in main.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)
        }
        spacing = values["length"] / values["segments"]
        self.assertGreater(spacing, 2 * values["radius"], f"spacing {spacing * 1e3:.2f} mm")


class _Array:
    """A device array as far as ``_pose_anchor`` reads and writes one."""

    def __init__(self, n: int) -> None:
        self.values = np.zeros(n)

    def numpy(self) -> np.ndarray:
        return self.values.copy()

    def assign(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float64)


def _pose(*, allow_grasp_latch: bool, arm: str = "right"):
    """``EntityRod._pose_anchor`` against a stand-in rod; returns the ``hold_segment`` mock."""
    from robosimstudio.domains.rod import entity as rod

    ik = SimpleNamespace(
        solve_q=lambda seed, target, approach: np.zeros(4),
        fk_ee=lambda q: (np.zeros(3), None),
    )
    index = SimpleNamespace(arm_coords=(0, 1, 2), gripper_coords=(3,))
    embodiment = SimpleNamespace(iks={arm: ik}, arms={arm: index})
    hold = mock.Mock()
    fake = SimpleNamespace(
        spec=SimpleNamespace(anchor_grasp=(0.0, 0.0, 0.0), anchor_grip=0.0, allow_grasp_latch=allow_grasp_latch),
        embodiment=embodiment,
        worlds=None,
        _fleet_model=None,
        _hold_flag=object(),
        _anchor_arm=lambda: arm,
        placed=lambda pose: np.asarray(pose, dtype=np.float64),
        _coord=lambda c, world: int(c),
        hold_segment=hold,
    )
    state = SimpleNamespace(joint_q=_Array(4), joint_qd=None, body_q=None)
    pipeline = SimpleNamespace(
        state_0=state,
        state_1=SimpleNamespace(joint_q=_Array(4), body_q=None),
        control=SimpleNamespace(joint_target_q=_Array(4)),
    )
    with mock.patch.object(rod.newton, "eval_fk"), mock.patch.object(rod.wp, "copy"):
        rod.EntityRod._pose_anchor(fake, pipeline, world=0)
    return hold, pipeline


class TestTheAnchorLatchesTheHandItPosed(unittest.TestCase):
    """``rod/entity.py``: the anchor's latch follows ``_anchor_arm`` and ``allow_grasp_latch``."""

    def test_the_latch_is_on_the_resolved_arm(self) -> None:
        hold, pipeline = _pose(allow_grasp_latch=True, arm="right")
        hold.assert_called_once_with(True, pipeline.state_0, arm="right", slot=1, world=0)

    def test_a_contact_only_cable_is_not_latched(self) -> None:
        hold, _ = _pose(allow_grasp_latch=False)
        hold.assert_not_called()


if __name__ == "__main__":
    unittest.main()
