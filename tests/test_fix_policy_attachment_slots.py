# SPDX-License-Identifier: Apache-2.0
"""Attachment slots: one planner per arm takes any small held object by an in-place write.

cuRobo is not a test dependency, so the planner is a stub whose kinematics
params hold a real (CPU) ``link_spheres`` tensor built from the tool link's
configured spheres, the way cuRobo lays them out.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is in every GPU environment
    torch = None

from robosimstudio_policy_agent.planners import curobo as planners
from robosimstudio_policy_agent.planners.curobo import (
    ATTACHMENT_SLOTS,
    PLANNER_CACHE_ENV,
    MotionBackendCuRobo,
    clear_planner_cache,
)

TIER = "fast"
TOOL = "link_6"
BASE_SPHERES = 3  # the tool link's own, before any attachment


class _Params:
    def __init__(self, tool_spheres: list[dict]) -> None:
        other = [[0.0, 0.0, -1.0, 0.05]] * 5  # spheres of other links, first in the buffer
        rows = other + [[*s["center"], s["radius"]] for s in tool_spheres]
        self.link_spheres = torch.tensor([rows], dtype=torch.float32)
        self.tool_index = torch.arange(len(other), len(rows))

    def get_sphere_index_from_link_name(self, link: str):
        assert link == TOOL
        return self.tool_index


class _Planner:
    built = 0

    def __init__(self, cfg) -> None:
        type(self).built += 1
        spheres = cfg["robot"]["robot_cfg"]["kinematics"]["collision_spheres"].get(TOOL) or []
        self.tool_frames = [TOOL]
        self.events: list[str] = []
        params = _Params([{"center": [0.0, 0.0, 0.01 * i], "radius": 0.02} for i in range(BASE_SPHERES)] + spheres)
        self.kinematics = mock.Mock()
        self.kinematics.config.kinematics_config = params
        self.ik_solver = mock.Mock()

    def warmup(self) -> None:
        self.events.append("warmup")

    def clear_scene_cache(self) -> None:
        self.events.append("clear")

    def reset_seed(self) -> None:
        self.events.append("reset_seed")


class _PlannerCfg:
    @staticmethod
    def create(**kw):
        return kw


def _stubs() -> dict:
    import types

    names = (
        "curobo",
        "curobo._src",
        "curobo._src.cost",
        "curobo._src.cost.tool_pose_criteria",
        "curobo.motion_planner",
        "curobo.types",
    )
    modules = {name: types.ModuleType(name) for name in names}
    for name in names:
        modules[name].__path__ = []
    modules["curobo._src.cost.tool_pose_criteria"].ToolPoseCriteria = object
    modules["curobo.motion_planner"].MotionPlanner = _Planner
    modules["curobo.motion_planner"].MotionPlannerCfg = _PlannerCfg
    for name in ("GoalToolPose", "JointState", "Pose"):
        setattr(modules["curobo.types"], name, object)
    return modules


def _attachment(n: int, scale: float = 1.0) -> dict:
    return {
        "spheres": [
            {"center": {"x": 0.01 * i * scale, "y": 0.0, "z": 0.1}, "radius": 0.01 + 0.001 * i} for i in range(n)
        ]
    }


@unittest.skipIf(torch is None, "needs torch")
class AttachmentSlotsTest(unittest.TestCase):
    def setUp(self) -> None:
        _Planner.built = 0
        clear_planner_cache()
        robot = {"robot_cfg": {"kinematics": {"collision_spheres": {}}}}
        for patch in (
            mock.patch.dict(sys.modules, _stubs()),
            mock.patch.dict(os.environ, {PLANNER_CACHE_ENV: ""}),
            mock.patch.object(planners, "_capture_thread_local", lambda: None),
            mock.patch.object(planners, "_current_device", lambda: "cuda:0"),
            mock.patch.object(MotionBackendCuRobo, "_robot_config", lambda self: robot),
            mock.patch.object(MotionBackendCuRobo, "_tool_from_tcp", lambda self, arm_id: np.eye(4)),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(clear_planner_cache)
        self.backend = MotionBackendCuRobo(None, None, config_path="franka.yml")
        self.backend._frame = TOOL

    def slots(self, planner) -> torch.Tensor:
        params = planner.kinematics.config.kinematics_config
        return params.link_spheres[0, params.tool_index[-ATTACHMENT_SLOTS:]]

    def test_small_attachments_share_one_planner_written_in_place(self) -> None:
        self.backend.planner  # noqa: B018 - the arm's base planner, built on first use
        base_builds = _Planner.built
        first = self.backend._planner_for(_attachment(5), 0)
        self.assertEqual(_Planner.built, base_builds + 1)
        self.assertEqual(first.events, ["warmup", "reset_seed"], "slots must be disabled before warmup, then written")
        slots = self.slots(first)
        self.assertTrue(torch.allclose(slots[:5, 3], torch.tensor([0.01 + 0.001 * i for i in range(5)])))
        self.assertTrue(bool((slots[5:, 3] == -100.0).all()), "unused slots are not disabled")
        tool = first.kinematics.config.kinematics_config.link_spheres[
            0, first.kinematics.config.kinematics_config.tool_index
        ]
        self.assertTrue(bool((tool[:BASE_SPHERES, 3] == 0.02).all()), "the tool link's own spheres were overwritten")

        second = self.backend._planner_for(_attachment(9, scale=1.3), 0)
        self.assertIs(second, first, "a second small attachment built a planner")
        self.assertEqual(_Planner.built, base_builds + 1)
        self.assertEqual(int((self.slots(second)[:, 3] > 0).sum()), 9)
        self.assertEqual(
            second.events.count("reset_seed"), 2, "a new attachment resets the seed, as a fresh planner would"
        )

        self.backend._planner_for(_attachment(9, scale=1.3), 0)
        self.assertEqual(second.events.count("reset_seed"), 2, "re-planning with the same attachment reset the seed")

    def test_a_large_attachment_still_gets_its_own_planner(self) -> None:
        small = self.backend._planner_for(_attachment(4), 0)
        built = _Planner.built
        large = self.backend._planner_for(_attachment(ATTACHMENT_SLOTS + 1), 0)
        self.assertIsNot(large, small)
        self.assertEqual(_Planner.built, built + 1)

    def test_no_attachment_is_the_base_planner(self) -> None:
        self.assertIs(self.backend._planner_for(None, 0), self.backend.planner)
        self.assertIs(self.backend._planner_for({"spheres": []}, 0), self.backend.planner)


if __name__ == "__main__":
    unittest.main()
