# SPDX-License-Identifier: Apache-2.0
"""Tests for the Lemi 2B embodiment and hands in RoboSimStudio."""

from __future__ import annotations

import types

import numpy as np

import robosimstudio as rss
from robosimstudio.robots.registry import get_embodiment, get_gripper
from robosimstudio_real2sim_agent import checks

TIER = "medium"
"""Builds a scene and steps it on a GPU."""


def test_lemi2b_registry_and_hands() -> None:
    emb = get_embodiment("lemi2b")
    assert emb.name == "lemi2b"
    assert len(emb.arms) == 2
    assert {a.name for a in emb.arms} == {"left", "right"}
    assert dict(emb.aux_home_q)["lift"] == 0.85

    lh = get_gripper("lemi2b_left_hand")
    rh = get_gripper("lemi2b_right_hand")
    assert lh.finger_names == ("LH_PI", "LH_DI", "LH_P3F", "LH_D3F")
    assert rh.finger_names == ("RH_PI", "RH_DI", "RH_P3F", "RH_D3F")


def test_lemi2b_scene_builds_and_settles_with_zero_penetration() -> None:
    scene = rss.Scene()
    scene.add(rss.robot("lemi2b"))
    scene.add(
        rss.CameraSpec(name="hat", width=640, height=480, fov_y=90.0, eye=(0.1, 0.0, 1.3), look_at=(0.6, 0.0, 0.7))
    )
    scene.add(rss.slab(name="target_box", half=(0.06, 0.06, 0.04), at=(0.75, 0.0, 0.045)))
    sim = scene.build(control_mode="joint_position")

    srv = types.SimpleNamespace(sim=sim, build_error="")
    res = checks.gate(srv, camera="hat", objects=["target_box"], steps=10, settle_mm=5.0, penetration_mm=2.0)
    assert res["passed"], res["report"]
    assert np.all(np.isfinite(sim.state.joint_q.numpy()))


if __name__ == "__main__":
    test_lemi2b_registry_and_hands()
    test_lemi2b_scene_builds_and_settles_with_zero_penetration()
    print("PASS test_lemi2b_embodiment.py")
