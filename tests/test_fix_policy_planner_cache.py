# SPDX-License-Identifier: Apache-2.0
"""ROBOSIMSTUDIO_PLANNER_CACHE: one warm cuRobo planner per robot model per process.

cuRobo is not a test dependency, so the planner is a stub that counts its
builds and records what was done to it.
"""

from __future__ import annotations

import os
import sys
import threading
import types
import unittest
from unittest import mock

from robosimstudio.core.streams import stream_lock
from robosimstudio_policy_agent.planners import curobo as planners
from robosimstudio_policy_agent.planners.curobo import (
    CACHED_ATTACHED_PLANNERS,
    PLANNER_CACHE_ENV,
    MotionBackendCuRobo,
    clear_planner_cache,
)

TIER = "fast"


class _Planner:
    built = 0
    fail_warmup = False

    def __init__(self, cfg) -> None:
        type(self).built += 1
        self.cfg = cfg
        self.tool_frames = ["link_6"]
        self.events: list[str] = []

    def warmup(self) -> None:
        if type(self).fail_warmup:
            raise RuntimeError("cudaErrorStreamCaptureInvalidated")
        self.events.append("warmup")

    def clear_scene_cache(self) -> None:
        self.events.append("clear")

    def reset_seed(self) -> None:
        self.events.append("reset_seed")


class _PlannerCfg:
    @staticmethod
    def create(**kw):
        return kw


def _curobo_stubs(fit=None) -> dict:
    names = (
        "curobo",
        "curobo._src",
        "curobo._src.cost",
        "curobo._src.cost.tool_pose_criteria",
        "curobo.motion_planner",
        "curobo.types",
        "curobo._src.geom",
        "curobo._src.geom.sphere_fit",
        "curobo._src.geom.sphere_fit.fit_spheres",
        "curobo.sphere_fit",
    )
    modules = {name: types.ModuleType(name) for name in names}
    for name in names:
        modules[name].__path__ = []  # a package, so importlib walks it
    modules["curobo._src.cost.tool_pose_criteria"].ToolPoseCriteria = object
    modules["curobo.motion_planner"].MotionPlanner = _Planner
    modules["curobo.motion_planner"].MotionPlannerCfg = _PlannerCfg
    for name in ("GoalToolPose", "JointState", "Pose"):
        setattr(modules["curobo.types"], name, object)
    if fit is not None:
        for name in ("curobo._src.geom.sphere_fit.fit_spheres", "curobo._src.geom.sphere_fit", "curobo.sphere_fit"):
            modules[name].fit_spheres_to_mesh = fit
    return modules


class _Case(unittest.TestCase):
    cache = ""

    def setUp(self) -> None:
        _Planner.built = 0
        _Planner.fail_warmup = False
        clear_planner_cache()
        patches = [
            mock.patch.dict(sys.modules, _curobo_stubs()),
            mock.patch.dict(os.environ, {PLANNER_CACHE_ENV: self.cache}),
            mock.patch.object(planners, "_capture_thread_local", lambda: None),
            mock.patch.object(planners, "_current_device", lambda: "cuda:0"),
            # The description is what the key is about; reading YAML is not.
            mock.patch.object(
                MotionBackendCuRobo, "_robot_config", lambda self: {"config": self.config_path, **self.lock_joints}
            ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(clear_planner_cache)

    @staticmethod
    def backend(config: str = "yam_curobo.yml", lock: dict | None = None) -> MotionBackendCuRobo:
        return MotionBackendCuRobo(None, None, config_path=config, lock_joints=lock)


class WithoutTheCacheTest(_Case):
    cache = ""

    def test_every_backend_builds_its_own_and_only_when_asked(self) -> None:
        first, second = self.backend(), self.backend()
        self.assertEqual(_Planner.built, 0, "an uncached backend built its planner before its first plan")
        self.assertIsNot(first.planner, second.planner)
        self.assertEqual(_Planner.built, 2)
        self.assertEqual(first.planner.events, ["warmup"])


class WithTheCacheTest(_Case):
    cache = "1"

    def test_the_next_backend_adopts_the_warm_planner(self) -> None:
        first = self.backend()
        self.assertEqual(_Planner.built, 1, "the shared planner was not warmed when its first backend was built")
        second = self.backend()
        self.assertEqual(_Planner.built, 1, "a second backend built a second planner")
        self.assertIs(second.planner, first.planner)
        self.assertEqual(second._frame, "link_6")
        # The state a fresh build leaves, again: no world, seeds reset.
        self.assertEqual(first.planner.events, ["warmup", "clear", "reset_seed"])

    def test_a_different_robot_description_is_a_different_planner(self) -> None:
        base = self.backend()
        other_config = self.backend(config="franka.yml")
        other_fingers = self.backend(lock={"left_finger": 0.04})
        same_fingers = self.backend(lock={"left_finger": 0.04 + 1e-9})
        self.assertEqual(_Planner.built, 3)
        self.assertIsNot(base.planner, other_config.planner)
        self.assertIsNot(base.planner, other_fingers.planner)
        self.assertIs(same_fingers.planner, other_fingers.planner)

    def test_attachment_planners_are_shared_and_bounded(self) -> None:
        first = self.backend()
        for i in range(CACHED_ATTACHED_PLANNERS):
            first._keep("attached", ("a", i), _Planner({}))
        oldest = first._recall("attached", ("a", 0))  # now the most recently used
        first._keep("attached", ("a", "new"), _Planner({}))
        shared = first._warm.attached
        self.assertEqual(len(shared), CACHED_ATTACHED_PLANNERS)
        self.assertNotIn(("a", 1), shared, "the least recently used planner was kept")

        second = self.backend()
        self.assertIs(second._recall("attached", ("a", 0)), oldest)
        self.assertIsNone(second._recall("attached", ("a", 1)))
        # Adoption reset the attachment planners too, not only the main one.
        self.assertEqual(oldest.events[-2:], ["clear", "reset_seed"])

    def test_a_backend_keeps_what_it_used_whatever_the_others_evict(self) -> None:
        """Eight worlds, eight held objects: the shared bound must not make one
        world rebuild its own planner between two of its own plans."""
        worlds = [self.backend() for _ in range(CACHED_ATTACHED_PLANNERS + 4)]
        mine = [_Planner({}) for _ in worlds]
        for i, (world, planner) in enumerate(zip(worlds, mine, strict=True)):
            world._keep("attached", ("held", i), planner)
        self.assertEqual(len(worlds[0]._warm.attached), CACHED_ATTACHED_PLANNERS)
        for i, (world, planner) in enumerate(zip(worlds, mine, strict=True)):
            self.assertIs(world._recall("attached", ("held", i)), planner)

    def test_world_loads_and_solves_hold_the_models_lock(self) -> None:
        backend = self.backend()
        held = []

        class Probe(MotionBackendCuRobo):
            @planners._holding_model_lock
            def solve(self) -> None:
                held.append(self._warm.lock._is_owned())

        probe = Probe.__new__(Probe)
        probe.__dict__.update(backend.__dict__)
        probe.solve()
        self.assertEqual(held, [True])

    def test_a_backend_waits_for_a_straggler_before_adopting(self) -> None:
        """The previous episode's graph thread may still be mid-plan on the
        shared planner; the next episode's adoption must not reset it under
        that plan."""
        first = self.backend()
        warm = first._warm
        order: list[str] = []
        started = threading.Event()

        def straggler() -> None:
            with warm.lock:
                started.set()
                order.append("plan starts")
                threading.Event().wait(0.2)
                order.append("plan ends")

        thread = threading.Thread(target=straggler)
        thread.start()
        started.wait()
        self.backend()
        order.append("adopted")
        thread.join()
        self.assertEqual(order, ["plan starts", "plan ends", "adopted"])

    def test_a_failed_warmup_is_left_to_the_first_plan(self) -> None:
        _Planner.fail_warmup = True
        with self.assertLogs(planners.logger, "WARNING"):
            backend = self.backend()
        with self.assertRaisesRegex(RuntimeError, "StreamCapture"):
            _ = backend.planner
        _Planner.fail_warmup = False
        self.assertEqual(backend.planner.events, ["warmup"])


class SphereFitTest(unittest.TestCase):
    def test_the_fitter_runs_under_the_stream_lock_every_capture_holds(self) -> None:
        """MORPHIT draws from torch's device generator; a draw landing while
        another thread captures a cuRobo solver is refused (``Offset increment
        outside graph capture``), so the two must not overlap."""
        held = []

        def fit(mesh, **kwargs):
            held.append(stream_lock._is_owned())
            return mesh, kwargs

        stubs = _curobo_stubs(fit=fit)
        with mock.patch.dict(sys.modules, stubs):
            planners._serialize_sphere_fit()
            planners._serialize_sphere_fit()  # idempotent: one layer, not two
            for name in ("curobo._src.geom.sphere_fit.fit_spheres", "curobo._src.geom.sphere_fit", "curobo.sphere_fit"):
                wrapped = stubs[name].fit_spheres_to_mesh
                self.assertIs(wrapped.__wrapped__, fit)
                self.assertEqual(wrapped("mesh", num_spheres=64), ("mesh", {"num_spheres": 64}))
        self.assertEqual(held, [True, True, True])


if __name__ == "__main__":
    unittest.main()
