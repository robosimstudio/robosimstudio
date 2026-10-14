# SPDX-License-Identifier: Apache-2.0
"""cuRobo world loading: the TSDF's frame, a TSDF with every mesh filtered, sphere paths.

cuRobo is not a test dependency, so the few names these paths import are stubbed.
"""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from robosimstudio_policy_agent.planners.curobo import MotionBackendCuRobo

TIER = "fast"


class _Feature(np.ndarray):
    def copy_(self, other) -> None:
        self[...] = other

    def clone(self):
        return self.copy()


class _SceneCfg:
    def __init__(self, **kw) -> None:
        self.kw = kw


class _Planner:
    def __init__(self) -> None:
        self.loaded = None

    def clear_scene_cache(self) -> None:
        self.loaded = None

    def update_world(self, cfg) -> None:
        self.loaded = cfg.kw


def _yaml(path):
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _curobo_stubs(configs_dir: str = "/nonexistent") -> dict:
    modules = {
        name: types.ModuleType(name)
        for name in (
            "curobo",
            "curobo._src",
            "curobo._src.geom",
            "curobo._src.geom.types",
            "curobo.config_io",
            "curobo.content",
        )
    }
    modules["curobo._src.geom.types"].SceneCfg = _SceneCfg
    modules["curobo._src.geom.types"].Mesh = lambda **kw: kw
    modules["curobo.config_io"].load_yaml = _yaml
    modules["curobo.config_io"].join_path = lambda a, b: str(Path(a) / b)
    modules["curobo.content"].get_robot_configs_path = lambda: configs_dir
    return modules


def _backend_with_grid(build_arm: int) -> MotionBackendCuRobo:
    backend = MotionBackendCuRobo.__new__(MotionBackendCuRobo)
    eye = np.eye(3)
    bases = {0: (eye, np.array([0.0, -0.3, 0.0])), 1: (eye, np.array([0.0, 0.3, 0.0]))}
    backend._base_cache = {-1: bases[0], **bases}
    n = 5
    feature = np.ones((n, n, n)).view(_Feature)
    backend._tsdf_grid = types.SimpleNamespace(feature_tensor=feature)
    backend._tsdf_pristine = feature.clone()
    backend._tsdf_axes = [(np.arange(n) - (n - 1) / 2) * 0.01 for _ in range(3)]
    backend._tsdf_base = bases[build_arm]
    backend._robot_spheres_world = lambda: np.zeros((0, 4))
    return backend


class TsdfWorldTest(unittest.TestCase):
    def test_tsdf_loads_when_every_mesh_is_filtered(self) -> None:
        backend = _backend_with_grid(0)
        planner = _Planner()
        world = {"use_tsdf": True, "meshes": [{"name": "empty", "vertices": [], "faces": []}]}
        with mock.patch.dict(sys.modules, _curobo_stubs()):
            backend._update_world(planner, world, 0)
        self.assertIsNotNone(planner.loaded, "the cleared world was never reloaded")
        self.assertIs(planner.loaded["voxel"][0], backend._tsdf_grid)

    def test_another_arms_grid_is_refused(self) -> None:
        backend = _backend_with_grid(0)
        with mock.patch.dict(sys.modules, _curobo_stubs()):
            backend._update_world(_Planner(), {"use_tsdf": True}, 0)
            with self.assertRaisesRegex(ValueError, "arm_id=1"):
                backend._update_world(_Planner(), {"use_tsdf": True}, 1)

    def test_carving_uses_the_grids_own_base(self) -> None:
        # Built for arm 1 (base y=+0.3); a contact sphere at world y=+0.3 is the
        # grid's centre voxel in that frame, and arm 0's frame would miss it.
        backend = _backend_with_grid(1)
        backend._tsdf_pristine[2, 2, 2] = 0.0
        backend._tsdf_grid.feature_tensor[...] = backend._tsdf_pristine
        backend._carve_tsdf_contact_spheres(backend._tsdf_grid, np.array([[0.0, 0.3, 0.0, 0.001]]))
        self.assertEqual(float(backend._tsdf_grid.feature_tensor[2, 2, 2]), 1.0)


class CollisionSpheresTest(unittest.TestCase):
    def _backend(self, config_path) -> MotionBackendCuRobo:
        backend = MotionBackendCuRobo.__new__(MotionBackendCuRobo)
        backend.config_path = config_path
        return backend

    def test_none_becomes_a_mapping(self) -> None:
        kinematics = {"collision_spheres": None}
        spheres = self._backend("franka.yml")._inline_collision_spheres(kinematics)
        self.assertEqual(spheres, {})
        self.assertIs(kinematics["collision_spheres"], spheres)

    def test_a_path_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "spheres").mkdir()
            (Path(tmp) / "spheres" / "arm.yml").write_text(
                "collision_spheres:\n  link0:\n    - {center: [0, 0, 0], radius: 0.05}\n"
            )
            config = Path(tmp) / "robot.yml"
            config.write_text("robot_cfg: {}\n")
            kinematics = {"collision_spheres": "spheres/arm.yml"}
            with mock.patch.dict(sys.modules, _curobo_stubs()):
                spheres = self._backend(str(config))._inline_collision_spheres(kinematics)
        self.assertEqual(list(spheres), ["link0"])
        self.assertIs(kinematics["collision_spheres"], spheres)

    def test_a_path_resolves_in_curobos_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "spheres").mkdir()
            (Path(tmp) / "spheres" / "franka.yml").write_text("collision_spheres:\n  hand: []\n")
            kinematics = {"collision_spheres": "spheres/franka.yml"}
            with mock.patch.dict(sys.modules, _curobo_stubs(tmp)):
                spheres = self._backend("franka.yml")._inline_collision_spheres(kinematics)
        self.assertEqual(spheres, {"hand": []})


if __name__ == "__main__":
    unittest.main()
