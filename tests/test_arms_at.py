# SPDX-License-Identifier: Apache-2.0
"""``rss.robot(..., arms_at=...)`` stands each arm's base where it says, with the robot's own file.

A real cell whose two arms stand on separate mounts used to need a copy of the
robot's MJCF with the arm bodies moved inside it. ``arms_at`` says the same
thing about the registered file, and these are the claims that make it the same
robot in the new place, each measured on a built scene:

* each base body lands on its pose in the work-surface frame -- on the
  collapsed ``yam_bimanual``, read off the collider the file puts on the base,
  since the collapse folds the base body itself into the world;
* the layout builds exactly what a moved copy of the file builds -- the
  workaround this replaces -- body for body;
* it is the registered robot: its file, joints, shapes and collision filtering
  are the default build's;
* IK reaches a target beside each moved base, and the arm gets there;
* on a robot whose base survives as a body (the Panda), the handle's
  ``base_body`` -- the frame a base-frame solver reads -- is at the pose.
"""

from __future__ import annotations

import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

TIER = "medium"
"""Builds and steps robot scenes on a GPU."""

CELL = "bench_075"
H = math.sqrt(0.5)
KEYBOX = {
    "left": (-0.36, 0.20, 0.06),
    "right": (0.12, -0.36, 0.22, 0.0, 0.0, H, H),
}
"""The keybox twin's arms, which it stood by copying the YAM's file: the left
base 60 mm up and back on the left, the right one 220 mm up on a riser, turned a
quarter turn to face across the bench. In the work-surface frame. The left is a
position, which keeps the heading the YAM's file gives it -- none."""


def _full(pose) -> tuple[float, ...]:
    """A keybox entry as a pose: a position keeps the YAM base's heading, which is the identity."""
    return tuple(pose) if len(pose) == 7 else (*pose, 0.0, 0.0, 0.0, 1.0)


POSITION_MM = 1.0
ANGLE_DEG = 0.1


def _pose_error(have, want) -> tuple[float, float]:
    """Position [mm] and rotation [deg] between two ``(x, y, z, qx, qy, qz, qw)`` poses."""
    have, want = np.asarray(have, dtype=np.float64), np.asarray(want, dtype=np.float64)
    position = float(np.linalg.norm(have[:3] - want[:3])) * 1000.0
    dot = abs(float(np.dot(have[3:] / np.linalg.norm(have[3:]), want[3:] / np.linalg.norm(want[3:]))))
    return position, math.degrees(2.0 * math.acos(min(1.0, dot)))


def _world(pose):
    from robosimstudio.scene.frames import in_surface
    from robosimstudio.scene.registry import get_cell

    return np.asarray(in_surface(get_cell(CELL), pose), dtype=np.float64)


def _build(robot):
    import robosimstudio as rss

    scene = rss.Scene(cell=CELL, name="arms_at")
    scene.add(robot)
    return scene.build()


def _yam_file() -> Path:
    from robosimstudio.paths import resolve_asset
    from robosimstudio.robots.registry import get_embodiment

    return Path(resolve_asset(get_embodiment("yam_bimanual").asset))


def _base_from_puck(sim, side: str) -> np.ndarray:
    """The world pose of ``{side}_arm``, read off the mount puck's collider.

    The collapse folds the base body into the world and keeps its shapes there at
    their world poses, so the base is the collider's pose less the collider's
    own pose in the base body, which the file states.
    """
    import newton

    from robosimstudio.core.math import transform_matrix, xform_from_matrix

    geom = ET.parse(_yam_file()).getroot().find(f".//body[@name='{side}_arm']/geom[@class='collision']")
    pos = [float(v) for v in geom.get("pos", "0 0 0").split()]
    w, x, y, z = (float(v) for v in geom.get("quat", "1 0 0 0").split())
    local = transform_matrix((*pos, x, y, z, w))
    model = sim.model
    flags = model.shape_flags.numpy()
    body = model.shape_body.numpy()
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    hits = [
        i
        for i, label in enumerate(model.shape_label)
        if label.split("/")[-2:-1] == [f"{side}_arm"] and int(flags[i]) & collide
    ]
    assert len(hits) == 1, f"{side}_arm has {len(hits)} colliders of its own"
    assert int(body[hits[0]]) == -1, "the base body was expected to be collapsed into the world"
    world = transform_matrix(model.shape_transform.numpy()[hits[0]])
    return np.asarray(xform_from_matrix(world @ np.linalg.inv(local)), dtype=np.float64)


def _reach(test: unittest.TestCase, sim, arm: str, target, steps: int = 400) -> float:
    """Solve for ``target`` with the tool pointing down, drive there, and return the TCP's miss [mm]."""
    ik = sim.arm(arm)
    seed = np.asarray(sim.pipeline.snapshot.joint_q, dtype=np.float64)
    solved = ik.solve_q(seed, target, (0.0, 0.0, -1.0))
    landed, _ = ik.fk_ee(np.asarray(solved, dtype=np.float32))
    test.assertLess(float(np.linalg.norm(landed - target)) * 1000.0, 2.0, f"{arm}: the solve itself missed")
    ik.set_joint_targets(np.asarray(solved)[ik.arm_coords], sim.control.joint_target_q)
    sim.run(steps).sync()
    position, _ = ik.ee_pose(sim.state)
    return float(np.linalg.norm(position - np.asarray(target))) * 1000.0


class TestTheYamArmsStandWhereArmsAtSays(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        cls.sim = _build(rss.robot("yam_bimanual", arms_at=KEYBOX))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def test_each_base_lands_on_its_pose(self) -> None:
        for side, pose in KEYBOX.items():
            with self.subTest(arm=side):
                position, angle = _pose_error(_base_from_puck(self.sim, side), _world(_full(pose)))
                print(f"[arms_at] {side} base off its pose by {position:.4f} mm, {angle:.4f} deg")
                self.assertLess(position, POSITION_MM)
                self.assertLess(angle, ANGLE_DEG)

    def test_it_is_the_registered_robot(self) -> None:
        """The registered file, its spec untouched -- cross-arm collisions as the robot states them."""
        from robosimstudio.robots.registry import get_embodiment

        registered = get_embodiment("yam_bimanual")
        built = self.sim.scene.embodiment
        self.assertEqual(registered.asset, built.spec.asset)
        self.assertEqual(
            (registered.cross_arm_collisions, registered.enable_self_collisions),
            (built.spec.cross_arm_collisions, built.spec.enable_self_collisions),
        )
        self.assertEqual(2, len(built.arm_shapes), "the arms were not grouped for cross-arm collisions")

    def test_it_is_the_default_robot_moved(self) -> None:
        """Same joints, coordinates, bodies, shapes and filtered pairs as the default build."""
        import robosimstudio as rss

        default = _build(rss.robot("yam_bimanual"))
        try:
            mine, theirs = self.sim.model, default.model
            for field in ("joint_count", "joint_coord_count", "joint_dof_count", "body_count", "shape_count"):
                with self.subTest(field=field):
                    self.assertEqual(getattr(theirs, field), getattr(mine, field))
            self.assertEqual(set(theirs.shape_collision_filter_pairs), set(mine.shape_collision_filter_pairs))
            self.assertEqual(list(theirs.joint_label), list(mine.joint_label))
        finally:
            default.close()

    def test_ik_reaches_beside_each_moved_base(self) -> None:
        """A target 300 mm out, 50 mm aside and 120 mm up from each base, in the base's own axes."""
        from robosimstudio.core.math import transform_point

        for side, pose in KEYBOX.items():
            with self.subTest(arm=side):
                target = transform_point(_world(_full(pose)), (0.30, 0.05, 0.12))
                miss = _reach(self, self.sim, side, target)
                print(f"[arms_at] {side} TCP {miss:.2f} mm from a target beside its moved base")
                self.assertLess(miss, 5.0)
                try:
                    import pyroki  # noqa: F401
                except ImportError:
                    continue
                # The URDF solver calibrates against the model and refuses an arm it
                # does not describe; serving the moved arm means it measured the new base.
                self.assertEqual("pyroki", self.sim.arm(side).solver_name)


class TestTheSameLayoutAsAMovedCopy(unittest.TestCase):
    def test_body_for_body(self) -> None:
        """The workaround this replaces, as the oracle: a copy of the file with the arms moved in it."""
        from dataclasses import replace

        import newton

        import robosimstudio as rss
        from robosimstudio.robots.registry import get_embodiment

        source = _yam_file()
        tree = ET.parse(source)
        root = tree.getroot()
        compiler = root.find("compiler")
        compiler.set("meshdir", str((source.parent / compiler.get("meshdir")).resolve()))
        for side, pose in KEYBOX.items():
            x, y, z, qx, qy, qz, qw = _full(pose)
            body = root.find(f"./worldbody/body[@name='{side}_arm']")
            body.set("pos", f"{x!r} {y!r} {z!r}")
            body.set("quat", f"{qw!r} {qx!r} {qy!r} {qz!r}")
        with tempfile.TemporaryDirectory() as folder:
            copy = Path(folder) / "arms.xml"
            tree.write(copy, encoding="unicode")
            moved = replace(get_embodiment("yam_bimanual"), asset=rss.AssetSpec(path=str(copy)))
            # The copy's arm poses are relative to its root, which at= stands at the surface frame's origin.
            oracle = _build(rss.robot(moved, at=(0.0, 0.0, 0.0)))
        sim = _build(rss.robot("yam_bimanual", arms_at=KEYBOX))
        try:
            poses = []
            for one in (sim, oracle):
                state = one.model.state()
                newton.eval_fk(one.model, one.model.joint_q, one.model.joint_qd, state)
                poses.append(state.body_q.numpy())
            self.assertEqual(poses[0].shape, poses[1].shape)
            worst = (0.0, 0.0)
            for have, want in zip(*poses, strict=True):
                position, angle = _pose_error(have, want)
                worst = (max(worst[0], position), max(worst[1], angle))
            print(f"[arms_at] arms_at vs a moved copy of the file: worst body {worst[0]:.5f} mm, {worst[1]:.5f} deg")
            self.assertLess(worst[0], 0.01)
            self.assertLess(worst[1], 0.01)
            np.testing.assert_allclose(
                sim.model.shape_transform.numpy(), oracle.model.shape_transform.numpy(), atol=1.0e-5
            )
        finally:
            sim.close()
            oracle.close()


class TestAnArmWithABaseBody(unittest.TestCase):
    """The Panda keeps its base as a body, so its handle names one and IK reads its frame off it."""

    POSE = (-0.45, 0.20, 0.05, 0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8))
    """Back on the left and raised 50 mm, yawed 45 degrees in."""

    def test_the_base_body_is_at_the_pose_and_ik_follows(self) -> None:
        import robosimstudio as rss

        sim = _build(rss.robot("panda", arms_at={"right": self.POSE}))
        try:
            handle = sim.handles.arms["right"]
            self.assertGreaterEqual(handle.base_body, 0)
            position, angle = _pose_error(sim.state.body_q.numpy()[handle.base_body], _world(self.POSE))
            print(f"[arms_at] panda base body off its pose by {position:.4f} mm, {angle:.4f} deg")
            self.assertLess(position, POSITION_MM)
            self.assertLess(angle, ANGLE_DEG)
            from robosimstudio.core.math import transform_point

            target = transform_point(_world(self.POSE), (0.45, 0.0, 0.25))
            miss = _reach(self, sim, "right", target)
            print(f"[arms_at] panda TCP {miss:.2f} mm from a target in front of its moved base")
            self.assertLess(miss, 5.0)
        finally:
            sim.close()

    def test_a_hand_welded_on_rides_the_moved_wrist(self) -> None:
        """``panda_robotiq`` welds its 2F-85 on after the arm is placed: the tool centre, seen
        from the base, is where it is on the robot left standing at its own mount."""
        import newton

        import robosimstudio as rss
        from robosimstudio.core.math import transform_matrix, xform_from_matrix

        def tcp_in_base(robot) -> np.ndarray:
            sim = _build(robot)
            try:
                state = sim.model.state()
                newton.eval_fk(sim.model, sim.model.joint_q, sim.model.joint_qd, state)
                base = state.body_q.numpy()[sim.handles.arms["right"].base_body]
                position, quat = sim.arm("right").ee_pose(state)
                tcp = transform_matrix((*position, *quat))
                return np.asarray(xform_from_matrix(np.linalg.inv(transform_matrix(base)) @ tcp), dtype=np.float64)
            finally:
                sim.close()

        moved = tcp_in_base(rss.robot("panda_robotiq", arms_at={"right": self.POSE}))
        default = tcp_in_base(rss.robot("panda_robotiq"))
        position, angle = _pose_error(moved, default)
        print(f"[arms_at] panda_robotiq TCP in its base frame differs by {position:.4f} mm, {angle:.4f} deg")
        self.assertLess(position, 0.01)
        self.assertLess(angle, 0.01)


if __name__ == "__main__":
    unittest.main()
