# SPDX-License-Identifier: Apache-2.0
"""The robot line reads the built robot's numbers, and a hanging thing is read by where it ends.

Three twins of one clip built the robot three ways: a copy of its file with the
wrist widened and stiffer servos, a copy that only changed a colour, and the
registered file built by a class of the twin's own that moved the arm bases. The
robot line read the spec, so the third passed. It now holds the built model's
per-DOF ranges, gains, force limits and armature, its joints' frames and axes,
its shapes' size, place, contact and what they collide with, and its links'
masses and inertia to the registered robot built alone, and names a class of
the twin's own. And a mug hung on a peg swings to where it swings: replays of one
twin left it 1 to 13 mm and up to 92 degrees from the written run, which failed
``built`` on a turn that means nothing.
"""

from __future__ import annotations

import types
import unittest

import numpy as np

from robosimstudio_real2sim_agent import physics

TIER = "fast"


def _arr(values) -> types.SimpleNamespace:
    a = np.asarray(values, dtype=np.float64)
    return types.SimpleNamespace(numpy=lambda: a)


def _model(name: str, joint6: float = 2.0944, gain: float = 80.0, mu: float = 1.0, base: float = 0.0):
    labels = [f"{name}/arm/joint1", f"{name}/arm/joint6", "table"]
    return types.SimpleNamespace(
        joint_label=labels,
        joint_qd_start=_arr([0, 1, 2, 3]),
        joint_limit_lower=_arr([-1.0, -joint6, 0.0]),
        joint_limit_upper=_arr([1.0, joint6, 0.0]),
        joint_target_ke=_arr([gain, 30.0, 0.0]),
        joint_target_kd=_arr([6.0, 2.0, 0.0]),
        joint_effort_limit=_arr([1e6, 1e6, 0.0]),
        joint_armature=_arr([0.1, 0.1, 0.0]),
        shape_label=[f"{name}/arm/finger_geom", "table_geom"],
        shape_material_mu=_arr([mu, 0.3]),
        body_label=[f"{name}/arm/link_1", "table"],
        body_mass=_arr([1.0, 5.0 + base]),
    )


class CompareModels(unittest.TestCase):
    def test_the_registered_robot_differs_in_nothing(self) -> None:
        self.assertEqual(physics.compare_models(_model("yam"), _model("yam"), "yam", "yam"), [])

    def test_a_widened_joint_a_stiffer_servo_and_grippier_fingers_are_named(self) -> None:
        got = physics.compare_models(_model("yam", joint6=3.1416, gain=200.0, mu=2.0), _model("yam"), "yam", "yam")
        text = "; ".join(got)
        self.assertIn("joint range joint6 -3.142 (registered -2.094)", text)
        self.assertIn("servo gain joint1 200 (registered 80)", text)
        self.assertIn("friction finger_geom 2 (registered 1)", text)

    def test_what_is_not_the_robot_and_another_name_are_matched_by_label(self) -> None:
        mine = _model("twin_robot", base=3.0)  # the table weighs more: not the robot's
        self.assertEqual(physics.compare_models(mine, _model("yam"), "twin_robot", "yam"), [])


def _parts(name: str, **edit):
    """A robot of a world-standing base shape, a link with one finger shape, and the base joint and one more."""
    base_x = edit.get("base_x", 0.0)
    finger = [0.01, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0]
    finger[2] += edit.get("finger_dz", 0.0)
    return types.SimpleNamespace(
        joint_label=[f"{name}/base_joint", f"{name}/joint1"],
        joint_qd_start=_arr([0, 0, 1]),
        joint_parent=_arr([-1, 0]),
        joint_X_p=_arr([[base_x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]]),
        joint_axis=_arr([[0.0, 0.0, 1.0], edit.get("axis", [0.0, 0.0, 1.0])]),
        shape_label=[f"{name}/base_geom", f"{name}/finger_geom"],
        shape_body=_arr([-1, 1]),
        shape_transform=_arr([[base_x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], finger]),
        shape_scale=_arr([[0.1, 0.1, 0.1], [0.01, 0.01, edit.get("finger_len", 0.03)]]),
        shape_flags=_arr([3, edit.get("finger_flags", 3)]),
        shape_material_mu=_arr([1.0, 1.0]),
        body_label=[f"{name}/base", f"{name}/link_1"],
        body_mass=_arr([2.0, 0.5]),
        body_inertia=_arr([np.eye(3) * 0.01, np.eye(3) * edit.get("inertia", 0.001)]),
    )


class CompareModelsParts(unittest.TestCase):
    def test_where_the_robot_stands_is_the_twins(self) -> None:
        self.assertEqual(physics.compare_models(_parts("yam", base_x=0.75), _parts("yam"), "yam", "yam"), [])

    def test_a_longer_finger_a_moved_shape_and_a_turned_axis_are_named(self) -> None:
        mine = _parts("yam", finger_len=0.05, finger_dz=0.004, axis=[0.0, 1.0, 0.0], inertia=0.002)
        text = "; ".join(physics.compare_models(mine, _parts("yam"), "yam", "yam"))
        self.assertIn("shape size finger_geom", text)
        self.assertIn("shape place on its link finger_geom 4.0 mm and 0.0 deg off", text)
        self.assertIn("joint axis joint1", text)
        self.assertIn("inertia link_1", text)

    def test_a_shape_made_not_to_collide_is_named_and_one_only_hidden_is_not(self) -> None:
        import newton

        visible = int(newton.ShapeFlags.VISIBLE)
        hidden = physics.compare_models(_parts("yam", finger_flags=3 & ~visible), _parts("yam"), "yam", "yam")
        self.assertEqual(hidden, [], "visibility is the look's")
        ghost = physics.compare_models(_parts("yam", finger_flags=visible), _parts("yam"), "yam", "yam")
        self.assertIn("what a shape collides with finger_geom", "; ".join(ghost))

    def test_float_noise_in_a_frame_is_not_a_departure(self) -> None:
        self.assertEqual(physics.compare_models(_parts("yam", finger_dz=1e-5), _parts("yam"), "yam", "yam"), [])


class Hangs(unittest.TestCase):
    def _end(self, things=("mug_tree",), static=False, robot=False) -> dict:
        return {"end_on": {"things": list(things), "static": static, "robot": robot}}

    def test_a_mug_on_the_same_peg_in_both_runs_hangs(self) -> None:
        self.assertTrue(physics._hangs(self._end(), self._end()))

    def test_one_resting_on_the_table_held_or_on_something_else_does_not(self) -> None:
        self.assertFalse(physics._hangs(self._end(static=True), self._end()))
        self.assertFalse(physics._hangs(self._end(robot=True), self._end()))
        self.assertFalse(physics._hangs(self._end(things=("rack",)), self._end()))
        self.assertFalse(physics._hangs(self._end(things=()), self._end(things=())))
        self.assertFalse(physics._hangs({}, self._end()), "a run that read no contacts is read as before")

    def test_a_hanging_mugs_turn_is_no_miss_but_its_place_is(self) -> None:
        written = {"things": {"mug": {"end": [0.36, -0.09, 0.94], "end_quat": [0, 0, 0, 1.0], **self._end()}}}
        swung = {"things": {"mug": {"end": [0.362, -0.091, 0.94], "end_quat": [0, 0, 0.7071, 0.7071], **self._end()}}}
        self.assertEqual(physics._misses(swung, written, ["mug"], bystanders=False, excursion=False), [])
        fell = {"things": {"mug": {"end": [0.36, -0.09, 0.76], "end_quat": [0, 0, 0, 1.0], **self._end(static=True)}}}
        self.assertEqual(len(physics._misses(fell, written, ["mug"], bystanders=False, excursion=False)), 1)
        lying = {"things": {"mug": {"end": [0.36, -0.09, 0.94], "end_quat": [0, 0, 0.7071, 0.7071]}}}
        self.assertEqual(
            len(
                physics._misses(
                    lying,
                    {"things": {"mug": {**written["things"]["mug"], "end_on": None}}},
                    ["mug"],
                    bystanders=False,
                    excursion=False,
                )
            ),
            1,
            "without end_on the turn still counts",
        )


class TheRegisteredBuildIsKept(unittest.TestCase):
    """The robot line compares against the registered robot built alone; that build is the robot's and the
    physics', so it is made once a process and read back off the host after its model is gone."""

    def test_a_kept_model_reads_as_the_model_did(self) -> None:
        kept = physics._Kept(_model("yam"))
        self.assertEqual(physics.compare_models(_model("yam"), kept, "yam", "yam"), [])
        self.assertIn("joint6", "; ".join(physics.compare_models(_model("yam", joint6=3.1416), kept, "yam", "yam")))

    def test_the_second_replay_builds_nothing(self) -> None:
        from unittest import mock

        import robosimstudio as rss

        builds: list[str] = []

        class FakeScene:
            def __init__(self, **kwargs) -> None:
                pass

            def add(self, spec) -> None:
                pass

            def build(self, **kwargs):
                builds.append("built")
                return types.SimpleNamespace(model=_model("yam"), close=lambda: None)

        sim = types.SimpleNamespace(
            scene=types.SimpleNamespace(embodiment=types.SimpleNamespace(spec=types.SimpleNamespace(name="yam"))),
            request=types.SimpleNamespace(physics=("a physics", 1)),
            model=_model("yam"),
        )
        reg = types.SimpleNamespace(name="yam_kept_test")
        physics._REGISTERED.clear()
        with mock.patch.object(rss, "Scene", FakeScene), mock.patch.object(rss, "robot", lambda spec: spec):
            first = physics.built_departures(sim, reg)
            second = physics.built_departures(sim, reg)
        self.assertEqual(first, second)
        self.assertEqual(["built"], builds, "the registered robot was built again for the second replay")
        physics._REGISTERED.clear()


if __name__ == "__main__":
    unittest.main()
