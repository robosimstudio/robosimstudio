# SPDX-License-Identifier: Apache-2.0
"""The robot line reads the built robot's numbers, and a hanging thing is read by where it ends.

Three twins of one clip built the robot three ways: a copy of its file with the
wrist widened and stiffer servos, a copy that only changed a colour, and the
registered file built by a class of the twin's own that moved the arm bases. The
robot line read the spec, so the third passed. It now holds the built model's
per-DOF ranges, gains, force limits and armature, its shapes' friction and its
bodies' masses to the registered robot built alone, and names a class of the
twin's own. And a mug hung on a peg swings to where it swings: replays of one
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


if __name__ == "__main__":
    unittest.main()
