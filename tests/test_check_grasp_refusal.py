# SPDX-License-Identifier: Apache-2.0
"""``sim.check_grasp`` must refuse an unknown arm, not answer held=False.

The silent path cost a study cell its last iterations: an agent guessed role
words ("work", "support") as the ``arm`` argument, ``arms.get`` missed, and
the tool answered a confident False on every genuine question after it.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from robosimstudio_policy_agent.refusal import ToolRefusal
from robosimstudio_policy_agent.sim_tools import SimTools

TIER = "fast"


def _tool(arms: dict, work_arm: str) -> SimTools:
    # __new__ skips __init__'s home-pose probe; check_grasp touches only these.
    tool = SimTools.__new__(SimTools)
    tool.env = SimpleNamespace(handles=SimpleNamespace(arms=arms, work_arm=work_arm))
    tool.bus = SimpleNamespace(call=lambda fn: fn())
    return tool


class TestCheckGraspRefusal(unittest.TestCase):
    def test_unknown_arm_refuses_and_names_the_vocabulary(self) -> None:
        tool = _tool({"left": object(), "right": object()}, "left")
        with self.assertRaises(ToolRefusal) as ctx:
            tool.check_grasp("wrench", arm="work")
        message = str(ctx.exception)
        self.assertIn("'work'", message)
        self.assertIn("left", message)
        self.assertIn("describe_arm", message)

    def test_known_arm_still_answers_without_probes(self) -> None:
        # No contact probe on the env: the documented held=False path.
        tool = _tool({"left": object()}, "left")
        self.assertEqual(
            {"object_name": "wrench", "held": False},
            tool.check_grasp("wrench", arm="left"),
        )

    def test_default_arm_still_answers_without_probes(self) -> None:
        tool = _tool({"left": object()}, "left")
        self.assertFalse(tool.check_grasp("wrench")["held"])


if __name__ == "__main__":
    unittest.main()
