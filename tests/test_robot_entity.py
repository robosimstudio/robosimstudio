# SPDX-License-Identifier: Apache-2.0
"""``EntityEmbodiment``'s builder-side reading of a hand: which fingers one equality ties together."""

from __future__ import annotations

import unittest

TIER = "fast"
"""No GPU: a stand-in builder's mimic rows."""


class TestAHandsLinkageIsFoundEitherWayRound(unittest.TestCase):
    """``ActuationSpec`` damping, friction and armature reach every finger an equality carries."""

    @staticmethod
    def _builder(rows, enabled=None):
        from types import SimpleNamespace

        # Five one-DOF joints; mimic rows as (joint0, joint1), joint0 following joint1.
        return SimpleNamespace(
            constraint_mimic_joint0=[a for a, _ in rows],
            constraint_mimic_joint1=[b for _, b in rows],
            constraint_mimic_enabled=list(enabled) if enabled is not None else [True] * len(rows),
            joint_qd_start=[0, 1, 2, 3, 4],
            joint_armature=[0.0] * 5,
        )

    def test_the_driven_finger_may_be_either_side_of_the_row(self) -> None:
        from robosimstudio.robots.entity import EntityEmbodiment

        # The arena YAM's equality: the driven finger (1) is joint0, following the carried one (2).
        geared = EntityEmbodiment._geared_dofs(self._builder([(1, 2)]))
        self.assertEqual((1, 2), geared[1])
        self.assertEqual((1, 2), geared[2])
        self.assertNotIn(0, geared)

    def test_a_chain_of_rows_is_one_linkage_and_a_disabled_row_ties_nothing(self) -> None:
        from robosimstudio.robots.entity import EntityEmbodiment

        geared = EntityEmbodiment._geared_dofs(self._builder([(2, 1), (3, 2), (4, 0)], enabled=(True, True, False)))
        self.assertEqual((1, 2, 3), geared[3])
        self.assertNotIn(4, geared)

    def test_a_row_ending_at_minus_one_joins_nothing(self) -> None:
        """Newton keeps -1 ends when it remaps after a collapse: read through -1, two hands' fingers were one linkage."""
        from robosimstudio.robots.entity import EntityEmbodiment

        geared = EntityEmbodiment._geared_dofs(self._builder([(1, -1), (3, -1), (4, 0)]))
        self.assertNotIn(1, geared)
        self.assertNotIn(3, geared)
        self.assertEqual((0, 4), geared[4])

    def test_the_replays_read_the_same_linkages_off_the_model(self) -> None:
        """``physics._mimic_linkage`` reads a finalized model, whose ``joint_qd_start`` has its closing entry."""
        from types import SimpleNamespace

        import numpy as np

        from robosimstudio_real2sim_agent.physics import _mimic_linkage

        def array(values):
            return SimpleNamespace(numpy=lambda v=values: np.asarray(v))

        model = SimpleNamespace(
            constraint_mimic_joint0=array([2, 3, 1]),
            constraint_mimic_joint1=array([1, -1, -1]),
            constraint_mimic_enabled=array([True, True, True]),
            joint_qd_start=array([0, 1, 2, 3, 4, 5]),
            joint_dof_count=5,
        )
        self.assertEqual({1: (1, 2), 2: (1, 2)}, _mimic_linkage(model))


if __name__ == "__main__":
    unittest.main()
