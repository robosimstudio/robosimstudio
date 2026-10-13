# SPDX-License-Identifier: Apache-2.0
"""A debugging snapshot, and the sign nobody would notice was backwards.

Two things here need holding and they need it at different tiers.

**The sign convention**, which is arithmetic and is tested as arithmetic. MuJoCo
reports ``contact.dist`` as a signed *gap* -- negative when two geoms overlap --
and :meth:`ContactQueries.penetrations` flips it so a depth is positive metres of
overlap. Flipped in one place precisely so it can be wrong in one place, and a
reading of "penetration 0.3" that actually meant three tenths of a millimetre
*apart* would look entirely reasonable in a report. ``ContactView`` takes plain
arrays, so this costs no device.

**What the file says about a real scene**, which needs one, and is where the
value is: a snapshot exists to answer "is this settled, and if not what is wrong",
so a test that only checked the shape of the JSON would pass over a scene sunk
ten millimetres into its own bin floor.
"""

from __future__ import annotations

import json
import unittest

import numpy as np

from robosimstudio.core.probes.contact import ContactView
from robosimstudio.scene.snapshot import FLAGGED_PENETRATION, summarise

TIER = "fast"
"""No GPU: numpy arrays and a dict. A snapshot of a built scene is not tested here."""


def _view(depths, geoms=None) -> ContactView:
    rows = len(depths)
    geoms = np.arange(2 * rows).reshape(rows, 2) if geoms is None else np.asarray(geoms)
    return ContactView(
        geoms=geoms,
        impulses=np.zeros((rows, 3)),
        depths=np.asarray(depths, dtype=float),
        # Identity, so a shape index reads as the geom that produced it and a
        # failure names something a reader can find.
        geom_to_shape=np.arange(int(geoms.max(initial=0)) + 2),
        dt=0.002,
    )


class TestPenetrationIsPositiveOverlap(unittest.TestCase):
    """MuJoCo's gap, flipped once, here."""

    def test_an_overlap_is_reported_as_a_positive_depth(self):
        rows = _view([-0.004]).penetrations()
        self.assertEqual(1, len(rows))
        self.assertAlmostEqual(0.004, float(rows[0][2]))

    def test_a_pair_merely_inside_the_margin_is_not_a_penetration(self):
        """The half that makes the flip worth testing.

        MuJoCo reports pairs it is *watching* as well as pairs that touch, with
        a positive gap. Counting those would make every scene look catastrophic
        and would make the worst-overlap figure meaningless.
        """
        self.assertEqual(0, len(_view([+0.002, +0.195]).penetrations()))

    def test_they_come_back_worst_first(self):
        rows = _view([-0.004, +0.002, -0.011]).penetrations()
        self.assertEqual([0.011, 0.004], [round(float(r[2]), 6) for r in rows])

    def test_a_floor_drops_the_shallow_ones(self):
        rows = _view([-0.0004, -0.011]).penetrations(deeper_than=0.001)
        self.assertEqual([0.011], [round(float(r[2]), 6) for r in rows])

    def test_no_contacts_is_an_empty_table_rather_than_a_raise(self):
        empty = _view([]).penetrations()
        self.assertEqual((0, 3), empty.shape)

    def test_the_pair_is_named_in_shape_indices(self):
        """Newton's, not MuJoCo's -- everything downstream indexes shapes."""
        rows = _view([-0.004], geoms=[[3, 5]]).penetrations()
        self.assertEqual([3, 5], [int(v) for v in rows[0][:2]])


class TestTheSummaryLine(unittest.TestCase):
    """What a reader sees before deciding whether to open the file."""

    def _snap(self, *, worst: float, moving: float = 0.0, finite: bool = True) -> dict:
        return {
            "task": {"step": 60},
            "finite": finite,
            "worst_penetration_mm": worst,
            "bodies": [{"name": "a", "static": False, "linear_speed": moving}],
        }

    def test_a_healthy_scene_reads_plainly(self):
        line = summarise(self._snap(worst=0.044))
        self.assertIn("step 60", line)
        self.assertIn("0.044 mm", line)
        self.assertNotIn("check", line)

    def test_a_millimetre_of_overlap_is_flagged(self):
        """The threshold separates two measured populations, not two round numbers.

        Settled objects sit at 0.01-0.09 mm; the contact model that was
        missing showed as 7.6-10.4 mm. Nothing has ever landed between.
        """
        self.assertIn("check", summarise(self._snap(worst=FLAGGED_PENETRATION * 1000 + 0.1)))

    def test_a_diverged_scene_says_so_first(self):
        self.assertIn("STATE NOT FINITE", summarise(self._snap(worst=0.0, finite=False)))

    def test_something_still_moving_is_counted(self):
        self.assertIn("1 still moving", summarise(self._snap(worst=0.0, moving=0.4)))
        self.assertIn("0 still moving", summarise(self._snap(worst=0.0, moving=0.0001)))


class TestTheSpeedsAreTheRightHalves(unittest.TestCase):
    """Newton's ``body_qd`` is ``(linear, angular)``, linear first."""

    def test_linear_is_the_first_three(self):
        from types import SimpleNamespace

        from robosimstudio.scene.snapshot import _objects

        def arr(values):
            return SimpleNamespace(numpy=lambda: np.asarray(values, dtype=float))

        env = SimpleNamespace(
            state=SimpleNamespace(
                body_q=arr([[0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]]),
                # Sliding at 0.3 m/s, spinning at 5 rad/s.
                body_qd=arr([[0.3, 0.0, 0.0, 0.0, 0.0, 5.0]]),
            ),
            handles=SimpleNamespace(objects={"cube": SimpleNamespace(body=0, static=False)}),
        )
        (row,) = _objects(env, SimpleNamespace(body_mass=arr([0.1])))
        self.assertAlmostEqual(0.3, row["linear_speed"])
        self.assertAlmostEqual(5.0, row["angular_speed"])


class TestItIsJsonAble(unittest.TestCase):
    def test_the_summary_input_round_trips(self):
        """A snapshot is written with ``json.dumps``, so nothing may be numpy.

        Cheap to assert and the failure is annoying to find: a stray
        ``np.float32`` raises inside the write, after the scene state that
        prompted the snapshot has already moved on.
        """
        snap = {"task": {"step": 1}, "finite": True, "worst_penetration_mm": 0.1, "bodies": []}
        self.assertEqual(snap, json.loads(json.dumps(snap)))


if __name__ == "__main__":
    unittest.main()
