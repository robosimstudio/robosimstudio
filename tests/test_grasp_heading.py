# SPDX-License-Identifier: Apache-2.0
"""The two headings `sim.get_object_obb` returns are perpendicular, and stay so.

`grasp_yaw` is the long axis's own heading; `grasp_heading_deg` is what a
top-down grasp closes along. Handing the first to
`robot.grasp_frame(close_heading_deg=...)` type-checks, validates, plans,
executes and closes the jaws down the length of the object -- so nothing but a
score ever reports it. Measured on the hammer grid: holding the roll at
`grasp_yaw` scores 0/12, holding it at `grasp_heading_deg` scores 8/12, and the
latter is the only configuration that has ever picked the hammer up at the
second staged yaw.
"""

from __future__ import annotations

import math
import unittest

from robosimstudio_policy_agent.sim_tools import _grasp_heading_deg

TIER = "fast"
"""A pure function of one angle: no GPU, no scene."""


def _perp(a_deg: float, b_deg: float) -> float:
    """Separation of two undirected headings, 0..90 degrees."""
    d = abs(a_deg - b_deg) % 180.0
    return min(d, 180.0 - d)


class GraspHeadingIsSquareToTheLongAxis(unittest.TestCase):
    def test_at_every_staged_yaw(self):
        for yaw_deg in (-175, -90, -5, 0, 5, 44, 90, 175):
            with self.subTest(yaw_deg=yaw_deg):
                self.assertAlmostEqual(_perp(_grasp_heading_deg(math.radians(yaw_deg)), yaw_deg), 90.0, places=9)


if __name__ == "__main__":
    unittest.main()
