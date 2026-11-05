# SPDX-License-Identifier: Apache-2.0
"""What lets the paper fold be done by two hands, by contact, each on its own side, read off the example's source.

``examples/03_materials/05_paper`` folds an A4 sheet in half with both of the robot's hands. A parallel
jaw cannot get a pad under a sheet lying flat (seven top-down pinches raised one 0 mm), so the sheet lies
on a box with its near edge 35 mm past the box's edge, where each hand slides its lower pad under it
from its own corner. The sheet lies across the middle so the two corners are one in each hand's half;
on the bench edge the old layout hung it over, only the right arm reached and the left hand crossed
into its space to help. These checks hold that layout and its realism settings in place without
building anything: the overhang the pads close on, the sheet on the box and across the middle, the arms
that collide with each other, a grip capped at 15 N or less and fed the sheet's push, a pad friction
under 1, and nothing pinned to a hand.
"""

from __future__ import annotations

import ast
import math
import sys
import unittest
from pathlib import Path

from tests.fixtures.examples import EXAMPLES

TIER = "fast"
"""Parses one file; builds nothing."""

FOLD = EXAMPLES / "03_materials" / "05_paper.py"


def _locals() -> dict:
    """The literal locals ``main()`` opens with: the sheet, the box, the overhang."""
    tree = ast.parse(FOLD.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    out = {}
    for node in main.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            if isinstance(target, ast.Name):
                out[target.id] = value
            elif isinstance(target, ast.Tuple):
                out.update({t.id: v for t, v in zip(target.elts, value, strict=True)})
    return out


def _keywords(source: str, call: str) -> dict:
    """The literal keyword arguments of the one call to ``call`` (an attribute or a name) in the source."""
    found = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and call in (getattr(node.func, "attr", ""), getattr(node.func, "id", ""))
    ]
    assert len(found) == 1, f"{len(found)} calls to {call}"
    return {kw.arg: ast.literal_eval(kw.value) for kw in found[0].keywords if isinstance(kw.value, ast.Constant)}


class TestTheFoldIsLaidOutForTwoHands(unittest.TestCase):
    def setUp(self) -> None:
        self.found = _locals()
        self.source = FOLD.read_text()

    def test_the_pads_close_on_the_overhang_clear_of_the_box(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from recording import takes

        overhang = self.found["overhang"]
        far_side = takes.FOLD_PINCH + 0.006  # the 12 mm pads' far side, in from the edge
        self.assertGreaterEqual(
            overhang - far_side, 0.005, f"the pads end {far_side * 1e3:.0f} mm in, the box at {overhang * 1e3:.0f}"
        )
        self.assertLessEqual(overhang, 0.05, f"the edge runs {overhang * 1e3:.0f} mm past the box")

    def test_the_sheet_lies_on_the_box_across_the_middle(self) -> None:
        size, box, corner, overhang = (self.found[k] for k in ("size", "box", "corner", "overhang"))
        self.assertAlmostEqual(corner[2], box[2] + 0.005, places=6, msg="laid 5 mm over the box's top")
        self.assertAlmostEqual(corner[1], -size[1] / 2, places=6, msg="one corner in each hand's half")
        self.assertGreaterEqual(box[0], size[0] - overhang, "the rest of the sheet is on the box")
        self.assertGreater(box[1], size[1], "and the box is wider than it")

    def test_the_arms_collide_with_each_other(self) -> None:
        from robosimstudio.robots.registry import get_embodiment

        self.assertTrue(get_embodiment("yam_bimanual").cross_arm_collisions, "the YAM's own setting")
        self.assertNotIn("cross_arm_collisions=False", self.source)

    def test_the_grip_is_capped_and_feels_the_sheet(self) -> None:
        hands = _keywords(self.source, "ActuationSpec")
        # One motor drives a hand's two fingers through the asset's joint equality: its cap bounds the driven
        # finger's slide, not either finger's contact push, which test_fold_paper_robot holds to 15 N.
        self.assertLessEqual(hands["effort_limit"], 15.0, "the motor's cap")
        # Stiff enough that a finger held 0.2 mm off shut already asks the whole cap: Close stalls at the cap.
        self.assertGreaterEqual(hands["target_ke"] * 0.0002, hands["effort_limit"])
        coupling = _keywords(self.source, "coupler")
        self.assertTrue(coupling["slide_feedback"], "the sheet's push reaches the fingers' slides")
        self.assertTrue(coupling["contact_forces"])

    def test_no_pad_grips_paper_harder_than_friction_one(self) -> None:
        """VBD mixes paper against a pad as the geometric mean: read the pads' friction off the hand's asset."""
        import xml.etree.ElementTree as ET

        from robosimstudio.paths import resolve_path
        from robosimstudio_materials.paper import PAPER

        asset = ET.parse(resolve_path("embodiments/yam/yam_bimanual_arms.xml"))
        pads = [float(g.get("friction").split()[0]) for g in asset.iter("geom") if g.get("friction")]
        self.assertTrue(pads)
        self.assertLessEqual(math.sqrt(PAPER.friction * max(pads)), 1.0)
        self.assertNotIn("shape_material_mu", self.source, "and the page writes no friction of its own")

    def test_nothing_on_the_page_pins_the_sheet_to_a_hand(self) -> None:
        for attachment in ("grab_set_requested", "follow_arm", "particle_sets", "pinned=", "attach"):
            self.assertNotIn(
                attachment, self.source, f"the fold is carried by contact; {attachment!r} is an attachment's"
            )


if __name__ == "__main__":
    unittest.main()
