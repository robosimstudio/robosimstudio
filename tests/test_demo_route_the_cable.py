# SPDX-License-Identifier: Apache-2.0
"""``demo/route_the_cable``: from the robot's home pose, the routine routes the cable and seats it, by friction.

The demo ends by opening the studio with a **Run** button, so it is run the way
a person would: ``main()`` with :func:`robosimstudio.view` replaced by a stub
that records what it was handed, then a studio session over that scene with no
browser on it (:mod:`tests.fixtures.grasp`), and the button pressed. The routine
drives the arm panels -- the gizmos and the Grip buttons -- and nothing else; the
cable has no latch and no pin, so every grasp that held is the pads' friction.

What is asserted is the reference task's goal (VABAR's easy cable-routing tier):
the cable passes each spool on its side and ends in the port with its end in
it, both hands gone and the cable at rest. And what the old version of this
demo held its cable to: it never sinks into the bench or into a spool's shaft,
and the state stays finite. Each grasp is held to one capsule pitch (12 mm) of
slip between Close and Open.

Bands, not bits: the engine documents no determinism guarantee.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import re
import sys
import unittest
from pathlib import Path

import numpy as np

TIER = "slow"
"""A GPU, one scene built once, and a 42 s routine: 90 s at 28 steps a second, measured."""

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "demo" / "route_the_cable"
SCRIPT = FOLDER / "route_the_cable.py"
NAME = "Route the cable"

SLIP = 0.012
"""How far a pinched capsule may slide in the fingers between Close and Open [m]: one pitch.
Measured 1.2-10.8 mm over the four grasps in 19 runs."""


def _load(stem: str):
    """A module of the demo's folder, by file path, with the folder importable as the script finds it."""
    if str(FOLDER) not in sys.path:
        sys.path.insert(0, str(FOLDER))
    name = "_rss_demo_" + re.sub(r"\W", "_", stem)
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, FOLDER / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _has_rope() -> bool:
    import robosimstudio as rss

    return hasattr(rss, "RodSpec")


@unittest.skipUnless(_has_rope(), "the rope (rss.RodSpec) is not in this tree")
class TestRouteTheCable(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss
        from tests.fixtures.grasp import open_session, tick

        demo = _load("route_the_cable")
        cls.board = sys.modules["board"]
        opened: list = []
        real = rss.view
        rss.view = lambda sim, **kwargs: opened.append((sim, kwargs))
        try:
            demo.main()
        finally:
            rss.view = real
        ((cls.sim, kwargs),) = opened
        cls.session = open_session(cls.sim, 9402, kwargs.get("page"))
        cls.routines = kwargs.get("routines") or {}
        tick(cls.session, 30)  # the cable settles on the bench
        cls.start = {
            "routed": cls.board.routed(cls.sim),
            "widths": [cls.sim.gripper(a).width(cls.sim.state) for a in ("left", "right")],
            "homes": [
                float(np.linalg.norm(p.ik.ee_pose(cls.sim.state)[0] - p.ik.home_pose[0]))
                for p in cls.session.arms.values()
            ],
            "cable": cls.board.capsules(cls.sim),
        }
        # Watched every step: the lowest capsule, and the nearest a capsule comes to a shaft's axis below its rim.
        cls.worst = {"low": 1.0, "shaft": 1.0, "finite": True}
        board, sim = cls.board, cls.sim

        def watch() -> None:
            c = board.capsules(sim)
            if not np.isfinite(c).all():
                cls.worst["finite"] = False
                return
            cls.worst["low"] = min(cls.worst["low"], float(c[:, 2].min()))
            for x, y in board.PEGS:
                low = c[c[:, 2] < board.GROOVE[1]]
                if len(low):
                    cls.worst["shaft"] = min(cls.worst["shaft"], float(np.hypot(low[:, 0] - x, low[:, 1] - y).min()))

        cls.session.each_step(watch)
        cls.log: list = []
        cls.session.add_routines({NAME: functools.partial(cls.routines[NAME], log=cls.log)})
        cls.panel = cls.session.panels[-1]
        cls.panel.requested = NAME
        tick(cls.session, 1)
        cls.steps = 0
        while cls.panel.gen is not None and cls.steps < 6000:
            tick(cls.session, 1)
            cls.steps += 1
        cls.said = str(cls.panel._status.value) if cls.panel._status is not None else ""
        hands = [p.ik.ee_pose(sim.state)[0] for p in cls.session.arms.values()]
        cls.end = board.done(sim, hands)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_it_starts_at_home_with_open_hands_and_the_cable_where_the_reference_lays_it(self) -> None:
        self.assertEqual([NAME], list(self.routines), "the page offers the routine")
        self.assertLess(max(self.start["homes"]), 0.01, "a hand is not at home")
        for width in self.start["widths"]:
            self.assertGreater(width, 0.08, "a hand starts closed")
        cable = self.start["cable"]
        self.assertLess(float(np.abs(cable[:, 0] - self.board.START[0]).max()), 0.01, "the cable is not on its line")
        self.assertEqual(0.0, max(self.start["routed"].values()), "the cable starts in a seat")

    def test_the_routine_plays_to_its_end(self) -> None:
        self.assertIn("done", self.said, f"the routine did not finish: {self.said}")
        self.assertTrue(self.worst["finite"] and self.sim.state_is_finite())

    def test_the_cable_ends_routed_through_both_spools_and_into_the_port(self) -> None:
        end = self.end
        self.assertGreaterEqual(end["seat0"], 0.04, "not beside the first spool, on the robot's side")
        self.assertGreaterEqual(end["seat1"], 0.04, "not beside the second spool, on the far side")
        self.assertGreaterEqual(end["port"], 0.06, "fewer than three capsules in the port")
        self.assertEqual(1.0, end["end"], "the free end is not in the port")
        self.assertGreater(end["hands"], 0.05, "a hand is still on the cable")
        self.assertLessEqual(end["speed"], 0.05, "the cable is still moving")
        self.assertTrue(end["routed"])

    def test_every_grasp_held_by_friction(self) -> None:
        self.assertEqual(4, len(self.log), "four grasps: three placements and the plug")
        for entry in self.log:
            with self.subTest(step=entry["step"]):
                self.assertLess(entry["slip"], SLIP, f"slid {entry['slip'] * 1e3:.1f} mm in the fingers")
        spec = self.sim["cable"].entity.spec
        self.assertFalse(spec.allow_grasp_latch, "a hand could latch the cable")

    def test_the_cable_stays_out_of_the_bench_and_the_spools(self) -> None:
        radius = self.board.RADIUS
        self.assertGreater(self.worst["low"], radius - 0.001, f"a capsule went {self.worst['low'] * 1e3:.2f} mm up")
        # Seated against a shaft a capsule's axis is 8.25 mm from the spool's; a millimetre of overlap is contact.
        self.assertGreater(self.worst["shaft"], self.board.SHAFT_R + radius - 0.001, "the cable went into a spool")


class TestTheShape(unittest.TestCase):
    """The rules a demo script keeps: one ``main()`` with no flags, the parameters at the top, the studio last."""

    def setUp(self) -> None:
        self.source = SCRIPT.read_text()
        self.tree = ast.parse(self.source)

    def test_it_imports_the_public_api_and_its_board(self) -> None:
        imports = {a.name for n in self.tree.body if isinstance(n, ast.Import) for a in n.names}
        imports |= {n.module for n in self.tree.body if isinstance(n, ast.ImportFrom)}
        self.assertEqual({"math", "numpy", "board", "robosimstudio", "robosimstudio_studio.routine"}, imports)
        self.assertNotIn("argparse", self.source)
        self.assertNotIn(".hold(", self.source, "the cable is held by the fingers, not pinned")

    def test_it_ends_by_opening_the_studio_with_the_routine(self) -> None:
        main = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        self.assertEqual([], main.args.args, "main() takes no arguments")
        last = main.body[-1].value
        self.assertEqual("rss.view", ast.unparse(last.func))
        self.assertIn("routines", [k.arg for k in last.keywords])
        self.assertEqual("main()", ast.unparse(self.tree.body[-1].body[0]).strip())

    def test_its_docstring_is_the_page_it_becomes(self) -> None:
        blocks = [block.strip() for block in (ast.get_docstring(self.tree) or "").split("\n\n")]
        self.assertEqual("Route the cable", blocks[0])
        self.assertNotIn("\n", blocks[1], "the description is one sentence on one line")
        self.assertEqual(["**Key features:**"], blocks[3:4], "one paragraph, then the features")
        bullets = blocks[4].splitlines()
        self.assertTrue(3 <= len(bullets) <= 5 and all(b.startswith("* ") for b in bullets))

    def test_the_folder_says_how_to_run_it(self) -> None:
        readme = (FOLDER / "README.md").read_text(encoding="utf-8")
        self.assertIn("demo/route_the_cable/route_the_cable.py", readme)
        self.assertIn("media/route_the_cable.webp", readme)


if __name__ == "__main__":
    unittest.main()
