# SPDX-License-Identifier: Apache-2.0
"""``examples/08_demos/fold_the_towel``: from the home pose to a towel folded in half, held by friction alone.

The demo's routine is played the way its page plays it: through the Run button, with
the arm panels taking each step (:mod:`tests.fixtures.grasp`), so the gizmos move, the
arms follow through their IK and setpoint law, and the towel comes over because the
pads hold it. Nothing in the scene is pinned to a hand.

The assertions are the cloth_fold suite's goal -- the span across the fold at most 0.6
of the settled span, a fold score of at least 0.85, the towel lying still and released
-- and the physical ones the old, latched demo was held to: the edge in the hand stays
in the hand to the top of the arc, the near half stays on the bench, the fingers do not
go through it, and the sheet stays finite.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import unittest

import numpy as np

from tests.fixtures.examples import ROOT
from tests.fixtures.grasp import open_session, tick

TIER = "medium"
"""Builds the towel scene and plays the 22 s fold through a page with no browser on it: about 75 s."""

DEMO = ROOT / "examples" / "08_demos" / "fold_the_towel" / "fold_the_towel.py"


def _load():
    name = "_rss_demo_fold_the_towel"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, DEMO)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _lowest(sim, shapes) -> float:
    """The lowest world z over these shapes [m]: spheres, capsules (along local z) and boxes."""
    import newton

    from robosimstudio.core.math import quat_to_mat

    model = sim.model
    body_q = sim.state.body_q.numpy().astype(np.float64)
    kinds, xforms = model.shape_type.numpy(), model.shape_transform.numpy().astype(np.float64)
    scales, bodies = model.shape_scale.numpy().astype(np.float64), model.shape_body.numpy()
    corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=np.float64)
    low = np.inf
    for shape in shapes:
        pose = body_q[bodies[shape]]
        rot = quat_to_mat(pose[3:7])
        centre = pose[:3] + rot @ xforms[shape][:3]
        up = (rot @ quat_to_mat(xforms[shape][3:7]))[2]  # the world z of each local axis
        kind, scale = int(kinds[shape]), scales[shape]
        if kind == int(newton.GeoType.SPHERE):
            low = min(low, centre[2] - scale[0])
        elif kind == int(newton.GeoType.CAPSULE):
            low = min(low, centre[2] - abs(up[2]) * scale[1] - scale[0])
        elif kind == int(newton.GeoType.BOX):
            low = min(low, float((centre[2] + (corners * scale) @ up).min()))
    return float(low)


class TestTheRoutineFoldsTheTowel(unittest.TestCase):
    """One build, one press of Run, and every reading taken along the way."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        demo = cls.demo = _load()
        sim = cls.sim = demo.build()
        cls.surface = rss.get_cell(demo.CELL).surface_z
        ids = np.asarray(sim["towel"].entity.particles, dtype=np.int64)
        model = sim.model
        labels = [str(label).rsplit("/", 1)[-1] for label in model.body_label]
        bodies = model.shape_body.numpy()
        fingers = [
            s
            for s in range(model.shape_count)
            if bodies[s] >= 0 and labels[bodies[s]].startswith("left_link_") and "finger" in labels[bodies[s]]
        ]

        session = cls.session = open_session(sim, 9590)
        session.add_routines({"Fold the towel": demo.routine})
        panel = session.panels[-1]
        left = session.arms["left"]
        hz = float(sim.request.physics.control_hz)

        def particles() -> np.ndarray:
            return sim.state.particle_q.numpy().astype(np.float64)[ids]

        def tool() -> np.ndarray:
            return np.asarray(left.ik.ee_pose(sim.state)[0], dtype=np.float64)

        tick(session, 5)
        rest = cls.rest = particles()
        near = rest[:, 1] < rest[:, 1].min() + 0.012  # the edge across from the hand
        r = cls.readings = {"near_lift": 0.0, "near_slide": 0.0, "fingers": np.inf, "top_z": -np.inf}
        panel.requested = "Fold the towel"
        tick(session, 1)
        squeezed_at, steps, pinched = None, 0, None
        while panel.gen is not None and steps < round(60 * hz):
            tick(session, 1)
            steps += 1
            q, hand = particles(), tool()
            if squeezed_at is None and left.gripper_command == demo.SQUEEZE:
                squeezed_at = steps
            # What the pads hold once the squeeze has risen and settled (0.2 + 0.3 + 1.0 s in the routine).
            if squeezed_at is not None and pinched is None and steps == squeezed_at + round(1.5 * hz):
                pinched = np.flatnonzero(np.linalg.norm(q - hand, axis=1) < 0.020)
                r["pinched"] = len(pinched)
            if pinched is not None and left.gripper_command == demo.SQUEEZE and hand[2] > r["top_z"]:
                r["top_z"] = float(hand[2])
                r["held_top"] = float((np.linalg.norm(q[pinched] - hand, axis=1) < 0.025).mean())
                r["rise_top"] = float(q[pinched, 2].mean() - rest[pinched, 2].mean())
            if steps % 5 == 0:
                r["near_lift"] = max(r["near_lift"], float(q[near, 2].max() - cls.surface))
                r["near_slide"] = max(
                    r["near_slide"], float(np.linalg.norm(q[near, :2] - rest[near, :2], axis=1).max())
                )
                r["fingers"] = min(r["fingers"], _lowest(sim, fingers) - cls.surface)
                r["finite"] = bool(np.isfinite(q).all()) and r.get("finite", True)
        r["seconds"] = steps / hz
        r["said"] = str(panel._status.value) if panel._status is not None else ""
        r["open"] = left.gripper_command == 0.0
        cls.end = particles()
        cls.pinched = pinched if pinched is not None else np.zeros(0, dtype=np.int64)
        cls.measures = demo.fold_measures(sim)
        print(f"\n[fold] {cls.readings} {cls.measures}", flush=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_routine_plays_to_its_end(self) -> None:
        self.assertIn("done", self.readings["said"])
        self.assertTrue(self.readings["finite"] and self.sim.state_is_finite())

    def test_the_towel_is_folded_in_half_and_put_down(self) -> None:
        """The cloth_fold suite's goal, read at the end of the routine's five still seconds."""
        m = self.measures
        self.assertLessEqual(m["fold_extent"], 0.6, "the span across the fold did not halve")
        self.assertGreaterEqual(m["fold_score"], 0.85, "the far half did not land on the near one")
        self.assertLess(m["settle_speed"], 0.02, "the towel is still moving")
        self.assertTrue(self.readings["open"], "the hand still holds it")

    def test_the_pads_held_the_edge_to_the_top_of_the_arc(self) -> None:
        r = self.readings
        self.assertGreater(r["pinched"], 10, "the pads closed on nothing")
        self.assertGreaterEqual(r["held_top"], 0.5, "the edge slipped out of the hand")
        self.assertGreater(r["rise_top"], 0.06, "the edge did not come up with the hand")

    def test_the_held_edge_was_carried_over_the_crease_and_laid_down(self) -> None:
        """Past the crease by 0.10 m: the edge is laid 12.5 mm short of its mirror, where the
        latched demo carried it to the mirror (0.12 m) -- a slab's crease loop takes up cloth."""
        edge = self.end[self.pinched].mean(axis=0)
        self.assertLess(edge[1], self.demo.CREASE_Y - 0.10)
        self.assertLess(edge[2] - self.surface, 0.025)

    def test_the_near_half_stayed_on_the_bench(self) -> None:
        """Lifted no more than the latched demo's 12 mm, and dragged only by the squeeze.

        The squeeze drags the near edge's free corner 26-31 mm: the inner pad sweeps 47 mm of
        towel as the jaws close, and the press holds the other side. A hand half shut before it
        came down swept less, and pinched nothing at 0.4-0.6 of its grip. The fold itself moves
        the near half 4-7 mm on average, the suite's drag term."""
        r = self.readings
        self.assertLess(r["near_lift"], 0.02, "the half on the bench was lifted")
        self.assertLess(r["near_slide"], 0.035, "and was dragged along")
        self.assertLess(self.measures["fixation_dist"], 0.01)

    def test_the_fingers_rest_on_the_bench_and_go_no_further(self) -> None:
        """The pinch is made with the fingertips on the bench, so they touch it: MuJoCo lets them in 0.3 mm."""
        self.assertGreater(self.readings["fingers"], -0.001)


class TestThePage(unittest.TestCase):
    def test_the_page_opens_looking_at_the_towel_and_offers_the_fold(self) -> None:
        """The page's own view is closer than the work surface's, on the towel, from across the bench."""
        from types import SimpleNamespace

        import robosimstudio as rss

        demo = _load()
        opened = []
        real = rss.view
        rss.view = lambda sim, **kwargs: opened.append((sim, kwargs))
        try:
            demo.main()
        finally:
            rss.view = real
        sim, kwargs = opened[0]
        try:
            self.assertEqual(["Fold the towel"], list(kwargs["routines"]))
            aimed = []
            kwargs["page"](SimpleNamespace(aim_camera=lambda eye, look_at: aimed.append((eye, look_at))))
            ((eye, look_at),) = aimed
            middle = sim["towel"].particle_q.mean(axis=0)
            np.testing.assert_allclose(look_at, middle, atol=0.01)
            eye = np.asarray(eye) - middle
            self.assertGreater(eye[0], 0.3, "from the bench's open side, across from the arms")
            self.assertGreater(eye[2], 0.2, "and from above")
            self.assertLess(float(np.linalg.norm(eye)), 0.6)
        finally:
            sim.close()


class TestTheDemoHasTheShapeOfOne(unittest.TestCase):
    """The rules ``tests/test_examples.py`` holds the numbered examples to, at a demo's length."""

    MAX_LINES = 300
    """A demo that finishes its job carries its routine and its measure, so it runs longer than a scene."""

    def setUp(self) -> None:
        self.source = DEMO.read_text()
        self.tree = ast.parse(self.source)

    def test_it_is_a_script_with_a_main_that_ends_on_the_studio(self) -> None:
        self.assertLessEqual(len(self.source.splitlines()), self.MAX_LINES)
        self.assertNotIn("argparse", self.source)
        for withheld in ("examples.tasks", "taskbuild", "robosimstudio_bench", "solutions", "rgbs", "vabar"):
            self.assertNotIn(withheld, self.source, "a demo stands on the public library alone")
        for latch in ("held_by", "ParticleDragger", "latch("):
            self.assertNotIn(latch, self.source, "nothing holds the towel but the fingers")
        main = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        self.assertEqual([], main.args.args)
        self.assertEqual("rss.view", ast.unparse(main.body[-1].value.func))
        self.assertEqual("__name__ == '__main__'", ast.unparse(self.tree.body[-1].test))

    def test_its_docstring_is_the_page_it_becomes(self) -> None:
        blocks = [block.strip() for block in (ast.get_docstring(self.tree) or "").split("\n\n")]
        self.assertTrue(2 <= len(blocks[0].split()) <= 5 and not blocks[0].endswith("."), blocks[0])
        self.assertNotIn("\n", blocks[1])
        self.assertTrue(blocks[1].endswith("."))
        self.assertIn(len(blocks), (2, 3), "at most one paragraph")


if __name__ == "__main__":
    unittest.main()
