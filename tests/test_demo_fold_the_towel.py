# SPDX-License-Identifier: Apache-2.0
"""``examples/09_demos/02_fold_the_towel.py``: the towel starts in the hand, and carrying the hand folds it.

The example ends by opening the studio, so :func:`robosimstudio.view` is replaced
(:mod:`tests.fixtures.examples`) and the hand is driven here the way the page drives it
when the wrist gizmo is dragged: an operational-space setpoint walked towards a cursor and
solved to joint targets once per control step (``PanelArm._drive``). The cursor goes over
in an arc about the towel's crease, which is the gesture that folds a towel in half.

The assertions are physical rather than about the page: the patch in the hand stays in
the hand, the half lying on the bench stays lying, the fingers never go into the bench,
the sheet stays finite, and it ends folded. Measured on Newton 1.5 and 1.6 alike: the
patch drifts 2.3 mm from the tool centre (the hand turning a little on the way over), the
near edge lifts 12 mm and slides 10 mm, the fingertips stay 8.6 mm over the bench, and the
span across the fold closes from 0.300 m to 0.175 m.

The seam the example stands on, ``rss.sheet(held_by=...)``, is
:class:`~robosimstudio.domains.cloth.EntityHeldCloth`, and its rules are checked here too:
a held patch weighs nothing, only what hangs from the holder stops touching particles, and
the hold survives a reset because it is a hook the entity contributed at build.
"""

from __future__ import annotations

import ast
import unittest

import numpy as np

from tests.fixtures.examples import EXAMPLES, _run

TIER = "medium"
"""Builds the towel scene once on a GPU, settles it and carries the edge over: about 30 s."""

REL = "09_demos/02_fold_the_towel.py"
CELL = "bench_0728"
CREASE_Y = 0.06
"""The towel's centreline across the fold, in the bench's work-surface frame [m]."""


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
        else:
            raise AssertionError(f"a finger shape of kind {kind} that this measure does not read")
    return float(low)


class TestTheTowelFolds(unittest.TestCase):
    """One build, and every test starts from ``sim.reset()``, which is what the page does when it opens."""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        cls.sim, cls.kwargs = _run(REL)
        cls.surface = rss.get_cell(CELL).surface_z
        cls.towel = cls.sim["towel"]
        cls.entity = cls.towel.entity
        cls.held = np.asarray(cls.entity.held, dtype=np.int64)
        cls.ids = np.asarray(cls.entity.particles, dtype=np.int64)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sim.close()

    def setUp(self) -> None:
        self.sim.reset()
        self.arm = self.sim.arm("left")

    # ------------------------------------------------------------- readings
    def particles(self) -> np.ndarray:
        return self.sim.state.particle_q.numpy().astype(np.float64)

    def tool(self) -> np.ndarray:
        return np.asarray(self.arm.ee_pose(self.sim.state)[0], dtype=np.float64)

    def drive(self, path, every, check) -> None:
        """Walk the hand along ``path`` as the page walks it to the gizmo, calling ``check`` every few steps."""
        from robosimstudio.solvers import OscSetpoint

        sim, arm = self.sim, self.arm
        _, quat0 = arm.ee_pose(sim.state)
        law = OscSetpoint(control_dt=1.0 / sim.request.physics.control_hz)
        law.reset(*arm.ee_pose(sim.state))
        for k, cursor in enumerate(path):
            position, quat = arm.ee_pose(sim.state)
            arm.set_target(*law.advance(position, quat, cursor, quat0))
            arm.solve_into(sim.state.joint_q, sim.control.joint_target_q)
            sim.run(1)
            if k % every == 0 or k == len(path) - 1:
                sim.sync()
                check(cursor)

    # ---------------------------------------------------------------- tests
    def test_the_edge_starts_in_the_hand_and_the_rest_lies_on_the_bench(self) -> None:
        from robosimstudio.domains.cloth import EntityHeldCloth, HeldClothSpec

        self.assertIsInstance(self.towel.spec, HeldClothSpec, "rss.sheet(held_by=...) is a held sheet")
        self.assertIsInstance(self.entity, EntityHeldCloth)
        self.assertEqual(28, len(self.held), "7 columns along the edge, 2 rows into it, on both faces")
        mass = self.sim.model.particle_mass.numpy()
        self.assertTrue(np.all(mass[self.held] == 0.0), "a held particle is kinematic")
        self.assertTrue(np.all(mass[np.setdiff1d(self.ids, self.held)] > 0.0), "and only a held one")

        q, tool = self.particles(), self.tool()
        patch = q[self.held].mean(axis=0)
        self.assertLess(float(np.linalg.norm(patch[:2] - tool[:2])), 0.01, "the patch is under the tool centre")
        self.assertTrue(0.01 < tool[2] - patch[2] < 0.04, "and just below the fingertips")
        bottom = q[self.ids[: len(self.ids) // 2], 2] - self.surface  # a slab's first half is its bottom face
        self.assertAlmostEqual(0.00125, float(np.median(bottom)), delta=0.001, msg="the towel lies on the bench")
        self.assertGreater(_lowest(self.sim, self.entity.freed) - self.surface, 0.005, "fingertips clear of the towel")

    def test_the_page_opens_looking_at_the_towel(self) -> None:
        """viser's default view is from outside the room the studio draws, so the page sets its own."""
        from types import SimpleNamespace

        camera = SimpleNamespace()
        self.kwargs["page"](SimpleNamespace(server=SimpleNamespace(initial_camera=camera)))
        middle = self.particles()[self.ids].mean(axis=0)
        np.testing.assert_allclose(camera.look_at, middle, atol=0.01)
        eye = np.asarray(camera.position) - middle
        self.assertGreater(eye[0], 0.3, "from the bench's open side, across from the arms")
        self.assertGreater(eye[2], 0.2, "and from above")
        self.assertEqual((0.0, 0.0, 1.0), tuple(camera.up))

    def test_only_what_hangs_from_the_wrist_stops_touching_the_cloth(self) -> None:
        import newton

        model = self.sim.model
        bit = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        flags, bodies = model.shape_flags.numpy(), model.shape_body.numpy()
        labels = [str(label).rsplit("/", 1)[-1] for label in model.body_label]
        freed = {labels[int(bodies[s])] for s in self.entity.freed}
        self.assertEqual({"left_link_left_finger", "left_link_right_finger"}, freed)
        self.assertTrue(all(not flags[s] & bit for s in self.entity.freed))
        for keeps in ("left_link_6", "right_link_left_finger", "right_link_right_finger"):
            shapes = [s for s in range(model.shape_count) if labels[int(bodies[s])] == keeps and bodies[s] >= 0]
            with self.subTest(body=keeps):
                self.assertTrue(any(flags[s] & bit for s in shapes), f"{keeps} still touches the cloth")

    def test_the_hold_survives_a_reset(self) -> None:
        """The page resets when it opens and on its Reset button; a hook added after the build would be gone."""
        start = self.tool()
        rest = self.particles()[self.held]
        lift = [start + np.array([0.0, 0.0, 0.05 * min(1.0, k / 40.0)]) for k in range(60)]
        self.drive(lift, every=59, check=lambda cursor: None)
        self.assertGreater(float(self.particles()[self.held, 2].mean() - rest[:, 2].mean()), 0.03, "it lifted")
        self.sim.reset()
        np.testing.assert_allclose(self.particles()[self.held], rest, atol=1e-5)
        self.drive(lift, every=59, check=lambda cursor: None)
        self.assertGreater(float(self.particles()[self.held, 2].mean() - rest[:, 2].mean()), 0.03, "and again")

    def test_carrying_the_edge_over_folds_the_towel(self) -> None:
        sim, entity, held = self.sim, self.entity, self.held
        hz = sim.request.physics.control_hz
        start = self.tool()
        q0 = self.particles()
        near = self.ids[q0[self.ids, 1] < q0[self.ids, 1].min() + 0.012]  # the edge across from the hand
        span0 = float(np.ptp(q0[self.ids, 1]))
        offset0 = q0[held] - start
        shape0 = np.linalg.norm(q0[held][:, None] - q0[held][None], axis=2)
        # Over in a half circle about the crease, landing a centimetre higher than it started.
        radius = start[1] - CREASE_Y
        arc = [np.pi * u * u * (3.0 - 2.0 * u) for u in np.linspace(0.0, 1.0, int(3.0 * hz))]
        path = [start + np.array([0.0, -radius * (1 - np.cos(a)), radius * np.sin(a) + 0.01 * a / np.pi]) for a in arc]
        path += [path[-1]] * int(1.5 * hz)

        worst = {"pin": 0.0, "drift": 0.0, "lift": 0.0, "slide": 0.0, "fingers": np.inf}

        def check(cursor) -> None:
            q = self.particles()
            shape = np.linalg.norm(q[held][:, None] - q[held][None], axis=2)
            worst["pin"] = max(worst["pin"], float(np.abs(shape - shape0).max()))
            worst["drift"] = max(worst["drift"], float(np.abs(q[held] - self.tool() - offset0).max()))
            worst["lift"] = max(worst["lift"], float(q[near, 2].max() - self.surface))
            worst["slide"] = max(worst["slide"], float(np.linalg.norm(q[near, :2] - q0[near, :2], axis=1).max()))
            worst["fingers"] = min(worst["fingers"], _lowest(sim, entity.freed) - self.surface)
            self.assertTrue(np.isfinite(q[self.ids]).all(), "the towel went non-finite")

        self.drive(path, every=5, check=check)
        self.assertTrue(sim.state_is_finite())
        self.assertLess(worst["pin"], 1.0e-4, "the patch in the hand moved as one piece")
        self.assertLess(worst["drift"], 0.005, "the held edge followed the hand")
        self.assertLess(worst["lift"], 0.02, "the half on the bench stayed on it")
        self.assertLess(worst["slide"], 0.025, "and was not dragged along")
        self.assertGreater(worst["fingers"], 0.0, "the fingers went into the bench")

        q = self.particles()
        edge = q[held].mean(axis=0)
        self.assertLess(edge[1], CREASE_Y - 0.12, "the held edge was carried over the crease")
        self.assertLess(edge[2] - self.surface, 0.025, "and laid down")
        self.assertLess(float(np.ptp(q[self.ids, 1])), 0.65 * span0, "the towel is folded across its middle")


class TestTheExampleHasTheShapeOfOne(unittest.TestCase):
    """The rules ``tests/test_examples.py`` holds the numbered examples to, at a demo's length."""

    MAX_LINES = 150

    def setUp(self) -> None:
        self.source = (EXAMPLES / REL).read_text()
        self.tree = ast.parse(self.source)

    def test_it_is_a_script_with_a_main_that_ends_on_the_studio(self) -> None:
        self.assertLessEqual(len(self.source.splitlines()), self.MAX_LINES)
        self.assertNotIn("argparse", self.source)
        for withheld in ("examples.tasks", "taskbuild", "solutions", "rgbs", "vabar", "routine"):
            self.assertNotIn(withheld, self.source, "a demo stands on the public library alone")
        main = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        self.assertEqual([], main.args.args)
        self.assertEqual("rss.view", ast.unparse(main.body[-1].value.func))
        self.assertEqual("__name__ == '__main__'", ast.unparse(self.tree.body[-1].test))

    def test_its_docstring_is_the_page_it_becomes(self) -> None:
        blocks = [block.strip() for block in (ast.get_docstring(self.tree) or "").split("\n\n")]
        self.assertTrue(2 <= len(blocks[0].split()) <= 5 and not blocks[0].endswith("."), blocks[0])
        self.assertNotIn("\n", blocks[1])
        self.assertTrue(blocks[1].endswith("."))
        at = blocks.index("**Key features:**")
        self.assertLessEqual(at - 2, 1, "at most one paragraph")
        bullets = blocks[at + 1].splitlines()
        self.assertTrue(3 <= len(bullets) <= 5 and all(b.startswith("* ") for b in bullets))
        for bullet in bullets:
            self.assertRegex(bullet, r":(func|class|meth|attr|data|mod):`|``")


if __name__ == "__main__":
    unittest.main()
