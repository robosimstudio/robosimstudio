# SPDX-License-Identifier: Apache-2.0
"""``examples/09_demos/01_route_the_cable.py``: the cable rests on its skin, rides the hand, and routes.

The example ends by opening the studio, so it is run the way
:mod:`tests.test_examples` runs every example: loaded by file path, with
:func:`robosimstudio.view` replaced by a stub that records what it was handed.
Its ``page`` hook is then given a stand-in studio -- a per-step list, a camera to
aim, and no scene to draw in, since the studio draws the cable itself -- and the
hand is driven the way the studio's arm panel drives it from the wrist gizmo:
an :class:`~robosimstudio.solvers.OscSetpoint` toward a target, IK into the
joint targets, the page's per-step hook, one control step.

What is asserted is physical. The padded mirrors put the cable's axis one
radius over the bench (0.93 mm unpadded, half the cable inside the wood); the
held end keeps up with the tool centre; the cable
never sinks into the bench or into a prong; and the route the docstring
describes -- lift, carry past the far clip, lower onto the line -- lays it into
all three clips. Each test starts from a reset, which is also how the hook is
held to its job: a reset lets go of everything, and the hand takes the end again
on the first step after it.

Bands, not bits: the engine documents no determinism guarantee.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import re
import sys
import types
import unittest
from pathlib import Path

import numpy as np

TIER = "medium"
"""A GPU, the [superdex] extra, one scene built once."""

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "09_demos" / "01_route_the_cable.py"

LAG = 0.004
"""How far the held end may trail the tool centre [m]. The engine carries the hold on the hand
through every substep, so what is left at the arm's top speed (~0.3 m/s under the setpoint's
rate limit) is the pin's own compliance: 2.1 mm. Re-aimed once per control step it was 5.2 mm."""


def _load():
    name = "_rss_demo_" + re.sub(r"\W", "_", EXAMPLE.stem)
    spec = importlib.util.spec_from_file_location(name, EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _has_rope() -> bool:
    """Whether this tree carries the rope pack, which ``rss.RodSpec`` comes from."""
    import robosimstudio as rss

    return hasattr(rss, "RodSpec")


@unittest.skipUnless(importlib.util.find_spec("superdex") is not None, "the [superdex] extra is not installed")
@unittest.skipUnless(_has_rope(), "the rope (rss.RodSpec) is not in this tree")
class TestRouteTheCable(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        cls.example = _load()
        opened: list = []
        real = rss.view
        rss.view = lambda sim, **kwargs: opened.append((sim, kwargs))
        try:
            cls.example.main()
        finally:
            rss.view = real
        ((cls.sim, kwargs),) = opened
        cls.steps: list = []
        cls.studio = types.SimpleNamespace(
            each_step=cls.steps.append,
            server=types.SimpleNamespace(initial_camera=types.SimpleNamespace()),
        )
        kwargs["page"](cls.studio)
        cls.cable = cls.sim["cable"]
        cls.ik = cls.sim.arm()
        cls.origin = np.array([0.6, 0.0, float(cls.sim.handles.surface_z)])

    @classmethod
    def tearDownClass(cls) -> None:
        with contextlib.suppress(Exception):
            cls.sim.close()

    # ------------------------------------------------------------------ driving
    def setUp(self) -> None:
        from robosimstudio.solvers import OscSetpoint

        self.sim.reset()
        self.law = OscSetpoint(control_dt=1.0 / float(self.sim.request.physics.control_hz))
        self.start, self.quat = (np.asarray(v, dtype=np.float64) for v in self.ik.ee_pose(self.sim.state))
        self.law.reset(self.start, self.quat)

    def _step(self, target=None) -> None:
        """One control step as the studio takes it: the arm toward ``target``, the page's hook, the physics."""
        sim = self.sim
        position, quat = self.ik.ee_pose(sim.state)
        goal = self.start if target is None else self.origin + np.asarray(target, dtype=np.float64)
        self.ik.set_target(*self.law.advance(position, quat, goal, self.quat))
        self.ik.solve_into(sim.state.joint_q, sim.control.joint_target_q)
        for fn in self.steps:
            fn()
        sim.step_teleop()

    def _drive(self, keys) -> dict[str, float]:
        """Ease the tool centre through ``keys`` ((x, y, z) on the bench, seconds); the worst of each reading."""
        worst = {"lag": 0.0, "low": 1.0, "post": 1.0}
        here = self._local(self.ik.ee_pose(self.sim.state)[0])
        hz = float(self.sim.request.physics.control_hz)
        for key, seconds in keys:
            there = np.asarray(key, dtype=np.float64)
            for u in np.linspace(0.0, 1.0, max(2, int(seconds * hz))):
                self._step(here + (there - here) * (u * u * (3.0 - 2.0 * u)))
                nodes = self._nodes()
                self.assertTrue(
                    np.isfinite(nodes).all(), f"the cable stopped being finite at step {self.sim.step_count}"
                )
                tcp = self._local(self.ik.ee_pose(self.sim.state)[0])
                worst["lag"] = max(worst["lag"], float(np.linalg.norm(nodes[-1] - tcp)))
                worst["low"] = min(worst["low"], float(nodes[:, 2].min()))
                worst["post"] = min(worst["post"], self._closest_post(nodes))
            here = there
        return worst

    # ------------------------------------------------------------------ reading
    def _local(self, point) -> np.ndarray:
        return np.asarray(point, dtype=np.float64) - self.origin

    def _nodes(self) -> np.ndarray:
        return self._local(self.cable.particle_q)

    def _closest_post(self, nodes: np.ndarray) -> float:
        """The nearest a point on the cable's axis below the posts' tops comes to a post's axis [m]."""
        ex = self.example
        posts = np.array([[ex.LINE_X + s * (ex.CHANNEL / 2 + ex.PRONG_R), y] for y in ex.CLIP_Y for s in (1.0, -1.0)])
        t = np.linspace(0.0, 1.0, 6)[None, :, None]
        points = (nodes[:-1, None, :] + t * np.diff(nodes, axis=0)[:, None, :]).reshape(-1, 3)
        low = points[points[:, 2] < ex.PRONG_R + 2 * ex.POST_HALF]
        if not len(low):
            return 1.0
        return float(np.linalg.norm(low[:, None, :2] - posts[None], axis=2).min())

    def _routed(self, nodes: np.ndarray) -> list[bool]:
        """Per clip: does the cable cross its channel, between the prongs and below 12 mm."""
        ex = self.example
        out = []
        for y in ex.CLIP_Y:
            a = nodes[:, 1] - y
            cross = np.flatnonzero(np.sign(a[:-1]) != np.sign(a[1:]))
            points = [nodes[i] + a[i] / (a[i] - a[i + 1]) * (nodes[i + 1] - nodes[i]) for i in cross]
            out.append(any(abs(p[0] - ex.LINE_X) <= ex.CHANNEL / 2 + 0.002 and p[2] <= 0.012 for p in points))
        return out

    # -------------------------------------------------------------------- tests
    def test_the_end_starts_in_the_hand_and_the_cable_on_its_skin(self) -> None:
        for _ in range(30):
            self._step()
        nodes = self._nodes()
        tcp = self._local(self.ik.ee_pose(self.sim.state)[0])
        self.assertLess(float(np.linalg.norm(nodes[-1] - tcp)), 0.003, "the hand is not holding the end")
        # The nodes still lying along the bench out of the socket, clear of the clamp and of the rise
        # into the hand (the stress-free cable straightens and lifts the bench run's last few).
        bench = nodes[2:10, 2]
        radius = self.example.RADIUS
        self.assertLess(float(np.abs(bench - radius).max()), 0.0007, f"the axis rests at {bench * 1e3} mm, not 2.5")
        # Drawn by the studio, as every centerline is: a tube of the cable's radius round its nodes.
        from robosimstudio_studio.garments import garment_visuals

        (drawn,) = (g for g in garment_visuals(self.sim.scene) if g.name == "cable")
        self.assertTrue(drawn.is_tube, "the studio does not draw the cable")
        rings = drawn.surface(self.sim.state.particle_q.numpy())[0][:-2].reshape(46, -1, 3)
        np.testing.assert_allclose(rings.mean(axis=1), self.cable.particle_q, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(rings[-1] - self.cable.particle_q[-1], axis=1), radius, rtol=1e-4)

    def test_the_held_end_follows_the_hand(self) -> None:
        """Lift and carry along the route: ~100 control steps."""
        x = self.example.LINE_X
        worst = self._drive((((x, self._local(self.start)[1], 0.16), 0.8), ((x, 0.05, 0.16), 0.9)))
        self.assertGreater(self.sim.step_count, 90)
        self.assertLess(worst["lag"], LAG, f"the end trailed the tool centre by {worst['lag'] * 1e3:.1f} mm")
        self.assertGreater(worst["low"], 0.0015, f"the cable's axis went {worst['low'] * 1e3:.2f} mm over the bench")
        self.assertGreater(worst["post"], self.example.PRONG_R, "the cable's axis went inside a prong")
        self.assertGreater(float(self._nodes()[-1, 2]), 0.12, "the end did not come up with the hand")

    def test_the_route_lays_it_into_all_three_clips(self) -> None:
        """The docstring's route at twice its speed: up, along past the far clip, straight down onto the line."""
        x, y = self.example.LINE_X, self._local(self.start)[1]
        keys = (
            ((x, y, 0.20), 0.7),
            ((x, 0.19, 0.20), 0.7),
            ((x, 0.19, 0.20), 0.25),
            ((x, 0.225, 0.07), 1.2),
            ((x, 0.235, 0.016), 0.6),
            ((x, 0.235, 0.016), 1.0),
        )
        worst = self._drive(keys)
        self.assertEqual([True, True, True], self._routed(self._nodes()), "not every clip holds the cable")
        self.assertLess(worst["lag"], LAG)
        self.assertGreater(worst["low"], 0.0015)
        # Pressed into a channel, the cable touches a prong (5.5 mm between the axes) and may
        # overlap it by about a millimetre; it never reaches the prong's axis side of its surface.
        self.assertGreater(worst["post"], self.example.PRONG_R + 0.0005, "the cable went into a prong")


class TestTheShape(unittest.TestCase):
    """The rules of a numbered example, for the one script ``tests/test_examples.py`` does not list."""

    def setUp(self) -> None:
        self.source = EXAMPLE.read_text()
        self.tree = ast.parse(self.source)

    def test_it_is_one_short_script_over_the_public_api(self) -> None:
        self.assertLessEqual(len(self.source.splitlines()), 160)
        imports = {a.name for n in self.tree.body if isinstance(n, ast.Import) for a in n.names}
        imports |= {n.module for n in self.tree.body if isinstance(n, ast.ImportFrom)}
        self.assertEqual({"math", "numpy", "robosimstudio"}, imports, "only the flat rss names, and numpy")
        self.assertNotIn("argparse", self.source)

    def test_it_ends_by_opening_the_studio(self) -> None:
        main = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        self.assertEqual([], main.args.args, "main() takes no arguments")
        self.assertEqual("rss.view", ast.unparse(main.body[-1].value.func))
        self.assertEqual("main()", ast.unparse(self.tree.body[-1].body[0]).strip())

    def test_its_docstring_is_the_page_it_becomes(self) -> None:
        blocks = [block.strip() for block in (ast.get_docstring(self.tree) or "").split("\n\n")]
        self.assertEqual("Route the cable", blocks[0])
        self.assertNotIn("\n", blocks[1], "the description is one sentence on one line")
        self.assertEqual(["**Key features:**"], blocks[3:4], "one paragraph, then the features")
        bullets = blocks[4].splitlines()
        self.assertTrue(3 <= len(bullets) <= 5 and all(b.startswith("* ") for b in bullets))
        self.assertIn("[superdex]", blocks[2], "an example that needs an extra says so")


if __name__ == "__main__":
    unittest.main()
