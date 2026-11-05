# SPDX-License-Identifier: Apache-2.0
"""What the SuperDex bridge feeds back to a hand: the sponge's push, in the world frame; nothing from a rod.

SuperDex's ``get_contact_force_from_actor_world`` is a world vector for a
volumetric soft body: the Panda closing on the sponge (``02_soft/00_sponge``)
reads each pad's push along that pad's own normal, into the foam. For a rod it
is not: the knot example's rope, held by both YAM hands at rest, reports about
40 N along +y on every pad, the same vector on both hands, on the pad at -x and
the pad at +x, whichever way the hands move, when a squeeze acts along the pads'
normals and cancels. Fed to the hands, that shoved a pinch sideways by 80 N
until the rope slipped out.

The bridge therefore harvests a soft body's force as the engine states it, and
drops a rod's with one warning. These tests pin both: the sponge's pads push
into the foam, and the rope's pads get no wrench.
"""

from __future__ import annotations

import importlib.util
import logging
import unittest

import numpy as np

TIER = "medium"
"""Two builds on a GPU: the sponge example with its hand closing, and the knot example with one hand closing: ~60 s."""

HAS_SUPERDEX = importlib.util.find_spec("superdex") is not None


def _harvest(sim):
    """Patch the host bridge to keep the last wrench it would have applied; returns the holder."""
    host = next(iter(sim.pipeline.program.hosts.values()))
    holder: dict = {}
    host.bridge.apply_wrenches = lambda w: holder.__setitem__("w", np.asarray(w, dtype=np.float64).copy())
    return host.bridge, holder


@unittest.skipUnless(HAS_SUPERDEX, "needs the [superdex] extra")
class TestTheSpongesPushIsInTheWorldFrame(unittest.TestCase):
    """``02_soft/00_sponge``: the hand closes on the foam; each pad's harvested force points into the foam."""

    @classmethod
    def setUpClass(cls) -> None:
        from tests.fixtures.examples import _run
        from tests.fixtures.grasp import go, open_session, tick

        cls.sim, kwargs = _run("02_soft/00_sponge.py")
        bridge, cls.holder = _harvest(cls.sim)
        session = open_session(cls.sim, 0, kwargs.get("page"))
        (arm,) = session.arms
        tick(session, 30)
        ids = np.asarray(cls.sim["sponge"].entity.particles, dtype=np.int64)
        points = lambda: cls.sim.state.particle_q.numpy()[ids].astype(np.float64)  # noqa: E731
        centre = points().mean(axis=0)
        # The example's gesture: fingertips a centimetre over the bench, then Close.
        go(session, arm, np.array([centre[0], centre[1], 0.75 + 0.012]), speed=0.1)
        session.arms[arm].grip_request = 1.0
        tick(session, 40)
        cls.body_q = cls.sim.state.body_q.numpy().astype(np.float64)
        cls.pads = [int(m.body) for m in bridge.mirrors if m.two_way and m.moving]
        # Each loaded pad's contact normal, pointed from the foam toward the pad.
        cls.outward = {}
        for b in cls.pads:
            c = bridge.contact_points("sponge", against=b)
            if len(c["pos"]) < 5:
                continue
            n = np.asarray(c["normal"], dtype=np.float64).mean(axis=0)
            n /= np.linalg.norm(n)
            toward_pad = cls.body_q[b, :3] - np.asarray(c["pos"], dtype=np.float64).mean(axis=0)
            cls.outward[b] = n if np.dot(n, toward_pad) > 0 else -n
        session.teardown()

    def test_each_pad_in_contact_is_pushed_out_along_its_normal(self) -> None:
        w = self.holder["w"]
        loaded = [b for b in self.outward if np.linalg.norm(w[b, :3]) > 5.0]
        self.assertGreaterEqual(len(loaded), 2, "fewer than two pads carry a force")
        for b in loaded:
            force = w[b, :3]
            # The foam pushes each pad out along that pad's own contact normal: a world vector, as the engine states it.
            self.assertGreater(float(np.dot(force / np.linalg.norm(force), self.outward[b])), 0.8, f"pad {b}: {force}")


@unittest.skipUnless(HAS_SUPERDEX, "needs the [superdex] extra")
class TestARodsPushIsNotHarvested(unittest.TestCase):
    """``03_materials/03_knot`` with feedback on: a hand closed on the rope gets no wrench, and one warning says why."""

    @classmethod
    def setUpClass(cls) -> None:
        from unittest import mock

        import robosimstudio as rss
        from tests.fixtures.examples import _run
        from tests.fixtures.grasp import open_session, tick

        real = rss.backend

        def backend(name, **kw):
            return real(name, **({**kw, "feedback": "auto"} if name == "superdex" else kw))

        with mock.patch.object(rss, "backend", backend):
            cls.sim, kwargs = _run("03_materials/03_knot.py")
        bridge, cls.holder = _harvest(cls.sim)
        cls.said: list[str] = []

        class _Keep(logging.Handler):
            def emit(self, record):
                cls.said.append(record.getMessage())

        keep = _Keep(level=logging.WARNING)
        logger = logging.getLogger("robosimstudio.engines.superdex.bridge")
        logger.addHandler(keep)  # from the first tick: the rod is declined on the first harvest, contact or not
        session = open_session(cls.sim, 0, kwargs.get("page"))
        arms = session.arms
        tick(session, 60)
        rest = np.asarray(cls.sim["rope"].particle_q, dtype=np.float64)
        # The left hand to its tail by a joint-space move: the panel's incremental solve stops 25 mm short there.
        ik = arms["left"].ik
        q = ik.solve_q_full(
            cls.sim.state.joint_q.numpy().copy(), rest[-1] - np.array([0.0, 0.04, -0.010]), ik.home_pose[1]
        )
        control = cls.sim.pipeline.control
        t = control.joint_target_q.numpy().copy()
        coords = list(ik.arm_coords)
        t[coords] = np.asarray(q)[coords]
        arms["left"].suspend()
        control.joint_target_q.assign(t)
        tick(session, 150)
        arms["left"].resume()
        try:
            for k in range(1, 49):
                arms["left"].grip_request = 0.9 * k / 48
                tick(session, 1)
            tick(session, 30)
        finally:
            logger.removeHandler(keep)
        cls.bridge = bridge
        session.teardown()

    def test_the_pads_touch_the_rope(self) -> None:
        for pad in (6, 7):
            self.assertGreater(
                len(self.bridge.contact_points("rope", against=pad)["pos"]), 10, f"pad {pad} is not on the rope"
            )

    def test_the_hand_gets_no_wrench_from_the_rope(self) -> None:
        w = self.holder["w"]
        self.assertLess(float(np.abs(w[[6, 7], :3]).max()), 1e-6, f"a rod's force reached the pads: {w[[6, 7], :3]}")

    def test_it_says_so_once(self) -> None:
        said = [line for line in self.said if "not harvested" in line]
        self.assertEqual(len(said), 1, self.said)


if __name__ == "__main__":
    unittest.main()
