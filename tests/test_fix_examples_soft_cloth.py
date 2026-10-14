# SPDX-License-Identifier: Apache-2.0
"""Regressions for the soft and cloth examples on the GPU: the VBD fallback and the shared settle."""

from __future__ import annotations

import unittest

import numpy as np

TIER = "medium"


class TestASoftBodyOnVbdFalls(unittest.TestCase):
    """EntitySoft cleared ACTIVE whatever the backend, so a body routed to VBD never moved."""

    def test_it_falls(self):
        import newton

        import robosimstudio as rss

        physics = rss.replace(rss.GENERIC, backends=tuple(e for e in rss.GENERIC.backends if e[0] != "soft"))
        scene = rss.Scene(cell="void", physics=physics, name="vbd_soft")
        # With a robot the arrangement couples, and the solvers copy the flags they are built with.
        scene.add(rss.robot("yam_bimanual"))
        body = scene.add(
            rss.SoftSpec(name="block", size=(0.04, 0.04, 0.02), cells=(4, 4, 2), pose=(0.0, 0.0, 0.3), settle_frames=0)
        )
        sim = scene.build(control_mode="joint_position")
        flags = sim.pipeline.model.particle_flags.numpy()[np.asarray(body.particles)]
        self.assertTrue(np.all(flags & int(newton.ParticleFlags.ACTIVE)))
        before = float(sim.pipeline.state_0.particle_q.numpy()[body.particles, 2].mean())
        for _ in range(30):
            sim.pipeline.step()
        after = float(sim.pipeline.state_0.particle_q.numpy()[body.particles, 2].mean())
        self.assertLess(after, before - 0.01)


class TestSheetsShareOneSettle(unittest.TestCase):
    """Each sheet settled the whole scene and snapshotted, so earlier sheets' baselines were stale."""

    def test_every_sheet_baselines_the_state_reset_restores(self):
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_0728", name="two_sheets")
        towel = rss.materials.TERRY_TOWELLING
        scene.add(rss.sheet(name="a", size=(0.2, 0.2), cells=(8, 8), at=(-0.3, 0.0, 0.5), material=towel))
        scene.add(rss.sheet(name="b", size=(0.2, 0.2), cells=(8, 8), at=(0.1, 0.0, 0.5), material=towel))
        sim = scene.build()
        things = {e.name: e for e in sim.scene.entities}
        a, b = things["a"], things["b"]
        self.assertEqual(len(sim.pipeline.particle_settles["runs"]), 1, "two like sheets settle once")
        np.testing.assert_array_equal(a.rest["particle_q"], b.rest["particle_q"])
        sim.reset()
        state = sim.pipeline.state_0.particle_q.numpy()
        for sheet in (a, b):
            baseline = np.asarray(sheet.probe.baseline).reshape(-1, 3)
            np.testing.assert_allclose(baseline, state[: len(baseline)], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
