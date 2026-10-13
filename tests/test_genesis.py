# SPDX-License-Identifier: Apache-2.0
"""The Genesis co-simulation, with libuipc actually in the loop.

The half of the story ``test_genesis_absent.py`` cannot tell: a cloth built into
the Newton model, simulated by Genesis's IPC coupler, read back through the
ordinary particle arrays, and pressing on a Newton body hard enough to measure.

Numbers are asserted as bands, not bits --- libuipc's Newton solve is iterative
and this stack's sims are nondeterministic besides. The one number asserted
tightly is the resting weight, because that one is not a simulation outcome: a
cloth at rest on a plate weighs what it weighs, and if it does not, the model
Genesis is integrating is not the model the scene declared.
"""

from __future__ import annotations

import importlib.util
import unittest
from dataclasses import replace

import numpy as np

TIER = "medium"
"""A GPU, the [genesis] extra, and a scene or two.

The extra is deliberately **not** installed in the main environment --- it moves
twenty packages there for an engine most runs never build. So this module skips
by default and is run against a sibling venv instead, with the checkout on the
path; ``pyproject.toml``'s ``[genesis]`` comment carries the exact commands.
``tests/test_genesis_absent.py`` is what covers the seam without any of that,
and it is the one that gates every commit.
"""

BENCH_TOP = 0.75
"""bench_075's work surface height [m], the number its name states."""


def _importable(module: str) -> bool:
    """Whether ``module`` can be found; False, not a raise, when a parent package is missing."""
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _physics():
    from robosimstudio.specs.physics import GENERIC, backend

    return replace(
        GENERIC,
        backends=(("cloth", backend("genesis", substeps=2)), ("*", backend("kinematic"))),
    )


def _seed_targets(sim) -> None:
    """Joint targets are not initialised from the spawn pose.

    Stepping a fresh scene from all-zero targets teleports every free body to
    the origin --- which, with a two-way mirror, then drags the cloth with it.

    A deformable-only scene has no joints and therefore no target array; that is
    not a case to seed, so it is skipped rather than guarded against downstream.
    """
    targets = getattr(sim.pipeline.control, "joint_target_q", None)
    if targets is not None:
        targets.assign(sim.pipeline.state_0.joint_q.numpy().copy())


@unittest.skipUnless(_importable("genesis"), "the [genesis] extra is not installed")
class TestClothOnGenesis(unittest.TestCase):
    """A sheet routed to the IPC coupler by one arrangement line."""

    @classmethod
    def setUpClass(cls):
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075", physics=_physics())
        cls.towel = scene.add(
            rss.sheet(name="towel", size=(0.2, 0.2), cells=(10, 10), at=(-0.1, -0.1, 0.04), settle_frames=0)
        )
        cls.sim = scene.build()
        _seed_targets(cls.sim)

    def test_it_is_a_shell_actor_and_drapes_onto_the_bench(self):
        program = self.sim.scene.program
        record = next(i for i in program.manifest.islands if i.host_engine)
        self.assertEqual(record.solver_class, "genesis")
        self.assertEqual([a["kind"] for a in record.host_recipe["actors"]], ["shell"])
        self.sim.run(60).sync()
        q = self.towel.particle_q
        self.assertTrue(self.sim.state_is_finite())
        self.assertAlmostEqual(float(q[:, 2].min()), BENCH_TOP, delta=0.008, msg="the sheet lies on the bench")
        self.assertLess(float(q[:, 2].max()), BENCH_TOP + 0.03, msg="and is flat-ish, not floating")

    def test_the_ground_plane_did_not_stop_the_scene_building(self):
        """A Plane and any non-plane rigid cannot coexist in an IPC scene ---
        ``build`` dies in ``_init_accessors``. The adapter mirrors a Newton
        ground plane as a large flat box for exactly this reason, and this
        asserts the workaround is still needed *and* still working."""
        bridge = self.sim.scene.program.solver.solver("genesis").bridge
        kinds = sorted(m["geom"].kind for m in bridge.mirrors)
        self.assertIn("plane", kinds, "the recipe still mirrors the ground")
        self.assertTrue(self.sim.state_is_finite())


@unittest.skipUnless(_importable("genesis"), "the [genesis] extra is not installed")
class TestTheReactionIsTheClothsWeight(unittest.TestCase):
    """The two-way harvest, against a number that is not a simulation outcome.

    A cloth resting on a plate weighs what the Newton model says it weighs. The
    plate rides ``backend("kinematic")`` and feels nothing, so the harvested
    wrench is a measurement of Genesis rather than of a feedback loop.

    This is also the only end-to-end check of
    :data:`~robosimstudio.engines.genesis.bridge.CLOTH_MASS_FACTOR`. Genesis
    exposes no simulated mass on an FEM entity, so the bridge applies the 2x
    convention and cannot confirm it at build; here it is confirmed by weighing.
    Unapplied, this assertion reads 2.0 instead of 1.0.
    """

    @classmethod
    def setUpClass(cls):
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075", physics=_physics())
        cls.plate = scene.add(rss.box(name="plate", half=(0.14, 0.14, 0.005), at=(0.0, 0.0, 0.10)))
        cls.towel = scene.add(
            rss.sheet(name="towel", size=(0.2, 0.2), cells=(10, 10), at=(-0.1, -0.1, 0.16), settle_frames=0)
        )
        cls.sim = scene.build()
        _seed_targets(cls.sim)
        cls.bridge = cls.sim.scene.program.solver.solver("genesis").bridge
        cls.applied = {}
        inner = cls.bridge.apply_wrenches

        def spy(w):
            cls.applied["w"] = np.array(w, copy=True)
            return inner(w)

        cls.bridge.apply_wrenches = spy
        cls.sim.run(300).sync()

    def test_feedback_is_live_for_a_two_way_mirror(self):
        self.assertTrue(self.bridge.feedback_enabled)
        record = next(i for i in self.sim.scene.program.manifest.islands if i.host_engine)
        self.assertTrue(
            any(m["two_way"] for m in record.host_recipe["mirrors"]),
            "the plate's contact rule should make its mirror two-way",
        )

    def test_the_plate_feels_exactly_the_cloth(self):
        count = int(self.towel.particle_q.shape[0])
        weight = float(self.sim.scene.model.particle_mass.numpy()[:count].sum()) * 9.81
        measured = float(self.bridge._acc.ema[0, 2])
        self.assertTrue(self.sim.state_is_finite())
        # Downward: the cloth rests ON the plate. Genesis reports the force on
        # the link, so unlike SuperDex's per-actor-pair query there is no
        # negation anywhere in the path.
        self.assertLess(measured, 0.0, "the reaction on the plate points down")
        self.assertAlmostEqual(
            abs(measured) / weight,
            1.0,
            delta=0.05,
            msg=f"measured {measured:+.4f} N against a cloth weighing {weight:.4f} N; a ratio near "
            f"2.0 means CLOTH_MASS_FACTOR stopped being applied",
        )

    def test_the_shaping_does_not_clip_a_physical_load(self):
        """The Delta-v cap exists for the transient, not for a resting load ---
        a cap that engages at rest would mean the arm feels the cap and not the
        cloth."""
        raw = float(self.bridge._acc.ema[0, 2])
        applied = float(self.applied["w"][0, 2])
        self.assertAlmostEqual(applied, raw, delta=1e-3, msg="a resting load passes through unclamped")


if __name__ == "__main__":
    unittest.main()
