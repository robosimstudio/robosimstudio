# SPDX-License-Identifier: Apache-2.0
"""A material field that never reaches the engine is said out loud at compose.

Measured before this existed: every shipped fabric states ``tri_ke``,
``edge_ke`` and the rest of its VBD stiffness, none states ``shell_young``, and
only six of ``ClothMaterialSpec``'s eighteen fields ever reach a foreign engine
-- so packing tape and bath towelling arrived at SuperDex and Genesis with the
same ``E = 1e5``, and nothing said so. Genesis is worse: of the six that cross,
it reads five, and ``friction``/``particle_radius``/``self_contact`` fall on the
floor between the entity and the engine with no message at all.

Two warnings, one per side of the seam, both naming the entity, the engine and
the keys:

* **engine side** -- a backend declares ``consumes=``; a ``HostActor`` param
  outside that set is a parameter the engine will never look at.
* **entity side** -- a field the spec's material states away from its default
  that does not appear in ``host_actor().params`` at all.

Warnings rather than errors: both shipped engines would refuse every cloth scene
today under an error, and the point is that a *person* learns which knob is
dead, not that a working scene stops. Stub engine, no GPU beyond a compile.
"""

from __future__ import annotations

import unittest
import warnings

TIER = "medium"


class TestALostMaterialFieldIsLoud(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio.specs.physics import BACKENDS, register_backend
        from tests.test_soft_domain import _StubHostSolver

        cls.alias = "probe_loud_engine"
        if cls.alias not in BACKENDS:
            register_backend(
                cls.alias,
                cls=_StubHostSolver,
                supports=("particles",),
                host_engine=True,
                consumes=("thickness", "areal_density", "young", "poisson"),
            )

    def _build_sheet(self, **material):
        import robosimstudio as rss
        from robosimstudio.specs.physics import GENERIC, backend

        physics = rss.replace(
            GENERIC,
            backends=(("articulated", backend("mujoco")), ("cloth", backend(self.alias)), ("*", backend("vbd"))),
        )
        scene = rss.Scene(cell="void", name="loud", physics=physics)
        scene.add(
            rss.sheet(
                name="towel",
                size=(0.2, 0.2),
                cells=(6, 6),
                at=(0.0, 0.0, 0.1),
                material=rss.ClothMaterialSpec(**material),
            )
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sim = scene.build()
        sim.close()
        return [str(w.message) for w in caught if "towel" in str(w.message)]

    def test_a_param_the_engine_never_reads_is_named(self) -> None:
        """Engine side: ``friction`` crosses in the actor and ``probe_loud_engine`` does not consume it."""
        messages = self._build_sheet()
        engine_side = [m for m in messages if self.alias in m and "friction" in m]
        self.assertTrue(engine_side, f"no engine-side warning named friction; got {messages}")
        self.assertIn("particle_radius", engine_side[0])

    def test_a_stated_field_that_never_crosses_is_named(self) -> None:
        """Entity side: a fabric that states ``edge_ke`` and never hands it to any engine."""
        messages = self._build_sheet(edge_ke=0.05, tri_ke=2.0e4)
        entity_side = [m for m in messages if "edge_ke" in m and "tri_ke" in m]
        self.assertTrue(entity_side, f"no entity-side warning named the stated fields; got {messages}")
        self.assertIn("shell_young", entity_side[0], "the message should say which knob does reach the engine")

    def test_a_field_left_at_its_default_is_not_nagged_about(self) -> None:
        """Only what was *stated* is lost; defaults were never a decision."""
        messages = self._build_sheet()
        self.assertFalse([m for m in messages if "tri_ke" in m], "tri_ke at its default is not a lost statement")

    def test_an_engine_that_declares_nothing_is_not_checked(self) -> None:
        from robosimstudio.engines.geometry import actor_consumes, engine_consumes
        from robosimstudio.specs.physics import get_backend

        get_backend("superdex")  # loads the engine bundle, which registers its builders
        self.assertNotEqual(engine_consumes("superdex"), (), "superdex declares, per builder")
        self.assertIn("friction", actor_consumes("superdex", "tet"))
        self.assertNotIn("friction", actor_consumes("superdex", "shell"), "the shell builder never reads it")
        self.assertEqual(get_backend("vbd").consumes, (), "a Newton solver reads the model, not the actor")

    def test_a_kind_s_own_builder_sharpens_the_engine_s_declaration(self) -> None:
        """The stub engine declares four keys; its shell builder declares two, and wins."""
        from robosimstudio.engines.geometry import actor_consumes, register_actor_builder

        if actor_consumes(self.alias, "shell") is None:
            register_actor_builder(self.alias, "shell", lambda *_a, **_k: None, consumes=("thickness", "young"))
        messages = self._build_sheet()
        engine_side = [m for m in messages if self.alias in m and "shell builder" in m]
        self.assertTrue(engine_side, f"no per-kind warning; got {messages}")
        self.assertIn("areal_density", engine_side[0], "stated by the sheet, read engine-wide, not by this builder")


if __name__ == "__main__":
    unittest.main()
