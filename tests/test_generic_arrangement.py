# SPDX-License-Identifier: Apache-2.0
"""The default arrangement can actually step every domain the library ships.

:data:`~robosimstudio.specs.physics.GENERIC` is what a scene gets when nobody has
staged anything. The composer already checks a backend against the state its
domain carries -- :func:`~robosimstudio.scene.physics.islands.build_islands`
raises when they disagree -- but it checks the arrangement *in front of it*, at
build time, for the entities that happen to be in that scene. That is too late
and too narrow for the default: a domain the library ships whose backend under
GENERIC cannot own its state is a scene nobody can build, and it should fail here
rather than in somebody's first five minutes.

Done statically, so it needs no GPU and no model: the same comparison
``build_islands`` makes, against the registry rather than against a graph.

The rod is the case that motivates the whole file. It owns bodies and joints,
which is exactly what MuJoCo supports, so a rod served by the dataclass's own
``"*"`` fallback passes the capability check and fails *later*, inside MuJoCo's
importer. ``GENERIC`` inverts the table -- the fallback is the solver that can
own everything, and only the robot is named -- and
:func:`test_the_fallback_is_a_solver_that_can_own_anything` pins that shape.
"""

from __future__ import annotations

import unittest

TIER = "fast"
"""No GPU. Gates every commit."""


class TestGenericStepsEveryShippedDomain(unittest.TestCase):
    def test_every_domain_has_a_backend_that_can_own_its_state(self) -> None:
        from robosimstudio.specs.physics import DOMAINS, GENERIC, get_backend, get_domain

        for name in DOMAINS.names():
            info = get_domain(name)
            if info.world_fixed:
                continue  # scenery owns no dynamics; build_islands sets it aside the same way
            with self.subTest(domain=name):
                spec = GENERIC.backend_for(name)
                self.assertIsNotNone(spec, f"GENERIC serves {list(GENERIC.domains())} and not {name!r}")
                supports = get_backend(spec.solver).supports
                if not supports:
                    continue  # a backend that declares no capability claims all of them
                # "shapes" excluded exactly as build_islands excludes it: a shape
                # is collision geometry, owned by whoever owns the body under it.
                needed = [kind for kind in info.state if kind != "shapes"]
                missing = [kind for kind in needed if kind not in supports]
                self.assertEqual(
                    [],
                    missing,
                    f"under GENERIC a {name!r} is solved by {spec.solver!r}, which integrates "
                    f"{list(supports)} and cannot own {missing}",
                )

    def test_the_fallback_is_a_solver_that_can_own_anything(self) -> None:
        """The silent case, and the reason the table is shaped the way it is.

        Under the dataclass default a **rod** landing on the ``"*"`` fallback
        does not raise -- bodies and joints are what MuJoCo is for -- and dies
        later, inside MuJoCo's own importer. ``GENERIC`` answers that by making
        the fallback the solver that can integrate *every* kind of state, so a
        domain nobody named -- a rod, a cloth, a kind you registered this
        morning -- lands on something that can hold it.

        Two entries are named beside it, each because the fallback would be
        *silently wrong* rather than loud: the robot for MuJoCo's joint
        drives, and ``soft`` for the SuperDex host engine -- a tetrahedral
        body passes VBD's capability check (particles are particles) and then
        integrates as the wrong physics with no message, which is exactly the
        failure mode this table exists to remove.
        """
        from robosimstudio.specs.physics import GENERIC, STATE_KINDS, get_backend

        fallback = get_backend(GENERIC.backend_for("*").solver)
        self.assertEqual(set(STATE_KINDS), set(fallback.supports), "the fallback cannot own every kind of state")
        self.assertEqual("mujoco", GENERIC.backend_for("articulated").solver, "the robot is the one named entry")
        self.assertTrue(
            get_backend(GENERIC.backend_for("soft").solver).host_engine, "soft is the other: a foreign engine"
        )
        self.assertEqual(("articulated", "soft", "*"), GENERIC.domains(), "and everything else falls through")

    def test_the_substep_count_survives_the_even_rule(self) -> None:
        """``ArrangementSpec`` refuses an odd substep count, and GENERIC is
        constructed at import: this asserts the value rather than the rule, so a
        future edit to the number is caught here rather than at first import."""
        from robosimstudio.specs.physics import GENERIC

        self.assertEqual(0, GENERIC.substeps % 2)
        self.assertGreater(GENERIC.substeps, 4, "GENERIC buys tunnelling margin on purpose; see its docstring")


if __name__ == "__main__":
    unittest.main()
