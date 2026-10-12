# SPDX-License-Identifier: Apache-2.0
"""Two implementations of one kind, answering one contract.

The project's central bet is that competing implementations of a cable, a cloth
or a soft body **coexist and are the same object to everything above the
solver**. The tree has carried the evidence for the first half of that for some
time -- ``EntityRod`` builds a capsule chain on cable joints, ``EntityRodSuperdex``
builds a Cosserat centerline of particles, they share one ``RodSpec``, and one
tuple in an arrangement picks between them -- and none of the evidence for the
second half. Measured before this file existed: of ``handle``, ``tunables``,
``published_states``, the friction pass and the hold latch, the SuperDex rod had
**none**. It was not clickable, not draggable, not scoreable through the handle
tables and not graspable.

What fixed that is the split this file pins: a **capability** is one geometry
question and everything derived from it, and it is a different axis from the
**domain**, which picks the solver. An implementation says where its nodes are;
``substrate_points``, the curve's own numbers and (through the handle) the page
follow from that and are written once.

No GPU: the contract is checked against the classes and against a stand-in
snapshot, which is the level it lives at. That two *built* rods agree is
``tests/test_extension_surface`` and ``scripts/engine_ab.py``'s job.
"""

from __future__ import annotations

import unittest

import numpy as np
from robosimstudio.domains.rod import Centerline, EntityRod, EntityRodSuperdex

from robosimstudio.domains.capabilities import ParticleMixin
from robosimstudio.scene.entity import Entity

TIER = "fast"


class _Straight(Centerline):
    """A curve that is a straight line of `n` nodes, one metre long."""

    def __init__(self, n: int = 5) -> None:
        self._pts = np.stack([np.linspace(0.0, 1.0, n), np.zeros(n), np.zeros(n)], axis=1)

    def centerline(self, snapshot):
        del snapshot
        return self._pts


class _Bent(Centerline):
    """A right angle: two metre-long legs meeting at the origin."""

    def centerline(self, snapshot):
        del snapshot
        return np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class TestBothRodsShareOneContract(unittest.TestCase):
    """The acceptance criterion for the capability split."""

    def test_both_implementations_declare_the_same_capability(self):
        """One kind, two formulations, one thing they are asked."""
        self.assertTrue(issubclass(EntityRod, Centerline))
        self.assertTrue(issubclass(EntityRodSuperdex, Centerline))
        self.assertEqual(EntityRod.domain, EntityRodSuperdex.domain, "same kind")

    def test_they_differ_only_in_where_the_curve_is_kept(self):
        """``centerline`` is each implementation's own; everything else is not.

        The point of the split: what is *overridden* is the one method that says
        where the nodes live. ``substrate_points`` -- what every rod measure
        actually calls -- is not overridden by either, so the two cannot drift
        apart on the question a goal asks.
        """
        for cls in (EntityRod, EntityRodSuperdex):
            with self.subTest(cls=cls.__name__):
                self.assertIn("centerline", vars(cls), "each implementation owns its own curve accessor")
                self.assertNotIn("substrate_points", vars(cls), "derived, not written twice")

    def test_the_particle_implementation_publishes_a_handle(self):
        """It published none, which is what made it second-class.

        A kind that publishes no handle is composable, steppable and invisible:
        no row in the handle tables, so nothing can click it, drag it or score it
        by name. The Newton rod published one and the SuperDex rod did not, and
        they are supposed to be the same thing.
        """
        self.assertTrue(issubclass(EntityRodSuperdex, ParticleMixin))
        self.assertIn("handle", dir(EntityRodSuperdex))

    def test_the_shared_lifecycle_is_written_once(self):
        """``bind``, ``reset`` and ``observe`` were byte-identical in three classes."""
        for name in ("bind", "reset", "observe", "handle", "snapshot_rest"):
            with self.subTest(member=name):
                self.assertIn(name, vars(ParticleMixin))
                self.assertNotIn(name, vars(EntityRodSuperdex), "inherited, not copied")


class TestWhatACurveDerives(unittest.TestCase):
    def test_substrate_points_follows_from_the_curve(self):
        pts = _Straight(7).substrate_points(None)
        self.assertEqual(pts.shape, (7, 3))
        self.assertEqual(pts.dtype, np.float64)

    def test_an_empty_curve_measures_as_nothing_rather_than_as_zero(self):
        """``None`` is "ask something else"; an array of zeros is a wrong answer."""

        class _Empty(Centerline):
            def centerline(self, snapshot):
                del snapshot
                return np.zeros((0, 3))

        self.assertIsNone(_Empty().substrate_points(None))
        self.assertEqual(_Empty().curve_states(None), {})

    def test_arc_length_and_sag(self):
        states = _Straight(11).curve_states(None)
        self.assertAlmostEqual(states["length"], 1.0, places=9)
        self.assertAlmostEqual(states["sag"], 0.0, places=9)

    def test_a_straight_curve_has_no_bend(self):
        """Collinear nodes have infinite radius rather than a divide by zero."""
        self.assertEqual(_Straight(5).curve_states(None)["min_bend_radius"], float("inf"))

    def test_a_right_angle_reports_its_own_radius(self):
        """Menger curvature over the corner: the circle through the three nodes.

        Two unit legs at 90 degrees are a chord of sqrt(2) subtending a right
        angle, so the circumradius is sqrt(2)/2 -- checked against the closed
        form rather than against a recorded number.
        """
        got = _Bent().curve_states(None)["min_bend_radius"]
        self.assertAlmostEqual(got, np.sqrt(2.0) / 2.0, places=9)


class TestTheCapabilityIsNotTheDomain(unittest.TestCase):
    """The separation this whole split exists for.

    A capability is a plain mixin over ``Entity``: it registers no domain, needs
    no backend and says nothing about which solver serves the thing. That is what
    lets a cable be simulated as a soft body -- ``domain="soft"``, routed to a
    foreign engine -- and still be a cable to every rod measure and to the grasp
    latch. While the two were welded, a cable could only ever be a ``rod``.
    """

    def test_a_capability_carries_no_domain(self):
        self.assertFalse(hasattr(Centerline, "domain"))
        self.assertFalse(issubclass(Centerline, Entity))

    def test_a_kind_may_be_asked_a_question_its_domain_did_not_imply(self):
        """A soft-domain thing that is also a curve. This is the case that was impossible."""

        class _FemCable(Centerline, Entity):
            domain = "soft"

            def centerline(self, snapshot):
                del snapshot
                return np.array([[0.0, 0.0, 0.0], [0.0, 0.3, 0.0]])

        thing = _FemCable.__new__(_FemCable)
        self.assertEqual(thing.domain, "soft", "the solver sees a soft body")
        self.assertAlmostEqual(thing.curve_states(None)["length"], 0.3, places=9)
        self.assertEqual(thing.substrate_points(None).shape, (2, 3), "and a rod measure sees a curve")


class TestTheCapabilityIsPublishedByItsPack(unittest.TestCase):
    """``Centerline`` ships in the rod pack and is *published* through the library.

    The library defines no capability of its own: one is earned by a kind, so
    it lives in that kind's domain pack, and ``register_capability`` is how the
    pack makes it importable as ``robosimstudio.domains.capabilities.<Name>``.
    That name is what an entity in another domain declares it by -- a
    tetrahedral cable in ``soft`` -- so the entity's module names no pack, and
    a checkout without the pack simply has no such attribute.
    """

    def test_the_published_name_is_the_packs_own_class(self):
        from robosimstudio.domains import capabilities

        self.assertIs(capabilities.Centerline, Centerline)
        self.assertEqual("robosimstudio.domains.rod.centerline", Centerline.__module__)
        self.assertIs(capabilities.CAPABILITIES.get("Centerline"), Centerline)

    def test_the_library_module_defines_none_itself(self):
        """Read off the source, because a resolved name is cached nowhere and
        ``hasattr`` answers yes either way."""
        import ast
        from pathlib import Path

        from robosimstudio.domains import capabilities

        classes = [
            n.name for n in ast.parse(Path(capabilities.__file__).read_text()).body if isinstance(n, ast.ClassDef)
        ]
        self.assertEqual(["ParticleMixin"], classes, "a capability belongs in the pack of the kind that earned it")

    def test_a_name_nobody_published_is_an_attribute_error_that_says_what_is(self):
        from robosimstudio.domains import capabilities

        with self.assertRaises(AttributeError) as caught:
            capabilities.Surface  # noqa: B018 -- the access is the test
        self.assertIn("Centerline", str(caught.exception), "the message lists what the installed packs publish")
        self.assertFalse(hasattr(capabilities, "Volume"))

    def test_one_name_is_one_class(self):
        from robosimstudio.domains.capabilities import CAPABILITIES, register_capability

        self.assertIs(register_capability(Centerline), Centerline, "the same class again is fine")
        impostor = type("Centerline", (), {})
        with self.assertRaises(ValueError):
            register_capability(impostor)
        self.assertIs(CAPABILITIES.get("Centerline"), Centerline, "the refused one did not replace it")


if __name__ == "__main__":
    unittest.main()


class TestOneSpecCarriesBothImplementationsMatter(unittest.TestCase):
    """``RodSpec`` builds either rod, and that has to be true of the *matter*.

    The two implementations state what a cable is made of in genuinely different
    vocabularies: the capsule chain wants VBD penalty stiffnesses
    (``stretch_ke``, ``bend_ke``), a Cosserat rod wants ``EA``/``EI``/``GJ``,
    and those are not the same quantities under another name. Both therefore sit
    on the one spec, each read by the implementation that means it.

    The continuum half was **unreachable** until this test's change:
    ``EntityRodSuperdex.host_actor`` read it as ``getattr(spec,
    "young_modulus", 5.0e6)``, no spec in the tree declared it, and nothing
    subclassed to add one -- so "a suite that has measured its own says so on
    its spec", which is what that module's own docstring tells an author to do,
    was advice nobody could take.
    """

    def test_the_shared_spec_declares_both_vocabularies(self) -> None:
        import dataclasses

        from robosimstudio.domains.rod import RodSpec

        fields = {f.name for f in dataclasses.fields(RodSpec)}
        self.assertLessEqual({"stretch_ke", "bend_ke"}, fields, "the penalty half")
        self.assertLessEqual({"young_modulus", "poisson_ratio", "stiffness_damping"}, fields, "the continuum half")

    def test_stating_a_modulus_reaches_the_foreign_actor(self) -> None:
        """The whole point: saying it on the spec changes what the engine is handed."""
        from robosimstudio.domains.rod import RodSpec
        from robosimstudio.domains.rod.superdex import EntityRodSuperdex

        soft = EntityRodSuperdex(RodSpec(young_modulus=1.0e6)).host_actor().params
        stiff = EntityRodSuperdex(RodSpec(young_modulus=9.0e6)).host_actor().params
        self.assertLess(soft["axial_stiffness"], stiff["axial_stiffness"])
        self.assertAlmostEqual(stiff["axial_stiffness"] / soft["axial_stiffness"], 9.0, places=6)

    def test_the_default_is_the_number_that_was_buried(self) -> None:
        """Declaring it must not have changed what a bare ``rss.rod`` is made of."""
        import math

        from robosimstudio.domains.rod import RodSpec
        from robosimstudio.domains.rod.superdex import EntityRodSuperdex

        spec = RodSpec()
        params = EntityRodSuperdex(spec).host_actor().params
        self.assertAlmostEqual(params["axial_stiffness"], 5.0e6 * math.pi * spec.radius**2, places=6)
