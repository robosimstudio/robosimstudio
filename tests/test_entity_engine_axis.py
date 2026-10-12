# SPDX-License-Identifier: Apache-2.0
"""One kind of thing, an implementation per engine, chosen by the arrangement.

``register_entity("cloth")(EntityCloth)`` used to bind a kind to exactly one
builder, so a sheet was discretised by Newton however many engines could
simulate it -- and a foreign engine got its elastic modulus from the *backend*
(``backend("superdex", shell_young=...)``, island-wide) because there was
nowhere per-sheet to put it. The engine axis is the fix: a kind may register an
implementation per backend, and the arrangement that already says which solver
serves a domain is what selects one.

What is checked here is :func:`~robosimstudio.scene.registry.resolve_entity_class`
rather than a built scene, and deliberately: the end-to-end path is medium tier
(it needs a GPU), so before this module the resolution rules had no fast-tier
test at all. Every rule below is one that was got wrong at least once while the
axis was being designed.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any, ClassVar

from robosimstudio.scene.entity import Entity
from robosimstudio.scene.registry import ANY_ENGINE, entity_engines, entity_key, register_entity, resolve_entity_class
from robosimstudio.specs.physics import ArrangementSpec, SimRequirements, backend, register_domain

TIER = "fast"
"""Resolves classes against registries; builds no model and touches no device."""

DOMAIN = "axis_probe"
"""A domain of this module's own, so nothing here depends on what the tree ships."""

if DOMAIN not in __import__("robosimstudio.specs.physics", fromlist=["DOMAINS"]).DOMAINS:
    register_domain(DOMAIN, state=("particles",), reads="cloth", role="deformables", summary="A probe kind.")


@register_entity("axis_probe")
class ProbeAny(Entity):
    """The engine-neutral implementation: what every kind has today."""

    domain = DOMAIN


@register_entity("axis_probe", engine="vbd")
class ProbeVbd(Entity):
    """One engine's own, registered against the alias an arrangement states."""

    domain = DOMAIN


@register_entity("axis_probe", engine="mypkg.SolverGenesis")
class ProbeDottedPath(Entity):
    """Registered against a dotted path, verbatim -- see the collision test."""

    domain = DOMAIN


@dataclass(frozen=True, kw_only=True)
class ProbeSpec:
    """A spec that states no ``sim``, so the domain comes off the class."""

    name: str = "probe"
    entity: str = "axis_probe"


@dataclass(frozen=True, kw_only=True)
class DeclaredSpec:
    """A spec that states its own ``sim``, which is where the domain then comes from."""

    name: str = "probe"
    entity: str = "axis_probe"
    sim: Any = field(default_factory=lambda: SimRequirements(domain=DOMAIN))


def _arrangement(*backends) -> ArrangementSpec:
    return ArrangementSpec(control_hz=60.0, substeps=4, backends=tuple(backends))


class TestTheKeyIsTheOldKeyUntilAnEngineIsNamed(unittest.TestCase):
    """The engine-neutral case must be byte-identical to what it was."""

    def test_a_neutral_key_is_the_bare_name(self) -> None:
        self.assertEqual("axis_probe", entity_key("axis_probe"))
        self.assertEqual("axis_probe", entity_key("axis_probe", ANY_ENGINE))

    def test_a_keyed_one_reads_as_a_pair(self) -> None:
        self.assertEqual("axis_probe@vbd", entity_key("axis_probe", "vbd"))

    def test_a_kind_reports_what_it_has_implementations_for(self) -> None:
        self.assertEqual(
            [ANY_ENGINE, "mypkg.SolverGenesis", "vbd"],
            sorted(entity_engines("axis_probe")),
        )


class TestResolution(unittest.TestCase):
    def test_no_arrangement_takes_the_neutral_implementation(self) -> None:
        """``Scene.add`` resolves a ``Ref`` class before any backend is known."""
        self.assertIs(ProbeAny, resolve_entity_class(ProbeSpec()))

    def test_the_arrangement_selects_the_engines_own(self) -> None:
        arrangement = _arrangement((DOMAIN, backend("vbd")))
        self.assertIs(ProbeVbd, resolve_entity_class(ProbeSpec(), arrangement))

    def test_an_engine_with_no_implementation_falls_back(self) -> None:
        """What keeps every kind registered before the axis existed working."""
        arrangement = _arrangement((DOMAIN, backend("mujoco")))
        self.assertIs(ProbeAny, resolve_entity_class(ProbeSpec(), arrangement))

    def test_the_star_fallback_is_honoured(self) -> None:
        self.assertIs(ProbeVbd, resolve_entity_class(ProbeSpec(), _arrangement(("*", backend("vbd")))))

    def test_a_declared_sim_supplies_the_domain(self) -> None:
        arrangement = _arrangement((DOMAIN, backend("vbd")))
        self.assertIs(ProbeVbd, resolve_entity_class(DeclaredSpec(), arrangement))

    def test_a_spec_may_override_the_arrangement(self) -> None:
        """``SimRequirements.backend`` is the asset's own say; it wins."""
        spec = DeclaredSpec(sim=SimRequirements(domain=DOMAIN, backend=backend("vbd")))
        arrangement = _arrangement((DOMAIN, backend("mujoco")))
        self.assertIs(ProbeVbd, resolve_entity_class(spec, arrangement))


class TestTheWaysThisWasGotWrong(unittest.TestCase):
    """Three rules that each cost a bug while the axis was being designed."""

    def test_a_missing_backend_resolves_rather_than_raising(self) -> None:
        """``backend_for`` returns ``None`` when the arrangement serves neither
        the domain nor ``"*"``, and several tests stage exactly that.

        The message worth reading is ``islands.partition``'s -- it names the
        domain and what the arrangement *does* serve. Raising here would
        pre-empt it with an ``AttributeError`` about ``NoneType.solver``.
        """
        arrangement = _arrangement(("articulated", backend("mujoco")))
        self.assertIsNone(arrangement.backend_for(DOMAIN))
        self.assertIs(ProbeAny, resolve_entity_class(ProbeSpec(), arrangement))

    def test_a_dotted_path_does_not_collide_with_an_alias(self) -> None:
        """The engine key is the solver string **verbatim**, never a normalised
        alias.

        An island's *name* reduces ``mypkg.SolverGenesis`` to ``"genesis"``,
        which is right for a label and wrong for a lookup: under that rule any
        third party's class ending in ``SolverGenesis`` would silently select
        the Genesis implementation of every kind. So the two spellings are two
        keys, and neither reaches the other.
        """
        dotted = _arrangement((DOMAIN, backend("mypkg.SolverGenesis")))
        self.assertIs(ProbeDottedPath, resolve_entity_class(ProbeSpec(), dotted))
        # ...and the reduced spelling reaches the neutral one, not the dotted.
        self.assertIs(ProbeAny, resolve_entity_class(ProbeSpec(), _arrangement((DOMAIN, backend("genesis")))))

    def test_an_unstated_domain_is_not_silently_rigid(self) -> None:
        """``SimRequirements.domain`` defaults to ``"rigid"`` and is therefore
        never falsy, so ``spec.sim.domain or cls.domain`` reads the class
        *never* -- and a spec stating contacts but no domain would resolve
        against the rigid backend.

        The rule is the one ``Entity.sim_requirements`` already applies: the
        declaration when there is one, the class when there is not.
        """
        no_sim = ProbeSpec()
        self.assertIs(ProbeVbd, resolve_entity_class(no_sim, _arrangement((DOMAIN, backend("vbd")))))
        # The class's domain is what was consulted: an arrangement that serves
        # only *rigid* leaves this kind on the neutral implementation.
        self.assertIs(ProbeAny, resolve_entity_class(no_sim, _arrangement(("rigid", backend("vbd")))))


class TestStatics(unittest.TestCase):
    def test_a_world_fixed_domain_is_not_keyed_on_the_star_backend(self) -> None:
        """Nothing integrates a static, so the arrangement never resolves a
        backend for one -- and keying it on whatever happens to serve everything
        else would select an implementation for an engine that will never see it.
        """
        from robosimstudio.scene.registry import get_entity

        @dataclass(frozen=True, kw_only=True)
        class Staticish:
            name: str = "peg"
            entity: str = "static"
            stage: ClassVar[str] = "scenery"

        neutral = get_entity("static")
        self.assertIs(neutral, resolve_entity_class(Staticish(), _arrangement(("*", backend("vbd")))))


class TestTheShippedCableHasTwo(unittest.TestCase):
    """The axis, on a kind the tree actually ships -- structural, no engine needed.

    A cable is the case the axis exists for. Newton solves one as a chain of
    capsules on cable joints; SuperDex solves the same cable as a Cosserat
    centerline, which owns particles instead of bodies and states its matter as
    axial/flexural/torsional stiffness instead of VBD's stretch/bend pair.
    Neither is a translation of the other, which is why one entity with a switch
    inside it would have had to pick a formulation and lie about the other.
    """

    def test_the_rod_kind_has_an_implementation_per_engine(self) -> None:
        from robosimstudio.scene.registry import entity_engines, get_entity

        self.assertEqual(["*", "superdex"], sorted(entity_engines("rod")))
        self.assertEqual("EntityRod", get_entity("rod").__name__)
        self.assertEqual("EntityRodSuperdex", get_entity("rod", engine="superdex").__name__)

    def test_both_are_reached_from_one_spec_by_the_arrangement_alone(self) -> None:
        from robosimstudio.domains.rod import RodSpec

        spec = RodSpec(name="cable")
        newton_side = resolve_entity_class(spec, _arrangement(("rod", backend("vbd"))))
        foreign = resolve_entity_class(spec, _arrangement(("rod", backend("superdex"))))
        self.assertEqual("EntityRod", newton_side.__name__)
        self.assertEqual("EntityRodSuperdex", foreign.__name__)

    def test_the_domain_admits_both_representations(self) -> None:
        """``state`` is what the composer checks a backend against, and a rod
        owns bodies under one implementation and particles under the other."""
        from robosimstudio.specs.physics import get_domain

        state = get_domain("rod").state
        self.assertIn("bodies", state)
        self.assertIn("particles", state)

    def test_the_foreign_one_declares_a_rod_actor(self) -> None:
        """And the kind it declares is one something knows how to read."""
        from robosimstudio.domains.rod import RodSpec
        from robosimstudio.domains.rod.superdex import EntityRodSuperdex

        from robosimstudio.engines.geometry import ACTOR_KINDS

        actor = EntityRodSuperdex(RodSpec(name="cable")).host_actor()
        self.assertEqual("rod", actor.kind)
        self.assertIn(actor.kind, ACTOR_KINDS.names())
        # Stiffnesses in the engine's own vocabulary, derived from the geometry
        # the shared spec states -- not VBD's stretch_ke wearing a new label.
        self.assertGreater(actor.params["axial_stiffness"], 0.0)
        self.assertEqual(2, len(actor.params["flexural_stiffness"]))
        self.assertGreater(actor.params["torsional_stiffness"], 0.0)
        self.assertGreater(actor.params["linear_density"], 0.0)

    def test_the_two_weigh_the_same_cable(self) -> None:
        """An A/B between them is a comparison rather than two different cables."""
        import math

        from robosimstudio.domains.rod import RodSpec
        from robosimstudio.domains.rod.superdex import EntityRodSuperdex

        spec = RodSpec(name="cable", length=0.5, radius=0.008, density=350.0)
        linear = EntityRodSuperdex(spec).host_actor().params["linear_density"]
        self.assertAlmostEqual(spec.density * math.pi * spec.radius**2, linear, places=9)


class TestTheAxisReachesMeasurement(unittest.TestCase):
    """Two implementations of one kind must agree about what may be *asked* of
    them, not only about the physics.

    ``SUBSTRATES`` names a state **layout** -- ``"rod"`` is "an ordered sequence
    of positions along a length" -- and its reader assumed those positions were
    body poses, because while a kind had one implementation that was the only
    way a rod could be built. The centerline implementation owns no body, so it
    built, stepped, came to rest correctly, and bound to nothing: every rod goal
    (``winding_turns``, ``straightness``, ``cable.point(0.6)``) failed on it.

    So an entity says where its points are and the reader stops guessing. An
    entity that says nothing keeps the old inference, which is what makes this
    safe for every rod that already worked.
    """

    def test_the_base_declares_it_and_answers_nothing(self) -> None:
        from robosimstudio.scene.entity import Entity

        self.assertIsNone(Entity(ProbeSpec()).substrate_points(object()))

    def test_the_newton_rod_answers_with_its_capsule_endpoints(self) -> None:
        """Endpoints, not centres: the same polyline the Cosserat rod answers with.

        Centres were one point fewer and spanned one segment less, so the two
        implementations disagreed about the length of one cable -- the 4%
        ``scripts/engine_ab.py`` showed for as long as it existed.
        """
        import numpy as np
        from robosimstudio.domains.rod import EntityRod, RodSpec

        entity = EntityRod(RodSpec(name="cable"))
        # ``bodies`` is resolved at bind, not derived from the build range --
        # the rod carries its own ordered index list because *order* is the
        # whole content of the substrate. The half-lengths are the build's.
        entity.bodies = np.arange(2, 6, dtype=np.int32)
        entity._half_heights = np.full(4, 0.05)

        class _Snap:
            # Four identical capsules, identity rotation: local +Z is world +Z.
            body_q = np.tile(np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]), (8, 1))

        points = entity.substrate_points(_Snap())
        self.assertEqual((5, 3), points.shape, "one point per capsule end, plus the last capsule's far end")
        np.testing.assert_allclose(
            points[:4, 2], 0.25, err_msg="each capsule's first endpoint is half a length below its centre"
        )
        np.testing.assert_allclose(points[4, 2], 0.35, err_msg="and the curve ends half a length past the last centre")

    def test_a_rod_assigned_bodies_by_hand_refuses_to_guess_its_endpoints(self) -> None:
        import numpy as np
        from robosimstudio.domains.rod import EntityRod, RodSpec

        entity = EntityRod(RodSpec(name="cable"))
        entity.bodies = np.arange(2, 6, dtype=np.int32)

        class _Snap:
            body_q = np.tile(np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]), (8, 1))

        with self.assertRaises(RuntimeError):
            entity.substrate_points(_Snap())

    def test_the_centerline_rod_answers_off_its_probe(self) -> None:
        """Off the probe, never off the model: the model's array is the
        as-built layout and a cable that has fallen reports itself where it was
        declared -- a plausible number, confidently wrong."""
        import numpy as np
        from robosimstudio.domains.rod import RodSpec
        from robosimstudio.domains.rod.superdex import EntityRodSuperdex

        entity = EntityRodSuperdex(RodSpec(name="cable"))
        # Resolved at bind. The probe holds the **whole model's** particle
        # array -- every deformable's, in build order -- so the accessor has to
        # index by the cable's own nodes or a rod measure is handed a towel's
        # vertices as well. Here the cable is nodes 4..12 of a longer array.
        entity.particles = np.arange(4, 13, dtype=np.int32)
        every = np.zeros((20, 3))
        every[4:13] = np.linspace([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], 9)

        class _Probe:
            positions = every

        class _Snap:
            particles: ClassVar[dict] = {"cable": _Probe()}

        points = entity.substrate_points(_Snap())
        self.assertEqual((9, 3), points.shape, "its own nodes, not the model's whole array")
        self.assertAlmostEqual(0.5, float(np.linalg.norm(points[-1] - points[0])), places=9)

        class _Empty:
            particles: ClassVar[dict] = {}

        self.assertIsNone(entity.substrate_points(_Empty()), "no probe, no claim -- the reader falls back")

    def test_both_implementations_answer_the_same_question(self) -> None:
        """The property that makes them two implementations of *one* kind."""
        from robosimstudio.domains.rod import EntityRod
        from robosimstudio.domains.rod.superdex import EntityRodSuperdex

        from robosimstudio.scene.entity import Entity

        for cls in (EntityRod, EntityRodSuperdex):
            self.assertIsNot(cls.substrate_points, Entity.substrate_points, f"{cls.__name__} must declare it")


class TestOnlyAnEntityCanBeRegisteredAsOne(unittest.TestCase):
    """What the registry will and will not accept as an entity class.

    The compiler calls Entity's surface on whatever it finds in the registry,
    so a non-Entity there does not fail at registration -- it fails during a
    scene build, as a missing attribute on an object nobody expected to be
    there. That is a long way from the line that caused it.

    And the line that causes it is specific: a helper class inserted directly
    beneath ``@register_entity`` takes the decorator, and the entity it was
    written for goes unregistered. Both halves are silent. This is the check
    that makes the first one loud, at import, naming the class.
    """

    def test_a_plain_class_is_refused_at_registration(self) -> None:
        class _Helper:
            """Whatever a task author put above their entity."""

        with self.assertRaises(TypeError) as caught:
            register_entity("not_an_entity")(_Helper)
        message = str(caught.exception)
        self.assertIn("_Helper", message, "name the class, not just the rule")
        self.assertIn("not_an_entity", message, "and the kind it was claiming")

    def test_a_refused_class_is_not_left_in_the_registry(self) -> None:
        """A half-registration would be worse than either outcome."""

        class _Helper:
            pass

        with self.assertRaises(TypeError):
            register_entity("also_not_an_entity")(_Helper)
        with self.assertRaises(KeyError):
            resolve_entity_class(_Spec("also_not_an_entity"))

    def test_an_entity_subclass_is_accepted(self) -> None:
        @register_entity("a_real_one")
        class _Real(Entity):
            pass

        self.assertIs(_Real, resolve_entity_class(_Spec("a_real_one")))


@dataclass
class _Spec:
    """The one field :func:`resolve_entity_class` reads off a spec."""

    entity: str


if __name__ == "__main__":
    unittest.main()
