# SPDX-License-Identifier: Apache-2.0
"""The composer: declarations plus an arrangement to islands, channels and routing.

What :mod:`robosimstudio.scene.physics` decides, checked against synthetic
declarations rather than against a compiled scene -- an interaction graph is a
graph over index arrays, and none of the deciding needs a model that steps.

Three claims this file exists to keep true, in the order they matter:

* **The partition follows the backends, not the kinds.** Two domains served by
  an *equal* backend share one island, which is how a cable and a towel end up
  in one VBD without anybody saying so; two different backends are two islands
  and therefore a coupling problem.
* **A pair is detected once.** Every declared contact reaches exactly one
  pipeline -- the main one, or the channel's own -- and the composer says so
  rather than leaving it to be discovered as a double-counted force.
* **A solver nobody registered still composes.** ``backend("some.module.Solver")``
  is the extensibility contract in one line: a Newton release lands as a string
  in a task's ``staging.py``, with no edit under ``robosimstudio/``.

The model here is Newton's own -- built by hand, finalized, never stepped -- so
this is a fast-tier test that nonetheless exercises the real index arithmetic.

**The cable's domain is declared here.** The scenes below put a capsule chain
beside an arm and a towel, which is the rope's shape -- and the rope is a domain
pack a release may leave out. The composer reads no pack: it partitions on what a
domain *declares*. So the declaration is restated under a name of this module's
own (:data:`CHAIN`), and every claim is the same claim, on the same indices, in a
tree that carries the rope and in one that does not.
"""

from __future__ import annotations

import unittest
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.scene.physics import CompositionError, build_graph, compose
from robosimstudio.scene.physics.islands import partition
from robosimstudio.specs import ArrangementSpec, PhysicsManifest, SimRequirements, backend, contacts_with, coupler
from robosimstudio.specs.physics import DOMAINS, register_backend, register_coupler, register_domain

TIER = "fast"
"""No GPU: warp runs the kernels this never launches, and a graph is arithmetic."""


class _Range:
    """The half-open index ranges an entity records at build."""

    def __init__(self, bodies=(0, 0), shapes=(0, 0), joints=(0, 0), particles=(0, 0)) -> None:
        self.bodies, self.shapes, self.joints, self.particles = bodies, shapes, joints, particles

    @property
    def body_indices(self) -> np.ndarray:
        return np.arange(*self.bodies, dtype=np.int32)

    @property
    def shape_indices(self) -> np.ndarray:
        return np.arange(*self.shapes, dtype=np.int32)

    @property
    def particle_indices(self) -> np.ndarray:
        return np.arange(*self.particles, dtype=np.int32)


class _Entity:
    """The half of the entity contract the composer reads, and nothing else.

    Deliberately not an :class:`~robosimstudio.scene.entity.Entity`: what
    is under test is the *contract* -- a declaration, the state it owns, and for
    a robot its coupling surfaces -- so a stub that answers exactly that is what
    says whether the contract is the whole of it.
    """

    def __init__(self, name, sim, *, range=None, hands=()) -> None:
        self.name, self.spec, self.range = name, _Spec(name, sim), range or _Range()
        self._hands = list(hands)

    def sim_requirements(self):
        return self.spec.sim

    def owned_bodies(self):
        return self.range.body_indices

    def owned_particles(self):
        return self.range.particle_indices

    def owned_shapes(self):
        return self.range.shape_indices

    def coupling_bodies(self):
        return list(self._hands)


class _Spec:
    def __init__(self, name, sim) -> None:
        self.name, self.sim = name, sim


def _builder(*, bodies: int = 4, shapes: int = 4, particles: int = 0) -> newton.ModelBuilder:
    """The builder :func:`_model` finalizes, so a test can replicate it first."""
    builder = newton.ModelBuilder()
    for i in range(bodies):
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.2 * (i + 1)), wp.quat_identity()), mass=1.0)
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    for _ in range(max(0, shapes - bodies)):
        builder.add_shape_box(-1, hx=0.5, hy=0.5, hz=0.02)
    if particles:
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=max(1, int(particles**0.5) - 1),
            dim_y=max(1, int(particles**0.5) - 1),
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
        )
    builder.color()
    return builder


def _model(*, bodies: int = 4, shapes: int = 4, particles: int = 0) -> newton.Model:
    """A finalized model of the right shape: bodies on free joints, a shape each."""
    return _builder(bodies=bodies, shapes=shapes, particles=particles).finalize()


def _replicated(num_worlds: int, **kwargs) -> tuple[newton.Model, newton.Model, Any]:
    """``(template, replicated, WorldMap)`` -- what a fleet compile hands the composer."""
    from robosimstudio.core.worlds import WorldMap

    base = _builder(**kwargs)
    fleet = newton.ModelBuilder()
    fleet.gravity = base.gravity
    fleet.replicate(base, num_worlds, spacing=(0.0, 0.0, 0.0))
    template, model = base.finalize(), fleet.finalize()
    return template, model, WorldMap.of(template, model, num_worlds)


CHAIN = "probe_chain"
"""A capsule chain on cable joints: bodies, joints and shapes, on a maximal-coordinate solver.

What the rope's own pack declares for its Newton half, less the particles of
its other one. Registered for the length of this module and taken back out:
registries are process-global, and the fast tier may run in one process.
"""


def setUpModule() -> None:
    register_domain(
        CHAIN,
        state=("bodies", "joints", "shapes"),
        reads="rigid",
        role="deformables",
        contacts=("surface", "fixtures", "objects", "hands", "deformables"),
        summary="A capsule chain on cable joints. Exists only in this test.",
    )


def tearDownModule() -> None:
    DOMAINS._items.pop(CHAIN, None)
    DOMAINS._origins.pop(CHAIN, None)


ARM = SimRequirements(domain="articulated", contacts=contacts_with("surface", "objects"))
CABLE = SimRequirements(domain=CHAIN, contacts=contacts_with("surface", "hands", "deformables"))
CLOTH = SimRequirements(domain="cloth", contacts=contacts_with("surface", "hands", "deformables"), self_contact=True)
TABLE = SimRequirements(domain="static")


class TestIslandsFollowTheBackends(unittest.TestCase):
    def _partition(self, arrangement, parts, robot=None):
        model = _model(bodies=6, shapes=8, particles=9)
        graph = build_graph(model, parts, robot, arrangement, scene_statics={"table"})
        return graph, partition(graph, arrangement, model)

    def test_two_domains_on_one_backend_share_an_island(self):
        """The mechanism the towel-on-a-cable scene turns on: the arrangement
        names the same VBD for the chain and ``cloth``, and nothing else says
        the two belong together."""
        vbd = backend("vbd", iterations=40)
        arrangement = ArrangementSpec(
            backends=(("articulated", backend("mujoco")), (CHAIN, vbd), ("cloth", vbd)),
        )
        parts = {
            "arm": _Entity("arm", ARM, range=_Range(bodies=(0, 2), shapes=(0, 2)), hands=[1]),
            "cable": _Entity("cable", CABLE, range=_Range(bodies=(2, 4), shapes=(2, 4), joints=(2, 4))),
            "towel": _Entity("towel", CLOTH, range=_Range(particles=(0, 9))),
        }
        _graph, islands = self._partition(arrangement, parts, parts["arm"])
        self.assertEqual(len(islands), 2)
        vbd_island = next(i for i in islands if i.backend is vbd)
        self.assertEqual(set(vbd_island.entities), {"cable", "towel"})

    def test_a_different_parameter_is_a_different_island(self):
        """Equality is the rule, so two VBDs tuned differently are two solvers --
        which is what a task asking for a stiff rod beside a limp sheet wants,
        and what it has to be able to say."""
        arrangement = ArrangementSpec(
            backends=(
                ("articulated", backend("mujoco")),
                (CHAIN, backend("vbd", iterations=40)),
                ("cloth", backend("vbd", iterations=10)),
            ),
        )
        parts = {
            "arm": _Entity("arm", ARM, range=_Range(bodies=(0, 2), shapes=(0, 2)), hands=[1]),
            "cable": _Entity("cable", CABLE, range=_Range(bodies=(2, 4), shapes=(2, 4), joints=(2, 4))),
            "towel": _Entity("towel", CLOTH, range=_Range(particles=(0, 9))),
        }
        _graph, islands = self._partition(arrangement, parts, parts["arm"])
        self.assertEqual(len(islands), 3)

    def test_the_robots_island_steps_first(self):
        """It is the proxy source in every channel, so its poses have to exist
        before the destination reads them."""
        arrangement = ArrangementSpec(
            backends=(("articulated", backend("mujoco")), ("cloth", backend("vbd"))),
        )
        parts = {
            "towel": _Entity("towel", CLOTH, range=_Range(particles=(0, 9))),
            "arm": _Entity("arm", ARM, range=_Range(bodies=(0, 2), shapes=(0, 2)), hands=[1]),
        }
        _graph, islands = self._partition(arrangement, parts, parts["arm"])
        self.assertEqual(islands[0].entities, ("arm",))

    def test_a_domain_with_no_backend_is_named_rather_than_defaulted(self):
        arrangement = ArrangementSpec(backends=(("articulated", backend("mujoco")),))
        parts = {"towel": _Entity("towel", CLOTH, range=_Range(particles=(0, 9)))}
        with self.assertRaises(CompositionError) as caught:
            self._partition(arrangement, parts)
        self.assertIn("names no backend", str(caught.exception))

    def test_a_backend_that_cannot_own_the_state_is_refused_by_its_own_claim(self):
        """The ``mujoco`` alias declares which state kinds it integrates, so this
        is the backend's own statement rather than a name match."""
        arrangement = ArrangementSpec(backends=(("cloth", backend("mujoco")),))
        parts = {"towel": _Entity("towel", CLOTH, range=_Range(particles=(0, 9)))}
        with self.assertRaises(CompositionError) as caught:
            self._partition(arrangement, parts)
        self.assertIn("cannot own", str(caught.exception))

    def test_the_statics_shapes_go_to_the_deformable_island(self):
        """A rod has to rest on the table it was laid on, and an entry that
        lists shapes sees only those."""
        arrangement = ArrangementSpec(
            backends=(("articulated", backend("mujoco")), (CHAIN, backend("vbd"))),
        )
        parts = {
            "arm": _Entity("arm", ARM, range=_Range(bodies=(0, 2), shapes=(0, 2)), hands=[1]),
            "cable": _Entity("cable", CABLE, range=_Range(bodies=(2, 4), shapes=(2, 4), joints=(2, 4))),
            "table": _Entity("table", TABLE, range=_Range(shapes=(6, 8))),
        }
        _graph, islands = self._partition(arrangement, parts, parts["arm"])
        vbd = next(i for i in islands if i.is_deformable)
        self.assertTrue(set(range(6, 8)).issubset(set(vbd.shapes.tolist())))
        mjc = next(i for i in islands if not i.is_deformable)
        self.assertEqual(mjc.shapes.size, 0, "an entry with owned shapes sees only those; the arm's are derived")


class TestTheInteractionGraph(unittest.TestCase):
    def test_a_hands_rule_reaches_the_coupling_surfaces_and_not_the_arm(self):
        arrangement = ArrangementSpec(backends=(("articulated", backend("mujoco")), (CHAIN, backend("vbd"))))
        arm = _Entity("arm", ARM, range=_Range(bodies=(0, 2), shapes=(0, 2)), hands=[1])
        parts = {"arm": arm, "cable": _Entity("cable", CABLE, range=_Range(bodies=(2, 4), shapes=(2, 4)))}
        graph = build_graph(_model(bodies=4), parts, arm, arrangement)
        edge = next(e for e in graph.edges if {e.a, e.b} == {"arm", "cable"})
        self.assertTrue(edge.hands_only)

    def test_a_declared_role_nothing_plays_is_reported_and_a_default_is_not(self):
        """A cloth that *says* it touches the fixtures in a scene with none has
        a claim nobody honoured; one that inherits the domain default has not
        said anything, and reporting it would bury the real lines."""
        arrangement = ArrangementSpec(backends=(("cloth", backend("vbd")),))
        declared = SimRequirements(domain="cloth", contacts=contacts_with("fixtures"))
        graph = build_graph(_model(bodies=1, particles=9), {"a": _Entity("a", declared)}, None, arrangement)
        self.assertTrue(any("fixtures" in line for line in graph.unresolved))

        default = build_graph(
            _model(bodies=1, particles=9), {"a": _Entity("a", SimRequirements(domain="cloth"))}, None, arrangement
        )
        self.assertEqual(default.unresolved, [])

    def test_an_unregistered_domain_names_itself(self):
        arrangement = ArrangementSpec(backends=(("*", backend("mujoco")),))
        parts = {"a": _Entity("a", SimRequirements(domain="jelly"))}
        with self.assertRaises(CompositionError) as caught:
            build_graph(_model(), parts, None, arrangement)
        self.assertIn("register_domain", str(caught.exception))


class TestComposingAWholeScene(unittest.TestCase):
    """The end of the pipeline: a solver, the pair lists, and a manifest of both."""

    def _scene(self, arrangement, *, particles: int = 9):
        model = _model(bodies=6, shapes=8, particles=particles)
        arm = _Entity("arm", ARM, range=_Range(bodies=(0, 2), shapes=(0, 2)), hands=[1])
        parts = {
            "arm": arm,
            "cable": _Entity("cable", CABLE, range=_Range(bodies=(2, 4), shapes=(2, 4), joints=(2, 4))),
            "towel": _Entity("towel", CLOTH, range=_Range(particles=(0, particles))),
            "table": _Entity("table", TABLE, range=_Range(shapes=(6, 8))),
        }
        return compose(model, parts, arm, arrangement, scene_statics={"table"})

    def test_a_lone_island_needs_no_coupler(self):
        program = self._scene(ArrangementSpec(backends=(("*", backend("mujoco")),)), particles=0)
        self.assertEqual(program.manifest.coupler, "none")
        self.assertFalse(program.coupled)

    def test_the_manifest_round_trips_through_json(self):
        """A repro bundle carries this and nothing else about the physics, so a
        field that does not survive the trip is a field a bundle silently drops."""
        import json

        program = self._scene(ArrangementSpec(backends=(("*", backend("mujoco")),)), particles=0)
        node = json.loads(json.dumps(program.manifest.to_dict()))
        self.assertEqual(PhysicsManifest.from_dict(node), program.manifest)

    def test_the_manifest_records_the_effective_parameters(self):
        """What the solver was actually built with, defaults and all -- a
        manifest that recorded only what was stated would replay a different
        solver."""
        program = self._scene(ArrangementSpec(backends=(("*", backend("mujoco", njmax=99)),)), particles=0)
        params = dict(program.manifest.islands[0].backend.params)
        self.assertEqual(params["njmax"], 99)
        self.assertEqual(params["jacobian"], "sparse", "the alias's own default, recorded rather than implied")


class TestRoutingCoversEveryWorld(unittest.TestCase):
    """The pair lists a replicated scene is stepped with, and the world in each.

    The cut is the *replicated* model's pair table against the *interaction
    graph's* shape sets, and the graph holds template indices -- because the
    template is the model every entity bound to. Cut one against the other with
    nothing to reconcile them and every list comes back holding world 0's pairs:
    world 0 has contacts, no other world does, and nothing says so.

    Measured on ``cable_route/port_easy`` before :func:`route` took a world map:
    three worlds identical at reset, diverging from the first step, rods 19 m
    below the bench by step 120. Free fall -- there was nothing under them. The
    states stayed finite, the episodes ran their budget, and the rows read like
    rows.
    """

    ARRANGEMENT = ArrangementSpec(
        backends=((CHAIN, backend("vbd")), ("*", backend("mujoco"))),
        coupling=coupler("proxy"),
    )

    def _routing(self, num_worlds: int):
        template, model, worlds = _replicated(num_worlds, bodies=6, shapes=8)
        arm = _Entity("arm", ARM, range=_Range(bodies=(0, 2), shapes=(0, 2)), hands=[1])
        parts = {
            "arm": arm,
            "cable": _Entity("cable", CABLE, range=_Range(bodies=(2, 4), shapes=(2, 4), joints=(2, 4))),
            "table": _Entity("table", TABLE, range=_Range(shapes=(6, 8))),
        }
        del template
        program = compose(model, parts, arm, self.ARRANGEMENT, scene_statics={"table"}, worlds=worlds)
        return program, worlds, model

    def _lists(self, program) -> dict:
        out = {}
        if program.routing.main_pairs is not None:
            out["main"] = program.routing.main_pairs
        for key, rows in program.routing.channel_pairs.items():
            if rows is not None:
                out[f"channel{key}"] = rows
        return out

    def test_every_list_grows_with_the_world_count(self):
        one = self._lists(self._routing(1)[0])
        three = self._lists(self._routing(3)[0])
        self.assertTrue(one, "the scene routed nothing, so this proves nothing")
        for name, rows in one.items():
            with self.subTest(list=name):
                self.assertEqual(len(three[name]), 3 * len(rows), f"{name} did not cover three worlds")

    def test_every_world_gets_the_same_number_of_pairs(self):
        """An uneven split is a world that is partly in and partly out."""
        program, worlds, _ = self._routing(3)
        for name, rows in self._lists(program).items():
            with self.subTest(list=name):
                per_world = np.bincount(
                    np.minimum(rows[:, 0] // worlds.shapes_per_world, worlds.num_worlds - 1),
                    minlength=worlds.num_worlds,
                )
                self.assertEqual(len(set(per_world.tolist())), 1, f"{name} is uneven across worlds: {per_world}")

    def test_no_pair_reaches_across_worlds(self):
        """Worlds are superimposed at the origin, so a cross-world pair is a
        contact between two scenes that cannot see each other and is geometrically
        certain to fire."""
        program, worlds, model = self._routing(3)
        limit = worlds.shapes_per_world * worlds.num_worlds
        for name, rows in self._lists(program).items():
            with self.subTest(list=name):
                local = rows[(rows[:, 0] < limit) & (rows[:, 1] < limit)]
                if not len(local):
                    continue
                same = (local[:, 0] // worlds.shapes_per_world) == (local[:, 1] // worlds.shapes_per_world)
                self.assertTrue(bool(same.all()), f"{name} pairs one world's shapes against another's")
        del model

    def test_one_world_is_unchanged(self):
        """The expansion is guarded, so a single-world scene routes as it always did."""
        program, _, _ = self._routing(1)
        self.assertTrue(self._lists(program))


class TestExtensibility(unittest.TestCase):
    """A solver, a domain and a coupler the library has never heard of."""

    def test_a_dotted_path_backend_needs_no_registration(self):
        """The contract in one line: a Newton release lands as a string in a
        task's staging file, with no edit under ``robosimstudio/``."""
        arrangement = ArrangementSpec(backends=(("*", backend("newton.solvers.SolverXPBD", iterations=8)),))
        model = _model(bodies=2, shapes=3)
        parts = {"box": _Entity("box", SimRequirements(domain="rigid"), range=_Range(bodies=(0, 2), shapes=(0, 2)))}
        graph = build_graph(model, parts, None, arrangement)
        islands = partition(graph, arrangement, model)
        self.assertEqual(islands[0].solver_class, "newton.solvers.SolverXPBD")
        self.assertEqual(dict(islands[0].backend.params), {"iterations": 8}, "no defaults invented for a path")

    def test_a_registered_alias_and_a_registered_coupler_compose(self):
        """Both registries are open and keyed by string, so a task may add to
        either from its own directory."""
        register_backend("probe_solver", cls="newton.solvers.SolverXPBD", supports=("bodies", "joints", "shapes"))
        register_coupler("probe_coupler")
        from robosimstudio.scene.physics.couplers import COUPLER_BUILDERS, register_coupler_builder

        seen = {}

        def build(ctx):
            seen["islands"] = [i.name for i in ctx.islands]
            return object()

        register_coupler_builder("probe_coupler", build)
        try:
            arrangement = ArrangementSpec(
                backends=(("articulated", backend("probe_solver")), ("cloth", backend("vbd"))),
                coupling=coupler("probe_coupler"),
            )
            model = _model(bodies=2, shapes=3, particles=9)
            arm = _Entity("arm", ARM, range=_Range(bodies=(0, 2), shapes=(0, 2)), hands=[1])
            parts = {"arm": arm, "towel": _Entity("towel", CLOTH, range=_Range(particles=(0, 9)))}
            program = compose(model, parts, arm, arrangement)
            self.assertEqual(program.manifest.coupler, "probe_coupler")
            self.assertEqual(sorted(seen["islands"]), ["probe_solver", "vbd"])
        finally:
            COUPLER_BUILDERS._items.pop("probe_coupler", None)

    def test_a_second_builder_under_a_taken_kind_is_refused(self):
        """The coupler builders were the one registry here with no duplicate
        check at all, so a second registration silently won -- which is a
        scene's coupling swapped out from under it at whichever point in an
        import order nobody controls. An identical copy still has to be
        accepted, because two tasks each shipping their own is the case the
        redundancy rule exists for."""
        from robosimstudio.scene.physics.couplers import (
            COUPLER_BUILDERS,
            register_coupler_builder,
        )

        def build_mine(ctx):
            return object()

        def build_theirs(ctx):
            return object()

        register_coupler_builder("probe_dup", build_mine)
        try:
            # The same builder again -- a second task shipping its own copy.
            register_coupler_builder("probe_dup", build_mine)
            # Sameness is judged on (module basename, qualname), so two builders
            # that genuinely differ are two names -- which is what a clash looks
            # like in the wild, one task's builder against another's.
            with self.assertRaises(ValueError) as caught:
                register_coupler_builder("probe_dup", build_theirs)
            self.assertIn("probe_dup", str(caught.exception))
            self.assertIn("already registered", str(caught.exception))
        finally:
            COUPLER_BUILDERS._items.pop("probe_dup", None)

    def test_a_coupler_kind_with_no_builder_says_so(self):
        arrangement = ArrangementSpec(
            backends=(("articulated", backend("mujoco")), ("cloth", backend("vbd"))),
            coupling=coupler("nothing_builds_this"),
        )
        model = _model(bodies=2, shapes=3, particles=9)
        arm = _Entity("arm", ARM, range=_Range(bodies=(0, 2), shapes=(0, 2)), hands=[1])
        parts = {"arm": arm, "towel": _Entity("towel", CLOTH, range=_Range(particles=(0, 9)))}
        with self.assertRaises(CompositionError) as caught:
            compose(model, parts, arm, arrangement)
        self.assertIn("nothing_builds_this", str(caught.exception))


class TestIslandCapabilitiesAreRegisteredFacts(unittest.TestCase):
    """Two of the four island predicates used to be alias lists in ``islands.py``.

    They are facts on ``register_backend`` now, beside ``supports`` and
    ``host_engine``, so a third engine states them where it states everything
    else. The ratchet: no alias set may come back."""

    def test_the_facts_are_on_the_registration(self) -> None:
        from robosimstudio.specs.physics import get_backend

        self.assertTrue(get_backend("vbd").maximal_coordinates)
        self.assertTrue(get_backend("xpbd").maximal_coordinates)
        self.assertFalse(get_backend("mujoco").maximal_coordinates)
        self.assertTrue(get_backend("mujoco").native_contacts)
        self.assertFalse(get_backend("vbd").native_contacts)

    def test_no_alias_list_remains(self) -> None:
        from robosimstudio.scene.physics import islands

        self.assertFalse(hasattr(islands, "MAXIMAL_COORDINATE_ALIASES"), "a registered fact, not a list")


if __name__ == "__main__":
    unittest.main()
