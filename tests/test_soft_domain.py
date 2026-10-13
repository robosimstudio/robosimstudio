# SPDX-License-Identifier: Apache-2.0
"""The soft domain and the host-engine seam, without the engine.

Everything here runs with SuperDex absent on purpose: the domain registers, the
specs validate, the composer splits a host island out and records a recipe, and
none of it may touch the wheel -- a base checkout composes and refuses legibly,
which is half the extra's contract. The half that needs the engine lives in
``test_superdex_contact.py`` and ``test_superdex_teleport.py`` at the medium tier.

The stub host solver in this file is the second half of the acceptance story
the extension-surface test tells: a *foreign engine* backend -- not a Newton
solver at all -- declared in a test module, registered with ``host_engine=True``,
composed, scheduled, and recorded, with no library edit beyond the seam itself.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from typing import ClassVar

import newton
import numpy as np
import warp as wp

from robosimstudio.scene.physics import compose
from robosimstudio.scene.physics.compose import _channels
from robosimstudio.scene.physics.islands import partition
from robosimstudio.specs import ArrangementSpec, PhysicsManifest, SimRequirements, backend, contacts_with
from robosimstudio.specs.physics import (
    BACKENDS,
    GENERIC,
    HostActor,
    get_backend,
    register_backend,
    register_coupler,
)

TIER = "fast"
"""No GPU and no SuperDex: composition is index arithmetic plus one stub class."""


# --------------------------------------------------------------------------- stubs
class _Range:
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


class _Spec:
    def __init__(self, name, sim) -> None:
        self.name, self.sim = name, sim


class _Entity:
    def __init__(self, name, sim, *, range=None, hands=(), actor_params=None) -> None:
        self.name, self.spec, self.range = name, _Spec(name, sim), range or _Range()
        self._hands = list(hands)
        self._actor_params = actor_params

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

    def host_actor(self):
        params = dict(self._actor_params or {"kind": "soft", "density": 60.0})
        return HostActor(kind=str(params.pop("kind", "tet")), params=params)


class _StubHostSolver:
    """A foreign engine in one screenful: the whole contract the composer needs."""

    built: ClassVar[list] = []

    def __init__(self, model, *, recipe, substeps=1, **params) -> None:
        self.model, self.recipe, self.substeps, self.params = model, recipe, substeps, params
        type(self).built.append(self)

    def step(self, state_in, state_out, control, contacts, dt) -> None:
        del state_in, state_out, control, contacts, dt

    def notify_model_changed(self, flags) -> None:
        del flags

    def host_tick(self, state, dt) -> None:
        del state, dt


def _model(*, bodies: int = 4, shapes: int = 6, particles: int = 9, site_shapes: int = 0) -> newton.Model:
    builder = newton.ModelBuilder()
    for i in range(bodies):
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.2 * (i + 1)), wp.quat_identity()), mass=1.0)
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    for _ in range(max(0, shapes - bodies)):
        builder.add_shape_box(-1, hx=0.5, hy=0.5, hz=0.02)
    for _ in range(site_shapes):
        # A marker in a collision body's shape list, the way a MuJoCo site
        # arrives: present, world-fixed here for simplicity, and flagged out of
        # collision -- what the mirror recipe must skip.
        index = builder.add_shape_box(-1, hx=10.0, hy=0.005, hz=0.005)
        builder.shape_flags[index] = int(builder.shape_flags[index]) & ~int(newton.ShapeFlags.COLLIDE_SHAPES)
    if particles:
        side = max(1, int(particles**0.5) - 1)
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=side,
            dim_y=side,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
        )
    builder.color()
    return builder.finalize()


ARM = SimRequirements(domain="articulated", contacts=contacts_with("surface", "objects"))
CLOTH = SimRequirements(domain="cloth", contacts=contacts_with("surface", "hands", "deformables"))
SOFT = SimRequirements(domain="soft", contacts=contacts_with("surface", "hands", "objects"))
TABLE = SimRequirements(domain="static")


def _register_stub(alias="probe_host"):
    return register_backend(alias, cls=_StubHostSolver, supports=("particles",), host_engine=True)


# --------------------------------------------------------------------------- specs
class TestSoftSpecs(unittest.TestCase):
    def test_lame_round_trips(self):
        from robosimstudio.domains.soft import lame_from_young_poisson, young_poisson_from_lame

        for young, poisson in ((2.0e4, 0.25), (8.0e4, 0.0), (1.0e6, 0.45)):
            mu, lam = lame_from_young_poisson(young, poisson)
            back = young_poisson_from_lame(mu, lam)
            self.assertAlmostEqual(back[0], young, delta=young * 1e-9)
            self.assertAlmostEqual(back[1], poisson, delta=1e-9)

    def test_the_incompressible_limit_is_refused(self):
        from robosimstudio.domains.soft import lame_from_young_poisson

        with self.assertRaises(ValueError):
            lame_from_young_poisson(1.0e4, 0.5)
        with self.assertRaises(ValueError):
            lame_from_young_poisson(-1.0, 0.3)

    def test_sponge_validates_and_counts(self):
        from robosimstudio.domains.soft import SoftSpec, sponge

        spec = sponge(name="s", size=(0.1, 0.08, 0.04), cells=(4, 3, 2), at=(0.0, 0.0, 0.0))
        self.assertEqual(spec.particle_count, 5 * 4 * 3)
        self.assertEqual(spec.stage, "deformables")
        self.assertEqual(spec.entity, "soft")
        with self.assertRaises(ValueError):
            SoftSpec(name="bad", size=(0.1, 0.0, 0.1))
        with self.assertRaises(ValueError):
            SoftSpec(name="bad", cells=(0, 1, 1))

    def test_actor_params_are_json_scalars(self):
        from robosimstudio.domains.soft import SOFT_FOAM

        params = SOFT_FOAM.actor_params()
        json.dumps(params)
        self.assertNotIn("young", params)  # elasticity travels in the model, once
        for key in ("density", "friction", "stiffness_damping", "contact_penalty", "contact_padding"):
            self.assertIn(key, params)

    def test_material_is_slab_for_the_render_path(self):
        """``render_parts`` branches on ``material.is_slab``; a volumetric body
        answers True -- its thickness is geometry -- so the shared line needs no
        special case."""
        from robosimstudio.domains.soft import SOFT_FOAM

        self.assertTrue(SOFT_FOAM.is_slab)


class TestNodeSelectors(unittest.TestCase):
    """``anchored=``/``rigid=`` keep ``particle_sets``' selector contract.

    A mask computed on a 36-node lattice and reused on a 45-node one anchored
    nodes 0..11 of a bottom layer that is 0..14, and a mask past the end
    published node 54 of 45 -- both built without a word.
    """

    @staticmethod
    def _build(**selectors):
        from robosimstudio.domains.soft import EntitySoft, SoftSpec

        entity = EntitySoft(SoftSpec(name="b", size=(0.04, 0.02, 0.01), cells=(4, 2, 2), settle_frames=0, **selectors))
        entity.build(newton.ModelBuilder(), None)
        return entity

    def test_a_mask_of_the_right_length_picks_its_nodes(self):
        entity = self._build(anchored=lambda rest: rest[:, 2] < 1.0e-6)
        self.assertEqual(entity.host_actor().params["anchors"], list(range(15)))

    def test_a_mask_of_another_length_is_refused(self):
        for field in ("anchored", "rigid"):
            for length in (36, 55):
                with self.subTest(field=field, length=length):
                    mask = np.zeros(length, dtype=bool)
                    mask[:12] = True
                    with self.assertRaisesRegex(ValueError, f"`{field}`.*mask of {length} for 45"):
                        self._build(**{field: lambda rest, m=mask: m})

    def test_an_index_past_the_end_is_refused(self):
        with self.assertRaisesRegex(ValueError, "`anchored`"):
            self._build(anchored=lambda rest: np.array([0, 54]))


# --------------------------------------------------------------------- registration
class TestRegistration(unittest.TestCase):
    def test_the_bundle_registers_domain_backend_entity(self):
        import robosimstudio.domains.soft as soft

        from robosimstudio.scene.registry import ENTITIES

        self.assertEqual(soft.DOMAIN.reads, ("volume", "cloth"))
        self.assertEqual(soft.DOMAIN.state, ("particles",))
        info = get_backend("superdex")
        self.assertTrue(info.host_engine)
        self.assertEqual(info.supports, ("particles",))
        self.assertNotIn(".", info.alias)
        self.assertIs(ENTITIES.get("soft"), soft.EntitySoft)

    def test_generic_serves_soft_with_superdex(self):
        spec = GENERIC.backend_for("soft")
        self.assertEqual(spec.solver, "superdex")
        self.assertGreaterEqual(spec.substeps, 2)

    def test_an_alias_a_bundle_registers_is_found_from_a_cold_process(self):
        """GENERIC names ``superdex`` and an arm-only process never imports the
        bundle -- ``get_backend`` must load the domain scan once and retry."""
        code = "from robosimstudio.specs.physics import get_backend\nassert get_backend('superdex').host_engine\n"
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)


# ----------------------------------------------------------------------- composing
class TestHostIslandComposition(unittest.TestCase):
    def test_a_registered_coupler_without_a_builder_is_a_composition_error(self):
        """The regression for the ``kind``/``method`` NameError at the exact
        line a mid-integration backend author reaches first."""
        from robosimstudio.scene.physics import CompositionError
        from robosimstudio.scene.physics.couplers import CouplerBuild, build_coupled
        from robosimstudio.scene.physics.islands import Island

        register_coupler("orphaned_kind")
        islands = [
            Island(name="a", backend=backend("mujoco"), solver_class="x"),
            Island(name="b", backend=backend("vbd"), solver_class="y"),
        ]
        ctx = CouplerBuild(
            model=_model(particles=0),
            arrangement=ArrangementSpec(backends=(("*", backend("mujoco")),)),
            islands=islands,
            channels=[],
            spec=__import__("robosimstudio.specs.physics", fromlist=["coupler"]).coupler("orphaned_kind"),
            collision=ArrangementSpec(backends=(("*", backend("mujoco")),)).collision,
        )
        with self.assertRaises(CompositionError):
            build_coupled(ctx)

    def test_partition_marks_host_islands_and_keeps_statics_for_newton(self):
        _register_stub()
        try:
            arrangement = ArrangementSpec(
                backends=(
                    ("articulated", backend("mujoco")),
                    ("cloth", backend("vbd")),
                    ("soft", backend("probe_host")),
                ),
            )
            model = _model(bodies=4, shapes=6, particles=18)
            parts = {
                "arm": _Entity("arm", ARM, range=_Range(bodies=(0, 2), shapes=(0, 2)), hands=[1]),
                "towel": _Entity("towel", CLOTH, range=_Range(particles=(0, 9))),
                "blob": _Entity("blob", SOFT, range=_Range(particles=(9, 18))),
                "table": _Entity("table", TABLE, range=_Range(shapes=(4, 5))),
            }
            from robosimstudio.scene.physics import build_graph

            graph = build_graph(model, parts, parts["arm"], arrangement, scene_statics={"table"})
            islands = partition(graph, arrangement, model)
            host = next(i for i in islands if i.owns("blob"))
            vbd = next(i for i in islands if i.owns("towel"))
            self.assertTrue(host.is_host_engine)
            self.assertFalse(vbd.is_host_engine)
            # The table's shapes go to the Newton deformable island, never the host one.
            self.assertIn(4, vbd.shapes.tolist())
            self.assertNotIn(4, host.shapes.tolist())
            channels = _channels(graph, islands, None, model)
            hosted = {c.destination for c in channels if c.destination == host.name}
            self.assertEqual(hosted, {host.name})
        finally:
            BACKENDS._items.pop("probe_host", None)

    def test_a_host_only_scene_composes_with_no_newton_solver(self):
        """The whole seam, end to end, engine stubbed: split, recipe, schedule
        stage, manifest -- and nothing Newton-side constructed at all."""
        _register_stub()
        _StubHostSolver.built = []
        try:
            arrangement = ArrangementSpec(backends=(("soft", backend("probe_host", substeps=3)),))
            model = _model(bodies=0, shapes=2, particles=9, site_shapes=1)
            lone = SimRequirements(domain="soft", contacts=contacts_with("surface"))
            parts = {
                "blob": _Entity("blob", lone, range=_Range(particles=(0, 9))),
                "table": _Entity("table", TABLE, range=_Range(shapes=(0, 1))),
            }
            program = compose(model, parts, None, arrangement, scene_statics=("table",))
            self.assertEqual(program.problems(), [])
            self.assertEqual(len(_StubHostSolver.built), 1)
            self.assertEqual(_StubHostSolver.built[0].substeps, 3)
            record = next(i for i in program.manifest.islands if i.host_engine)
            self.assertEqual(record.solver_class, "probe_host")
            recipe = record.host_recipe
            self.assertEqual([a["name"] for a in recipe["actors"]], ["blob"])
            self.assertEqual(recipe["actors"][0]["particle_count"], 9)
            self.assertEqual(recipe["actors"][0]["params"]["density"], 60.0)
            self.assertTrue(any(m["body"] < 0 for m in recipe["mirrors"]))  # the table
            # The non-colliding marker in the shape list -- a site, the way a
            # MuJoCo hand carries them -- must not become a phantom collider.
            flags = model.shape_flags.numpy()
            site = int(np.flatnonzero((flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) == 0)[0])
            self.assertNotIn(site, [m["shape"] for m in recipe["mirrors"]])
            self.assertIn(("after", "cosim:" + record.name), program.manifest.stages)
            # The record must survive the manifest round trip as plain JSON.
            data = json.loads(json.dumps(program.manifest.to_dict()))
            self.assertNotIn("robosimstudio", json.dumps(data))
            again = PhysicsManifest.from_dict(data)
            self.assertEqual(again.island(record.name).host_recipe, recipe)
        finally:
            BACKENDS._items.pop("probe_host", None)


if __name__ == "__main__":
    unittest.main()
