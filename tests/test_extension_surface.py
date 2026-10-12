# SPDX-License-Identifier: Apache-2.0
"""A kind of thing the library has never heard of, composed on the same terms as a cloth.

**The acceptance criterion for the whole extensibility refactor.** A physics
domain, an entity that builds it, a spec that names it, a substep kernel it
contributes, a solver named by bare dotted path with no registration at all, and
a coupling scheme -- every one of them defined *in this file*, none of them known
to anything under ``robosimstudio/``. Then the scene builds and steps.

If any of it needed one line of library code, the seam is not open, and that is
what this module is for. It is written the way a third party would write it,
including the parts that are easy to get wrong:

* the spec is a plain frozen dataclass -- no base class, because
  ``build_stage`` and ``_make_entity`` both duck-type;
* the entity subclasses :class:`~robosimstudio.scene.entity.Entity` for the index
  bookkeeping and could decline to: ``steps_of`` finds lifecycle phases by
  *method name* and checks no base class;
* the domain declares its ``role``, its default ``contacts`` and its
  ``world_fixed`` flag, which is what stops it falling through to "an object that
  touches nothing" the way a content domain silently used to;
* the backend is ``"newton.solvers.SolverMuJoCo"`` -- a dotted path, resolved at
  compose time, registered nowhere.

The scene deliberately holds **two** islands: the probe domain on MuJoCo and a
cloth on VBD. One island takes the bare solver and never reaches a coupler at
all, so a single-island scene would leave the coupling seam untested.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

import newton
import numpy as np
import warp as wp
from robosimstudio.domains.cloth import ClothSpec

from robosimstudio.scene import BuildRequest, compile_scene
from robosimstudio.scene.entity import Entity
from robosimstudio.scene.physics.couplers import CouplerBuild, build_proxy, register_coupler_builder
from robosimstudio.scene.registry import register_entity
from robosimstudio.specs import CollisionSpec, SimRequirements, backend, contacts_with, coupler
from robosimstudio.specs.physics import ArrangementSpec, register_coupler, register_domain

TIER = "medium"
"""Builds and steps a real scene on a GPU."""

# ---------------------------------------------------------------------------
# Everything below here is what a third party writes. Nothing above the fold in
# robosimstudio/ knows any of these names.
# ---------------------------------------------------------------------------

register_domain(
    "probe_gravel",
    state=("bodies", "joints", "shapes"),
    reads="rigid",
    role="objects",
    contacts=("surface", "fixtures", "objects", "deformables"),
    summary="A handful of loose stones. Exists only in this test.",
)

register_coupler("probe_glue", summary="Proxy coupling under another name, to prove the seam is reached.")

COUPLED: list[str] = []
"""Which kinds this file's own builder was asked for. Empty means it was never reached."""


def build_glue(ctx: CouplerBuild) -> Any:
    """Record the call, then delegate.

    Delegating on purpose: what is under test is that a coupler kind registered
    from *outside* the library is dispatched to, not that a novel coupling
    algorithm converges. Writing a real one here would test Newton, and a test
    that fails for two unrelated reasons tells you neither.
    """
    COUPLED.append(ctx.spec.method)
    return build_proxy(ctx)


register_coupler_builder("probe_glue", build_glue)


@wp.kernel
def _settle_kernel(body_qd: wp.array(dtype=wp.spatial_vector), damping: float):
    i = wp.tid()
    body_qd[i] = body_qd[i] * damping


@dataclass(frozen=True, kw_only=True)
class GravelSpec:
    """A plain frozen dataclass. No library base class, and none needed.

    **It declares no build stage, and that is the change worth noticing.** It
    used to say ``stage = "extras"`` and explain at length why: ``objects`` was
    not "anything with bodies", it was the ``ObjectSpec`` contract, and a novel
    entity claiming it failed on an attribute rather than on anything to do with
    physics. So an unknown kind bought composability by opting out of handles,
    probes and every panel that reads a bin.

    The views are derived from what a thing *is* now, so there is no bin to
    choose and nothing to opt out of. ``extras`` is what anything declaring no
    stage already got; saying it added nothing but the belief that it mattered.
    """

    entity: str = "probe_gravel"
    name: str = "gravel"
    stones: int = 3
    half: float = 0.02
    at: tuple[float, float, float] = (0.0, 0.0, 0.12)
    damping: float = 0.995
    sim: Any = field(
        default_factory=lambda: SimRequirements(
            domain="probe_gravel", contacts=contacts_with("surface", "fixtures", "deformables")
        )
    )


@register_entity("probe_gravel")
class EntityGravel(Entity):
    """Builds a row of free boxes and damps them every substep."""

    domain = "probe_gravel"

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        cfg = builder.default_shape_cfg.copy()
        cfg.density = 700.0
        self.bodies = []
        for i in range(spec.stones):
            pose = self.placed(np.array([*spec.at, 0.0, 0.0, 0.0, 1.0], dtype=np.float64))
            pose[0] += (i - (spec.stones - 1) / 2) * 4.0 * spec.half
            body = builder.add_body(
                xform=wp.transform(tuple(float(v) for v in pose[:3]), wp.quat(*[float(v) for v in pose[3:7]])),
                label=f"{spec.name}_{i:02d}",
            )
            builder.add_shape_box(body, hx=spec.half, hy=spec.half, hz=spec.half, cfg=cfg, label=f"{spec.name}_{i:02d}")
            self.bodies.append(body)

    def contribute_hooks(self, hooks) -> None:
        """A device kernel in a named phase, at a cadence, from outside the library."""
        hooks.add("post_solve", self._damp, name="probe-gravel-damp", cadence=1)

    def _damp(self, state, dt) -> None:
        del dt
        wp.launch(_settle_kernel, dim=state.body_qd.shape[0], inputs=[state.body_qd, float(self.spec.damping)])


PHYSICS = ArrangementSpec(
    control_hz=60.0,
    substeps=16,
    backends=(
        # Registered nowhere. importlib resolves it at compose time.
        ("probe_gravel", backend("newton.solvers.SolverMuJoCo")),
        ("cloth", backend("vbd")),
        ("*", backend("newton.solvers.SolverMuJoCo")),
    ),
    coupling=coupler("probe_glue"),
    collision=CollisionSpec(),
)


class TestAnUnknownKindOfThingComposes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        COUPLED.clear()
        request = BuildRequest(
            name="probe",
            cell="bench_075",
            entities=(GravelSpec(), ClothSpec(name="rag", size=(0.2, 0.2), cells=(12, 12), pose=(0.0, 0.25, 0.10))),
            physics=PHYSICS,
        )
        cls.scene = compile_scene(request)
        for _ in range(60):
            cls.scene.pipeline.step()
        cls.scene.pipeline.sync()

    def test_the_scene_stepped(self) -> None:
        self.assertGreater(self.scene.pipeline.sim_time, 0.0)

    def test_the_unknown_domain_owns_its_bodies(self) -> None:
        """Not "it built". The interaction graph has to have *given* it its
        bodies, which is what a domain that fell through every branch never
        got."""
        node = self.scene.pipeline.program.graph.nodes["gravel"]
        self.assertEqual("probe_gravel", node.domain)
        self.assertEqual(3, node.bodies.size)
        self.assertEqual({"objects"}, set(node.roles))

    def test_the_declared_contacts_became_edges(self) -> None:
        """A domain registered by content used to touch **nothing** by default
        and nothing said so: a scene composed, stepped, and silently never
        collided."""
        edges = self.scene.pipeline.program.graph.edges_of("gravel")
        self.assertTrue(edges, "the probe domain collides with nothing in this scene")
        self.assertIn("rag", {e.a for e in edges} | {e.b for e in edges}, "the declared cloth contact is missing")

    def test_the_dotted_path_solver_was_resolved(self) -> None:
        islands = {i.name for i in self.scene.pipeline.program.islands}
        self.assertGreaterEqual(len(islands), 2, f"expected two islands, got {islands}")

    def test_this_files_coupler_was_the_one_that_built(self) -> None:
        self.assertEqual(["probe_glue"], COUPLED)

    def test_the_contributed_kernel_is_in_the_schedule(self) -> None:
        names = [name for _, name in self.scene.pipeline.program.hooks.schedule.names()]
        self.assertIn("probe-gravel-damp", names)

    def test_the_gravel_settled(self) -> None:
        """The damping kernel ran on the device, so the stones are on the bench
        rather than still falling. 0.775 is the bench top plus the half extent
        -- the same number :mod:`tests.test_bare_scenes` pins for a cube."""
        bodies = self.scene.pipeline.program.graph.nodes["gravel"].bodies
        z = self.scene.pipeline.state_0.body_q.numpy()[bodies, 2]
        self.assertLess(float(z.max()), 0.80, "a stone is still in the air")
        self.assertGreater(float(z.min()), 0.74, "a stone fell through the bench")


if __name__ == "__main__":
    unittest.main()
