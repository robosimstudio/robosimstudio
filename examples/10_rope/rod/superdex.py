# SPDX-License-Identifier: Apache-2.0
"""The same cable, simulated as a Cosserat rod by a foreign engine.

:class:`~robosimstudio.domains.rod.entity.EntityRod` builds a rod the way Newton
can solve one: twenty-eight overlapping capsules on cable joints, integrated by
VBD in maximal coordinates. That is a *representation*, not the rod -- and it is
the only one that existed, because a kind of thing had exactly one
implementation and that implementation authored Newton bodies.

This is the second. It builds the same cable as a **centerline**: one particle
per node, flagged inactive so no Newton solver touches them, integrated by
SuperDex's Cosserat rod -- which carries a twist per node and states its matter
as axial, flexural and torsional stiffness rather than as VBD's stretch/bend
pair. Newton still owns the memory, so the particle probe, the renderer, the
read-back and the repro tape work on it without hearing about SuperDex; the same
trick the soft kind uses for a sponge.

**Which one a scene gets is one line of arrangement:**

    ("rod", backend("vbd"))        -> EntityRod,         capsules on cable joints
    ("rod", backend("superdex"))   -> EntityRodSuperdex, a Cosserat centerline

and the spec is the same ``RodSpec`` either way. That is the point of the engine
axis: the two formulations disagree about *everything* internal -- degrees of
freedom, material vocabulary, what a "segment" even is -- and agree about what a
cable is, which is what a task states.

Why the material is derived rather than declared. A Cosserat rod wants ``EA``,
``EI`` and ``GJ``; ``RodSpec`` states ``stretch_ke`` and ``bend_ke``, which are
VBD's penalty stiffnesses and are not those quantities under another name. So
this derives all three from the geometry the spec *does* state -- radius,
density -- and one elastic modulus, the way a cable is actually specified. A
suite that has measured its cord subclasses ``RodSpec``, adds the field, and
this reads it: the same escape hatch every other entity here offers.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import newton
import numpy as np
import warp as wp

from robosimstudio.core.probes.particle import ParticleProbe
from robosimstudio.domains.capabilities import ParticleMixin, host_bridge
from robosimstudio.scene.entity import Entity
from robosimstudio.specs.physics import HostActor

from .centerline import Centerline
from .entity import RodSpec

if TYPE_CHECKING:
    from robosimstudio.core.pipeline import ScenePipeline

__all__ = ["EntityRodSuperdex"]

INACTIVE = int(newton.ParticleFlags.ACTIVE) & ~int(newton.ParticleFlags.ACTIVE)
"""The default particle flags with ``ACTIVE`` cleared.

Written as the mask rather than as ``0`` because zero is only correct while
``ACTIVE`` is the only flag there is, and this says what it means: every Newton
solver and the soft-contact pass gate on this bit, so a whole-model VBD in the
same scene integrates past these nodes without touching them. SuperDex writes
the same arrays from the eager phase instead. Newton owns the memory; the
foreign engine owns the physics -- the idiom the soft kind established.
"""


class EntityRodSuperdex(ParticleMixin, Centerline, Entity):
    """One cable, as a centerline of inactive particles.

    Args:
        spec: The same :class:`~robosimstudio.domains.rod.entity.RodSpec` the
            Newton implementation takes.
        world: Which world.
    """

    spec: RodSpec
    domain = "rod"

    def __init__(self, spec: RodSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.particles: np.ndarray = np.zeros(0, dtype=np.int32)
        """Global indices of this cable's centerline nodes, in order along it."""
        self.probe: ParticleProbe | None = None
        self.tube_radius = float(spec.radius)
        """What a renderer sweeps the centerline by: the cable's own radius."""
        self.rest: dict[str, np.ndarray] = {}
        self.bridge: Any = None
        """The engine's bridge, once the scene is ready; ``None`` on a Newton solver."""
        self._pins: dict[int, tuple[int, int]] = {}
        """Slot -> ``(pin id, global node)``. The engine carries the pin on the hand's body."""

    # -------------------------------------------------------------- phase 1
    def nodes(self) -> np.ndarray:
        """The centerline at rest, ``[segments + 1, 3]``, in world coordinates.

        ``segments`` is a capsule count in the Newton implementation and a
        *segment* count here, so the two discretise the same cable at the same
        resolution and a spec's number means one thing.
        """
        # The spec's own rest polyline -- a stated one, or the straight line --
        # lifted node by node into the world, so a coiled cable reaches this
        # engine exactly as it reaches the capsule chain.
        return np.asarray(
            [self.placed((*node, 0.0, 0.0, 0.0, 1.0))[:3] for node in self.spec.rest_polyline()], dtype=np.float64
        )

    def stress_free_nodes(self) -> np.ndarray:
        """:attr:`RodSpec.stress_free` as nodes: the same legs as :meth:`nodes`, in the stated shape."""
        nodes = self.nodes()
        legs = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
        shape = self.spec.stress_free
        if isinstance(shape, str):
            if shape != "straight":
                raise ValueError(f"rod {self.spec.name!r}: stress_free is 'straight' or a polyline, got {shape!r}")
            axis = (nodes[1] - nodes[0]) / legs[0]
            return nodes[0] + np.concatenate([[0.0], np.cumsum(legs)])[:, None] * axis[None, :]
        free = np.asarray(shape, dtype=np.float64).reshape(-1, 3)
        if free.shape != nodes.shape:
            raise ValueError(f"rod {self.spec.name!r}: stress_free has {len(free)} nodes, the rod {len(nodes)}")
        mismatch = np.abs(np.linalg.norm(np.diff(free, axis=0), axis=1) - legs).max()
        if mismatch > 1.0e-6:
            raise ValueError(f"rod {self.spec.name!r}: stress_free's legs differ from the rod's by {mismatch:.2e} m")
        return free

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        nodes = self.nodes()
        # Total mass from the cable's own geometry, spread over its nodes -- so
        # the two implementations weigh the same thing, which is what makes an
        # A/B between them a comparison rather than two different cables.
        area = math.pi * float(spec.radius) ** 2
        node_mass = float(spec.density) * area * float(spec.length) / len(nodes)
        for point in nodes:
            builder.add_particle(
                pos=wp.vec3(*[float(v) for v in point]),
                vel=wp.vec3(0.0, 0.0, 0.0),
                mass=node_mass,
                radius=float(spec.radius),
                flags=INACTIVE,
            )

    def host_actor(self) -> HostActor:
        """What SuperDex needs beyond the nodes: the rod's matter, in its terms.

        ``EA``, ``EI`` and ``GJ`` for a circular section, from the modulus and
        the radius. Derived here rather than stated because the model carries no
        array to invert them back out of -- unlike a tet body, whose Lame pair
        *is* in ``tet_materials`` and is read from there so there is only one
        statement of it.
        """
        spec = self.spec
        radius = float(spec.radius)
        young = float(spec.young_modulus)
        poisson = float(spec.poisson_ratio)
        area = math.pi * radius**2
        second_moment = math.pi * radius**4 / 4.0  # I, circular section
        shear = young / (2.0 * (1.0 + poisson))
        params: dict[str, Any] = {
            "radius": radius,
            "axial_stiffness": float(spec.axial_stiffness) if spec.axial_stiffness is not None else young * area,
            "flexural_stiffness": [young * second_moment, young * second_moment],
            # Polar second moment is 2I for a circle, so GJ = G * 2I.
            "torsional_stiffness": shear * 2.0 * second_moment,
            "linear_density": float(spec.density) * area,
            # rho * J per unit length, J = 2I for a circle.
            "linear_rotational_inertia": float(spec.density) * 2.0 * second_moment,
            "stiffness_damping": float(spec.stiffness_damping),
            "friction": float(spec.mu_cable) ** 0.5,
            "self_contact": bool(self.sim_requirements().self_contact),
            "contact_padding": float(spec.contact_gap),
            "contact_penalty": float(spec.contact_penalty),
            "skin": bool(spec.skin),
            "skin_segments": int(spec.skin_segments),
        }
        if spec.skin_samples:
            params["skin_samples"] = str(spec.skin_samples)
        if spec.stress_free is not None:
            params["stress_free"] = self.stress_free_nodes().tolist()
        for key in ("contact_threshold", "contact_smoothing", "friction_falloff"):
            value = getattr(spec, key)
            if value is not None:
                params[key] = float(value)
        count = int(spec.segments) + 1
        # ``anchors`` was a RodSpec field this implementation never read: a
        # cable strung between two posts fell off both of them here.
        ends = {"start": 0, "end": count - 1}
        unknown = [a for a in spec.anchors if a not in ends]
        if unknown:
            raise ValueError(f"rod {spec.name!r}: anchors must name 'start' or 'end', got {unknown}")
        if spec.anchors:
            # A clamp, as the capsule chain's fixed joint is: the end element's
            # two nodes (its position and tangent) and its twist. The end node
            # alone was a pin: a level 300 mm cable anchored at its start swung
            # 74 degrees down about it in 120 steps here (288 mm of tip drop),
            # where the capsule chain's first segment stayed within 0.2 degrees.
            elements = {"start": 0, "end": count - 2}
            params["anchors"] = sorted({n for a in spec.anchors for n in (elements[a], elements[a] + 1)})
            params["anchor_twists"] = sorted({elements[a] for a in spec.anchors})
        welds = []
        if spec.joins:
            arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.nodes(), axis=0), axis=1))])
            for end, other, length in spec.joins:
                if end not in ends:
                    raise ValueError(f"rod {spec.name!r}: a join names 'start' or 'end', got {end!r}")
                from_end = arc if end == "start" else arc[-1] - arc
                nodes = np.flatnonzero(from_end <= float(length) + 1.0e-9)
                welds.append({"to": str(other), "nodes": [int(i) for i in nodes], "contact": False})
            params["welds"] = welds
        return HostActor(kind="rod", params=params)

    # -------------------------------------------------------------- phase 2
    # -------------------------------------------------------------- phase 4
    def on_scene_ready(self, pipeline: ScenePipeline) -> None:
        """Let it settle, then record what a reset returns to."""
        self.bridge = host_bridge(pipeline, self.name)
        state = pipeline.state_0
        self.rest = {
            "particle_q": state.particle_q.numpy().copy(),
            "particle_qd": state.particle_qd.numpy().copy(),
        }
        # Every world's probe, not just the first: this class re-implements
        # `ParticleMixin.snapshot_rest` inline rather than calling it, so it did
        # not get that method's fix and worlds 1..W-1 would have had no baseline
        # to measure against. Unreachable today -- the compiler refuses to
        # replicate a scene with this entity in it, since a foreign engine
        # builds one scene from the whole model -- and cheap to keep right for
        # whoever lifts that refusal.
        for probe in self.probes or ([self.probe] if self.probe is not None else []):
            if probe is not None:
                probe.update(state)
                probe.snapshot_baseline()

    # -------------------------------------------------------------- phase 5
    # ---------------------------------------------------------------- runtime
    def positions(self, source=None) -> np.ndarray:
        """The centerline now, ``[n, 3]``, in order along the cable.

        The **same accessor name and the same shape** as
        :meth:`~robosimstudio.domains.rod.entity.EntityRod.positions`, and
        deliberately: what the ``rod`` substrate means is "an ordered sequence
        of positions along a length", and a measure over one -- winding turns,
        straightness, a point at arc length 0.6 -- is a statement about that
        sequence, not about where it is stored. The Newton implementation reads
        its capsule centres out of ``body_q``; this reads its nodes out of
        ``particle_q``. Neither of those is the substrate.

        **The caller passes the instant, and that is not decoration.** Reading
        the model's own array instead looks right and returns the *build-time*
        layout forever: a cable that has fallen half a metre and come to rest
        reports itself still hanging where it was declared, which is a plausible
        number and completely wrong. Measured while writing this -- the rod was
        simulating correctly the whole time and the accessor was not.

        Args:
            source: A :class:`newton.State`, or a ``particle_q`` array, or
                ``None`` for the model's own -- which is the as-built layout and
                is right only before anything has stepped.
        """
        if not len(self.particles):
            return np.zeros((0, 3), dtype=np.float64)
        if source is None:
            if self.model is None:
                return np.zeros((0, 3), dtype=np.float64)
            q = self.model.particle_q.numpy()
        elif hasattr(source, "particle_q"):
            q = source.particle_q.numpy()
        else:
            q = np.asarray(source)
        return np.asarray(q, dtype=np.float64)[self.particles]

    # ---------------------------------------------------------------- holding
    def hold(
        self, on: bool, state=None, *, arm: str = "right", slot: int = 0, at: int | None = None, world: int = 0
    ) -> int:
        """Pin the node nearest ``arm``'s jaws through the engine, riding the hand.

        The particle answer to :meth:`~robosimstudio.domains.rod.centerline.Centerline.hold`.
        Where the capsule chain's latch is a kernel pinning a *body* to a
        gripper frame, this asks the foreign engine for a compliant node
        constraint and has the engine carry it on the hand's body
        (``bridge.ride``): re-aimed before every engine substep from the hand's
        pose swept through the control step, as the mirrored fingers are, so the
        node keeps up with the hand rather than trailing it by a step.
        Returns the global node index taken, or ``-1``.
        """
        if world:
            # The contract carries the world so a capsule chain in a wave can latch
            # per world; the particle rod pins through one engine scene and has no
            # per-world pin yet. Refusing is honest; pinning world 0 would not be.
            raise NotImplementedError(f"rod {self.name!r} on a host engine holds in world 0 only, not world {world}")
        if not on:
            entry = self._pins.pop(slot, None)
            if entry is not None and self.bridge is not None:
                self.bridge.unpin(entry[0])
            return -1
        if self.bridge is None:
            raise RuntimeError(f"rod {self.name!r} is not on a host engine; nothing to pin through")
        if state is None:
            raise ValueError("hold(True) needs the instant, to find the jaws")
        body_q = np.asarray(state.body_q.numpy() if hasattr(state.body_q, "numpy") else state.body_q)
        grip, hand_pos, _hand_quat = self.hand_tcp(arm, body_q)
        nodes = self.positions(state)
        local = self.nearest_to(nodes, hand_pos, at=at)
        if local < 0:
            return -1
        node = int(self.particles[local])
        from robosimstudio.core.math import quat_to_mat

        body = np.asarray(body_q[grip], dtype=np.float64)
        if slot in self._pins:
            self.hold(False, slot=slot)
        pin = self.bridge.pin_nodes(self.name, np.array([node]), nodes[local][None, :].copy())
        self.bridge.ride(pin, grip, (quat_to_mat(body[3:7]).T @ (nodes[local] - body[:3]))[None, :])
        self._pins[slot] = (int(pin), node)
        return node

    def holding(self, slot: int = 0) -> bool:
        """Whether ``slot`` holds a node -- asked of the engine, so a reset reads as letting go.

        **A reset lets go, as on the capsule chain, and nothing re-takes the
        node.** The capsule chain's reset zeroes every latch -- "a new episode
        starts with nothing grasped" -- and one ``RodSpec`` is one cable
        whichever engine steps it, so a studio Reset lets go of a held cable on
        both. The engine's own reset already removes every pin, as a constraint
        the rest it restores never had. And a hold is an offset from where the
        hand *was*: the reset sends the hand home, and a pin re-made from that
        offset would drag the node across the scene to it. A caller that wants
        the cable in the hand from the start takes it again after the reset --
        on the first step, as ``examples/09_demos/01_route_the_cable.py`` does.

        Asked rather than remembered because this rod's record of its pins
        outlived the engine's: ``held`` read 1.0 after a reset with nothing
        held. A slot the engine let go of is forgotten here.
        """
        entry = self._pins.get(slot)
        if entry is None or self.bridge is None:
            return False
        if not self.bridge.has_pin(entry[0]):
            del self._pins[slot]
            return False
        return True

    def published_states(self, snapshot) -> dict[str, float]:
        """The same two flags the capsule chain publishes, so a goal reads either."""
        del snapshot
        return {"held": float(self.holding(0)), "anchor_held": float(self.holding(1))}

    def centerline(self, snapshot) -> np.ndarray:
        """This centerline's nodes, in order -- this implementation's :class:`Centerline`.

        Off the probe rather than off the model, for the same reason
        :meth:`positions` takes an instant: the model's array is the as-built
        layout and never moves. The probe is filled once per control step by
        whoever is driving the scene.

        Indexed by this cable's own nodes. ``ParticleProbe.positions`` is the
        **whole model's** particle array -- one probe per entity, each holding
        every particle in the scene -- so returning it raw would hand a rod
        measure a towel's vertices as well, in whatever order the builder
        happened to lay them out. Order is the entire content of this layout, so
        that is not a small error.
        """
        if not len(self.particles):
            return np.zeros((0, 3), dtype=np.float64)
        # Two snapshot shapes reach this. A probe snapshot carries one probe per
        # entity under ``particles``; a ``HostSnapshot`` carries the whole
        # model's ``particle_q`` and no ``particles`` at all -- and reading the
        # attribute unguarded raised on every read of that kind, which took
        # every ``Centerline`` measure down with it on this engine.
        probes = getattr(snapshot, "particles", None)
        if probes is not None:
            probe = probes.get(self.name)
            if probe is None:
                return np.zeros((0, 3), dtype=np.float64)
            return np.asarray(probe.positions, dtype=np.float64)[self.particles]
        q = getattr(snapshot, "particle_q", None)
        if q is None or not len(q):
            return np.zeros((0, 3), dtype=np.float64)
        return np.asarray(q, dtype=np.float64)[self.particles]

    def update_probe(self, state: newton.State, step: int | None = None, *, readback=None) -> ParticleProbe:
        """Read this step's node positions. Once per control step."""
        if self.probe is None:
            raise RuntimeError(f"rod {self.spec.name!r} is not bound yet")
        if readback is not None and getattr(readback, "particle_q", None) is not None and len(readback.particle_q):
            self.probe.update_arrays(readback.particle_q, readback.particle_qd, step)
        else:
            self.probe.update(state, step)
        return self.probe
