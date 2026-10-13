# SPDX-License-Identifier: Apache-2.0
"""Newton shapes as SuperDex collider actors.

The deformable never lives in two engines, but the *boundary* it presses
against does: whatever rigid geometry the contact rules say the soft island
touches -- gripper pads, a bench top, the ground plane -- is rebuilt here as a
static SuperDex rigid actor with an SDF collider, and then driven by pose
alone. Static-with-a-moving-transform is SuperDex's kinematic idiom: the
engine freezes the stage-start transform each step, so consecutive
``set_root_transform`` calls read as swept motion and friction sees the right
surface velocity.

Everything is derived from the finalized ``newton.Model`` shape arrays, never
from specs -- the same rule the soft actors follow, so a repro bundle can
rebuild the mirrors from the model and the manifest alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robosimstudio.engines.geometry import ShapeGeometry, iter_mirrors

__all__ = ["MirrorActor", "build_mirror_actors", "make_collider_shape"]

_XYZW = slice(3, 7)
_POS = slice(0, 3)


def make_collider_shape(physics, geom: ShapeGeometry):
    """One :class:`~robosimstudio.engines.geometry.ShapeGeometry` as a SuperDex
    shape handle, or ``None`` to skip.

    Planes come back as implicit planes and spheres as implicit spheres --
    SuperDex evaluates both exactly. Everything else takes the tessellation and
    is left to the SDF collider bake: contact quality then depends on the SDF
    resolution rather than on how faithful the triangles are, which is why a
    coarse capsule is fine.
    """
    if geom.kind == "plane":
        # Local frame: the plane is z=0 with +z outward. World placement is the
        # actor's transform, so the implicit plane is stated in local terms.
        return physics.create_plane_shape(normal=[0.0, 0.0, 1.0], distance=0.0)
    if geom.kind == "sphere":
        return physics.create_sphere_shape(center=[0.0, 0.0, 0.0], radius=float(geom.scale[0]))

    mesh = geom.mesh()
    if mesh is None:
        return None
    vertices, faces = mesh
    return physics.create_tri_mesh_shape(
        coordinates=np.asarray(vertices, dtype=np.float64).ravel(),
        connectivity=np.asarray(faces, dtype=np.int32).ravel(),
    )


@dataclass
class MirrorActor:
    """One Newton shape mirrored into the SuperDex scene.

    Attributes:
        shape: Newton shape index.
        body: Newton body index, ``-1`` for world-fixed.
        actor: The SuperDex actor.
        local: The shape's body-local transform ``[7]``, xyz + xyzw.
        two_way: Whether contact wrenches against this mirror are harvested
            back onto the Newton body.
        world: Which world of a replicated model this collider belongs to.

            0 when nothing replicated it. Above that it decides *which scene*
            the collider is built in: a replicated model superimposes its worlds
            at the origin, which a foreign engine cannot pair contacts within,
            so each world has a scene of its own. One Newton shape can therefore
            be mirrored more than once -- a world-fixed bench is one shape and W
            colliders, one per scene.
    """

    shape: int
    body: int
    actor: object
    local: np.ndarray
    two_way: bool = False
    world: int = 0
    prev: np.ndarray = field(default_factory=lambda: np.zeros(7))

    @property
    def moving(self) -> bool:
        return self.body >= 0


def build_mirror_actors(
    physics,
    scenes,
    model,
    *,
    shapes: list[dict],
    layer: str = "mirror",
    contact_smoothing: float = 5.0e-4,
    friction_falloff: float = 1.0e-2,
    contact_threshold: float | None = None,
    contact_padding: float = 0.0,
    exact_boxes: bool = True,
) -> list[MirrorActor]:
    """Create one static SuperDex actor per mirrored Newton shape.

    Args:
        scenes: One engine scene, or a sequence of them indexed by world -- a
            record's ``world`` picks which it is built in.
        shapes: Mirror records -- ``{"shape": int, "body": int, "two_way":
            bool}`` -- exactly what the manifest's host-island record carries.
        layer: Contact layer the mirrors live on.
        contact_smoothing: ``penalty_smoothing_half_distance`` [m] on every
            mirror. **The collider's value is the one that governs the pair**:
            setting it on the deformable actor instead changes nothing
            measurable, which is why it belongs here and not on ``SoftSpec``.

            The engine ships 5 mm, which is wider than the ~3 mm at which the
            penalty engages, so a contact stays on the soft part of the ramp
            for its whole useful range. Measured on a fingertip pinch of a
            45 mm foam block, jaws commanded 6 mm past its faces:

            ======  ==========  =========  ======
            value   deepest gap  peak dent  grip
            ======  ==========  =========  ======
            5.0 mm  -2.060 mm    1.23 mm    1.28 N
            2.0 mm  -0.646 mm    1.72 mm    2.31 N
            1.0 mm  -0.032 mm    1.96 mm    2.74 N
            0.5 mm  +0.000 mm    2.03 mm    3.00 N
            ======  ==========  =========  ======

            At the engine's default the pads sit **two millimetres inside the
            foam** and press it with 700 Pa; at 0.5 mm they rest exactly on the
            surface and press it properly. ``penalty_coefficient`` barely
            enters into it -- 5e8 and 2e9 land on the same numbers at the same
            smoothing -- so this, not stiffness, is the knob.
        friction_falloff: ``friction_falloff_vel`` [m/s] on every mirror: the
            sliding speed at which the regularised Coulomb friction reaches
            its full ``mu * N``. Below it friction is proportional to speed,
            so there is **no static friction**: a held object creeps at
            roughly ``load / (mu * N) * friction_falloff``.

            **It does not reach a contact.** Every mirror is a static actor,
            and against a static collider SuperDex uses the *colliding* body's
            falloff, not the collider's (its ``ContactParams`` docs say so).
            Measured on 2026-10-13: a Panda pinching a 50 mm sponge at 4 N a
            finger slid it 13.9-14.0 mm in the first 1.5 s of a hold and dropped
            it, with its pads' mirrors at 10, 1 and 0.1 mm/s alike; with the
            sponge's own falloff at 1 mm/s it held, creeping 1.8 mm in 2.5 s.
            The falloff that decides a hold is the deformable's:
            ``SoftMaterialSpec.friction_falloff``, ``RodSpec.friction_falloff``.
            What follows was recorded earlier, with this value, and is kept as
            it was recorded. The engine ships
            10 mm/s, and at that value a 9 g sponge pinched with 4-6 N per pad
            at mu 1.0 -- friction capacity forty times its weight -- slid down
            the pads at 0.4-2.5 mm/s through a stationary hold and dropped in
            about a second, in the serial env and in a wave alike; the same
            recorded joint targets replayed with 1 mm/s held it at the tool
            centre to within a millimetre, and 0.1 mm/s did the same. This is
            why the surface-preparation suite's routine pins the foam to the pads: a
            friction-only grasp could not hold it. 1 mm/s makes that grasp
            static for the seconds a pick lasts (a 0.09 N load against 4 N
            creeps ~0.02 mm/s) without changing the solver's convergence.

            **The default is the engine's 10 mm/s all the same**, and the reason
            is the other interface. Static friction on every mirror means the
            bar holds a pressed sponge with the full press force while the pads
            hold it with only the pinch force, so the moment a wipe presses
            harder than it pinches the sponge sticks to the bar and slides out
            of the jaws instead of along the bar: the surface-preparation seed policy,
            which wiped 0.4-1.0 of the stripe in half its serial episodes at
            10 mm/s, fell to 0-0.6 at 1 mm/s in the serial env and to at most
            0.38 in a wave, with the foam extruding upward between the pads at
            the first press. A stroke at 300 mm/s is at the Coulomb cap either
            way; what the value decides is which interface yields first. A
            task that wants a static pinch AND a pressed wipe needs the two
            interfaces told apart, not one number for both.

        contact_threshold: ``penalty_threshold_default`` [m] on every mirror,
            or ``None`` for the engine's 1 mm. **The collider's threshold is the
            one that governs a pair**, and a mirror is always the collider, so
            this is the stand-off at which every gripper pad, bench and fixture
            starts pushing: measured, a free block rests 0.98 mm above a box at
            the default and 0.000 mm at 2e-5. A sponge does not mind; a plug in
            a jack with 0.15 mm of clearance is squeezed by every wall at once.
        contact_padding: ``penalty_threshold_extra_padding`` [m] on every
            mirror: a stand-off the engine adds to the threshold **only against
            a point-cloud (or collider-less) actor**, which is what a rod is.
            0, the engine's own, leaves the mirrors as they were. A rod built
            without its skin (``RodSpec(skin=False)``) samples contact on its
            axis, so it rests with its axis on whatever it touches: measured, a
            5 mm cable's axis 0.93 mm over the bench, its lower half inside the
            wood, and a clip's 3 mm prong 2.6 mm from it -- inside the prong.
            Padded by the radius less the engine's own 1 mm threshold (1.5 mm),
            the axis rests 2.43 mm up and stays 4.4 mm from a prong's axis:
            the cable's surface on every mirror -- bench, fixtures and fingers
            alike -- for a third of the skin's step cost (3.4 ms against 9.9 ms
            a control step at rest). A skinned rod (the default) already rests
            on its skin (3.07 mm for the same cable) and padding would float it,
            which is why this is an option and not the rods' default.
        exact_boxes: Mirror a Newton box as the engine's own ``BOX`` collider
            (its local AABB, which for a box *is* the box) rather than an SDF
            baked from twelve triangles. The bake's default voxel is a quarter
            of the mean edge -- 4 mm for a 20 x 10 x 5 mm pad -- so its corners
            and thin faces were whatever the grid made of them.

    World-fixed shapes are posed once, here. Body-carried shapes are posed at
    every bridge tick and only *placed* here, at their body-local transform --
    the first tick overwrites it before anything can touch the sponge.
    """
    if contact_padding < 0.0:
        raise ValueError(f"mirror contact_padding must not be negative (the engine refuses it), got {contact_padding}")
    per_world = list(scenes) if isinstance(scenes, (list, tuple)) else [scenes]
    out: list[MirrorActor] = []
    for geom in iter_mirrors(model, shapes):
        handle = make_collider_shape(physics, geom)
        if handle is None:
            continue
        local = geom.local
        placed = local.copy()
        if geom.kind in ("plane", "sphere"):
            collider = physics.ColliderType.AUTO
        elif geom.kind == "box" and exact_boxes:
            collider = physics.ColliderType.BOX
        else:
            collider = physics.ColliderType.SDF
        contact = physics.ContactParams(
            coulomb_friction_coefficient=geom.mu,
            penalty_smoothing_half_distance=float(contact_smoothing),
            friction_falloff_vel=float(friction_falloff),
        )
        if contact_threshold is not None:
            contact.penalty_threshold_default = float(contact_threshold)
        if contact_padding:
            contact.penalty_threshold_extra_padding = float(contact_padding)
        actor = per_world[geom.world % len(per_world)].create_rigid_actor(
            name=f"mirror:{geom.shape}" if len(per_world) == 1 else f"mirror:{geom.shape}#w{geom.world}",
            layer=layer,
            shape=handle,
            is_static=True,
            collider_type=collider,
            world_from_local=physics.TransformRT(
                rotation=physics.Quaternion(*[float(v) for v in placed[_XYZW]]),
                translation=[float(v) for v in placed[_POS]],
            ),
            contact=contact,
        )
        out.append(
            MirrorActor(
                shape=geom.shape,
                body=geom.body,
                actor=actor,
                local=local,
                two_way=geom.two_way,
                world=geom.world,
                prev=placed.copy(),
            )
        )
    return out
