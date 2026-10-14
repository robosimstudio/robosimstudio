# SPDX-License-Identifier: Apache-2.0
"""What a compiled scene knows about itself.

:class:`SceneHandles` is the single place anything downstream looks up "where is
the hammer", "which shapes are the left gripper's", "which particles are the
tab", "what should the segmentation id of the bin be". Evaluators, ground-truth
tools, renderers, recorders and teleop panels all read it, and none of them
walks the model or re-derives an index.

That concentration is what keeps the core/toolkit seam honest. Core's mechanisms
take explicit data -- index arrays, part lists, shape-to-id lookups -- and this
is where the toolkit layer produces it. A renderer never learns what a bin is; it is handed
a table saying shape 42 is label 3.

Handles carry a ``world`` index throughout. Scenes are single-world today
because the in-process graph runtime is; batching is the documented growth path
and the indices are shaped for it from the start rather than retrofitted.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

import newton
import numpy as np

from robosimstudio.core.render.finish import Finish
from robosimstudio.solvers import ObbGrasp

__all__ = ["ArmHandle", "DeformableHandle", "EntityHandle", "RegionHandle", "SceneHandles"]


@dataclass
class EntityHandle:
    """One rigid entity's place in the model.

    Attributes:
        name: The spec's name.
        category: Its category, for goals written against categories.
        body: Index into ``State.body_q``, or -1 for something bodiless.
        coord_start: First free-joint coordinate, or -1 if it has no free joint.
        shapes: Every shape on the body.
        visual_shapes: Those that are drawn.
        collision_shapes: Those that collide. The split matters: a boxified tool
            collides as a box and is drawn as a mesh, so a segmentation LUT
            built from collision shapes labels pixels nothing draws.
        obb: The grasp box, or ``None``.
        bbox: Bounding-box extents [m], for predicates that need one.
        bodies: Every body it owns, in order, when it owns more than one -- a
            chain's segments. Empty for the ordinary case of one body, and
            :attr:`body` is the first either way, so everything reading a single
            body keeps reading one.
        static: Whether it is welded to the world.
        xform: Where it sits, for something welded and therefore **bodiless**.
            A free body's pose is read from ``body_q`` and this stays identity;
            a static's is not in ``body_q`` at all, so anything measuring against
            a fixture -- a bore on a receptacle, a slot on a bracket -- has
            nowhere else to get it. The regions already resolved this at compile
            time for their own frames; this is the same fact, for features.
    """

    name: str
    category: str = ""
    body: int = -1
    bodies: tuple[int, ...] = ()
    coord_start: int = -1
    shapes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    visual_shapes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    collision_shapes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    xform: tuple[float, float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    obb: ObbGrasp | None = None
    bbox: np.ndarray | None = None
    static: bool = False
    world: int = 0

    def pose(self, body_q: np.ndarray) -> np.ndarray:
        """World pose as ``(px, py, pz, qx, qy, qz, qw)``.

        A static that owns no body carries ``body=-1`` and its pose in
        :attr:`xform`; indexing ``body_q[-1]`` would read the last body's.
        """
        if self.body < 0:
            return np.asarray(self.xform, dtype=np.float64)
        return np.asarray(body_q[self.body], dtype=np.float64)

    def position(self, body_q: np.ndarray) -> np.ndarray:
        return self.pose(body_q)[:3]


@dataclass
class ArmHandle:
    """One arm's DOFs, tool-centre point and finger shapes.

    Attributes:
        name: The arm's name.
        arm_coords: ``joint_q`` indices of the actuated joints.
        arm_dofs: ``joint_qd`` indices of the same.
        gripper_coords: The finger joints' ``joint_q`` indices.
        gripper_dofs: The same in ``joint_qd``.
        ee_body: Tool-centre body.
        base_body: The body this arm is mounted on -- the spec's, else the
            body its first joint hangs on -- or -1 when that is the world.
            What a base-frame solver and a Cartesian action expressed in the
            robot's own axes read their frame from.
        ee_offset: Tool-centre point in that body's frame [m].
        ee_offset_rot: Rotation from that body's frame to the **grasp** frame,
            xyzw -- the frame whose +z is the approach direction, which is the
            convention every core solver and every commanded pose speaks.

            Carried here because leaving it off cost a whole robot. The
            Cartesian servo measures where the hand is by reading ``ee_body``'s
            transform and composing this on; with no such field to read it
            composed identity and measured the **raw wrist** instead. On a hand
            whose grasp frame is a half-turn from its wrist -- the i2rt YAM's
            is, because the gripper extends along body -z -- the measured
            approach axis then points backwards, the rotation error reads about
            180 degrees on a perfectly aimed hand, and every ``go_to_pose``
            spends its entire step budget before giving up quietly. Invisible
            on the two robots whose grasp frame *is* their wrist, which is
            exactly why it went unnoticed until a third one ran.
        approach_axis: Gripper-local axis from the wrist towards the grasp
            point. Measured, not assumed.
        close_axis: Gripper-local axis the jaws close along, measured from the
            pad centres.
        gripper_open: Finger travel at full open [m]. For a geared hand this is
            the driven joint's travel, which is an angle -- read
            :attr:`gripper_span` for a distance.
        gripper_span: Fingertip separation at full open [m].
        gripper_open_q: Per driven finger joint, the coordinate at full open.
            Empty means the mirrored prismatic convention.
        gripper_closed_q: The same at full closed.
        gripper_geared: The hand's travel is not proportional to its pad gap
            (:attr:`GripperSpec.geared`), so :meth:`jaw_gap` refuses it.
        finger_groups: Per side, ``(body, shapes)`` pairs for grasp tests. A
            robot whose fingertip is a separate body lists both.
        shapes: Every shape belonging to **this arm only** -- its links,
            fingers and welded hand, from the embodiment index's limb walk.
            A shared base or torso is in neither arm's set, and "ignore the
            whole robot" wants :attr:`SceneHandles.robot_shapes`, not this.
            (It used to alias ``robot_shapes`` on every arm, which made the
            two handles of a bimanual cell identical and arm-vs-arm
            clearance a self-measurement.)
        home_q: The arm joints' home coordinates [rad], in ``arm_coords`` order.
            Carried here so task code can restore a posture without importing a
            constant from an embodiment module -- an import that pins the task
            to one robot and, being strict-zipped against a joint count, breaks
            loudly on any other.

    The four axis and travel fields exist for the same reason ``home_q`` does,
    and the module docstring's claim that a handle is "the single place anything
    downstream looks up" was not true until they did: the studio, the sim tools
    and the connector each re-found the :class:`ArmSpec` by name to read them,
    with a ``spec.arms[0]`` fallback that names a different hand on the two YAM
    cells.
    """

    name: str
    arm_coords: tuple[int, ...] = ()
    arm_dofs: tuple[int, ...] = ()
    gripper_coords: tuple[int, ...] = ()
    gripper_dofs: tuple[int, ...] = ()
    ee_body: int = -1
    base_body: int = -1
    ee_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ee_offset_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    approach_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    close_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    gripper_open: float = 0.04
    gripper_span: float = 0.08
    gripper_open_q: tuple[float, ...] = ()
    gripper_closed_q: tuple[float, ...] = ()
    gripper_geared: bool = False
    home_q: tuple[float, ...] = ()
    finger_groups: tuple[tuple[tuple[int, np.ndarray], ...], ...] = ()
    open_axis_signs: tuple[float, ...] = (1.0, -1.0)
    shapes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    world: int = 0

    def _finger_limits(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        """Per driven finger coordinate, its value at full open and at full closed."""
        if self.gripper_open_q:
            open_q = np.asarray(self.gripper_open_q, dtype=np.float64)
        elif count == 2:
            # The mirrored prismatic convention, as the Gripper drives it.
            open_q = np.array([self.gripper_open, -self.gripper_open])
        else:
            open_q = np.full(count, float(self.gripper_open))
        closed_q = (
            np.asarray(self.gripper_closed_q, dtype=np.float64) if self.gripper_closed_q else np.zeros(len(open_q))
        )
        return open_q, closed_q

    def closing_travel(self, finger_q) -> float:
        """How far the driven fingers have closed from full open, mean over fingers.

        In joint units, and positive towards closed **whichever way the joints
        run**: a mirrored prismatic pair closes towards zero from ``+/-open``,
        a Robotiq 2F-85's driver opens at zero and closes upward. Reading a raw
        coordinate as "smaller is more closed" gets the second backwards.

        Args:
            finger_q: The coordinates at :attr:`gripper_coords`, in that order.
        """
        q = np.asarray(finger_q, dtype=np.float64).reshape(-1)
        open_q, closed_q = self._finger_limits(len(q))
        sign = np.where(closed_q >= open_q, 1.0, -1.0)
        return float(np.mean((q - open_q) * sign))

    def jaw_gap(self, finger_q) -> float:
        """Pad separation at the driven fingers' coordinates [m].

        The same derivation :meth:`GripperSpec.jaw_gap` makes, from the numbers
        the compiler copied onto this handle -- so a measure that has a scene
        does not have to go back to the registry for the spec that built it --
        with the travel read direction-aware (:meth:`closing_travel`).

        Args:
            finger_q: The coordinates at :attr:`gripper_coords`, in that order.

        Raises:
            ValueError: On a geared hand, as :meth:`GripperSpec.jaw_gap` does:
                its pad separation is not linear in finger travel.
        """
        from robosimstudio.specs import jaw_gap_from

        if self.gripper_geared:
            raise ValueError(
                f"arm {self.name!r} has a geared hand, so pad separation is not linear in finger travel. "
                "A gap for this hand needs a measured curve, not this derivation."
            )
        return jaw_gap_from(
            span=self.gripper_span,
            open_width=self.gripper_open,
            fingers=len(self.gripper_open_q) or len(self.gripper_coords) or 2,
            joint_q=self.gripper_open - self.closing_travel(finger_q),
        )


@dataclass
class DeformableHandle:
    """One deformable's particles and named sets.

    Attributes:
        name: The spec's name.
        particles: Global particle indices.
        sets: ``"<cloth>/<set>" -> indices``.
        triangles: Triangles into a local vertex numbering, for renderers.
    """

    name: str
    particles: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    sets: dict[str, np.ndarray] = field(default_factory=dict)
    triangles: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int32))
    world: int = 0
    tube_radius: float = 0.0
    """For a centerline -- a rope whose particles are its nodes and which owns
    no triangle -- the radius a renderer sweeps a tube of. 0 for anything with a
    surface of its own."""


@dataclass
class RegionHandle:
    """A goal region resolved into world coordinates.

    The frame origin is looked up once at compile time from whatever the region
    is attached to. That is correct for a bin welded into the arena and wrong
    for a region on something that moves, so :meth:`origin_now` re-reads it from
    a live ``body_q`` when the frame has a body.

    **Where the box is, not what being in it means.** This carried a
    ``contains()`` answering the second question, and a region's *meaning* is the
    ``containment`` atom's -- which is what every goal binds to, and which
    disagreed with this one on the boundary faces for as long as both existed.
    One measure, one implementation, in the layer that owns the vocabulary:
    ``assetlib.vocabulary.regions``. What is left here is geometry, which is what a
    handle is for.
    """

    name: str
    frame: str
    frame_body: int = -1
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    xy_ranges: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    z_range: tuple[float, float] = (0.0, 0.15)
    world: int = 0

    def origin_now(self, body_q: np.ndarray | None = None) -> np.ndarray:
        """The frame origin, re-read if the frame is a moving body."""
        if self.frame_body >= 0 and body_q is not None:
            return np.asarray(body_q[self.frame_body][:3], dtype=np.float64)
        return self.origin

    def center(self, body_q: np.ndarray | None = None) -> np.ndarray:
        """The region's centre in world coordinates [m], at the frame origin's height."""
        origin = self.origin_now(body_q)
        x0, y0, x1, y1 = self.xy_ranges
        return origin + np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5, 0.0])


@dataclass
class SceneHandles:
    """Everything a compiled scene knows about where things are.

    Attributes:
        objects: Manipulable bodies by name.
        statics: Scenery by name.
        arms: Arms by name.
        deformables: Deformables by name.
        regions: Goal regions by name.
        robot_shapes: Every shape belonging to the embodiment, for exclusions.
        meta: Whatever the scene's JSON sidecars carried.
        world: Which world these indices are for.
        scene_name: Which registered scene this was staged in, or ``""``.
        surface_z: World z of the work surface [m]. Read by anything that needs
            "the height things rest at" -- which used to be a module constant
            imported from an *embodiment*, and is a fact about the place.
        work_arm: Name of the arm that does the task, or ``""`` if there is no
            embodiment. Read this rather than a ``"right"`` literal: the two YAM
            cells order their arms oppositely, so a literal means different
            hands on different robots.
        support_arm: Name of the arm that holds or anchors, or ``""`` when the
            task needs only one.
        finishes: ``shape index -> Finish`` for every shape in the model -- what
            each surface does with light, beside the colour ``model.shape_color``
            holds. Complete by construction: the compiler fills the gaps an
            entity's own painting left (an appearance hook's shells, the ground
            plane) from the colour those shapes carry. Read by the path-traced
            backend the way ``segmentation_labels`` is read by the tiled one.
        deformable_finishes: ``deformable name -> Finish`` for cloth, rods'
            sheets and soft bodies, which are not shapes and so have no row in
            :attr:`finishes`.
        live_shapes: Shapes whose colour a running scene rewrites -- a mark
            being polished off, a seam parting -- so a backend that bakes colour
            at load knows which few to re-read each frame. See
            :attr:`~robosimstudio.scene.entity.Entity.recolors_live`.
        active: The entity names this table was :meth:`restricted` to, or
            ``None`` for the whole model's. A reader that must know whether a
            name is *absent* or *parked* asks here.
    """

    objects: dict[str, EntityHandle] = field(default_factory=dict)
    statics: dict[str, EntityHandle] = field(default_factory=dict)
    arms: dict[str, ArmHandle] = field(default_factory=dict)
    deformables: dict[str, DeformableHandle] = field(default_factory=dict)
    regions: dict[str, RegionHandle] = field(default_factory=dict)
    robot_shapes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    meta: dict = field(default_factory=dict)
    world: int = 0
    scene_name: str = ""
    surface_z: float = 0.0
    work_arm: str = ""
    support_arm: str = ""
    finishes: dict[int, Finish] = field(default_factory=dict)
    deformable_finishes: dict[str, Finish] = field(default_factory=dict)
    live_shapes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    active: frozenset[str] | None = None

    # ------------------------------------------------------------- restriction
    def restricted(self, active: Iterable[str] | None) -> SceneHandles:
        """This table with only the named objects, everything else shared.

        What one episode of a superset model sees: the objects it runs, the
        regions whose frame is one of them or is scenery, and every other
        table -- statics, arms, deformables, finishes, the robot's shapes --
        the same objects as this one's, because those are facts about the
        model and the model is shared. ``None`` is the whole table, returned as
        is, so a caller that always restricts pays nothing for the common case.

        **Only free bodies are subject to the set.** An object with no free
        joint (``coord_start < 0``: a fixture, a bin the arena declared) is
        welded in every world and stays listed whether or not the set names
        it -- it cannot be parked, and a region on it must go on resolving.
        The same rule :func:`~robosimstudio.scene.parking.parked_objects`
        parks by, so what this table lists and what the world holds agree.

        A parked object's *body* is still in the model and still in
        ``body_q``; what this removes is every way of finding it by name, which
        is what the evaluator, the observation, the segmentation LUT and the
        privileged tools enumerate. A region on a parked object goes with it:
        ``origin_now`` would otherwise re-read a frame six metres in the air.

        Dictionaries are new; handles are shared. Nothing here mutates a
        handle, and a reader holding one from the full table holds the same
        object.
        """
        if active is None:
            return self
        keep = frozenset(str(name) for name in active)
        objects = {name: handle for name, handle in self.objects.items() if name in keep or int(handle.coord_start) < 0}
        dropped = {name for name in self.objects if name not in objects}
        regions = {name: region for name, region in self.regions.items() if region.frame not in dropped}
        return replace(self, objects=objects, regions=regions, active=keep)

    # -------------------------------------------------------------------- roles
    def arm(self, role: str = "work") -> ArmHandle:
        """The arm filling one role (``"work"``, ``"support"``), or named directly.

        The role indirection is what makes task code portable -- it asks for the
        hand that does the job, and the binding says which hand that is on this
        robot. A literal arm name (``"left"``, ``"right"``, ...) is resolved
        straight to that arm instead, which is what :class:`~robosimstudio.core.
        cameras.CameraSpec`'s ``parent_frame`` promises for a rig that wants a
        *specific* arm regardless of which role it plays on a given document --
        without this branch, any name other than ``"work"`` fell through to
        ``support_arm``, so two cameras named for the two different arms both
        rode the support arm's body.
        """
        if role in self.arms:
            return self.arms[role]
        name = self.work_arm if role == "work" else self.support_arm
        if not name:
            raise KeyError(f"this scene has no {role} arm; arms are {sorted(self.arms)}")
        return self.arms[name]

    # ------------------------------------------------------------------ lookup
    def find(self, name: str) -> EntityHandle | DeformableHandle | RegionHandle | None:
        """The handle under one name, whichever table holds it, or ``None``.

        The one search over the named tables. ``Ref.handle`` and
        ``Sim.__getitem__`` each walked the four in turn with their own copy of
        this loop, which read as proof the tables should be one map with the
        kind on the handle. The consumers say otherwise: of the fifty-odd sites
        that read a table, three look a bare name up without knowing its kind
        (this, and :meth:`entity`), and the rest either address a kind by name
        -- ``arms[name]`` is driven, ``regions[name]`` is only read -- or iterate
        one table for everything of that kind: every object's pose for an
        observation, every deformable's mesh for a renderer, every region for
        the overlay. A single map would hand each of those a filter. So the
        tables stay, and the search is written once.

        Arms are not searched. An arm is addressed by role through :meth:`arm`,
        and a bare name has never resolved to one.
        """
        for table in (self.objects, self.deformables, self.statics, self.regions):
            if name in table:
                return table[name]
        return None

    def entity(self, name: str) -> EntityHandle:
        """An object or a static, by name."""
        if name in self.objects:
            return self.objects[name]
        if name in self.statics:
            return self.statics[name]
        raise KeyError(f"no entity {name!r}; objects {sorted(self.objects)}, statics {sorted(self.statics)}")

    def objects_of_category(self, category: str) -> list[EntityHandle]:
        """Every manipulable object of one category."""
        return [h for h in self.objects.values() if h.category == category]

    def particle_set(self, name: str) -> np.ndarray:
        """A named particle set from whichever cloth owns it."""
        for cloth in self.deformables.values():
            if name in cloth.sets:
                return cloth.sets[name]
            key = f"{cloth.name}/{name}"
            if key in cloth.sets:
                return cloth.sets[key]
        raise KeyError(f"no particle set {name!r}")

    # ------------------------------------------------------------------ tables
    def segmentation_labels(self, *, background: int = 0) -> dict[int, int]:
        """``shape index -> label id`` for a segmentation render.

        Ids are assigned in a stable order -- statics, then objects, then the
        robot -- so a recorded segmentation means the same thing across runs of
        the same task. The mapping itself is returned alongside through
        :meth:`segmentation_names`, since an id with no name is not an
        annotation.
        """
        del background
        lut: dict[int, int] = {}
        for label_id, name in enumerate(self._segmentation_order(), start=1):
            handle = self.entity(name) if name in self.objects or name in self.statics else None
            shapes = handle.shapes if handle is not None else self.robot_shapes
            for shape in shapes:
                lut[int(shape)] = label_id
        return lut

    def segmentation_names(self) -> dict[int, str]:
        """``label id -> name``, the key to :meth:`segmentation_labels`."""
        return dict(enumerate(self._segmentation_order(), start=1))

    def _segmentation_order(self) -> list[str]:
        return [*sorted(self.statics), *sorted(self.objects), "__robot__"]

    def summary(self) -> str:
        """One line per group, for a build log."""
        lines = [
            f"objects: {', '.join(sorted(self.objects)) or '-'}",
            f"statics: {', '.join(sorted(self.statics)) or '-'}",
            f"arms:    {', '.join(f'{n} ({len(a.arm_coords)} dof)' for n, a in sorted(self.arms.items())) or '-'}",
        ]
        if self.deformables:
            lines.append(
                "deformables:  "
                + ", ".join(f"{n} ({len(c.particles)} particles)" for n, c in sorted(self.deformables.items()))
            )
        if self.regions:
            lines.append(f"regions: {', '.join(sorted(self.regions))}")
        return "\n".join(lines)


def split_shape_flags(model: newton.Model, shapes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a shape set into the drawn ones and the colliding ones."""
    if not len(shapes):
        return shapes, shapes
    flags = model.shape_flags.numpy()
    idx = np.asarray(shapes, dtype=np.int32)
    visible = idx[(flags[idx] & int(newton.ShapeFlags.VISIBLE)) != 0]
    colliding = idx[(flags[idx] & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0]
    return visible.astype(np.int32), colliding.astype(np.int32)
