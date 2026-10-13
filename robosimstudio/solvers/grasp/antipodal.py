# SPDX-License-Identifier: Apache-2.0
"""Where to grasp a body, found by looking for opposed faces on its own mesh.

The heuristic in :mod:`~robosimstudio.solvers.grasp.obb` answers a different
question and answers it deliberately: it is a transcription of VAB-industrial's
baseline, kept identical so a pose that works here is the pose that works there.
What it is not is a *good* grasp finder, and three of its limits are structural
rather than tunable:

* It boxes a body's shapes in the **body frame**, so the box is only tight when
  the author happened to align the mesh with it, and it grasps at the box centre
  whether or not there is material there. An L-bracket is grasped at the corner
  of the L, which is air.
* Its yaw is ``arctan2`` of the long axis projected into the table plane, which
  is *degenerate for anything upright*: a standing bottle's long axis is world
  vertical, both components are zero, and every such object proposes the same
  0 rad regardless of how it is turned.
* Its orientation is the arm's **home** wrist quaternion yawed about world +z,
  so the roll about the approach direction is whatever the home posture happened
  to have. Every embodiment publishes a *measured* approach axis and closing
  axis for its hand, and the heuristic reads neither.

So this module does the ordinary thing instead. A grasp is two contacts on
opposed surfaces, and the mesh already says where those are: sample its
triangles, keep the pairs whose separation fits between the jaws and whose
normals face each other within a friction cone, and rank what survives. The
width is then *measured* rather than inferred from a box, which is also what
lets the close command be "just inside this object" rather than "all the way".

**The click is a hint, not a filter.** ``near`` biases the search towards where a
person pointed, so clicking a hammer's handle proposes the handle and clicking
its head proposes the head -- one object, several grasps, chosen by pointing at
the part you want. Passing ``near=None`` searches the whole body.

**A pair is only a grasp if the jaws can get to it**, and there are two ways they
cannot. The hand may have nowhere to stand, which :meth:`AntipodalSampler._clearance`
answers by casting a ray back from the tool centre; or the *plates* may be where
something else already is, which is a different question and needs
:meth:`AntipodalSampler._pads_clear`. Only the second one catches a neighbour,
because straddling an object puts the plates outside its silhouette while the
tool centre stays safely in the middle of it -- so on ``astro/wiring_sort``, with
a screwdriver lying alongside a pair of pliers, every grasp the ray called clear
put a plate inside the screwdriver.

Nothing here touches Warp, an arm, or a solver: it is arithmetic over triangles
and a pose out, so reachability stays the caller's problem. The studio filters
the returned candidates through :meth:`~robosimstudio.solvers.ArmIK.solve_best`
because it is the thing that owns an IK.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import newton
import numpy as np

from robosimstudio.core.math import mat_to_quat, quat_from_axis_angle, quat_multiply, quat_to_mat, transform_matrix
from robosimstudio.core.render.visual import shape_meshes

__all__ = ["AntipodalSampler", "GraspCandidate", "PadBox", "grasp_quat"]

UP = np.array([0.0, 0.0, 1.0])
"""World up. Gravity is -Z everywhere in this package."""

_MAX_PAIRS = 400
"""Antipodal pairs carried into the approach search, best-opposed first.

One flat face proposes hundreds of pairs that are all the same grasp, and the
fan and the pad-reach test are paid per pair. Sorting by how opposed the faces
are and taking the front of that list keeps the ones worth the work.
"""
_REACH_POINTS = 800
"""Points of the object the pad-reach test measures against.

Coarse on purpose: it has to notice a can's lid standing between a finger and
its contact, and it does not need every triangle of the lid to do that.
"""

_SKIN = 0.003
"""How far outboard of a contact the object's own surface stops being the contact [m].

The same 3 mm :meth:`AntipodalSampler._reach` uses, and for the same reason: a
contact sits on a triangle whose neighbours are always a hair around it, so a
volume test starting exactly at the contact plane reads the contact itself as an
obstruction.
"""

_LOCAL_RADIUS = 0.025
"""How far around a grasp counts as "here" for :attr:`GraspCandidate.local_width` [m].

Half the jaw span of the narrower shipped hands, which is the right scale
because it is the scale of the thing the pads are about to close on. Wider and a
hammer's claw starts contributing to its handle's cross-section, which is the
whole failure the local measurement exists to avoid; narrower and a pair on a
gently curved face reads less material than the pads will actually meet.
"""

BALANCE_WEIGHT = 3.0
BALANCE_DECAY = 0.015
BALANCE_PERP = True
"""How hard :meth:`AntipodalSampler._score` presses for a grasp over the mass.

The arms of the A/B that set them: ``1.5``, ``0.05`` and ``False`` restore the
term exactly as it stood before -- weak, and measuring the *omnidirectional*
offset rather than the one that turns a held object. They were environment
variables while that A/B ran; a measurement knob that stays is a constant.
"""


@dataclass(frozen=True)
class PadBox:
    """One jaw's plate, in the grasp frame, over the whole of its stroke.

    The grasp frame is the one :func:`grasp_quat` builds: **+z is the approach**,
    so negative z is back towards the wrist; **+x is the closing direction**, so
    the pad's *gripping face* is the low-x face of the ``+`` side's plate; and y
    is the plate's width.

    Every number is a union or an endpoint of a *sweep*, never a snapshot, and
    that is the whole design of this class. A pad is not a plane at the
    fingertips and it does not sit still: a 2F-85's four-bar swings its plate
    about 13 mm along the approach between open and shut, and the distal links
    hang on ``<connect>`` equalities that sag when the hand points down, so the
    same plate measured at the home posture and measured wrist-down differ by
    4 mm. Both traps have already been paid for on this branch -- a clearance
    rule written against a fixed fingertip plane and a home-posture band
    produced a phantom "29 mm of material" on an object that has 52.8 -- so
    :attr:`back` and :attr:`front` are stated at *both* ends of the stroke and
    the test takes the interval spanned between them.

    Attributes:
        half_width: Half the plate across the grasp frame's y [m].
        thickness: How far the plate stands **outboard** of its own gripping
            face along the closing axis [m]. The number that decides this test:
            an inner face parked just outside a neighbour still puts this much
            of a plate inside it.
        back: How far the plate reaches back towards the wrist [m], at full open
            and at full shut. Decreasing, because closing swings it forward.
        front: How far it reaches past the tool centre [m], at the same two
            commands. Increasing, for the same reason.

    The defaults are the Robotiq 2F-85's, measured: the lateral pair off the pad
    colliders in the grasp frame (``x`` 42.5 to 50.6 mm at full open against a
    span of 85, and ``y`` -11.0 to +11.0), and the along-approach pair swept
    under physics with the wrist held down rather than at home. A caller with a
    different hand should measure it the same way and say so; a caller with no
    measurement gets a plate rather than a plane, which is the error worth
    making.
    """

    half_width: float = 0.011
    thickness: float = 0.008
    back: tuple[float, float] = (0.0372, 0.0196)
    front: tuple[float, float] = (0.0010, 0.0102)

    def along_approach(self, opened: float, shut: float) -> tuple[float, float]:
        """The z interval either plate occupies while closing from ``opened`` to ``shut``.

        Both commands are 0 open .. 1 shut. The union is the open end's reach
        back and the shut end's reach forward, because the plate only travels
        one way as the jaws close.
        """
        lo, hi = (min(opened, shut), max(opened, shut))
        return -_lerp(self.back, lo), _lerp(self.front, hi)


def grasp_quat(
    approach_local: Sequence[float],
    close_local: Sequence[float],
    ee_offset_rot: Sequence[float],
    approach_world: Sequence[float],
    close_world: Sequence[float],
) -> np.ndarray:
    """A wrist target, xyzw, that points a hand along an approach and closes across a direction.

    The general form of what the embodiment layer's ``top_down_grasp_site_quat``
    does for the one case of a top-down grasp -- and what that function is now
    written over: map the gripper's *measured* local axes onto two world
    directions and read off the rotation between the two bases.
    Both pairs are orthogonalised against their approach, so a caller may pass a
    closing direction that is merely near-perpendicular -- which is what an
    antipodal pair actually gives you -- and get a valid frame back.

    The result is composed with ``ee_offset_rot`` because
    :class:`~robosimstudio.solvers.ArmIK` states orientation targets in the
    offset frame. That composition is also what makes ``quat_to_mat(result)[:, 2]``
    equal ``approach_world``, which is the convention ``ArmIK.solve_q`` reads its
    approach axis in.

    Args:
        approach_local: Gripper-local axis pointing from the wrist towards the
            grasp point.
        close_local: Gripper-local axis the jaws close along.
        ee_offset_rot: Rotation from the tool body's frame to the grasp frame, xyzw.
        approach_world: Where the hand should point [world].
        close_world: What the jaws should close across [world].

    Returns:
        The wrist orientation as an xyzw quaternion.

    Raises:
        ValueError: If either pair is degenerate -- a zero axis, or a closing
            direction parallel to its approach, which names no frame at all.
    """
    local = _basis(approach_local, close_local, "gripper-local")
    world = _basis(approach_world, close_world, "world")
    return quat_multiply(mat_to_quat(world @ local.T), np.asarray(ee_offset_rot, dtype=np.float64))


def _basis(approach: Sequence[float], close: Sequence[float], what: str) -> np.ndarray:
    """An orthonormal frame whose first column is ``approach`` and second is ``close``."""
    a = np.asarray(approach, dtype=np.float64)
    norm = float(np.linalg.norm(a))
    if norm < 1.0e-9:
        raise ValueError(f"{what} approach axis is zero")
    a = a / norm
    c = np.asarray(close, dtype=np.float64)
    c = c - float(np.dot(c, a)) * a
    norm = float(np.linalg.norm(c))
    if norm < 1.0e-9:
        raise ValueError(f"{what} closing axis is parallel to the approach axis")
    return np.column_stack([a, c / norm, np.cross(a, c / norm)])


@dataclass(frozen=True)
class GraspCandidate:
    """One proposed grasp: where the jaws meet, how they are turned, and how good it looks.

    Attributes:
        position: World point midway between the two contacts [m]. This is the
            tool-centre target, not a stand-off -- the caller backs off along
            :attr:`approach` for its own approach leg.
        quat: Wrist target, xyzw, in the ``ee_offset`` frame.
        approach: World direction the hand comes in along. Equals
            ``quat_to_mat(quat)[:, 2]``, which is what ``ArmIK`` expects.
        close_axis: World direction the jaws close across.
        width: Measured separation of the two contacts [m]. The whole point of
            sampling the mesh rather than boxing it: this is what the object
            actually is between the pads, not what its bounding box is.
        contacts: The two surface points, ``[2, 3]``.
        cone_angle: Worst of the two contacts' deviations from a perfectly
            opposed normal [deg]. 0 is a flat pair of parallel faces.
        clearance: Free length along the approach ahead of the grasp [m], up to
            the stand-off that was tested. Less than that means something is in
            the way.
        local_width: How wide the object *is* across the closing axis here [m],
            measured over its own surface within :data:`_LOCAL_RADIUS` of the
            grasp. The companion to :attr:`width`, and the number that says
            whether a pair is on the outside of the object or inside a recess in
            it: an outer pair spans nearly all of it, a pair in a cavity spans a
            fraction.

            **A whole-body box cannot answer this and a caller using one gets
            elongated objects wrong.** Measured on a tool-sorting scene: the
            sampler's best handle grasps are 30.1-42.5 mm pairs, and the box
            across the same direction is 131.7 mm because the box is sized by
            the claw at the other end -- so a perfectly good handle grasp reads
            as 0.27 of its object and is indistinguishable from a pair inside a
            breadboard's mouldings, which reads 0.11-0.74. Measured *here*, the
            handle's own cross-section is the handle, and the ratio separates.
        score: Higher is better. Only meaningful against candidates from the
            same :meth:`AntipodalSampler.propose` call.
    """

    position: np.ndarray
    quat: np.ndarray
    approach: np.ndarray
    close_axis: np.ndarray
    width: float
    contacts: np.ndarray
    cone_angle: float
    clearance: float
    local_width: float
    score: float

    def grip_command(self, span: float, *, squeeze: float = 0.004) -> float:
        """How far to close on this object, 0 open .. 1 fingers together.

        Closes to just inside the *measured* width, so the grip force is what
        the position servo makes of a few millimetres of standing error rather
        than what it makes of driving the pads together. The failure this avoids
        is recorded on a sharps-disposal scene: commanded fully shut, a smooth
        24 mm cylinder is extruded out of the jaws mid-carry.

        The reference's own routine has to fall back to closing fully whenever
        its bounding box is wider than the span (a claw hammer reads 132 mm
        across a 36 mm handle). Here there is no such case: the width came from
        the two faces the fingers will actually touch.

        Args:
            span: Fingertip separation at full open [m].
            squeeze: How far inside the measured width to aim [m].
        """
        if span <= 0.0:
            return 1.0
        return float(np.clip(1.0 - max(0.0, self.width - squeeze) / span, 0.0, 1.0))

    @property
    def top_down(self) -> bool:
        """Whether this grasp comes in from above rather than from the side."""
        return float(np.dot(self.approach, UP)) < -0.7

    def flipped(self) -> GraspCandidate:
        """The same grasp with the jaws swapped over: the wrist rolled half a turn.

        Which contact of an antipodal pair is called the first one is arbitrary,
        so the closing direction's *sign* is arbitrary, and a parallel hand is
        symmetric under the flip -- both wrists put the same pads on the same two
        faces. They are not interchangeable to the *arm*, though: they differ by
        180 degrees on the last wrist joint, and on a Panda reaching across
        itself one of them runs that joint into its limit while the other is
        comfortable. Measured on a tool-sorting scene: 40.9 degrees of residual
        roll against a hard stop, where the flip solves to 0.9.

        The old box heuristic did this by wrapping its yaw into [-90, 90). A
        sampler that reads real faces cannot wrap an angle it never computed, so
        it hands both over and lets whoever owns an IK pick.
        """
        return GraspCandidate(
            position=self.position,
            quat=quat_multiply(quat_from_axis_angle(self.approach, np.pi), self.quat),
            approach=self.approach,
            close_axis=-self.close_axis,
            width=self.width,
            contacts=self.contacts[::-1].copy(),
            cone_angle=self.cone_angle,
            clearance=self.clearance,
            local_width=self.local_width,
            score=self.score,
        )


class AntipodalSampler:
    """Proposes grasps on one model's bodies by searching their meshes.

    Built once per compiled model and reused: the triangles are gathered lazily
    through :func:`~robosimstudio.core.render.visual.shape_meshes`, which the
    studio's picker shares, and only the per-query transform is paid per click.

    Args:
        model: The finalized model.
        span: Fingertip separation at full open [m] -- what a pair has to fit
            inside. Read this off the arm's :class:`~robosimstudio.solvers.Gripper`
            rather than off a spec default; a geared hand's span is not its
            joint travel.
        approach_axis: Gripper-local axis pointing from the wrist towards the
            grasp point, as the embodiment's arm spec measures it.
        close_axis: Gripper-local axis the jaws close along.
        ee_offset_rot: Rotation from the tool body's frame to the grasp frame, xyzw.
        friction_deg: Half-angle of the friction cone at each contact. A pair
            whose normals deviate by more than this cannot be squeezed without
            sliding. 30 degrees is a coefficient of about 0.58, which is
            optimistic for smooth plastic and about right for the rest.
        finger_clearance: Margin left on each side of the object inside the
            span [m], so a pair that only just fits is not proposed.
        finger_depth: How far the pads reach past the palm [m]. A grasp deeper
            into the object than this cannot be taken: the fingers bottom out on
            it and the palm arrives first. Measured on the failure it exists to
            stop -- a 100 mm tall soup can proposes a beautiful 65 mm antipodal
            pair across its middle, and a top-down hand asked for it presses the
            can into the table instead of lifting it, because reaching a contact
            50 mm below the lid means putting 50 mm of finger through the lid.
        finger_radius: How wide a corridor each pad needs [m]. Object geometry
            inside it and above the contact is geometry the finger has to go
            through.
        pad: The jaws' own plates, for the clearance test that asks where they
            **land** -- see :class:`PadBox` and :meth:`_pads_clear`. ``None``
            takes the 2F-85's measured plate.
        squeeze: How far inside the measured width the jaws will be driven [m].
            :meth:`GraspCandidate.grip_command`'s own default, restated here
            because the pad test has to know where the plates stop.
        max_triangles: Ceiling on the triangles searched per body. A scanned
            grocery mesh carries tens of thousands and the pairwise search is
            quadratic in the *seeds*, not in these, but the world transform is
            linear in them and a click should not cost a tenth of a second.
        max_seeds: Ceiling on the sampled contact points. The pair search is
            this squared, so it is the number that sets the cost.
        approach_fan: How many approach directions to try per pair, spread
            around the plane perpendicular to the closing axis.
        standoff: How far ahead of the grasp the approach is tested for
            obstructions [m]. Matches the stand-off a caller's approach leg
            backs off by, and is therefore also how far
            :meth:`_pads_clear` sweeps the plates back along the approach for
            the descent -- one distance, because it is one leg.
        ignore_shapes: Model shape indices that are **not** obstacles -- the
            grasping robot's own, which the caller knows and this does not.

            Not an optimisation. :meth:`_obstacles` boxes everything that is not
            the body being grasped, and that includes the hand doing the
            grasping; a caller that queries with the arm parked anywhere near
            the object casts its upward stand-off straight into its own gripper
            and every top-down candidate is dropped for want of room. Measured
            on a wiring-sort workcell: called with the arm at home the wire
            spool offers 17 candidates of which 3 pass, and called with the hand
            hovering a few centimetres away over the previous goal, 16 of which
            **0** pass, with an empty refusal. Not "nothing found": found, and
            every one of them a side grasp.

            Defaults to empty, which is the old behaviour, because a caller with
            no robot in the scene has nothing to name.
        shapes: How many shapes make up **one** world. ``0``, the default, is
            "all of the model's" and is right for every single-world scene.

            It exists for a **replicated** one. A fleet's model holds every
            world's geometry, and a world's connector reads ``body_q`` back as
            the *template's* rows -- which is what lets a tool written against
            one scene index this world's bodies with the indices a handle
            resolved once. The two disagree here and nowhere else:
            :meth:`_obstacles` boxes everything that is not the body being
            grasped, so it reaches world 3's shapes, looks their body up in a
            template-length pose array and raises ``index 27 is out of bounds
            for axis 0 with size 27`` -- every world, on its first grasp. Even
            bounds-checked it would be wrong: another world's crate is not an
            obstacle to this one's hand.

            Counted in **shapes** rather than bodies because a shape index is
            world-strided whatever it hangs off, and a body index is not
            available for the ones that hang off nothing: a two-world tool-sort
            model carries sixteen bodiless shapes, eight per world, and a
            body-index rule keeps every one of them. Shapes at or past this
            belong to a later world and are dropped from both geometry sets.

            Which leaves *world 0's* geometry, whichever world is asking, and
            that is right rather than lucky: a replica differs from the template
            in where its bodies are and not in what they are made of, and where
            they are is the ``body_q`` the caller passes in.
    """

    def __init__(
        self,
        model: newton.Model,
        *,
        span: float,
        approach_axis: Sequence[float] = (0.0, 0.0, 1.0),
        close_axis: Sequence[float] = (0.0, 1.0, 0.0),
        ee_offset_rot: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        friction_deg: float = 30.0,
        finger_clearance: float = 0.004,
        finger_depth: float = 0.045,
        finger_radius: float = 0.010,
        max_triangles: int = 4000,
        max_seeds: int = 256,
        approach_fan: int = 12,
        standoff: float = 0.10,
        ignore_shapes: Sequence[int] = (),
        pad: PadBox | None = None,
        squeeze: float = 0.004,
        shapes: int = 0,
    ) -> None:
        if span <= 0.0:
            raise ValueError(f"span must be positive, got {span}")
        self.model = model
        self.span = float(span)
        self.approach_axis = np.asarray(approach_axis, dtype=np.float64)
        self.close_axis = np.asarray(close_axis, dtype=np.float64)
        self.ee_offset_rot = np.asarray(ee_offset_rot, dtype=np.float64)
        self.friction_deg = float(friction_deg)
        self.finger_clearance = float(finger_clearance)
        self.finger_depth = float(finger_depth)
        self.finger_radius = float(finger_radius)
        self.max_triangles = int(max_triangles)
        self.max_seeds = int(max_seeds)
        self.approach_fan = int(approach_fan)
        self.standoff = float(standoff)
        self.ignore_shapes = frozenset(int(s) for s in np.asarray(ignore_shapes).reshape(-1))
        self.pad = pad or PadBox()
        self.squeeze = float(squeeze)
        self.shapes = max(0, int(shapes))

        self._meshes: list | None = None
        self._by_body: dict[int, list] | None = None
        self._colliders: list | None = None
        self.last_refusal = ""
        """Why the last :meth:`propose` returned nothing, in words a readout can show."""

    # ----------------------------------------------------------------- geometry
    def invalidate(self) -> None:
        """Drop the cached geometry. Cheap; it is rebuilt on the next query."""
        self._meshes = None
        self._by_body = None
        self._colliders = None

    def _this_world(self, meshes: list) -> list:
        """Drop the shapes belonging to a world this query is not about.

        A no-op unless ``shapes`` was given -- see the class docstring for what
        it is for and what happens without it.
        """
        if self.shapes <= 0:
            return meshes
        return [mesh for mesh in meshes if int(mesh.index) < self.shapes]

    def _geometry(self) -> tuple[list, dict[int, list]]:
        if self._meshes is None:
            self._meshes = self._this_world(shape_meshes(self.model))
            self._by_body = {}
            for mesh in self._meshes:
                self._by_body.setdefault(int(mesh.body), []).append(mesh)
        return self._meshes, self._by_body

    def _collision_geometry(self) -> list:
        """The shapes an obstacle test has to be about: the ones that collide.

        A different set from :meth:`_geometry`, not a subset of it. The mesh
        search reads what is *drawn*, because that is where the real surface is
        and where a width worth commanding comes from; an obstacle test reads
        what would actually stop a finger. On every ``astro`` object those two
        share no shape at all: the drawn mesh's collider bit is off and the
        collider is an invisible substitute the compiler fitted for it.

        How far apart the two then are is a property of the substitute and not
        of this method, and it has been both extremes on this suite inside a
        week -- the convex hull it carries today agrees with the drawing, and the
        fitted box before it stood up to 28.6 mm proud, which is more than a
        plate's width of geometry a drawn-set test would not have known about.
        Reading the colliding set is right in both cases and only *visibly*
        right in one.
        """
        if self._colliders is None:
            self._colliders = self._this_world(shape_meshes(self.model, flag=int(newton.ShapeFlags.COLLIDE_SHAPES)))
        return self._colliders

    def _surface(self, body: int, body_q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One body's triangles in world: ``(centroids, unit normals, areas)``.

        Sampling triangle centroids rather than vertices is what keeps a contact
        on a *face* -- a vertex is shared between faces pointing several ways, so
        its normal names none of them and the antipodal test on it is noise.
        """
        _, by_body = self._geometry()
        pose = np.eye(4) if int(body) < 0 else transform_matrix(body_q[int(body)])
        centroids, normals, areas = [], [], []
        for mesh in by_body.get(int(body), ()):
            world = pose @ mesh.local
            verts = mesh.vertices @ world[:3, :3].T + world[:3, 3]
            tri = verts[mesh.faces]
            cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            area = 0.5 * np.linalg.norm(cross, axis=1)
            keep = area > 1.0e-12
            centroids.append(tri[keep].mean(axis=1))
            normals.append(cross[keep] / (2.0 * area[keep])[:, None])
            areas.append(area[keep])
        if not centroids:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        return np.concatenate(centroids), np.concatenate(normals), np.concatenate(areas)

    # ---------------------------------------------------------------- proposing
    def propose(
        self,
        body: int,
        body_q: np.ndarray,
        *,
        near: Sequence[float] | None = None,
        k: int = 8,
        rng: np.random.Generator | None = None,
    ) -> list[GraspCandidate]:
        """Grasps on one body, best first.

        Args:
            body: Index into ``State.body_q``.
            body_q: The step's body transforms, ``[body_count, 7]``.
            near: A world point to search around, normally where a click landed.
                ``None`` searches the whole body.
            k: How many candidates to return.
            rng: Source for the triangle subsampling. Defaults to a fixed seed,
                which is what makes clicking the same object twice cycle a
                *stable* list rather than re-roll the sample into a new order
                and step somewhere unrelated.

        Returns:
            Up to ``k`` candidates, highest score first. Empty means no pair of
            opposed faces fits the span; :attr:`last_refusal` says so in words.
        """
        rng = rng or np.random.default_rng(0)
        centroids, normals, areas = self._surface(body, body_q)
        if not len(centroids):
            self.last_refusal = "that body has no mesh to search"
            return []

        points, surface_normals = self._seed(centroids, normals, areas, near, rng)
        if len(points) < 2:
            self.last_refusal = "nothing to grasp near where you clicked"
            return []

        pairs = self._pairs(points, surface_normals)
        if not len(pairs):
            widest = self.span - 2.0 * self.finger_clearance
            self.last_refusal = f"no opposed faces closer than the {widest * 1000:.0f} mm the jaws fit"
            return []

        com = self._body_com(body, body_q)
        blocked = self._obstacles(body, body_q)
        # A coarse copy of the object for the pad-reach test, which runs per
        # fanned direction per pair and does not need every triangle to notice a
        # can's lid standing between a finger and its contact.
        surface = centroids
        if len(surface) > _REACH_POINTS:
            surface = surface[rng.choice(len(surface), size=_REACH_POINTS, replace=False, p=_weights(areas))]

        # Best-opposed first and capped: hundreds of pairs across one flat face
        # are the same grasp, and the fan is paid per pair.
        pairs.sort(key=lambda pair: pair[3])
        pairs = pairs[:_MAX_PAIRS]

        candidates = []
        blocked_by_depth = 0
        blocked_by_pads = 0
        blocked_by_body = 0
        for i, j, width, cone in pairs:
            centre = 0.5 * (points[i] + points[j])
            close_world = points[j] - points[i]
            contacts = np.stack([points[i], points[j]])
            approach, clearance, reach, pads, on_body = self._approach(
                centre, close_world, width, blocked, surface, contacts
            )
            if approach is None:
                blocked_by_depth += reach > self.finger_depth
                blocked_by_pads += reach <= self.finger_depth and not pads and not on_body
                blocked_by_body += reach <= self.finger_depth and on_body
                continue
            try:
                quat = grasp_quat(self.approach_axis, self.close_axis, self.ee_offset_rot, approach, close_world)
            except ValueError:
                continue  # a fanned direction that came out parallel to the closing axis
            candidates.append(
                GraspCandidate(
                    position=centre,
                    quat=quat,
                    approach=approach,
                    close_axis=close_world / max(width, 1.0e-12),
                    width=float(width),
                    contacts=contacts,
                    cone_angle=float(cone),
                    clearance=float(clearance),
                    local_width=self._local_width(surface, centre, close_world / max(width, 1.0e-12)),
                    score=self._score(centre, approach, width, cone, clearance, reach, com, near, close_world),
                )
            )
        if not candidates:
            # Which of the three, because they send a reader to different fixes:
            # a grasp too deep is answered by clicking elsewhere on the object, a
            # jaw landing on a neighbour by moving the neighbour, and no room at
            # all by approaching from somewhere else.
            # The **dominant** count, not the first non-zero one. Testing them
            # in a fixed order meant a single self-blocked pair spoke for four
            # hundred: ``phillips_screwdriver_0`` reported "the plates would
            # land in the object itself" on 81 such pairs while 319 were
            # rejected against its neighbours. A refusal that names the wrong
            # cause is worse than none, because it is acted on.
            reasons = (
                (blocked_by_depth, "every grasp on it is deeper than the fingers reach -- try nearer an edge"),
                (
                    blocked_by_body,
                    "every pair on it is one the jaws cannot reach from outside -- "
                    "the plates would land in the object itself",
                ),
                (blocked_by_pads, "every way of straddling it puts a jaw inside something else -- clear around it"),
            )
            count, worst = max(reasons, key=lambda pair: pair[0])
            self.last_refusal = worst if count else "every grasp on it is blocked by something in the way"
            return []

        self.last_refusal = ""
        candidates.sort(key=lambda c: -c.score)
        return self._dedupe(candidates)[:k]

    # -------------------------------------------------------------- the search
    def _seed(
        self,
        centroids: np.ndarray,
        normals: np.ndarray,
        areas: np.ndarray,
        near: Sequence[float] | None,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Contact points to search over: near the click if it is on the body.

        The locality radius is generous on purpose. A click lands on the surface
        and the far contact of the pair is up to a span away *through* the
        object, so a radius of one span would keep the near face and drop the
        one it has to close against.
        """
        keep = np.arange(len(centroids))
        if near is not None:
            radius = 1.2 * self.span
            close = np.flatnonzero(np.linalg.norm(centroids - np.asarray(near, dtype=np.float64), axis=1) <= radius)
            # A click on a face whose neighbourhood holds nothing to close
            # against is a hint that could not be honoured, not a failure: fall
            # back to the whole body rather than refuse a graspable object.
            if len(close) >= 2:
                keep = close

        if len(keep) > self.max_triangles:
            keep = rng.choice(keep, size=self.max_triangles, replace=False, p=_weights(areas[keep]))
        if len(keep) > self.max_seeds:
            keep = rng.choice(keep, size=self.max_seeds, replace=False, p=_weights(areas[keep]))
        return centroids[keep], normals[keep]

    def _pairs(self, points: np.ndarray, normals: np.ndarray) -> list[tuple[int, int, float, float]]:
        """Antipodal pairs: opposed faces close enough to fit between the jaws.

        The textbook test, vectorised over the whole seed set at once. For a
        pair ``(p_i, n_i)``, ``(p_j, n_j)`` with ``u`` the unit vector from i to
        j, both contacts must lie inside the friction cone: the grasp axis has
        to be within ``friction_deg`` of *into* the surface at i and of *out of*
        it at j. That is what distinguishes squeezing a bar from pinching two
        points on the same face, which no width test would catch.
        """
        delta = points[None, :, :] - points[:, None, :]
        width = np.linalg.norm(delta, axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            u = delta / width[:, :, None]

        cos_cone = float(np.cos(np.radians(self.friction_deg)))
        # u points i -> j, so it runs *into* i's surface (against n_i) and *out
        # of* j's (along n_j).
        into_i = -np.einsum("ijk,ik->ij", u, normals)
        out_of_j = np.einsum("ijk,jk->ij", u, normals)

        fits = (width > 1.0e-6) & (width <= self.span - 2.0 * self.finger_clearance)
        opposed = (into_i >= cos_cone) & (out_of_j >= cos_cone)
        # Only the upper triangle: (i, j) and (j, i) are the same grasp.
        ii, jj = np.nonzero(np.triu(fits & opposed, k=1))
        if not len(ii):
            return []
        worst = np.minimum(into_i[ii, jj], out_of_j[ii, jj])
        cone = np.degrees(np.arccos(np.clip(worst, -1.0, 1.0)))
        return list(zip(ii.tolist(), jj.tolist(), width[ii, jj].tolist(), cone.tolist(), strict=True))

    def _local_width(self, surface: np.ndarray, centre: np.ndarray, close_axis: np.ndarray) -> float:
        """How wide the object is across ``close_axis`` at ``centre`` [m].

        The object's own surface points within :data:`_LOCAL_RADIUS` of the
        grasp, projected onto the closing axis; the width is the span of that
        projection. Restricted to a neighbourhood because the question is about
        the *feature* being grasped and not about the body: a hammer's handle
        and its claw are one body, and only one of them is between the pads.

        The radius is measured across the closing axis rather than in all three
        directions, so a long thin feature contributes its whole cross-section
        and none of what lies further along it.

        Returns ``width`` unchanged when fewer than two points fall in the
        neighbourhood -- a caller comparing a ratio then gets 1.0 and admits the
        pair, which is the right way to fail for a test whose whole job is to
        *reject*.
        """
        rel = surface - centre
        along = rel @ close_axis
        perpendicular = np.linalg.norm(rel - np.outer(along, close_axis), axis=1)
        near = perpendicular <= _LOCAL_RADIUS
        if int(np.count_nonzero(near)) < 2:
            return 0.0
        span = along[near]
        return float(span.max() - span.min())

    def _reach(self, surface: np.ndarray, contacts: np.ndarray, approach: np.ndarray) -> float:
        """How much of the object each pad has to pass through to reach its contact [m].

        The object's own geometry standing between a pad and where that pad has
        to end up. Measured per contact, over the points inside a corridor of
        :attr:`finger_radius` about the line the finger comes down, and only
        those *above* the contact -- a finger arriving at the side of a soup can
        halfway up has the whole top half of the can in front of it, while one
        arriving at the side of a hammer's handle has 18 mm of handle and then
        air.

        This is deliberately not the approach-clearance test: that one asks
        whether anything *else* is in the way and excludes the body being
        grasped, because the tool centre is *supposed* to end up inside that
        body's silhouette. This one asks how far inside, which is the question
        the pads answer.
        """
        depth = 0.0
        for contact in contacts:
            rel = surface - contact
            height = rel @ (-approach)
            perpendicular = np.linalg.norm(rel - np.outer(height, -approach), axis=1)
            # 3 mm of skin: the contact sits on a triangle, and its own
            # neighbours are always a hair above it.
            near = (perpendicular < self.finger_radius) & (height > 0.003)
            if np.any(near):
                depth = max(depth, float(height[near].max()))
        return depth

    def _approach(
        self,
        centre: np.ndarray,
        close_world: np.ndarray,
        width: float,
        blocked: tuple[np.ndarray, ...],
        surface: np.ndarray,
        contacts: np.ndarray,
    ) -> tuple[np.ndarray | None, float, float, bool, bool]:
        """The best direction to come in from, how much room it has, how deep it reaches, whether the jaws fit, and whether they land in the object.

        The hand may approach along anything perpendicular to the closing axis,
        which is a whole circle of choices rather than a single answer, so this
        fans the circle and keeps the best one. Ranked on what disqualifies a
        grasp first: pads that cannot reach their contacts beat everything, then
        plates that land on something else, then an approach with no room to
        stand off, then how near vertical it is, then how much room it has.
        Coming in from above is preferred and not required -- a plate lying flat
        has no vertical approach that does not close the jaws through the table,
        and reaching in from the side is better than refusing.

        The pad test belongs *inside* the fan rather than after it because the
        plates turn with the approach: rolling the wrist about the closing axis
        sweeps them through a different volume, so a heading whose jaws land in a
        neighbour at one roll can be clear at another, and only the fan can find
        it.
        """
        axis = close_world / max(float(np.linalg.norm(close_world)), 1.0e-12)
        # Seed the fan from the most-vertical direction perpendicular to the
        # closing axis, so the first sample is the one a top-down grasp wants.
        down = -UP - float(np.dot(-UP, axis)) * axis
        if float(np.linalg.norm(down)) < 1.0e-6:
            # The jaws close vertically, so every approach is horizontal.
            down = np.cross(axis, UP)
            if float(np.linalg.norm(down)) < 1.0e-6:
                down = np.cross(axis, np.array([1.0, 0.0, 0.0]))
        down = down / float(np.linalg.norm(down))
        side = np.cross(axis, down)

        best, best_clearance, best_reach, best_pads, best_key = None, 0.0, np.inf, False, None
        best_on_body = False
        for step in range(max(1, self.approach_fan)):
            angle = 2.0 * np.pi * step / max(1, self.approach_fan)
            direction = np.cos(angle) * down + np.sin(angle) * side
            reach = self._reach(surface, contacts, direction)
            clearance = self._clearance(centre, direction, blocked)
            pads = self._pads_clear(centre, direction, close_world, width, blocked)
            # And the body being grasped, which _pads_clear excludes: a plate
            # inside *it* is a pair across a recess rather than a grasp. Kept as
            # its own flag, because "the neighbour is in the way" and "there is
            # nowhere on this object to put a jaw" send a reader to different
            # fixes -- move the neighbour, or grasp somewhere else entirely.
            on_body = self._pads_on_body(centre, direction, close_world, width, surface)
            pads = pads and not on_body
            key = (
                reach <= self.finger_depth,
                pads,
                clearance >= self.standoff,
                -float(np.dot(direction, UP)),
                clearance,
            )
            if best_key is None or key > best_key:
                best, best_clearance, best_reach, best_pads, best_key = direction, clearance, reach, pads, key
                best_on_body = on_body
        if best is None:
            return None, 0.0, np.inf, False, False
        # The pads bottom out on the object before they reach their contacts:
        # this grasp is inside the thing, from every direction.
        if best_reach > self.finger_depth:
            return None, best_clearance, best_reach, best_pads, best_on_body
        # A jaw would have to pass through a neighbour, or through the object
        # itself, to straddle this pair.
        if not best_pads:
            return None, best_clearance, best_reach, False, best_on_body
        # Nowhere to put the hand at all. Half the stand-off is the shortest
        # approach leg worth flying; below that the fingers start inside
        # whatever is in the way.
        if best_clearance < 0.5 * self.standoff:
            return None, best_clearance, best_reach, best_pads, best_on_body
        return best, best_clearance, best_reach, best_pads, best_on_body

    def _clearance(self, centre: np.ndarray, approach: np.ndarray, blocked: tuple[np.ndarray, ...]) -> float:
        """How far back from the grasp the hand can stand [m], capped at the stand-off.

        Cast *backwards* from the grasp point along the approach and stop at the
        first thing in the way, which is the question the approach leg asks: not
        "is the path clear" but "where can it start". A grasp deep inside a bin
        answers a centimetre and gets ranked below one on the same object that
        is reachable from above.

        Boxes rather than triangles: this runs once per fanned direction per
        pair, and what it has to catch is a hand reaching through a neighbour or
        into a bin wall, which an axis-aligned box already answers.

        **It is a ray from the tool centre, so it says nothing about the jaws**,
        and that is what :meth:`_pads_clear` is for. A tool centre inside an
        object's own silhouette casts straight up into free air while the plates
        either side of it are buried in the neighbour.
        """
        room = self.standoff
        segment = -approach * self.standoff
        for box, axes, half in zip(blocked[0], blocked[1], blocked[2], strict=True):
            # Into the box's own frame, where it is the axis-aligned slab test
            # again. Rotating the ray is exact; boxing the box is not.
            origin = axes @ (centre - box)
            hit = _segment_box(origin, axes @ segment, -half, half)
            if hit is not None:
                room = min(room, hit * self.standoff)
        return float(max(0.0, room))

    def _plate_frame(
        self, centre: np.ndarray, approach: np.ndarray, close_world: np.ndarray, width: float
    ) -> tuple[np.ndarray, float, float, float, float] | None:
        """The grasp frame and the plate's extents along it, shared by both pad tests.

        Returns ``(axes, inner, outer, back, front)`` or ``None`` for a hand with
        no plate to speak of, or a frame too degenerate to orient one in.
        """
        span = max(self.span, 1.0e-9)
        wide = min(width + 2.0 * self.finger_clearance, span)
        narrow = max(width - self.squeeze, 0.0)
        inner, outer = 0.5 * narrow, 0.5 * wide + self.pad.thickness
        if outer <= inner:
            return None
        back, front = self.pad.along_approach(1.0 - wide / span, 1.0 - narrow / span)
        x = close_world / max(float(np.linalg.norm(close_world)), 1.0e-12)
        z = approach / max(float(np.linalg.norm(approach)), 1.0e-12)
        y = np.cross(z, x)
        norm = float(np.linalg.norm(y))
        if norm < 1.0e-9:
            return None
        return np.stack([x, y / norm, z]), inner, outer, back, front

    def _pads_on_body(
        self,
        centre: np.ndarray,
        approach: np.ndarray,
        close_world: np.ndarray,
        width: float,
        surface: np.ndarray,
    ) -> bool:
        """Whether a plate would land inside the object it is trying to grasp.

        **The question every width heuristic was a proxy for.** A pair is a
        grasp when the pads can reach its contacts *from outside*; it is a pair
        across a recess when they cannot, because the object's own material is
        where a plate has to be. Asking that directly needs no reference width,
        and a reference width is what every version of this test has got wrong:
        the whole bounding box scores a claw hammer's handle 0.27 because the
        box is sized by the claw, and a fixed-radius neighbourhood scores a
        pistol-grip drill's grip 0.18 because 25 mm still spans both limbs of
        the L. Each shape wanted its own constant, which is the sign that the
        quantity was never the right one.

        This is :meth:`_pads_clear`'s test with the one body it excludes put
        back. That exclusion is right for the *tool centre* -- which is supposed
        to end up inside the object's silhouette -- and wrong for the plates,
        which are supposed to straddle it.

        Only the volume strictly **outboard** of the contact plane counts. The
        contacts themselves lie on the surface at ``+/- width / 2``, and the
        commanded close drives the gripping face a squeeze inside that, so the
        near part of the plate is *meant* to be in contact. :data:`_SKIN` is the
        margin that keeps the contact's own triangles from reading as an
        obstruction, and it is the one this class already uses for the same
        reason in :meth:`_reach`.

        Fails **open**: a hand whose plate geometry was never measured has
        ``half_width`` and ``thickness`` of zero, so the volume is empty and
        every pair passes -- which is the behaviour those hands have today.
        """
        if not len(surface) or not self.pad.half_width:
            return False
        frame = self._plate_frame(centre, approach, close_world, width)
        if frame is None:
            return False
        axes, _, outer, back, front = frame
        inner = 0.5 * width + _SKIN
        if outer <= inner:
            return False
        local = (surface - centre) @ axes.T  # into the grasp frame
        across = np.abs(local[:, 1]) <= self.pad.half_width
        along = (local[:, 2] >= back - self.standoff) & (local[:, 2] <= front)
        reach = np.abs(local[:, 0])
        outboard = (reach >= inner) & (reach <= outer)
        return bool(np.any(across & along & outboard))

    def _pads_clear(
        self,
        centre: np.ndarray,
        approach: np.ndarray,
        close_world: np.ndarray,
        width: float,
        blocked: tuple[np.ndarray, ...],
    ) -> bool:
        """Whether both plates can get where this grasp puts them without going through something.

        The question :meth:`_clearance` never asks. A grasp is two plates
        straddling an object, and straddling means the plates stand *outside*
        it -- so on anything in a bin they are the parts of the hand nearest to
        whatever is next to it, while the tool centre the approach ray is cast
        from sits safely in the middle of the object being taken. Measured on
        ``astro/wiring_sort``: a screwdriver's collision box runs alongside the
        combination pliers' for 160 mm with a few millimetres between them, and
        every grasp the graph could fly reached 8.5 to 10.8 mm past the pliers'
        own boundary and into it -- while reporting the full 100 mm of
        clearance, because the ray it was measured on goes up the middle.

        **The volume, and why it survives a pad that moves.** Each plate is a box
        in the grasp frame, and every extent of it is a *sweep* rather than a
        pose:

        * along the closing axis, from the gripping face at the width the jaws
          will be commanded to -- ``width - squeeze``, halved -- out through the
          width they descend at -- ``width`` plus :attr:`finger_clearance` a
          side, which is what a preshape leaves -- and then
          :attr:`PadBox.thickness` further for the plate itself. The jaws are at
          the wide end when they arrive and at the narrow end when they finish,
          so the plate really does occupy all of it;
        * along the approach, :meth:`PadBox.along_approach` between those same
          two commands, which is where the four-bar's ~13 mm of swing is
          accounted for, extended back by :attr:`standoff` for the descent;
        * across, :attr:`PadBox.half_width` either side.

        The plate is oriented and the obstacles are axis-aligned, so the overlap
        test is the box-box separating-axis theorem rather than an AABB of a
        rotated AABB -- which on a grasp taken at 45 degrees would inflate the
        plate by half its length and refuse grasps that are fine.
        """
        boxes, box_axes, box_half = blocked
        if not len(boxes):
            return True
        span = max(self.span, 1.0e-9)
        # The two jaw gaps this grasp passes through: what the descent flies at
        # and what the close is commanded to, each as a distance and as the 0..1
        # command the plate's own position is a function of.
        wide = min(width + 2.0 * self.finger_clearance, span)
        narrow = max(width - self.squeeze, 0.0)
        inner, outer = 0.5 * narrow, 0.5 * wide + self.pad.thickness
        if outer <= inner:  # a hand with no plate to speak of has nothing to test
            return True
        back, front = self.pad.along_approach(1.0 - wide / span, 1.0 - narrow / span)
        # **The plate is swept its full authored depth, including past whatever
        # the object rests on, and that is a known limitation rather than an
        # oversight.** A thin object lying on a bench has no top-down grasp at
        # all under this test: on a sharps-disposal scene all 400 opposed pairs
        # are refused, and all 134 of them return the moment the bench is taken
        # out of the obstacle set. A real hand would bring its fingertips down
        # to the surface there, so the refusal is wrong.
        #
        # Truncating the sweep at the object's own underside fixes exactly that
        # and was tried, in two forms -- for every obstacle, and then only for
        # the world-welded ones, on the argument that a bench may be touched
        # where a loose neighbour may not. **Both cost more than they bought.**
        # Measured, three episodes each: a heap-sorting scene 2/3 -> 0/3 either
        # way, while the objects the truncation recovered (a syringe, a
        # breadboard, a screwdriver) turned into no additional task success at
        # all -- the sharps-disposal scene stays 0/3 because what actually stops it
        # is wrist-roll authority, not the grasp. Four more candidates on a
        # bench are not worth a suite that was passing.
        #
        # So the honest state is: this is over-strict for anything resting on a
        # surface, the recovery is known and is one line, and it needs the
        # obstacle geometry to stop being an axis-aligned box of a rotated one
        # before it is worth having. See ``_obstacles``.
        x = close_world / max(float(np.linalg.norm(close_world)), 1.0e-12)
        z = approach / max(float(np.linalg.norm(approach)), 1.0e-12)
        y = np.cross(z, x)
        norm = float(np.linalg.norm(y))
        if norm < 1.0e-9:
            return True  # degenerate frame; grasp_quat will refuse it anyway
        y = y / norm
        axes = np.stack([x, y, z])
        half = np.array([0.5 * (outer - inner), self.pad.half_width, 0.5 * (front - back + self.standoff)])
        # The plate's centre in the grasp frame, mirrored for the second jaw.
        along = 0.5 * (front + back - self.standoff)
        for side in (1.0, -1.0):
            middle = centre + axes.T @ np.array([side * 0.5 * (inner + outer), 0.0, along])
            if _obb_hits_obbs(middle, axes, half, boxes, box_axes, box_half):
                return False
        return True

    def _obstacles(self, body: int, body_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World AABBs of every *collider* that is not the body being grasped.

        :attr:`ignore_shapes` comes out too. The hand that is going to fly the
        approach is not an obstacle to it -- it is the thing that moves -- and
        leaving it in is how a query made with the arm hovering over the object
        refuses every top-down grasp on it. See the constructor.

        **Oriented, not axis-aligned, and that is the whole of this method.**
        Measured over the benchmark's scenes, the AABB claims up to **11.9x** the
        volume the shape occupies (a level-2 tool-sorting scene; 10.6x and 5.3x on
        two other sorting scenes) -- a 200 mm screwdriver lying
        at 40 degrees in a bin asks for a box an order of magnitude too big, and
        when *every* object in the bin does that there is no candidate left to
        lose. That level-2 scene is what it cost: 0 of 8 placed, and not for
        want of clock -- given three times the step budget the graph finishes
        early, having skipped every object in turn.

        Returns:
            ``(centres, axes, halfs)``: the box centre in world, its unit axes
            as **rows** of a ``[3, 3]``, and its half-extents along them, so
            both obstacle tests can run over the whole stack at once.
        """
        centres, axes, halfs = [], [], []
        for mesh in self._collision_geometry():
            if int(mesh.body) == int(body) or int(mesh.index) in self.ignore_shapes:
                continue
            world = mesh.local if int(mesh.body) < 0 else transform_matrix(body_q[int(mesh.body)]) @ mesh.local
            rotation = np.asarray(world[:3, :3], dtype=np.float64)
            lo = np.asarray(mesh.lo, dtype=np.float64)
            hi = np.asarray(mesh.hi, dtype=np.float64)
            centres.append(rotation @ (0.5 * (lo + hi)) + np.asarray(world[:3, 3], dtype=np.float64))
            # Rows are the box's own axes, which is the convention
            # _obb_hits_obbs and _segment_obb both read. A shape transform is a
            # rigid motion, so the columns are already unit.
            axes.append(rotation.T)
            halfs.append(0.5 * (hi - lo))
        if not centres:
            return np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3))
        return np.asarray(centres), np.asarray(axes), np.asarray(halfs)

    # --------------------------------------------------------------- selection
    def _score(
        self,
        centre: np.ndarray,
        approach: np.ndarray,
        width: float,
        cone: float,
        clearance: float,
        reach: float,
        com: np.ndarray,
        near: Sequence[float] | None,
        close_world: np.ndarray | None = None,
    ) -> float:
        """Rank one candidate. Higher is better.

        Six terms, and the weights say what the studio is for: a grasp that
        holds beats a grasp that is convenient. Antipodal quality and the moment
        arm are about whether the object stays in the hand; verticality and
        clearance are about whether the hand can get there; the click term only
        breaks ties between grasps that are otherwise as good as each other.
        """
        antipodal = 1.0 - min(1.0, cone / max(self.friction_deg, 1.0e-6))
        margin = np.clip((self.span - width) / self.span, 0.0, 1.0)
        lever = self._lever(centre, com, close_world if BALANCE_PERP else None)
        balance = float(np.exp(-lever / BALANCE_DECAY))
        vertical = 0.5 * (1.0 - float(np.dot(approach, UP)))
        room = float(np.clip(clearance / max(self.standoff, 1.0e-9), 0.0, 1.0))
        # Shallow beats deep even when both fit. A pad that has to slide most of
        # its length past the object catches on every irregularity on the way,
        # and it is the grasp that a millimetre of settling turns into a miss.
        shallow = 1.0 - float(np.clip(reach / max(self.finger_depth, 1.0e-9), 0.0, 1.0))
        pointed = 1.0 if near is None else float(np.exp(-float(np.linalg.norm(centre - np.asarray(near))) / 0.05))
        return float(
            3.0 * antipodal
            + BALANCE_WEIGHT * balance
            + 1.5 * shallow
            + 1.0 * vertical
            + 1.0 * room
            + 0.5 * margin
            + 1.0 * pointed
        )

    @staticmethod
    def _lever(centre: np.ndarray, com: np.ndarray, close_world: np.ndarray | None) -> float:
        """Gravity's moment arm about the grasp axis [m].

        The two contacts of an antipodal pair lie on one line -- the closing
        axis -- so friction at them has **no arm about that line**, and the only
        thing resisting a turn about it is torsional friction over the pad's own
        patch. What drives the turn is ``m g`` times the horizontal offset from
        that line to the centre of mass, *perpendicular* to the closing
        direction; the component along it is carried by the pads.

        The distinction is not academic. Measured on ``astro/wiring_sort``, with
        the object's rotation read relative to the hand: ``combination_pliers``
        goes from 0.2 to 62.8 degrees in twenty control steps the moment the
        lift takes the bench's support away, then hangs there oscillating for
        the rest of the carry with the jaw gap moving 3 mm -- a pivot, not a
        slip, about an axis within 18 degrees of the commanded closing axis.

        ``close_world`` is optional so a caller with no closing direction still
        gets the old omnidirectional reading rather than an exception.
        """
        offset = np.asarray(centre, dtype=np.float64) - np.asarray(com, dtype=np.float64)
        offset = np.array([offset[0], offset[1], 0.0])
        if close_world is None:
            return float(np.linalg.norm(offset))
        heading = np.array([float(close_world[0]), float(close_world[1]), 0.0])
        norm = float(np.linalg.norm(heading))
        if norm < 1.0e-9:
            # The jaws close vertically, so the grasp axis is vertical and
            # gravity has no moment about it at all -- whatever the offset.
            # `mesh_grasp`'s level-close filter never lets one of these through,
            # but the studio's picker takes the sampler unfiltered.
            return 0.0
        heading = heading / norm
        return float(np.linalg.norm(offset - float(np.dot(offset, heading)) * heading))

    def _dedupe(
        self, candidates: list[GraspCandidate], *, radius: float = 0.015, angle_deg: float = 20.0
    ) -> list[GraspCandidate]:
        """Drop candidates that are the same grasp as a better one already kept.

        A mesh proposes hundreds of pairs across one flat face and they are all
        the same grasp; without this, cycling through candidates in the page
        would step a millimetre at a time and never reach the other end of the
        object.
        """
        cos_limit = float(np.cos(np.radians(angle_deg)))
        kept: list[GraspCandidate] = []
        for candidate in candidates:
            duplicate = any(
                float(np.linalg.norm(candidate.position - other.position)) < radius
                and abs(float(np.dot(candidate.close_axis, other.close_axis))) > cos_limit
                for other in kept
            )
            if not duplicate:
                kept.append(candidate)
        return kept

    def _body_com(self, body: int, body_q: np.ndarray) -> np.ndarray:
        """The body's centre of mass in world [m]."""
        com_local = np.asarray(self.model.body_com.numpy()[int(body)], dtype=np.float64)
        xform = np.asarray(body_q[int(body)], dtype=np.float64)
        return xform[:3] + quat_to_mat(xform[3:7]) @ com_local


def _weights(areas: np.ndarray) -> np.ndarray:
    """Area weights for subsampling, normalised, uniform if the areas are useless.

    Weighting by area is what keeps a mesh whose detail is concentrated in one
    fillet from spending every seed there and none on the faces worth grasping.
    """
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(areas), 1.0 / max(len(areas), 1))
    return areas / total


def _lerp(ends: tuple[float, float], t: float) -> float:
    """Between the two ends of a stroke, at ``t`` in [0, 1]."""
    low, high = float(ends[0]), float(ends[1])
    return low + (high - low) * float(np.clip(t, 0.0, 1.0))


def _box_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The 8 corners of an axis-aligned box, ``[8, 3]``."""
    signs = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float64)
    return lo + signs * (hi - lo)


def _obb_hits_boxes(centre: np.ndarray, axes: np.ndarray, half: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    """Whether an oriented box overlaps any of a stack of world AABBs.

    Ericson's ``TestOBBOBB``, with the second box axis-aligned so its rotation
    is the first one's own axes, and vectorised over the stack: fifteen
    candidate separating axes, and a pair is disjoint the moment one of them
    separates it.

    The exact test rather than the AABB of the rotated box, because the loose
    version's error is not small where it matters. A jaw plate is 8 mm through
    and 74 mm long, so boxing it axis-aligned at 45 degrees claims a 58 mm
    footprint and refuses grasps with centimetres to spare.

    Args:
        centre: The oriented box's centre [world].
        axes: Its unit axes as rows, ``[3, 3]``.
        half: Its half-extents along those axes, ``[3]``.
        lo: Per-box minimum corner, ``[n, 3]``.
        hi: Per-box maximum corner, ``[n, 3]``.
    """
    if not len(lo):
        return False
    box_half = 0.5 * (hi - lo)
    delta = 0.5 * (lo + hi) - centre
    # The translation in the oriented box's own frame, and |R| with the epsilon
    # Ericson's write-up adds so that two parallel boxes do not separate on a
    # cross product that is numerically zero.
    local = delta @ axes.T
    abs_r = np.abs(axes) + 1.0e-9

    disjoint = np.zeros(len(lo), dtype=bool)
    for i in range(3):
        disjoint |= np.abs(local[:, i]) > half[i] + box_half @ abs_r[i]
    for j in range(3):
        disjoint |= np.abs(delta[:, j]) > half @ abs_r[:, j] + box_half[:, j]
    for i in range(3):
        for j in range(3):
            a, b = (i + 1) % 3, (i + 2) % 3
            u, v = (j + 1) % 3, (j + 2) % 3
            reach = half[a] * abs_r[b, j] + half[b] * abs_r[a, j]
            reach = reach + box_half[:, u] * abs_r[i, v] + box_half[:, v] * abs_r[i, u]
            disjoint |= np.abs(local[:, b] * axes[a, j] - local[:, a] * axes[b, j]) > reach
    return bool(np.any(~disjoint))


def _obb_hits_obbs(
    centre: np.ndarray,
    axes: np.ndarray,
    half: np.ndarray,
    boxes: np.ndarray,
    box_axes: np.ndarray,
    box_half: np.ndarray,
) -> bool:
    """Whether an oriented box overlaps any of a stack of oriented boxes.

    Ericson's ``TestOBBOBB`` with **both** boxes oriented, vectorised over the
    stack: fifteen candidate separating axes -- three of each box's own, and the
    nine cross products -- and a pair is disjoint the moment one of them
    separates it.

    Args:
        centre: The query box's centre [world].
        axes: Its unit axes as rows, ``[3, 3]``.
        half: Its half-extents along those axes, ``[3]``.
        boxes: Per obstacle, its centre, ``[n, 3]``.
        box_axes: Per obstacle, its unit axes as rows, ``[n, 3, 3]``.
        box_half: Per obstacle, its half-extents, ``[n, 3]``.
    """
    if not len(boxes):
        return False
    # R[k, i, j] = a_i . b_kj, and the epsilon Ericson's write-up adds so that
    # two parallel boxes do not separate on a cross product that is numerically
    # zero.
    rot = np.einsum("ip,kjp->kij", axes, box_axes)
    abs_r = np.abs(rot) + 1.0e-9
    delta = boxes - centre
    local = np.einsum("ip,kp->ki", axes, delta)

    disjoint = np.zeros(len(boxes), dtype=bool)
    for i in range(3):
        disjoint |= np.abs(local[:, i]) > half[i] + np.einsum("kj,kj->k", box_half, abs_r[:, i, :])
    other = np.einsum("kjp,kp->kj", box_axes, delta)
    for j in range(3):
        disjoint |= np.abs(other[:, j]) > np.einsum("i,ki->k", half, abs_r[:, :, j]) + box_half[:, j]
    for i in range(3):
        for j in range(3):
            a, b = (i + 1) % 3, (i + 2) % 3
            u, v = (j + 1) % 3, (j + 2) % 3
            reach = half[a] * abs_r[:, b, j] + half[b] * abs_r[:, a, j]
            reach = reach + box_half[:, u] * abs_r[:, i, v] + box_half[:, v] * abs_r[:, i, u]
            disjoint |= np.abs(local[:, b] * rot[:, a, j] - local[:, a] * rot[:, b, j]) > reach
    return bool(not np.all(disjoint))


def _segment_box(origin: np.ndarray, direction: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float | None:
    """Where a segment first enters a box, as a fraction of its length, or ``None``.

    The standard slab test. ``direction`` spans the whole segment rather than
    being a unit vector, so the answer is in [0, 1].
    """
    t0, t1 = 0.0, 1.0
    for axis in range(3):
        d = float(direction[axis])
        if abs(d) < 1.0e-12:
            if origin[axis] < lo[axis] or origin[axis] > hi[axis]:
                return None
            continue
        near = (float(lo[axis]) - float(origin[axis])) / d
        far = (float(hi[axis]) - float(origin[axis])) / d
        if near > far:
            near, far = far, near
        t0, t1 = max(t0, near), min(t1, far)
        if t0 > t1:
            return None
    return t0
