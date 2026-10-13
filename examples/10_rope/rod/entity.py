# SPDX-License-Identifier: Apache-2.0
"""Building a rod, and the pegs it wraps: the mechanism ``cable_rod`` takes.

**Why this is library code, when it was content.** It used to live in
the cable suite's own ``rod.py``, and its docstring argued for staying there:
"an asset is content whether or not it carries code, and a library that shipped
a 0.91 m rod cable would be shipping a benchmark's props." That argument was
right about the *props* and wrong about the *mechanism*, and what exposed the
difference was a second task needing the mechanism. ``threaded_needle``'s suture
is a rod too -- a shorter, stiffer one carrying a needle -- and its entity
subclasses :class:`EntityRod` outright. Two tasks may not reach each other,
so the shared half had to become
either a duplicate or a layer. A 968-line duplicate is not checked redundancy;
it is drift with a delay.

So the split is: **the capsule chain, its latch, its tuning views and the peg
fixtures are here; every measurement of a particular cable stays a default that
a task overrides.** :class:`RodSpec`'s field defaults are still the weave
reference's own numbers -- 0.9144 m of 11 mm-radius cable at 350 kg/m³ (~176 g
as built), stretch 2e5, bend 0.04 -- and they are documented as such rather than
as the library's opinion of what a rod is. ``cable_route`` takes them as they
stand; a shorter, stiffer rod overrides most of them. Neither reads the other.

The names stay with the content. Nothing here calls ``register_entity``: the
task's own ``rod.py`` binds ``cable_rod``, ``cable_peg`` and
``cable_peg_round`` to these classes, which keeps every entity name declared by
the task that stages it -- the same seam every other entity in the tree honours.

The rod is **not** a cloth. The reference cable simulation it was ported from
builds it as a rigid rod -- 28 overlapping capsules chained by cable joints
(``ModelBuilder.add_rod``), solved by rigid VBD/AVBD inside the coupled-proxy
solver -- so it enters a scene as an *extras* entity rather than through the
particle path.

Friction is a *scene* statement here, made at bind: the reference fills every
shape to ``mu = 2.0`` and then carves out the three materials that matter --
cable 60, gripper blades 60 (grip = sqrt(60·60) = 60), and the resting surface
0.02 (cable-vs-table = sqrt(60·0.02) ≈ 1.1). The absurd-looking 60s are the
point: Newton combines friction as ``sqrt(mu_a · mu_b)``, so pairwise targets
this far apart are only reachable by splitting them across the two shapes.

The pegs are cylinders, which :class:`~robosimstudio.specs.StaticSpec`
cannot say -- its procedural shape is a box, and a cable wrapped around a box
is a different task. ``EntityPeg`` is the static domain's own post
(:class:`~robosimstudio.domains.static.posts.EntityPost`), which reads the
spec's ``half_extents`` as ``(radius, _, half_height)`` and says so loudly.

``EntityPegRound`` is the other fixture: ManipulationNet's F1, imported whole
rather than drawn. A spool with a groove holds a route the hand has let go of
where a post lets it ride up and off, which is why the weaves stand on it and
why its collider is the authored mesh rather than a hull -- the hull of a spool
is the cylinder it was chosen instead of.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import newton
import numpy as np
import warp as wp

from robosimstudio.core.render.palette import ROBOT_DARK
from robosimstudio.domains.static.entity import EntityStatic, PlacedAtBuild
from robosimstudio.domains.static.posts import EntityPost, fixture_shape_cfg
from robosimstudio.scene.entity import Entity
from robosimstudio.specs.params import LIVE, ui

from .centerline import Centerline
from .latch import build_hooks as _build_latch
from .spec import RodMaterialSpec, RodSpec

__all__ = [
    "CableDriveSpec",
    "CableLatchSpec",
    "EntityCable",
    "EntityPeg",
    "EntityPegRound",
    "EntityRod",
    "RodSpec",
    "SpecCable",
    "fixture_shape_cfg",
]


CableDriveSpec = RodMaterialSpec
"""The rod's constitutive stiffness, under the name ``cable_route`` imports it.

The class itself is :class:`~robosimstudio.domains.rod.spec.RodMaterialSpec`, and
this used to be a second copy of it -- forty lines, field for field, ``ui()``
metadata included, one in ``specs`` and one here. Two dataclasses that must stay
identical with no way to say so is a drift waiting to happen, and the duplicate
here was the live one while the ``specs`` copy was reachable from nothing at all.

The material belongs in ``specs`` because a *material* is data about a kind of
thing, readable with no entity and no solver present -- the argument every other
spec in that package rests on.
"""


@dataclass(frozen=True, kw_only=True)
class CableLatchSpec:
    """The grasp latch's gains and ceiling, live.

    Attributes:
        kp: Position gain pinning the pinched segment to the hand [N/m].
        kd: Damping against the relative velocity [N·s/m].
        f_max: Force ceiling [N]. Four rod weights as built -- the clamp that
            turned a 72 m/s ejection into a slip.
    """

    kp: float = field(default=400.0, metadata=ui(tier=LIVE, unit="N/m", range=(1.0, 1e4), log=True, group="latch"))
    kd: float = field(default=12.0, metadata=ui(tier=LIVE, unit="N*s/m", range=(1e-2, 1e2), log=True, group="latch"))
    f_max: float = field(default=7.0, metadata=ui(tier=LIVE, unit="N", range=(1e-2, 1e2), log=True, group="latch"))


def _bow_cable(points, quats, spec, episode):
    """Bend a straight spawn into a half-cosine bow, per episode. See ``bow_range``.

    Returns the inputs untouched when the range is zero-width, so the straight
    path stays byte-identical rather than merely equivalent -- a rod that moved
    by a rounding error would change every recorded episode in the corpus.

    The quaternions are turned with the curve. ``add_rod`` reads them as each
    segment's frame, and leaving them straight while bending the positions
    hands the solver a rod whose every segment is twisted against its own
    neighbours -- it relaxes with a snap on the first step, which looks like a
    physics bug and is a spawn bug.
    """
    lo, hi = (float(v) for v in getattr(spec, "bow_range", (0.0, 0.0)))
    if abs(hi - lo) <= 0.0 and abs(lo) <= 0.0:
        return points, quats
    # The EPISODE's unit fraction, scaled by the CABLE's range. Read from the
    # sampled overrides rather than re-derived from the seed: an override is
    # what the model key is built from, so a bow taken straight off the seed
    # would vary in this function and not in the key -- the env would re-pose
    # instead of rebuilding and every episode would silently get episode 0's
    # rod. Measured before it was: four episodes, four identical bows.
    overrides = dict(getattr(episode, "overrides", ()) or ())
    unit = overrides.get("cable_bow")
    if unit is None:
        return points, quats
    amplitude = 0.5 * (lo + hi) + 0.5 * (hi - lo) * float(unit)
    if abs(amplitude) < 1.0e-6:
        return points, quats

    axis = np.asarray([float(v) for v in spec.direction], dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    up = np.array([0.0, 0.0, 1.0])
    lateral = np.cross(up, axis)
    if float(np.linalg.norm(lateral)) < 1.0e-9:  # a vertical rod has no "sideways"
        lateral = np.cross(np.array([1.0, 0.0, 0.0]), axis)
    lateral /= max(float(np.linalg.norm(lateral)), 1.0e-12)

    pts = np.asarray([[float(c) for c in p] for p in points], dtype=np.float64)
    n = len(pts)
    t = np.linspace(0.0, 1.0, n)
    # A SINGLE ARC IS NOT WHAT A LAID CABLE LOOKS LIKE. One half-cosine pins both
    # tips and bulges once in the middle, which at these amplitudes reads as a
    # nearly straight rod with a slight bend -- the shape a cable takes when it
    # is under tension, not when it is put down.
    #
    # A real run has slack, and slack meanders: it crosses its own axis. The
    # second harmonic is what adds that -- ``sin(2*pi*t)`` is antisymmetric, so
    # it swings the rod one way over the first half and the other way over the
    # second WITHOUT moving the extremes any further out than its own amplitude.
    # That matters here, because how far the rod may stray toward the stations
    # is bounded by the idle floor and not by taste: the near-side seat boxes
    # begin about 60 mm off the spawn line, and a rod that reaches one scores
    # before the robot has moved.
    offset = amplitude * np.sin(np.pi * t)  # zero at both tips, max at the middle
    wave_lo, wave_hi = (float(v) for v in getattr(spec, "wave_range", (0.0, 0.0)))
    u_wave = overrides.get("cable_wave")
    if u_wave is not None and abs(wave_hi - wave_lo) > 0.0:
        w_amp = 0.5 * (wave_lo + wave_hi) + 0.5 * (wave_hi - wave_lo) * float(u_wave)
        offset = offset + w_amp * np.sin(2.0 * np.pi * t)

    # TWO MORE DEGREES OF FREEDOM, both optional and both zero when unsampled.
    #
    # A bow alone varies the rod's MIDDLE and pins its two tips, so every episode
    # still starts with the ends in the same place -- and the ends are what the
    # anchor grasp and the first crossing are aimed at. A policy can be right
    # about those by remembering a coordinate. Shift and yaw move them:
    #
    #   cable_shift  slides the whole rod along the lateral axis (toward or away
    #                from the station line), so the standoff is not a constant.
    #   cable_yaw    turns it in the bench plane about its own midpoint, so the
    #                rod is not parallel to the station line and each station
    #                sees a different standoff.
    #
    # Both are unit fractions scaled by the spec's own ranges, read from the
    # sampled overrides for the same reason the bow is -- an override is what the
    # model key is built from, and anything re-derived from the seed here would
    # vary in this function and not in the key.
    shift_lo, shift_hi = (float(v) for v in getattr(spec, "shift_range", (0.0, 0.0)))
    yaw_lo, yaw_hi = (float(v) for v in getattr(spec, "yaw_range", (0.0, 0.0)))
    u_shift = overrides.get("cable_shift")
    u_yaw = overrides.get("cable_yaw")
    if u_shift is not None and abs(shift_hi - shift_lo) > 0.0:
        offset = offset + (0.5 * (shift_lo + shift_hi) + 0.5 * (shift_hi - shift_lo) * float(u_shift))

    bowed = pts + offset[:, None] * lateral[None, :]

    if u_yaw is not None and abs(yaw_hi - yaw_lo) > 0.0:
        ang = math.radians(0.5 * (yaw_lo + yaw_hi) + 0.5 * (yaw_hi - yaw_lo) * float(u_yaw))
        # About the rod's own midpoint, so a yaw does not also translate it.
        mid = bowed.mean(axis=0)
        c, s_ = math.cos(ang), math.sin(ang)
        rel = bowed - mid
        bowed = mid + np.stack([rel[:, 0] * c - rel[:, 1] * s_, rel[:, 0] * s_ + rel[:, 1] * c, rel[:, 2]], axis=1)

    # Each segment's frame follows its own chord. The straight quats already
    # carry the rod's local-axis-to-direction rotation, so what is composed on
    # is only the turn from `direction` to this segment's tangent.
    out = []
    for i, q in enumerate(quats):
        a, b = bowed[min(i, n - 2)], bowed[min(i + 1, n - 1)]
        tangent = b - a
        norm = float(np.linalg.norm(tangent))
        if norm < 1.0e-12:
            out.append(q)
            continue
        tangent /= norm
        cross = np.cross(axis, tangent)
        sin_a = float(np.linalg.norm(cross))
        cos_a = float(np.dot(axis, tangent))
        if sin_a < 1.0e-12:
            out.append(q)
            continue
        turn = wp.quat_from_axis_angle(wp.vec3(*(cross / sin_a)), float(np.arctan2(sin_a, cos_a)))
        out.append(wp.mul(turn, q))
    return [wp.vec3(*p) for p in bowed], out


HOLD_KP_REFERENCE = 400.0
HOLD_KD_REFERENCE = 12.0
"""The latch's position gains [N/m, N.s/m], as the reference tuned them.

Forces, not accelerations -- which is the whole reason they need scaling. See
:meth:`EntityRod.hold_gains`.
"""


def _half_heights_of(points) -> np.ndarray:
    """Each capsule's half-length, from the centreline nodes it spans.

    What ``add_rod_graph`` gives the capsule it builds for that chord
    (``half_height = 0.5 * seg_length``), restated here so
    :meth:`EntityRod.centerline` can answer the curve as its *endpoints* rather
    than its centres. A bowed rod's chords are not equal to each other and not
    equal to ``length / segments``: measured over fifty ``port_easy`` draws, a
    chord runs up to 1.0809 of the nominal pitch.
    """
    pts = np.asarray([[float(v) for v in p] for p in points], dtype=np.float64)
    return 0.5 * np.linalg.norm(np.diff(pts, axis=0), axis=1)


def _segment_mass(spec: RodSpec) -> float:
    """One capsule's mass [kg], from what the spec declares.

    Analytic rather than read off the model, because the gains are wanted before
    a body exists and because the ratio is what matters, not the absolute: the
    overlap `add_rod` builds in cancels between two rods built the same way.
    """
    return float(spec.density) * math.pi * float(spec.radius) ** 2 * (float(spec.length) / int(spec.segments))


class EntityRod(Centerline, Entity):
    """The capsule-chain cable: built by ``add_rod``, indexed for everyone else."""

    spec: RodSpec
    domain = "rod"

    def __init__(self, spec: RodSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.bodies: np.ndarray = np.zeros(0, dtype=np.int32)
        self.joints: np.ndarray = np.zeros(0, dtype=np.int32)
        self.shapes: np.ndarray = np.zeros(0, dtype=np.int32)
        self.embodiment = None
        """Backfilled by the suite's solver factory, which sees every entity;
        the anchor pre-pose needs the arm's IK and nothing else does."""
        self._rest_joint_q: np.ndarray | None = None
        self._rod_spans: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0), (0, 0))
        self._half_heights: np.ndarray = np.zeros(0, dtype=np.float64)
        self._half_heights_per_world: dict[int, np.ndarray] = {}
        """Each world's capsule half-lengths, one row per world :meth:`place` drew.

        Empty in a single-world build, where :attr:`_half_heights` -- the built
        rod's -- is the only answer there is, and :meth:`centerline` falls back to
        it for any world not in here. A capsule's chord runs up to 8% over the
        nominal 12 mm, so answering a world with another's is ~1 mm on
        ``cable.end_b``, which ``portbox`` scores at 1.0."""
        self._anchor_welds: tuple[tuple[int, int], ...] = ()
        """``(joint, body)`` per declared anchor, template-indexed. See :meth:`_weld_anchors`."""
        self._hold_flag: wp.array | None = None
        self._hold_seg: wp.array | None = None
        self._hold_rel: wp.array | None = None
        self._hold_grip: wp.array | None = None
        self._hold_gains: wp.array | None = None
        self._grip_bodies: dict[str, int] = {}
        self._tcp_offsets: dict[str, np.ndarray] = {}

    # -------------------------------------------------------------- phase 1
    def _shape_cfg(self) -> Any:
        """The capsules' contact settings. Shared by :meth:`build` and :meth:`place`."""
        spec = self.spec
        return newton.ModelBuilder.ShapeConfig(
            density=float(spec.density),
            ke=float(spec.contact_ke),
            kd=float(spec.contact_kd),
            mu=float(spec.mu_base),
            margin=0.0,
            gap=float(spec.contact_gap),
        )

    def _spawn(self, episode) -> tuple[list, list | None]:
        """This episode's centreline nodes and segment frames, in world.

        Split out of :meth:`build` because :meth:`place` has to draw *the same
        rod* for another world, and a second copy of the lift-and-bow would be
        two ways for a wave's rods to stop being the ones a serial build makes.
        """
        from newton.utils import create_straight_cable_points_and_quaternions

        spec = self.spec
        # ``start`` is stated in the cell's work-surface frame, like every
        # rigid pose, and lifted here. This was the last place a task-owned
        # entity handed a pose to the builder raw: a rod laid at z = 0.01 on a
        # 750 mm bench landed on the floor and looked like a rod that fell, and
        # both shipped rod tasks compensated by hand. The frame carries no
        # rotation, so ``direction`` is the same in both.
        if spec.nodes is not None:
            # A stated polyline: every node lifted into the world the same way
            # a straight start is, and the frames left to ``add_rod`` to derive
            # from consecutive legs, which is what it does for ``None``.
            return [
                wp.vec3(*[float(v) for v in self.placed((*node, 0.0, 0.0, 0.0, 1.0))[:3]]) for node in spec.nodes
            ], None
        start = wp.vec3(*[float(v) for v in self.placed((*spec.start, 0.0, 0.0, 0.0, 1.0))[:3]])
        direction = wp.vec3(*[float(v) for v in spec.direction])
        points, quats = create_straight_cable_points_and_quaternions(
            start=start, direction=direction, length=float(spec.length), num_segments=int(spec.segments)
        )
        # Only the straight spawn is bowed. A stated polyline already says
        # where every node goes, and has no frames to re-derive a bend from
        # -- ``quats`` is ``None`` there by design.
        return _bow_cable(points, quats, spec, episode)

    def _add_rod(self, builder: newton.ModelBuilder, points, quats) -> None:
        """``add_rod`` at this spec's own numbers. Shared by :meth:`build` and :meth:`place`."""
        spec = self.spec
        builder.add_rod(
            positions=points,
            quaternions=quats,
            radius=float(spec.radius),
            body_frame_origin="com",
            cfg=self._shape_cfg(),
            color=spec.color,
            stretch_stiffness=float(spec.stretch_ke),
            stretch_damping=float(spec.stretch_kd),
            bend_stiffness=float(spec.bend_ke),
            bend_damping=float(spec.bend_kd),
            label=spec.name,
            wrap_in_articulation=self.wrap_rod_in_articulation(),
        )

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        spec = self.spec
        if spec.joins:
            # Refused rather than ignored: a capsule chain has no node to weld,
            # and a cable that silently fell out of its plug would read as a
            # physics failure rather than a missing implementation.
            raise NotImplementedError(
                f"rod {spec.name!r}: `joins` is read by the foreign-engine rod; route the 'rod' domain to "
                "backend('superdex') to mould a cable into another body"
            )
        from newton.solvers import SolverVBD

        # The VBD custom attributes (per-joint hard flags, Dahl state) must be
        # registered before add_rod writes any -- the reference registers at
        # scene start; here the first cable in the scene does it.
        SolverVBD.register_custom_attributes(builder)

        self._decompose_working_tips(builder)
        # The tip hulls just added belong to the *robot*; the entity's bracketed
        # range covers them anyway, so the rod's own spans are recorded
        # explicitly from here.
        body_start, joint_start, shape_start = builder.body_count, builder.joint_count, builder.shape_count

        points, quats = self._spawn(episode)
        self._add_rod(builder, points, quats)
        self._rod_spans = (
            (body_start, builder.body_count),
            (joint_start, builder.joint_count),
            (shape_start, builder.shape_count),
        )
        self._half_heights = _half_heights_of(points)
        self._weld_anchors(builder, body_start, builder.body_count)

    def _weld_anchors(self, builder: newton.ModelBuilder, body_start: int, body_end: int) -> None:
        """Fix the declared ends to the world where they were built.

        A cable strung between two posts has no hand holding it: its ends are
        tied. A fixed joint from the world to the end capsule, at the capsule's
        build pose, is that tie -- VBD solves FIXED joints as hard constraints.
        Recorded in the rod's own joint span so the cable's island owns them.
        """
        anchors = tuple(self.spec.anchors or ())
        if not anchors or body_end <= body_start:
            return
        welds: list[tuple[int, int]] = []
        for where in anchors:
            body = body_start if where == "start" else body_end - 1
            if where not in ("start", "end"):
                raise ValueError(f"cable {self.spec.name!r}: anchors name 'start' or 'end', got {where!r}")
            pose = builder.body_q[body]
            # Recorded with the body it holds, because the weld's ``joint_X_p``
            # **is** that capsule's pose -- so a world whose rod was drawn
            # differently needs this row rewritten with it (:meth:`place`).
            welds.append((int(builder.joint_count), int(body)))
            builder.add_joint_fixed(
                parent=-1,
                child=body,
                parent_xform=wp.transform(
                    wp.vec3(*[float(v) for v in pose[:3]]), wp.quat(*[float(v) for v in pose[3:7]])
                ),
                child_xform=wp.transform_identity(),
                label=f"{self.spec.name}_anchor_{where}",
            )
        self._anchor_welds = tuple(welds)
        (b0, b1), (j0, _), (s0, s1) = self._rod_spans
        self._rod_spans = ((b0, b1), (j0, builder.joint_count), (s0, s1))

    def wrap_rod_in_articulation(self) -> bool:
        """Whether ``add_rod`` should immediately close its articulation.

        Ordinary cables are complete at this point. A derived asset may return
        false when it must append another child joint before wrapping the full
        contiguous joint range itself.
        """
        return True

    def _hand_body_labels(self) -> tuple[frozenset[str], frozenset[str]]:
        """The fingertip labels and the whole blade set, off the hands' own declarations.

        Two sets because two different steps want them: the mesh decomposition
        wants the *tips*, and the collision filter wants the tips **and** the
        body the tool-centre point is measured in, which on a jaw gripper is the
        palm the blades hang off.

        Read from :attr:`~robosimstudio.specs.GripperSpec.finger_bodies` and
        :attr:`~robosimstudio.specs.GripperSpec.tcp_body`, already templated per
        arm by :meth:`~robosimstudio.specs.GripperSpec.for_arm` -- so
        ``("{arm}_tip_left",)`` on the shipped YAM hand arrives here as
        ``right_tip_left`` and ``left_tip_left`` without this module knowing that
        the robot has arms called right and left, or that its tips are called
        tips.

        Empty for a scene with no robot, and empty for a bare arm. Both are real
        scenes and neither has a fingertip, which is a different thing from a
        label that matched nothing -- and the caller distinguishes them.
        """
        if self.embodiment is None:
            return frozenset(), frozenset()
        wrists = {arm.name: arm.ee_body_name for arm in self.embodiment.spec.arms}
        tips: set[str] = set()
        palms: set[str] = set()
        for name, hand in self.embodiment.hands.items():
            if hand is None:
                continue
            for side in hand.finger_bodies:
                tips.update(label for label in side if label)
            # An unstated tcp_body means the arm's own wrist, which is what
            # GripperSpec.tcp_body documents and what a hand welded onto a
            # flange leaves blank.
            palm = hand.tcp_body or wrists.get(name, "")
            if palm:
                palms.add(palm)
        return frozenset(tips), frozenset(tips | palms)

    def _decompose_working_tips(self, builder: newton.ModelBuilder) -> None:
        """CoACD the working arm's fingertip colliders, the reference's own step.

        The linear_4310 tip is a non-convex mesh; through the proxy pipeline
        its raw triangles give the rod a flickering contact patch and the
        friction grip bleeds. The reference decomposes the *active* arm's tips
        (threshold 0.3, up to 6 hulls) and falls back to a convex hull with a
        printed warning when CoACD is unavailable -- ported verbatim, wrap arms
        included via ``proxy_arms``.

        **Which bodies those are is the hand's declaration, not this module's
        guess.** This used to read ``arms = ("right", "left")`` crossed with
        ``f"{arm}_tip_left"`` -- the i2rt YAM's labels, written into a library
        module. On a Franka or a Robotiq that matches nothing, and
        matching nothing here is silent in the worst way: the tips are never
        decomposed, the friction grip bleeds, and the episode reads as a grasp
        that slipped rather than as a step that never ran. See
        :meth:`_hand_body_labels`.
        """
        tip_labels, blade_labels = self._hand_body_labels()
        if not tip_labels:
            return
        labels = list(builder.body_label)
        tips = [
            s
            for s in range(builder.shape_count)
            if int(builder.shape_body[s]) >= 0
            and builder.shape_source[s] is not None
            and labels[int(builder.shape_body[s])].split("/")[-1] in tip_labels
        ]
        if not tips:
            # The declared labels found nothing, which used to be the whole
            # failure and used to be silent. It is legitimate -- a hand whose
            # fingers are primitives has no mesh to decompose -- so it is a line
            # rather than a raise, but it is a line.
            print(f"[cable] no fingertip meshes matched {sorted(tip_labels)}; skipping the tip decomposition")
            return
        # WHAT COLOUR EACH TIP ALREADY IS, read off the hand before it is taken
        # apart. ``approximate_meshes`` gives its new shapes -- the hulls, and
        # the visible copy it keeps of each original mesh -- no colour of their
        # own, so whatever this entity paints them is what the fingertips LOOK
        # like from here on. The answer is "the same as before", and the only
        # way to say that for an arbitrary hand is to have read it.
        tip_colors = {int(builder.shape_body[s]): tuple(float(v) for v in builder.shape_color[s]) for s in tips}
        before = builder.shape_count
        try:
            builder.approximate_meshes(
                method="coacd",
                shape_indices=tips,
                keep_visual_shapes=True,
                threshold=0.3,
                max_convex_hull=6,
                preprocess_resolution=40,
            )
        except Exception as exc:
            print(f"[cable] CoACD unavailable ({type(exc).__name__}: {exc}); convex-hulling the fingertips")
            builder.approximate_meshes(method="convex_hull", shape_indices=tips, keep_visual_shapes=True)

        # Everything the decomposition just added, filed under the colour of the
        # tip it came from. :meth:`apply_display` paints from this rather than
        # from a constant; see the note there for what a constant cost.
        self._tip_colors = {
            s: tip_colors[int(builder.shape_body[s])]
            for s in range(before, builder.shape_count)
            if int(builder.shape_body[s]) in tip_colors
        }

        if not self.spec.filter_fingertip_scenery:
            return

        # The blades never need the scenery: the grasp puts the TCP at the
        # rod's centre, which hangs the blade tips below the table top, and
        # that contact goes through the proxy at 20x mass scaling -- measured
        # as fingers shoved from open to closed against their own servo and a
        # steady ~8 mm wrist deflection while 'at rest'. The rod still rests on
        # the table; the blades pass through it.
        blade_bodies = {b for b in range(builder.body_count) if labels[b].split("/")[-1] in blade_labels}
        # Every static, found by having no body rather than by being named.
        # This was a literal set -- {"table", "peg0", "peg1", "peg2"} -- which
        # was true of the three-peg bench and silently wrong the moment a layout
        # stood up a fourth: peg3 kept its blade contacts, and a fingertip
        # shoved against a peg it should pass through reads as a grasp that
        # drifts rather than as a filter that was never applied.
        statics = [t for t in range(builder.shape_count) if int(builder.shape_body[t]) < 0]
        for s in range(builder.shape_count):
            if int(builder.shape_body[s]) in blade_bodies:
                for t in statics:
                    builder.add_shape_collision_filter_pair(s, t)

    def apply_display(self, builder: newton.ModelBuilder) -> None:
        """The rod's colour to the rod, and the robot's to the robot.

        The base implementation paints everything in the entity's bracketed
        range, which is right for an entity whose range holds only its own
        shapes and wrong for this one: :meth:`_decompose_working_tips` adds
        fingertip hulls -- and, because it keeps the visual shapes, their
        original meshes -- to the *robot*, inside this range. Painted with the
        rod's colour they came out near-white, so the same YAM had pale
        fingertips in a cable scene and dark ones everywhere else. That is the
        palette's own warning arriving from the other direction: stating a
        colour is not enough if the thing stated is not the thing painted.

        :meth:`bind` already draws this distinction for the solver, using
        ``_rod_spans`` rather than the range so a hull is not filed as a cable
        body. This is the same split, for appearance.

        The tips are painted rather than left alone, because "left alone" is not
        neutral -- the embodiment closed its range before these existed, so an
        unpainted hull takes a build-order palette colour that moves when
        anything upstream adds a shape.

        **What they are painted is the tip's OWN colour, not a constant**, and
        the constant is the second half of this bug rather than the fix for it.
        They were given :data:`~robosimstudio.core.render.palette.ROBOT_DARK`,
        which is the library's generic robot grey at ``(0.30, 0.30, 0.30)`` --
        while the i2rt YAM paints itself ``(0.040, 0.040, 0.043)``. So the same
        hand came out of a cable scene with mid-grey fingertips on a black
        gripper, against black ones in every other suite; the near-white
        version of this was reported first and "dark" was still the wrong
        answer, just a quieter one. :meth:`_decompose_working_tips` reads each
        tip's colour off the hand before it takes it apart, and this repaints
        the copies with it, so a hand of any colour keeps the fingers it came
        with. A shape with nothing recorded still falls back to ROBOT_DARK,
        which is the "you did not say" case the palette exists for.
        """
        from robosimstudio.core.render.finish import finish_for

        rod = range(*self._rod_spans[2])
        rgb = tuple(float(v) for v in self.spec.color)
        stated = getattr(self.spec, "finish", None)
        rod_finish = stated if stated is not None else finish_for(rgb, "deformables")
        tips = getattr(self, "_tip_colors", {})
        for s in range(*self.range.shapes):
            if s in rod:
                color, finish = rgb, rod_finish
            else:
                color = tips.get(s, ROBOT_DARK)
                finish = finish_for(color, "robot")
            builder.shape_color[s] = color
            self.finishes[s] = finish

    def owned_bodies(self) -> np.ndarray:
        """The rod's own capsules -- not the entity's bracketed range, which
        also covers the fingertip hulls the tip decomposition added to the robot."""
        (b0, b1), _, _ = self._rod_spans
        return np.arange(b0, b1, dtype=np.int32)

    def owned_shapes(self) -> np.ndarray:
        """Nothing world-fixed: every capsule sits on a rod body and is derived from it."""
        return np.zeros(0, dtype=np.int32)

    world_aware = True
    """Every phase of this rod carries a world axis.

    What it took, in the order the failures would have arrived: the latch is a
    row per hand *per world*; the segment and gripper indices written into those
    rows are the replicated model's, not the template's; the ``post_solve``
    recovery runs against the model being stepped rather than the one names were
    resolved in; and the anchor is solved once and posed into every world.

    The one that is easiest to miss is the third. ``bind`` is handed the
    template deliberately, so ``self.model`` is the *wrong* model for anything a
    solver will see -- and ``eval_ik`` against it does not fail, it writes a
    sixteenth of the array and leaves the rest holding the previous substep.
    """

    @property
    def _num_worlds(self) -> int:
        """Worlds in this build. 1 when nothing replicated it."""
        return int(getattr(self.worlds, "num_worlds", 1) or 1)

    @property
    def _fleet_model(self) -> Any:
        """The model that steps, which is the template unless one replicated it."""
        return self.stepping_model if self.stepping_model is not None else self.model

    def _global(self, index: int, world: int) -> int:
        """A template body index, as this world's copy of it."""
        return self.worlds.body(int(index), int(world)) if self.worlds is not None else int(index)

    def _coord(self, index: int, world: int) -> int:
        """A template joint coordinate, as this world's copy of it."""
        return self.worlds.coord(int(index), int(world)) if self.worlds is not None else int(index)

    def _template_body_q(self, state, world: int) -> np.ndarray:
        """``state``'s body transforms as one world sees them, template-indexed.

        Two kinds of caller reach the latch picker and they hold different
        arrays. One is above the tool boundary and has already been handed a
        per-world view -- that is what a fleet's world view is for -- so its
        array is the template's length and there is nothing to slice. The other
        is this entity's own anchor, which holds the whole replicated array and
        has to take its world out of it.

        Told apart by length, which is the only thing that actually
        distinguishes them, and the alternative -- a flag threaded through the
        tool surface -- would put worlds back above a boundary built to keep
        them below it.
        """
        values = np.asarray(state.body_q.numpy())
        if self.worlds is None or len(values) == self.worlds.bodies_per_world:
            return values
        return self.worlds.body_view(values, world)

    # -------------------------------------------------------------- phase 2
    def bind(self, model: newton.Model) -> None:
        super().bind(model)
        # The rod's own spans, not the entity's bracketed range: the range also
        # covers the fingertip hulls the tip decomposition added, and a hull
        # listed as a cable shape puts one index in two coupled-solver entries.
        (b0, b1), (j0, j1), (s0, s1) = self._rod_spans
        self.bodies = np.arange(b0, b1, dtype=np.int32)
        self.joints = np.arange(j0, j1, dtype=np.int32)
        self.shapes = np.arange(s0, s1, dtype=np.int32)

        # Off the embodiment's own index, not a f"{side}_gripper" label match.
        # The pattern happens to hold for this robot and matches nothing on the
        # next one -- and matching nothing here is silent: the latch simply
        # never finds a hand, so the cable is never held and the run looks like
        # a grasp that slipped.
        # A scene with no robot leaves both empty, and that is the ordinary case
        # for a rod on a bench rather than an error: the latch kernel reads
        # ``_hold_flag``, which nothing ever sets when there is no hand to set
        # it.
        if self.embodiment is not None:
            self._grip_bodies = {name: arm.ee_body for name, arm in self.embodiment.arms.items()}
            # Off the hand, not the arm: the tool-centre point is where *this*
            # gripper puts it, and the two YAM hands put it in opposite
            # directions along the wrist's z.
            self._tcp_offsets = {
                name: np.asarray(hand.tcp_offset if hand else (0.0, 0.0, 0.0), dtype=np.float64)
                for name, hand in self.embodiment.hands.items()
            }
        # The latch's device state and its launch come from ``latch.py`` beside
        # this module -- the one module a repro bundle copies verbatim -- rather
        # than from a second copy here. The five attribute names stay, because the repro
        # taps in ``cable_route/staging.py`` and the picker below read them.
        # Sized for the fleet: the latch is a row per hand per world, and a
        # wave's worlds each pinch their own rod with their own hands. One
        # world is the identity, so a single-world build is untouched.
        self._latch = _build_latch(model, {}, num_worlds=self._num_worlds)
        sinks = self._latch.aux_sinks
        self._hold_flag = sinks["hold_flag"]
        self._hold_seg = sinks["hold_seg"]
        self._hold_grip = sinks["hold_grip"]
        self._hold_rel = sinks["hold_rel"]
        self._hold_gains = sinks["hold_gains"]

    # ------------------------------------------------------------- phase 2b
    SPAWN_FIELDS: ClassVar[tuple[str, ...]] = ("cable_bow", "cable_shift", "cable_yaw", "cable_wave")
    """The sampled keys :func:`_bow_cable` reads: this rod's drawn spawn.

    Named on the class that CONSUMES them rather than in the planner, unlike
    :data:`~robosimstudio_bench.run.fleet.PLACEABLE_FIELDS`. A static's episode
    pose is a field of every task's vocabulary; ``cable_bow`` means nothing to a
    task with no rod, so the entity is what can say the field exists and what
    can say it is written per world."""

    places_spawn_per_world: ClassVar[bool] = True
    """Whether :meth:`place` writes this class's drawn spawn into each world.

    True here, and the wave planner drops :attr:`SPAWN_FIELDS` from the model key
    on the strength of it (:func:`~robosimstudio_bench.run.fleet.spawn_fields_placed_per_world`).
    What it is for is a SUBCLASS: one that draws its rod some other way in
    ``build``, or appends geometry :meth:`place` does not know to rewrite, has to
    answer False, or a wave would share one model between episodes whose rods it
    cannot draw and every world would silently run the build episode's -- the
    same trap :attr:`~robosimstudio.domains.static.entity.EntityStatic.claims_body`
    exists for, and the one the usbc port fell into. The domain's *other*
    implementation (the Cosserat rod, whose state is particles) is not a subclass
    and declares nothing at all, which the planner reads as no."""

    @classmethod
    def fields_placed_per_world(cls, spec) -> tuple[str, ...]:
        """The sampled fields a replicated build of ``spec`` writes per world.

        The rod's half of the planner's question; the statics' half is
        :meth:`~robosimstudio.domains.static.entity.EntityStatic.places_per_world`.
        Answers nothing for a rod whose nodes are stated outright: a polyline
        says where every node goes and :func:`_bow_cable` never reaches it, so
        the fields change nothing about that rod and there is nothing to place.
        """
        return cls.SPAWN_FIELDS if cls.places_spawn_per_world and getattr(spec, "nodes", None) is None else ()

    def place(self, model: newton.Model, episode, *, world: int = 0) -> None:
        """Draw THIS world's rod into the replicated model, before anything reads it.

        The rod's spawn is geometry, not a pose: ``add_rod`` sizes each capsule
        from its own chord, puts the body at the chord's centre, and anchors the
        cable joints at ``+-0.5 * chord``. So a bowed rod differs from a straight
        one in ``body_q``, in mass and inertia, in ``shape_scale`` and in every
        ``joint_X_p``/``joint_X_c`` -- rows a model holds once per world, which is
        exactly why this is a per-world *write* and not a per-world reset.

        **The rows are copied from a scratch build rather than re-derived.** A
        throwaway :class:`newton.ModelBuilder` runs the same ``add_rod`` this
        world's episode would have got from :meth:`build`, and its rows are copied
        into this world's slots. Re-deriving the masses and inertias here would be
        a second implementation of ``add_rod``'s arithmetic, and the property that
        matters -- that world *k* is what a serial build of episode *k* produces,
        to the bit -- is one a copy has by construction.

        Running before ``_assemble`` is what makes the spawn stress-free per
        world: SolverVBD initialises its rest bend and twist from the model's
        joint frames and ``body_q`` when it is constructed, so every world's rest
        state is its own rod's. A rod re-posed afterwards would be a rod the
        solver thinks is bent away from a straight rest shape, and it would
        relax with a snap on the first step.

        No solver notification, and none is needed: there is no solver yet.
        """
        if not self.fields_placed_per_world(self.spec) or not len(self.bodies):
            return
        points, quats = self._spawn(episode)
        scratch = newton.ModelBuilder()
        # The same registration ``build`` makes, for the same reason: ``add_rod``
        # writes VBD's per-joint custom attributes and they must exist first.
        from newton.solvers import SolverVBD

        SolverVBD.register_custom_attributes(scratch)
        self._add_rod(scratch, points, quats)
        (b0, b1), (j0, _), (s0, s1) = self._rod_spans
        if scratch.body_count != b1 - b0 or scratch.shape_count != s1 - s0:
            raise RuntimeError(
                f"rod {self.name!r}: world {world}'s draw made {scratch.body_count} bodies and "
                f"{scratch.shape_count} shapes against the template's {b1 - b0} and {s1 - s0}; a wave's "
                "worlds must be the same model and this one is not"
            )
        # FINALIZED, not read off the builder, and the difference is measurable:
        # ``finalize`` validates and corrects the inertias on the device, so a
        # model's ``body_inv_mass`` is ``1/mass`` computed in the array's own
        # precision rather than the builder's. Copying the builder's float64
        # reciprocal instead left every capsule half an ULP out -- 4.883e-04 on an
        # inverse mass of ~8192, with ``body_mass`` itself already equal. Harmless
        # physically and not equal, and equal is the property this is for.
        scratch_model = scratch.finalize(device=model.device)
        worlds = self.worlds
        rows = {
            "body_q": (b0, None if worlds is None else worlds.body),
            "body_com": (b0, None if worlds is None else worlds.body),
            "body_mass": (b0, None if worlds is None else worlds.body),
            "body_inv_mass": (b0, None if worlds is None else worlds.body),
            "body_inertia": (b0, None if worlds is None else worlds.body),
            "body_inv_inertia": (b0, None if worlds is None else worlds.body),
            "shape_scale": (s0, None if worlds is None else worlds.shape),
            # The cable joints only. ``scratch`` holds no anchor weld -- the
            # template's own is rewritten below, off the capsule it holds.
            "joint_X_p": (j0, None if worlds is None else worlds.joint),
            "joint_X_c": (j0, None if worlds is None else worlds.joint),
        }
        for name, (start, remap) in rows.items():
            array = getattr(model, name)
            values = array.numpy()
            source = getattr(scratch_model, name).numpy()
            index = np.arange(start, start + len(source))
            if remap is not None:
                index = np.asarray([remap(int(i), world) for i in index])
            values[index] = source
            array.assign(values)
        # The weld's ``parent_xform`` **is** the anchored capsule's build pose
        # (:meth:`~robosimstudio.domains.static.welded.WeldedFixture.weld_in_place`
        # states the same identity for a fixture), so a world whose rod was drawn
        # differently needs the row rewritten or its free end is tied to where
        # world 0's rod started. VBD solves a FIXED joint as a hard constraint,
        # so the error does not drift -- it yanks.
        if self._anchor_welds:
            parents = model.joint_X_p.numpy()
            bodies = model.body_q.numpy()
            for joint, body in self._anchor_welds:
                row = joint if worlds is None else worlds.joint(joint, world)
                parents[row] = bodies[body if worlds is None else worlds.body(body, world)]
            model.joint_X_p.assign(parents)
        # This world's chords, for the endpoint curve. See
        # :attr:`_half_heights_per_world`.
        self._half_heights_per_world[int(world)] = _half_heights_of(points)

    def _blade_shapes(self, model: newton.Model) -> set[int]:
        """Which shapes are a gripper's, off the hand's own declaration.

        The same resolution :meth:`_decompose_working_tips` uses, and for the
        same reason. This matched shape-label *tails* against the literals
        ``"tip_left"``, ``"tip_right"`` and ``"gripper"`` -- the i2rt YAM's
        spelling, written into a library module. ``yam_bimanual`` stages
        ``yam_linear_arena``, whose hand declares
        ``("{arm}_link_left_finger",)``, so on the embodiment every rod example
        uses the rule matched **nothing**: the pads fell through to the ``else``
        and took :attr:`RodSpec.mu_base` (2.0) where the spec says
        :attr:`RodSpec.mu_gripper` (60). A grip friction thirty times lower than
        the number beside it, silently, on the arm the examples ship.

        The *tips*, not the whole blade set. ``_hand_body_labels`` also offers
        the tips-plus-palm set, which the collision filter wants because a hand
        must not collide with the scenery it reaches over -- but friction is
        about what presses on the rod, and that is the pads. Taking the wider
        set here would put grip friction on a whole wrist link.
        """
        tip_labels, _ = self._hand_body_labels()
        if not tip_labels:
            return set()
        labels = list(model.body_label)
        bodies = {b for b in range(model.body_count) if labels[b].split("/")[-1] in tip_labels}
        return {s for s in range(model.shape_count) if int(model.shape_body.numpy()[s]) in bodies}

    def apply_model_materials(self, model: newton.Model) -> None:
        """The reference's post-finalize material pass, restated.

        Everything to the base, then the three materials that matter --
        sqrt-combined: grip 60, cable-on-table ~1.1. With ``scene_friction``
        off, only the rod's own capsules are written, and the rest of the scene
        keeps whatever its own entities stated.
        """
        mu = model.shape_material_mu.numpy()
        ke = model.shape_material_ke.numpy()
        kd = model.shape_material_kd.numpy()
        own = set(self.shapes.tolist())
        labels = list(model.shape_label)
        blades = self._blade_shapes(model)
        for s in range(model.shape_count):
            tail = labels[s].split("/")[-1]
            if s in own:
                mu[s] = float(self.spec.mu_cable)
                ke[s] = float(self.spec.contact_ke)
                kd[s] = float(self.spec.contact_kd)
            elif not self.spec.scene_friction:
                continue
            elif s in blades:
                mu[s], ke[s], kd[s] = (
                    float(self.spec.mu_gripper),
                    float(self.spec.contact_ke),
                    float(self.spec.contact_kd),
                )
            elif tail in self.spec.surface_labels:
                mu[s], ke[s], kd[s] = (
                    float(self.spec.mu_surface),
                    float(self.spec.contact_ke),
                    float(self.spec.contact_kd),
                )
            else:
                mu[s], ke[s], kd[s] = float(self.spec.mu_base), float(self.spec.contact_ke), float(self.spec.contact_kd)
        model.shape_material_mu.assign(mu)
        model.shape_material_ke.assign(ke)
        model.shape_material_kd.assign(kd)

    # -------------------------------------------------------------- phase 3
    def contribute_hooks(self, hooks) -> None:
        """Recover joint coordinates from the body transforms VBD writes.

        The proxy solver integrates the cable in maximal coordinates;
        ``eval_ik`` writes them back through the joints so ``joint_q`` stays
        the state's mirror -- the reference runs it every substep, after the
        solve.
        """
        # The model under the solver, not the one names resolved in. Against
        # the template this writes one world's worth of an array holding many,
        # and the rest keep the previous substep's coordinates -- a rod that
        # reads as frozen in fifteen worlds out of sixteen.
        model = self._fleet_model

        def recover(state, dt) -> None:
            del dt
            newton.eval_ik(model, state, state.joint_q, state.joint_qd)

        hooks.post_solve.append(recover)

        if not self.spec.allow_grasp_latch:
            return
        kp_hold, kd_hold = self.hold_gains()
        # A device array rather than launch floats: the captured graph bakes a
        # float and reads an array, and the array is what the studio's latch
        # sliders write. The array is the latch module's own, seeded here.
        self._hold_gains.assign([float(kp_hold), float(kd_hold), float(self.hold_force_max())])
        hooks.rigid_force.extend(self._latch.stages["rigid_force"])

    # -------------------------------------------------------------- phase 4
    def on_scene_ready(self, pipeline) -> None:
        for world in range(self._num_worlds):
            self._pose_anchor(pipeline, world=world)
        # After every world is posed, so the rest field a reset restores from is
        # the whole fleet's and not world 0's repeated.
        self._rest_joint_q = pipeline.state_0.joint_q.numpy().copy()

    def _anchor_arm(self) -> str | None:
        """The arm :attr:`RodSpec.anchor_arm` names, or the last one declared.

        ``None`` when the robot has no arm the anchor could use, which is not an
        error: a scene may stage a rod with an anchor point and a one-armed
        robot, and the honest outcome is no anchor rather than a ``KeyError``
        raised from inside a solver-ready hook.

        A *named* arm that does not exist is a different thing and says so --
        that is a task pointed at the wrong robot, and it should be loud.
        """
        if self.embodiment is None:
            return None
        arms = list(self.embodiment.iks)
        wanted = self.spec.anchor_arm
        if wanted:
            if wanted not in arms:
                raise KeyError(
                    f"cable {self.name!r} anchors with arm {wanted!r}, but {self.embodiment.spec.name!r} has {arms}"
                )
            return wanted
        if not arms:
            print(f"[cable] {self.name}: an anchor grasp was staged and the robot has no solvable arm; skipping")
            return None
        return arms[-1]

    def _pose_anchor(self, pipeline, *, world: int = 0) -> None:
        """Pre-pose the left arm gripping the cable, the wrap variant's anchor.

        Solved with the arm's full-pose LM (the closing axis must land across
        the cable, which an approach-axis-only solve leaves free), written into
        state and targets, and never commanded again -- the position servo
        holds it for the episode, exactly as the reference's one-off
        ``_solve_arm_ik`` does.
        """
        spec = self.spec
        if spec.anchor_grasp is None or self.embodiment is None:
            return

        arm = self._anchor_arm()
        if arm is None:
            return
        ik = self.embodiment.iks[arm]
        index = self.embodiment.arms[arm]
        # Position + approach only, roll free -- an anchor pinches wherever
        # its pads land, and demanding the grasp roll here is what parked the
        # hand 40 mm off, where it shoved the rod across the table before the
        # working arm's plan ever read it. The reference's own anchor solve
        # under-converged rotation the same way. Chained through a point above
        # the grasp so the descent leg stays in one branch.
        # Surface-frame, like ``start``; the IK wants world.
        target = np.asarray(self.placed((*spec.anchor_grasp, 0.0, 0.0, 0.0, 1.0))[:3], dtype=np.float64)
        # This world's coordinates, template-shaped -- which is what the arm's
        # solver takes. Seeding from the whole replicated vector hands a solver
        # built for one arm a vector holding W robots' worth of joints.
        joint_q = np.asarray(pipeline.state_0.joint_q.numpy(), dtype=np.float64)
        seed = np.asarray(self.worlds.coord_view(joint_q, world) if self.worlds is not None else joint_q)
        above = ik.solve_q(seed, target + np.array([0.0, 0.0, 0.08]), (0.0, 0.0, -1.0))
        solved = ik.solve_q(np.asarray(above, dtype=np.float64), target, (0.0, 0.0, -1.0))
        landed, _ = ik.fk_ee(np.asarray(solved, dtype=np.float32))
        pos_err = float(np.linalg.norm(landed - target))
        if pos_err > 0.01:
            print(f"[cable] anchor solve landed {pos_err * 1000:.1f} mm off; the wrap's far hand may not hold")

        for state in (pipeline.state_0, pipeline.state_1):
            q = state.joint_q.numpy()
            for c in index.arm_coords:
                q[self._coord(c, world)] = solved[c]
            for c in index.gripper_coords:
                q[self._coord(c, world)] = float(spec.anchor_grip)
            state.joint_q.assign(q)
        target = pipeline.control.joint_target_q.numpy()
        for c in index.arm_coords:
            target[self._coord(c, world)] = solved[c]
        for c in index.gripper_coords:
            target[self._coord(c, world)] = float(spec.anchor_grip)
        pipeline.control.joint_target_q.assign(target)
        # The stepping model, and over the whole fleet: forward kinematics is
        # what the latch below reads its poses from, and it is cheaper to
        # recompute every world than to reason about which of them is stale.
        newton.eval_fk(self._fleet_model, pipeline.state_0.joint_q, pipeline.state_0.joint_qd, pipeline.state_0)
        wp.copy(pipeline.state_1.body_q, pipeline.state_0.body_q)
        # Pin the anchored segment to the left hand. A pinch is not a clamp:
        # under the wind's drag the rod slides axially through the pads, and
        # the anchor's whole job is that it does not.
        if self._hold_flag is not None:
            self.hold_segment(True, pipeline.state_0, arm="left", slot=1, world=world)

    # -------------------------------------------------------------- phase 5
    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        """Straighten the cable back to its build pose, and re-assert the anchor.

        The anchor half is load-bearing: the embodiment's own reset re-writes
        every arm's targets to *home*, and entities reset in build order -- so
        without this the anchor arm spends the settle climbing away from its
        pinch, sweeping the rod 150 mm across the table on the way out.

        **Split by what it is a property of, not by what is convenient.** The
        straighten, the forward kinematics and the solver reset all act on the
        whole model at once and are done on the first world's call; the anchor
        is one world's hand on one world's rod and is done on every call. The
        order is the load-bearing part and was got wrong once already: the flag
        clear has to precede the first anchor, or it wipes the anchor it just
        set and ``(:init (anchor-held cable))`` reads 0.0 with nothing else
        looking wrong.
        """
        del episode
        if self._rest_joint_q is None:
            return
        if world == 0:
            self._straighten(pipeline)
        self._pose_anchor(pipeline, world=world)

    def _straighten(self, pipeline) -> None:
        """Put every world's rod back on its build pose. Once per wave.

        The rest field is the fleet's, captured after every world was posed, so
        the slice to restore is per world -- the rod's coordinate run, offset
        into each world's block. Restoring only the first block leaves the other
        worlds holding the previous episode's rod, bent where the last policy
        left it, which is a start state nobody chose and nothing reports.

        **What restores the capsules is the solver reset below, not these lines,
        and that is worth stating because it does not look that way.** ``eval_fk``
        skips CABLE joints outright (``newton/_src/sim/articulation.py``: "CABLE
        joints are skipped by generic forward kinematics"), and the 196 cable
        coordinates it would have read are all zero and carry nothing -- so the
        coordinate restore moves no capsule at all. ``reset_solvers`` returns the
        deformable island to the rest it captured, which IS every world's own
        built rod. Measured: with three worlds drawn differently, shoving every
        capsule 150 mm along y and resetting brings all three back to the first
        reset's poses exactly (0.0000 mm), and it did so identically before this
        method learned to write ``body_q`` -- which is why it does not.

        The coupling is the thing to know about: a rod's start pose survives a
        reset because a *solver* captured it, so anything that stops capturing it
        has to put a per-world ``body_q`` restore here in its place.
        """
        start = int(self.model.joint_q_start.numpy()[self.joints[0]])
        end_joint = int(self.joints[-1])
        template_end = (
            int(self.model.joint_q_start.numpy()[end_joint + 1]) if end_joint + 1 < self.model.joint_count else None
        )
        for state in (pipeline.state_0, pipeline.state_1):
            q = state.joint_q.numpy()
            end = (
                template_end
                if template_end is not None
                else (self.worlds.coords_per_world if self.worlds is not None else len(q))
            )
            for w in range(self._num_worlds):
                lo, hi = self._coord(start, w), self._coord(end, w)
                q[lo:hi] = self._rest_joint_q[lo:hi]
            state.joint_q.assign(q)
        newton.eval_fk(self._fleet_model, pipeline.state_0.joint_q, pipeline.state_0.joint_qd, pipeline.state_0)
        wp.copy(pipeline.state_1.body_q, pipeline.state_0.body_q)
        # A teleport leaves the coupled solver holding last episode's contact
        # anchors and warm starts, and the first step yanks the body toward
        # them -- measured as a freshly reset plug flung 5 m in 5 steps. The
        # solver ships the reset for exactly this.
        reset_solvers = getattr(pipeline, "reset_solvers", None)
        if reset_solvers is not None:
            reset_solvers()
        # Drop every latch FIRST, then re-establish the anchor -- in that order,
        # because the anchor is one of the latches.
        #
        # These two lines used to run the other way round, and the anchor never
        # survived a reset: `_pose_anchor` set slot 1 and `zero_()` cleared it
        # again a line later, so `(:init (anchor-held cable))` reported 0.0 on
        # every variant that declares it. Quiet in exactly the way this file
        # keeps finding: the arm still holds its solved pose, the rod still
        # starts where it should, and only the kinematic pin -- the thing that
        # stops the whole rod following the working hand -- was missing.
        #
        # A new episode starts with nothing grasped, which is what the zero is
        # for; the anchor is not a grasp the policy made and has to be put back.
        #
        # The re-anchoring is ``reset``'s, once per world, and this runs on the
        # first of those calls -- so the clear still precedes every anchor, in
        # every world, which is the whole of what the order above is protecting.
        if self._hold_flag is not None:
            self._hold_flag.zero_()

    # ---------------------------------------------------------------- control
    def hold_gains(self) -> tuple[float, float]:
        """This rod's latch gains, scaled off the reference's by segment mass.

        **The gains are forces and nothing scaled them.** 400 N/m and 12 N.s/m
        were tuned on the reference rod -- 11 mm at 28 capsules over 0.91 m,
        about 4.4 g a segment -- and applied unchanged to any rod. The weave's
        is 5 mm at 42 capsules over 0.50 m, about 0.33 g: thirteen times
        lighter, so the same spring delivered thirteen times the acceleration.

        What that looked like was not a stiff latch. It was a rod flung off the
        table at 0.9 m/s on a good run and a diverged solve on a bad one, from a
        plan whose every waypoint solved to under a tenth of a millimetre --
        which is the worst kind of failure to read, because everything upstream
        of it is provably right.

        Scaling by mass holds the latch's *dynamics* fixed instead of its force:
        same natural frequency, same damping ratio, whatever the rod is made of.
        The reference rod scales by 1.0 and is untouched.
        """
        scale = _segment_mass(self.spec) / _segment_mass(RodSpec())
        return HOLD_KP_REFERENCE * scale, HOLD_KD_REFERENCE * scale

    def hold_force_max(self) -> float:
        """The latch's force ceiling [N]: four rod weights.

        Sized by the whole rod rather than the pinched segment because that is
        what the latch actually moves -- the segment is chained to the rest, so
        hoisting the rod by one segment routes the full weight through this one
        wrench. One weight lifts, ~0.6 drags the half-rod against the table's
        combined mu of 1.1, and four covers both with the dynamics of a brisk
        carry. What it forbids is the runaway the kernel docstring describes:
        under a contact fight the latch saturates and slips instead of pumping
        the rod up to escape velocity.
        """
        return 4.0 * _segment_mass(self.spec) * self.spec.segments * 9.81

    # --------------------------------------------------------------- tunables
    def _drive_dofs(self) -> tuple[np.ndarray, np.ndarray]:
        """The cable joints' dof rows, split ``(stretch+shear, bend+twist)``.

        ``add_rod`` builds each cable joint with four dofs in a fixed order --
        stretch, shear, bend, twist -- and seeds shear from stretch and twist
        from bend, the isotropic split. Writing the pairs together is what
        keeps that isotropy under tuning: a bend slider that left twist behind
        would make a rod that flexes freshly soft and still refuses to roll.
        """
        starts = self.model.joint_qd_start.numpy()
        stretch: list[int] = []
        bend: list[int] = []
        for j in (int(v) for v in self.joints):
            begin = int(starts[j])
            end = int(starts[j + 1]) if j + 1 < len(starts) else begin
            if end - begin < 4:
                continue
            stretch += [begin, begin + 1]
            bend += [begin + 2, begin + 3]
        return np.asarray(stretch, dtype=np.int64), np.asarray(bend, dtype=np.int64)

    def tunables(self, pipeline=None):
        """The rod's stiffness, its contact material, and the latch -- live.

        What the studio's Parameters folder renders when the cable is
        selected. Three groups:

        * **drive** -- bend and stretch stiffness/damping over the cable
          joints' own dof rows. ``~``: the write is live but reaches the
          coupled solver's view through ``notify_model_changed``, so a
          captured graph drops and retakes.
        * **contact** -- the standard shape material over the rod's capsules.
        * **latch** -- the hold's gains and force ceiling, in the device array
          the capture reads: genuinely live.
        """
        from robosimstudio.core.tuning import ArrayWriter, ShapeWriter, Tunable
        from robosimstudio.specs import ContactMaterialSpec

        if self.model is None:
            return ()
        out = []

        stretch_dofs, bend_dofs = self._drive_dofs()
        if len(bend_dofs):
            joints = tuple(int(j) for j in self.joints)

            class _DriveWriter(ArrayWriter):
                flags = 1 << 1
                """``ModelFlags.JOINT_DOF_PROPERTIES``, as ``JointWriter`` spells it."""

                ROWS: ClassVar[dict[str, tuple[str, np.ndarray]]] = {
                    "bend_ke": ("joint_target_ke", bend_dofs),
                    "bend_kd": ("joint_target_kd", bend_dofs),
                    "stretch_ke": ("joint_target_ke", stretch_dofs),
                    "stretch_kd": ("joint_target_kd", stretch_dofs),
                }

                def read(self) -> dict[str, float]:
                    return {name: self._read(attr, rows) for name, (attr, rows) in self.ROWS.items()}

                def __call__(self, field_name: str, value) -> bool:
                    row = self.ROWS.get(field_name)
                    if row is None:
                        return False
                    landed = self.landed(self._write(row[0], row[1], float(value)))
                    if landed:
                        self._reland_vbd_penalties()
                    return landed

                def _reland_vbd_penalties(self) -> None:
                    """Push the new rows into the running VBD solve's own arrays.

                    ``SolverVBD`` copies ``joint_target_ke/kd`` into
                    per-constraint penalty arrays at construction
                    (``_init_joint_penalty_k``), and its
                    ``notify_model_changed`` refreshes only the rest-geometry
                    cache -- measured on this build as a slider that moved the
                    model *and* the coupled view while the rod hung exactly as
                    before. So after the standard write-and-notify, the four
                    slots each cable joint owns are recomputed in place from
                    the view's now-current rows: ``k_max`` becomes the new
                    stiffness, the live AVBD penalty is clamped under it so a
                    softened rod softens now rather than after a ramp-down
                    that never comes, and the damping row follows. In-place,
                    so the pointers a captured graph baked stay valid; the
                    guards make a newton that renames these internals degrade
                    to the write-and-notify behaviour rather than raise.
                    """
                    program = getattr(self.pipeline, "program", None)
                    solvers = list(program.solvers().values()) if program is not None else []
                    for inner in solvers:
                        if inner is None or not hasattr(inner, "joint_penalty_k_max"):
                            continue
                        try:
                            jc_start = inner.joint_constraint_start.numpy()
                            qd_start = inner.model.joint_qd_start.numpy()
                            target_ke = inner.model.joint_target_ke.numpy()
                            target_kd = inner.model.joint_target_kd.numpy()
                            k_max = inner.joint_penalty_k_max.numpy()
                            k_cur = inner.joint_penalty_k.numpy()
                            k_damp = inner.joint_penalty_kd.numpy()
                        except AttributeError:
                            continue
                        touched = False
                        for j in joints:
                            if j + 1 >= len(qd_start) or j >= len(jc_start):
                                continue
                            c0, d0 = int(jc_start[j]), int(qd_start[j])
                            if c0 + 3 >= len(k_max) or d0 + 3 >= len(target_ke):
                                continue
                            for offset in range(4):
                                k_max[c0 + offset] = target_ke[d0 + offset]
                                k_cur[c0 + offset] = min(k_cur[c0 + offset], target_ke[d0 + offset])
                                k_damp[c0 + offset] = target_kd[d0 + offset]
                            touched = True
                        if touched:
                            inner.joint_penalty_k_max.assign(k_max)
                            inner.joint_penalty_k.assign(k_cur)
                            inner.joint_penalty_kd.assign(k_damp)

            drive = _DriveWriter(self.model, pipeline)
            out.append(
                Tunable(
                    name="drive",
                    spec=RodMaterialSpec(**drive.read()),
                    write=drive,
                    summary=f"Bend and stretch over {len(self.joints)} cable joints. Bend is why it feels stiff.",
                )
            )

        if len(self.shapes):
            shapes = ShapeWriter(self.model, self.shapes, pipeline)
            out.append(
                Tunable(
                    name="contact",
                    spec=ContactMaterialSpec(**shapes.read()),
                    write=shapes,
                    summary=f"What the rod's {len(self.shapes)} capsules feel on contact.",
                )
            )

        if self._hold_gains is not None:
            gains = self._hold_gains
            fields = ("kp", "kd", "f_max")

            def write_latch(field_name: str, value) -> bool:
                # A device array the captured launch reads: nothing to notify.
                if field_name not in fields:
                    return False
                host = gains.numpy()
                host[fields.index(field_name)] = float(value)
                gains.assign(host)
                return True

            current = self._hold_gains.numpy()
            out.append(
                Tunable(
                    name="latch",
                    spec=CableLatchSpec(kp=float(current[0]), kd=float(current[1]), f_max=float(current[2])),
                    write=write_latch,
                    summary="The grasp hold's gains and its force ceiling. Live, even captured.",
                )
            )

        return tuple(out)

    def hold(
        self, on: bool, state=None, *, arm: str = "right", slot: int = 0, at: int | None = None, world: int = 0
    ) -> int:
        """The body-latch answer to :meth:`~robosimstudio.domains.rod.centerline.Centerline.hold`."""
        return self.hold_segment(on, state, arm=arm, slot=slot, at=at, world=world)

    def hold_segment(
        self, on: bool, state=None, *, arm: str = "right", slot: int = 0, at: int | None = None, world: int = 0
    ) -> int:
        """Engage or drop one latch slot: the segment nearest ``arm``'s jaws.

        Records the segment's current pose in that gripper's frame, so the rod
        rides the hand exactly as taken. Slot 0 is the working hand's, slot 1
        the anchor's. Returns the segment index, or -1.

        Args:
            state: Whose ``body_q`` to read. **Template-indexed**, which is what
                both callers already hand over: a single-world state is one, and
                a fleet's ``WorldView`` slices its world out before anything
                above the tool boundary sees it. So the *read* side of this
                needs no world and does not take one.
            world: Which world is doing the latching. The *write* side does need
                it, and in two places: the latch row, and the body indices
                written into it. Those indices go to a kernel reading the
                replicated model, so a template index there pins world 0's rod
                to world 0's gripper -- from every world at once, which reads as
                fifteen rods that will not be picked up.

        Returns:
            The segment index, template-indexed like everything else a caller
            sees, or -1.
        """
        if on and not self.spec.allow_grasp_latch:
            raise RuntimeError("This cable permits contact/friction grasps only; artificial grasp latches are disabled")
        row = self._latch.row(slot, world) if self._latch is not None else int(slot)
        flags = self._hold_flag.numpy()
        if not on:
            flags[row] = 0
            self._hold_flag.assign(flags)
            return -1
        body_q = self._template_body_q(state, world)
        from robosimstudio.core.math import quat_to_mat

        grip_body = self._grip_bodies[arm]
        g = body_q[grip_body]
        positions = self.positions(body_q)
        tcp = np.asarray(g[:3]) + quat_to_mat(g[3:7]) @ self._tcp_offsets[arm]
        seg_local = self.nearest_to(positions, tcp, at=at)
        if seg_local < 0:
            return -1
        seg = int(self.bodies[seg_local])
        g_tf = wp.transform(wp.vec3(*[float(v) for v in g[:3]]), wp.quat(*[float(v) for v in g[3:7]]))
        s_row = body_q[seg]
        s_tf = wp.transform(wp.vec3(*[float(v) for v in s_row[:3]]), wp.quat(*[float(v) for v in s_row[3:7]]))
        rel = wp.transform_multiply(wp.transform_inverse(g_tf), s_tf)
        rels = list(self._hold_rel.numpy())
        rels[row] = rel
        self._hold_rel.assign(rels)
        segs = self._hold_seg.numpy()
        segs[row] = self._global(seg, world)
        self._hold_seg.assign(segs)
        grips = self._hold_grip.numpy()
        grips[row] = self._global(grip_body, world)
        self._hold_grip.assign(grips)
        flags[row] = 1
        self._hold_flag.assign(flags)
        return seg

    # ---------------------------------------------------------------- runtime
    def positions(self, body_q: np.ndarray) -> np.ndarray:
        """Segment centre positions, ``[n, 3]``."""
        return np.asarray(body_q[self.bodies, :3], dtype=np.float64)

    def handle(self, model):
        """A chain publishes one handle over all its segments.

        ``body`` is the first segment -- the single-body view a pose readout and
        an ``Attach handle`` dropdown want -- and ``bodies`` is the run, which is
        what lets a click land on any segment.
        """
        del model
        bodies = tuple(int(b) for b in self.bodies) if getattr(self, "bodies", None) is not None else ()
        if not bodies:
            return None
        from robosimstudio.scene.handles import EntityHandle

        return EntityHandle(
            name=self.name,
            body=bodies[0],
            bodies=bodies,
            shapes=np.asarray(getattr(self, "shapes", ()), dtype=np.int32).ravel(),
        )

    def centerline(self, snapshot) -> np.ndarray:
        """The capsule **endpoints**, in order -- this implementation's :class:`Centerline`.

        ``segments + 1`` points, the same polyline the Cosserat implementation
        answers with, so a measure over the two reads the same curve. It was the
        capsule *centres* (``segments`` points), and the two implementations
        therefore disagreed about the length of one and the same cable: 24
        centres span 23/24 of it, which is the 4% an A/B run of the two
        implementations showed. ``add_rod`` places the centreline endpoints at
        local ``(0, 0, +-half_height)`` with ``body_frame_origin="com"``; each
        segment's first endpoint, then the last segment's second, is the curve.

        :meth:`positions` still answers with the centres, body-indexed, because
        the latch picker needs a *body* per point and there is one per centre.
        """
        if not len(self.bodies):
            return np.zeros((0, 3), dtype=np.float64)
        from robosimstudio.core.math import quat_to_mat

        body_q = np.asarray(snapshot.body_q)[self.bodies]
        # This world's chords when a wave drew it its own rod, the template's
        # otherwise. The world comes off the context the same way
        # :meth:`published_states` reads it -- a wave that took the template's
        # would answer every world the first episode's capsule lengths, which is
        # up to 8% of a 12 mm capsule on the endpoint the port scores.
        context = getattr(snapshot, "ctx", snapshot)
        half = self._half_heights_per_world.get(int(getattr(context, "world", 0)), self._half_heights)
        half = np.asarray(half, dtype=np.float64)
        if len(half) != len(body_q):
            raise RuntimeError(
                f"rod {self.name!r}: {len(body_q)} bodies but {len(half)} capsule half-lengths -- "
                "centerline() needs the rod built, not bodies assigned by hand"
            )
        axes = np.stack([quat_to_mat(q[3:7])[:, 2] for q in body_q])  # local +Z, per capsule
        starts = body_q[:, :3] - axes * half[:, None]
        last_end = body_q[-1, :3] + axes[-1] * half[-1]
        return np.vstack([starts, last_end[None, :]])

    def published_states(self, snapshot) -> dict[str, float]:
        """The latch flags, which nothing outside this entity can see.

        The wrap's anchor is a kinematic coupling written into a warp array, not
        a contact and not a pose -- so ``(anchor-held cable)`` has nothing to
        measure from outside, exactly as the retired RJ45 assembly's internal
        bodies did not. Declared on the record and read here, which is what lets the wrap
        document *state* its anchor as an init constraint rather than leave it
        as a loader side effect nobody can check.

        Slot 1 is the anchor's, slot 0 the working hand's; see
        :meth:`hold_segment`.
        """
        # Length and segment count as well: a graph that may not name the
        # suite's cable reader still has to know how much rod there is, and
        # `get_states` was the one generic read that answered for the cable
        # at all -- with two latch flags. Both are properties of the built
        # rod, not of this instant, so they are cheap and never wrong.
        try:
            chain = self.centerline(snapshot)
            length = float(np.linalg.norm(np.diff(chain, axis=0), axis=1).sum()) if len(chain) >= 2 else 0.0
        except Exception:
            length = 0.0
        shape = {"length_m": length, "n_segments": float(len(self.bodies))}
        if self._hold_flag is None:
            return {"held": 0.0, "anchor_held": 0.0, **shape}
        # This world's two rows. The context carries the world because an
        # entity's published states are read through it and nothing else; a
        # wave that read rows 0 and 1 would hand every world world 0's verdict,
        # which is the same answer W times and right by accident once.
        context = getattr(snapshot, "ctx", snapshot)
        world = int(getattr(context, "world", 0))
        flags = self._hold_flag.numpy()
        held, anchor = self._latch.row(0, world), self._latch.row(1, world)
        return {"held": float(flags[held]), "anchor_held": float(flags[anchor]), **shape}


EntityPeg = EntityPost
"""The post, under the name the cable suites import it by.

The class is :class:`~robosimstudio.domains.static.posts.EntityPost`. It was
defined here while a welded cylinder was only ever a cable's peg; ``rss.post``
made it the library's, and a constructor in the ``static`` domain whose entity
only this domain registered was a post that could not be built without a rod
installed. So the class and :func:`fixture_shape_cfg` live with the static
domain, which registers ``"post"`` itself, and this domain reuses them -- one
object under two names, so ``cable_peg`` and ``hang_post`` go on building
exactly what they built.
"""


PEG_MESH_SCALE = 0.01
"""The vendored fixture is authored in centimetres. See its PROVENANCE.md, which
is where the unit is stated -- it is the one fact about that file a reader cannot
recover from the file."""

PEG_SHAFT_RADIUS = 0.00475
PEG_FLANGE_RADIUS = 0.0125
PEG_ROUND_HEIGHT = 0.0235
"""ManipulationNet's F1, measured off the mesh at :data:`PEG_MESH_SCALE`.

Restated here so a spec can be built without loading trimesh, and asserted
against the live measurement by :class:`EntityPegRound` -- the same contract the
USB-C stage keeps, and for the same reason: two copies of one number that can
disagree silently is the failure this codebase keeps finding.
"""


class EntityPegRound(PlacedAtBuild, EntityStatic):
    """ManipulationNet's F1 round peg: a spool, imported rather than drawn.

    **The groove is the whole point.** A cable laid against a plain post is held
    there by nothing -- it rides up under tension and lifts off the top, which is
    the failure a weave spends its retention on. This fixture captures the rod
    between a 12.5 mm flange below and 17 mm of shaft above, so a seated cable
    stays seated without the grip holding it there.

    That also moves the number a goal reads: a cable seats against the **shaft**
    at 4.75 mm, not against the flange at 12.5, so it rests 9.75 mm from the axis
    where a 13 mm post would have put it at 18. The seat regions in the weave
    documents are stated against the measured value.

    Authored geometry, not a hull of it: the mesh goes in whole, because a convex
    hull of a spool is a cylinder and a cylinder is the thing this replaces.
    ``half_extents`` is ignored where :class:`EntityPeg` requires it -- the mesh
    carries its own size, and taking one from the spec would let a static declare
    a peg the fixture is not.

    **The mesh comes from the spec, and it has to.** So the path is the
    static's, resolved from what its ``AssetSpec`` declares: the library realises
    the fixture and the task says where its geometry is, which is the direction
    the import table already runs in. The dimensions below stay asserted, so a
    static that points this entity at some other mesh is refused rather than
    silently staged.
    """

    claims_body = True
    """``build`` below claims its per-world body before drawing. See :class:`EntityPeg`."""

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        # The episode's pose, and in a replicated build a body to carry it per
        # world; see :class:`EntityPeg`.
        self._claim_body(builder, episode)
        del episode
        import trimesh

        from robosimstudio.paths import resolve_path

        spec = self.spec
        if spec.asset is None or not spec.asset.path:
            raise ValueError(
                f"static {spec.name!r} realises cable_peg_round but declares no asset; this fixture is an "
                "imported mesh and the suite that owns it is what says where it lives. Pass "
                "StaticSpec(asset=declared_asset(CABLE_PEG_ROUND))."
            )
        loaded = trimesh.load(str(resolve_path(spec.asset.path)))
        vertices = np.asarray(loaded.vertices, dtype=np.float64) * PEG_MESH_SCALE
        radius = float(np.hypot(vertices[:, 0], vertices[:, 1]).max())
        height = float(vertices[:, 2].max() - vertices[:, 2].min())
        for measured, want, what in (
            (radius, PEG_FLANGE_RADIUS, "flange radius"),
            (height, PEG_ROUND_HEIGHT, "height"),
        ):
            if abs(measured - want) > 1.0e-6:
                raise RuntimeError(f"round_peg.stl: {what} measured {measured}, reference {want}")
        # Seated on the surface: the mesh's own origin is its base, and the
        # xform a loader computes puts that base on the table.
        vertices[:, 2] -= float(vertices[:, 2].min())
        # ``_shape_frame``, not ``spec.xform``: the episode may have placed this
        # fixture, and reading the declared pose here moves the SEAT and leaves
        # the SPOOL behind -- exactly the split ``episode_xform`` warns about.
        # Measured before the fix: three episodes with sampled board yaws of
        # +0.071, -0.124 and +0.230 rendered the spools at pixel-identical
        # centroids, while their seat boxes moved up to 14 mm.
        body, xform = self._shape_frame()
        # The procedural peg's contact settings, unchanged -- what is different
        # here is the shape, and changing both at once would make a retention
        # difference impossible to attribute.
        cfg = fixture_shape_cfg(builder)
        builder.add_shape_mesh(
            body=body,
            xform=xform,
            mesh=newton.Mesh(
                np.ascontiguousarray(vertices, dtype=np.float32),
                np.ascontiguousarray(np.asarray(loaded.faces, dtype=np.int64).reshape(-1), dtype=np.int32),
            ),
            cfg=cfg,
            label=spec.name,
        )


# ---------------------------------------------------------------------------
# The names content already imports
# ---------------------------------------------------------------------------

SpecCable = RodSpec
"""The rod spec under the name three task directories import it by.

The class is :class:`RodSpec`. It was ``SpecCable`` while a rod was only ever a
cable, and the domain has been called ``rod`` since it was registered -- so the
class and the domain disagreed on what the kind of thing is called, in a module
whose whole job is to be the one place a rod is described.

Kept as an alias rather than swept, because renaming a class in
``robosimstudio/`` to tidy it is not a reason to edit ten files under
``tasks/``: the four constructors in ``cable_route`` go on meaning exactly
what they meant. It is one
object under two names, so a subclass of either is a subclass of both.
"""

EntityCable = EntityRod
"""The rod entity under the name three task directories import it by. See
:data:`SpecCable` for why both names exist and why neither is a shim."""
