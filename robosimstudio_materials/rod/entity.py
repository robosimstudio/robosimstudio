# SPDX-License-Identifier: Apache-2.0
"""Building a rod, and the pegs it wraps: the mechanism ``cable_rod`` takes.

The capsule chain, its latch, its tuning views and the peg fixtures. Every
measurement of a particular cable is a default a task overrides:
:class:`RodSpec`'s defaults are the weave reference cable (0.9144 m of
11 mm-radius cable at 350 kg/m³, ~176 g as built, stretch 2e5, bend 0.04).
``cable_route`` takes them as they stand; ``threaded_needle``'s shorter, stiffer
suture subclasses :class:`EntityRod` and overrides most of them.

This module calls no ``register_entity``: each task's own ``rod.py`` binds
``cable_rod``, ``cable_peg`` and ``cable_peg_round`` to these classes.

The rod is 28 overlapping capsules chained by cable joints
(``ModelBuilder.add_rod``), solved by rigid VBD/AVBD inside the coupled-proxy
solver: bodies, not particles. It is built in the ``deformables`` pass
(:attr:`RodSpec.stage`).

Friction is a *scene* statement here, made at bind: the reference fills every
shape to ``mu = 2.0`` and then carves out the three materials that matter --
cable 60, gripper blades 60 (grip = sqrt(60·60) = 60), and the resting surface
0.02 (cable-vs-table = sqrt(60·0.02) ≈ 1.1). The absurd-looking 60s are the
point: Newton combines friction as ``sqrt(mu_a · mu_b)``, so pairwise targets
this far apart are only reachable by splitting them across the two shapes.

The pegs are cylinders, which :class:`~robosimstudio.specs.StaticSpec` cannot
say (its procedural shape is a box). ``EntityPeg`` is the static domain's post
(:class:`~robosimstudio.domains.static.posts.EntityPost`), which reads the spec's
``half_extents`` as ``(radius, _, half_height)``.

``EntityPegRound`` is ManipulationNet's F1 spool, imported whole. Its groove
holds a route the hand has let go of, where a post lets it ride off; its
collider is the authored mesh, since its hull would be a cylinder.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
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
    "add_capsule_chain",
    "fixture_shape_cfg",
]


CableDriveSpec = RodMaterialSpec
"""Alias of :class:`~robosimstudio_materials.rod.spec.RodMaterialSpec`, under the name ``cable_route`` imports."""


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

    Returns the inputs untouched when nothing was drawn (no bow, wave, shift or
    yaw), so a straight spawn stays byte-identical and recorded episodes do not
    change. The four are independent: a shift or a yaw with no bow still moves
    the rod.

    The quaternions turn with the curve: ``add_rod`` reads them as each
    segment's frame, and straight frames on a bent rod snap on the first step.
    """
    lo, hi = (float(v) for v in getattr(spec, "bow_range", (0.0, 0.0)))
    # The episode's unit fraction, scaled by the spec's range. Read from
    # episode.overrides, not the seed: overrides form the model key, so a
    # seed-derived bow would not trigger a rebuild and every episode would get
    # episode 0's rod. The same holds for wave, shift and yaw.
    overrides = dict(getattr(episode, "overrides", ()) or ())
    unit = overrides.get("cable_bow")
    amplitude = 0.0
    if unit is not None and (abs(hi - lo) > 0.0 or abs(lo) > 0.0):
        amplitude = 0.5 * (lo + hi) + 0.5 * (hi - lo) * float(unit)
    if abs(amplitude) < 1.0e-6:
        amplitude = 0.0
    wave_lo, wave_hi = (float(v) for v in getattr(spec, "wave_range", (0.0, 0.0)))
    u_wave = overrides.get("cable_wave")
    w_amp = 0.0
    if u_wave is not None and abs(wave_hi - wave_lo) > 0.0:
        w_amp = 0.5 * (wave_lo + wave_hi) + 0.5 * (wave_hi - wave_lo) * float(u_wave)
    shift_lo, shift_hi = (float(v) for v in getattr(spec, "shift_range", (0.0, 0.0)))
    u_shift = overrides.get("cable_shift")
    shift = 0.0
    if u_shift is not None and abs(shift_hi - shift_lo) > 0.0:
        shift = 0.5 * (shift_lo + shift_hi) + 0.5 * (shift_hi - shift_lo) * float(u_shift)
    yaw_lo, yaw_hi = (float(v) for v in getattr(spec, "yaw_range", (0.0, 0.0)))
    u_yaw = overrides.get("cable_yaw")
    yaw = 0.0
    if u_yaw is not None and abs(yaw_hi - yaw_lo) > 0.0:
        yaw = 0.5 * (yaw_lo + yaw_hi) + 0.5 * (yaw_hi - yaw_lo) * float(u_yaw)
    if amplitude == 0.0 and w_amp == 0.0 and shift == 0.0 and yaw == 0.0:
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
    # Half-cosine bow plus an optional sin(2*pi*t) wave. The wave is
    # antisymmetric, so slack meanders across the axis without pushing the
    # extremes past its own amplitude. The near-side seat boxes start about
    # 60 mm off the spawn line, and a rod that reaches one scores before the
    # robot moves, so keep amplitudes under that.
    offset = amplitude * np.sin(np.pi * t)  # zero at both tips, max at the middle
    if w_amp != 0.0:
        offset = offset + w_amp * np.sin(2.0 * np.pi * t)

    # Shift and yaw move the tips too, which a bow leaves in place:
    #   cable_shift  slides the whole rod along the lateral axis.
    #   cable_yaw    turns it in the bench plane about its own midpoint.
    if shift != 0.0:
        offset = offset + shift

    bowed = pts + offset[:, None] * lateral[None, :]

    if yaw != 0.0:
        ang = math.radians(yaw)
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
"""The latch's position gains [N/m, N.s/m], as the reference tuned them. They
are forces, so :meth:`EntityRod.hold_gains` scales them by segment mass."""

ROLL_STEP = 0.1
"""How far :meth:`EntityRod.roll` turns a pinned capsule per call at most [rad]:
a full turn in 63 control steps at 60 Hz. The rate is not what the cable
feels -- 0.26 rad (15 degrees) per step puts the same uniform 0.348 turns into
the middle third of the twist example's cables with straightness 1.000
throughout, as 0.05 does (``probe6.py``, GPU 1) -- it is what a hand
turning a cable end looks like on the page, and it keeps a slider yanked
across its range from being a jump the joint never sees."""


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


def add_capsule_chain(
    builder: newton.ModelBuilder,
    rod: newton.Rod,
    *,
    cfg: Any,
    color,
    gains: Mapping[str, float],
    label: str,
    wrap_in_articulation: bool = True,
) -> None:
    """One capsule chain on cable joints, ``add_rod`` in the ``rod=`` form of Newton 1.6.

    Geometry and frames on a :class:`newton.Rod`, the eight stiffness and
    damping numbers stated outright, keyed as :class:`RodMaterialSpec` names
    them (``stretch_ke`` ... ``twist_kd``). All eight, because the builder pairs
    a slot's damping with its stiffness only while *both* are left ``None`` --
    a shear stiffness given alone would silently take 0 damping -- so whoever
    calls this resolves the pairs first (:meth:`RodSpec.drive_gains` does for
    a cable). ``body_frame_origin="com"`` puts each capsule's frame at its
    centre with local ``+z`` toward the next node, which is what
    :meth:`EntityRod.centerline` reads back. A module function rather than a
    method so a kind built *out of* capsule chains -- a brush's bristles --
    builds them the way a cable is built without being a cable.

    The chain's joints are wrapped here, not by ``add_rod``: Newton 1.7's own
    wrapping also gives the first capsule a free joint to the world, which
    adds seven coordinates ahead of every cable's and reorders its joint span,
    and VBD solves a free joint as no constraint at all. Wrapped by hand the
    articulation is the one Newton 1.6 built -- the cable joints alone, under
    ``{label}_articulation`` -- so the model a cable is solved on is unchanged.
    """
    _, joints = builder.add_rod(
        rod=rod,
        body_frame_origin="com",
        cfg=cfg,
        color=color,
        stretch_stiffness=float(gains["stretch_ke"]),
        stretch_damping=float(gains["stretch_kd"]),
        shear_stiffness=float(gains["shear_ke"]),
        shear_damping=float(gains["shear_kd"]),
        bend_stiffness=float(gains["bend_ke"]),
        bend_damping=float(gains["bend_kd"]),
        twist_stiffness=float(gains["twist_ke"]),
        twist_damping=float(gains["twist_kd"]),
        label=label,
        wrap_in_articulation=False,
    )
    if wrap_in_articulation and joints:
        builder.add_articulation(list(joints), label=f"{label}_articulation" if label else None)


def rod_slot_dofs(model: newton.Model, joints) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The dof rows of the rod joints among ``joints``, one array per slot: ``(stretch, shear, bend, twist)``.

    ``add_rod`` builds each rod joint with four dofs in that fixed order
    (``STRETCH=0, SHEAR=1, BEND=2, TWIST=3``). Told apart by type, so the welds
    and a free joint that share a span with them are left out. A module function
    beside :func:`add_capsule_chain` for the same reason: a cable's drive
    tunables and a brush's bristles both read their chains' slots here.
    """
    starts = model.joint_qd_start.numpy()
    types = model.joint_type.numpy()
    slots: tuple[list[int], list[int], list[int], list[int]] = ([], [], [], [])
    for j in (int(v) for v in joints):
        if int(types[j]) != int(newton.JointType.ROD):
            continue
        begin = int(starts[j])
        for slot, rows in enumerate(slots):
            rows.append(begin + slot)
    return tuple(np.asarray(rows, dtype=np.int64) for rows in slots)


def _segment_mass(spec: RodSpec) -> float:
    """One capsule's mass [kg], from what the spec declares.

    Analytic, because the gains are wanted before a body exists. Only the ratio
    matters, so the overlap ``add_rod`` builds in cancels.
    """
    return float(spec.density) * math.pi * float(spec.radius) ** 2 * (float(spec.length) / int(spec.segments))


class EntityRod(Centerline, Entity):
    """The capsule-chain cable: built by ``add_rod(rod=newton.Rod(...))``, indexed for everyone else."""

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

        Empty in a single-world build; :meth:`centerline` falls back to
        :attr:`_half_heights` for any world not in here. A chord runs up to 8%
        over the nominal 12 mm, so another world's is ~1 mm off on
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
        self._pins: dict[int, dict[str, Any]] = {}
        """Capsules held by :meth:`pin`, by model body: which segment and world, the pose held, the roll since, and the mass to give back."""

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

        Shared by :meth:`build` and :meth:`place`, so a wave's rods match a
        serial build.
        """
        spec = self.spec
        # ``start`` is in the work-surface frame; lift it to world. The frame
        # has no rotation, so ``direction`` is unchanged.
        if spec.nodes is not None:
            # A stated polyline: lift every node; ``newton.Rod`` parallel-transports
            # the frames along it when given ``None``, so an untwisted polyline
            # is built untwisted (the deprecated chain form aligned each frame
            # to its own leg, which put a twist between legs that turn).
            return [
                wp.vec3(*[float(v) for v in self.placed((*node, 0.0, 0.0, 0.0, 1.0))[:3]]) for node in spec.nodes
            ], None
        start = wp.vec3(*[float(v) for v in self.placed((*spec.start, 0.0, 0.0, 0.0, 1.0))[:3]])
        direction = wp.vec3(*[float(v) for v in spec.direction])
        rod = newton.Rod.create_straight(
            start=start,
            direction=direction,
            length=float(spec.length),
            segment_count=int(spec.segments),
            radius=float(spec.radius),
        )
        # The same points and frames the deprecated helper gave, to the bit
        # (``tests/test_rod_new_api.py``), as lists so the bow can turn them.
        points = [wp.vec3(*[float(v) for v in p]) for p in rod.points]
        quats = [wp.quat(*[float(v) for v in q]) for q in rod.quaternions]
        # Only the straight spawn is bowed.
        return _bow_cable(points, quats, spec, episode)

    def _add_rod(self, builder: newton.ModelBuilder, points, quats) -> None:
        """``add_rod`` at this spec's own numbers. Shared by :meth:`build` and :meth:`place`.

        :func:`add_capsule_chain` with this rod's geometry and
        :meth:`RodSpec.drive_gains`, which is where the ``None`` pairs are
        resolved.
        """
        spec = self.spec
        add_capsule_chain(
            builder,
            newton.Rod(points, quaternions=quats, radius=float(spec.radius)),
            cfg=self._shape_cfg(),
            color=spec.color,
            gains=spec.drive_gains(),
            label=spec.name,
            wrap_in_articulation=self.wrap_rod_in_articulation(),
        )

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        spec = self.spec
        if spec.closed:
            # Refused rather than built open: a capsule chain has two ends.
            raise NotImplementedError(
                f"rod {spec.name!r}: a closed rod (a band, an O-ring) is built by the foreign-engine rod; route the "
                "'rod' kind to backend('superdex')"
            )
        if spec.joins:
            # Refused rather than ignored: a capsule chain has no node to weld.
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

        ``(tips, tips + palms)``: the mesh decomposition wants the tips, the
        collision filter also wants the body the tool-centre point is measured in
        (the palm). Read from
        :attr:`~robosimstudio.specs.GripperSpec.finger_bodies` and
        :attr:`~robosimstudio.specs.GripperSpec.tcp_body`, already templated per
        arm by :meth:`~robosimstudio.specs.GripperSpec.for_arm`. Empty with no
        robot or a bare arm.
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

        Which tips: the hands' own declaration (see :meth:`_hand_body_labels`).
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
            # Legitimate (primitive fingers have no mesh), so warn rather than raise.
            print(f"[cable] no fingertip meshes matched {sorted(tip_labels)}; skipping the tip decomposition")
            return
        # Record each tip's colour before decomposing: ``approximate_meshes``
        # leaves its new shapes (hulls and kept visual copies) uncoloured, and
        # :meth:`apply_display` repaints them with this.
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

        # Every shape the decomposition added, under its tip's colour.
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
        # Every static shape (no body), not a named list.
        statics = [t for t in range(builder.shape_count) if int(builder.shape_body[t]) < 0]
        for s in range(builder.shape_count):
            if int(builder.shape_body[s]) in blade_bodies:
                for t in statics:
                    builder.add_shape_collision_filter_pair(s, t)

    def apply_display(self, builder: newton.ModelBuilder) -> None:
        """The rod's colour to the rod, and the robot's to the robot.

        The fingertip hulls and kept meshes :meth:`_decompose_working_tips` adds
        fall inside this entity's range but belong to the robot. They are
        repainted with the tip's own recorded colour
        (:data:`~robosimstudio.core.render.palette.ROBOT_DARK` if none); left
        unpainted they would take a build-order palette colour. :meth:`bind`
        makes the same split for the solver via ``_rod_spans``.
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
    """Every phase of this rod carries a world axis: the latch is a row per hand
    per world; the segment and gripper indices in those rows are the replicated
    model's; ``post_solve`` runs against the stepping model; and the anchor is
    solved once and posed into every world.

    ``bind`` gets the template, so ``self.model`` is wrong for solver-side work:
    ``eval_ik`` on it silently writes one world's slice. Use :attr:`_fleet_model`.
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

        Callers pass either a per-world view (template length; tools above the
        boundary) or the full replicated array (this entity's anchor), told
        apart by length.
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

        # Grip bodies and TCP offsets from the embodiment's own index, not a
        # label match. Empty with no robot; then nothing sets ``_hold_flag``.
        if self.embodiment is not None:
            self._grip_bodies = {name: arm.ee_body for name, arm in self.embodiment.arms.items()}
            # Off the hand, not the arm: the two YAM hands put the TCP in
            # opposite directions along the wrist's z.
            self._tcp_offsets = {
                name: np.asarray(hand.tcp_offset if hand else (0.0, 0.0, 0.0), dtype=np.float64)
                for name, hand in self.embodiment.hands.items()
            }
        # Latch state from ``latch.py`` (the module repro bundles copy), sized
        # per world. The five attribute names are read by the repro taps in
        # ``cable_route/staging.py`` and the picker below.
        self._latch = _build_latch(model, {}, num_worlds=self._num_worlds)
        sinks = self._latch.aux_sinks
        self._hold_flag = sinks["hold_flag"]
        self._hold_seg = sinks["hold_seg"]
        self._hold_grip = sinks["hold_grip"]
        self._hold_rel = sinks["hold_rel"]
        self._hold_gains = sinks["hold_gains"]

    # ------------------------------------------------------------- phase 2b
    SPAWN_FIELDS: ClassVar[tuple[str, ...]] = ("cable_bow", "cable_shift", "cable_yaw", "cable_wave")
    """The sampled keys :func:`_bow_cable` reads: this rod's drawn spawn. Named on
    the consuming class, unlike :data:`~robosimstudio_bench.run.fleet.PLACEABLE_FIELDS`,
    because ``cable_bow`` means nothing to a task with no rod."""

    places_spawn_per_world: ClassVar[bool] = True
    """Whether :meth:`place` writes this class's drawn spawn into each world. When
    True the wave planner drops :attr:`SPAWN_FIELDS` from the model key
    (:func:`~robosimstudio_bench.run.fleet.spawn_fields_placed_per_world`). A
    subclass that draws its rod differently in ``build``, or adds geometry
    :meth:`place` does not rewrite, must set False, or every world runs the build
    episode's rod. The Cosserat rod declares nothing, which reads as False."""

    @classmethod
    def fields_placed_per_world(cls, spec) -> tuple[str, ...]:
        """The sampled fields a replicated build of ``spec`` writes per world.

        The statics' counterpart is
        :meth:`~robosimstudio.domains.static.entity.EntityStatic.places_per_world`.
        Empty for a rod with stated ``nodes``, which :func:`_bow_cable` never
        touches.
        """
        return cls.SPAWN_FIELDS if cls.places_spawn_per_world and getattr(spec, "nodes", None) is None else ()

    def place(self, model: newton.Model, episode, *, world: int = 0) -> None:
        """Draw this world's rod into the replicated model, before solvers exist.

        ``add_rod`` sizes each capsule from its chord, so a bow changes
        ``body_q``, mass, inertia, ``shape_scale`` and the cable joints'
        ``joint_X_p``/``joint_X_c``. The rows are copied from a finalized scratch
        ``add_rod`` of this world's episode, so world *k* matches a serial build
        of episode *k* to the bit.

        Running before ``_assemble`` keeps each world's rest shape its own:
        SolverVBD takes its rest bend and twist from the model at construction,
        and a rod re-posed afterwards snaps on the first step. No solver
        notification is needed.
        """
        if not self.fields_placed_per_world(self.spec) or not len(self.bodies):
            return
        points, quats = self._spawn(episode)
        scratch = newton.ModelBuilder()
        # As in ``build``: VBD's custom attributes must exist before ``add_rod``.
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
        # Finalized, so inverse masses match to the bit (the builder's float64
        # reciprocal is half an ULP off: 4.883e-04 on ~8192).
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
        # The weld's ``parent_xform`` is the anchored capsule's build pose, so
        # each world's row is rewritten from its own rod. Otherwise the end is
        # tied to world 0's start, and VBD's hard FIXED joint yanks it there.
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

        Tips only, from :meth:`_hand_body_labels`: friction is about the pads,
        and the palm set would put grip friction on a whole wrist link. A shape
        not found here takes :attr:`RodSpec.mu_base` (2.0), not
        :attr:`RodSpec.mu_gripper` (60).
        """
        tip_labels, _ = self._hand_body_labels()
        if not tip_labels:
            return set()
        labels = list(model.body_label)
        bodies = {b for b in range(model.body_count) if labels[b].split("/")[-1] in tip_labels}
        body = model.shape_body.numpy()
        return {s for s in range(model.shape_count) if int(body[s]) in bodies}

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
        own = {int(s) for s in self.shapes.tolist()}
        # EVERY WORLD'S CAPSULES, on a replicated model. ``self.shapes`` are
        # template indices, and this pass runs on the replicated model (see
        # ``_compile_multi``: "every entity writes its own rows per world"), so
        # reading them as rows wrote ``mu_cable`` into world 0's capsules alone.
        # Every other world's rod then fell through to ``mu_base``: measured on
        # a cable_route/port_easy wave, mu 60 in world 0 and 2.0 in worlds 1+,
        # i.e. a pad-on-cable grip of sqrt(2 x 60) = 11 instead of 60. The
        # terminal carry's shallow pinch let go of the cable in those worlds,
        # and a rod pressed against a fixture moved 0.4-0.8 mm differently from
        # world 0 and the single-world build, which agreed within 0.02-0.06.
        worlds = self.worlds
        if worlds is not None and int(model.shape_count) != int(worlds.shapes_per_world):
            own = {int(worlds.shape(s, w)) for w in range(self._num_worlds) for s in own}
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
        self._apply_dahl(model)

    def _cable_joints(self, model: newton.Model) -> np.ndarray:
        """The rod joints among :attr:`joints`, template-indexed.

        :attr:`joints` is the rod's whole span, which also holds the anchor
        welds (``FIXED``); the stiffness rows and the Dahl rows belong to the
        ``ROD`` joints alone.
        """
        if not len(self.joints):
            return np.zeros(0, dtype=np.int64)
        types = model.joint_type.numpy()
        return np.asarray([int(j) for j in self.joints if int(types[j]) == int(newton.JointType.ROD)], dtype=np.int64)

    def _apply_dahl(self, model: newton.Model) -> None:
        """Author the cable joints' Dahl rows, in every world, when the spec asks for memory.

        Here rather than in ``build`` because the attributes are per-joint
        *model* arrays: the builder's custom-attribute defaults are what
        ``finalize`` writes, and this runs on the finalized model -- the
        replicated one in a wave -- in the window before the solver is
        constructed. That window is the whole point: ``SolverVBD`` decides at
        construction whether to run its Dahl kernels at all, from whether any
        joint has both rows positive, and binds the arrays by pointer after
        that. A rod built at ``dahl_eps_max=0`` writes nothing, so a scene with
        no memory in it is the scene it always was.

        The anchor welds are left at 0: they are ``FIXED`` joints and the
        kernels skip them, but a row that reads 2.0 on a weld is a row somebody
        will one day have to explain.
        """
        spec = self.spec
        if float(spec.dahl_eps_max) <= 0.0:
            return
        vbd = getattr(model, "vbd", None)
        eps_array = getattr(vbd, "dahl_eps_max", None)
        tau_array = getattr(vbd, "dahl_tau", None)
        if eps_array is None or tau_array is None:
            raise RuntimeError(
                f"rod {self.name!r}: dahl_eps_max={spec.dahl_eps_max} needs VBD's custom attributes on the model, "
                "and this model has none; the rod is built on a backend that is not VBD"
            )
        joints = self._cable_joints(model)
        if not len(joints):
            return
        worlds = self.worlds
        if worlds is not None:
            joints = np.asarray(
                [worlds.joint(int(j), w) for w in range(self._num_worlds) for j in joints], dtype=np.int64
            )
        eps = eps_array.numpy()
        tau = tau_array.numpy()
        eps[joints] = float(spec.dahl_eps_max)
        tau[joints] = float(spec.dahl_tau)
        eps_array.assign(eps)
        tau_array.assign(tau)

    # -------------------------------------------------------------- phase 3
    def contribute_hooks(self, hooks) -> None:
        """Recover joint coordinates from the body transforms VBD writes.

        The proxy solver integrates the cable in maximal coordinates;
        ``eval_ik`` writes them back through the joints so ``joint_q`` stays
        the state's mirror -- the reference runs it every substep, after the
        solve.
        """
        # The stepping model, not the template (see :attr:`world_aware`).
        model = self._fleet_model

        def recover(state, dt) -> None:
            del dt
            newton.eval_ik(model, state, state.joint_q, state.joint_qd)

        hooks.post_solve.append(recover)

        if not self.spec.allow_grasp_latch:
            return
        kp_hold, kd_hold = self.hold_gains()
        # Seed the latch's gain array (see ``latch.hold_segment``).
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

        ``None`` when the robot has no usable arm (not an error). A named arm
        that does not exist raises ``KeyError``.
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
        """Pre-pose the anchor arm (:meth:`_anchor_arm`) gripping the cable, the wrap variant's anchor.

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
        # Position + approach only, roll free: demanding the grasp roll parked
        # the hand 40 mm off, shoving the rod across the table. Chained through
        # a point above the grasp so the descent stays in one IK branch.
        # Surface-frame, like ``start``; the IK wants world.
        target = np.asarray(self.placed((*spec.anchor_grasp, 0.0, 0.0, 0.0, 1.0))[:3], dtype=np.float64)
        # Seed with this world's coordinates, template-shaped, as the solver takes.
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
        # FK over the whole fleet on the stepping model; the latch reads poses from it.
        newton.eval_fk(self._fleet_model, pipeline.state_0.joint_q, pipeline.state_0.joint_qd, pipeline.state_0)
        wp.copy(pipeline.state_1.body_q, pipeline.state_0.body_q)
        # Latch the anchored segment to the anchor hand, or the rod slides
        # through the pads under drag. Without ``allow_grasp_latch`` it stays a
        # plain pinch.
        if self._hold_flag is not None and spec.allow_grasp_latch:
            self.hold_segment(True, pipeline.state_0, arm=arm, slot=1, world=world)

    # -------------------------------------------------------------- phase 5
    def reset(self, pipeline, episode, *, world: int = 0) -> None:
        """Straighten every world (first call only), then re-pose this world's anchor.

        The embodiment's reset sends every arm home, so without the re-pose the
        anchor arm leaves its pinch and sweeps the rod 150 mm across the table.
        :meth:`_straighten` clears every latch, so it must run before any anchor
        is re-set, or ``(:init (anchor-held cable))`` reads 0.0.
        """
        del episode
        if self._rest_joint_q is None:
            return
        if world == 0:
            self._release_pins(pipeline)
            self._straighten(pipeline)
        self._pose_anchor(pipeline, world=world)

    def _straighten(self, pipeline) -> None:
        """Put every world's rod back on its build pose. Once per wave.

        Restores the rod's coordinate run in each world's block of the fleet's
        rest field.

        ``eval_fk`` skips CABLE joints (``newton/_src/sim/articulation.py``), so
        the ``joint_q`` restore moves no capsule; ``reset_solvers()`` returns the
        deformable island to each world's built rest. With three worlds drawn
        differently, shoving every capsule 150 mm along y and resetting returns
        all three exactly (0.0000 mm). If the solver stops capturing that rest,
        restore ``body_q`` per world here.
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
        # anchors and warm starts; without the reset a plug was flung 5 m in
        # 5 steps.
        reset_solvers = getattr(pipeline, "reset_solvers", None)
        if reset_solvers is not None:
            reset_solvers()
        # Clear every latch before reset() re-anchors (the anchor is slot 1).
        if self._hold_flag is not None:
            self._hold_flag.zero_()

    # ---------------------------------------------------------------- control
    def hold_gains(self) -> tuple[float, float]:
        """This rod's latch gains, scaled off the reference's by segment mass.

        400 N/m and 12 N.s/m were tuned on the reference rod (11 mm, 28 capsules
        over 0.91 m, ~4.4 g a segment). Scaling by segment mass keeps the natural
        frequency and damping ratio fixed; the reference scales by 1.0.
        Unscaled, the weave's rod (5 mm, 42 capsules over 0.50 m, ~0.33 g, 13x
        lighter) was flung off the table at 0.9 m/s or diverged.
        """
        scale = _segment_mass(self.spec) / _segment_mass(RodSpec())
        return HOLD_KP_REFERENCE * scale, HOLD_KD_REFERENCE * scale

    def hold_force_max(self) -> float:
        """The latch's force ceiling [N]: four rod weights.

        Sized by the whole rod: lifting by one segment carries the full weight.
        One weight lifts, ~0.6 drags the half-rod against the table's combined
        mu of 1.1, and four covers both in a brisk carry. Under a contact fight
        the latch saturates and slips (see ``latch.hold_segment``).
        """
        return 4.0 * _segment_mass(self.spec) * self.spec.segments * 9.81

    # --------------------------------------------------------------- tunables
    def _drive_dofs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The cable joints' dof rows, one array per slot: ``(stretch, shear, bend, twist)``.

        :func:`rod_slot_dofs` over the rod's span. Four arrays rather than the
        two pairs this used to return, so a slider moves one slot: the pairs
        are equal only while the spec left ``shear_*``/``twist_*`` at ``None``.
        """
        return rod_slot_dofs(self.model, self.joints)

    def tunables(self, pipeline=None):
        """The rod's stiffness, its contact material, and the latch -- live.

        What the studio's Parameters folder renders when the cable is
        selected. Three groups:

        * **drive** -- stretch, shear, bend and twist stiffness/damping over
          the cable joints' own dof rows, a slot each, and the Dahl pair.
          ``~``: the write is live but reaches the
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

        stretch_dofs, shear_dofs, bend_dofs, twist_dofs = self._drive_dofs()
        if len(bend_dofs):
            joints = tuple(int(j) for j in self.joints)
            cable_joints = self._cable_joints(self.model)

            class _DriveWriter(ArrayWriter):
                flags = 1 << 1
                """``ModelFlags.JOINT_DOF_PROPERTIES``, as ``JointWriter`` spells it."""

                ROWS: ClassVar[dict[str, tuple[str, np.ndarray]]] = {
                    "bend_ke": ("joint_target_ke", bend_dofs),
                    "bend_kd": ("joint_target_kd", bend_dofs),
                    "stretch_ke": ("joint_target_ke", stretch_dofs),
                    "stretch_kd": ("joint_target_kd", stretch_dofs),
                    "shear_ke": ("joint_target_ke", shear_dofs),
                    "shear_kd": ("joint_target_kd", shear_dofs),
                    "twist_ke": ("joint_target_ke", twist_dofs),
                    "twist_kd": ("joint_target_kd", twist_dofs),
                    "dahl_eps_max": ("vbd.dahl_eps_max", cable_joints),
                    "dahl_tau": ("vbd.dahl_tau", cable_joints),
                }

                LIVE_ROWS: ClassVar[frozenset[str]] = frozenset({"dahl_eps_max", "dahl_tau"})
                """Rows VBD reads by pointer every substep: written, not notified
                (``flags=0``) and not relanded -- there is no penalty copy of them.
                Per *joint*, where the stiffness rows are per dof."""

                def read(self) -> dict[str, float]:
                    return {name: self._read(attr, rows) for name, (attr, rows) in self.ROWS.items()}

                def __call__(self, field_name: str, value) -> bool:
                    row = self.ROWS.get(field_name)
                    if row is None:
                        return False
                    if field_name in self.LIVE_ROWS:
                        return self._write(row[0], row[1], float(value))
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
                    cache, so a slider alone leaves the rod unchanged. After
                    the write-and-notify, each cable joint's four slots are
                    recomputed in place: ``k_max`` becomes the new stiffness,
                    the live AVBD penalty is clamped under it (so softening
                    takes effect now), and the damping row follows. In place,
                    so a captured graph's pointers stay valid; if newton
                    renames these internals, this falls back to
                    write-and-notify rather than raising -- which is where
                    Newton 1.6 lands: it has no ``joint_penalty_k_max``, and its
                    ``notify_model_changed(JOINT_DOF_PROPERTIES)`` reseeds the
                    ROD slots' penalty, floor and damping itself
                    (``_refresh_joint_material_params``), so the notify alone is
                    the reland there.
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
                    summary=(
                        f"Stretch, shear, bend and twist over {len(cable_joints)} cable joints. Bend is why it "
                        "feels stiff; twist is whether a roll reaches the far end. The plastic pair is live, and "
                        "inert when built at 0."
                    ),
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
        """The body-latch answer to :meth:`~robosimstudio_materials.rod.centerline.Centerline.hold`."""
        return self.hold_segment(on, state, arm=arm, slot=slot, at=at, world=world)

    def hold_segment(
        self, on: bool, state=None, *, arm: str = "right", slot: int = 0, at: int | None = None, world: int = 0
    ) -> int:
        """Engage or drop one latch slot: the segment nearest ``arm``'s jaws.

        Records the segment's current pose in that gripper's frame, so the rod
        rides the hand exactly as taken. Slot 0 is the working hand's, slot 1
        the anchor's. Returns the segment index, or -1.

        Args:
            state: Template-indexed ``body_q`` source (a single-world state, or
                a fleet ``WorldView``'s slice).
            world: Selects the latch row and the global body indices written to
                it.

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

    # ---------------------------------------------------------- a hand's hold
    def pin(self, pipeline, *, segment: int = -1, on: bool = True, world: int = 0) -> int:
        """Hold one capsule where it is now, kinematically, or let it go. Returns its body index.

        A hand with a cable end in a hard latch: the capsule keeps the pose it
        has when pinned, whatever the chain does, until :meth:`roll` or another
        pin moves it. Its inverse mass and inertia go to zero and it is flagged
        kinematic in the model and in every island
        (:func:`robosimstudio.core.tuning.set_kinematic` says why both are
        needed once a robot shares the scene). Let go, it gets its mass back
        where it is, so a cable released mid-roll unwinds from there.
        Pinning a pinned capsule re-reads its pose; releasing a free one does
        nothing. A reset releases every pin. This is a kinematic hold and not
        a grasp: the page's object handle is a clamped wrench, which cannot
        roll a capsule (its torque is bounded by the capsule's own inertia,
        3.2e-6 N*m on this chain's 6 mm capsules, against the 1e-3 to 1e-1
        N*m a turn of twist takes; lifting the bound whips both cables into
        loops -- ``probe4.py``, GPU 1).

        Args:
            segment: Which capsule, chain-indexed; ``-1`` is the free end of a
                cable clamped at its start.
            on: ``False`` releases it.
            world: The fleet world, for a batched build.
        """
        from robosimstudio.core.tuning import notify, set_kinematic

        body = self._global(int(self.bodies[int(segment)]), world)
        model = self._fleet_model
        pin = self._pins.get(body)
        if not on:
            if pin is None:
                return body
            inv_mass, inv_inertia = model.body_inv_mass.numpy(), model.body_inv_inertia.numpy()
            inv_mass[body], inv_inertia[body] = pin["inv_mass"], pin["inv_inertia"]
            model.body_inv_mass.assign(inv_mass)
            model.body_inv_inertia.assign(inv_inertia)
            set_kinematic(pipeline, [body], False, restore=pin["flags"])
            notify(pipeline, int(newton.ModelFlags.BODY_INERTIAL_PROPERTIES))
            del self._pins[body]
            return body
        pose = np.asarray(pipeline.state_0.body_q.numpy()[body], dtype=np.float64).copy()
        if pin is None:
            inv_mass, inv_inertia = model.body_inv_mass.numpy(), model.body_inv_inertia.numpy()
            pin = {
                "segment": int(segment),
                "world": int(world),
                "inv_mass": float(inv_mass[body]),
                "inv_inertia": inv_inertia[body].copy(),
            }
            inv_mass[body], inv_inertia[body] = 0.0, 0.0
            model.body_inv_mass.assign(inv_mass)
            model.body_inv_inertia.assign(inv_inertia)
            pin["flags"] = set_kinematic(pipeline, [body], True)
            notify(pipeline, int(newton.ModelFlags.BODY_INERTIAL_PROPERTIES))
            self._pins[body] = pin
        pin["pose"], pin["angle"] = pose, 0.0
        self._write_pin(pipeline, body)
        return body

    def roll(self, pipeline, angle: float, *, segment: int = -1, max_step: float = ROLL_STEP, world: int = 0) -> float:
        """Turn a pinned capsule about its own axis to ``angle`` from where it was pinned, and say where it got to.

        Right-handed about the capsule's axis (its local ``+Z``, pointing along
        the chain), in radians, absolute since the pin: calling with the same
        angle every step holds it there, and ``0.0`` rolls it back. The turn
        is rate-limited to ``max_step`` per call because the capsule is
        kinematic: a jump of a full turn is the same frame as no turn at all
        to the joint and to :func:`~robosimstudio_materials.rod.measures.twist_turns`,
        and a jump of half a turn is a shock. Pins the capsule first when it
        is free. Returns the angle it now stands at, which lags the request
        while it is still turning.
        """
        from robosimstudio.core.math import quat_from_axis_angle, quat_multiply, quat_to_mat

        body = self._global(int(self.bodies[int(segment)]), world)
        if body not in self._pins:
            self.pin(pipeline, segment=segment, world=world)
        pin = self._pins[body]
        step = float(max_step) if max_step and max_step > 0 else float("inf")
        pin["angle"] += float(np.clip(float(angle) - pin["angle"], -step, step))
        rest = pin["pose"]
        axis = quat_to_mat(rest[3:7])[:, 2]
        self._write_pin(pipeline, body, quat_multiply(quat_from_axis_angle(axis, pin["angle"]), rest[3:7]))
        return float(pin["angle"])

    def _write_pin(self, pipeline, body: int, quat=None) -> None:
        """Put a pinned capsule's pose into both state buffers, at rest: what the solver reads next substep."""
        pose = self._pins[body]["pose"].copy()
        if quat is not None:
            pose[3:7] = quat
        for state in (pipeline.state_0, pipeline.state_1):
            q = state.body_q.numpy()
            q[body] = pose
            state.body_q.assign(q)
            qd = state.body_qd.numpy()
            qd[body] = 0.0
            state.body_qd.assign(qd)

    def _release_pins(self, pipeline) -> None:
        """Every pin let go, with the mass given back: a reset's rod is nobody's."""
        for pin in list(self._pins.values()):
            self.pin(pipeline, segment=pin["segment"], world=pin["world"], on=False)

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

        ``segments + 1`` points, the same polyline the Cosserat rod returns
        (centres would span only (n-1)/n of the length, 4% short at 24).
        ``add_rod`` places the endpoints at local ``(0, 0, +-half_height)`` with
        ``body_frame_origin="com"``: each segment's first endpoint, then the last
        segment's second.

        :meth:`positions` still returns centres, one per body, for the latch picker.
        """
        if not len(self.bodies):
            return np.zeros((0, 3), dtype=np.float64)
        from robosimstudio.core.math import quat_to_mat

        body_q = np.asarray(snapshot.body_q)[self.bodies]
        # Per-world chords when the wave drew its own rod (up to 8% of a 12 mm
        # capsule on the scored endpoint otherwise).
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
        """Latch flags (``held`` = slot 0, ``anchor_held`` = slot 1, this world's rows
        via the snapshot context), ``n_segments`` always, and ``length_m`` when the
        snapshot has ``body_q``.

        The latch is a warp array, so ``(anchor-held cable)`` can only be read
        here. A rod :meth:`centerline` cannot read raises rather than reporting
        0 m.
        """
        shape = {"n_segments": float(len(self.bodies))}
        if getattr(snapshot, "body_q", None) is not None:
            chain = self.centerline(snapshot)
            shape["length_m"] = float(np.linalg.norm(np.diff(chain, axis=0), axis=1).sum()) if len(chain) >= 2 else 0.0
        if self._hold_flag is None:
            return {"held": 0.0, "anchor_held": 0.0, **shape}
        context = getattr(snapshot, "ctx", snapshot)
        world = int(getattr(context, "world", 0))
        flags = self._hold_flag.numpy()
        held, anchor = self._latch.row(0, world), self._latch.row(1, world)
        return {"held": float(flags[held]), "anchor_held": float(flags[anchor]), **shape}


EntityPeg = EntityPost
"""Alias of :class:`~robosimstudio.domains.static.posts.EntityPost`, under the name the cable suites import."""


PEG_MESH_SCALE = 0.01
"""The vendored fixture mesh is in centimetres (see its PROVENANCE.md)."""

PEG_SHAFT_RADIUS = 0.00475
PEG_FLANGE_RADIUS = 0.0125
PEG_ROUND_HEIGHT = 0.0235
"""ManipulationNet's F1, measured off the mesh at :data:`PEG_MESH_SCALE`. Restated
so a spec builds without loading trimesh; :class:`EntityPegRound` asserts them
against the mesh."""


class EntityPegRound(PlacedAtBuild, EntityStatic):
    """ManipulationNet's F1 round peg: a spool, imported rather than drawn.

    The groove captures the rod between a 12.5 mm flange below and 17 mm of
    shaft above, so a seated cable stays seated where a plain post lets it ride
    up and off. A cable seats on the 4.75 mm shaft, 9.75 mm from the axis (a
    13 mm post would put it at 18); the weave documents' seat regions use that.

    The mesh goes in whole (its convex hull would be a cylinder).
    ``spec.half_extents`` is ignored; the mesh path comes from ``spec.asset``,
    and its dimensions are asserted, so a different mesh is refused.
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
        # ``_shape_frame``, not ``spec.xform``: the episode may have moved this
        # fixture, and the declared pose would leave the spool behind its seat.
        body, xform = self._shape_frame()
        # The procedural peg's contact settings, so only the shape differs.
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
"""Alias of :class:`RodSpec` for task code."""

EntityCable = EntityRod
"""Alias of :class:`EntityRod` for task code."""
