# SPDX-License-Identifier: Apache-2.0
"""The robot: imported, indexed, actuated, and checked.

Everything here is a check the port learned to make the hard way:

* **Register MuJoCo's custom attributes before importing.** Without
  ``SolverMuJoCo.register_custom_attributes(builder)`` the importer drops
  MJCF-specific attributes, above all ``gravcomp="1"``, and a
  position-servoed arm sags several centimetres under gravity while every
  target is tracked correctly.
* **Re-key those attributes after collapsing fixed joints.** They are
  BODY-frequency, so the collapse renumbers the bodies and leaves the values
  keyed by the old indices -- half the arm ends up reading the arena's entries.
  See :func:`~robosimstudio.core.builder.rekey_custom_attributes`.
* **Assert the actuation mode.** Newton reads ``joint_target_q`` only for a DOF
  whose actuator is a ``<position>``; a torque-actuated import takes zero
  torque from every target and droops while IK solves happily against nothing.
* **Find joints by label, not by index.** An importer's numbering is not an
  interface, and an ambiguous match raises rather than taking the first.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.core.builder import restate_body_attribute, set_body_attribute
from robosimstudio.core.suction import Suction
from robosimstudio.paths import resolve_asset
from robosimstudio.robots.registry import resolve_gripper
from robosimstudio.scene.build.bodies import preserve_body_inertia
from robosimstudio.scene.build.colliders import copy_shape_material
from robosimstudio.scene.build.imports import strip_free_bodies
from robosimstudio.scene.entity import Entity, label_index
from robosimstudio.solvers import ArmIK, Gripper
from robosimstudio.specs import IDENTITY as _IDENTITY_XFORM
from robosimstudio.specs import ArmSpec, EmbodimentSpec, GripperSpec

__all__ = ["ArmIndex", "EntityEmbodiment"]


def _in_last_articulation(builder: newton.ModelBuilder, body: int) -> bool:
    """Whether ``body`` belongs to the articulation the builder added last.

    Newton's own rule for ``parent_body``: a hand may only be welded onto the
    most recently added articulation. Read off ``articulation_start``, which
    holds each articulation's first joint, and the joint whose child is the body.
    A body no articulation claims (a free body) is not in the last one.
    """
    starts = [int(v) for v in builder.articulation_start]
    if not starts:
        return True
    joints = [j for j in range(builder.joint_count) if int(builder.joint_child[j]) == int(body)]
    if not joints:
        return False
    return int(joints[0]) >= starts[-1]


def _limb_bodies(joint_parent, joint_child, joints: Sequence[int], seed_joints: Sequence[int]) -> set[int]:
    """Every body hanging off ``seed_joints``, following the tree downwards only.

    What makes one arm of a two-armed robot nameable without naming anything.
    An arm's own joints are resolved from its spec (the prefix or the explicit
    names ``bind`` already indexes with); their child bodies are its links, and
    descending from there picks up the fingers, the wrist camera and any welded
    hand for free.

    **Downwards only**, which is the whole trick: both arms of a bimanual cell
    attach to the world, so an ascent from either one meets the other there.
    Descending stops at the leaves, so a shared base or torso ends up in neither
    arm -- correctly, since it is not part of either.
    """
    children: dict[int, list[int]] = {}
    for j in joints:
        child = int(joint_child[j])
        if child >= 0:
            children.setdefault(int(joint_parent[j]), []).append(child)
    bodies: set[int] = set()
    stack = [int(joint_child[j]) for j in seed_joints if int(joint_child[j]) >= 0]
    while stack:
        body = stack.pop()
        if body in bodies:
            continue
        bodies.add(body)
        stack.extend(children.get(body, ()))
    return bodies


class ArmIndex:
    """One arm's indices in a finalized model.

    Attributes:
        name: The arm's name.
        arm_coords: ``joint_q`` indices of the actuated joints.
        arm_dofs: ``joint_qd`` indices of the same joints.
        gripper_coords: The driven finger joints' ``joint_q`` indices.
        gripper_dofs: The same in ``joint_qd``.
        ee_body: Tool-centre body.
        base_body: The body this arm is mounted on: the one the spec names
            (:attr:`~robosimstudio.specs.ArmSpec.base_body_name`), else the
            body its first joint hangs on -- the URDF's root link, which is
            what a base-frame solver and a Cartesian action expressed in the
            robot's own axes read their frame from. -1 only when that body
            is the world, as it is after an import collapses the mount's
            fixed joints away.
        finger_bodies: Per side, the finger link bodies used by grasp tests.
        bodies: Every body this arm carries -- its links, its fingers, its
            wrist camera, and any hand welded to it. Walked *downward* from the
            arm's own joints, so on a two-armed robot each arm gets its own and
            a shared torso belongs to neither.
        shapes: This arm's shapes. What a clearance query is asked about: "how
            close does this arm come to that one" is a question about two sets
            of colliders, and deriving them at each call site by matching body
            labels is the failure mode
            :meth:`EntityEmbodiment.gripper_bodies` exists to avoid.
        gripper_open: Finger travel at full open [m].
        gripper_open_q: Per driven finger joint, the coordinate at full open.
            Empty means the mirrored prismatic convention.
        gripper_closed_q: The same at full closed.
        gripper_follower_coords: ``joint_q`` indices of finger joints an equality
            or mimic constraint carries. Never written a target -- the solver
            ignores it -- but read, because on a mirrored pair whose twin is a
            follower they are what makes the gap exactly measurable.
        gripper_follower_dofs: ``joint_qd`` indices of the same follower joints,
            so a reset can zero their rates along with the driven fingers'.
        gripper_span: Fingertip separation at full open [m].
        home_q: Home joint positions, or an empty array.
        approach_axis: Gripper-local axis pointing from the wrist towards the
            grasp point, copied off the arm's :class:`GripperSpec`. Carried here
            so a caller composing a grasp frame -- the studio, a routine -- reads
            it off the index it already holds instead of re-finding the hand.
        close_axis: Gripper-local axis the jaws close along.
    """

    __slots__ = (
        "approach_axis",
        "arm_coords",
        "arm_dofs",
        "base_body",
        "bodies",
        "close_axis",
        "ee_body",
        "finger_bodies",
        "gripper_closed_q",
        "gripper_coords",
        "gripper_dofs",
        "gripper_follower_coords",
        "gripper_follower_dofs",
        "gripper_open",
        "gripper_open_q",
        "gripper_span",
        "home_q",
        "name",
        "shapes",
    )

    def __init__(self, name: str) -> None:
        self.name = name
        self.arm_coords: list[int] = []
        self.arm_dofs: list[int] = []
        self.gripper_coords: list[int] = []
        self.gripper_dofs: list[int] = []
        self.ee_body: int = -1
        self.base_body: int = -1
        self.finger_bodies: tuple[tuple[int, ...], ...] = ()
        self.bodies: tuple[int, ...] = ()
        self.shapes: np.ndarray = np.zeros(0, dtype=np.int32)
        self.gripper_open: float = 0.04
        self.gripper_open_q: tuple[float, ...] = ()
        self.gripper_closed_q: tuple[float, ...] = ()
        self.gripper_follower_coords: tuple[int, ...] = ()
        self.gripper_follower_dofs: tuple[int, ...] = ()
        self.gripper_span: float = 0.08
        self.home_q: np.ndarray = np.zeros(0, dtype=np.float32)
        self.approach_axis: np.ndarray = np.array([0.0, 0.0, 1.0])
        self.close_axis: np.ndarray = np.array([0.0, 1.0, 0.0])

    def __repr__(self) -> str:
        return f"ArmIndex({self.name!r}, coords={self.arm_coords}, fingers={self.gripper_coords}, ee={self.ee_body})"


def _mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume of a closed triangle mesh [m^3], by the divergence theorem.

    Here rather than through trimesh: this runs at scene build, and the repo
    keeps trimesh to offline asset builders.
    """
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


class EntityEmbodiment(Entity):
    """A robot in a scene: import, index, actuate.

    Args:
        spec: The embodiment.
        pose: Where its base goes, in world coordinates. The compiler derives
            this from the scene's work-surface frame and the spec's
            ``base_offset``; identity means the asset already places itself,
            which is what every asset does until it is re-origined.
        world: Which world.
    """

    domain = "articulated"

    spec: EmbodimentSpec

    def __init__(self, spec: EmbodimentSpec, *, pose=None, world: int = 0, scene_resident: bool = False) -> None:
        super().__init__(spec, world=world)
        self.scene_resident = bool(scene_resident)
        """Whether this robot already exists inside the task's scene asset.

        Separate from ``spec.asset is None``, which is what this used to be
        keyed on, because the two stopped meaning the same thing once an
        embodiment could be *both* importable on its own and present inside a
        monolithic export. the tool-sorting suite's Panda is exactly that: its native pairing
        takes the one baked into the robosuite arena, and any other pairing
        imports the extracted standalone asset.
        """
        self.pose = pose if pose is not None else spec.base_xform
        # The hand on each arm, looked up and templated. Resolved once here
        # rather than at each read: it is a registry lookup, and the alternative
        # is every consumer knowing the arm-then-embodiment precedence rule.
        self.hands: dict[str, GripperSpec | None] = {a.name: resolve_gripper(spec, a) for a in spec.arms}
        self.arms: dict[str, ArmIndex] = {}
        self.iks: dict[str, ArmIK] = {}
        self.grippers: dict[str, Gripper] = {}
        self.suctions: dict[str, Suction] = {}
        """Per arm whose hand is a vacuum cup, its :class:`~robosimstudio.core.suction.Suction`, filled at bind."""
        self._suction_rows: dict[str, dict[int, int]] = {}
        self._suction_owners: dict[int, str] = {}
        self._hand_bodies: dict[str, tuple[int, int]] = {}
        """Per arm whose hand is welded on, the body range that hand's import added."""
        self.bodies: np.ndarray = np.zeros(0, dtype=np.int32)
        """Every body in this robot's tree, filled at bind."""
        self.arm_shapes: tuple[tuple[int, ...], ...] = ()
        """Per arm, its colliding shapes -- filled only when
        :attr:`~robosimstudio.specs.EmbodimentSpec.cross_arm_collisions`
        grouped them. Reported rather than re-derived, since the grouping is
        what decides which pairs were left live."""
        self.trunk_shapes: tuple[int, ...] = ()
        """Colliding shapes no arm claimed, filled alongside :attr:`arm_shapes`."""
        self._home_target: np.ndarray | None = None
        self._body_cache: list[int] | None = None

    # -------------------------------------------------------------- phase 1
    def build_into(self, builder: newton.ModelBuilder, episode) -> None:
        """Import, colour, then run the robot's own appearance pass.

        The order is the point. :meth:`Entity.apply_display` paints every shape
        in this entity's range with the spec's one ``color``, which is right for
        a converter-authored asset whose per-geom ``rgba`` is noise -- and it
        would flatten anything a finer pass had already painted. So the hook
        runs *after* it, and repaints the few shapes one colour gets wrong.
        """
        super().build_into(builder, episode)
        if self.spec.appearance is not None:
            self.spec.appearance(builder, self.range.shapes)

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        import os

        from newton.solvers import SolverMuJoCo

        # Harvest MJCF-specific attributes -- above all gravcomp. Registration
        # must happen before the import, not after.
        SolverMuJoCo.register_custom_attributes(builder)

        if self.resident:
            # The robot arrives inside the task's scene asset; importing it here
            # would put a second one in the model.
            attached = [a.name for a in self.spec.arms if (h := self.hands.get(a.name)) is not None and h.asset]
            if attached:
                raise RuntimeError(
                    f"{self.spec.name}: arm(s) {attached} carry a hand with an asset of its own, but this robot "
                    "arrives inside the scene's import and is never imported here -- so there is no wrist for the "
                    "weld to bolt to. A resident robot can only take a hand its own export already contains."
                )
            if self.spec.cross_arm_collisions:
                raise RuntimeError(
                    f"{self.spec.name}: cross_arm_collisions needs this robot's own import, and this one arrives "
                    "inside the scene's asset -- which was imported with self-collisions off, so the pairs this "
                    "would re-admit have already been filtered and cannot be recovered. Import the robot "
                    "separately (scene_resident=False with a stripped scene) to ask for it."
                )
            return

        path = resolve_asset(self.spec.asset)
        source = str(path)
        stripped = strip_free_bodies(source) if path.suffix.lower() in (".xml", ".mjcf") else None
        shape_start = builder.shape_count
        # The mount, composed onto every root the asset declares. Never through
        # ``override_root_xform``: both YAM cells are two separate articulation
        # roots in one file, and overriding would collapse the pair onto a
        # single pose rather than carrying them rigidly to a new place.
        mount = self._mount_xform()
        try:
            if path.suffix.lower() in (".xml", ".mjcf"):
                if self.spec.collapse_fixed_joints:
                    from robosimstudio.scene.build.imports import add_mjcf_collapsed

                    add_mjcf_collapsed(
                        builder,
                        stripped or source,
                        scale=self.spec.asset.scale,
                        floating=False,
                        enable_self_collisions=self.spec.import_self_collisions,
                        **({} if mount is None else {"xform": mount}),
                    )
                else:
                    builder.add_mjcf(
                        stripped or source,
                        scale=self.spec.asset.scale,
                        floating=False,
                        collapse_fixed_joints=False,
                        enable_self_collisions=self.spec.import_self_collisions,
                        verbose=False,
                        **({} if mount is None else {"xform": mount}),
                    )
            elif path.suffix.lower() in (".usd", ".usda", ".usdc", ".usdz"):
                # An Isaac-exported arm, which arrives as a stage rather than a
                # description file. add_usd takes the stage's own units, so the
                # asset scale is not forwarded here any more than it is for a
                # static -- see EntityStatic._add_usd.
                from robosimstudio.scene.build.imports import add_usd_stage

                add_usd_stage(
                    builder,
                    source,
                    collapse_fixed_joints=self.spec.collapse_fixed_joints,
                    verbose=False,
                    **({} if mount is None else {"xform": mount}),
                )
            elif path.suffix.lower() == ".urdf":
                joints_before = builder.joint_count
                builder.add_urdf(
                    source,
                    scale=self.spec.asset.scale,
                    floating=False,
                    # Collapsed below, folding only this import's fixed joints:
                    # the importer's own flag folds the whole builder.
                    collapse_fixed_joints=False,
                    enable_self_collisions=self.spec.import_self_collisions,
                    # The URDF's own <inertial> tags, not a recomputation from
                    # geometry. A benchmark's robot is characterised by the
                    # inertias its own file states -- a system-identification fit
                    # made against them is not a fit against whatever a mesh
                    # volume implies -- and Newton's importer would otherwise
                    # honour the file only because its default happens to.
                    # Stated, so a change of default is not a change of dynamics.
                    ignore_inertial_definitions=False,
                    # A URDF's <dynamics damping> as the drive damping, Newton 1.6's
                    # reading; 1.7 warns when this is left unsaid and 1.8 switches
                    # the default to passive joint damping, which neither VBD nor
                    # XPBD reads. Stated for the same reason as the line above.
                    joint_damping_mode="drive",
                    **({} if mount is None else {"xform": mount}),
                )
                if self.spec.collapse_fixed_joints:
                    from robosimstudio.scene.build.imports import collapse_own_fixed_joints

                    collapse_own_fixed_joints(builder, joints_before)
            else:
                raise ValueError(f"embodiment {self.spec.name!r}: unsupported asset {path.suffix!r}")
        finally:
            if stripped is not None:
                os.unlink(stripped)

        self._attach_hands(builder)

        if self.spec.show_collision_shapes:
            # The importer leaves colliding geoms invisible; for an asset with
            # one dual-purpose geom per link that hides the whole robot.
            #
            # Sites are excluded, and that exclusion is the whole reason this is
            # a filter rather than a blanket OR: MuJoCo sites import *invisible
            # by design* (they are frames, not geometry), Newton's importer
            # drops their authored rgba, and their shapes therefore fall to the
            # build-order palette. Un-hiding them puts a handful of arbitrarily
            # coloured markers on camera, floating inside the gripper.
            visible = int(newton.ShapeFlags.VISIBLE)
            site = int(newton.ShapeFlags.SITE)
            for s in range(shape_start, builder.shape_count):
                if int(builder.shape_flags[s]) & site:
                    continue
                builder.shape_flags[s] = int(builder.shape_flags[s]) | visible

        if self.spec.gravcomp_restate:
            # The blanket fallback. Correct here because after the strip and
            # collapse every surviving body is an arm body, and every one
            # carries gravcomp="1" in the MJCF.
            restate_body_attribute(builder, "mujoco:gravcomp", 1.0)

        if self.spec.hull_colliders:
            self._hull_colliders(builder)

        if self.spec.decompose_colliders:
            self._decompose_colliders(builder, shape_start)

        if self.spec.cross_arm_collisions and not self.spec.enable_self_collisions:
            # After the hands and the hulls, because both add shapes and every
            # shape this robot owns has to land in exactly one group.
            self._filter_within_arms(builder, shape_start)

    # --------------------------------------------------------- self-collision
    def _filter_within_arms(self, builder: newton.ModelBuilder, shape_start: int) -> None:
        """Filter each arm against itself, and leave the arms free to hit each other.

        The import ran with self-collisions **on**, which is more than
        :attr:`~robosimstudio.specs.EmbodimentSpec.cross_arm_collisions`
        asks for: it also lets each hand's two fingers meet, and a hand whose
        pads cannot meet is a hand that stalls against its own geometry (the
        measurement is in
        :func:`~robosimstudio.scene.build.colliders.filter_self_collisions`). So the
        blanket filter is put back *per arm* and the arm-against-arm pairs are
        what survive.

        Membership is walked, not matched. Each arm's own joints (the spec's
        prefix or names, which is the same resolution ``bind`` indexes with)
        give a set of child bodies, and the walk descends the joint graph from
        there -- so the fingers, the wrist camera and any welded hand come along
        without being named, and the walk stops at the world rather than
        crossing into the other arm. Anything the arms do not claim -- a shared
        torso, and the two mount pucks this cell's collapse welds to the world
        -- is a *trunk*, filtered against everything exactly as it was before,
        because "the arms collide" is not a licence to change what the base
        does.
        """
        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        mine = [s for s in range(shape_start, builder.shape_count) if int(builder.shape_flags[s]) & collide]
        if len(self.spec.arms) < 2:
            raise ValueError(
                f"{self.spec.name}: cross_arm_collisions needs two arms to be about, and this robot declares "
                f"{len(self.spec.arms)}. On a one-armed robot the blanket filter already says what this would."
            )

        labels = builder.joint_label
        joints = range(self.range.joints[0], builder.joint_count)
        groups: list[list[int]] = []
        claimed: set[int] = set()
        for arm in self.spec.arms:
            arm_j, finger_j = self._arm_joints(labels, arm, joint_types=builder.joint_type)
            if not arm_j:
                raise RuntimeError(f"{self.spec.name} arm {arm.name!r} matched no joints; cannot group its shapes")
            bodies = _limb_bodies(builder.joint_parent, builder.joint_child, joints, (*arm_j, *finger_j))
            shapes = [s for s in mine if int(builder.shape_body[s]) in bodies]
            if not shapes:
                raise RuntimeError(
                    f"{self.spec.name} arm {arm.name!r} owns no colliding shapes, so it cannot be told from the "
                    "other arm. cross_arm_collisions would filter nothing and change nothing."
                )
            groups.append(shapes)
            claimed.update(shapes)

        from robosimstudio.scene.build.colliders import filter_self_collisions

        for shapes in groups:
            filter_self_collisions(builder, shapes)
        trunk = [s for s in mine if s not in claimed]
        filter_self_collisions(builder, trunk)
        for base in trunk:
            for shape in claimed:
                builder.add_shape_collision_filter_pair(base, shape)
        self.arm_shapes = tuple(tuple(g) for g in groups)
        self.trunk_shapes = tuple(trunk)

    # ------------------------------------------------------------- the weld
    def _attach_hands(self, builder: newton.ModelBuilder) -> None:
        """Bolt on every hand that carries a file of its own.

        Newton composes hierarchically: ``add_mjcf(parent_body=idx,
        floating=False)`` puts the imported root in the parent's articulation
        with a fixed joint, which is exactly a hand bolted to a wrist. Newton's
        own ``test_robot_composer`` builds a UR5e and a Robotiq 2F-85 this way.

        Three things this does beyond the import, each of which a hand needs
        and none of which the importer can infer:

        * **The driven finger joints are put in POSITION mode.** A hand's own
          file drives its jaw however its author liked -- the 2F-85 ships one
          tendon-coupled ``<general>`` actuator -- and Newton maps anything that
          is not a ``<position>`` to a direct-control source that ignores
          ``joint_target_q`` entirely. Every controller in this stack lands on
          ``joint_target_q``, so an attached hand is *declared* to be driven that
          way rather than being asserted into it: ``apply_actuation``'s check
          exists to catch an arm nobody meant to leave torque-actuated, and a
          welded hand is not that case.
        * **Its MuJoCo actuators are switched to ``JOINT_TARGET``.** Leaving the
          hand's own actuator live alongside the joint target makes the pair
          fight; the same setting appears in Newton's composer test, with a link
          to the mujoco_warp discussion behind it.
        * **Its bodies get ``gravcomp="1"``.** The arm convention, extended to
          the mass just bolted onto it: every shipped robot's file states
          gravity compensation for itself, and ``compose_yam_i2rt`` bakes the
          same onto the hand it extracts, because a hand hanging off a wrist
          sags. A vendored hand file states none -- Menagerie's 2F-85 is
          authored for its own tendon actuator, which does not need the help --
          so without this the welded mass hangs on the wrist servos: ~3 mm of
          static droop, and any loop that re-anchors targets to the *measured*
          state (``idle_action``'s hold, the studio gizmo's catch-up re-sync)
          turns that droop into a steady descent, measured at 47.6 mm over 4 s
          on ``tape_peel @ ur5`` and 0.000 mm with the restate. Restating 1.0
          on a hand whose file already declares it changes nothing.

        On a multi-root asset only the arm declared last can take a hand: Newton
        allows only the most recently added articulation as ``parent_body``, so
        on a robot whose file declares two arms the second arm's hand attaches
        and the first arm's would raise. Both YAM cells are that shape. A hand
        named on the last arm's own :attr:`~robosimstudio.specs.ArmSpec.gripper`
        is welded (``yam_bimanual_suction``'s cup); a hand for any other arm is
        refused here with that reason, and ``registry.admits`` refuses a hand
        named for the whole robot before the compiler gets this far.
        """
        wanted = [(arm, hand) for arm in self.spec.arms if (hand := self.hands.get(arm.name)) and hand.asset]
        if not wanted:
            return
        # This robot's own roots, not the builder's: the compiler builds the
        # scene's furniture first, and a stage that arrives as an articulation
        # would otherwise be counted as one of the arm's.
        roots = sum(1 for j in range(self.range.joints[0], builder.joint_count) if int(builder.joint_parent[j]) < 0)
        ctrl_source = builder.custom_attributes.get("mujoco:ctrl_source")
        for arm, hand in wanted:
            parent = self._wrist_body(builder, arm, hand)
            if roots > 1 and not _in_last_articulation(builder, parent):
                raise RuntimeError(
                    f"{self.spec.name}: cannot weld hand {hand.name!r} onto arm {arm.name!r} of an asset that "
                    f"declares {roots} articulation roots -- Newton accepts only the most recently added "
                    "articulation as a weld parent, and this arm is not it. Put the hand on the arm the asset "
                    "declares last, or split the asset per arm."
                )
            if hand.finger_names and self.spec.gripper_actuation is None and hand.actuation is None:
                # Switching the hand's own actuator to JOINT_TARGET below is what
                # stops it fighting the joint target -- and it also means nothing
                # drives the jaw unless somebody states gains. A limp hand that
                # reports its command faithfully is the worst failure this could
                # have, so it is refused at build rather than discovered in a
                # grasp that never closes.
                raise ValueError(
                    f"{self.spec.name} {arm.name}: attached hand {hand.name!r} states no actuation, and neither does "
                    "the robot. A welded hand is driven from joint_target_q, so its gains cannot come from its own "
                    "file -- set GripperSpec.actuation or EmbodimentSpec.gripper_actuation."
                )
            path = resolve_asset(hand.asset)
            if path.suffix.lower() not in (".xml", ".mjcf"):
                raise ValueError(
                    f"{self.spec.name} {arm.name}: hand {hand.name!r} is attached as {path.suffix!r}; "
                    "the weld imports MJCF only"
                )
            joint_start = builder.joint_count
            body_start = builder.body_count
            actuators_before = 0 if ctrl_source is None else len(ctrl_source.values)
            # The arm's word on where its flange is wins over the hand's: a hand
            # measured against one flange names it (mount_body) and carries the
            # transform; a hand that bolts to anything leaves it to the arm.
            mount = arm.gripper_mount if arm.gripper_mount is not None else hand.mount_xform
            builder.add_mjcf(
                str(path),
                xform=wp.transform(
                    wp.vec3(*(float(v) for v in mount[:3])),
                    wp.quat(*(float(v) for v in mount[3:7])),
                ),
                scale=hand.asset.scale,
                floating=False,
                parent_body=parent,
                collapse_fixed_joints=False,
                enable_self_collisions=self.spec.import_self_collisions,
                verbose=False,
            )
            if ctrl_source is not None and len(ctrl_source.values) > actuators_before:
                # 0 is SolverMuJoCo.CtrlSource.JOINT_TARGET. Spelled as the enum
                # rather than the number, but reached lazily: importing the
                # solver at module scope would pull mujoco into every build.
                from newton.solvers import SolverMuJoCo

                target = int(SolverMuJoCo.CtrlSource.JOINT_TARGET)
                for a in range(actuators_before, len(ctrl_source.values)):
                    ctrl_source.values[a] = target
            # A hand hanging off a wrist sags -- the third bullet above.
            set_body_attribute(builder, "mujoco:gravcomp", 1.0, range(body_start, builder.body_count))
            self._hand_bodies[arm.name] = (body_start, builder.body_count)
            self._drive_by_target(builder, arm, hand, joint_start)

    def _hand_body(self, labels: Sequence[str], arm: ArmSpec, name: str) -> int:
        """The body ``name`` of the hand on ``arm``: looked up in that hand's own import first.

        Two arms carrying the same welded hand import its labels twice, so a
        name the hand gives (its pad, its tool centre) matches once per arm in
        the whole model and once in its own import. A hand the arm's asset
        already contains, or a name its import does not hold once, is looked up
        in the whole model as before.
        """
        span = self._hand_bodies.get(arm.name)
        if span is not None:
            lo, hi = span
            hits = [lo + i for i, label in enumerate(labels[lo:hi]) if label.split("/")[-1] == name]
            if len(hits) == 1:
                return hits[0]
        return label_index(labels, name)

    def after_scene_built(self, builder: newton.ModelBuilder, entities: Sequence[Any]) -> None:
        """Couple each vacuum cup to every part the scene built: one disabled weld row per part.

        Called by the compiler once every entity has built and before the model
        is finalized, the only window in which both ends of a weld exist and rows
        can still be added. A part is a dynamic body of another entity with a box
        or mesh shape, the faces a cup can seal on. The rows are MuJoCo weld
        equalities on Newton's own ``mujoco:equality_constraint`` table, pad as
        body 1 and part as body 2, switched off;
        :class:`~robosimstudio.core.suction.Suction` turns one on at hold with
        the pose the part has under the face then.
        """
        cups = [(arm, hand) for arm in self.spec.arms if (hand := self.hands.get(arm.name)) and hand.suction_pad]
        if not cups:
            return
        from newton.solvers import SolverMuJoCo

        kinematic = int(newton.BodyFlags.KINEMATIC)
        flat = (int(newton.GeoType.BOX), int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH))
        faced = {int(builder.shape_body[s]) for s in range(builder.shape_count) if int(builder.shape_type[s]) in flat}
        owners: dict[int, str] = {}
        for entity in entities:
            span = getattr(getattr(entity, "range", None), "bodies", None)
            if entity is self or span is None:
                continue
            for b in range(*span):
                if b in faced and float(builder.body_mass[b]) > 0.0 and not (int(builder.body_flags[b]) & kinematic):
                    owners[b] = entity.name
        # Every attribute of the row's frequency is written, so finalize has no
        # column to pad with defaults and nothing to warn about.
        defaults = {
            key: attr.default
            for key, attr in builder.custom_attributes.items()
            if getattr(attr, "frequency", None) == "mujoco:equality_constraint"
        }
        for arm, hand in cups:
            pad = self._hand_body(builder.body_label, arm, hand.suction_pad)
            rows: dict[int, int] = {}
            for part in owners:
                values = {
                    **defaults,
                    "mujoco:equality_constraint_type": int(SolverMuJoCo.EqType.WELD),
                    "mujoco:equality_constraint_objtype": 1,  # mjOBJ_BODY
                    "mujoco:equality_constraint_body1": int(pad),
                    "mujoco:equality_constraint_body2": int(part),
                    "mujoco:equality_constraint_torquescale": 1.0,
                    "mujoco:equality_constraint_label": f"{self.spec.name}/{arm.name}/suction/{builder.body_label[part]}",
                    "mujoco:equality_constraint_enabled": False,
                    "mujoco:equality_constraint_world": builder.current_world,
                }
                indices = builder.add_custom_values(**values)
                rows[int(part)] = int(indices["mujoco:equality_constraint_type"])
            self._suction_rows[arm.name] = rows
        self._suction_owners = owners

    def _wrist_body(self, builder: newton.ModelBuilder, arm: ArmSpec, hand: GripperSpec) -> int:
        """The body a hand bolts to, by the same precedence ``bind`` uses.

        The hand's own ``mount_body``, then the arm's ``ee_body_name``, then the
        last arm joint's child -- which is the wrist by construction, and is what
        lets an arm whose hand names its own mount leave ``ee_body_name`` unsaid.
        """
        for label in (hand.mount_body, arm.ee_body_name):
            if label:
                return label_index(builder.body_label, label)
        arm_j, _ = self._arm_joints(builder.joint_label, arm, joint_types=builder.joint_type)
        if not arm_j:
            raise RuntimeError(
                f"{self.spec.name} arm {arm.name!r}: no joints matched, so the wrist to weld "
                f"hand {hand.name!r} onto cannot be derived"
            )
        return int(builder.joint_child[arm_j[-1]])

    def _drive_by_target(self, builder: newton.ModelBuilder, arm: ArmSpec, hand: GripperSpec, joint_start: int) -> None:
        """Put the welded hand's driven joints on ``joint_target_q``.

        Restricted to the joints the hand *names* as driven: a follower an
        equality carries stays exactly as its file imported it, because switching
        it to a position target is what makes a mirrored pair fight its own
        constraint.
        """
        driven = set(hand.finger_names)
        if not driven:
            return
        labels = builder.joint_label
        for j in range(joint_start, builder.joint_count):
            if labels[j].split("/")[-1] not in driven:
                continue
            begin = int(builder.joint_qd_start[j])
            end = (
                int(builder.joint_qd_start[j + 1])
                if j + 1 < len(builder.joint_qd_start)
                else len(builder.joint_target_mode)
            )
            for dof in range(begin, end):
                builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)

    @property
    def resident(self) -> bool:
        """Whether the robot is already in the scene rather than imported here."""
        return self.scene_resident or self.spec.asset is None

    def _mount_xform(self):
        """This robot's base pose as a Warp transform, or ``None`` at identity.

        ``None`` rather than an identity transform so the importer is called
        with exactly the arguments it was called with before mounting existed --
        an identity compose is a no-op in principle and a chance to differ in
        practice, and every shipped asset still places itself.
        """
        pose = np.asarray(self.pose, dtype=np.float64).reshape(-1)
        if pose.shape[0] != 7 or np.allclose(pose, _IDENTITY_XFORM):
            return None
        return wp.transform(
            wp.vec3(float(pose[0]), float(pose[1]), float(pose[2])),
            wp.quat(float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6])),
        )

    def _hull_colliders(self, builder: newton.ModelBuilder) -> None:
        """Convex-hull the arm's link meshes, keeping the visual meshes.

        Worth about an eighth of the physics budget and invisible on camera --
        the same trade Newton's own Franka example makes. Restricted to this
        entity's own shapes, so a scene's tools and scenery are untouched.
        """
        shapes = self.builder_shapes(builder)
        targets = [
            s
            for s in shapes
            if builder.shape_type[s] == newton.GeoType.MESH
            and (builder.shape_flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES))
        ]
        if targets:
            builder.approximate_meshes(method="convex_hull", shape_indices=targets, keep_visual_shapes=True)

    SLIVER_FLOOR = 1.0e-9
    """Smallest convex piece a decomposition may keep [m^3].

    One cubic millimetre, and the floor exists because MuJoCo's compiler refuses
    the model outright over a degenerate one -- "mesh volume is too small ... Try
    setting inertia to shell" -- which is a build failure, not a warning. A piece
    that small is a decomposition artefact rather than geometry: on this hand the
    smallest real piece is 2.2 mm^3 and the pad plate's pieces are hundreds.
    """

    def _decompose_colliders(self, builder: newton.ModelBuilder, shape_start: int) -> None:
        """Give the named links a convex DECOMPOSITION instead of a hull.

        Restricted to :attr:`EmbodimentSpec.decompose_colliders`, by body-label
        tail. A blanket decomposition would be both slower and pointless: an arm
        link is convex enough that its hull *is* its shape, and it is only where
        a link has a hole in it -- a fingertip yoke -- that the hull invents
        geometry. See the spec field for what that invention measured.

        The shape range is the import's own, not :meth:`builder_shapes`: this
        runs inside ``build``, and the entity's body range does not close until
        ``build_into`` returns, so asking which shapes are "mine" here answers
        none of them.

        **Written out rather than handed to ``approximate_meshes``,** which
        offers exactly this and cannot be used here: it decomposes each
        *connected component* separately, and this fingertip is three of them.
        Split that way the decomposition shatters -- 113 pieces where the whole
        mesh gives 45, twenty of them under a cubic millimetre, and MuJoCo then
        refuses to compile the model at all. Decomposed whole, with CoACD's own
        preprocessing to make the components one manifold, the same mesh comes
        back in 38-45 clean pieces at 1.24x its volume against the hull's 3.13x.

        The substitution is this package's usual one, for its usual reasons: the
        drawn mesh stays drawn and stops colliding, each piece takes the
        material of the collider it replaces rather than the builder's default
        (:func:`~robosimstudio.scene.build.colliders.copy_shape_material`), and the
        body's mass is held across the swap
        (:func:`~robosimstudio.scene.build.bodies.preserve_body_inertia`) because the
        original shape is still there carrying it.
        """
        tails = tuple(self.spec.decompose_colliders)
        labels = builder.shape_label
        collides = int(newton.ShapeFlags.COLLIDE_SHAPES)
        visible = int(newton.ShapeFlags.VISIBLE)
        targets = [
            s
            for s in range(shape_start, builder.shape_count)
            if builder.shape_type[s] == newton.GeoType.MESH
            and (builder.shape_flags[s] & collides)
            and any(tail in str(labels[s]).rsplit("/", 1)[-1] for tail in tails)
        ]
        if not targets:
            raise ValueError(
                f"embodiment {self.spec.name!r}: decompose_colliders={tails} matched no mesh collider; "
                f"a name that selects nothing is silent, and the link keeps its hull"
            )

        cache: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for source in targets:
            mesh = builder.shape_source[source]
            key = hash(mesh)
            pieces = cache.get(key)
            if pieces is None:
                pieces = self._convex_pieces(mesh, float(self.spec.decompose_threshold))
                cache[key] = pieces
            body = int(builder.shape_body[source])
            xform = builder.shape_transform[source]
            scale = builder.shape_scale[source]
            label = str(labels[source])
            # A piece has to inherit the collider it replaces, not the builder's
            # default -- its material (see `copy_shape_material`), its contact
            # gap, and CRUCIALLY its collision group. A shape that lands in the
            # default group is filtered against a different set from the rest of
            # the arm it belongs to, which does not fail loudly: it drops
            # arm-vs-scenery contacts, or invents arm-vs-self ones, and the only
            # symptom is a link that stops in the wrong place. `filter_parent`
            # is off for the same reason it is off in Newton's own version --
            # the pieces are siblings of a collider that already resolved its
            # parent relationship, not new geometry making its own.
            cfg = builder.default_shape_cfg.copy()
            cfg.density = 0.0  # the drawn mesh is still there carrying the mass
            cfg.gap = builder.shape_gap[source]
            cfg.margin = builder.shape_margin[source]
            cfg.is_solid = builder.shape_is_solid[source]
            cfg.collision_group = builder.shape_collision_group[source]
            cfg.collision_filter_parent = False
            filtered = sorted(
                other
                for pair in builder._shape_collision_filter_pairs
                for other in pair
                if source in pair and other != source and builder.shape_body[other] != body
            )
            # Drawn, and no longer colliding: the pieces do the colliding now.
            builder.shape_flags[source] = (int(builder.shape_flags[source]) | visible) & ~collides
            with preserve_body_inertia(builder, body):
                for index, (vertices, indices) in enumerate(pieces):
                    added = builder.add_shape_mesh(
                        body=body,
                        xform=xform,
                        mesh=newton.Mesh(vertices, indices),
                        scale=scale,
                        cfg=cfg,
                        label=f"{label}_convex_{index:02d}",
                    )
                    builder.shape_flags[added] = (int(builder.shape_flags[added]) | collides) & ~visible
                    copy_shape_material(builder, source, added)
                    for other in filtered:
                        builder.add_shape_collision_filter_pair(other, added)

    def _convex_pieces(self, mesh, threshold: float) -> list[tuple[np.ndarray, np.ndarray]]:
        """One mesh's convex decomposition, slivers dropped. See :meth:`_decompose_colliders`."""
        import coacd

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        indices = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
        parts = coacd.run_coacd(
            coacd.Mesh(vertices, indices),
            threshold=float(threshold),
            preprocess_mode="on",
            merge=True,
            mcts_nodes=20,
            mcts_iterations=5,
            mcts_max_depth=1,
            max_convex_hull=int(getattr(mesh, "maxhullvert", 64)),
        )
        kept: list[tuple[np.ndarray, np.ndarray]] = []
        for part_vertices, part_faces in parts:
            v = np.asarray(part_vertices, dtype=np.float32)
            f = np.asarray(part_faces, dtype=np.int32).reshape(-1)
            if _mesh_volume(v, f.reshape(-1, 3)) < self.SLIVER_FLOOR:
                continue
            kept.append((v, f))
        if not kept:
            raise RuntimeError(
                f"embodiment {self.spec.name!r}: convex decomposition left no piece above "
                f"{self.SLIVER_FLOOR * 1e9:.1f} mm^3; the collider would be empty"
            )
        return kept

    def owned_bodies(self) -> np.ndarray:
        """Every body of the robot, resident or imported -- what its solver island owns."""
        return np.asarray(list(self.bodies), dtype=np.int32)

    def coupling_bodies(self, *arms: str) -> list[int]:
        """The bodies allowed to touch a deformable: the embodiment's coupling surfaces.

        The spec's ``coupling_surfaces`` when it names any (body-label tails),
        else every arm's tool link and fingers -- :meth:`gripper_bodies`. What
        the composer mirrors into a deformable island as proxies, so a hand
        pushes a rod or pinches a towel without the arm's whole tree riding
        along.
        """
        tails = tuple(getattr(self.spec, "coupling_surfaces", ()) or ())
        if tails and self.model is not None:
            labels = list(self.model.body_label)
            mine = {int(b) for b in self.bodies}
            out = [
                i
                for i, label in enumerate(labels)
                if i in mine and any(str(label).rsplit("/", 1)[-1].endswith(t) for t in tails)
            ]
            if out:
                return out
        return self.gripper_bodies(*arms)

    def gripper_bodies(self, *arms: str) -> list[int]:
        """The hand bodies of the named arms: the tool-centre link and its fingers.

        What a coupled-solver factory needs to mirror a grasp into a second
        solver, and the reason it is a method rather than a label match at each
        call site. Matching ``tail.startswith(f"{arm}_") and "tip_left" in
        tail`` happens to describe this robot; on another it selects nothing,
        and selecting nothing is *silent* -- the proxy is simply never driven,
        so the cable is never held and the run reads as a grasp that slipped.

        Args:
            *arms: Arm names; empty means every arm.

        Returns:
            Body indices, de-duplicated, in a stable order.
        """
        wanted = list(arms) if arms else list(self.arms)
        out: list[int] = []
        for name in wanted:
            index = self.arms.get(name)
            if index is None:
                raise KeyError(f"embodiment {self.spec.name!r} has no arm {name!r}; has {sorted(self.arms)}")
            for body in (index.ee_body, *(b for side in index.finger_bodies for b in side)):
                if int(body) >= 0 and int(body) not in out:
                    out.append(int(body))
        return out

    def builder_shapes(self, builder: newton.ModelBuilder) -> list[int]:
        """Every shape belonging to this robot, pre-finalize."""
        bodies = set(self.body_indices(builder))
        return [s for s in range(builder.shape_count) if int(builder.shape_body[s]) in bodies]

    def body_indices(self, builder: newton.ModelBuilder) -> list[int]:
        """Every body in this robot's kinematic tree.

        For an imported embodiment that is simply the range it added. For one
        that arrives inside a scene asset there is no such range, so the tree is
        walked instead: start from an arm joint's child and follow the joint
        graph, which reaches the base, the links and the fingers and stops at
        the world -- so a scene's free-jointed tools and its fixed-jointed bins
        each stay in their own articulation.
        """
        if not self.resident:
            return list(range(*self.range.bodies))
        if self._body_cache is not None:
            return self._body_cache

        labels = builder.joint_label
        edges: dict[int, set[int]] = {}
        for j in range(builder.joint_count):
            parent, child = int(builder.joint_parent[j]), int(builder.joint_child[j])
            if parent < 0 or child < 0:
                continue
            edges.setdefault(parent, set()).add(child)
            edges.setdefault(child, set()).add(parent)

        seeds: set[int] = set()
        for arm in self.spec.arms:
            arm_j, finger_j = self._arm_joints(labels, arm, joint_types=builder.joint_type)
            for j in [*arm_j, *finger_j]:
                child = int(builder.joint_child[j])
                if child >= 0:
                    seeds.add(child)
                parent = int(builder.joint_parent[j])
                if parent >= 0:
                    seeds.add(parent)

        seen: set[int] = set()
        stack = list(seeds)
        while stack:
            body = stack.pop()
            if body in seen:
                continue
            seen.add(body)
            stack.extend(edges.get(body, ()) - seen)
        self._body_cache = sorted(seen)
        return self._body_cache

    @staticmethod
    def _unlimit_ctrlrange(builder: newton.ModelBuilder, dofs: Sequence[int]) -> int:
        """Widen the MuJoCo *actuator* range for ``dofs``, not just the joint's.

        Widening ``joint_limit_lower/upper`` alone does not unlimit anything the
        operator can feel, and the reason is a detail of the import: an asset
        whose ``<position inheritrange="1"/>`` makes each actuator's ``ctrlrange``
        *the joint's range* has that resolved into a
        ``mujoco:actuator_ctrlrange`` attribute while newton reads the MJCF. By
        the time actuation is applied the copy is already taken, so the solver
        goes on clamping the **target** at the old bound however wide the
        joint's own limit has become -- measured on ``yam_bimanual``, a joint
        commanded to 115 degrees sat at 90.0 with the joint limit alone
        widened and reached 115.0 with this.

        The attribute is per *actuator*. Which DOF each one drives is read from
        the importer's own ``mujoco:actuator_trnid`` table (its first component,
        for a joint transmission) rather than assumed from joint order: a robot
        that keeps its fixed joints as bodies has more joints than actuators,
        and counting joints there addresses nothing -- an earlier version that
        did widened 12 ranges on the collapsed YAM and 0 on every other shipped
        arm. Only an actuator that declares a range is rewritten; one without
        (``actuator_has_ctrlrange`` 0) clamps nothing and is left alone.

        Returns:
            How many actuator ranges were widened. 0 when the model carries no
            such attributes -- leaving the ranges alone is the safe failure, and
            the caller's joint limits are still widened.
        """
        attrs = builder.custom_attributes
        ranges = attrs.get("mujoco:actuator_ctrlrange")
        targets = attrs.get("mujoco:actuator_trnid")
        if ranges is None or targets is None or not ranges.values or len(targets.values) != len(ranges.values):
            return 0
        kinds = attrs.get("mujoco:actuator_trntype")
        declared = attrs.get("mujoco:actuator_has_ctrlrange")

        def entries(attr) -> list:
            # A list while the importer fills it, a dict by actuator index once
            # anything wrote a single entry: both are read in actuator order.
            values = attr.values
            return [values[key] for key in sorted(values)] if isinstance(values, dict) else list(values)

        keys = sorted(ranges.values) if isinstance(ranges.values, dict) else list(range(len(ranges.values)))
        target_of = entries(targets)
        kind_of = entries(kinds) if kinds is not None and len(kinds.values) == len(keys) else [0] * len(keys)
        has_range = entries(declared) if declared is not None and len(declared.values) == len(keys) else [1] * len(keys)
        wanted = {int(dof) for dof in dofs}
        written = 0
        for i, key in enumerate(keys):
            if int(kind_of[i]) != 0 or not int(has_range[i]) or int(target_of[i][0]) not in wanted:
                continue
            ranges.values[key] = type(ranges.values[key])(-1.0e6, 1.0e6)
            written += 1
        return written

    def apply_actuation(self, builder: newton.ModelBuilder) -> None:
        """Set the servo gains, and check the actuation mode is position.

        Called by the compiler after every entity has built, since the arm's
        DOF indices are only known once the joints exist.
        """
        if self.spec.hull_colliders:
            # Deferred to here for a scene-resident robot: its own shapes can
            # only be told from the scene's once the joints exist to walk.
            self._hull_colliders(builder)
        if self.spec.gravcomp:
            # Here rather than in build(), for two reasons that both matter: a
            # scene-resident robot's bodies can only be walked once its joints
            # exist, and these keys are body *indices* -- writing them before a
            # fixed-joint collapse would leave them pointing at renumbered
            # bodies, which is the failure rekey_custom_attributes exists for.
            set_body_attribute(builder, "mujoco:gravcomp", 1.0, self.body_indices(builder))
        unlimited_dofs: list[int] = []
        for arm in self.spec.arms:
            index = self._index_arm_builder(builder, arm)
            hand = self.hands.get(arm.name)
            for dofs, actuation, what in (
                (index.arm_dofs, self.spec.arm_actuation, "arm"),
                # The embodiment's gains win over the hand's own: a robot may
                # need a stiffer servo than the linkage's default, and the hand
                # cannot know which robot it is on.
                (index.gripper_dofs, self.spec.gripper_actuation or (hand.actuation if hand else None), "gripper"),
            ):
                if actuation is None:
                    continue
                for i, dof in enumerate(dofs):
                    if actuation.require_position_mode:
                        mode = newton.JointTargetMode(builder.joint_target_mode[dof])
                        if mode not in (newton.JointTargetMode.POSITION, newton.JointTargetMode.POSITION_VELOCITY):
                            raise RuntimeError(
                                f"{self.spec.name} {arm.name} {what} dof {dof} imported as "
                                f"JointTargetMode.{mode.name}, so joint_target_q is ignored for it. "
                                f"The asset must declare <position> actuators."
                            )
                    builder.joint_target_ke[dof] = actuation.ke_for(i)
                    builder.joint_target_kd[dof] = actuation.kd_for(i)
                    effort = actuation.effort_for(i)
                    if effort:
                        builder.joint_effort_limit[dof] = effort
                    if actuation.unlimited:
                        # Widened rather than flagged off: newton carries the
                        # bound as a number the solver and the IK both read, and
                        # a range this far outside any reachable pose is what
                        # "no limit" means to everything downstream without
                        # asking each reader to special-case an infinity.
                        builder.joint_limit_lower[dof] = -1.0e6
                        builder.joint_limit_upper[dof] = 1.0e6
                        unlimited_dofs.append(dof)
                    if actuation.joint_damping is not None:
                        builder.joint_damping[dof] = float(actuation.joint_damping)
                    if actuation.joint_friction is not None:
                        builder.joint_friction[dof] = float(actuation.joint_friction)
                if actuation.armature:
                    self._set_armature(builder, dofs, actuation.armature)
                if actuation.armature_exact is not None:
                    # After ``armature``, so stating a value outright wins over
                    # the field that cannot say zero; the group's own DOFs only.
                    for dof in dofs:
                        builder.joint_armature[dof] = float(actuation.armature_exact)
            if arm.home_q:
                for coord, value in zip(index.arm_coords, arm.home_q, strict=False):
                    builder.joint_q[coord] = float(value)
        if unlimited_dofs:
            self._unlimit_ctrlrange(builder, unlimited_dofs)

    @staticmethod
    def _set_armature(builder: newton.ModelBuilder, dofs: Sequence[int], armature: float) -> int:
        """Give ``dofs`` -- and anything geared to them -- this armature.

        The followers are the point. A geared hand names one driven joint, so a
        spec's gripper DOFs are that joint alone; leaving the five passive joints
        weightless is what lets the drive overpower their equality constraints.
        They are found through the builder's own mimic table rather than by name,
        so this works for any linkage the importer closed.

        Returns:
            How many DOFs were written.
        """
        followers = list(getattr(builder, "constraint_mimic_joint0", ()))
        leaders = list(getattr(builder, "constraint_mimic_joint1", ()))
        starts = builder.joint_qd_start
        driven = {int(d) for d in dofs}
        # A joint's DOFs, by its position in joint_qd_start.
        for follower, leader in zip(followers, leaders, strict=False):
            if int(starts[int(leader)]) not in driven:
                continue
            begin = int(starts[int(follower)])
            end = int(starts[int(follower) + 1]) if int(follower) + 1 < len(starts) else len(builder.joint_armature)
            driven.update(range(begin, end))
        written = 0
        for dof in sorted(driven):
            if 0 <= dof < len(builder.joint_armature):
                builder.joint_armature[dof] = float(armature)
                written += 1
        return written

    # -------------------------------------------------------------- indexing
    def _arm_joints(
        self, labels: Sequence[str], arm: ArmSpec, *, joint_types: Sequence[int] | None = None
    ) -> tuple[list[int], list[int]]:
        """This arm's joint indices, split into arm joints and **driven** finger joints.

        The hand supplies the finger names; the arm supplies its own. A joint the
        linkage carries but nothing drives -- ``GripperSpec.follower_names`` --
        is in neither list: it must not be written a target (the solver ignores
        it, and ``apply_actuation`` asserts it cannot happen) and it must not be
        mistaken for an arm joint either.

        Args:
            joint_types: ``builder.joint_type`` or ``model.joint_type``, so the
                prefix scan can skip free joints. **A free joint is a body
                floating in the world and never an arm's**, and without this an
                object whose name begins with an arm's prefix is swept into that
                arm: a cube called ``left_cube`` on a robot with a ``left`` arm
                gets its ``left_cube_free_joint`` counted as a seventh arm
                joint, and the failure surfaces as
                ``joint_dof_mask partially masks joint 16 (FREE)`` out of
                Newton's IK -- a true statement about a mask that says nothing
                about the name that caused it.

                It only bites in prefix mode, and only for a scene assembled
                outside the binding matrix, which is why no shipped task hits
                it: a benchmark's objects are named for what they are.
        """
        hand = self.hands.get(arm.name)
        driven = list(hand.finger_names) if hand else []
        followers = set(hand.follower_names) if hand else set()
        if arm.joint_names:
            arm_j = [label_index(labels, n) for n in arm.joint_names]
            finger_j = [label_index(labels, n) for n in driven]
            return arm_j, finger_j
        # Prefix mode: the arm's joints are its prefix minus whatever the hand
        # claimed. Naming the hand's joints explicitly is what stops a token
        # scan sweeping an undriven follower in with the driven pair -- the
        # arena YAM's failure, and one that only surfaced on the two suites that
        # state gripper gains.
        named = {*driven, *followers}
        token = hand.finger_token if hand else ""
        arm_j, finger_j = [], []
        free = int(newton.JointType.FREE)
        types = list(joint_types) if joint_types is not None else []
        for j, label in enumerate(labels):
            tail = label.split("/")[-1]
            if tail in named or not tail.startswith(arm.joint_prefix):
                continue
            if types and int(types[j]) == free:
                continue
            # The token branch survives only for a hand that named no joints at
            # all. It is the silent path -- a token matching nothing yields a
            # hand that never moves -- so a hand that can name its joints does.
            (finger_j if token and token in tail else arm_j).append(j)
        if driven:
            finger_j = [label_index(labels, n) for n in driven]
        return arm_j, finger_j

    def _index_arm_builder(self, builder: newton.ModelBuilder, arm: ArmSpec) -> ArmIndex:
        labels = builder.joint_label
        arm_j, finger_j = self._arm_joints(labels, arm, joint_types=builder.joint_type)
        if not arm_j:
            raise RuntimeError(f"{self.spec.name} arm {arm.name!r} matched no joints in {len(labels)} labels")
        index = ArmIndex(arm.name)
        index.arm_coords = [int(builder.joint_q_start[j]) for j in arm_j]
        index.arm_dofs = [int(builder.joint_qd_start[j]) for j in arm_j]
        index.gripper_coords = [int(builder.joint_q_start[j]) for j in finger_j]
        index.gripper_dofs = [int(builder.joint_qd_start[j]) for j in finger_j]
        return index

    # -------------------------------------------------------------- phase 2
    def bind(self, model: newton.Model) -> None:
        super().bind(model)
        self.bodies = np.asarray(
            list(range(*self.range.bodies)) if not self.resident else (self._body_cache or []),
            dtype=np.int32,
        )
        joint_labels = list(model.joint_label)
        body_labels = list(model.body_label)
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        shape_body = model.shape_body.numpy()
        upper = model.joint_limit_upper.numpy()
        # Every joint, not this robot's range: a free-jointed object's parent is
        # the world, and the walk never looks the world up, so nothing outside
        # the arm can be reached from inside it.
        all_joints = range(model.joint_count)

        for arm in self.spec.arms:
            arm_j, finger_j = self._arm_joints(joint_labels, arm, joint_types=model.joint_type.numpy().tolist())
            if not arm_j:
                raise RuntimeError(f"{self.spec.name} arm {arm.name!r} matched no joints after finalize")
            index = ArmIndex(arm.name)
            index.arm_coords = [int(q_start[j]) for j in arm_j]
            index.arm_dofs = [int(qd_start[j]) for j in arm_j]
            index.gripper_coords = [int(q_start[j]) for j in finger_j]
            index.gripper_dofs = [int(qd_start[j]) for j in finger_j]
            bodies = _limb_bodies(joint_parent, joint_child, all_joints, (*arm_j, *finger_j))
            index.bodies = tuple(sorted(bodies))
            index.shapes = np.asarray(
                [s for s in range(model.shape_count) if int(shape_body[s]) in bodies], dtype=np.int32
            )
            hand = self.hands.get(arm.name)
            # The tool-centre body: the hand's own if it has one, else the arm's
            # wrist, else the last arm joint's child. A hand that names a body
            # is naming where *its* point is measured from, which is the
            # difference between a Robotiq folded into panda_link7 and a YAM
            # gripper with a body of its own.
            tcp_body = (hand.tcp_body if hand else "") or arm.ee_body_name
            if hand is not None and hand.tcp_body:
                index.ee_body = self._hand_body(body_labels, arm, tcp_body)
            else:
                index.ee_body = label_index(body_labels, tcp_body) if tcp_body else int(joint_child[arm_j[-1]])
            index.base_body = (
                label_index(body_labels, arm.base_body_name) if arm.base_body_name else int(joint_parent[arm_j[0]])
            )
            sides = hand.finger_bodies if hand else ()
            index.finger_bodies = tuple(tuple(label_index(body_labels, n) for n in side) for side in sides)
            # Finger travel from the model's own limits, so a retuned gripper is
            # picked up rather than silently overridden by the spec's constant.
            travel = float(upper[index.gripper_dofs[0]]) if index.gripper_dofs else 0.0
            if hand and hand.open_width > 0:
                index.gripper_open = hand.open_width
            elif np.isfinite(travel) and travel > 1.0e-4:
                index.gripper_open = travel
            # The followers, for measurement only. label_index raises on a name
            # that matches nothing, so a hand claiming a follower it does not
            # have is a loud error rather than a silently unmeasurable jaw.
            followers = [label_index(joint_labels, n) for n in hand.follower_names] if hand else []
            index.gripper_follower_coords = tuple(int(q_start[j]) for j in followers)
            index.gripper_follower_dofs = tuple(int(qd_start[j]) for j in followers)
            index.gripper_open_q = tuple(float(v) for v in hand.finger_open_q) if hand else ()
            index.gripper_closed_q = tuple(float(v) for v in hand.finger_closed_q) if hand else ()
            # effective_span rather than the arithmetic restated: one rule, on
            # the hand, and this used to be one of the two places it was spelled.
            index.gripper_span = hand.effective_span if hand else 2.0 * index.gripper_open
            index.home_q = np.asarray(arm.home_q, dtype=np.float32)
            index.approach_axis = np.asarray(hand.approach_axis if hand else (0.0, 0.0, 1.0), dtype=np.float64)
            index.close_axis = np.asarray(hand.close_axis if hand else (0.0, 1.0, 0.0), dtype=np.float64)
            self.arms[arm.name] = index

        # IK needs a state at the home posture to read the resting EE pose from.
        home_state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, home_state)
        for arm in self.spec.arms:
            index = self.arms[arm.name]
            hand = self.hands.get(arm.name)
            self.iks[arm.name] = ArmIK(
                model,
                arm_coords=index.arm_coords,
                arm_dofs=index.arm_dofs,
                ee_body=index.ee_body,
                home_state=home_state,
                # The TCP is the hand's, not the arm's: put a different hand on
                # this wrist and the point moves while the arm does not.
                ee_offset=hand.tcp_offset if hand else (0.0, 0.0, 0.0),
                ee_offset_rot=hand.tcp_offset_rot if hand else (0.0, 0.0, 0.0, 1.0),
                iterations=self.spec.ik_iterations,
                # The PyRoKi solver is built from these: the embodiment's URDF,
                # the base body the URDF root stands in, and the home posture
                # it calibrates the URDF against the model at.
                urdf=self.spec.urdf,
                base_body=index.base_body,
                home_q=index.home_q,
            )
            if index.gripper_coords:
                # One driven joint plus one follower is a parallel jaw whose gap
                # the two coordinates measure exactly. Saying so keeps width()
                # exact on a hand that cannot use the mirrored branch because
                # only one of its joints may be driven.
                pair = (
                    (index.gripper_coords[0], index.gripper_follower_coords[0])
                    if len(index.gripper_coords) == 1 and len(index.gripper_follower_coords) == 1
                    else ()
                )
                self.grippers[arm.name] = Gripper(
                    model,
                    index.gripper_coords,
                    open_width=index.gripper_open,
                    open_q=index.gripper_open_q,
                    closed_q=index.gripper_closed_q,
                    span=index.gripper_span,
                    width_coords=pair,
                    action_closed_at_high=hand.action_closed_at_high if hand else True,
                )

            if hand is not None and hand.suction_pad:
                self.suctions[arm.name] = Suction(
                    model,
                    arm=arm.name,
                    pad_body=self._hand_body(body_labels, arm, hand.suction_pad),
                    pad_radius=float(hand.pad_radius),
                    rows=self._suction_rows.get(arm.name, {}),
                    owners=self._suction_owners,
                )

        self._home_target = self.home_target_q()

    # -------------------------------------------------------------- phase 3
    @property
    def adds_substep_kernels(self) -> bool:
        """Whether this robot puts a kernel in the substep: only a vacuum cup's wrench does.

        The compiler's replication refusal reads it, so a robot without a cup
        replicates and one with a cup is refused by name: the cup's wrench and
        weld rows address world 0 alone.
        """
        return any(hand is not None and hand.suction_pad for hand in self.hands.values())

    def contribute_hooks(self, hooks) -> None:
        """Each vacuum cup's wrench kernel, so it is in the graph from the first capture."""
        for suction in self.suctions.values():
            suction.contribute_hooks(hooks)

    def on_scene_ready(self, pipeline) -> None:
        """Each vacuum cup finds the solver that carries its weld rows."""
        for suction in self.suctions.values():
            suction.attach(pipeline)

    # ---------------------------------------------------------------- runtime
    def home_joint_q(self) -> np.ndarray:
        """A full ``joint_q`` vector with each arm at its home posture.

        Free-joint coordinates keep the model's own values, so restoring the
        home posture never drags a loose object with it.
        """
        q = np.array(self.model.joint_q.numpy(), dtype=np.float32)
        for arm_spec in self.spec.arms:
            index = self.arms[arm_spec.name]
            for coord, value in zip(index.arm_coords, index.home_q, strict=False):
                q[coord] = float(value)
            # Where the hand starts, expressed the same way the Gripper drives
            # it. Normally full open -- the declared open coordinates, or the
            # mirrored travel when a spec states none, since a geared hand's open
            # position is not +/-travel and can be zero, so assuming the mirrored
            # form closes it at reset. An arm that states its own starting
            # aperture wins over all of that.
            if arm_spec.home_gripper_q:
                for coord, value in zip(index.gripper_coords, arm_spec.home_gripper_q, strict=True):
                    q[coord] = float(value)
            elif index.gripper_open_q:
                for coord, value in zip(index.gripper_coords, index.gripper_open_q, strict=True):
                    q[coord] = float(value)
            elif index.gripper_coords:
                q[index.gripper_coords[0]] = index.gripper_open
                q[index.gripper_coords[1]] = -index.gripper_open
        return q

    def home_target_q(self) -> np.ndarray:
        """The matching ``joint_target_q`` vector.

        Indexed by DOF unless ``newton.use_coord_layout_targets`` is set, which
        the compiler does -- so this indexes the same way ``joint_q`` does.
        """
        target = np.zeros(len(self.model.control().joint_target_q), dtype=np.float32)
        home = self.home_joint_q()
        target[: min(len(target), len(home))] = home[: len(target)]
        return target

    def reset(self, pipeline, episode) -> None:
        del episode
        for suction in self.suctions.values():
            suction.reset()
        for name, gripper in self.grippers.items():
            # The command that *produces* the home aperture, which is 0 (open)
            # for every hand that starts open and is not for one that does not.
            # Leaving it at 0 while the coordinates say otherwise would make the
            # measured and commanded closures disagree from step zero, which is
            # exactly the pair a grasp check reads.
            gripper.command = self._home_gripper_command(name, gripper)
        if self._home_target is not None:
            pipeline.control.joint_target_q.assign(self._home_target)
        if self.spec.reset_joint_q:
            self._write_home_joint_q(pipeline)

    def _home_gripper_command(self, arm: str, gripper: Gripper) -> float:
        """The 0..1 command that lands the fingers on their home coordinates.

        Inverted from the same interpolation :meth:`Gripper.apply` runs, using
        the first driven joint -- the rest track it by construction. 0 for a hand
        whose arm states no starting aperture, which is every robot but one.
        """
        spec = next((a for a in self.spec.arms if a.name == arm), None)
        if spec is None or not spec.home_gripper_q:
            return 0.0
        low, high = float(gripper.open_q[0]), float(gripper.closed_q[0])
        if abs(high - low) < 1.0e-12:
            return 0.0
        return float(np.clip((float(spec.home_gripper_q[0]) - low) / (high - low), 0.0, 1.0))

    def _write_home_joint_q(self, pipeline) -> None:
        """Put the arm back at its home posture, coordinates and all.

        Only this robot's own coordinates are touched -- read back, overwritten
        at the arm and finger indices, written whole -- so a loose object's free
        joint keeps whatever an object entity's own reset put there, whichever
        order the two run in.

        Both states, because the compiler's double buffer is swapped by the
        substep loop and a caller cannot know which half a reset lands on.
        """
        home = self.home_joint_q()
        coords = [c for index in self.arms.values() for c in (*index.arm_coords, *index.gripper_coords)]
        coords += [c for index in self.arms.values() for c in index.gripper_follower_coords]
        if not coords:
            return
        for state in (pipeline.state_0, pipeline.state_1):
            q = np.array(state.joint_q.numpy(), dtype=np.float32)
            qd = np.array(state.joint_qd.numpy(), dtype=np.float32)
            for c in coords:
                q[c] = home[c]
            state.joint_q.assign(q)
            # The velocities too: an arm restored to a posture while still
            # carrying the previous episode's joint rates lurches on the first
            # step, which reads as a controller overshooting rather than as a
            # reset that stopped halfway. Only this robot's DOFs -- the coord
            # and DOF layouts coincide for revolute and prismatic joints, which
            # is every joint an arm has. The followers as well: a mimic finger
            # still swinging from the last episode drags its driver with it.
            for index in self.arms.values():
                for d in (*index.arm_dofs, *index.gripper_dofs, *index.gripper_follower_dofs):
                    qd[d] = 0.0
            state.joint_qd.assign(qd)

    def observe(self, state: newton.State) -> dict[str, np.ndarray]:
        """Per-arm proprioception: joint positions, gripper closure, TCP pose.

        ``<arm>/gripper`` is the **measured** closure, not the commanded one, and
        the distinction is the whole content of the signal. A hand told to shut on
        a 0.07 m object stalls at about 0.18 of its stroke; the same command on air
        reaches 1.0. Reporting the command made those identical -- so a policy or a
        graph checking whether its grasp took anything saw a hand that always
        closed perfectly, and concluded it was holding air. Every consumer here
        (the graph-policy facade's ``gripper_fraction``, the remote observation adapter's
        proprioception vector) wants the measurement, because on a real robot that
        is the only one it can have.
        """
        joint_q = state.joint_q.numpy()
        out: dict[str, np.ndarray] = {}
        for name, index in self.arms.items():
            out[f"{name}/joint_q"] = np.array([joint_q[c] for c in index.arm_coords], dtype=np.float32)
            gripper = self.grippers.get(name)
            out[f"{name}/gripper"] = np.float32(gripper.closed_fraction(state) if gripper is not None else 0.0)
            # What was asked for, beside what happened. A graph never needs it, but
            # a run whose gripper never reaches its command is diagnosable from the
            # pair and not from either alone.
            out[f"{name}/gripper_command"] = np.float32(gripper.command if gripper is not None else 0.0)
            pos, quat = self.iks[name].ee_pose(state)
            out[f"{name}/ee_pose"] = np.concatenate([pos, quat]).astype(np.float32)
        return out
