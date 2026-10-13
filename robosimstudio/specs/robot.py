# SPDX-License-Identifier: Apache-2.0
"""What a robot is: its arms, its hands, and how they are driven.

Depends on :mod:`robosimstudio.specs.asset` for one thing only -- an
:class:`~robosimstudio.specs.asset.AssetSpec` names the file an arm or a hand is
imported from. Nothing here knows about tasks or scenes: an embodiment is an
independent axis, composed with the other two at compile time by a
:class:`~robosimstudio.specs.task.Binding`.

**Produced by** the modules under :mod:`robosimstudio.robots.embodiments` and
:mod:`robosimstudio.robots.grippers`, which register one of these per robot.
**Consumed by** the scene compiler, and by the robot library's IK and motion
backends, which read reach, joint sets and tool frames off them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from robosimstudio.core.render.finish import Finish

from .asset import AssetSpec
from .base import IDENTITY, Quat, Vec3, Xform

__all__ = ["ActuationSpec", "ArmSpec", "EmbodimentSpec", "GripperSpec", "jaw_gap_from"]


@dataclass(frozen=True, kw_only=True)
class ActuationSpec:
    """Position-servo gains for one group of joints.

    Newton only reads ``joint_target_q`` for a DOF whose actuator is a
    ``<position>``; a torque-actuated import maps to a direct-control source and
    silently ignores every target, so an arm droops under gravity while IK
    solves happily against something nothing is tracking.
    :attr:`require_position_mode` asserts rather than trusts.

    Attributes:
        target_ke: Position gain [N·m/rad or N/m].
        target_kd: Damping gain.
        armature: Rotor inertia reflected at the joint [kg·m²], added to the
            diagonal of the mass matrix. 0 leaves the import's own.

            **On a geared hand this is what makes the linkage hold together, and
            no other knob substitutes for it.** A Robotiq 2F-85 is one driven
            joint plus five geared to it by solver equality constraints, and with
            zero armature -- which is what the Isaac export carries -- the drive
            simply overpowers them: measured on the grocery stage, the passive
            joints ran up to 41.8 degrees away from what their gearing demands and
            the two fingertip pads ended up 95.8 degrees off parallel, so the hand
            closed on a can edge-first instead of flat.

            Omni Physics' own 2F-85 tuning example prescribes exactly this, and
            its 5.0e-3 is a derived rotor inertia rather than a fitted constant.
            Measured here, closing on the soup can:

                armature   pad tilt   worst mimic residual
                0            95.8°          41.8°
                1e-3         28.6°          19.5°
                5e-3          6.4°           4.5°
                1e-2          3.2°           2.3°

            Note what does *not* help, both measured and both the obvious guesses:
            raising the solver to 50 iterations changes the tilt by 0.00 degrees,
            and re-tuning the drive gains changes it by 0.05. The linkage is not
            under-solved or over-driven, it is under-inertial.

            Applied to the whole linkage -- the driven joint and every joint
            geared to it -- since armature on the driver alone leaves the
            followers exactly as light as before.
        require_position_mode: Raise if a DOF in the group is not
            position-driven.
        target_ke_per_dof: Per-DOF stiffness, overriding :attr:`target_ke` where
            stated. Empty uses the scalar for every DOF.

            **A system-identified arm has no single gain.** SimplerEnv's WidowX
            was fitted per joint against real trajectories and its stiffnesses
            run 730 to 1272; the Google robot's run 433 to 2000. Those are as
            much a part of the benchmark as its geometry -- how the arm tracks is
            what an open-loop policy's behaviour depends on -- and collapsing
            them to a mean is a different robot that happens to build.

            Stated per DOF rather than per joint, so a multi-DOF joint is
            expressible; every arm here is one DOF per joint, so the two
            coincide and the tuple reads as the reference's own list.
        target_kd_per_dof: The same for damping.
        effort_limit: Torque or force ceiling per DOF [N·m or N]. 0 leaves the
            import's own.
        effort_limit_per_dof: Per-DOF ceilings, overriding :attr:`effort_limit`.
            Also a benchmark quantity: the WidowX's first two joints are limited
            to 200 and the rest to 100, which is what stops a fitted gain from
            driving the shoulder through the table.
        unlimited: Remove this group's joint limits entirely.

            For a **leader-driven** arm, where the authority on where a joint may
            go is a physical arm in someone's hand rather than the asset. The
            two disagree more easily than they look: a leader whose own travel
            exceeds the model's range drives the target past a stop the solver
            then clamps, and the gap that leaves never closes. Measured on a
            bimanual YAM driven from its leader, one wrist joint sat 0.287 rad
            -- 16.5 degrees -- outside its +-90 degree range with the joint
            *stationary*, which
            :mod:`~robosimstudio.robots.teleop.force_feedback` renders as a
            permanent contact force on a joint touching nothing.

            Off by default and deliberately not a general fix. It costs two real
            things: a joint at its limit stops being something the operator can
            feel, which is one of the forces that module exists to render; and
            :class:`~robosimstudio.solvers.ArmIK` solves against these
            bounds, so an unlimited arm lets IK-driven front ends reach poses
            the hardware cannot. Set it for a teleop pairing, not for a policy
            or a benchmark.
        armature_exact: Rotor inertia to set outright, ``0.0`` included, or
            ``None`` to leave it alone. :attr:`armature` cannot express zero --
            it treats it as "leave the import's own", which is the right default
            and the wrong thing when the intent is to match a source model that
            declares none. Written to the group's own DOFs only, after
            :attr:`armature`; a linkage's geared followers are not touched.
        joint_damping: Per-DOF joint damping, or ``None`` to keep the import's.
        joint_friction: Per-DOF joint friction loss, or ``None`` to keep the
            import's.
    """

    target_ke: float
    target_kd: float
    armature: float = 0.0
    require_position_mode: bool = True
    target_ke_per_dof: tuple[float, ...] = ()
    target_kd_per_dof: tuple[float, ...] = ()
    effort_limit: float = 0.0
    effort_limit_per_dof: tuple[float, ...] = ()
    unlimited: bool = False
    armature_exact: float | None = None
    joint_damping: float | None = None
    joint_friction: float | None = None

    def ke_for(self, index: int) -> float:
        """Stiffness for the ``index``-th DOF of the group."""
        return float(self.target_ke_per_dof[index]) if index < len(self.target_ke_per_dof) else float(self.target_ke)

    def kd_for(self, index: int) -> float:
        """Damping for the ``index``-th DOF of the group."""
        return float(self.target_kd_per_dof[index]) if index < len(self.target_kd_per_dof) else float(self.target_kd)

    def effort_for(self, index: int) -> float:
        """Effort ceiling for the ``index``-th DOF, or 0 to leave the import's."""
        if index < len(self.effort_limit_per_dof):
            return float(self.effort_limit_per_dof[index])
        return float(self.effort_limit)


def jaw_gap_from(*, span: float, open_width: float, fingers: int, joint_q: float) -> float:
    """Pad separation at one finger coordinate, from a hand's own geometry [m].

    ``span - fingers * (open_width - joint_q)``: each finger has travelled
    ``open_width - joint_q`` in from full open, and that many of them are closing
    on one object.

    A free function because two things need it and neither owns the other -- a
    :class:`GripperSpec` before a scene exists, and an ``ArmHandle`` after one
    does, which carries the same three numbers copied out of the spec. Written
    twice it would disagree once.
    """
    return float(span) - int(fingers) * (float(open_width) - float(joint_q))


@dataclass(frozen=True, kw_only=True)
class GripperSpec:
    """A hand: how to find it, how it closes, and how big it is.

    Registered on its own axis, beside scenes and embodiments, because a hand is
    not a property of the arm it is bolted to.

    Two modes, told apart by :attr:`asset`:

    * **Descriptive** (``asset is None``): the hand already arrives inside the
      arm's or the scene's import, and this spec only *indexes* it -- which
      joints drive it, which bodies are the pads, where the tool-centre point
      sits. All four shipped hands are descriptive, and migrating them is a pure
      refactor with nothing measured newly.
    * **Attached** (:attr:`asset` set): the hand carries its own file and is
      welded onto :attr:`mount_body`. Read :attr:`asset` before reaching for it
      -- the runtime weld is real and narrow, and the route that has actually
      shipped is the offline one: the hand composed into the arm's own file
      before import.

    Any name field may carry ``{arm}``, substituted with the owning
    :class:`ArmSpec`'s name by :meth:`for_arm`. That is what lets one entry
    describe both hands of a bimanual robot whose fingers are ``right_joint7``
    and ``left_joint7`` -- the same hand, a different prefix.

    **A measurement nobody has taken is stated as 0, never as a plausible
    number.** The mechanism that reads it then keeps its own default, which is
    the convention :attr:`span` and :attr:`open_width` already follow. A
    fabricated constant passes review and fails on contact; a zero says out loud
    that the hand has not been measured, and a test can list what is missing.

    Attributes:
        name: Registry key.

        finger_names: The **driven** finger joints' label suffixes -- the ones an
            actuator is attached to, and only those. A joint the linkage carries
            but nothing drives belongs in :attr:`follower_names`.
        follower_names: Finger joints an equality or mimic constraint carries
            rather than an actuator.

            Split from :attr:`finger_names` because the two need opposite
            treatment and lumping them together breaks a robot each way. Naming a
            follower as driven puts a ``JointTargetMode.NONE`` DOF into the group
            :meth:`EntityEmbodiment.apply_actuation` asserts over, which is what
            refuses to build the arena YAM on any task that states gripper gains.
            Dropping it instead would work, but it would also switch
            :class:`~robosimstudio.solvers.Gripper` off its mirrored branch
            -- and on a hand whose two joints *are* a mirrored pair that changes
            what ``width()`` and ``closed_fraction`` report on a robot the tape
            suite is calibrated against. Listed here, the joint is excluded from
            the drive and still available to anything measuring the jaw.
        finger_token: Substring separating finger joints from arm joints inside
            an :attr:`ArmSpec.joint_prefix` scan. Used only when
            :attr:`finger_names` is empty, and worth avoiding: the prefix path is
            the one that matches *silently*, so a token that catches an undriven
            joint, or catches nothing, produces a robot that builds and a hand
            that never moves.
        finger_open_q: Per joint in :attr:`finger_names`, the coordinate at full
            open. Empty uses the mirrored prismatic convention
            ``(+open_width, -open_width)``, which needs exactly two joints. A
            geared hand states its own: the Robotiq 2F-85 has one driven revolute
            whose *zero* is open, so the mirrored form describes neither its sign
            nor its count.
        finger_closed_q: The same at full closed. Empty uses zeros.
        action_closed_at_high: Whether the **high** end of a normalized gripper
            command closes this hand.

            A reference benchmark's gripper controller maps a normalized action
            linearly onto the joint's own ``[low, high]``, so which end closes is
            a fact about the *joint*, not about a convention anyone chose -- and
            the two SimplerEnv robots come out opposite. The WidowX's prismatic
            fingers run 0.014 (nearly shut) to 0.038 (open), so its low end
            closes; the Google robot's revolute pair runs -1.31 (open) to +1.31
            (shut), so its high end does. SimplerEnv's own README says the
            meaning of open and close depends on the URDF, and this is the field
            that carries it.

            Load-bearing only once a *trained checkpoint* drives the hand.
            Everything written against this package's own 0-open..1-closed
            convention is unaffected, which is exactly why getting it wrong is
            invisible: a scripted routine that asks for "closed" still closes,
            and a policy that asks for -1 opens the hand it meant to shut.
        linear_travel: Whether the driven joint's travel is proportional to the
            jaw gap.

            True for any parallel jaw, which is three of the four hands here --
            including one whose two fingers run on the *same* sign rather than
            mirrored, and one whose twin is carried by an equality. Neither of
            those is a linkage, and inferring "geared" from a sign convention or
            an actuator count calls them one: it would tell every caller their
            travel is a curve when it is a straight line.

            False is the real thing: a 2F-85's driven revolute sweeps an arc and
            the pads follow it, so a command is monotone in the gap and nothing
            more. State it and a caller knows to calibrate rather than divide.
        open_width: Finger travel at full open [m], the number
            :class:`~robosimstudio.solvers.Gripper` interpolates against. 0
            reads it off the imported joint limits. For a geared hand this is an
            *angle* -- read :attr:`span` for a distance.
        span: Fingertip separation at full open [m]: the gap an object has to
            pass through, and what every grasp-fit test measures against. 0
            derives ``2 * open_width``, which is right for a mirrored prismatic
            pair and meaningless for a linkage.

        tcp_body: Body label suffix the tool-centre point is measured in. Empty
            means the arm's own :attr:`ArmSpec.ee_body_name` -- right for a hand
            whose bodies the arm's asset does not separately name (the DROID's
            Robotiq collapses into ``panda_link7``) and wrong for one that does
            (the i2rt YAM has a ``{arm}_gripper`` body of its own).
        tcp_offset: Tool-centre point in :attr:`tcp_body`'s frame [m].

            **On the hand, not on the arm, and that is the argument for this
            class in one field.** Put a different hand on the same wrist and the
            TCP moves; the arm did not change.
        tcp_offset_rot: Rotation from that body's frame to the grasp frame,
            xyzw. The grasp frame's +z is the approach direction, which is the
            convention every core solver speaks.
        approach_axis: Gripper-local axis pointing from the wrist towards the
            grasp point -- the direction the hand extends. Measured, not assumed:
            it is what a top-down grasp maps to world -z.
        close_axis: Gripper-local axis the jaws close along, measured from the
            **pad centres**. Worth measuring rather than reading off the
            fingertip *body origins*, which on the i2rt YAM is 84.5 degrees wrong
            and whose origins move apart as the blades close.

        finger_bodies: Body label suffixes of the finger links, per side, for
            contact-based grasp tests. **Empty disables those tests silently**:
            :meth:`~robosimstudio.core.probes.contact.ContactProbe.check_grasp`
            returns ``bool(flags) and all(flags)``, so no sides means ``False``
            forever and every graph that verifies a grasp concludes it is holding
            air. That is what the arena YAM did.
        open_axis_signs: Which way along each finger body's local +y that side
            opens.

        pad_half_span: Half the separation of the pad *centres* at full open [m].
            Deliberately not ``span / 2``: :attr:`span` is the gap between the
            gripping faces and the pads stand outboard of it. On the 2F-85 that
            is 0.052 against 0.0425, and the difference decides whether placing
            an item sweeps its neighbour.
        pad_clearance: Margin left each side of an object inside :attr:`span`
            [m], so a pair that only just fits is not proposed.
        pad_thickness: A pad's own height along the approach axis [m].
        pad_depth: How far the pads reach past the housing along the approach
            axis [m] -- a grasp deeper into an object than this cannot be taken.
        pad_radius: How wide a corridor each pad needs [m].
        tip_reach: Distance from the tool-centre point to the fingertips along
            the approach axis [m]. What a routine planning a hover clears by, and
            today a sentence in two suite docstrings rather than a number
            anything can read.
        housing_half: Half-width of the hand's body across its widest axis [m].
        housing_rise: How far above the fingertips the housing's underside sits
            [m].

        asset: The hand's own file, or ``None`` when it already arrives inside
            the arm's or the scene's import.
        mount_body: Which body of the *arm* the hand bolts to. Empty means the
            arm's :attr:`ArmSpec.ee_body_name`, or its derived wrist.

            **It also says which arms this hand fits**, and that is not a second
            job so much as the same one stated once. :attr:`mount_xform` is
            measured against a particular flange; naming that flange's body is
            how a hand records which. ``registry.admits`` refuses a robot whose
            wrist is called something else, because the alternative is a compile
            that dies on ``'wrist_3_link' matched 0 labels`` -- a true statement
            about labels that tells a reader nothing about the pairing.
        mount_xform: The hand's root pose in :attr:`mount_body`'s frame.
        actuation: The hand's own servo gains, or ``None`` to leave the import's.
            Here because they are a fact about the linkage rather than about the
            robot: the 2F-85's are derived from *its* five geared joints, and the
            day a second arm carries the same hand, restating them on both
            embodiments is the duplication this class exists to end. An
            :attr:`EmbodimentSpec.gripper_actuation` overrides it.
    """

    name: str

    # ---------------------------------------------------------------- joints
    finger_names: tuple[str, ...] = ()
    follower_names: tuple[str, ...] = ()
    finger_token: str = "finger"
    finger_open_q: tuple[float, ...] = ()
    finger_closed_q: tuple[float, ...] = ()
    action_closed_at_high: bool = True
    linear_travel: bool = True
    open_width: float = 0.0
    span: float = 0.0

    # ------------------------------------------------------------- the point
    tcp_body: str = ""
    tcp_offset: Vec3 = (0.0, 0.0, 0.0)
    tcp_offset_rot: Quat = (0.0, 0.0, 0.0, 1.0)
    approach_axis: Vec3 = (0.0, 0.0, 1.0)
    close_axis: Vec3 = (0.0, 1.0, 0.0)

    # ----------------------------------------------------------- the fingers
    finger_bodies: tuple[tuple[str, ...], ...] = ()
    open_axis_signs: tuple[float, ...] = (1.0, -1.0)

    # ---------------------------------------------------------- the envelope
    pad_half_span: float = 0.0
    pad_clearance: float = 0.0
    pad_thickness: float = 0.0
    pad_depth: float = 0.0
    pad_radius: float = 0.0
    tip_reach: float = 0.0
    housing_half: float = 0.0
    housing_rise: float = 0.0

    # -------------------------------------------------------------- mounting
    asset: AssetSpec | None = None
    mount_body: str = ""
    mount_xform: Xform = IDENTITY
    actuation: ActuationSpec | None = None

    @property
    def effective_span(self) -> float:
        """:attr:`span`, or the mirrored-prismatic derivation of it [m].

        One place, because the rule ``span or 2 * (open_width or 0.04)`` was
        written out twice -- in the compiler and in ``registry.admits`` -- and a
        rule spelled twice is a rule that will disagree once.
        """
        return float(self.span) or 2.0 * (float(self.open_width) or 0.04)

    @property
    def geared(self) -> bool:
        """Whether travel and distance are *not* proportional on this hand.

        The inverse of :attr:`linear_travel`, named for the thing it usually
        means. A caller converting metres to a command needs this: on a geared
        hand that conversion is a monotone approximation until someone
        calibrates it.
        """
        return not self.linear_travel

    def jaw_gap(self, joint_q: float) -> float:
        """The surface gap between the pads at one finger coordinate [m].

        The fingers each travel ``open_width - joint_q`` in from full open, and
        there are :attr:`finger_names` of them closing on one object, so the gap
        shrinks by that many times the travel::

            gap = span - len(finger_names) * (open_width - joint_q)

        **Derived rather than fitted, and checked against a fit.** The usbc rig
        measures this relation at build time by sweeping the finger command
        through forward kinematics and reading the actual collider separation off
        the fingertip meshes -- a polyfit over the sweep. On the i2rt linear_4310
        that fit is ``gap = 2.000014 q - 0.0010397``; this derivation gives
        ``gap = 2 q - 0.00104``, agreeing to 1.4e-5 in the slope and 3.1e-7 in
        the offset. The residual is the polyfit's own numerical noise.

        So the fit was never a new calibration constant. It is :attr:`span` and
        :attr:`open_width` -- both already measured, both already carrying their
        own provenance -- and a hand that recorded it separately would be
        recording the same measurement twice, with two chances to update one and
        not the other.

        Raises:
            ValueError: On a geared hand. Travel and distance are not
                proportional there, so this arithmetic is not an approximation of
                the right answer -- it is a different curve. The Robotiq's five
                geared joints are the case; whoever needs a gap for one owes a
                calibration, not a formula.
        """
        if self.geared:
            raise ValueError(
                f"gripper {self.name!r} is geared, so pad separation is not linear in finger travel. "
                "A gap for this hand needs a measured curve, not this derivation."
            )
        return jaw_gap_from(
            span=self.effective_span,
            open_width=self.open_width,
            fingers=len(self.finger_names) or 2,
            joint_q=joint_q,
        )

    def for_arm(self, arm_name: str) -> GripperSpec:
        """This hand with ``{arm}`` substituted, for the arm it sits on.

        Pure spec arithmetic, so it stays here rather than in the registry: the
        registry's job is to find the entry, this is what the entry means on one
        particular wrist. A name with no ``{arm}`` in it comes back unchanged, so
        a single-arm hand never has to write a template.
        """

        def fmt(text: str) -> str:
            return text.format(arm=arm_name) if "{" in text else text

        return replace(
            self,
            finger_names=tuple(fmt(n) for n in self.finger_names),
            follower_names=tuple(fmt(n) for n in self.follower_names),
            finger_bodies=tuple(tuple(fmt(n) for n in side) for side in self.finger_bodies),
            tcp_body=fmt(self.tcp_body),
            mount_body=fmt(self.mount_body),
        )


@dataclass(frozen=True, kw_only=True)
class ArmSpec:
    """One arm of an embodiment, and how to find it in an imported model.

    The arm proper: base to wrist. What is bolted *to* the wrist is a
    :class:`GripperSpec` named by :attr:`gripper`, because the same arm carries
    different hands and the same hand goes on different arms -- and while the two
    were one dataclass, that pairing was only expressible as a whole new robot.

    Joints are located by *label suffix* rather than by index, because an
    importer's numbering is not a stable interface and a wrong index surfaces
    much later as an arm that moves the wrong axis.

    Attributes:
        name: Key this arm appears under in handles, observations and actions.
            Also what ``{arm}`` substitutes to in the gripper's name templates.
        joint_prefix: Label prefix selecting this arm's joints, e.g. ``"left_"``.
            Used when :attr:`joint_names` is empty.
        joint_names: Explicit joint label suffixes, in order. Wins over
            :attr:`joint_prefix`, and worth preferring: a missing explicit name
            is a hard error where a prefix that matches nothing is silent.
        ee_body_name: Body label suffix of the **wrist** -- where a hand bolts
            on, and the frame its tool-centre point is measured in when the hand
            names no body of its own. Empty derives it as the last arm joint's
            child, which is the wrist by construction.
        base_body_name: Body label suffix of the link this arm is mounted on.
            Empty takes the body the first arm joint hangs on -- the URDF's
            root link on every import that keeps it -- which is right for
            everything that never asks and for most that do.

            What asks is a **Cartesian action expressed in the robot's own
            axes**: ``ActionEEDelta(convention="align")`` over a base frame is
            what a SAPIEN/ManiSkill-trained checkpoint emits, and a base standing
            yawed 180 degrees from world -- which is how both SimplerEnv robots
            stand -- turns a world-frame reading of those same numbers into an
            arm driven backwards along x and y. The *body*, not a constant
            quaternion, because reading it from ``body_q`` costs nothing and
            keeps a scene that re-mounts the robot from needing a second
            statement of where it went.
        gripper: Registry key of the hand on this wrist, or ``""`` for a bare
            arm. Empty falls through to :attr:`EmbodimentSpec.gripper`, so a
            bimanual robot with one kind of hand states it once.
        home_q: Home joint positions [rad or m]. Empty leaves the import's own.
        home_gripper_q: Where the *fingers* start, per driven finger joint.
            Empty starts them at the hand's own full-open coordinates, which is
            what every robot here did before this field existed.

            Separate from the hand's ``finger_open_q`` because "where the hand
            starts" and "how far it opens" are not the same fact and one robot
            here proves it: SimplerEnv resets the Google robot's fingers to
            **0 rad**, the middle of their -1.31 to +1.31 sweep, not to either
            end. Starting them at full open instead would begin every episode
            with an aperture the reference never holds -- and, worse, would put
            the servo target a whole stroke away from the measured position, so
            the hand springs on the first step of every reset.
    """

    name: str
    joint_prefix: str = ""
    joint_names: tuple[str, ...] = ()
    ee_body_name: str = ""
    base_body_name: str = ""
    gripper: str = ""
    home_q: tuple[float, ...] = ()
    home_gripper_q: tuple[float, ...] = ()


@dataclass(frozen=True, kw_only=True)
class EmbodimentSpec:
    """A robot: its asset, its arms, and how it is actuated.

    Attributes:
        name: Registry key.
        asset: The MJCF, URDF or USD to import, or ``None`` when the robot
            already arrives inside the task's scene asset. A benchmark that
            ships a merged export its own compiler produced contains its robot,
            and importing it a second time would put two of them in the model --
            so such an embodiment only *indexes* what is already there.
        collapse_fixed_joints: Collapse the import's fixed joints. Goes through
            ``add_mjcf_collapsed``, which
            keeps the remap tables so custom attributes can be re-keyed.
        gravcomp_restate: Blanket-restate ``mujoco:gravcomp`` on every body
            after the collapse. The proven fallback for an import whose remap
            tables are unavailable, and only correct when every surviving body
            wants gravity compensation -- see
            :func:`~robosimstudio.core.builder.restate_body_attribute`.
        gravcomp: Set ``mujoco:gravcomp`` on *this robot's* bodies only. What an
            arm imported from USD needs: an MJCF Panda declares ``gravcomp="1"``
            for itself and the importer harvests it, but a USD stage carries no
            such attribute, so the servo alone has to hold the arm against
            gravity and it creeps under load. Unlike
            :attr:`gravcomp_restate` this is safe in a scene that also holds
            loose objects, since it names the bodies.
        enable_self_collisions: Passed to the importer. Off means the importer
            filters **every** pair of the robot's own colliding shapes against
            each other -- not just the adjacent ones -- which is what a linkage
            wants (see
            ``filter_self_collisions``) and
            what a *bimanual* cell does not: both YAM cells are two arms in one
            file, so with this off the two arms sweep straight through each
            other. :attr:`cross_arm_collisions` is the middle setting.
        cross_arm_collisions: Let the arms collide **with each other** while
            each arm stays internally filtered.

            Two arms in one asset are one import, and the importer's
            self-collision switch cannot tell the two cases apart. All the way
            on is not the answer either: a gripper's fingers are held apart by
            its joints and its equality constraints, not by its collision
            geometry, and the pads have to be able to meet when the hand closes
            on nothing -- the measured failure in
            ``filter_self_collisions``, where a
            self-colliding linkage stalled at 9-16% of its travel against
            51-675 N of its own contact forces.

            So this imports with self-collisions **on** and then filters each
            arm's shapes against themselves, leaving exactly the arm-against-arm
            pairs live. Measured on the arena YAM: 2850 pairs filtered outright
            with this off, 1369 arm-against-arm pairs live with it on, and the
            976 within-arm pairs filtered either way. Shapes the arms do not
            claim -- a shared base or torso, and the two mount pucks this cell
            leaves welded to the world -- stay filtered against everything, so
            turning this on changes one thing and names it.

            Ignored when :attr:`enable_self_collisions` is on, which already
            asks for more than this. Refused on a scene-resident robot, whose
            shapes arrive through the scene's own import and never pass through
            here.
        reset_joint_q: Write the home posture into ``joint_q`` on every episode
            reset, not only the servo targets.

            Off by default, and the default is not laziness: for a task whose
            episodes are long enough for the arm to converge, re-seeding the
            *targets* is enough, and writing ``joint_q`` teleports an arm that
            may be mid-contact. It is on for a robot whose benchmark resets the
            articulation outright -- SimplerEnv calls ``agent.reset(qpos)``, and
            without it episode ``n + 1`` starts from wherever episode ``n``
            left the hand, which on a 60-step episode is most of the difference
            between a reproduction and an approximation.
        hull_colliders: Convex-hull the robot's mesh colliders, keeping the
            visual meshes. Worth about an eighth of the physics budget and
            invisible on camera.
        decompose_colliders: Body-label tails whose mesh collider is replaced by
            a convex DECOMPOSITION rather than left to be hulled, keeping the
            visual mesh. For a link that is genuinely convex a hull is free
            fidelity; for one with a hole in it a hull is a lie, and this names
            the links where that matters.

            **What it is for, measured on the YAM's fingertips.** ``tip_left``
            is an open A-frame -- a pad plate on a triangular yoke -- and MuJoCo
            collides every mesh geom as its convex hull (see
            ``newton``'s own note at ``solver_mujoco.py``: "MuJoCo compiles every
            mesh geom through its convex-hull path"). The hull fills the yoke in
            solid: **3.13x the blade's volume**, with an inner face standing
            ~13 mm proud of the pad. So the hand grips on a surface that is not
            drawn and is not where the rubber is. Probed on a pinch of a thin strip
            at the instant the pick reports 35 N: all four pads are **13-14 mm clear
            of the strip** while their hulls are 1.2-2.3 mm inside it.

            It also explains a discrepancy the gripper module had recorded and
            not explained -- that "the jaw-gap derivation and the pad faces are
            not the same surface". The gap formula is fitted to the blade tips;
            the contact is on the hull, 13 mm inboard of them.
        decompose_threshold: CoACD concavity threshold for
            :attr:`decompose_colliders`. Lower is finer and slower. On the YAM
            fingertip, at Newton's own CoACD settings (~0.9 s per distinct mesh,
            cached per builder by mesh hash, so a bimanual robot pays it twice):
            0.05 gives 77 pieces at 1.21x the blade's volume, 0.08 gives 51 at
            1.25x, 0.12 gives 20 at 1.32x -- against the hull's 3.13x.
        show_collision_shapes: Mark the robot's imported collision shapes
            visible. For an asset whose geoms serve as both collider and
            visual -- one mesh per link, no ``_visual`` twins -- Newton's MJCF
            importer classifies every colliding geom as collision-only and no
            renderer draws the robot at all. Flipping the flag post-import
            changes nothing physical: no masses, no shape indices, no asset
            regeneration.
        arms: The arms, in the order their slices appear in an action vector.
        gripper: Default hand for every arm that names none, as a
            :class:`GripperSpec` registry key.

            What makes the pairing declarable in one word::

                EmbodimentSpec(name="yam_robotiq", asset=..., arms=(...),
                               gripper="robotiq_2f85")

            A bimanual robot has two arms and one kind of hand, and stating it
            per arm was the duplication in miniature. An arm that wants a
            different hand states its own, which wins.

            **This is the robot's own hand, and it is no longer the only way to
            say which hand.** :attr:`Binding.gripper` names one per pairing, and
            it was held back until the two things that make it safe were done
            together, which is what the note here used to insist on: the axis is
            on ``Binding.key`` *and* on ``admits``. Without the first, two
            bindings differing only in gripper collide as duplicates; without the
            second, ``min_gripper_span`` is gated against a hand that will not be
            built. A hand named on a binding replaces every arm's -- see
            ``compile.compiler``, which clears the per-arm entries as it sets
            this one, since an arm's own name wins over the robot's default.
        arm_actuation: Gains for the arm joints.
        gripper_actuation: Gains for the finger joints, normally stiffer so a
            grasp does not slip open. Overrides the hand's own
            :attr:`GripperSpec.actuation`.
        monolithic: This robot's asset arrives with a **scene** around it, so it
            can only be used where that scene is already there.

            The mirror of :attr:`TaskRequirements.monolithic`, which says the
            same thing from the task's side, and it exists for the same reason:
            neither is a preference, both are what the file on disk is. A pairing
            a task *declares* is still admissible -- the declaration is somebody
            saying they have run it -- and every undeclared one is refused rather
            than compiled.
        unbuildable: Why this robot cannot be compiled yet, or ``""`` when it
            can. Refused by the benchmark tier's pairing matrix
            on every pairing the task does not itself declare, with this string
            as the reason.

            The same shape as :attr:`monolithic` and
            :func:`~robosimstudio.robots.registry._admits_gripper`, one level
            up: a fact readable off the specs, said as a sentence about the
            pairing rather than found out by a compile that dies about a label.
            Nothing else in ``admits`` can know it -- arm count, work surface
            and gripper span all pass for an arm whose asset imports fine and
            whose *indexing* does not -- so a half-landed robot was offered as
            ``(untried)`` on every task whose arm count it met, and each of
            those was a ``KeyError`` traceback one click away.

            A pairing the task **declares** still wins, exactly as it does for
            :attr:`monolithic`, and for a reason that survives the difference:
            the suite that declares a robot nobody has run is the *record* of
            the break, and deleting its own pairing would delete the record.
            What this buys is that the break stays in that suite instead of
            being offered on six others. The one page that must also refuse a
            declared pairing -- a menu, where the click is the whole
            interaction -- reads this field directly; see
            :meth:`~robosimstudio_studio.session.StudioSession.load_task`.

        base_xform: Where the robot's base sits, if the asset does not say.
        base_offset: Where the base sits **in a scene's work-surface frame**, so
            a mount can be derived rather than measured per pairing.

            This is the field that makes a robot portable. Across the shipped
            cells the absolute base height ranges over 0.003, 0.75 and 0.912 m
            while the base's height *above the work surface* is only ever 0,
            +0.022 or +0.11 -- the datum is a property of the room and the
            offset is a property of the robot. Stating the offset here and the
            datum on the scene lets either change without touching the other.

            Identity means "the asset already places itself", which is true of
            every embodiment until its asset is re-origined.
        ik_iterations: LM iterations per tracked frame.
        honour_roll: Whether the graph-facing IK backend should solve this
            robot's wrist roll rather than leave it free.

            A fact about the arm, which is why it is declared here rather than
            chosen per run. Wrist roll is *free* to the default solver
            (the graph-policy runtime's IK backend
            aligns position and an approach axis), and a parallel jaw closes
            along one direction -- so on a robot that leaves it free, a graph
            that computed the direction the jaws should close along has that
            direction discarded, and the width it measured along it is
            commanded along whatever the arm chose instead. On a UR5e sorting
            elongated parts, asked headings of 0 and +-90 degrees landed near
            +55 to +62 with it free -- one object in four placed, the one that
            measures the same width at every heading -- and within 0.3 degrees
            with it declared, with most objects placed.

            **The precondition is the tool centre sitting on the last joint's
            axis.** :meth:`~robosimstudio.solvers.ArmIK.solve_q_full`
            closes the roll by turning that one joint, so on a hand whose TCP is
            off it the correction sweeps the TCP around a circle -- 62 mm at 90
            degrees on the YAM -- and the position polish is capped. Set this
            only where ``ee_offset`` has no component perpendicular to that
            axis; the UR5e's is ``(0, 0, 0.1488)`` and the Panda's is zero, and
            every other hand here is off-axis.

            It costs about twice the solver iterations per waypoint, and the
            episode gets longer for a second reason worth separating from it: a
            grasp that *holds* is then carried and placed, which is work a
            missed grasp never does -- on that sorting task, more than twice the
            control steps per episode. Check the headroom before setting this on
            a robot whose graphs already run near their budget.
        urdf: The arm's kinematics as a URDF, resolved against the asset
            pool, for the solvers that work from one -- the PyRoKi family,
            which is the default. An arm that names none runs on Newton's
            on-device solver instead. The URDF has to describe the arm the
            simulator steps: the solver measures its tool-link-to-TCP transform
            against the model at two configurations and refuses one that
            drifts.
        curobo_config: cuRobo robot config for this arm, resolved against
            cuRobo's own content directory, or ``""`` for a robot with no
            planner model. Empty is the honest default and the majority case:
            it makes the planner degrade per *embodiment* rather than per
            install, so a cell whose robot has never been measured against a
            URDF falls back to the IK backend instead of planning against a
            model nobody has checked.
        curobo_urdf: The kinematics URDF the config's collision spheres are
            posed on, repository-relative. Separate from :attr:`asset` because
            the two answer to different masters -- the asset is whatever the
            benchmark exported, and this is a chain measured against it.
            ``assets/PROVENANCE.md`` records the FK disagreement per robot;
            nothing here is trusted that has not been measured there.
        curobo_time_scale: Execution-duration multiplier for planned
            trajectories. ``1`` preserves planner timing; values above one
            slow a friction-held payload without changing its geometric path.
        curobo_lock_joint: The finger joint the planner holds fixed while it
            plans, or ``""`` for a model without one. Locked where the *scene*
            holds it, not at a nominal open: the Panda reads tens of
            millimetres of phantom disagreement with its finger locked at
            40 mm while the scene holds it at 5.6. One name, because a mimic
            joint drives the other finger.
        color: Display RGB in [0, 1] for every shape the robot imports, or
            ``None`` to keep what the asset authored. State one whenever the
            asset's own colours are conversion artefacts rather than intent --
            a mesh-per-link MJCF written out by a converter carries a different
            arbitrary ``rgba`` on every geom, which renders as a harlequin and
            reads as a broken import. An asset with real materials (a
            ``class="visual"`` split and named materials) wants ``None``.
        finish: How the robot's shapes reflect light, or ``None`` for the
            finish keyed to :attr:`color` -- satin for the flat dark most
            converter-authored arms get. Shapes an :attr:`appearance` hook adds
            afterwards take the finish keyed to *their* colour, which is how a
            white shell comes out glossy beside a satin-black arm. Display only.
        appearance: Optional ``appearance(builder, shape_range) -> int``, run
            *after* :attr:`color` has been applied, for a robot whose real look
            one colour cannot state.

            The escape hatch :attr:`color` needs. Flattening a converter's
            rainbow to one grey fixes the harlequin and costs the robot any
            genuine two-tone it had -- and on an asset that welds one mesh per
            link, the parts that should differ cannot be named, only cut out.
            So this runs last and repaints the few shapes that a single colour
            gets wrong.
            :func:`~robosimstudio.robots.embodiments.yam_look.split_yam_white_tubes`
            is the one that exists.

            It may add visual-only shapes and hide originals; it must not touch
            collision geometry or inertia. Returning 0 means nothing matched,
            which leaves the robot flat -- a degraded look and never a broken
            model.
    """

    name: str
    asset: AssetSpec | None = None
    collapse_fixed_joints: bool = False
    gravcomp_restate: bool = False
    gravcomp: bool = False
    enable_self_collisions: bool = False
    cross_arm_collisions: bool = False
    reset_joint_q: bool = False
    hull_colliders: bool = False
    decompose_colliders: tuple[str, ...] = ()
    decompose_threshold: float = 0.08
    show_collision_shapes: bool = False
    arms: tuple[ArmSpec, ...] = ()
    gripper: str = ""
    arm_actuation: ActuationSpec | None = None
    gripper_actuation: ActuationSpec | None = None
    monolithic: bool = False
    unbuildable: str = ""
    base_xform: Xform = IDENTITY
    base_offset: Xform = IDENTITY
    ik_iterations: int = 8
    honour_roll: bool = False
    urdf: str = ""
    curobo_config: str = ""
    curobo_urdf: str = ""
    curobo_time_scale: float = 1.0
    curobo_lock_joint: str = ""
    color: Vec3 | None = None
    finish: Finish | None = None
    appearance: Callable[..., int] | None = None

    @property
    def import_self_collisions(self) -> bool:
        """What the *importer* is told, which is not what the robot ends up with.

        :attr:`cross_arm_collisions` needs the import to leave the pairs live so
        the per-arm filter can put back exactly the ones it means; the blanket
        filter is unrecoverable once applied, since it names no arms.
        """
        return bool(self.enable_self_collisions or self.cross_arm_collisions)
