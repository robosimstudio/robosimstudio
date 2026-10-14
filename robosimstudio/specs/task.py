# SPDX-License-Identifier: Apache-2.0
"""What to do, what it is done to, and what counts as having done it.

The top of the IR, and the only module here that reaches across all the others:
a :class:`TaskSpec` names the objects (:mod:`robosimstudio.specs.asset`), the
arrangement that steps them (:mod:`robosimstudio.specs.cell`) and the actuation
its embodiment is driven with (:mod:`robosimstudio.specs.robot`), while a
:class:`Binding` is one admissible (scene, embodiment, gripper) triple.

**Nothing writes a ``TaskSpec`` by hand.** A task is a document; the task
language lowers it to one of these, and the compiler, the ``Env``, the studio
and the runner consume it. That is what lets the asset library's vocabulary
change without touching the runtime -- a compiled goal tree is the ``Evaluator``
:attr:`TaskSpec.evaluator` has always held.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.core.render.spec import RenderSpec

from .asset import BUILD_STAGES, ObjectSpec, RegionSpec, build_stage
from .base import IDENTITY, Xform
from .physics import ArrangementSpec
from .robot import ActuationSpec

__all__ = [
    "PARK_HEIGHT",
    "PARK_PITCH",
    "ActionSpec",
    "Binding",
    "Episode",
    "EpisodeSpec",
    "ReproHooksSpec",
    "TaskRequirements",
    "TaskSpec",
    "TaskSuite",
    "park_lattice",
    "park_poses",
    "restage",
]

PARK_HEIGHT = 6.0
"""Where a parked object floats [m above the world origin], when a task states no park pose.

An episode may run a *subset* of the objects its model holds -- an instance of
a superset document -- and the rest are parked: floated out of the way, held
against gravity per world, touching nothing. Six metres up is outside every
fixed camera frustum any shipped rig declares (the deepest far plane is 5 m from
an eye below 2 m), above every workcell's ceiling, and far enough that the
lattice below never overlaps a table. A park is never *under* the floor: the
ground plane is a half-space and a body beneath it is in deep penetration.
"""

PARK_PITCH = 0.5
"""Spacing of the default park lattice [m]. Wider than any shipped object."""

_PARK_ROW = 8
"""Parked objects per lattice row, centred on x = 0."""

VIEW_OF_STAGE: dict[str, str] = {"scenery": "statics"}
"""The one view whose name is not its stage's.

A view is a :data:`~robosimstudio.specs.asset.BUILD_STAGES` stage read under the
name a task author uses for it: ``objects``, ``deformables`` and ``extras`` are
the same word on both sides, and ``statics`` selects what is built at
``scenery``. A stage not listed here is its own view name.
"""


@dataclass(frozen=True, kw_only=True)
class TaskRequirements:
    """What a pairing must provide for a task to be worth *trying*.

    The point is arithmetic, not curation. Sixteen variants against five scenes
    and four embodiments is 320 cells, almost all of them nonsense, and nobody
    is going to hand-write which. So a task states the few things that make a
    pairing impossible rather than merely untested, and the matrix tool runs
    what survives and reports what happened.

    Deliberately narrow. Only claims that can be checked off the specs go here:
    "this needs two arms" is a fact about the task, whereas "this arm can reach
    that shelf" is a measurement, and a predicate that guessed at reach would
    filter out working pairings while admitting broken ones. Reach is left to
    the tool, which finds out by compiling.

    Attributes:
        arms: How many arms the task needs, including any it merely parks. A
            routine that lays out a two-arm action vector needs two arms even
            if only one of them moves.
        needs_surface: Whether the task needs a raised work surface. A tabletop
            layout staged on bare ground puts its fixtures at ankle height.
        min_gripper_span: The widest thing that must be grasped [m]. 0 states
            nothing, which is right for a task whose graspables are not measured.
        monolithic: The task's scene *and* its robot arrive in one import, so
            neither axis can be swapped until a binding names a robot-stripped
            variant of the import. True for a task imported from one exported
            scene file. Note this blocks both axes, not just the
            scene: pairing such a task with another embodiment would import that
            robot *alongside* the one already inside the arena.
    """

    arms: int = 1
    needs_surface: bool = True
    min_gripper_span: float = 0.0
    monolithic: bool = False


@dataclass(frozen=True, kw_only=True)
class Binding:
    """One admissible (scene, embodiment) pairing for a task, and its overrides.

    A task declares the combinations it supports; a binding is one of them, plus
    whatever that particular pairing needs said about it. Anything a binding
    leaves unstated falls through to the task's own value and then the
    embodiment's, so the common case -- "this robot works here, nothing special"
    -- is a two-field entry.

    Attributes:
        static_overrides: ``(static name, asset path)`` pairs replacing what one
            of the task's statics imports, for this pairing only.

            What makes a monolithic export hostable. The tool-sorting suite imports an
            arena with a Panda welded into it, so another robot in that arena
            needs the *stripped* arena instead -- and which file that is depends
            on the pairing, not on the task. Set this together with
            ``scene_resident=False``, since the point is that the robot is no
            longer in there and the compiler must import one.
        scene_resident: The robot is already inside the scene's asset, so the
            compiler must not import it again. Distinct from the embodiment
            simply having no asset of its own: an extracted Panda is importable
            *and* present inside the robosuite arena, and which of the two a
            pairing means is a property of the pairing.
        scene: Registry name, or ``""`` for the task's native scene.
        embodiment: Registry name, or ``""`` for the task's native embodiment.
        gripper: Registry name of the hand to put on every arm, or ``""`` for
            whichever hand the embodiment declares.

            ``""`` is not the same as naming the embodiment's own hand: it means
            *this pairing does not care*, which is what keeps every binding
            written before this field existed reading the same.
        mount: Where the robot's base goes, in world coordinates. ``None``
            derives it from the scene's surface frame and the embodiment's
            :attr:`EmbodimentSpec.base_offset`, which is what makes a new
            pairing a declaration rather than a measurement.
        mount_point: Name a mount the scene already declares, instead of
            deriving one. Ignored when ``mount`` is set.
        work_arm: Which arm does the task. ``""`` is the embodiment's first.
        support_arm: Which arm holds, anchors or steadies. ``""`` means the task
            needs only one arm; naming one on an embodiment that has no such arm
            is what makes a bimanual task inadmissible on a single-armed robot.
        arm_actuation: Gains for this pairing specifically. A robot stiff enough
            to seat a connector is a fact about the *pair*, not about either
            half, which is why this sits here as well as on the task.
        gripper_actuation: The same for the fingers.
        cross_arm_collisions: Whether the two arms collide with each other, per
            :attr:`EmbodimentSpec.cross_arm_collisions`. ``None`` takes the
            robot's own setting.

            Here as well as on the embodiment for the same reason the gains are:
            whether two arms may pass through each other only *matters* on a
            task that puts them in the same place, and a task that keeps them
            apart should not pay for the pairs. The fold is the first case --
            the carrying hand lands its edge on the half the other hand is
            pressing -- so it is the first binding to state it.
        graphs: Which of the task's shipped graphs are known to work on this
            pairing. Empty means none -- the task still compiles and can be
            teleoperated, which is the honest state for a pairing nobody has
            written a policy for yet.
        notes: Anything a reader of the compatibility matrix should know.
    """

    scene: str = ""
    embodiment: str = ""
    gripper: str = ""
    mount: Xform | None = None
    mount_point: str = ""
    work_arm: str = ""
    support_arm: str = ""
    arm_actuation: ActuationSpec | None = None
    gripper_actuation: ActuationSpec | None = None
    cross_arm_collisions: bool | None = None
    graphs: tuple[str, ...] = ()
    scene_resident: bool = False
    static_overrides: tuple[tuple[str, str], ...] = ()
    notes: str = ""

    def key(self, task: TaskSpec) -> tuple[str, str, str]:
        """This binding's ``(scene, embodiment, gripper)``, task defaults filled in.

        The gripper is *not* defaulted to the embodiment's own hand, because
        resolving that needs the registry and a spec may not reach for it. ``""``
        stays ``""`` here and ``registry.pairing_key`` is what resolves it when a
        caller needs the two spellings to compare equal.
        """
        return (self.scene or task.scene, self.embodiment or task.embodiment, self.gripper)


@dataclass(frozen=True, kw_only=True)
class ActionSpec:
    """What a variant's Cartesian action vectors *mean*.

    Every field here describes a **convention a policy was trained against**,
    not a capability of the robot -- which is why it sits on the task rather
    than on the embodiment, and why the defaults are this package's own rather
    than any benchmark's. A task reproducing a reference states the reference's
    conventions here and its checkpoints then step unaltered.

    The distinction that motivates the whole class: ``[dx, dy, dz, ...]`` is not
    a pose, it is six numbers plus an agreement about what they are six numbers
    *of*. Get the agreement wrong and nothing raises -- the arm simply moves,
    somewhere else, every step.

    Attributes:
        ee_delta_frame: Which axes ``ee_delta``'s translations and rotations are
            in -- ``"world"``, or ``"base"`` for the work arm's own base link
            (:attr:`ArmSpec.base_body_name`). A robot mounted yawed from world
            makes these differ by exactly that yaw, which for the SimplerEnv
            cells is 180 degrees.
        ee_delta_convention: One of
            :data:`~robosimstudio.solvers.EE_DELTA_CONVENTIONS`.
        ee_delta_pos_scale: Metres per unit of ``ee_delta`` translation. 1.0
            means the action is already in metres, which is what a controller
            configured ``normalize_action=False`` emits.
        ee_delta_rot_scale: Radians per unit of ``ee_delta`` rotation.
        gripper_range: The interval a gripper command arrives in, for both
            ``ee_delta`` and ``osc_pose``. ``(0.0, 1.0)`` is this package's
            0 open / 1 closed; ``(-1.0, 1.0)`` is what robosuite- and
            ManiSkill-trained checkpoints emit.
        ee_delta_use_target: Whether ``ee_delta`` composes onto the standing
            setpoint (``True``) or onto the arm's measured pose (``False``).
            ManiSkill spells these ``arm_pd_ee_target_delta_pose`` and
            ``arm_pd_ee_delta_pose``, and **one benchmark can use both**:
            SimplerEnv drives its WidowX with a target and its Google robot
            without one. A policy trained against the second and stepped through
            the first accumulates error it never saw in training.
        gripper_delta: Whether the gripper number is a change to the standing
            command rather than an absolute position -- ManiSkill's
            ``gripper_pd_joint_target_delta_pos`` against ``gripper_pd_joint_pos``.
            Also per robot in the same benchmark, and RT-1's own wrapper emits
            to match: relative for the Google robot, absolute for the WidowX.
        arm_planner_limits: ``(velocity, acceleration, jerk)`` for the arm's
            plan, in rad/s, rad/s^2 and rad/s^3.
        gripper_planner_limits: The same for the jaw. Separate because a jaw is
            far lighter than an arm and its limits are not the arm's.
    """

    ee_delta_frame: str = "world"
    ee_delta_convention: str = "align"
    ee_delta_pos_scale: float = 1.0
    ee_delta_rot_scale: float = 1.0
    gripper_range: tuple[float, float] = (0.0, 1.0)
    ee_delta_use_target: bool = True
    gripper_delta: bool = False
    arm_planner_limits: tuple[float, float, float] = (1.5, 2.0, 50.0)
    gripper_planner_limits: tuple[float, float, float] = (1.0, 7.0, 50.0)

    def __post_init__(self) -> None:
        if self.ee_delta_frame not in ("world", "base"):
            raise ValueError(f"ee_delta_frame must be 'world' or 'base', got {self.ee_delta_frame!r}")


@dataclass(frozen=True, kw_only=True)
class EpisodeSpec:
    """How episodes of a task vary from one another.

    Attributes:
        count: How many episodes the task defines.
        seed: Base seed. Episode ``i`` is sampled with ``seed + i``, so an
            episode id reproduces exactly regardless of what ran before it.
        sampler: ``(spec, episode_id, rng) -> dict`` returning the per-episode
            overrides -- object poses, a chosen instruction, a chosen layout.
            A plain callable, referenced directly: Python-first means a task
            names its sampler function, not a registry key that resolves to one.
        model_key_fields: Which sampled keys change the *model* rather than
            only its state. An environment rebuilds the scene only when one of
            these differs from the last episode, and merely re-poses otherwise.
    """

    count: int = 1
    seed: int = 0
    sampler: Callable[..., dict[str, Any]] | None = None
    model_key_fields: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Episode:
    """One concrete episode: what varies, plus what identifies it.

    Attributes:
        episode_id: Index within the task.
        seed: The seed this episode was sampled with.
        instruction: The language instruction for this episode.
        overrides: Sampled values the compiler and entities read.
        model_key: Identity of the *model* this episode needs. Two episodes
            sharing a key can run back to back without rebuilding the scene.
    """

    episode_id: int
    seed: int
    instruction: str = ""
    overrides: tuple[tuple[str, Any], ...] = ()
    model_key: str = ""

    def get(self, key: str, default: Any = None) -> Any:
        """One override, or ``default``."""
        for name, value in self.overrides:
            if name == key:
                return value
        return default

    @property
    def as_dict(self) -> dict[str, Any]:
        return dict(self.overrides)

    @property
    def active(self) -> frozenset[str] | None:
        """Which of the model's entities this episode runs, or ``None`` for every one.

        The override ``("active", (names...))``, as a set. A sampler emits it
        when a document is an *instance* of a superset -- four of the eight
        syringes, this document's tools out of the tier's catalogue -- and the
        environment parks whatever is not named: floated out of reach, held
        against gravity, touching nothing, and hidden from the handles the
        evaluator, the observation and the privileged tools enumerate.

        **State, never model.** The set is deliberately not a model-key field:
        the whole point is that every instance of one superset shares one
        model, so a wave can hold twelve of them. ``None`` -- no override --
        is "every entity", which is what every document written before this
        existed means.
        """
        names = self.get("active")
        if names is None:
            return None
        return frozenset(str(name) for name in names)


@dataclass(frozen=True, kw_only=True)
class ReproHooksSpec:
    """A suite's standalone substep *hooks*, declared for repro export.

    The solvers themselves are rebuilt from the
    :class:`~robosimstudio.specs.physics.PhysicsManifest` the composer recorded,
    generically, inside the bundle. What a manifest cannot carry is a kernel: a
    grasp latch, a bond field, a handle pin. A suite whose entities contribute
    such stages declares here the *standalone* module (imports only newton,
    warp, numpy; names nothing of this project -- the bundle leak scan reads it)
    that rebuilds them from plain parameters, plus the callables the exporter
    uses to compute those parameters and to tape per-step aux state.

    The module must expose ``build_hooks(model, params) -> hooks`` where
    ``hooks`` carries ``stages`` (phase name -> list of ``(state, dt)``
    callables; the phases are ``rigid_force``, ``pre_solve``, ``post_solve``,
    ``mirror``) and ``aux_sinks`` (name -> device array the replay uploads each
    control step).

    Attributes:
        module: Dotted import path or file path of the standalone module. The
            exporter copies its file into the bundle as ``task_physics.py``
            verbatim, and scans it like every other bundle file.
        params: ``params(scene: CompiledScene) -> dict`` of what ``build_hooks``
            needs -- index lists, gains. Values must be JSON scalars/lists or
            numpy arrays (arrays travel in the bundle's npz).
        taps: Optional ``taps(scene) -> {name: () -> np.ndarray}``: per-step
            readers of task state a replay must re-upload (latch flags, hold
            targets). Recorded by the tape as ``aux/<name>`` and fed back into
            ``aux_sinks[name]`` each step.
    """

    module: str
    params: Callable[..., dict[str, Any]]
    taps: Callable[..., dict[str, Any]] | None = None


@dataclass(frozen=True, kw_only=True)
class TaskSpec:
    """One benchmark variant: a scene, an instruction, and how it is scored.

    Attributes:
        name: Variant name. Resolved as ``"<suite>/<name>"``.
        instruction: Default language instruction. An episode may override it.
        embodiment: Registry name of the robot.
        gripper: Registry name of the hand this task is currently pinned to, or
            ``""`` for the robot's own. Never authored by a suite -- a task does
            not own a hand any more than it owns a robot. It is where ``Env``
            records the ``--gripper`` selection, so that the one
            ``binding_for()`` the compiler makes resolves to the pairing that was
            asked for, exactly as :attr:`scene` and :attr:`embodiment` already do
            for their axes.
        embodiment_pose: Where the robot's base goes.
        entities: Everything in the scene except the robot, in one list.

            A spec says its own :data:`~robosimstudio.specs.asset.BUILD_STAGES`
            stage now, so the compiler runs one loop and a new kind declares
            where it belongs rather than being filed under whichever tuple was
            least wrong.

            The four names survive as *views* -- :attr:`statics`,
            :attr:`objects`, :attr:`deformables`, :attr:`extras` -- because plenty of
            readers legitimately want one sort ("every cloth's probe"), and a
            view over one list cannot drift from it the way a second tuple can.

            A suite-owned spec the generic kinds cannot express -- a
            capsule-chain cable, an articulated fixture. Each names its entity
            class via an ``entity`` attribute (the same seam
            :attr:`StaticSpec.entity` and a :class:`ClothSpec` subclass use)
            and carries a ``name``; the compiler builds them after the deformables
            and hands the built entities to the evaluator through
            ``WorldState.extras``. This is the "adding a new *kind* of thing
            adds one entity class" story made literal: the spec class, the
            entity class and the physics all live in the suite's directory.
        regions: Goal regions.
        cameras: The camera rig.
        evaluator: A direct reference to an evaluator class or factory, called
            as ``evaluator(task)``. Not a registry key: it is handed in by
            whoever built the spec, which for a document is the goal tree
            compiled from its ``(:goal ...)``, so a reference is the shortest
            correct thing and a name would only be a second way to lose it.
        routine: An optional scripted policy shipped with the task, called as
            ``routine(env)``.
        tools: An optional ``(registry, env, bus) -> None`` registering the
            suite's own privileged tools, the same shape the graph-policy
            runtime's ``register_sim_tools`` has.

            A direct reference for the same reason ``evaluator`` is one. What it
            exists for is that the generic ``sim.*`` set deliberately knows
            nothing about what task is running, and some ground-truth graphs
            need something that does -- which goal is still outstanding, what a
            bond field says. A suite that needs one contributes it here rather
            than growing the generic set a task-shaped tool.
        repro_hooks: The standalone twin of any entity substep kernels, for
            repro export, or ``None``. The solvers need none: the composer's
            manifest rebuilds them. A task whose entities contribute stages is
            refused by ``Env.export_repro`` until it declares one -- see
            :class:`ReproHooksSpec`.
        graphs: Shipped graph directories, ``name -> path`` relative to the
            suite. ``"gt"`` is the ground-truth graph by convention.
        physics: The arrangement: rates, a backend per physics domain, the
            coupler and the collision policy. See
            :class:`~robosimstudio.specs.physics.ArrangementSpec`. Never
            authored on a task directly: a document stages a versioned name and
            lowering resolves it.
        arm_actuation: Per-task override of the embodiment's arm servo gains,
            or ``None`` for the embodiment's own. An insertion task runs the
            same arm several times stiffer than a transport task does, and that
            is a fact about the task rather than the robot.
        gripper_actuation: The same for the finger servos.
        episodes: How episodes vary.
        control_mode: ``"joint_position"``, ``"ee_delta"``, ``"osc_pose"`` or
            ``"ee_pose_abs"``. What the variant's own routine and shipped graph
            speak; a run may override it (see the benchmark tier's ``Env``).
        action: What this variant's Cartesian action vectors *mean* -- see
            :class:`ActionSpec`. Separate from :attr:`control_mode`, which names
            *which* space a bare ``step(action)`` speaks: one is the layout, this
            is the convention inside it.
        render: How this variant is lit, for whichever backend is asked, or
            ``None`` to leave every backend on its own default. A task
            reproducing a benchmark that photographs its scene states the
            benchmark's own rig here, because on a visual-matching task the
            lighting is part of the observation rather than a display setting.
        max_steps: Control steps before truncation.
        terminate_on_success: Whether a successful evaluator result immediately
            ends the episode. Disable this for scripted benchmark routines that
            must finish post-success motions such as retracting and returning
            the robot home; success is still reported in ``info``.
        servo_speed_mps: The Cartesian speed [m/s] this task's graphs were
            written and measured with the tool servoed at, or ``None`` for the
            setpoint law's own cruise (about 0.31 m/s from the OSC gains and
            ``output_max``). ``(:servo-speed-mps X)`` in the document's
            staging. A graph's budgets, stall windows and fine legs are tuned
            against how fast the tool actually moved, and that used to be a
            fact about the host: a serial servo turn spanned two or three
            simulator steps and the law advanced once per turn, so the tool
            crossed the cell at 0.10-0.13 m/s on the machines the paper's
            graphs were authored on. Both runners now advance the law once per
            simulator step, so the speed a suite relied on is stated here as a
            physical quantity rather than left to whatever the machine did.
            The servo turns it into a base factor ``min(1, X / cruise)`` on
            the law's ceiling and a graph's own ``speed_scale`` is relative to
            that -- see the graph-policy runtime's ``CartesianServo.base_speed_scale``
            and the fleet stepper's ``_lead``.
        render_skip: Entity names a visual-matching composite supplies from a
            photograph rather than drawing. An entity whose camera has a
            ``background`` may legitimately come from the picture -- SimplerEnv's
            sink is physics-only and is not drawn.
        fleet_base: The ``"<suite>/<variant>"`` whose **model** a fleet builds
            for this document, or ``""`` for its own. ``(:fleet-base X)`` in the
            document's staging. What a superset instance states: its objects
            are a subset of the base's, drawn at this document's poses, and a
            wave runs it as one world of the base's model with everything the
            document does not name parked (:attr:`Episode.active`). The serial
            environment ignores it and builds the document's own model, which
            stays the paper's reference.
        parking: Where an object goes when an episode does not run it --
            ``(name, xform)`` pairs, world coordinates. An object not listed
            takes :func:`park_poses`' lattice. A suite states these when its
            rig looks somewhere the default lattice would show.
    """

    name: str
    instruction: str = ""
    scene: str = ""
    embodiment: str = ""
    gripper: str = ""
    embodiment_pose: Xform = IDENTITY
    bindings: tuple[Binding, ...] = ()
    requires: TaskRequirements = field(default_factory=TaskRequirements)
    entities: tuple[Any, ...] = ()
    regions: tuple[RegionSpec, ...] = ()
    cameras: tuple[CameraSpec, ...] = ()
    evaluator: Callable[..., Any] | None = None
    routine: Callable[..., Any] | None = None
    tools: Callable[..., Any] | None = None
    repro_hooks: ReproHooksSpec | None = None
    graphs: tuple[tuple[str, str], ...] = ()
    physics: ArrangementSpec = field(default_factory=ArrangementSpec)
    arm_actuation: ActuationSpec | None = None
    gripper_actuation: ActuationSpec | None = None
    episodes: EpisodeSpec = field(default_factory=EpisodeSpec)
    control_mode: str = "joint_position"
    action: ActionSpec = field(default_factory=ActionSpec)
    render: RenderSpec | None = None
    max_steps: int = 600
    terminate_on_success: bool = True
    servo_speed_mps: float | None = None
    render_skip: tuple[str, ...] = ()
    fleet_base: str = ""
    parking: tuple[tuple[str, Xform], ...] = ()

    suite: str = ""
    """Filled in by :class:`TaskSuite` when the suite is constructed."""

    @property
    def full_name(self) -> str:
        return f"{self.suite}/{self.name}" if self.suite else self.name

    def object(self, name: str) -> ObjectSpec:
        for obj in self.objects:
            if obj.name == name:
                return obj
        raise KeyError(f"task {self.full_name!r} has no object {name!r}")

    def region(self, name: str) -> RegionSpec:
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"task {self.full_name!r} has no region {name!r}")

    def graph_path(self, name: str) -> str:
        for key, path in self.graphs:
            if key == name:
                return path
        raise KeyError(f"task {self.full_name!r} ships no graph {name!r}; has {[k for k, _ in self.graphs]}")

    # ------------------------------------------------------------- bindings
    @property
    def native_binding(self) -> Binding:
        """The pairing this task was authored against.

        Synthesized when :attr:`bindings` is empty, which is what keeps a task
        that has never thought about being portable working unchanged: it is
        admissible on exactly its own scene and robot, and every graph it ships
        is supported there.
        """
        if self.bindings:
            return self.bindings[0]
        return Binding(scene=self.scene, embodiment=self.embodiment, graphs=tuple(k for k, _ in self.graphs))

    # ------------------------------------------------------------ build
    STAGE_OF_VIEW: ClassVar[dict[str, str]] = {VIEW_OF_STAGE.get(stage, stage): stage for stage in BUILD_STAGES}
    """View name -> the :data:`~robosimstudio.specs.asset.BUILD_STAGES` stage it selects.

    Derived, not written out: the stages are the one table and a view is a stage
    under the name its readers use, which for all but :data:`VIEW_OF_STAGE`'s one
    entry is the stage's own. A second literal here was a second vocabulary to
    keep in step with the first.
    """

    def at_stage(self, stage: str) -> tuple[Any, ...]:
        """Everything built at one stage, in declaration order."""
        return tuple(spec for spec in self.entities if build_stage(spec) == stage)

    @property
    def statics(self) -> tuple[Any, ...]:
        """Scenery. A *view* over :attr:`entities`, not a second tuple."""
        return self.at_stage(self.STAGE_OF_VIEW["statics"])

    @property
    def objects(self) -> tuple[Any, ...]:
        """Manipulable rigid bodies."""
        return self.at_stage(self.STAGE_OF_VIEW["objects"])

    @property
    def deformables(self) -> tuple[Any, ...]:
        """Deformables."""
        return self.at_stage(self.STAGE_OF_VIEW["deformables"])

    @property
    def extras(self) -> tuple[Any, ...]:
        """Suite-owned entities the generic kinds cannot express."""
        return self.at_stage(self.STAGE_OF_VIEW["extras"])

    def build_order(self) -> tuple[Any, ...]:
        """:attr:`entities`, in the order the compiler builds them."""
        return tuple(spec for stage in BUILD_STAGES for spec in self.at_stage(stage))

    def with_entities(self, **by_view: Any) -> TaskSpec:
        """A copy with some stages replaced and the rest kept.

        What ``dataclasses.replace(spec, statics=...)`` used to do and cannot any
        more, now that the four tuples are one list: swapping one sort means
        rebuilding the list around it. Every task loader restages *something*
        after lowering -- a static's export, an object's collider, its own extras
        -- so this is the seam that has to stay pleasant.

        Raises:
            ValueError: On a view name that is not one of :attr:`STAGE_OF_VIEW`,
                rather than silently doing nothing. A typo here would drop an
                entity out of the scene, and a scene missing a fixture compiles
                perfectly well.
        """
        unknown = sorted(set(by_view) - set(self.STAGE_OF_VIEW))
        if unknown:
            raise ValueError(f"unknown entity view(s) {unknown}; have {sorted(self.STAGE_OF_VIEW)}")
        kept = {view: tuple(by_view.get(view, getattr(self, view))) for view in self.STAGE_OF_VIEW}
        return replace(self, entities=tuple(spec for view in self.STAGE_OF_VIEW for spec in kept[view]))

    def binding_for(self, scene: str = "", embodiment: str = "", gripper: str = "") -> Binding:
        """The declared pairing for one scene, embodiment and hand.

        Empty arguments mean the task's own, so ``binding_for()`` is the native
        pairing. A combination nobody declared raises rather than being compiled
        on optimistic defaults -- a robot silently mounted at a pose no one chose
        produces a scene that looks plausible and reaches nothing.
        """
        want = (scene or self.scene, embodiment or self.embodiment, gripper or self.gripper)
        for binding in self.bindings or (self.native_binding,):
            if binding.key(self) == want:
                return binding
        offered = ["/".join(x or "-" for x in b.key(self)) for b in (self.bindings or (self.native_binding,))]
        raise KeyError(
            f"task {self.full_name!r} declares no binding for scene={want[0]!r} embodiment={want[1]!r} "
            f"gripper={want[2]!r}; has {offered}"
        )

    @property
    def binding_keys(self) -> tuple[tuple[str, str, str], ...]:
        """Every ``(scene, embodiment, gripper)`` this task declares."""
        return tuple(b.key(self) for b in (self.bindings or (self.native_binding,)))

    def with_pairing(self, scene: str = "", embodiment: str = "", gripper: str = "") -> TaskSpec:
        """This task pinned to a pairing, declaring one if it did not.

        For the callers that *explore* rather than measure -- compiling the whole
        admissible product (:mod:`robosimstudio_bench.taskbuild.matrix`),
        measuring a hand that no task has been welded onto yet, and the studio's
        task menu, whose ``(untried)`` options are exactly these combinations
        offered for a click. All of them need to build a combination nobody has
        claimed.

        This is deliberately not what ``Env`` does. ``Env`` refuses an undeclared
        pairing, because a robot silently mounted at a pose no one chose builds a
        scene that looks plausible and reaches nothing, and that refusal is worth
        keeping in the path everything else goes through. Synthesising the
        binding is the *explicit* opt-out, and it reads as one at the call site.
        """
        scene = scene or self.scene
        embodiment = embodiment or self.embodiment
        gripper = gripper or self.gripper
        bindings = self.bindings
        if (scene, embodiment, gripper) not in self.binding_keys:
            bindings = (
                *(bindings or (self.native_binding,)),
                Binding(scene=scene, embodiment=embodiment, gripper=gripper),
            )
        return replace(self, scene=scene, embodiment=embodiment, gripper=gripper, bindings=bindings)

    def with_arms_solid(self, on: bool = True) -> TaskSpec:
        """This task, with its **native** pairing's two arms solid to each other.

        The one-line form of what every bimanual suite on this robot has to say
        for itself. ``cross_arm_collisions`` is a field on a
        :class:`Binding` rather than on the robot because whether the arms may
        pass through each other is a fact about the pairing -- it only costs
        anything on a task that puts both hands in the same place, and the pairs
        are not free. But *most* of the suites on a two-armed cell want it, and
        the alternative to this method is each of them reaching into
        ``lower()``'s synthesised binding with its own ``replace``, which is four
        copies of one line and four chances to edit ``bindings[1]`` by mistake.

        Only the native row is touched. A foreign pairing declared for the matrix
        -- a cable-routing variant's ``yam_bimanual`` row, which compiles and has
        no policy -- is a claim about a robot this task was never authored
        against, and quietly changing its physics too would make that claim
        about something nobody measured.

        Args:
            on: ``True`` makes the arms solid, ``False`` filters them apart. The
                field is tri-state and this method only ever writes the two
                decided values, because a task calling this has an opinion --
                ``None`` is what a task that has never thought about it leaves.

        Returns:
            A copy whose ``bindings[0]`` carries the setting, with the native row
            synthesised first when the task declared none.
        """
        bindings = self.bindings or (self.native_binding,)
        return replace(self, bindings=(replace(bindings[0], cross_arm_collisions=on), *bindings[1:]))


def _check_bindings(task: TaskSpec) -> None:
    """The invariants a variant's declared bindings must hold.

    Checked at suite construction, i.e. at import, so a mistake in a binding
    table is an error where it was written rather than a compile failure much
    later on whichever pairing somebody happened to select first.
    """
    if not task.bindings:
        return
    native = task.bindings[0]
    if native.key(task) != (task.scene, task.embodiment, ""):
        raise ValueError(
            f"task {task.name!r}: bindings[0] must be the native pairing "
            f"{(task.scene, task.embodiment, '')}, got {native.key(task)}"
        )
    if native.arm_actuation is not None or native.gripper_actuation is not None:
        # The native pairing's gains belong on the task, where they already are.
        # Allowing both would give one pairing two sources of truth and a
        # precedence rule nobody would remember.
        raise ValueError(f"task {task.name!r}: the native binding must not override actuation; state it on the task")
    seen: set[tuple[str, str, str]] = set()
    for binding in task.bindings:
        key = binding.key(task)
        if key in seen:
            raise ValueError(
                f"task {task.name!r}: duplicate binding for scene={key[0]!r} embodiment={key[1]!r} gripper={key[2]!r}"
            )
        seen.add(key)
        shipped = {k for k, _ in task.graphs}
        unknown = set(binding.graphs) - shipped
        if unknown:
            raise ValueError(
                f"task {task.name!r}: binding {key} claims graphs {sorted(unknown)} the task does not ship "
                f"({sorted(shipped)})"
            )


def restage(spec: TaskSpec, **changes: Any) -> TaskSpec:
    """``dataclasses.replace`` for a task, entity views included.

    A loader's last act is almost always to restage what the document could not
    say -- a static's export for this pairing, an object's collider, its own
    extras -- mixed in with ordinary field changes:

        restage(lowered.spec, objects=(), deformables=(_towel(),), routine=DrapeRoutine)

    ``replace`` cannot do the first half any more: the four tuples are one
    ``entities`` list, so swapping one sort means rebuilding the list rather than
    assigning a field.

    **The view keyword is not decorative.** It names which of the existing
    entities are *replaced*; where the new spec is then built comes from its own
    stage. ``restage(spec, deformables=(x,))`` drops the deformables and keeps
    the extras, ``restage(spec, extras=(x,))`` the other way round, and ``x``
    lands at :func:`~robosimstudio.specs.asset.build_stage` ``(x)`` either way --
    so a stage-less spec handed in as ``deformables=`` is built among the extras.

    Raises:
        ValueError: On an unknown entity view -- see :meth:`TaskSpec.with_entities`.
    """
    views = {k: v for k, v in changes.items() if k in TaskSpec.STAGE_OF_VIEW}
    rest = {k: v for k, v in changes.items() if k not in TaskSpec.STAGE_OF_VIEW}
    out = spec.with_entities(**views) if views else spec
    return replace(out, **rest) if rest else out


def park_lattice(index: int) -> Xform:
    """The ``index``-th slot of the default park lattice, in world coordinates.

    :data:`_PARK_ROW` slots a row centred on x = 0, :data:`PARK_PITCH` apart,
    :data:`PARK_HEIGHT` up, identity orientation. The one rule both
    :func:`park_poses` and a hand-assembled scene's parking use.
    """
    row, col = divmod(index, _PARK_ROW)
    return ((col - (_PARK_ROW - 1) * 0.5) * PARK_PITCH, row * PARK_PITCH, PARK_HEIGHT, 0.0, 0.0, 0.0, 1.0)


def park_poses(task: TaskSpec) -> dict[str, Xform]:
    """Where each of a task's objects goes when an episode parks it. World coordinates.

    :attr:`TaskSpec.parking` first, then the default rule for every object the
    task did not place: a lattice :data:`PARK_HEIGHT` above the world origin,
    :data:`PARK_PITCH` apart, in ``task.objects`` order, so the poses are a
    function of the document alone and two builds of one superset park a body
    in the same place. Every pose is distinct, so parked bodies never touch
    each other; every pose is far above the scene, so they touch nothing else
    and no fixed camera frames them. Orientation is identity -- a parked body
    is not looked at.

    Only ``objects`` are listed. Scenery is welded and never parked; a
    deformable has no free joint to park with, and an episode that names one
    inactive is refused where the parking is written.
    """
    stated = {str(name): tuple(float(v) for v in xform) for name, xform in task.parking}
    out: dict[str, Xform] = {}
    for index, spec in enumerate(task.objects):
        if spec.name in stated:
            out[spec.name] = stated[spec.name]  # type: ignore[assignment]
            continue
        out[spec.name] = park_lattice(index)
    return out


@dataclass(frozen=True, kw_only=True)
class TaskSuite:
    """A family of variants sharing an evaluator, a routine and shipped graphs.

    The registered unit. Constructing one stamps each variant with the suite's
    name, so ``TaskSpec.full_name`` and the ``"<suite>/<variant>"`` lookup agree
    without a suite file repeating itself.

    Attributes:
        name: Registry key.
        variants: The task variants.
        description: One line, for listings.
        provenance: Where the benchmark this reproduces came from.
        home: The directory this suite was loaded from, when it came from one.

            Stamped by the discovery pass rather than declared, and it is what a
            ``(:graph ...)`` reference resolves against. A suite is a directory
            now, so "where its graphs are" is a fact about where it was found --
            not something to recompute by importing a package and taking its
            parent, which is what it used to be and which cannot work for a
            directory that was never on ``sys.path``.
    """

    name: str
    variants: tuple[TaskSpec, ...]
    description: str = ""
    provenance: str = ""
    home: Path | None = None

    def __post_init__(self) -> None:
        if not self.variants:
            raise ValueError(f"suite {self.name!r} declares no variants")
        stamped = tuple(replace(v, suite=self.name) for v in self.variants)
        object.__setattr__(self, "variants", stamped)
        names = [v.name for v in stamped]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"suite {self.name!r} has duplicate variants {sorted(duplicates)}")
        for variant in stamped:
            _check_bindings(variant)

    def variant(self, name: str) -> TaskSpec:
        for spec in self.variants:
            if spec.name == name:
                return spec
        raise KeyError(f"suite {self.name!r} has no variant {name!r}; has {[v.name for v in self.variants]}")

    @property
    def variant_names(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.variants)
