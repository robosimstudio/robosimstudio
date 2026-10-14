# SPDX-License-Identifier: Apache-2.0
"""One model, many worlds, one episode each.

:class:`~robosimstudio_bench.envs.env.Env` is one episode at a time: apply an
action, step, read the state back, score it. Every one of those reads is a device
synchronize, and at one world they are free next to the step. At five hundred
they are the whole cost, so this class inverts the arrangement:

* **Nothing is read back per step.** The step is a graph launch and returns to a
  host that has learned nothing. What the run needs -- has this world finished,
  where is its hammer -- comes from a :class:`FleetSnapshot`, taken every
  ``eval_every`` steps as *one* batched copy covering every world.
* **A snapshot's per-world views index like the single-world model.** That is
  :class:`~robosimstudio.core.worlds.WorldMap`'s guarantee, and it is what lets
  the task evaluators, the handles and the success predicates run unmodified
  against world *w*: they were written against one world's arrays, and a world's
  slice *is* one world's arrays.
* **A world that is not being worked on is masked, not stopped.** It keeps
  stepping -- it is in the same model, and the graph does not branch -- but
  nothing writes its targets, so it holds its last pose. That is what makes an
  episode finishing early cost nothing but physics.

Resets are per wave, not per step: every world's start pose is composed into one
host array and written with one assign, outside any captured graph. The scene is
built once and re-posed thereafter, which is the same economy
:meth:`Env.reset` already relies on.

What this class deliberately does *not* do is drive the robot. It offers the
batched primitives -- IK, joint segments, grippers -- and lets whatever is
driving the fleet decide which world gets what. A wave of scripted policies and a
wave of graph runtimes want very different things from those primitives, and the
single-world path has the same split between ``Env`` and the action spaces.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.core.readback import launch_finite_check
from robosimstudio.core.streams import capturing, reading
from robosimstudio.core.worlds import WorldMap
from robosimstudio.scene.compiler import CannotReplicate, compile_scene
from robosimstudio.scene.entity import Entity
from robosimstudio.scene.ik_model import build_ik_rig, plan_ik_model
from robosimstudio.scene.parking import GravcompParking, parked_objects, restrict_contacts, scene_park_poses
from robosimstudio.scene.state import WorldState
from robosimstudio.solvers.control.batched import GripperBank, SegmentDriver
from robosimstudio.specs import Episode, TaskSpec
from robosimstudio_bench.goals.context import EvalResult
from robosimstudio_bench.taskbuild.episodes import make_episode

__all__ = ["FleetEnv", "FleetSnapshot", "scoring_refusal"]


def scoring_refusal(scene) -> str:
    """Why this scene's *scoring* cannot be replicated yet, or empty when it can.

    The compiler refuses a scene whose **physics** cannot be replicated -- an
    entity contributing substep kernels with world-0 indices baked in. This is
    the other half, and it belongs here rather than there because it is about
    what :meth:`FleetEnv.context` can fill: ``particles`` and ``entities`` are
    both empty, so a task whose goal reads either would score every world from
    world 0's arrays, or raise ``BindError`` reaching for a published state that
    is not there.

    A deformable's physics *does* replicate -- measured on
    ``cloth_fold/towel_half``: four worlds settle and step to within 2e-10 of
    each other -- so this is a gap in the scoring, stated as one. Lift it by
    giving :meth:`FleetEnv.context` a per-world :class:`ParticleProbe` and a
    per-world view of each extra entity, **both**, not the probe alone.
    """
    # An entity that opened a probe per world is not refused: a wave hands a
    # goal that world's probe, over that world's block of the particle array.
    # One that did not would answer every world with the first world's
    # particles -- the physics replicates correctly either way, which is what
    # makes it worth refusing rather than discovering.
    particled = sorted(
        n for n, e in scene.by_name.items() if e.carries_particles and not callable(getattr(e, "probe_for", None))
    )
    if particled:
        return (
            f"the scene has entities made of particles ({', '.join(particled)}) and a wave has no per-world "
            "particle probe for them, so their success would be scored from one world's particles. The "
            "physics replicates correctly -- only the scoring is missing"
        )
    # "Does this publish", not "which view holds it". The old test was
    # ``scene.extras``, which is a build-order bin: a publisher that happened to
    # own a handle sat in ``objects`` and slipped through, and one that published
    # nothing sat in ``extras`` and was refused for it.
    #
    # A publisher that has declared itself world aware is *not* refused. It reads
    # ``WorldState.world`` and answers off that world's rows, which is the whole
    # of what was missing -- and :meth:`FleetEnv.context` now hands it the world
    # and hands the goal language the entity objects to ask.
    publishers = sorted(
        name
        for name, entity in scene.by_name.items()
        if type(entity).published_states is not Entity.published_states
        and not getattr(type(entity), "world_aware", False)
    )
    if publishers:
        return (
            f"the scene has entities publishing states ({', '.join(publishers)}) that a wave cannot read "
            "per world. Each one needs a world axis over whatever it publishes from and "
            "`world_aware = True` to say so"
        )
    return ""


_BUFFER_PARAMS = frozenset({"njmax", "nconmax", "rigid_contact_max"})
"""Backend params a superset base may raise: they size arrays, not the solve."""


def _material_physics(arrangement: Any) -> tuple:
    """What an arrangement integrates, with the buffer sizes left out.

    A superset base exists partly to give the union bigger constraint buffers --
    ``vab_sharps_union@v1`` is ``vab_500hz@v1`` with ``njmax`` 8192 -- and an
    instance staging the smaller one is not thereby staging different physics.
    Everything that changes the *result* is here: the rate, the substeps, the
    coupler, and which solver owns which domain.
    """
    if arrangement is None:
        return ()
    backends = tuple(
        (domain, getattr(spec, "solver", ""), tuple(sorted((k, v) for k, v in spec.params if k not in _BUFFER_PARAMS)))
        for domain, spec in getattr(arrangement, "backends", ())
    )
    return (
        float(getattr(arrangement, "control_hz", 0.0)),
        int(getattr(arrangement, "substeps", 0)),
        repr(getattr(arrangement, "coupling", None)),
        repr(getattr(arrangement, "collision", None)),
        backends,
    )


def _staged_name(field: str, value: Any) -> str:
    """What a document staged, named rather than dumped.

    ``physics`` and ``cameras`` are *registered* specs, and a refusal that
    printed the value printed the whole ``ArrangementSpec`` -- every solver
    parameter, several hundred characters of it -- where the one thing the
    reader needs is which arrangement it was. So the registry is read backwards
    for a name, and only an unregistered value falls back to a repr (truncated,
    because the reason to truncate is the same reason to look the name up).
    """
    registry = None
    if field == "physics":
        from robosimstudio_bench.taskbuild.arrangements import PHYSICS as registry
    elif field == "cameras":
        from robosimstudio_bench.taskbuild.arrangements import CAMERA_RIGS as registry
    if registry is not None and value is not None:
        for name in registry.names():
            if registry.get(name) is value or registry.get(name) == value:
                return repr(name)
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


_UNREAD = object()
"""Sentinel for a document nobody has asked for yet, since ``None`` is an answer."""


@dataclass
class FleetSnapshot:
    """Every world's state, from one batched read.

    Attributes:
        step: The wave's control step this was taken at.
        sim_time: Simulated seconds since the wave reset.
        body_q: Every world's body transforms, ``[worlds * bodies, 7]``.
        body_qd: Every world's body velocities.
        joint_q: Every world's joint coordinates.
        joint_target_q: Every world's commanded targets.
        diverged: Per world, whether it has ever produced a non-finite state.
        worlds: The stride map its views are taken with.
    """

    step: int
    sim_time: float
    body_q: np.ndarray
    body_qd: np.ndarray
    joint_q: np.ndarray
    joint_target_q: np.ndarray
    diverged: np.ndarray
    worlds: WorldMap

    def body(self, world: int) -> np.ndarray:
        """World ``world``'s body transforms, indexed like the template's."""
        return self.worlds.body_view(self.body_q, world)

    def body_vel(self, world: int) -> np.ndarray:
        return self.worlds.body_view(self.body_qd, world)

    def coords(self, world: int) -> np.ndarray:
        return self.worlds.coord_view(self.joint_q, world)

    def targets(self, world: int) -> np.ndarray:
        return self.worlds.coord_view(self.joint_target_q, world)


class FleetEnv:
    """A replicated scene, stepped as one, scored per world.

    Args:
        task: The variant to run. A registered ``"<suite>/<variant>"`` name or a
            spec.
        num_worlds: How many worlds to replicate.
        scene: Override the task's place, or ``""`` for its own.
        embodiment: Override the task's robot.
        gripper: Override every arm's hand.
        device: Warp device.
        max_steps: Overrides the task's own step budget when non-zero.
        eval_every: Control steps between snapshots. 1 scores every step, which
            is what a run compared against a serial baseline wants; higher trades
            up to that many steps of latency in noticing an episode ended for
            proportionally less host work.
        use_cuda_graph: Overrides the physics spec's own setting.
        verbose: Print what was built.
    """

    def __init__(
        self,
        task: str | TaskSpec,
        *,
        num_worlds: int,
        scene: str = "",
        embodiment: str = "",
        gripper: str = "",
        device: str | None = None,
        max_steps: int = 0,
        eval_every: int = 1,
        use_cuda_graph: bool | None = None,
        renderer: str = "",
        renderer_kwargs: dict[str, Any] | None = None,
        verbose: bool = False,
        episodes: Sequence[Episode] | None = None,
    ) -> None:
        from robosimstudio_bench.taskbuild.registry import get_task

        spec = get_task(task) if isinstance(task, str) else task
        if scene or embodiment or gripper:
            from robosimstudio.specs import replace
            from robosimstudio_bench.taskbuild.matrix import declared_key

            # The declared spelling of the pairing, as Env takes it.
            scene, embodiment, gripper = declared_key(spec, scene, embodiment, gripper) or (scene, embodiment, gripper)
            # Validate before rewriting, as Env does: after the rewrite a task
            # without bindings would admit whatever pairing it was handed.
            spec.binding_for(scene, embodiment, gripper)
            spec = replace(
                spec,
                scene=scene or spec.scene,
                embodiment=embodiment or spec.embodiment,
                gripper=gripper or spec.gripper,
            )
        self.num_worlds = int(num_worlds)
        self.device = device
        self.eval_every = max(1, int(eval_every))
        self.verbose = bool(verbose)

        self.episode = make_episode(spec, 0)
        # The wave this model is being built FOR, when the caller knows it. A
        # task whose sampler draws something built into the model needs every
        # world placed before anything is built on top of the model -- see
        # ``_place_worlds`` in the compiler for why a reset is too late for a
        # foreign engine's actors. ``None`` builds every world at episode 0's
        # layout, which is what a replicated scene has always been and is right
        # for every task that draws nothing.
        self._built_for: tuple[Episode, ...] = tuple(episodes or ())
        # The model a wave builds is the document's FLEET BASE when it names
        # one: a superset holding every object any instance of it runs, with
        # this document's episodes as state and what they do not name parked.
        # The document stays ``task`` -- its evaluator, graphs, tools and name
        # are what a run is about -- and the base is ``build_task``.
        from robosimstudio_bench.run.fleet import fleet_build_task

        build = fleet_build_task(spec)
        self.scene = compile_scene(
            build,
            self.episode,
            num_worlds=self.num_worlds,
            device=device,
            use_cuda_graph=use_cuda_graph,
            verbose=verbose,
            episodes=list(episodes) if episodes else None,
        )
        # The entity objects a goal asks for published states, which is what
        # ``entities`` is for -- ``_published`` calls ``published_states`` on
        # them rather than indexing them. Only the world-aware ones: an entity
        # that reads row 0 whatever world it is asked about would answer every
        # world with world 0's verdict, and ``scoring_refusal`` below is what
        # keeps one out of a wave in the first place.
        # The deformables a goal can be handed a probe for. Only the ones that
        # opened a probe per world: an entity holding one probe would answer
        # every world with the first world's particles, and ``scoring_refusal``
        # below is what keeps one out of a wave.
        self._particled = {
            name: entity
            for name, entity in self.scene.by_name.items()
            if getattr(entity, "carries_particles", False) and callable(getattr(entity, "probe_for", None))
        }
        self._publishers = {
            name: entity for name, entity in self.scene.by_name.items() if getattr(type(entity), "world_aware", False)
        }
        refusal = scoring_refusal(self.scene)
        if refusal:
            raise CannotReplicate(
                f"{self.scene.request.full_name} cannot run replicated ({self.num_worlds} worlds) yet: {refusal}"
            )
        self.build_task = self.scene.request
        """What the model was compiled from: the fleet base when the document
        declares one, else the document itself."""
        self.task = self.build_task if build is spec else spec
        """The document being run -- what the evaluator, the graphs and the
        rows are about. The same object as :attr:`build_task` for every
        document that names no fleet base."""
        self.pipeline = self.scene.pipeline
        self.model = self.pipeline.model
        self.handles = self.scene.handles
        # One world compiles through the single-world path, which leaves
        # ``CompiledScene.worlds`` unset -- so a fleet of one used to die here
        # on the first stride read (``'NoneType' object has no attribute
        # 'coords_per_world'``). Derived rather than pushed into the compiler
        # because a *serial* scene's ``worlds`` being ``None`` is read as "not
        # replicated" in several places, and this is the batched path's problem
        # alone. Every stride is then the identity, which is what world 0 of one
        # world means -- and it is what lets the superset proving test run its
        # middle rung, superset-W1 between serial and superset-W12.
        self.worlds: WorldMap = self.scene.worlds or WorldMap.of(self.model, self.model, 1)
        self.max_steps = int(max_steps or self.task.max_steps or 0)
        self.control_dt = 1.0 / float(self.pipeline.control_hz)
        self.cameras: tuple = ()
        """The rig, resolved against the **template** model.

        Template indices, like every other index a wave hands out, so a
        ``parent_body`` here means the same thing it means in a handle and
        :meth:`render_worlds` offsets it per world. Empty when the job asked
        for no renderer -- which anything deriving a config from an environment
        still reads, and an attribute that exists and is empty is the difference
        between "no cameras" and a connector that cannot be built at all.
        """
        self.renderer = None
        """The rig's backend, or ``None``. Owned and closed here."""

        # Each object's own declared pose, captured before anything resets it, so
        # every wave places every world from the same base rather than from
        # whatever the previous wave left behind.
        # ``declared_q`` is ``EntityRigid``'s, not the ``Entity`` contract's, and
        # the objects view is "offers a handle" rather than "is an ObjectSpec"
        # since the bins were derived -- so a kind that publishes a handle and
        # declares no rest pose belongs here and has nothing to restore.
        self._declared = {
            name: entity.declared_q.copy()
            for name, entity in self.scene.objects.items()
            if getattr(entity, "declared_q", None) is not None
        }
        self._home_q = (
            np.asarray(self.scene.embodiment.home_joint_q(), dtype=np.float32)
            if self.scene.embodiment is not None
            else np.asarray(self.scene.index_model.joint_q.numpy(), dtype=np.float32)
        )
        self._home_targets = (
            np.asarray(self.scene.embodiment.home_target_q(), dtype=np.float32)
            if self.scene.embodiment is not None
            else np.zeros(self.worlds.coords_per_world, dtype=np.float32)
        )
        self._start_q = np.asarray(self.scene.index_model.joint_q.numpy(), dtype=np.float32).copy()
        """The articulation every world starts from: the vector the model was BUILT
        with until the warm-up has run, then the vector the warm-up settled to
        (see :meth:`_settled_start`). Not the declared home: the single-world
        runner never writes a robot's home coordinates (its spec restores
        targets only), so its episodes start from the built articulation --
        on yam_tool_hanging the fingers at 0.0 with their target at 0.0475 --
        and a wave that laid down the home posture started every world with
        the hand 32 mm more open than the serial run's."""

        self.contacts = self._make_contact_probe()
        self.diverged = wp.zeros(self.num_worlds, dtype=wp.int32, device=self.model.device)
        self._settled: list[np.ndarray | None] = [None] * self.num_worlds
        self.step_count = 0
        # The wave's last picture, keyed on the step it was taken at: render_worlds()
        # hands the same frames to every world that asks within one control step.
        self._frames: list | None = None
        self._frames_at = -1
        self._host_joint_q: tuple[int, np.ndarray] | None = None
        self._doc: Any = _UNREAD
        self.episodes: list[Episode | None] = [None] * self.num_worlds
        self.documents: list[TaskSpec | None] = [None] * self.num_worlds
        """Per world, the DOCUMENT it runs, or ``None`` for :attr:`task`.

        A wave of one document leaves this empty and reads :attr:`task`, which
        is every run written before mixed-document waves existed. A wave of
        several instances of one superset fills it: the model is the base's,
        and each world's active set, regions, budget and evaluator are its own
        document's. See :meth:`reset_wave`."""
        self._declared_docs: dict[str, dict[str, np.ndarray]] = {}
        """Declared object poses by document name; see :meth:`_declared_for`."""
        self._doc_regions: dict[str, Any] = {}
        """Region tables by document name, resolved once each. A document's
        regions are geometry on the base's frames, so they are built from the
        base's handles and cached -- a wave of twelve instances has at most
        twelve tables (:meth:`handles_for`)."""
        self._park_poses = scene_park_poses(self.scene)
        """Where each object goes when a world parks it; see :func:`~robosimstudio.specs.park_poses`."""
        self._parking: GravcompParking | None = None
        """The per-world gravity-compensation table, built on the first park."""
        self._parked: list[tuple[str, ...]] = [()] * self.num_worlds
        """Per world, the objects it parks, in build order."""
        self._restricted: dict[frozenset[str] | None, Any] = {}
        """Restricted handle tables, one per distinct active set in the wave."""
        self._warmed = False
        self._hook_floor: int | None = None
        """How many solve-stage hooks the scene itself installed; everything past
        it belongs to a policy and is dropped at the next wave reset."""
        self._snapshot_lock = threading.RLock()
        """Serialises :meth:`snapshot` and :meth:`context` across the fleet thread
        and the worlds' graph threads -- see :meth:`snapshot`."""
        self._control_graph = None
        self._control_key: tuple[int, int] | None = None

        # ---- batched control -------------------------------------------------
        self.ik_plan = plan_ik_model(self.build_task, self.episode, self.scene)
        self.ik = build_ik_rig(self.scene, self.ik_plan, num_worlds=self.num_worlds)
        self.segments: dict[str, SegmentDriver] = {}
        self.grippers: dict[str, GripperBank] = {}
        if self.scene.embodiment is not None:
            for name, index in self.scene.embodiment.arms.items():
                self.segments[name] = SegmentDriver(
                    num_worlds=self.num_worlds,
                    scene_coords=index.arm_coords,
                    coords_per_world=self.worlds.coords_per_world,
                    control_dt=self.control_dt,
                    device=self.model.device,
                )
                hand = self.scene.embodiment.grippers.get(name)
                if hand is not None:
                    self.grippers[name] = GripperBank.like(
                        hand, num_worlds=self.num_worlds, coords_per_world=self.worlds.coords_per_world
                    )
        # Last, because the rig is built over everything above it -- the model
        # it photographs, and the handles a mounted camera resolves its arm
        # through.
        if renderer:
            self._open_renderer(renderer, dict(renderer_kwargs or {}))

        if verbose:
            print(
                f"[fleet] {self.task.full_name}: {self.num_worlds} worlds, "
                f"ik tier {self.ik_plan.tier}, arms {list(self.segments)}"
                + (f", rig {[c.name for c in self.cameras]} on {renderer}" if self.cameras else "")
            )

    # ------------------------------------------------------------------ lifecycle
    # --------------------------------------------------------------- the rig
    def _open_renderer(self, backend: str, kwargs: dict[str, Any]) -> None:
        """Build the observation rig over the whole fleet, or say why not.

        The renderer is constructed on the **replicated** model -- it is the one
        being photographed -- while the cameras carry **template** parent
        indices, which :meth:`render_worlds` offsets per world. Those are the
        same two models every other part of a wave keeps apart, and for the same
        reason: names resolve in one, state lives in the other.
        """
        from robosimstudio.solvers.render import backend_class, create_renderer, supported_channels

        self.cameras = self._resolve_cameras(supported_channels(backend))
        if not self.cameras:
            return
        cls = backend_class(backend)
        if not hasattr(cls, "render_worlds"):
            # Refusing rather than rendering world 0 W times. A backend with no
            # batched form still *runs* -- it would return one world's picture
            # for every world -- and a run that produced sixteen identical
            # frames labelled with sixteen episode ids is the kind of number
            # nothing downstream can tell apart from a real one.
            raise CannotReplicate(
                f"--renderer {backend!r} has no batched form: a wave renders every camera of every world in "
                f"one launch. Run it at --num-worlds 1, or choose a backend that does."
            )
        render = kwargs.pop("render", None)
        if render is None:
            # The scene's own stated lighting, else the house look -- exactly
            # what ``Sim.make_renderer`` resolves for a serial rig. A wave used
            # to hand the backend ``None`` here, which the backends read as
            # "your own defaults": measured on tool_hanging/wrenches_level1 at
            # two worlds, the path tracer's frame came out with the bench and
            # pegboard a stop darker than the serial frame the hook prompt was
            # calibrated on, and the hook detector found nothing in either world.
            from robosimstudio.core.render.look import look_for

            render = look_for(getattr(self.scene.request, "render", None))
        for key, provide in self._scene_dressing().items():
            if key in getattr(cls, "wants", ()) and key not in kwargs:
                value = provide()
                if value is not None:
                    kwargs[key] = value
        self.renderer = create_renderer(backend, self.model, self.cameras, render=render, **kwargs)

    def _scene_dressing(self) -> dict[str, Any]:
        """What a serial rig is dressed with, restated for a replicated model.

        ``Sim._make_renderer`` hands a backend the facts it declares in
        ``wants`` -- the finishes that carry every material and texture, the
        shapes a running scene recolours, the work surface, the cell -- and a
        wave used to hand it none of them. The geometry was right and nothing
        had a material: measured on ``tool_sort/level1_1`` at two worlds, the
        destination bin's four printed compartment labels came out blank, and
        ``gap_perception_loop``'s first node is the one that reads those
        labels. Both worlds failed it with "the surfaces appear to be blank".

        **Two of the tables are keyed by SHAPE INDEX**, and a replicated model
        holds ``W`` copies of every shape. A table built for the template
        dresses world 0 and leaves the rest bare, so those are restated at each
        world's stride -- which is what the segmentation labels already did, and
        the reason that one was right when these were missing.
        """
        handles = self.scene.handles
        worlds = range(self.num_worlds)

        def finishes() -> Any:
            table = handles.finishes
            return {self.worlds.shape(int(i), w): f for i, f in table.items() for w in worlds} or None

        def live_shapes() -> Any:
            base = np.asarray(handles.live_shapes, dtype=np.int32)
            if not len(base):
                return None
            return np.asarray([self.worlds.shape(int(i), w) for w in worlds for i in base], dtype=np.int32)

        def shape_labels() -> Any:
            labels = handles.segmentation_labels()
            return {self.worlds.shape(int(i), w): int(v) for i, v in labels.items() for w in worlds} or None

        def cell() -> Any:
            name = handles.scene_name
            if not name:
                return None
            from robosimstudio.scene.registry import get_cell

            try:
                return get_cell(name)
            except KeyError:
                return None

        return {
            "finishes": finishes,
            "live_shapes": live_shapes,
            "shape_labels": shape_labels,
            "surface_z": lambda: handles.surface_z,
            "cell": cell,
            "deformables": lambda: handles.deformables,
            "deformable_finishes": lambda: handles.deformable_finishes,
        }

    def _resolve_cameras(self, channels: tuple[str, ...]) -> tuple:
        """The rig with every mounted camera bound to a **template** body.

        The same resolution :class:`~robosimstudio.sim.sim.Sim` does, against
        the template rather than the model being stepped -- a label resolved
        against a replicated model finds one copy of a name that is there
        ``W`` times, and which copy is an accident of the matcher.

        Narrowed to what the backend can photograph, for the reason ``Sim``
        narrows: a rig is built from a *name* here, on the way to an
        environment, and one camera wanting one channel a backend lacks would
        otherwise cost the whole rig.
        """
        from robosimstudio.scene.entity import label_index
        from robosimstudio.specs import replace

        wanted = set(channels)
        resolved = []
        for spec in getattr(self.scene.request, "cameras", ()):
            keep = tuple(c for c in spec.channels if c in wanted)
            if not keep:
                continue
            narrowed = replace(spec, channels=keep)
            if not narrowed.parent_frame:
                resolved.append(narrowed)
                continue
            try:
                body = int(self.scene.handles.arm(narrowed.parent_frame).ee_body)
            except Exception:
                # An arm role is what an eye-in-hand camera means; a body label
                # is what a camera screwed to one link means. Both exist on real
                # rigs, so failing the first is a reason to try the second.
                body = label_index(list(self.scene.index_model.body_label), narrowed.parent_frame)
            resolved.append(replace(narrowed, parent_body=body, parent_frame=""))
        return tuple(resolved)

    def render_worlds(self, worlds: Sequence[int] | None = None) -> list[dict]:
        """The frames of the worlds that asked. ``[num_worlds]`` of FrameSet.

        Args:
            worlds: Which worlds want a picture, or ``None`` for all. What that
                saves depends on the backend and the backend says so: the tiled
                tracer renders every world in one megakernel either way and only
                the host copy narrows, while the path tracer is a pass per world
                and skips the ones nobody asked about.

        A world that did not ask comes back with an empty frame set rather than a
        missing entry, so the result is always indexable by world.

        Empty throughout when no rig was opened, which is the ordinary case: a
        privileged graph reads ground truth and never asks for a picture.

        Cached on the step count: a wave of W graphs asking for a picture in the
        same step costs one launch, not W. Fleet thread only -- it moves the
        device's current stream, which is why a world's thread reaches it
        through its bus rather than calling it. The cache is keyed on the step
        alone and not on *which* worlds asked, so it is only reused for a
        request that wanted them all.
        """
        if self.renderer is None:
            return [{} for _ in range(self.num_worlds)]
        if worlds is None:
            if self._frames is None or self._frames_at != self.step_count:
                with reading():
                    self._frames = self.renderer.render_worlds(self.pipeline.state_0, None)
                self._frames_at = self.step_count
            return self._frames
        with reading():
            return self.renderer.render_worlds(self.pipeline.state_0, worlds)

    @property
    def places_per_world(self) -> bool:
        """Whether this scene puts part of a world's layout in at BUILD time.

        True when any entity offers ``place`` -- the compiler's per-world
        placement phase, which is what a task whose sampler moves something the
        model is *converted* from has to use. Such a scene is built for one
        wave's benches and is not reusable across waves with different ones,
        which is the whole of why a runner asks.
        """
        return any(getattr(type(e), "world_aware", False) and hasattr(e, "place") for e in self.scene.by_name.values())

    def built_for(self, episodes: Sequence[Episode | None]) -> bool:
        """Whether this model was built for exactly these benches.

        Compared on the sampled overrides rather than on episode ids: two ids
        can draw the same bench (a ``fixed`` sampler draws one for all of them),
        and rebuilding for a bench already standing costs seconds and buys
        nothing.
        """
        if not self.places_per_world:
            return True
        mine = [getattr(e, "overrides", None) for e in self._built_for]
        theirs = [getattr(e, "overrides", None) for e in episodes if e is not None]
        return mine[: len(theirs)] == theirs

    def reset_wave(
        self,
        episodes: Sequence[Episode | None],
        documents: Sequence[TaskSpec | None] | None = None,
    ) -> None:
        """Put every world at its episode's start pose, in one write.

        A ``None`` episode parks that world: it holds the home posture and is
        never scored. That is the tail of a wave that did not divide evenly, and
        parking is cheaper than the alternative of shrinking the model.

        Args:
            episodes: One per world, in world order.
            documents: One per world, for a **mixed-document** wave -- several
                instances of one superset, each scored on its own goal over its
                own objects. ``None`` (and a ``None`` entry) means
                :attr:`task`, which is what every single-document wave passes
                and what the runner passed before this existed. Every document
                must key to the same model
                (:func:`~robosimstudio_bench.run.fleet.fleet_model_key`); the
                runner's planner guarantees that and :meth:`_check_documents`
                refuses it here rather than letting a world run the wrong
                model's episode.

        Warm-up runs before the first wave takes effect, exactly as
        :meth:`~robosimstudio.core.env.EnvBase.reset` orders it: capture steps
        real physics, so the wave has to be laid down again afterwards or it
        starts from wherever the warm-up left it.
        """
        if len(episodes) != self.num_worlds:
            raise ValueError(f"expected {self.num_worlds} episodes, got {len(episodes)}")
        if documents is not None and len(documents) != self.num_worlds:
            raise ValueError(f"expected {self.num_worlds} documents, got {len(documents)}")
        self.episodes = list(episodes)
        self.documents = list(documents) if documents is not None else [None] * self.num_worlds
        self._check_documents()
        # A document's active set decides what a world parks, so a table
        # restricted under the last wave's documents is not this wave's.
        self._restricted.clear()
        # Before any state is composed, because :meth:`_write_wave` is what runs
        # the forward kinematics that carries a moved fixture's children with it.
        self._place_documents()

        if not self._warmed:
            self._write_wave()
            self.pipeline.warmup_and_capture()
            self._warmed = True
            self._start_q = self._settled_start()

        self._write_wave()
        # Drop every hook a policy installed during the last wave, exactly as
        # ``EnvBase._retire_episode_hooks`` does between episodes: a pin a
        # graph's ``sim.attach`` put into the solve stage would otherwise keep
        # holding into the next wave. The floor is the first reset's count, so
        # the hooks the scene itself installed while being built stay.
        hooks = getattr(getattr(self.pipeline, "hooks", None), "post_solve", None)
        if hooks is not None:
            if self._hook_floor is None:
                self._hook_floor = len(hooks)
            else:
                del hooks[self._hook_floor :]
        # Before the entities, because this is what they place things *from*: a
        # host engine's own reset returns its island to the rest it captured, and
        # an entity's reset then moves this world's copy to where this episode
        # drew it. Never called here at all until surface_prep's foam started
        # being drawn -- which was survivable only while a wave was the whole run
        # and nothing carried into a second one. It does now: a cell of 42
        # episodes at twelve worlds is four waves, and without this the sponge in
        # wave two starts wherever wave one crushed it.
        self.pipeline.reset_solvers(flags=0)
        self._reset_entities()
        mark_dirty = getattr(self.pipeline, "mark_dirty", None)
        if callable(mark_dirty):
            # An entity's reset may have moved what a host engine mirrors; a
            # rig driven from the last host snapshot must not drive the first
            # tick from the pre-reset one.
            mark_dirty()
        self._settled = [None] * self.num_worlds
        self.pipeline.sim_time = 0.0
        self.pipeline.step_count = 0
        self.step_count = 0
        self._frames = None  # a new wave is a new picture
        self._host_joint_q = None
        self.diverged.zero_()
        for driver in self.segments.values():
            driver.clear()
        for _name, bank in self.grippers.items():
            # The hand's HOME aperture, as the single-world reset leaves it --
            # ``home_target_q`` carries the arm's stated starting aperture (or
            # full open when it states none). A command of zero is full open
            # whatever the arm declared: on yam_tool_hanging the serial reset
            # leaves the jaws at their stated home while a wave opened them
            # wide, and the wrist camera's first frames -- the ones the held
            # wrench is registered against -- showed different jaws.
            bank.set_all(np.full(self.num_worlds, self._home_fraction(bank), dtype=np.float32))
            bank.flush()
        if self.ik is not None:
            for arm in self.ik.arms.values():
                arm.set_enabled(np.zeros(self.num_worlds, dtype=np.int32))
            self.ik.flush()
        # Every world's start pose, taken now from the post-reset state rather
        # than whenever a world first asks: a policy whose first query comes
        # after its first motion would otherwise measure "has it moved" from
        # wherever the motion left things, not from where the evaluator does.
        snapshot = self.snapshot()
        self._settled = [snapshot.body(world).copy() for world in range(self.num_worlds)]

    def _home_fraction(self, bank) -> float:
        """Where the home posture puts this hand, as the bank's 0 (open) .. 1 (closed) command."""
        with reading():
            coords = np.asarray(bank.coords.numpy(), dtype=np.int64)
            open_q = np.asarray(bank.open_q.numpy(), dtype=np.float64)
            closed_q = np.asarray(bank.closed_q.numpy(), dtype=np.float64)
        home = np.asarray(self._home_q, dtype=np.float64)[coords]
        span = closed_q - open_q
        live = np.abs(span) > 1.0e-9
        if not live.any():
            return 0.0
        fraction = float(np.mean((home[live] - open_q[live]) / span[live]))
        return float(np.clip(fraction, 0.0, 1.0))

    def _reset_entities(self) -> None:
        """Run the ``reset`` phase once per world, for the entities that can take it.

        :meth:`_write_wave` composes a start state out of the home posture and
        each object's episode pose, which is the whole of what a rigid scene
        needs -- and for a long time was the whole of what a wave did. It is not
        the whole of what a *scene* does. An entity's own ``reset`` is where a
        rod straightens and re-asserts its anchor, where a marked surface gets
        its decals back, where a board clears the swept angles it scores from.
        Skipping it leaves those carrying the previous episode into the next one.

        **Only the entities that declare** :attr:`~robosimstudio.scene.entity.Entity.world_aware`.
        A reset written for one world, called once per world, writes world 0's
        state ``W`` times -- which is not a partial fix but a new way to be
        wrong, and quietly: every world would start from the same restored
        state and only the first would be the right one. So an unclaimed entity
        is skipped exactly as it was before, and the compiler's replication
        refusal is what stops that mattering for anything with substep kernels.

        A parked world is not reset. It is running no episode, so there is no
        start state for it to be put into.
        """
        entities = [e for e in self.scene.by_name.values() if getattr(type(e), "world_aware", False)]
        if not entities:
            return
        held = getattr(self, "_parked", None) or ()
        for world, episode in enumerate(self.episodes):
            if episode is None:
                continue
            parked = set(held[world]) if world < len(held) else set()
            for entity in entities:
                # A parked entity is not reset: its reset would write the
                # episode pose its layout drew for it, over the park.
                if getattr(entity, "name", None) in parked:
                    continue
                entity.reset(self.pipeline, episode, world=world)

    def _settled_start(self) -> np.ndarray:
        """The articulation the warm-up settled world 0 to, as one world's coordinate row.

        What the single-world runner starts every episode from: ``Sim.reset``
        captures ``_start_joint_q`` after the warm-up has stepped real physics
        from the built articulation, so a robot whose spec restores only its
        servo *targets* begins each episode where two warm-up iterations left
        it -- on yam_tool_hanging the fingers at 0.0071 / -0.0068 m on their
        way from the built 0.0 to the 0.0475 target -- and not at the declared
        home. A wave that laid the declared home down before and after its own
        warm-up started every world 34 mm from where the serial run starts,
        and the gap took three hundred control steps to decay. The objects'
        coordinates in this row are overwritten per episode by
        :meth:`_write_wave`, exactly as the serial entities re-place theirs.
        """
        snapshot = self.snapshot()
        return np.asarray(snapshot.coords(0), dtype=np.float32).copy()

    def _write_wave(self) -> None:
        """Compose every world's start state on the host and assign it once.

        Every object of every world is written here: its episode pose when the
        world's episode runs it, its **park pose** when it does not
        (:attr:`~robosimstudio.specs.Episode.active`; see
        :mod:`robosimstudio.scene.parking`). Then the parking table is
        applied, so a body parked in this wave is held in this wave and a body
        parked in the last one and run in this one is released -- the table is
        rewritten whole from the current episodes, never patched.
        """
        rows = []
        parked: list[tuple[str, ...]] = []
        for world, episode in enumerate(self.episodes):
            row = self._start_q.copy()
            declared = self._declared_for(self.world_document(world))
            names = parked_objects(self.scene, self._active_for(episode, world))
            out = set(names)
            for name, entity in self.scene.objects.items():
                # ``getattr``, because not every object is a free rigid body
                # with a floating start pose to write. A rod is a chain of
                # capsules on joints and a cloth is particles; neither declares
                # a coordinate start, and both are placed by their own ``reset``
                # rather than by a pose written here. Reading the attribute
                # outright made a wave refuse every scene holding one, with an
                # ``AttributeError`` that named the entity and not the reason.
                if int(getattr(entity, "coord_start", -1)) < 0:
                    continue
                if name in out:
                    pose = self._park_poses[name]
                else:
                    pose = entity.episode_start_q(episode, base=declared[name])
                start = int(entity.coord_start)
                row[start : start + 7] = np.asarray(pose[:7], dtype=np.float32)
            rows.append(row)
            parked.append(tuple(names))
        joint_q = self.worlds.stack_coords(rows)

        state_0, state_1 = self.pipeline.state_0, self.pipeline.state_1
        state_0.joint_q.assign(joint_q)
        state_1.joint_q.assign(joint_q)
        state_0.joint_qd.zero_()
        state_1.joint_qd.zero_()
        if self.model.body_count:
            newton.eval_fk(self.model, state_0.joint_q, state_0.joint_qd, state_0)
            wp.copy(state_1.body_q, state_0.body_q)
            # The back buffer's velocities too. ``Sim.restore`` says why: after
            # everything else matched, ``state_1.body_qd`` was the one half of
            # the double buffer still holding the previous episode's motion.
            wp.copy(state_1.body_qd, state_0.body_qd)
        targets = self.pipeline.control.joint_target_q
        if targets is not None:
            targets.assign(self.worlds.tile_coords(self._home_targets))
        self._parked = parked
        self._apply_parking()

    # ---------------------------------------------------------------- documents
    def _declared_for(self, document: TaskSpec) -> dict[str, np.ndarray]:
        """Each object's declared pose as **this document** states it.

        A document's start layout is a *staging* clause -- ``(:start-poses
        poses/tray_t01.json)`` -- baked into its object specs at lowering,
        not an episode override. So a wave that measured every world from the
        built scene's poses would place every world at the base document's
        layout and apply only the episode overrides on top. On
        ``sharps_disposal`` that is not a subtle error: ``tray_pair_both_t00``
        declares ``syringe_0`` at (0.073, -0.342) and ``t01`` at (0.002,
        -0.373), 72 mm apart, and both worlds would have run t00's.

        The base's table for anything the document does not declare, so a
        document naming a subset of the superset's objects inherits the rest --
        which is what a parked object wants anyway.
        """
        if document is self.build_task:
            return self._declared
        key = document.full_name
        table = self._declared_docs.get(key)
        if table is None:
            table = dict(self._declared)
            for spec in getattr(document, "objects", ()):
                if spec.name in table and spec.pose is not None:
                    table[spec.name] = np.concatenate(
                        [np.asarray(spec.pose, dtype=np.float64), np.asarray(spec.quat, dtype=np.float64)]
                    )
            self._declared_docs[key] = table
        return table

    def _place_documents(self) -> None:
        """Stand every world's per-world-placeable FIXTURE where its own document puts it.

        The fixture half of :meth:`_declared_for`, and it exists for the same
        reason: a start layout is a *staging* clause baked into the document's
        specs at lowering, not an episode override, so a wave that placed
        fixtures from the built scene would stand every world's at the base
        document's. Objects take that correction through ``joint_q``
        (:meth:`_write_wave`); a welded fixture has no coordinate to write and
        takes it through the mocap channel instead
        (:class:`~robosimstudio.domains.static.welded.WeldedFixture`).

        Duck-typed on ``place_from_document(pipeline, spec, *, world, document)``
        rather than on a domain: the entity that needs it is gear_mesh's board,
        which is an ``extras`` entity and not a static, and what the hook means is
        "your own spec, as the document this world runs states it". An entity
        without the method is left alone, which is every other entity in the
        corpus -- a static's per-world pose is an episode override and
        :meth:`_reset_entities` already carries it.

        Whatever the entity refuses, it refuses by name: a document whose fixture
        is a different SHAPE rather than a different place is a different model,
        and running it would score a world against geometry it never had.
        """
        for world, episode in enumerate(self.episodes):
            if episode is None:
                continue
            document = self.world_document(world)
            if document is self.build_task:
                continue
            specs = {getattr(spec, "name", ""): spec for spec in document.entities}
            for name, entity in self.scene.by_name.items():
                place = getattr(entity, "place_from_document", None)
                if place is None or name not in specs:
                    continue
                place(self.pipeline, specs[name], world=world, document=document.full_name)

    def world_document(self, world: int) -> TaskSpec:
        """The document world ``world`` runs: its own, or :attr:`task`.

        Never ``None``, so every caller can ask without branching. A parked
        world answers :attr:`task` like any other; it is never scored.
        """
        document = self.documents[int(world)]
        return self.task if document is None else document

    def _check_documents(self) -> None:
        """Refuse a wave whose documents do not all key to the model that was built.

        A wave is ``W`` copies of one model, so a document whose pairing,
        physics, control mode or rig differs is a different model and not a
        different episode. The planner groups on
        :func:`~robosimstudio_bench.run.fleet.fleet_model_key` and would never
        offer such a wave; this is the check that says so at the point the
        state is written, because the failure it prevents is silent -- a world
        stepping the wrong model's physics and being scored as though it were
        its own.

        Raises:
            CannotReplicate: Naming the document and what it disagrees about.
        """
        from robosimstudio_bench.run.fleet import fleet_build_task

        base = self.build_task.full_name
        running = [(w, d) for w, d in enumerate(self.documents) if d is not None]
        if not running:
            return
        # Compared against the first DOCUMENT of the wave, not against
        # :attr:`task`. They are the same thing when the runner opened the fleet
        # from a document of the wave, which it does -- but ``task`` is
        # "whichever document this env was built from", so a check written
        # against it means something different depending on how the env was
        # opened. Built from the *base* of a superset it refused every instance,
        # on a buffer difference that is the base's whole reason to exist.
        _, reference = running[0]
        for world, document in running:
            built = fleet_build_task(document)
            if built.full_name != base:
                raise CannotReplicate(
                    f"world {world} runs {document.full_name!r}, whose fleet base is "
                    f"{built.full_name!r}; this wave's model is {base!r}. A wave holds one model, "
                    "so two documents share a wave only when they share a base"
                )
            if document is reference:
                continue
            for field in ("scene", "embodiment", "gripper", "physics", "control_mode", "cameras"):
                mine, theirs = getattr(reference, field, None), getattr(document, field, None)
                if mine != theirs:
                    raise CannotReplicate(
                        f"world {world} runs {document.full_name!r}, which stages "
                        f"{field}={_staged_name(field, theirs)} against world {running[0][0]}'s "
                        f"{_staged_name(field, mine)}. That is a different model, not a different "
                        "episode of one"
                    )
        # And against the model that was actually built. A base may differ from
        # its instances in the constraint BUFFERS -- sizing them for the union is
        # what a superset base is for -- and may not differ in anything that
        # changes the integration, because then an instance would be run under
        # physics it was never measured under.
        theirs, mine = _material_physics(reference.physics), _material_physics(self.build_task.physics)
        if theirs != mine:
            raise CannotReplicate(
                f"this wave's documents stage {theirs} and the model built for them "
                f"({base!r}) is {mine}. A superset base may size the constraint buffers for the "
                "union and may not change the integration"
            )

    # ------------------------------------------------------------------ parking
    def _active_for(self, episode: Episode | None, world: int = 0) -> frozenset[str] | None:
        """The entities one world's episode runs, or ``None`` for every one.

        The episode's own set when it states one. An instance run on its
        FLEET BASE's model that states none runs *its own* entities -- the
        document's, which are a subset of the base's -- because "every
        entity" would mean every object of the superset, and a document that
        never heard of the base's other seventy tools did not ask for them.
        A parked world runs everything: it is never scored, and a wave of
        parked worlds is what the warm-up runs on.

        The document is **this world's** (:meth:`world_document`), not the
        fleet's one ``task``: in a mixed-document wave the whole point is that
        world 0 and world 1 activate different subsets of one model, and
        reading ``self.task`` here gave every world the first document's set.
        """
        if episode is None:
            return None
        active = episode.active
        document = self.world_document(world)
        if active is None and self.build_task is not document:
            active = frozenset(str(getattr(spec, "name", "")) for spec in document.entities)
        return active

    def world_active(self, world: int) -> frozenset[str] | None:
        """What world ``world`` runs, as of now, or ``None`` for everything.

        The episode's set, corrected by what is actually parked: every free
        body the world is not holding out of the way is running, whether the
        episode named it or a policy released it since (:meth:`unpark`), and
        every free body it is holding is not, whichever the episode said. For
        everything without a free joint the episode's statement stands.
        """
        active = self._active_for(self.episodes[int(world)], int(world))
        parked = set(self._parked[int(world)])
        if active is None and not parked:
            return None
        names = set(active) if active is not None else set(self.scene.by_name)
        free = {n for n, e in self.scene.objects.items() if int(getattr(e, "coord_start", -1)) >= 0}
        return frozenset((names | free) - parked)

    def parked(self, world: int) -> tuple[str, ...]:
        """The objects world ``world`` is holding out of the way, in build order."""
        return self._parked[int(world)]

    def handles_for(self, world: int):
        """This world's handles: the template's, restricted to what the world runs.

        :meth:`~robosimstudio.scene.handles.SceneHandles.restricted` over
        :meth:`world_active`, cached per distinct ``(document, set)`` -- a wave
        of twelve instances of one superset has at most twelve tables, and
        usually fewer. The whole table, the same object as :attr:`handles`, for
        a world that parks nothing and runs the document the model was built
        from.

        **Regions are the world's document's, not the base's.** A superset holds
        the base's fixtures and every instance declares its own goal regions
        over them: ``tool_sort/level1_1`` puts ``bin2_multimeter_region`` in one
        quadrant of bin2 and ``level1_13`` puts it in another, under the same
        name. Scoring a wave of both off one table would credit the wrong
        compartment, silently and in exactly the half of the worlds that did
        not build the model. The document's regions are resolved against the
        base's handles -- a region is geometry on a named frame -- and cached by
        document name.
        """
        active = self.world_active(world)
        document = self.world_document(world)
        key = (document.full_name, active)
        table = self._restricted.get(key)
        if table is None:
            table = self.handles.restricted(active)
            regions = self._regions_of(document)
            if regions is not None:
                from dataclasses import replace as _replace

                # Restricted first, then the document's own regions filtered by
                # the same rule ``restricted`` applies: a region on an object
                # this world parked goes with it.
                dropped = {name for name in self.handles.objects if name not in table.objects}
                table = _replace(table, regions={n: r for n, r in regions.items() if r.frame not in dropped})
            self._restricted[key] = table
        return table

    def _regions_of(self, document: TaskSpec) -> dict[str, Any] | None:
        """``document``'s regions over the base's frames, or ``None`` when they are the base's own.

        ``None`` for the document the model was compiled from, so the ordinary
        path hands back the table the compiler already built and costs nothing.
        """
        if document is self.build_task or not getattr(document, "regions", ()):
            return None
        name = document.full_name
        table = self._doc_regions.get(name)
        if table is None:
            from robosimstudio.scene.compiler import resolve_regions

            table = resolve_regions(self.handles, document, self.scene.index_model)
            self._doc_regions[name] = table
        return table

    def _make_parking(self) -> GravcompParking:
        if self._parking is None:
            solver = self.scene.program.mujoco_solver if self.scene.program is not None else None
            if solver is None:
                raise CannotReplicate(
                    f"{self.task.full_name} runs an episode with an active set, but its physics has no MuJoCo "
                    "island to hold a parked body against gravity with; parking needs one"
                )
            self._parking = GravcompParking(solver, self.model, self.worlds)
        return self._parking

    def _apply_parking(self) -> None:
        """Write the whole gravity-compensation table from :attr:`_parked`."""
        if self._parking is None and not any(self._parked):
            return
        parking = self._make_parking()
        parking.clear()
        for world, names in enumerate(self._parked):
            parking.set(world, [int(self.scene.objects[name].body) for name in names])
        parking.apply()

    def _write_object(self, world: int, name: str, pose: np.ndarray) -> None:
        """One object of one world to ``pose`` at rest, both state buffers, kinematics re-run."""
        entity = self.scene.objects[name]
        start = int(getattr(entity, "coord_start", -1))
        if start < 0:
            raise ValueError(f"{name!r} has no free joint; only a free body can be parked or released")
        state_0, state_1 = self.pipeline.state_0, self.pipeline.state_1
        with reading():
            q = np.asarray(state_0.joint_q.numpy(), dtype=np.float32).copy()
            qd = np.asarray(state_0.joint_qd.numpy(), dtype=np.float32).copy()
        c = self.worlds.coord(start, int(world))
        q[c : c + 7] = np.asarray(pose[:7], dtype=np.float32)
        dof_start = int(getattr(entity, "dof_start", -1))
        if dof_start >= 0:
            d = self.worlds.dof(dof_start, int(world))
            qd[d : d + 6] = 0.0
        for state in (state_0, state_1):
            state.joint_q.assign(q)
            state.joint_qd.assign(qd)
        if self.model.body_count:
            newton.eval_fk(self.model, state_0.joint_q, state_0.joint_qd, state_0)
            wp.copy(state_1.body_q, state_0.body_q)
            wp.copy(state_1.body_qd, state_0.body_qd)
        mark_dirty = getattr(self.pipeline, "mark_dirty", None)
        if callable(mark_dirty):
            mark_dirty()

    def park(self, world: int, name: str) -> None:
        """Float one object of one world to its park pose and hold it there.

        The per-body form of what :meth:`_write_wave` does for a whole wave:
        the park pose is written into that world's coordinates at zero
        velocity, and ``body_gravcomp[world, body]`` is set to 1 through the
        solver's ``mjc_body_to_newton`` table (both arrays; see
        :mod:`robosimstudio.scene.parking` for why two). The parked object
        leaves this world's :meth:`handles_for` table, its contact rows and
        its observation. Every other world is untouched.

        Outside a captured step: it reads and writes state on the host.
        """
        world = int(world)
        if name not in self.scene.objects:
            raise KeyError(f"no object {name!r}; objects are {sorted(self.scene.objects)}")
        self._write_object(world, name, self._park_poses[name])
        held = set(self._parked[world]) | {name}
        self._parked[world] = tuple(n for n in self.scene.objects if n in held)
        self._apply_parking()

    def unpark(self, world: int, name: str, pose: np.ndarray | None = None) -> None:
        """Release one parked object into one world at ``pose`` -- its episode pose when omitted.

        Gravity compensation off, the pose written at rest. What a later wave
        does through :meth:`_write_wave` when it activates a body an earlier
        wave parked; offered per body for a policy that reveals objects.
        """
        world = int(world)
        entity = self.scene.objects[name]
        if pose is None:
            base = self._declared_for(self.world_document(world))[name]
            pose = entity.episode_start_q(self.episodes[world], base=base)
        self._write_object(world, name, np.asarray(pose, dtype=np.float64))
        self._parked[world] = tuple(n for n in self._parked[world] if n != name)
        self._apply_parking()

    def gravcomp(self, world: int, name: str) -> float:
        """The solver's gravity-compensation row for one object in one world, read back from the device."""
        return self._make_parking().row(int(world), int(self.scene.objects[name].body))

    def _parked_shapes(self, world: int) -> list[int]:
        """Every template shape of every object world ``world`` parks."""
        names = self._parked[int(world)]
        if not names:
            return []
        objects = self.handles.objects
        return [int(s) for name in names if name in objects for s in objects[name].shapes]

    def _make_contact_probe(self):
        """The fleet's contact probe, or ``None`` when this scene cannot have one.

        Fails soft, exactly as the single-world environment does: a task with no
        contact-based conditions should not be unable to run because its solver
        was built without MuJoCo's own collision pipeline. What must not happen
        is failing *silently* on a task that does need it -- so the reason is
        printed once, here, rather than surfacing later as every grasp check
        answering "not held".
        """
        from robosimstudio.core.probes.contact import FleetContactProbe

        solver = self.scene.program.mujoco_solver if self.scene.program is not None else None
        if solver is None or self.worlds is None:
            return None
        try:
            return FleetContactProbe(solver, self.model, self.pipeline.sim_dt, worlds=self.worlds)
        except Exception as exc:
            print(f"[fleet] no contact probe ({type(exc).__name__}: {exc}); contact conditions will read False")
            return None

    def settled_body_q(self, world: int) -> np.ndarray:
        """This world's start pose, as :meth:`reset_wave` captured it.

        What "has it moved" is measured against, by the policy's tools and the
        evaluator alike. Copied rather than viewed because the next snapshot
        overwrites the buffer underneath; captured here only when asked for
        before any wave was reset.
        """
        index = int(world)
        if self._settled[index] is None:
            self._settled[index] = self.snapshot().body(index).copy()
        return self._settled[index]

    # ---------------------------------------------------------------------- step
    def step(self) -> None:
        """One control period for every world: write targets, then step physics.

        Two graph launches and no synchronize. The control stage is captured
        lazily on the array pair, like every other capture in this stack, and
        falls back to eager launches when capture fails.
        """
        self._control_stage()
        self.pipeline.step()
        self.step_count += 1

    @property
    def doc(self):
        """The document this task was defined by, or ``None``.

        :attr:`~robosimstudio_bench.envs.env.Env.doc` for a wave, and it has to
        exist for the same reason: a raw capture's manifest names each entity's
        *asset type* from the document, and the declared feature geometry hangs
        off that name. Without it every batched tape carries an empty type and
        an empty ``features`` -- which is the failure that manifest's own
        comment records having already happened once, to ``category``.

        Built from an evaluator made for the task rather than for a world,
        because a document is a property of the task; cached because building
        one is not free and nothing about it changes per wave.
        """
        if self._doc is _UNREAD:
            factory = getattr(self.task, "evaluator", None)
            try:
                evaluator = factory(self.task) if factory is not None else None
            except Exception:
                evaluator = None
            self._doc = getattr(evaluator, "doc", None)
        return self._doc

    def host_joint_q(self) -> np.ndarray:
        """This step's joint coordinates on the host, read once for the fleet.

        Every world wants its own slice of them and the transfer is the whole
        array whoever asks, so a read per world is one copy of the same bytes
        and one stream sync per world. A wave of twelve asking once each per
        control step -- which is what every arm holding position does, through
        :meth:`~robosimstudio_bench.envs.world_view.WorldView.idle_action` --
        is twelve of both for one array.

        Keyed on the step count, so what a world reads is what this step
        produced and a stale array cannot outlive the step that made it. The
        one thing that could defeat that is state written without a step, which
        is why :meth:`reset_wave` drops it explicitly rather than trusting the
        counter it also resets.
        """
        if self._host_joint_q is None or self._host_joint_q[0] != self.step_count:
            with reading():
                rows = np.asarray(self.pipeline.state_0.joint_q.numpy(), dtype=np.float32)
            self._host_joint_q = (self.step_count, rows)
        return self._host_joint_q[1]

    def _control_stage(self) -> None:
        targets = self.pipeline.control.joint_target_q
        if targets is None:
            return
        joint_q = self.pipeline.state_0.joint_q
        key = (joint_q.ptr, targets.ptr)
        if key != self._control_key:
            self._control_key = key
            self._control_graph = None
            if self.model.device.is_cuda:
                try:
                    self._control_eager(joint_q, targets)
                    with capturing(self.model.device) as capture:
                        self._control_eager(joint_q, targets)
                    self._control_graph = capture.graph
                except Exception as exc:
                    self._control_graph = None
                    print(f"[fleet] control capture failed ({type(exc).__name__}: {exc}); running eager")
        if self._control_graph is not None:
            wp.capture_launch(self._control_graph)
        else:
            self._control_eager(joint_q, targets)

    def _control_eager(self, joint_q: wp.array, targets: wp.array) -> None:
        """IK, then joint segments, then hands -- the body the graph captures.

        The order is the precedence: a world under Cartesian control has its arm
        coordinates written by the solve, a world running a joint trajectory has
        them overwritten by its segment, and the hand is independent of both.
        Every stage is gated per world, so a world only ever has one of them
        writing it -- the order settles what happens if something asks for two.
        """
        # Guarded, because Newton's Levenberg-Marquardt solver runs each
        # objective on its own CUDA stream and so moves the device's
        # process-global current stream -- and a world's script thread may be
        # solving its own boundary IK at that moment. Free once the stage is
        # captured: a graph launch does not run any of this Python, so the lock
        # is taken only on the eager path and during the capture itself.
        with reading():
            if self.ik is not None:
                self.ik.solve(joint_q, targets)
            for driver in self.segments.values():
                driver.drive(targets)
            for bank in self.grippers.values():
                bank.apply(targets)

    def flush(self) -> None:
        """Push every host-side plan to the arrays the captured stage reads."""
        if self.ik is not None:
            self.ik.flush()
        for driver in self.segments.values():
            driver.flush()
        for bank in self.grippers.values():
            bank.flush()

    # ------------------------------------------------------------------ snapshot
    def snapshot(self) -> FleetSnapshot:
        """One batched read of every world's state.

        The only device synchronize in a wave, and the reason nothing else needs
        one: everything the run decides -- has this world succeeded, diverged,
        finished its motion -- is decided from this.

        Serialised, because it is not only the fleet thread that asks: a
        world's graph thread reaches it through ``WorldView.make_context`` for
        every ``sim.query``, and the contact probe it refreshes keeps one
        step's rows for the whole fleet. Two threads in here at once left one
        of them a view built from the previous step's contacts, cached for the
        step -- a held object reported as not held, intermittently, more often
        the wider the wave.
        """
        with self._snapshot_lock:
            return self._snapshot()

    def _snapshot(self) -> FleetSnapshot:
        state = self.pipeline.state_0
        if self.contacts is not None:
            self.contacts.update_all(self.step_count)
        launch_finite_check(
            state,
            self.diverged,
            coords_per_world=self.worlds.coords_per_world,
            dofs_per_world=self.worlds.dofs_per_world,
            particles_per_world=self.worlds.particles_per_world,
            bodies_per_world=self.worlds.bodies_per_world,
        )
        # Every read below moves the device's process-global current stream, and
        # a world's script thread may be reading at the same moment. Taken once
        # around the whole snapshot rather than per array: it is the same lock
        # either way, and this is one read of one state.
        with reading():
            body_q = (
                np.asarray(state.body_q.numpy(), dtype=np.float64)
                if self.model.body_count
                else np.zeros((0, 7), dtype=np.float64)
            )
            body_qd = (
                np.asarray(state.body_qd.numpy(), dtype=np.float64)
                if self.model.body_count
                else np.zeros((0, 6), dtype=np.float64)
            )
            targets = self.pipeline.control.joint_target_q
            return FleetSnapshot(
                step=self.step_count,
                sim_time=self.step_count * self.control_dt,
                body_q=body_q,
                body_qd=body_qd,
                joint_q=np.asarray(state.joint_q.numpy(), dtype=np.float64),
                joint_target_q=(
                    np.asarray(targets.numpy(), dtype=np.float64)
                    if targets is not None
                    else np.zeros(0, dtype=np.float64)
                ),
                diverged=self.diverged.numpy().astype(bool),
                worlds=self.worlds,
            )

    def context(self, world: int, snapshot: FleetSnapshot, *, settled_body_q: np.ndarray) -> WorldState:
        """One world's scoring inputs, as the single-world path assembles them.

        Every array is a *view* into the snapshot, so this costs no copy and no
        read. The handles it carries are the template's, which index a world's
        view exactly as they index a single-world model's arrays -- that identity
        is what lets task evaluators run here without knowing worlds exist.

        **``particles`` is empty, and ``entities`` holds only the publishers
        that answer per world; both are load-bearing on :func:`scoring_refusal`.**
        The single-world path fills them with the task's own entity objects
        (:meth:`~robosimstudio_bench.envs.env.Env.make_context`); an entity that
        holds one world's indices, handed to world *w*, would read world 0 and
        report it as *w*. So an entity is handed over only when it declares
        ``world_aware`` -- its ``published_states`` reads ``world`` off
        this context and keeps its latch and windows per world (the gear board
        is the first) -- and the refusal declines any scene with a publisher
        that does not, or with particles, which still have no per-world probe.
        Everything else scores from ``body_q`` and contacts, both per world.

        :func:`scoring_refusal` invites lifting it, so: **both dicts have to be
        filled in that change, not just the probe.** ``entities`` is what
        published states are read from (``assetlib.bind._published``), and a
        peel lifted here with ``entities`` still empty would not score world 0 by
        mistake -- it would raise ``BindError`` on every episode, which is the
        better failure but still not the one anybody wants to discover from a
        fleet run.

        Serialised with :meth:`snapshot`, for the reason given there.
        """
        with self._snapshot_lock:
            return self._context(world, snapshot, settled_body_q=settled_body_q)

    def _context(self, world: int, snapshot: FleetSnapshot, *, settled_body_q: np.ndarray) -> WorldState:
        """The unlocked body of :meth:`context`."""
        contacts = self.contacts.view(world) if self.contacts is not None else None
        return WorldState(
            # This world's table: the template's, less what its episode parks,
            # so a goal enumerating objects scores the instance and not the
            # superset. The same object as ``handles`` when nothing is parked.
            handles=self.handles_for(world),
            state=self.pipeline.state_0,
            step=snapshot.step,
            sim_time=snapshot.sim_time,
            body_q=snapshot.body(world),
            body_qd=snapshot.body_vel(world),
            settled_body_q=settled_body_q,
            joint_target_q=snapshot.targets(world),
            # No row of a parked body, as a promise rather than a measurement.
            contacts=restrict_contacts(contacts, self._parked_shapes(world)),
            particles=self._particles(world),
            entities=self._publishers,
            # Which world this is a read of. An entity's ``published_states`` is
            # handed this object and nothing else, so without it a wave scores
            # every world off row 0 -- the same verdict W times, right once.
            world=world,
            active=self.world_active(world),
        )

    def _particles(self, world: int) -> dict[str, Any]:
        """This world's particle probes, by entity name, updated for this step.

        Updated here rather than on a schedule because a probe is only read by
        a goal that asks, and a wave that refreshed every world's probe every
        step would pay for sixteen deformables nobody measured. The update is a
        device read, so it is the expensive half.
        """
        if not self._particled:
            return {}
        state = self.pipeline.state_0
        out: dict[str, Any] = {}
        for name, entity in self._particled.items():
            probe = entity.probe_for(world)
            if probe is None:
                continue
            probe.update(state, self.step_count)
            out[name] = probe
        return out

    def evaluate(self, world: int, evaluator, snapshot: FleetSnapshot, *, settled_body_q: np.ndarray) -> EvalResult:
        """Score one world from a snapshot, with its own evaluator instance."""
        return evaluator(self.context(world, snapshot, settled_body_q=settled_body_q))

    # -------------------------------------------------------------- observation
    def frames(self, world: int) -> dict[str, Any]:
        """One world's frames, by camera name. Fleet thread only; see :meth:`render_worlds`."""
        return self.render_worlds()[int(world)]

    def observe(self, world: int, snapshot: FleetSnapshot) -> dict[str, Any]:
        """One world's observation, from the snapshot rather than the device."""
        body_q = snapshot.body(world)
        obs: dict[str, Any] = {
            "objects": {
                name: np.asarray(handle.pose(body_q), dtype=np.float32)
                for name, handle in self.handles_for(world).objects.items()
            },
            "sim_time": snapshot.sim_time,
        }
        embodiment = self.scene.embodiment
        if embodiment is not None:
            from robosimstudio.solvers.ik.tool_frame import tcp_pose, tool_offsets

            coords = snapshot.coords(world)
            for name, index in embodiment.arms.items():
                obs[f"{name}_joint_q"] = np.asarray(coords[index.arm_coords], dtype=np.float32)
                # The tool-centre pose, as the single-world observation reports
                # it -- not the wrist row, which sits a tool's length away.
                position, quat = tcp_pose(body_q, index.ee_body, *tool_offsets(self, name))
                obs[f"{name}_ee_pose"] = np.concatenate([position, quat]).astype(np.float32)
        return obs

    def close(self) -> None:
        """Release what the fleet opened. Idempotent."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self._control_graph = None
