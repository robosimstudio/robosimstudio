# SPDX-License-Identifier: Apache-2.0
"""Spec plus episode to a steppable, indexed scene -- in one fixed order.

The order in :func:`compile_scene` is not stylistic. Every step is there
because doing it later (or earlier) fails, usually silently:

1. **Device, then ``newton.use_coord_layout_targets = True``.** With free-joint
   objects in the model, ``joint_target_q`` is otherwise sized by DOF count and
   indexes differently from ``joint_q`` -- so every target write lands on the
   wrong coordinate the moment a loose object exists.
2. **Every backend's ``register_custom_attributes`` before any import**, read
   off the staged arrangement. Registered after, the importer has already
   dropped ``gravcomp`` and the arm sags.
3. **Entities build**, each recording its index ranges from the builder's counts.
4. **Actuation and object indexing**, which need the joints to exist.
5. **Re-key custom attributes** after any fixed-joint collapse -- see
   :func:`~robosimstudio.core.builder.rekey_custom_attributes`.
6. **Colour, ground, finalize.**
7. **Bind** every entity to the finalized model, then let each write its
   model-wide materials -- the one window before the solvers take their copies.
8. **Compose the physics**: the islands, the channels, the routing and the
   solver, from what the entities declared and the arrangement stages -- see
   :mod:`robosimstudio.scene.physics`. Then the entities' stages, and the pipeline.
9. **``on_scene_ready``**, which is the first moment physics can run: cloth
   settles here, and a task that must act on the *flat* sheet (creating bonds,
   say) does so in its own override before calling up.
10. **Build the handles.**

Nothing in this file knows what a hammer or a tape strip is. It knows specs,
entities and phases; the entities know the rest.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.core.lifecycle import TaskSchedule, steps_of
from robosimstudio.core.pipeline import ScenePipeline
from robosimstudio.core.render.finish import Finish
from robosimstudio.core.render.palette import FLOOR
from robosimstudio.core.worlds import WorldMap
from robosimstudio.scene.physics.world import REPLICATED_HOSTS
from robosimstudio.specs import IDENTITY, Binding, CellSpec, EmbodimentSpec, Episode, Xform, replace
from robosimstudio.specs.asset import build_stage
from robosimstudio.specs.physics import DOMAINS, get_domain

from .build.imports import clear_texture_tints
from .entity import Entity
from .frames import in_surface, surface_frame
from .handles import ArmHandle, RegionHandle, SceneHandles
from .physics import PhysicsProgram, register_custom_attributes
from .physics import compose as _compose
from .registry import get_cell
from .request import SceneRequest

Composer = Callable[..., PhysicsProgram]
"""Builds the physics for a finalized model. See :func:`compile_scene`."""

__all__ = ["CannotReplicate", "CompiledScene", "compile_scene", "resolve_regions"]


class CannotReplicate(NotImplementedError):
    """This scene may not be built at more than one world. See :func:`_replication_refusal`.

    A named type rather than a bare ``NotImplementedError`` because one caller
    needs to tell *this* refusal from every other way a compile can fail: an
    interactive page, where the world count is a knob and the task is the
    request, answers this one by dropping to a single world rather than by
    refusing the selection. Nothing else may be answered that way -- an
    inadmissible pairing or a missing asset is not fixed by narrowing the grid.

    Subclasses ``NotImplementedError`` so every existing handler, and every
    message already written about it, keeps working unchanged.
    """


@dataclass
class CompiledScene:
    """Everything one compiled scene consists of.

    Attributes:
        request: What it was compiled from -- a `SceneRequest`, which for a
            benchmark run is the `TaskSpec` and for a hand-assembled scene is a
            `BuildRequest`. The compiler reads ten things off it and does not
            care which.
        episode: The episode it was compiled for.
        model: The finalized model.
        builder: The builder, kept because grasp boxes and visual parts are
            read off pre-finalize geometry.
        pipeline: The substep loop.
        handles: Where everything landed.
        entities: Every entity, in build order.
        embodiment: The robot's entity, or ``None`` for a scene with no robot.
        objects: Rigid entities by name.
        num_worlds: Replicated worlds in :attr:`model`. 1 for a single-world
            scene, where :attr:`template_model` is :attr:`model` itself.
        template_model: The *unreplicated* model, finalized from the same builder.
            Every entity is bound to this one, so labels stay unambiguous and an
            index resolved once is right for every world -- see
            :class:`~robosimstudio.core.worlds.WorldMap`. It is the model handles,
            arm index sets and host-side planning are expressed in.
        worlds: The stride map from template indices to replicated ones, or
            ``None`` for a single-world scene.
    """

    request: SceneRequest
    episode: Episode
    model: newton.Model
    builder: newton.ModelBuilder
    pipeline: ScenePipeline
    handles: SceneHandles
    entities: list[Entity] = field(default_factory=list)
    embodiment: Entity | None = None
    num_worlds: int = 1
    template_model: newton.Model | None = None
    worlds: WorldMap | None = None
    program: PhysicsProgram | None = None
    """What the composer built: islands, channels, routing, the manifest."""

    @cached_property
    def views(self) -> dict[str, dict[str, Entity]]:
        """The four scene views, derived once.

        Cached rather than recomputed per access, because ``entities`` is fixed
        from the moment this exists and the derivation is not free: ``Ref.entity``
        scans all four, so a single lookup was four walks of the entity list with
        a registry lookup per entity -- 34 us, measured, for a thirteen-entity
        scene, on a path the studio takes per frame.
        """
        return derive_views(self.entities, self.embodiment)

    @cached_property
    def by_name(self) -> dict[str, Entity]:
        """Every built entity except the robot, by name.

        The view-independent lookup. Anything asking "which entity is called
        this" wants this and not a view -- keying such a question on a view means
        a kind that moves between them silently stops being found, which is
        exactly what happens the first time an entity starts offering a handle.
        """
        return {e.name: e for e in self.entities if e is not self.embodiment}

    @property
    def objects(self) -> dict[str, Entity]:
        """Free bodies whose parts are individually nameable. See :func:`view_of`."""
        return self.views["objects"]

    @property
    def statics(self) -> dict[str, Entity]:
        """Scenery: domains declaring ``world_fixed``."""
        return self.views["statics"]

    @property
    def model_key(self) -> str:
        return self.episode.model_key

    @property
    def index_model(self) -> newton.Model:
        """The model indices are resolved against: the template when replicated.

        Single-world, this is :attr:`model` and the distinction costs nothing.
        Replicated, reading it off :attr:`model` would be the bug that produces
        world 0's answer for every world -- or, for a name lookup, raises.
        """
        return self.template_model if self.template_model is not None else self.model


def _resolve_entity(path: str) -> type:
    """A class by dotted path, imported when it is needed.

    One caller left: the embodiment. ``robots`` imports the entity contract from
    this package, so naming its class at module scope would close the cycle that
    moving the entities out was meant to open.
    """
    import importlib

    module_name, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module_name), name)


_EMBODIMENT_ENTITY = "robosimstudio.robots.entity.EntityEmbodiment"
"""What builds the robot, by path for the same reason the others are.

``robots`` imports the entity contract from this package, so naming its class
here would close the cycle that moving the entities out was meant to open."""


def _make_entity(spec, arrangement=None, *, gripper_span: float = 0.08) -> Entity:
    """The entity a spec names, for the engine that will integrate it.

    One rule, where there were two. A table here held one class per build
    stage for a spec that named none, plus a fourth reached by sniffing the
    *shape* of the spec -- scenery that stripped its free bodies and collapsed
    its fixed joints got ``EntityArena``. That branch selected nothing across
    all ten shipped tasks, so the compiler carried a heuristic for a class
    nobody could reach, and ``EntityArena`` was unreachable code. It is
    registered under ``"arena"`` now and a spec asks for it by name.

    **A kind may have an implementation per engine**, and which one this is
    depends on the arrangement -- which is why it is resolved here rather than
    at ``Scene.add``. The rules are
    :func:`~robosimstudio.scene.registry.resolve_entity_class`'s; what matters
    at this call site is that the arrangement is already in hand (it has to be,
    for ``register_custom_attributes``) and that a kind with no engine-specific
    implementation resolves exactly as it did before the axis existed.

    ``gripper_span`` is offered rather than passed: it gates every object's
    ``ObbGrasp.fits_gripper`` and means nothing to a cloth, so it goes only to a
    class that declares it. Asked of the signature, so a suite's own entity
    states its own needs -- the same rule the control modes follow for
    ``control_hz``.
    """
    from .registry import resolve_entity_class

    cls = resolve_entity_class(spec, arrangement)
    takes_span = "gripper_span" in inspect.signature(cls).parameters
    return cls(spec, **({"gripper_span": gripper_span} if takes_span else {}))


def compile_scene(
    request: SceneRequest,
    episode: Episode | None = None,
    *,
    num_worlds: int = 1,
    device: str | None = None,
    use_cuda_graph: bool | None = None,
    verbose: bool = False,
    compose: Composer | None = None,
    pipeline_cls: type[ScenePipeline] = ScenePipeline,
    episodes: Sequence[Episode] | None = None,
) -> CompiledScene:
    """Build one scene for one episode.

    Args:
        request: What to build -- anything satisfying
            :class:`~robosimstudio.scene.request.SceneRequest`. A `TaskSpec`
            does; so does a `BuildRequest` assembled by hand.
        episode: Which episode's layout to build it at. ``None`` is the single
            episode a scene that nobody varies has, which is what a scene with
            no benchmark over it wants.
        num_worlds: Replicated worlds. 1 builds the single-world model this
            function has always built, on the same code path. Above 1, the scene
            is built once and replicated -- see :func:`_compile_multi` for what
            that costs and what it refuses.
        episodes: One episode per world, when the worlds are to hold *different*
            drawn layouts. ``None`` gives every world ``episode``'s, which is
            what a replicated scene has always been. See :func:`_compile_multi`:
            what an episode draws is written into the model **before** the
            solvers and any host engine are built, so a world's foreign-engine
            actors are *created* where that world's episode put them rather than
            created at one layout and moved to another.
        device: Warp device. ``None`` uses the current one.
        use_cuda_graph: Overrides the physics spec's own setting.
        verbose: Print what was built.
        compose: What builds the physics from the built entities and the
            arrangement, returning a
            :class:`~robosimstudio.scene.physics.PhysicsProgram`. ``None`` is
            :func:`robosimstudio.scene.physics.compose`, and the realistic use
            of the parameter is to *wrap* that rather than replace it -- call
            it, amend the schedule it returns, hand it back.
        pipeline_cls: The substep executor class. ``ScenePipeline`` runs the nine
            phases in their fixed order; a subclass may run its own. Both this
            and ``compose`` exist because the two of them were the last places
            the compiler decided something a caller might legitimately want to
            decide, and neither was reachable without editing this file.

    Returns:
        The compiled scene, ready to :meth:`~robosimstudio.core.pipeline.ScenePipeline.step`.
    """
    if num_worlds < 1:
        raise ValueError(f"num_worlds must be >= 1, got {num_worlds}")
    if episode is None:
        # A scene nobody varies has exactly one episode, and this is it. The
        # compiler threads an episode through every ``build_into`` because a
        # benchmark varies layouts per episode; a scene assembled by hand
        # varies nothing and should not have to say so.
        episode = Episode(episode_id=0, seed=0)

    with wp.ScopedDevice(device) if device else _null_scope():
        if num_worlds == 1:
            return _compile(
                request,
                episode,
                use_cuda_graph=use_cuda_graph,
                verbose=verbose,
                compose=compose,
                pipeline_cls=pipeline_cls,
            )
        return _compile_multi(
            request,
            episode,
            num_worlds=num_worlds,
            use_cuda_graph=use_cuda_graph,
            verbose=verbose,
            compose=compose,
            pipeline_cls=pipeline_cls,
            episodes=episodes,
        )


class _null_scope:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _mount(request: SceneRequest, binding: Binding, scene: CellSpec | None, spec: EmbodimentSpec) -> Xform:
    """Where this pairing puts the robot's base, in world coordinates.

    Three sources, most specific first: the binding names a pose, or it names
    one of the scene's own mount points, or the pose is *derived* from the
    scene's work-surface frame and the embodiment's ``base_offset``.

    Derivation is the point. The absolute base height across the shipped cells
    spans 0.003 to 0.912 m while the height above the work surface is only ever
    0, +0.022 or +0.11 -- so a new pairing is a statement about a robot and a
    place, not a measurement somebody has to take. An explicit ``mount`` stays
    available for the pairing where the derivation is wrong, which a contact
    sheet shows and nothing else does.

    ``request.embodiment_pose`` is honoured when the binding derives and the
    embodiment states no offset, which is what keeps a task that set it before
    any of this existed compiling to the same model.
    """
    if binding.mount is not None:
        return binding.mount
    if binding.mount_point:
        if scene is None:
            raise ValueError(
                f"scene {request.full_name!r}: binding names mount point {binding.mount_point!r} but no scene"
            )
        return in_surface(scene, scene.mount(binding.mount_point))
    if spec.base_offset != IDENTITY:
        return in_surface(scene, spec.base_offset)
    # Neither axis has stated an offset: the asset places itself, and the task's
    # own pose (identity everywhere today) is the only thing left to honour.
    return request.embodiment_pose


def _roles(binding: Binding, spec: EmbodimentSpec | None, request: SceneRequest) -> tuple[str, str]:
    """Which arm works and which supports, resolved to real arm names.

    Unstated, the work arm is the embodiment's first -- which is only a default,
    not a convention, because the two YAM cells order their arms oppositely.
    Anything that cares states the role and reads it back from the handles.

    A support arm the embodiment does not have is a hard error: it is exactly
    the signal that a bimanual task has been pointed at a single-armed robot,
    and the alternative is a scene that compiles and then silently never
    anchors anything.
    """
    if spec is None or not spec.arms:
        return "", ""
    names = [a.name for a in spec.arms]
    work = binding.work_arm or names[0]
    if work not in names:
        raise ValueError(f"scene {request.full_name!r}: work arm {work!r} is not one of {names}")
    support = binding.support_arm
    if support and support not in names:
        raise ValueError(
            f"scene {request.full_name!r}: needs a support arm {support!r}, but embodiment {spec.name!r} has {names}"
        )
    return work, support


@dataclass
class _BaseBuild:
    """One world's worth of built-but-unfinalized scene.

    The output of steps 0 through 6 of this module's docstring, stopping just
    before ``finalize``. Split out because that is exactly the point where the
    single-world and replicated paths diverge: one finalizes this builder, the
    other replicates it first and finalizes both.
    """

    builder: newton.ModelBuilder
    entities: list[Entity]
    embodiment: Entity | None
    binding: Binding
    scene_spec: CellSpec | None
    work_arm: str
    support_arm: str


def _embodiment_spec(request: SceneRequest, binding: Binding) -> EmbodimentSpec:
    """The robot to build: the request's own spec, or a registry lookup by name.

    A request may carry an :class:`~robosimstudio.specs.EmbodimentSpec` inline
    instead of a name -- a robot loaded from a file that nothing registered,
    which is the one place a hand-assembled scene and a benchmark request differ
    in *kind* rather than in content. The inline spec is used as it stands
    unless the binding asks for a different robot by name, which is a pairing
    override and resolves through the registry exactly as a name would.

    The import is lazy for the same reason the entity classes are reached by
    path: ``robots`` imports the entity contract from this package, so naming it
    at module scope would close a cycle. Once per compile; the cost is nothing.
    """
    inline = request.embodiment if isinstance(request.embodiment, EmbodimentSpec) else None
    wanted = binding.embodiment or (inline.name if inline is not None else request.embodiment)
    if inline is not None and wanted == inline.name:
        return inline
    from robosimstudio.robots.registry import get_embodiment

    return get_embodiment(wanted)


def _takes_per_world_body(entity: Entity) -> bool:
    """Whether this static, told it is replicated, will build itself a kinematic body.

    The static domain's own rule (:func:`~robosimstudio.domains.static.entity.placeable_static`,
    less the subclasses that weld themselves), asked here so the build loop can
    order it after the robot. Anything that is not a static answers no.
    """
    from robosimstudio.domains.static.entity import EntityStatic

    return isinstance(entity, EntityStatic) and type(entity).places_per_world(entity.spec)


def _build_base(
    request: SceneRequest, episode: Episode, *, dynamics: bool = True, replicated: bool = False
) -> _BaseBuild:
    """Build one world into a fresh builder, in the order this module documents.

    ``replicated`` says the builder is about to be replicated into a fleet, so
    every static the task declares is told before it builds
    (:attr:`~robosimstudio.domains.static.entity.EntityStatic.replicated`) and
    a placeable one takes the per-world channel. The cell's own furniture is
    not told: a bench is the surface, not a fixture an episode draws.

    Everything up to but not including ``finalize``. Both compile paths run this
    and nothing else builds a scene, so "the replicated model is the single-world
    model, N times" is true by construction rather than by two code paths
    agreeing.

    ``dynamics=False`` stops after the robot, leaving out the task's objects,
    deformables and extras. That is not a scene anybody can step -- it exists so
    an IK solver can be built on the arm's kinematics without the things that
    make a *scene* hostile to it: a cable, a D6-jointed connector, or simply a bin
    of loose tools whose free joints push the coordinate count past the solver's
    ceiling. Because those all build *after* the robot, dropping them leaves every
    earlier index untouched, and this stays the same import of the same asset by
    the same code rather than a second opinion about the arm's kinematics. The
    caller still pairs the two index spaces explicitly and checks the pairing --
    see :func:`~robosimstudio.scene.ik_model.build_ik_rig`.
    """
    # joint_q-aligned targets, so joint_target_q indexes the way joint_q does
    # even with free-joint objects in the same model.
    newton.use_coord_layout_targets = True

    arrangement = request.physics
    collision = arrangement.collision
    builder = newton.ModelBuilder()
    if arrangement.gravity != -9.81:
        builder.gravity = float(arrangement.gravity)
    if collision.rigid_gap is not None:
        # Before anything imports: this is the default gap for every shape that
        # states none, and a tight-clearance task needs it at the scale of its
        # fits rather than Newton's centimetre-scale default.
        builder.rigid_gap = float(collision.rigid_gap)
    # Also before anything imports, and for the same reason: USD carries no
    # contact stiffness, so every imported shape takes ShapeConfig's default and
    # this is the one place a task can state otherwise.
    if collision.contact_ke is not None:
        builder.default_shape_cfg.ke = float(collision.contact_ke)
    if collision.contact_kd is not None:
        builder.default_shape_cfg.kd = float(collision.contact_kd)
    if collision.force_sdf:
        # Before anything imports, because an SDF is baked at finalize from the
        # config a shape was *added* with. This is what makes
        # ``full_surface_contact`` usable on a robot whose colliders are meshes:
        # analytic boxes and capsules carry closed-form SDFs and need nothing,
        # but a mesh without a baked one is silently skipped by the edge and face
        # passes -- and Newton raises rather than let a soft body sail through it.
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
    # Every backend the arrangement names gets to register its custom attributes
    # before any entity builds -- MuJoCo's gravcomp, VBD's per-joint flags. Done
    # here, from the arrangement, so no entity has to import a solver to know
    # which attributes its builder calls will be allowed to keep.
    register_custom_attributes(builder, arrangement)

    entities: list[Entity] = []
    embodiment: Entity | None = None

    # The pairing being compiled, and the frame the other two axes agree in.
    binding = request.binding_for()
    scene_spec = get_cell(binding.scene) if binding.scene else None
    # Task-owned entities state their poses in the scene's work-surface frame;
    # the scene's own furniture states them in world. Identity without a scene,
    # so a suite that has not moved onto the frame keeps world coordinates.
    frame = surface_frame(scene_spec)

    # --- 0. the place -----------------------------------------------------
    # Before the task's own scenery, so a scene's floor cannot land on top of a
    # fixture standing on it, and so the surface frame exists before anything
    # is placed in it.
    if scene_spec is not None:
        for spec in scene_spec.statics:
            entity = _make_entity(spec, arrangement)
            # The cell's own furniture is the *surface*, not a fixture. Both are
            # `static` and the domain cannot tell them apart; the build loop can,
            # because this is the loop that builds the place rather than the task.
            entity.role = "surface"
            entity.build_into(builder, episode)
            entities.append(entity)

    # Everything the task declares, in the order the stages run. Two loops
    # rather than four, and the split is the robot: scenery builds before it,
    # everything else after -- which is also what ``dynamics=False`` drops.
    ordered = request.build_order()

    # --- 1. scenery -------------------------------------------------------
    overrides = dict(binding.static_overrides)
    # In a replicated build the placeable statics -- the ones that take a
    # kinematic body of their own -- build AFTER the robot (step 2b), and this
    # is not a style choice. An embodiment imported with ``add_mjcf_collapsed``
    # collapses the fixed joints of the WHOLE builder, and a fixture welded to
    # the world by a fixed joint is exactly what that folds away: its body is
    # merged into the world, every body after it is renumbered, and the entity
    # goes on holding the index it was given -- which is now the robot's first
    # link. Measured on a tool-hanging scene at two worlds: the
    # pegboard's per-world placement moved ``left_link_1`` onto the pegboard.
    # Nothing that imports after the robot collapses the builder, so a static
    # built there keeps its body. The single-world build keeps the order it
    # always had; a world-hung static owns no body for a collapse to take.
    deferred: list = []
    for declared in (s for s in ordered if build_stage(s) == "scenery"):
        # This pairing may import a different file for a static -- the stripped
        # arena, for a monolithic export hosting a foreign robot.
        spec = (
            replace(declared, asset=replace(declared.asset, path=overrides[declared.name]))
            if declared.name in overrides
            else declared
        )
        entity = _make_entity(spec, arrangement)
        entity.frame = frame
        if replicated and hasattr(entity, "replicated"):
            entity.replicated = True
            if _takes_per_world_body(entity):
                deferred.append(entity)
                continue
        entity.build_into(builder, episode)
        entities.append(entity)

    # --- 2. the robot -----------------------------------------------------
    if request.embodiment:
        embodiment_spec = _embodiment_spec(request, binding)
        if binding.gripper:
            # A hand named by the pairing replaces every arm's. It has to clear
            # the per-arm names as well as set the robot's, because an arm's own
            # entry wins over the robot's default -- leaving them would put the
            # requested hand on nothing and read as the flag being ignored.
            from dataclasses import replace as _replace

            embodiment_spec = _replace(
                embodiment_spec,
                gripper=binding.gripper,
                arms=tuple(_replace(a, gripper="") for a in embodiment_spec.arms),
            )
        arm_gains = binding.arm_actuation or request.arm_actuation
        gripper_gains = binding.gripper_actuation or request.gripper_actuation
        if arm_gains is not None or gripper_gains is not None:
            # Gains fall through binding, then request, then the embodiment's own.
            # A binding's are the most specific statement there is: how stiff an
            # arm has to be to seat a connector is a fact about *that arm doing
            # that task*, which neither half knows alone.
            from dataclasses import replace as _replace

            embodiment_spec = _replace(
                embodiment_spec,
                arm_actuation=arm_gains or embodiment_spec.arm_actuation,
                gripper_actuation=gripper_gains or embodiment_spec.gripper_actuation,
            )
        if binding.cross_arm_collisions is not None:
            # Whether the arms may pass through each other is a fact about the
            # pairing: it only matters on a task that puts both hands in the
            # same place, and the pairs are not free. Tri-state, so a binding can
            # also switch it *off* for a robot that ships with it on.
            from dataclasses import replace as _replace

            embodiment_spec = _replace(embodiment_spec, cross_arm_collisions=bool(binding.cross_arm_collisions))
        embodiment = _resolve_entity(_EMBODIMENT_ENTITY)(
            embodiment_spec,
            pose=_mount(request, binding, scene_spec, embodiment_spec),
            scene_resident=binding.scene_resident,
        )
        embodiment.build_into(builder, episode)
        embodiment.apply_actuation(builder)
        entities.append(embodiment)

    work_arm, support_arm = _roles(binding, embodiment_spec if embodiment is not None else None, request)

    gripper_span = 0.08
    if embodiment is not None and embodiment_spec.arms:
        # The *work* arm's span, not arms[0]'s: the two YAM cells list their
        # arms in opposite orders, so an index here would silently mean the left
        # hand on one robot and the right on the other -- and this number gates
        # every object's ObbGrasp.fits_gripper.
        #
        # The hand's own span wins over twice the open coordinate: for a geared
        # hand the joint coordinate is an angle, so twice it is not a distance
        # and an OBB fit test made against it would pass everything. That rule is
        # GripperSpec.effective_span, in one place, rather than restated here.
        arm = next((a for a in embodiment_spec.arms if a.name == work_arm), embodiment_spec.arms[0])
        hand = embodiment.hands.get(arm.name)
        if hand is not None:
            gripper_span = hand.effective_span

    # --- 2b. the placeable statics of a replicated build ------------------
    # After the robot, for the reason step 1 gives. Still before the objects,
    # so an object's reset pose is measured against fixtures that already
    # stand -- and still dropped by ``dynamics=False``, which stops at the
    # robot: an IK rig has no use for a peg.
    for entity in deferred if dynamics else ():
        entity.build_into(builder, episode)
        entities.append(entity)

    # --- 3. everything the robot can touch --------------------------------
    # One loop over three stages. What used to differ between them was a rigid
    # body's shape collection, collider approximation and grasp box, which now
    # belong to ``EntityRigid.after_build`` -- so the compiler no longer knows
    # that an object has a grasp box and a cloth does not.
    #
    # Every entity is offered the embodiment, where only ``extras`` used to be.
    # The ones that couple to the robot need it -- a cable latched to a gripper,
    # a rig measuring blade geometry -- and re-deriving it from body-name
    # patterns is the failure that selects nothing and reports nothing. The rest
    # ignore it.
    for spec in (s for s in ordered if build_stage(s) != "scenery") if dynamics else ():
        entity = _make_entity(spec, arrangement, gripper_span=gripper_span)
        entity.frame = frame
        entity.embodiment = embodiment
        # Told here as well as at step 1, because a *fixture* that is not a
        # ``StaticSpec`` builds in this loop: a gear-meshing board is an ``extras``
        # entity whose plate, pegs, pinion and button are the model's own
        # geometry, and which takes a kinematic root of its own when it is about
        # to be replicated so that each world's copy can stand where its own
        # document drew it. Nothing else in the corpus declares the attribute, so
        # for every other entity this is the no-op it was before.
        if replicated and hasattr(entity, "replicated"):
            entity.replicated = True
        entity.build_into(builder, episode)
        entities.append(entity)

    # --- 5. finalize ------------------------------------------------------
    # Scene-wide, and after every entity has built: a texture bound by one
    # import must not be tinted by a colour another import happened to leave on
    # the same shape. See clear_texture_tints for why the two disagree at all.
    clear_texture_tints(builder)
    builder.color()
    # The scene's own floor height, which every shipped scene states as 0 and one
    # family does not: SimplerEnv's rooms stand on a 17 mm slab, and that slab is
    # where its robots' mount heights are measured from.
    # Coloured, which it never was: `add_ground_plane` defaults to Newton's
    # (0.125, 0.125, 0.15) -- pixel 27 under the tiled tracer -- so every scene
    # shipped a near-black floor regardless of what its table was painted. A
    # step below GROUND rather than equal to it, or the table edge stops reading.
    builder.add_ground_plane(
        height=float(scene_spec.ground_z) if scene_spec is not None else 0.0,
        color=FLOOR,
    )
    return _BaseBuild(
        builder=builder,
        entities=entities,
        embodiment=embodiment,
        binding=binding,
        scene_spec=scene_spec,
        work_arm=work_arm,
        support_arm=support_arm,
    )


def _lifecycle(parts) -> TaskSchedule:
    """The task-level schedule for everything in this scene.

    One schedule built from every part rather than a loop per phase, so the
    order lives in :data:`~robosimstudio.core.lifecycle.TASK_PHASES` instead of
    in the sequence of ``for`` statements below it -- and so a part that owns no
    geometry can still take part. Contribution order inside a phase is the order
    the parts arrive in, which is the build order the compiler already fixed.
    """
    schedule = TaskSchedule()
    for part in parts:
        schedule.extend(steps_of(part))
    return schedule


def _assemble(
    request: SceneRequest,
    base: _BaseBuild,
    model: newton.Model,
    *,
    use_cuda_graph: bool | None,
    worlds: WorldMap | None = None,
    verbose: bool = False,
    compose: Composer | None = None,
    pipeline_cls: type[ScenePipeline] = ScenePipeline,
) -> ScenePipeline:
    """Compose the physics for a finalized model, then the substep loop over it.

    Shared by both compile paths because the physics of N worlds is the physics
    of one: the composer expands every island's ownership by the world map, the
    solvers are world-aware, and the substep block never mentions a world at
    all. What differs between the paths is the model handed in.

    The composer gets every built entity by name and the embodiment, and reads
    what each declared it needs; the arrangement says which backend serves each
    domain. Nothing here knows whether the result is one bare MuJoCo or a
    proxy-coupled pair -- see :mod:`robosimstudio.scene.physics`.
    """
    entities, embodiment = base.entities, base.embodiment
    arrangement = request.physics
    parts: dict[str, Any] = {e.name: e for e in entities if e is not embodiment}
    scene_statics = {spec.name for spec in base.scene_spec.statics} if base.scene_spec is not None else set()
    program = (compose or _compose)(
        model,
        parts,
        embodiment,
        arrangement,
        scene_statics=scene_statics,
        worlds=worlds,
        verbose=verbose,
    )

    for step in _lifecycle(entities).steps("kernels"):
        step.fn(program.hooks)

    return pipeline_cls(
        model,
        program=program,
        control_hz=arrangement.control_hz,
        substeps=arrangement.substeps,
        use_cuda_graph=arrangement.use_cuda_graph if use_cuda_graph is None else use_cuda_graph,
        worlds=worlds,
    )


def _start_pose(
    model: newton.Model, pipeline: ScenePipeline, embodiment: Entity | None, *, num_worlds: int = 1
) -> None:
    """Put the scene at its start pose before anything steps it.

    A freshly built State has no evaluated body transforms, and integrating from
    that yields NaN within one step -- which then gets baked into a captured
    graph and silently poisons every later step.

    A deformable-only scene -- a material rig -- has no bodies at all, so there
    is no rigid start pose to evaluate and nothing to copy.

    ``num_worlds`` only reaches the target vector: ``model.joint_q`` is already
    the replicated model's own, tiled by ``replicate``, but the home targets come
    from the embodiment and describe one robot.
    """
    if model.body_count:
        newton.eval_fk(model, model.joint_q, model.joint_qd, pipeline.state_0)
        wp.copy(pipeline.state_1.body_q, pipeline.state_0.body_q)
    if embodiment is not None:
        home = np.asarray(embodiment.home_target_q(), dtype=np.float32).reshape(-1)
        pipeline.control.joint_target_q.assign(np.tile(home, num_worlds) if num_worlds > 1 else home)


def view_of(entity: Entity) -> str:
    """Which of the four scene views this entity belongs to.

    ``build_stage`` is a build-order statement -- which pass adds this to the
    builder -- and the compiler used it as the taxonomy deciding which handle you
    get, which probe you get, and which studio panel can see you. A spec that
    declares no stage at all falls to ``extras``, so a chain of capsules whose
    spec said nothing was invisible to every bin reader while owning bodies and
    shapes exactly as a box does.

    Each test below is a contract the entity already answers:

    * ``world_fixed`` on its domain -- scenery, declared where the domain is.
    * owning particles -- what a deformable *is*, and the same question
      :meth:`Sim.deformables` asks.
    * offering a :meth:`Entity.handle` -- whether its parts are individually
      nameable. A box publishes one; a rod's segments are bodies the scene never
      publishes, which is exactly what made it an extra.

    **The order is the content.** A cloth both owns particles and offers a
    handle; a static offers one and is scenery. And no capability test alone
    reproduces the partition -- a capsule rod and a box own the same state and
    land in different views -- which is why the handle question is the one that
    separates them.
    """
    domain = getattr(entity, "domain", "")
    if domain in DOMAINS and get_domain(domain).world_fixed:
        return "statics"
    if len(entity.owned_particles()):
        return "deformables"
    return "objects" if type(entity).handle is not Entity.handle else "extras"


def derive_views(entities: list[Entity], embodiment: Entity | None) -> dict[str, dict[str, Entity]]:
    """The four scene views, from :func:`view_of`.

    A *partition* rather than four capability queries, because ``describe``
    iterates ``objects`` and ``deformables`` separately and a rod appearing in
    both would be counted twice.
    """
    views: dict[str, dict[str, Entity]] = {"objects": {}, "deformables": {}, "statics": {}, "extras": {}}
    for entity in entities:
        if entity is embodiment:
            continue  # its own axis; never in a view
        views[view_of(entity)][entity.name] = entity
    return views


def _finish(
    request: SceneRequest,
    episode: Episode,
    base: _BaseBuild,
    model: newton.Model,
    pipeline: ScenePipeline,
    *,
    template_model: newton.Model | None = None,
    worlds: WorldMap | None = None,
    num_worlds: int = 1,
    verbose: bool,
) -> CompiledScene:
    """Handles, the scene record, and the one line that says what was built.

    Handles come off ``template_model`` when the scene is replicated: they are
    label lookups, and every world carries the same labels. See
    :class:`~robosimstudio.core.worlds.WorldMap` for why an index resolved
    against one world is right for all of them.
    """
    index_model = template_model if template_model is not None else model
    handles = _build_handles(request, index_model, base.builder, base.embodiment, base.entities, scene=base.scene_spec)
    # Which place, and which hand does what: the two facts task code must read
    # off the compiled scene rather than assume, since both change with the
    # pairing while the task's own source does not.
    handles.scene_name = base.binding.scene
    handles.surface_z = float(base.scene_spec.surface_z) if base.scene_spec is not None else 0.0
    handles.work_arm, handles.support_arm = base.work_arm, base.support_arm
    scene = CompiledScene(
        request=request,
        episode=episode,
        model=model,
        builder=base.builder,
        pipeline=pipeline,
        handles=handles,
        entities=base.entities,
        embodiment=base.embodiment,
        num_worlds=int(num_worlds),
        template_model=template_model,
        worlds=worlds,
        program=pipeline.program,
    )
    if verbose:
        fleet = f" x {num_worlds} worlds" if num_worlds > 1 else ""
        print(
            f"[compile] {request.full_name} episode {episode.episode_id}{fleet}: "
            f"{model.body_count} bodies, {model.shape_count} shapes, "
            f"{model.particle_count} particles, {model.tri_count} tris"
        )
        print(handles.summary())
        print(pipeline.program.summary())
    return scene


def _compile(
    request: SceneRequest,
    episode: Episode,
    *,
    use_cuda_graph: bool | None,
    verbose: bool,
    compose: Composer | None = None,
    pipeline_cls: type[ScenePipeline] = ScenePipeline,
) -> CompiledScene:
    """One world: build it, finalize it, bind to it, step it."""
    base = _build_base(request, episode)
    model = base.builder.finalize()

    lifecycle = _lifecycle(base.entities)
    for step in lifecycle.steps("bind"):
        step.fn(model)
    # The one window before the solvers take their copies of the model's arrays.
    for step in lifecycle.steps("materials"):
        step.fn(model)

    pipeline = _assemble(
        request,
        base,
        model,
        use_cuda_graph=use_cuda_graph,
        verbose=verbose,
        compose=compose,
        pipeline_cls=pipeline_cls,
    )
    _start_pose(model, pipeline, base.embodiment)

    # --- 7. the first moment physics can run ------------------------------
    _run_ready(pipeline, lifecycle)

    return _finish(request, episode, base, model, pipeline, verbose=verbose)


def _run_ready(pipeline, lifecycle) -> None:
    """The first moment physics can run, for one world or for many.

    A host engine is attached *before* anything settles -- an entity's settle
    steps the pipeline, and the engine's tick has to know both states to write
    by then -- and records its rest *after* everything has, because the drape a
    deformable lands in is what a reset returns to.

    Shared rather than written twice. The replicated path used to run the
    ``ready`` steps and neither of the host phases, so a fleet whose deformable
    was routed to a foreign engine built, stepped, and reset to a rest nothing
    had captured. Nothing caught it: ``_replication_refusal`` looks for substep
    kernels, and no ``host_actor`` implementation contributes one.
    """
    hosts = list(getattr(pipeline.program, "hosts", {}).values())
    for host in hosts:
        attach = getattr(host, "attach_pipeline", None)
        if callable(attach):
            attach(pipeline)
    for step in lifecycle.steps("ready"):
        step.fn(pipeline)
    for host in hosts:
        capture = getattr(host, "capture_rest", None)
        if callable(capture):
            capture()


def _replication_refusal(base: _BaseBuild, request: SceneRequest) -> str:
    """Why this scene must not be replicated yet, or empty when it may be.

    Refusing by *capability* rather than by a list of task names, and the
    capability is narrow: an entity that contributes **substep kernels** with
    world-0 particle and body indices baked into them. Replicated, every world's
    physics would run against world 0's arrays -- plausible motion in the wrong
    place, which is the failure worth refusing where it can still be named. An
    entity whose kernels are written per world says so with
    ``world_aware = True`` and is let through.

    Deformables and coupled solvers are *not* refused any more: the composer
    expands every island's ownership by the world map, the coupled solver
    carries a world mask, and a settle is already world-wide. What a replicated
    deformable still lacks is per-world *scoring* -- the fleet's context has no
    per-world particle probe -- which is the fleet's to refuse for a task whose
    goal asks, not the compiler's to refuse for every task that has cloth.
    """
    contributors = sorted(
        entity.name
        for entity in base.entities
        if type(entity).contribute_hooks is not Entity.contribute_hooks and not getattr(entity, "world_aware", False)
    )
    if contributors:
        return f"these entities contribute substep physics written for one world: {', '.join(contributors)}"

    # A foreign engine builds *one* scene from the whole model, so a replicated
    # world's actors would land in it as extra bodies of the first rather than as
    # worlds of their own.
    #
    # **And the worlds are superimposed, which is what makes this harder than the
    # index bookkeeping it looks like.** Replication places every world at the
    # origin -- right for Newton, whose broadphase pairs within a world -- but a
    # host scene has no world axis to pair within. Sixteen sponges and sixteen
    # grippers at the same coordinates are one pile that collides with itself, so
    # emitting an actor record per world would not produce sixteen cells; it
    # would produce one cell with sixteen of everything in it, pressing on each
    # other, and reporting a contact force for it.
    #
    # Lifting this therefore needs the worlds *separated inside the host scene* --
    # a per-world translation added to each actor's rest coordinates and to every
    # mirrored collider's pose, and subtracted on read-back, which is the same
    # shape of answer the path tracer needs for the same reason. The recipe half
    # (an actor record per world, and the graph-node shape sets spread the way
    # ``routing.route`` spreads them) is the easy half and is not the blocker.
    #
    # Nothing in the corpus has ever built this pairing, so
    # it is refused by name rather than composed and quietly wrong -- which is
    # what it was: the replicated path ran no host attach and no rest capture at
    # all, and the substep-kernel test above cannot see a host engine because no
    # ``host_actor`` implementation contributes a kernel.
    arrangement = getattr(request, "physics", None)
    if arrangement is not None:
        from robosimstudio.specs.physics import get_backend

        hosted = []
        for entity in base.entities:
            domain = getattr(entity, "domain", "")
            if not domain:
                continue
            spec = getattr(entity.sim_requirements(), "backend", None) or arrangement.backend_for(domain)
            if spec is None:
                continue
            try:
                backend = get_backend(spec.solver)
                if backend.host_engine and spec.solver not in REPLICATED_HOSTS:
                    hosted.append(f"{entity.name} ({spec.solver})")
            except KeyError:
                continue
        if hosted:
            return (
                "these entities are served by a foreign engine, which builds one scene from the whole "
                f"model and has never been validated across worlds: {', '.join(sorted(hosted))}"
            )
    return ""


def _place_worlds(entities: list[Entity], model: newton.Model, episodes: Sequence[Episode], num_worlds: int) -> None:
    """Write each world's drawn layout into ``model``, before anything reads it.

    Only the entities that claim :attr:`~robosimstudio.scene.entity.Entity.world_aware`
    and offer a ``place`` -- the same claim a per-world ``reset`` needs, for the
    same reason: a placement written for one world and run ``W`` times puts
    world 0's bench in every world, which is the failure the whole per-world
    story exists to prevent.

    ``place`` takes a **model**, not a pipeline, because there is no pipeline
    yet -- that is the entire point of doing it here. An entity with per-world
    state that only a running scene can hold (a field to clear, a window to
    empty) leaves it to ``reset``; what belongs here is what something built
    *after* this will read and never re-read.
    """
    placers = [e for e in entities if getattr(type(e), "world_aware", False) and hasattr(e, "place")]
    if not placers:
        return
    for world, episode in enumerate(episodes[:num_worlds]):
        if episode is None:
            continue
        for entity in placers:
            entity.place(model, episode, world=world)


def _compile_multi(
    request: SceneRequest,
    episode: Episode,
    *,
    num_worlds: int,
    use_cuda_graph: bool | None,
    verbose: bool,
    compose: Composer | None = None,
    pipeline_cls: type[ScenePipeline] = ScenePipeline,
    episodes: Sequence[Episode] | None = None,
) -> CompiledScene:
    """``num_worlds`` copies of one scene in one model, stepped together.

    The scene is built exactly once, by the same :func:`_build_base` the
    single-world path uses, and then replicated. Two models come out of it and
    both are needed:

    * ``model`` -- the replicated one. This is what steps, what the solvers are
      built on, and what holds every world's state.
    * ``template_model`` -- the unreplicated one. Entities bind to it, so
      ``label_index`` still sees each name once, and every index they resolve is
      the offset :class:`~robosimstudio.core.worlds.WorldMap` applies per world.

    Replication happens *before* either finalize because finalize is what bakes a
    builder into device arrays, and ``replicate`` reads the builder.

    Spacing is deliberately zero -- every world sits on top of every other. Worlds
    do not collide with each other (the broadphase pairs shapes within a world, or
    with the global world the ground plane lives in), so separation would buy only
    a visual, and Newton's own guidance is that keeping worlds at the origin is
    better numerically. A viewer spreads them out at draw time instead.
    """
    base = _build_base(request, episode, replicated=True)
    refusal = _replication_refusal(base, request)
    if refusal:
        raise CannotReplicate(f"{request.full_name} cannot run replicated ({num_worlds} worlds) yet: {refusal}")

    fleet = newton.ModelBuilder()
    fleet.gravity = base.builder.gravity
    fleet.replicate(base.builder, num_worlds, spacing=(0.0, 0.0, 0.0))

    template_model = base.builder.finalize()
    model = fleet.finalize()
    worlds = WorldMap.of(template_model, model, num_worlds)

    # Before ``bind``, because ``bind`` is where an entity allocates its device
    # arrays and a world-aware one has to size them by the fleet. After the map
    # exists, because there is nothing to tell until both models are finalized.
    for entity in base.entities:
        entity.worlds = worlds
        entity.stepping_model = model

    lifecycle = _lifecycle(base.entities)
    for step in lifecycle.steps("bind"):
        step.fn(template_model)

    # Each world's own drawn layout, written into the model **before** anything
    # is built on top of it.
    #
    # A replicated scene is one episode's, W times, and a task that draws its
    # layout has had to move the other worlds afterwards -- at reset, once the
    # solvers and any host engine already exist. For Newton that is enough: an
    # array is an array whenever it is written. For a **foreign engine** it is
    # not. A host bridge builds its actors out of ``model.particle_q`` when it
    # is constructed, so a deformable moved afterwards is an actor created at
    # one layout and teleported to another -- and a surface-preparation foam measured
    # every static quantity identical either way while the two cases pressed
    # the bar differently, which is the shape of a difference that lives in an
    # engine's own state rather than in anything a test can read back.
    #
    # So the placement happens here: after ``bind``, because an entity resolves
    # its indices there, and before :func:`_assemble`, which is what builds the
    # solvers and the host island. Every world is then *built* at its own
    # layout rather than moved to it, which is the configuration a rebuild
    # produces and the only one this task has ever scored on.
    if episodes is not None:
        _place_worlds(base.entities, model, episodes, num_worlds)
    # Materials are written to the *replicated* model -- it is the one the
    # solvers read -- and every entity writes its own rows per world, which for
    # a model-wide scalar or a whole-array fill is the same write. Note the two
    # phases take *different* models here and the schedule does not hide that:
    # it orders the steps, the call site passes the arguments.
    for step in lifecycle.steps("materials"):
        step.fn(model)

    pipeline = _assemble(
        request,
        base,
        model,
        use_cuda_graph=use_cuda_graph,
        worlds=worlds,
        verbose=verbose,
        compose=compose,
        pipeline_cls=pipeline_cls,
    )
    _start_pose(model, pipeline, base.embodiment, num_worlds=num_worlds)

    # --- 7. the first moment physics can run ------------------------------
    # A settle steps every world at once, and a rest snapshot copies the whole
    # particle array, so the deformable entities' readiness is already per
    # world. Measured on the towel fold: four replicated worlds settle to
    # within 2e-10 of each other.
    _run_ready(pipeline, lifecycle)

    return _finish(
        request,
        episode,
        base,
        model,
        pipeline,
        template_model=template_model,
        worlds=worlds,
        num_worlds=num_worlds,
        verbose=verbose,
    )


def _build_handles(
    request: SceneRequest,
    model: newton.Model,
    builder: newton.ModelBuilder,
    embodiment: Entity | None,
    entities: list[Entity],
    scene: CellSpec | None = None,
) -> SceneHandles:
    handles = SceneHandles()

    # Each entity says where it landed.
    for entity in entities:
        if entity is embodiment:
            continue
        where = view_of(entity)
        if where == "statics":
            # Scenery contributes its sidecar whether or not it offers a handle
            # -- reached defensively, because ``meta`` is ``EntityStatic``'s JSON
            # sidecar and not part of the ``Entity`` contract. This view is
            # membership by *domain* now (``world_fixed``) rather than by the
            # ``scenery`` build stage, and those sets differ for exactly the case
            # this refactor is for: a novel welded kind with its own entity used
            # to land in ``extras`` and be merely invisible, and would now be
            # classified correctly and crash on an attribute nothing declares.
            handles.meta.update(getattr(entity, "sidecar", None) or {})
        made = entity.handle(model)
        if made is None:
            continue
        made.name = entity.name
        table = {"statics": handles.statics, "deformables": handles.deformables}.get(where, handles.objects)
        table[entity.name] = made

    if embodiment is not None:
        shape_body = model.shape_body.numpy()
        robot_bodies = {int(b) for b in embodiment.bodies}
        robot_shapes = np.asarray(
            [s for s in range(model.shape_count) if int(shape_body[s]) in robot_bodies], dtype=np.int32
        )
        handles.robot_shapes = robot_shapes
        for arm_spec in embodiment.spec.arms:
            index = embodiment.arms[arm_spec.name]
            hand = embodiment.hands.get(arm_spec.name)
            finger_groups = tuple(
                tuple((int(body), np.flatnonzero(shape_body == body).astype(np.int32)) for body in side)
                for side in index.finger_bodies
            )
            handles.arms[arm_spec.name] = ArmHandle(
                name=arm_spec.name,
                arm_coords=tuple(index.arm_coords),
                arm_dofs=tuple(index.arm_dofs),
                gripper_coords=tuple(index.gripper_coords),
                gripper_dofs=tuple(index.gripper_dofs),
                ee_body=index.ee_body,
                base_body=index.base_body,
                # The TCP and the grasp frame come off the *hand*: swap the hand
                # and both move while the arm does not.
                ee_offset=tuple(float(v) for v in (hand.tcp_offset if hand else (0.0, 0.0, 0.0))),
                # The grasp frame, and the two measured axes it is built from.
                # Off the *index* where the bind resolved them, off the spec
                # where it did not -- and carried at all so nothing downstream
                # has to re-find the ArmSpec by name to compose a pose. See
                # ArmHandle.ee_offset_rot for what leaving it off cost.
                ee_offset_rot=tuple(float(v) for v in (hand.tcp_offset_rot if hand else (0.0, 0.0, 0.0, 1.0))),
                approach_axis=tuple(float(v) for v in index.approach_axis),
                close_axis=tuple(float(v) for v in index.close_axis),
                gripper_open=index.gripper_open,
                gripper_span=index.gripper_span,
                gripper_open_q=tuple(float(v) for v in index.gripper_open_q),
                gripper_closed_q=tuple(float(v) for v in index.gripper_closed_q),
                gripper_geared=bool(hand.geared) if hand else False,
                finger_groups=finger_groups,
                open_axis_signs=tuple(hand.open_axis_signs if hand else (1.0, -1.0)),
                # THIS arm's shapes, from the index's own limb walk -- not the
                # whole embodiment's. `robot_shapes` here made both handles of
                # a bimanual cell identical, so `sim.clearance("arm:work",
                # "arm:support")` measured the union against itself and
                # answered every corridor question with the deepest self-
                # overlap of one arm's own links (measured -228 mm, constant,
                # on a USB-C insertion scene) -- a silent lie in the one tool
                # built to pre-flight a bimanual move. The whole-robot set
                # lives on `handles.robot_shapes`.
                shapes=index.shapes if len(index.shapes) else robot_shapes,
                home_q=tuple(float(v) for v in arm_spec.home_q),
            )

    handles.regions.update(resolve_regions(handles, request, model))

    del builder
    _collect_finishes(handles, model, entities)
    return handles


def resolve_regions(handles: SceneHandles, request: Any, model: newton.Model) -> dict[str, RegionHandle]:
    """Every region a request declares, resolved against an already-built scene.

    Split out of :func:`_build_handles` because a **superset** wave resolves
    these more than once: the model holds the base's fixtures, and each world
    runs a document whose regions are its own. One tool-sorting document puts
    ``bin2_multimeter_region`` in one quadrant of bin2 and another puts it
    in another, so a wave of both needs two tables over one model -- and a
    region is pure geometry on a named frame
    (:class:`~robosimstudio.specs.RegionSpec`), so the only thing resolving it
    needs is the frame's origin, which is what this looks up.

    ``model`` is the **template** model when the scene is replicated, as
    everywhere else a name becomes an index: a region's ``frame_body`` is a
    template index and :meth:`RegionHandle.origin_now` re-reads that world's
    row from it.

    Returns a fresh table rather than mutating ``handles.regions``, so a caller
    holding the base's handles can build a document's table without disturbing
    it.
    """
    out: dict[str, RegionHandle] = {}
    body_q = None
    for region in request.regions:
        frame_body = -1
        origin = np.zeros(3)
        if region.frame in handles.objects or region.frame in handles.statics:
            handle = handles.entity(region.frame)
            frame_body = handle.body
            if frame_body >= 0:
                if body_q is None:
                    state = model.state()
                    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                    body_q = state.body_q.numpy()
                origin = np.asarray(body_q[frame_body][:3], dtype=np.float64)
        # A frame the scene does not carry as a body: a *declared* static's
        # handle already carries its compile-resolved pose (stored at
        # construction for exactly this class of lookup -- a welded fixture has
        # no `body_q` row), and anything else is looked up in whatever metadata
        # the arena shipped, where a merged export records its bin origins.
        # Before the first branch existed, a region targeting a welded fixture
        # silently anchored at the world origin and its containment could never
        # fire -- the weave seats' failure mode.
        if not region.frame:
            # No target: the work surface, the frame every ``at=`` is stated in,
            # so a square drawn on the bench is drawn where the bench is.
            binding = request.binding_for()
            origin = np.asarray(surface_frame(get_cell(binding.scene) if binding.scene else None)[:3], dtype=np.float64)
        elif frame_body < 0:
            static = handles.statics.get(region.frame)
            if static is not None and any(s.name == region.frame for s in request.statics):
                # The handle's xform, which is this episode's -- so a seat
                # declared against a station follows the station when an episode
                # arranges the board. See `EntityStatic.handle`.
                origin = np.asarray(static.xform[:3], dtype=np.float64)
            else:
                origin = _origin_from_meta(handles.meta, region.frame, origin)
        out[region.name] = RegionHandle(
            name=region.name,
            frame=region.frame,
            frame_body=frame_body,
            origin=origin,
            xy_ranges=region.xy_ranges,
            z_range=region.z_range,
        )
    return out


def _collect_finishes(handles: SceneHandles, model: newton.Model, entities: list[Entity]) -> None:
    """Every shape's finish, every deformable's, and the live-recoloured set.

    The entities painted their own shapes in :meth:`~robosimstudio.scene.entity.Entity.apply_display`
    and recorded a finish beside each colour. What they could not record is
    anything added after their range closed -- the YAM's white shells come from
    an appearance hook that runs afterwards, the ground plane from the compiler
    -- so the gaps are filled here from the colour those shapes carry, and the
    table is complete for every shape in the model. A backend then never has to
    decide what an unlisted shape looks like.
    """
    from robosimstudio.core.render.finish import finish_for

    finishes: dict[int, Finish] = {}
    for entity in entities:
        finishes.update(getattr(entity, "finishes", {}) or {})
    colors = getattr(model, "shape_color", None)
    colors = (
        colors.numpy()
        if hasattr(colors, "numpy")
        else (np.asarray(colors, dtype=np.float32) if colors is not None else None)
    )
    robot = {int(s) for s in handles.robot_shapes}
    for s in range(int(model.shape_count)):
        if s in finishes:
            continue
        color = tuple(float(v) for v in colors[s][:3]) if colors is not None and s < len(colors) else None
        finishes[s] = finish_for(color, "robot" if s in robot else "objects")
    handles.finishes = finishes

    for entity in entities:
        if entity.name not in handles.deformables:
            continue
        stated = getattr(entity.spec, "finish", None)
        color = getattr(entity.spec, "color", None)
        handles.deformable_finishes[entity.name] = stated if stated is not None else finish_for(color, "deformables")

    live = sorted(
        {s for entity in entities if getattr(entity, "recolors_live", False) for s in range(*entity.range.shapes)}
    )
    handles.live_shapes = np.asarray(live, dtype=np.int32)


def _origin_from_meta(meta: dict[str, Any], frame: str, default: np.ndarray) -> np.ndarray:
    """A region frame's origin from the scene's JSON sidecar."""
    for key in (f"{frame}_pos", frame):
        value = meta.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return np.asarray(value[:3], dtype=np.float64)
    frames = meta.get("container_frames") or {}
    if frame in frames:
        return np.asarray(frames[frame][:3], dtype=np.float64)
    return default
