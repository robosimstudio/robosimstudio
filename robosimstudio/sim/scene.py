# SPDX-License-Identifier: Apache-2.0
"""Things in a place, and the arrangement that steps them.

``Scene`` is the front door. It holds a cell, a physics arrangement and a list
of things, and it turns them into something you can step::

    scene = rss.Scene(cell="bench_075")
    cube = scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0, 0, 0.2)))
    sim = scene.build()
    sim.run(120)
    print(cube.position)

**There is exactly one way to put something in: :meth:`add`.** There is
deliberately no ``add_box`` and no ``add_cloth``, and the reason is the whole
argument of this tier: a method per kind is a *closed set*, and a library that
owns the set is a library you have to edit to add a kind. Convenience
constructors exist -- ``box``, ``sheet``, ``sponge``, ``body`` -- and every one
of them is a **free function that returns a spec**, on exactly the same terms as
one you write yourself::

    def picnic_blanket(*, side, at):
        return rss.sheet(name="blanket", size=(side, side), cells=(40, 40), at=at)

    scene.add(picnic_blanket(side=0.6, at=(0.0, 0.0, 0.05)))

The library's constructors are not privileged. That is not a slogan -- it is
checked: ``tests/test_extension_surface.py`` defines a domain, an entity, a spec,
a solver and a coupler inside the test file and puts the result in a ``Scene``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from robosimstudio.scene import BuildRequest, compile_scene
from robosimstudio.scene.frames import in_surface
from robosimstudio.scene.ref import Ref
from robosimstudio.scene.registry import get_cell, get_entity
from robosimstudio.specs import (
    GENERIC,
    ActionSpec,
    ActuationSpec,
    ArmSpec,
    ArrangementSpec,
    AssetSpec,
    Binding,
    CameraSpec,
    EmbodimentSpec,
    RegionSpec,
    RenderSpec,
    Xform,
)

from .sim import Sim

__all__ = ["Robot", "Scene", "robot"]


@dataclass(frozen=True, kw_only=True)
class Robot:
    """One robot placement: which embodiment, where, and how it is driven.

    A separate type rather than three ``Scene`` arguments because it is one
    decision, and because :meth:`Scene.add` taking it means a robot arrives the
    same way everything else does.
    """

    name: str | EmbodimentSpec
    at: Xform | None = None
    """The base's pose in the cell's work-surface frame, or ``None`` for the
    robot's own mount (its spec's ``base_offset``)."""
    arms_at: tuple[tuple[str, tuple[float, ...]], ...] = ()
    """``(arm name, pose)`` pairs: single arms' bases, each in the cell's
    work-surface frame and absolute -- not relative to :attr:`at`. A pose is
    seven numbers, or three for a position that keeps the base's heading. A
    mapping is taken too, and held as pairs."""
    arm_actuation: ActuationSpec | None = None
    gripper_actuation: ActuationSpec | None = None

    def __post_init__(self) -> None:
        # One spelling, checked once, however the placement was written: the
        # mapping rss.robot takes, or pairs -- and pairs keep it hashable.
        object.__setattr__(self, "arms_at", _arms_at(self.arms_at))

    @property
    def label(self) -> str:
        """The robot's name whether it arrived as a registry key or a spec."""
        return self.name.name if isinstance(self.name, EmbodimentSpec) else self.name


def robot(
    which: str | EmbodimentSpec | None = None,
    *,
    asset: str | AssetSpec | None = None,
    arms: Sequence[ArmSpec] = (),
    name: str = "",
    at: Sequence[float] | None = None,
    arms_at: Mapping[str, Sequence[float]] | None = None,
    arm: ActuationSpec | None = None,
    gripper: ActuationSpec | None = None,
) -> Robot:
    """A :class:`Robot` placement, by registry name or from a file.

    Three spellings, one result::

        scene.add(rss.robot("panda", at=(-0.5, 0.0, 0.0)))                # registered
        scene.add(rss.robot(EmbodimentSpec(name="mine", asset=..., arms=...)))
        scene.add(rss.robot(asset="arms/mine.xml", arms=[ArmSpec(name="right",
                            joint_names=("j1", ..., "j6"), ee_body_name="wrist")]))

    The last two are pyroki's ``Robot.from_urdf`` shape: a robot that nothing
    registered, built from a file and a description of its arms. **A bare arm
    needs no registry at all** -- name the joints and the wrist and it builds,
    solves IK and takes joint-position commands. A gripper still resolves by
    name through the hand registry, because a hand's finger bodies, travel and
    tool-centre point are measured facts and the registry is where they live.

    Args:
        which: A registry name, or an :class:`~robosimstudio.specs.EmbodimentSpec`.
        asset: With ``arms``, builds a spec inline: a path under the asset roots
            or an :class:`~robosimstudio.specs.AssetSpec`.
        arms: One :class:`~robosimstudio.specs.ArmSpec` per arm.
        name: The inline spec's name; ``"robot"`` when unstated.
        at: Where its base goes, in the cell's work-surface frame like every
            other ``at``: a position ``(x, y, z)``, or a pose ``(x, y, z, qx,
            qy, qz, qw)``. Omitted, the robot stands at its own mount -- its
            spec's ``base_offset``, which for the shipped arms is at the edge of
            the work surface, facing it.
        arms_at: Single arms placed on their own, for a cell whose arms stand
            on separate mounts: ``{arm name: pose}``, each pose spelled as
            ``at`` is and in the same work-surface frame. A pose is where that
            arm's **base body** goes -- the body its
            :attr:`~robosimstudio.specs.ArmSpec.base_body_name` names, else
            the one its first moving joint hangs on in the robot's file (on
            ``yam_bimanual``, ``left_arm`` and ``right_arm``: the bottom of each
            mount puck). Absolute, not relative to ``at``: a named arm stands
            there wherever ``at`` puts the robot, and an arm left out follows
            ``at`` as before. A full pose ``(x, y, z, qx, qy, qz, qw)`` sets the
            base's heading too; a position ``(x, y, z)`` moves the base there
            and keeps the heading it already has from the file and ``at`` --
            unlike ``at``, whose position means no rotation, so that moving one
            arm of a turned robot does not turn the arm. The robot's file is not
            edited or copied -- each named arm is carried rigidly, its joints,
            actuators, hand and colliders with it -- and the arms' collision
            settings are the spec's, as without it::

                scene.add(rss.robot("yam_bimanual", arms_at={
                    "left": (-0.36, 0.20, 0.06),
                    "right": (0.12, -0.36, 0.22, 0.0, 0.0, 0.7071068, 0.7071068),
                }))

            An arm name the robot does not have is refused at build, naming
            the ones it has; so is a robot that arrives inside the scene's own
            import, an arm whose base hangs on a moving joint or shares the
            body welded to the world with another arm, and an arm an equality
            constraint ties to something that does not move with it.
        arm, gripper: Servo gains overriding the spec's own.
    """
    if isinstance(which, EmbodimentSpec):
        spec: str | EmbodimentSpec = which
    elif isinstance(which, str) and which:
        if asset is not None or arms:
            raise TypeError("rss.robot: give a registry name, or asset= and arms= to build one inline -- not both")
        spec = which
    elif asset is not None:
        if not arms:
            raise TypeError("rss.robot(asset=...) needs arms=: at least one ArmSpec naming its joints and its wrist")
        spec = EmbodimentSpec(
            name=name or "robot",
            asset=asset if isinstance(asset, AssetSpec) else AssetSpec(path=str(asset)),
            arms=tuple(arms),
        )
    else:
        raise TypeError("rss.robot: give a registry name, an EmbodimentSpec, or asset= and arms=")
    return Robot(name=spec, at=_base_at(at), arms_at=arms_at or (), arm_actuation=arm, gripper_actuation=gripper)


def _base_at(at: Sequence[float] | None) -> Xform | None:
    """``at`` as a pose: a position gets the identity rotation, anything else is refused."""
    if at is None:
        return None
    values = _spelled(at, "at=")
    return (*values, 0.0, 0.0, 0.0, 1.0) if len(values) == 3 else values


def _spelled(pose: Sequence[float], what: str) -> tuple[float, ...]:
    """Three numbers or seven, as floats, or the reason it is neither."""
    values = tuple(float(v) for v in pose)
    if len(values) in (3, 7):
        return values
    raise ValueError(
        f"rss.robot: {what} is a position (x, y, z) or a pose (x, y, z, qx, qy, qz, qw) in the "
        f"work-surface frame, and it was given {len(values)} numbers"
    )


def _arms_at(
    arms_at: Mapping[str, Sequence[float]] | Sequence[tuple[str, Any]] | None,
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """``arms_at`` as ``(arm name, pose)`` pairs, each pose three numbers or seven.

    Takes the mapping :func:`robot` is given or the pairs :class:`Robot` holds, so
    a placement built either way reads the same. A position stays a position: it
    keeps the base's heading, so it is not filled out to a pose here. Only the
    spelling is checked; whether the names are the robot's, and once each, is the
    compiler's question (``_arm_mounts``): only it knows which robot gets built.
    """
    if not arms_at:
        return ()
    items = arms_at.items() if isinstance(arms_at, Mapping) else arms_at
    pairs: list[tuple[str, tuple[float, ...]]] = []
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2 or not isinstance(item[0], str) or not item[0]:
            raise TypeError(f"rss.robot: arms_at= maps an arm's name to its base's pose, and got {item!r}")
        name, pose = item
        if pose is None:
            raise TypeError(f"rss.robot: arms_at[{name!r}] is None; leave the arm out to keep it where at= puts it")
        values = _spelled(pose, f"arms_at[{name!r}]")
        if len(values) == 7 and not sum(v * v for v in values[3:]) > 1.0e-12:
            # The surface frame's own composition reads a zero quaternion as no
            # rotation, so it would turn the base silently rather than fail here.
            raise ValueError(f"rss.robot: arms_at[{name!r}] has a zero quaternion; give a position to keep the heading")
        pairs.append((name, values))
    return tuple(pairs)


class Scene:
    """A cell, an arrangement, and the things standing in it.

    Args:
        cell: A registered cell name -- ``"bench_075"``, ``"void"``, or one a
            task directory declares. ``""`` is a bare world with a ground plane
            and no work surface, which is what ``"void"`` also is; naming
            ``void`` says you meant it.
        physics: How to step it. Defaults to
            :data:`~robosimstudio.specs.physics.GENERIC`, which is **not a tuned
            configuration** -- read its docstring before trusting a number that
            comes out of a scene you did not stage yourself.
        name: What to call this scene, for messages and recordings.
        binding: The pairing fields the compiler reads that a bare scene has no
            other place for -- a gripper by name, which arm works and which
            supports, a mount, ``cross_arm_collisions``, ``scene_resident``,
            ``static_overrides``. A task carries several and picks one per
            pairing; a scene has one, so it says so here.
        render: The look, when a renderer is asked for -- read by ``Sim``, not
            the compiler.
    """

    def __init__(
        self,
        *,
        cell: str = "",
        physics: ArrangementSpec = GENERIC,
        action: ActionSpec | None = None,
        name: str = "scene",
        binding: Binding | None = None,
        render: RenderSpec | None = None,
    ) -> None:
        self.cell = cell
        self.physics = physics
        self.action = action or ActionSpec()
        """How an action is read -- scales, conventions, gripper range. The
        defaults are, in ``ActionSpec``'s own words, "this package's own rather
        than any benchmark's", which is what makes leaving it unstated right."""
        self.name = name
        self.binding = binding
        self.render = render
        self._specs: list[Any] = []
        self._refs: dict[str, Ref] = {}
        self._robot: Robot | None = None
        self.regions: list[RegionSpec] = []
        self.cameras: list[CameraSpec] = []

    # ------------------------------------------------------------------ adding
    def add(self, spec: Any) -> Ref:
        """Put one thing in the scene and get back a handle on it.

        The handle works **after** :meth:`build`, and goes on working across a
        rebuild: it holds a name and a pointer to the current build, so the
        object you are holding is never the one from two builds ago.

        Args:
            spec: A build spec -- anything carrying a ``name`` and naming an
                entity, which is duck-typed on purpose -- a :class:`Robot`
                placement, a :class:`~robosimstudio.specs.RegionSpec` or a
                :class:`~robosimstudio.specs.CameraSpec`.

        Returns:
            A :class:`~robosimstudio.scene.ref.Ref`, of whatever class the
            entity declares. A robot's reads the robot's bodies (its
            :attr:`~robosimstudio.scene.ref.Ref.pose` is its base's), a
            region's has the region's :attr:`~robosimstudio.scene.ref.Ref.handle`,
            and a camera's carries its spec as ``.spec`` -- its name is what
            ``sim.render(camera)`` takes.

        Raises:
            ValueError: On a second thing with a name already taken, robots,
                regions and cameras included. Two things under one name is a
                scene where half the accessors silently answer about the wrong
                one. Also on a second robot: a scene holds one, and a two-armed
                setup is one :class:`~robosimstudio.specs.EmbodimentSpec` with
                two arms.
            KeyError: When the spec names an entity class nothing has
                registered -- raised **here**, where the line that added it is
                on the stack, rather than inside the compiler.
        """
        if isinstance(spec, Robot):
            if self._robot is not None:
                raise ValueError(
                    f"the scene already has a robot, {self._robot.label!r}, and holds one: a two-armed setup is "
                    "one EmbodimentSpec with two ArmSpecs. Scene.remove the first to replace it"
                )
            ref = self._claim(spec.label, _RobotRef)
            self._robot = spec
            return ref
        if isinstance(spec, RegionSpec):
            ref = self._claim(spec.name, Ref)
            self.regions.append(spec)
            return ref
        if isinstance(spec, CameraSpec):
            ref = self._claim(spec.name, _CameraRef)
            ref._spec = spec
            self.cameras.append(spec)
            return ref

        name = getattr(spec, "name", "")
        if not name:
            raise ValueError(f"{type(spec).__name__} has no name, and a name is how everything else finds it")
        ref = self._claim(name, _ref_class(spec))
        self._specs.append(spec)
        return ref

    def _claim(self, name: str, ref_cls: type[Ref]) -> Ref:
        """Reserve ``name`` and make the ref :meth:`build` will bind, or refuse a name already taken."""
        if name in self._refs:
            raise ValueError(f"the scene already has something called {name!r}")
        ref = ref_cls(None, name)
        self._refs[name] = ref
        return ref

    def add_all(self, specs) -> tuple[Ref, ...]:
        """:meth:`add` each, in order. Returns the refs in the same order."""
        return tuple(self.add(spec) for spec in specs)

    def remove(self, name: str) -> None:
        """Take one thing out again, by name -- an entity, the robot, a region or a camera.

        Raises:
            KeyError: If nothing in the scene is called that.
        """
        if name not in self._refs:
            raise KeyError(f"nothing called {name!r} in this scene; there is {sorted(self._refs)}")
        if self._robot is not None and self._robot.label == name:
            self._robot = None
        self._specs = [s for s in self._specs if getattr(s, "name", "") != name]
        self.regions = [r for r in self.regions if r.name != name]
        self.cameras = [c for c in self.cameras if c.name != name]
        del self._refs[name]

    def __contains__(self, name: str) -> bool:
        return name in self._refs

    def __len__(self) -> int:
        return len(self._specs)

    def __repr__(self) -> str:
        return f"Scene(cell={self.cell!r}, {len(self._specs)} things)"

    # ----------------------------------------------------------------- building
    @property
    def request(self) -> BuildRequest:
        """What this scene asks the compiler for.

        A :class:`~robosimstudio.scene.request.BuildRequest`, which satisfies the
        same :class:`~robosimstudio.scene.request.SceneRequest` protocol a
        ``TaskSpec`` does -- so a scene assembled by hand and a scene lowered
        from a benchmark document go into the *same* compiler by the same door.
        """
        return BuildRequest(
            name=self.name,
            cell=self.cell,
            physics=self.physics,
            action=self.action,
            entities=tuple(self._specs),
            regions=tuple(self.regions),
            cameras=tuple(self.cameras),
            embodiment=self._robot.name if self._robot else "",
            arm_actuation=self._robot.arm_actuation if self._robot else None,
            gripper_actuation=self._robot.gripper_actuation if self._robot else None,
            binding=self._binding_with_mount(),
            render=self.render,
        )

    def _binding_with_mount(self) -> Binding | None:
        """The scene's binding, with the robot's ``at`` stated as its mount and ``arms_at`` as its arm mounts.

        ``at`` and ``arms_at`` are in the work-surface frame and a binding's
        mounts are in world, so this is where the two meet. A mount is the first
        thing the compiler reads when it places a base, which is what makes
        ``at`` hold for a registered robot whose spec has an offset of its own.
        """
        at = _base_at(self._robot.at) if self._robot else None
        arms = self._robot.arms_at if self._robot else ()
        if at is None and not arms:
            return self.binding
        if at is not None and self.binding is not None and (self.binding.mount is not None or self.binding.mount_point):
            raise ValueError(
                f"scene {self.name!r}: rss.robot(at=...) and the binding's mount both say where the robot "
                "stands; give one"
            )
        if arms and self.binding is not None and self.binding.arm_mounts:
            raise ValueError(
                f"scene {self.name!r}: rss.robot(arms_at=...) and the binding's arm_mounts both say where the "
                "arms stand; give one"
            )
        cell = get_cell(self.cell) if self.cell else None
        binding = self.binding or Binding(scene=self.cell, embodiment=self._robot.label)
        changes: dict[str, Any] = {"scene": binding.scene or self.cell}
        if at is not None:
            changes["mount"] = in_surface(cell, at)
        if arms:
            # A position is carried over as a position -- the point in world --
            # so the base keeps its heading; a pose is a pose in world.
            changes["arm_mounts"] = tuple(
                (name, in_surface(cell, pose) if len(pose) == 7 else in_surface(cell, (*pose, 0.0, 0.0, 0.0, 1.0))[:3])
                for name, pose in arms
            )
        return replace(binding, **changes)

    def build(
        self,
        *,
        episode=None,
        num_worlds: int = 1,
        device="auto",
        use_cuda_graph: bool | None = None,
        verbose: bool = False,
        compose=None,
        pipeline=None,
        control_mode: str = "",
        max_steps: int = 0,
        renderer: str = "",
        render_every: int = 1,
        renderer_kwargs: dict[str, Any] | None = None,
        warmup: bool = True,
    ) -> Sim:
        """Compile the scene and return something you can step.

        ``episode`` through ``pipeline`` are the compiler's own -- ``compose``
        and ``pipeline`` being the two seams that let a caller replace the
        physics assembly or the substep program outright without subclassing
        anything here.

        **``device`` is ``"auto"``**: ``ROBOSIMSTUDIO_DEVICE`` where it names
        one, else CUDA where there is a GPU and the CPU where there is not.
        Name one (``"cpu"``, ``"cuda:1"``) to choose. The physics follows the
        device -- on a CPU, MuJoCo's C library and SuperDex, with ``GENERIC``
        built as :data:`~robosimstudio.specs.physics.CPU` -- and the renderer
        runs where the physics does. The build prints the whole choice on one
        line, and :attr:`Sim.platform` keeps it. ``control_mode`` through ``renderer_kwargs`` are the
        :class:`~robosimstudio.sim.Sim`'s: which control mode starts active,
        when to truncate, and whether to attach an observation rig.

        **``warmup`` is why this returns something already reset.** Compiling a
        scene does not compile its *kernels*, and it does not capture the substep
        block into a CUDA graph -- ``ScenePipeline.warmup_and_capture`` does both,
        and it runs on the first ``reset``. A ``Sim`` nobody had reset therefore
        stepped **eager**, which is a supported way to run and about an order of
        magnitude slower, with nothing anywhere saying so. ``Env`` never showed
        it because every caller of an ``Env`` resets it to pick an episode;
        ``scene.build().run(1000)`` has no such moment, and that is exactly the
        line this tier exists for.

        The reset is behaviour-neutral -- the warm-up steps real physics and
        restores the scene afterwards, and a cube released 200 mm over a bench
        settles at the same 0.7749 either way -- so ``warmup=False`` is for a
        caller who wants the model and does not intend to step it.

        **Building twice rebinds the refs to the newer build.** That is what makes
        an edit-and-rebuild loop work -- the ``cube`` you are holding reads the
        scene you just built, rather than the one before it, which is the hardest
        kind of wrong to see in a number that is nearly right. The cost is that
        two ``Sim`` s from one ``Scene`` are not independent: the older one's refs
        follow the newer. Hold two scenes if you want two.
        """
        extra = {} if pipeline is None else {"pipeline_cls": pipeline}
        compiled = compile_scene(
            self.request,
            episode,
            num_worlds=num_worlds,
            device=device,
            use_cuda_graph=use_cuda_graph,
            verbose=verbose,
            compose=compose,
            **extra,
        )
        sim = Sim(
            compiled,
            self._refs,
            control_mode=control_mode,
            max_steps=max_steps,
            renderer=renderer,
            render_every=render_every,
            renderer_kwargs=renderer_kwargs,
            verbose=verbose,
        )
        if warmup:
            # Not in ``Sim.__init__``: ``Env`` is a ``Sim`` and its callers reset
            # it themselves to pick an episode, so warming there would reset
            # every environment twice.
            sim.reset()
        return sim

    def with_physics(self, **changes: Any) -> Scene:
        """Amend the arrangement in place and return ``self``, so it chains.

        ``scene.with_physics(substeps=4)`` rather than rebuilding the whole
        ``Scene`` around a ``replace(GENERIC, ...)``, which is the same edit
        written twice.
        """
        self.physics = replace(self.physics, **changes)
        return self


class _RobotRef(Ref):
    """The robot :meth:`Scene.add` placed. It is the scene's embodiment rather
    than a named entity, so that is what it reads."""

    __slots__ = ()

    @property
    def entity(self) -> Any:
        return self._scene.embodiment


class _CameraRef(Ref):
    """A camera :meth:`Scene.add` declared. Its name is what ``sim.render``
    takes; it has no body, so the ref carries the spec it was added as."""

    __slots__ = ("_spec",)

    @property
    def spec(self) -> CameraSpec:
        return self._spec

    @property
    def pose(self):
        raise ValueError(
            f"{self.name!r} is a camera and owns no body; where it looks from is .spec (eye, look_at, xform), "
            f"and sim.render({self.name!r}) photographs through it"
        )


def _ref_class(spec: Any) -> type[Ref]:
    """The ``Ref`` subclass the spec's entity declares, or the base.

    Resolved at ``add`` time on purpose. An entity name nothing has registered is
    a mistake in the line that added it, and this is the only moment that line is
    still on the stack -- the compiler would raise the same ``KeyError`` much
    later, about a scene rather than about a statement.
    """
    name = getattr(spec, "entity", "")
    if not name:
        return Ref
    return getattr(get_entity(name), "ref", Ref)
