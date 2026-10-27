# SPDX-License-Identifier: Apache-2.0
"""A built scene you can drive: the thing :meth:`Scene.build` hands back.

``Sim`` owns a compiled scene, the control modes over it, the camera rig, the
observation providers and the loop. It has no task, no episode, no evaluator, no
goal and no notion of success -- those belong to the benchmark tier's
``Env``, which is a ``Sim`` with a job to do.

**The cut is where it is because of what is on each side of it.** Everything
here works on a scene somebody assembled by hand: an arm's IK does not know what
it is reaching for, a camera does not know what it is photographing, and
``ee_delta`` is a way of interpreting six numbers rather than a benchmark's
opinion. Everything ``Env`` adds needs a document to have declared it -- which
episode, what the evaluator says, whether the initial state holds. The check that
the cut is in the right place is that ``Env`` does not override :meth:`read` or
any observation provider: there is nothing task-shaped left to add to them.

The read surface is deliberately small. :meth:`run` and :meth:`step` advance it,
:meth:`read` gathers a :class:`~robosimstudio.scene.state.WorldState`, and a
:class:`~robosimstudio.scene.ref.Ref` per thing you added answers the question a
caller actually has -- is the cube on the shelf, has the rope stopped moving --
without the caller learning the index layout of a replicated model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from robosimstudio.core.env import EnvBase
from robosimstudio.core.render.base import RendererBase
from robosimstudio.core.render.look import look_for
from robosimstudio.core.render.spec import RenderSpec
from robosimstudio.machine import Platform, platform_of
from robosimstudio.scene.parking import GravcompParking, parked_objects, restrict_contacts, scene_park_poses
from robosimstudio.scene.ref import Ref
from robosimstudio.scene.state import WorldState
from robosimstudio.solvers import make_action_spaces
from robosimstudio.solvers.render import backend_class as _backend_class
from robosimstudio.solvers.render import create_renderer
from robosimstudio.specs import ActionSpec, CameraSpec, replace

__all__ = ["Sim", "scene_deformables"]


class Sim(EnvBase):
    """A compiled scene, its controllers, its cameras and the loop over them.

    Args:
        scene: What :func:`~robosimstudio.scene.compiler.compile_scene` returned.
        refs: The refs handed out by :meth:`Scene.add`, to be bound to this
            build. Rebinding rather than recreating is what lets the object a
            caller is holding go on working across a rebuild.
        control_mode: Which control mode starts active. Empty takes the first.
        max_steps: Control steps before an episode truncates. 0 never truncates,
            which is right for a scene with nothing to succeed at.
        renderer: Backend name for the observation rig, ``"auto"`` for the
            platform's (see :attr:`platform`), or ``""`` for none.
        render_every: Render on one step in this many.
        renderer_kwargs: Forwarded to the backend.
        verbose: Print what could not be built rather than silently doing without.
    """

    def __init__(
        self,
        scene: Any,
        refs: dict[str, Ref] | None = None,
        *,
        control_mode: str = "",
        max_steps: int = 0,
        renderer: str = "",
        obs_renderer: str = "",
        render_every: int = 1,
        renderer_kwargs: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> None:
        self.scene = scene
        self.verbose = bool(verbose)
        self.refs: dict[str, Ref] = dict(refs or {})
        for ref in self.refs.values():
            ref._scene = scene

        super().__init__(
            scene.pipeline,
            action_space=self._make_controllers(),
            control_mode=control_mode,
            max_steps=max_steps,
        )

        self.contacts = self._make_contact_probe()
        self._clearance = None
        if not hasattr(self, "episode"):
            # Only when a subclass has not already said. ``Env`` makes its
            # episode *before* compiling -- it has to, the compiler takes one --
            # so it is already set by the time this constructor runs, and today
            # it is the same object the scene carries. Assigning unconditionally
            # would be correct only for as long as that stays true, which is a
            # thing to notice rather than a thing to rely on.
            self.episode = getattr(scene, "episode", None)
            """The episode this scene was compiled for. A ``Sim`` never changes
            it -- it has no episodes to index -- and ``Env`` reassigns it per
            reset. Read by :meth:`reset_scene`, which hands it to each entity's
            own ``reset``."""
        self.reset_count = 0
        """Resets since the build: a reader that sampled ``step_count`` before tells a reset from a pause by it."""
        self._start_joint_q: np.ndarray | None = None
        """The articulation this run started from. Captured rather than compiled,
        because the warm-up steps real physics between the compiled pose and the
        first observation."""
        self._warned_nonfinite = False
        """Whether :meth:`run` has already said the state diverged. Once per
        episode: a loop calling ``run`` a hundred times should say it once, and
        :meth:`reset` clears it with the finite flags it reports."""
        self.settled_body_q = np.zeros((self.model.body_count, 7))
        self._parking: GravcompParking | None = None
        """The per-body gravity-compensation table, built on the first park."""
        self._parked: tuple[str, ...] = ()
        """Objects the current episode parks, in build order. See :attr:`parked`."""
        self._restricted: tuple[frozenset[str] | None, Any] | None = None
        """The last :attr:`handles` answer, keyed on the active set it was cut for."""

        self.render_every = max(1, int(render_every))
        self.renderer_name = renderer
        self.obs_renderer_name = obs_renderer
        self._narrowed_reported = False
        """Whether :meth:`_observable_rig` has already said what the backend
        cannot photograph. Once per ``Sim``, not once per rebuild."""
        self.renderer_kwargs = dict(renderer_kwargs or {})
        self._cameras = self._resolve_cameras()
        self.renderer: RendererBase | None = None
        self.obs_renderer: RendererBase | None = None
        """Separate renderer for obs["cameras"] (perception). When set, ``renderer``
        is used only for the debug video and ``obs_renderer`` feeds the policy."""
        self.obs_render_enabled = True
        """Whether the observation keeps ``obs["cameras"]`` fresh. See :attr:`frames`."""
        self.frames: dict[str, dict[str, np.ndarray]] = {}
        self.video_frames: dict[str, dict[str, np.ndarray]] = {}
        """Frames from the observer renderer when it differs from obs_renderer.
        Registered as obs["video_cameras"] so the recorder can use them."""
        self._fallback_said = False
        if obs_renderer and self._cameras:
            self.obs_renderer = self._make_renderer(obs_renderer)
        if renderer and self._cameras:
            self.renderer = self._make_renderer(renderer)
        self._still_renderer: RendererBase | None = None
        self._still_key: tuple | None = None

        self._register_obs_providers()

    # ------------------------------------------------------------------ what it is
    @property
    def request(self):
        """What this scene was compiled from: a ``BuildRequest``, or a ``TaskSpec``."""
        return self.scene.request

    @property
    def action_spec(self) -> ActionSpec:
        """How an action is interpreted.

        ``getattr`` rather than a protocol member, because
        :class:`~robosimstudio.scene.request.SceneRequest` is exactly what the
        *compiler* reads and this is read by a *driver* of the result. A request
        that names none gets ``ActionSpec``'s own defaults, which its docstring
        says are this package's rather than any benchmark's.
        """
        return getattr(self.request, "action", None) or ActionSpec()

    @property
    def num_worlds(self) -> int:
        return int(self.scene.num_worlds)

    @property
    def platform(self) -> Platform:
        """Where this scene runs and what draws it, and what the build adapted -- one line when printed.

        The compiler's answer (:attr:`CompiledScene.platform`); every renderer
        this object builds is chosen through it.
        """
        return platform_of(self.scene)

    # ------------------------------------------------------------------ build
    def _resolve_cameras(self) -> tuple[CameraSpec, ...]:
        """The rig with every mounted camera bound to a real body.

        A :class:`~robosimstudio.core.cameras.CameraSpec` states a *role*
        (``parent_frame``) because a scene is written before a model exists; a
        renderer needs a body *index*. This is where the two meet, once per
        environment, so the renderer and the observation both see one resolved
        rig rather than resolving it twice and possibly differently.

        An arm role wins, then a body **label**. The order is the point: a role
        is what an eye-in-hand camera means and survives a change of robot,
        while a label is what a camera screwed to one particular link means and
        does not. Both exist on real rigs, and a benchmark whose overhead camera
        is bolted to a pan-tilt head has no role to name it by.

        Raises:
            ValueError: If a camera names neither an arm this scene has nor a
                body this model carries. A mounted camera silently degrading to
                a world-fixed one at the origin renders the inside of the floor,
                which is a long way from its cause.
        """
        resolved: list[CameraSpec] = []
        for spec in getattr(self.request, "cameras", ()):
            if not spec.parent_frame:
                resolved.append(spec)
                continue
            body = -1
            try:
                body = int(self.scene.handles.arm(spec.parent_frame).ee_body)
            except Exception as arm_exc:
                from robosimstudio.scene.entity import label_index

                try:
                    body = label_index(list(self.model.body_label), spec.parent_frame)
                except Exception as body_exc:
                    raise ValueError(
                        f"camera {spec.name!r} rides {spec.parent_frame!r}, which is neither an arm of "
                        f"{self.request.full_name!r} on embodiment {self.request.embodiment!r} ({arm_exc}) "
                        f"nor a body in its model ({body_exc})"
                    ) from body_exc
            resolved.append(replace(spec, parent_body=body, parent_frame=""))
        return tuple(resolved)

    def _make_controllers(self):
        """Every control mode this scene supports, for switching between.

        All of them, not the one the request names: they are thin wrappers over
        the same IK and gripper objects, so the set costs one construction each,
        and a script that wants a planned joint transit followed by an OSC pick
        needs both live at once. ``control_mode`` picks which starts active.

        The :class:`~robosimstudio.specs.ActionSpec` is folded in **per mode**,
        never as shared keyword arguments: handing ``convention=`` to the
        joint-space class is a ``TypeError``. A bad option raises here; only a
        mode the embodiment cannot do is left out.
        """
        embodiment = self.scene.embodiment
        if embodiment is None or not embodiment.iks:
            return {}
        names = list(embodiment.iks)
        arms = [(embodiment.iks[name], embodiment.grippers.get(name)) for name in names]
        action = self.action_spec
        request = self.request
        frame_bodies = (
            [int(embodiment.arms[name].base_body) for name in names] if action.ee_delta_frame == "base" else []
        )
        if action.ee_delta_frame == "base" and any(b < 0 for b in frame_bodies):
            raise ValueError(
                f"scene {request.full_name!r} states base-frame ee_delta actions, but embodiment "
                f"{request.embodiment!r} has no base body for "
                f"{[n for n, b in zip(names, frame_bodies, strict=True) if b < 0]} (the spec names no "
                "ArmSpec.base_body_name and the first joint hangs on the world). Deltas would silently be "
                "read in world axes, which on a yawed mount drives the arm the wrong way on every step."
            )
        options = {
            "ee_delta": {
                "pos_scale": action.ee_delta_pos_scale,
                "rot_scale": action.ee_delta_rot_scale,
                "convention": action.ee_delta_convention,
                "gripper_range": action.gripper_range,
                "use_target": action.ee_delta_use_target,
                "gripper_delta": action.gripper_delta,
                **({"frame_bodies": frame_bodies} if frame_bodies else {}),
            },
            "osc_pose": {"gripper_range": action.gripper_range},
        }
        spaces = make_action_spaces(arms, control_hz=request.physics.control_hz, options=options)
        return spaces

    def _make_contact_probe(self):
        """A contact probe, unless the scene cannot support one.

        A scene whose MuJoCo solver does not own its collision pipeline has no
        geom table to map through, and a caller may legitimately not need
        contacts -- so this returns ``None`` rather than raising, and the
        predicates that need it fail closed.
        """
        try:
            from robosimstudio.core.probes import ContactProbe

            solver = self.scene.program.mujoco_solver if self.scene.program is not None else None
            if solver is None:
                raise RuntimeError("no MuJoCo island in this scene's physics")
            return ContactProbe(solver, self.model, self.scene.pipeline.sim_dt, world=self.scene.handles.world)
        except Exception as exc:
            if self.verbose:
                print(f"[sim] no contact probe ({type(exc).__name__}: {exc}); contact predicates will read False")
            return None

    def make_renderer(
        self,
        backend: str,
        *,
        cameras: Sequence[CameraSpec] | None = None,
        render: RenderSpec | None = None,
        **kwargs: Any,
    ) -> RendererBase:
        """A renderer over this scene, for a caller that owns its lifecycle.

        The scene-derived arguments a backend needs are supplied here rather
        than by everyone who builds one: the tiled backend's ``shape -> label``
        table and the visual backends' part list both come off the compiled
        scene, and a caller that forgets them gets segmentation in Newton's own
        shape numbering, or a photograph of an empty world. Explicit ``kwargs``
        still win, so a caller may override either.

        The returned renderer is **not** attached to this object and is not
        closed by :meth:`close` -- that is the point of it being public. A live
        view in a browser, an export through another backend and the
        observation rig are three renderers over one model with three different
        lifetimes.

        Args:
            backend: One of :func:`~robosimstudio.solvers.render.list_renderers`,
                or any dotted path to a renderer class.
            cameras: Which cameras, defaulting to the scene's own rig. A live
                preview passes ``[c.scaled(w, h) for c in ...]``.
            render: How to light the scene, in terms every backend understands.
                ``None`` -- the default -- takes :attr:`look`, which is the
                scene's own stated lighting if it has one and the house look
                otherwise. It no longer means "leave each backend on its own":
                that spread was 52/66/102/56 in mean frame brightness across four
                backends drawing one scene, and a caller who genuinely wants it
                asks for it by passing a bare ``RenderSpec()``.
            **kwargs: Forwarded to the backend, including ``channels``.

        ``backend`` may also be ``"auto"`` or ``""``, for the platform's pick; a
        name is checked against this scene's device first, so ``rtx`` on a CPU
        fails by name rather than inside the path tracer.
        """
        backend = self.platform.renderer_for(backend)[0]
        render = self.look if render is None else render
        specs = list(self.cameras if cameras is None else cameras)
        for key, provide in self._scene_facts(specs).items():
            if key in _backend_class(backend).wants and key not in kwargs:
                value = provide()
                if value is not None:
                    kwargs[key] = value
        return create_renderer(backend, self.model, specs, render=render, **kwargs)

    def render(
        self,
        camera: CameraSpec | str | None = None,
        *,
        channels: Sequence[str] = ("rgb",),
    ) -> dict[str, np.ndarray]:
        """One frame of the scene as it stands, through the renderer.

        The renderer is the :attr:`platform`'s: ``ROBOSIMSTUDIO_RENDERER``
        where it names one, else path traced (``rtx``) -- the room, the
        materials and the lights -- where ovrtx is installed and this scene's
        GPU has RT cores, and drawn from the colliders alone (``tiled``) where
        not, a CPU included, with one printed line saying why. The
        renderer is built on the first call and kept for the next one with the
        same camera, so a loop that photographs every few steps pays for the
        stage once.

        Args:
            camera: A :class:`~robosimstudio.specs.CameraSpec`, the name of one
                the scene declares, or ``None`` for the scene's first camera --
                or, on a scene that declares none, a three-quarter view of the
                work surface.
            channels: Which of ``"rgb"``, ``"depth"`` to produce.

        Returns:
            ``{"rgb": uint8 [H, W, 3], "depth": float32 [H, W], ...}``.

        Example::

            frame = sim.render()
            imageio.imwrite("still.png", frame["rgb"])
        """
        from robosimstudio.solvers.render import FALLBACK

        spec = self._camera_for(camera)
        backend, why = self.platform.renderer_for("")
        key = (backend, spec, tuple(channels))
        if self._still_renderer is None or self._still_key != key:
            if self._still_renderer is not None:
                self._still_renderer.close()
            self._say_fallback(backend, why)
            self._still_renderer = self.make_renderer(backend, cameras=[spec], channels=list(channels))
            self._still_key = key
            if backend != FALLBACK:
                self._still_renderer.render(self.state)  # the first frame only builds the stage
        self.sync()
        frame = self._still_renderer.render(self.state)[spec.name]
        return {c: np.asarray(frame[c]) for c in channels if c in frame}

    def record(
        self,
        path: str,
        steps: int,
        *,
        camera: CameraSpec | str | None = None,
        fps: int = 30,
        every: int = 1,
    ) -> str:
        """Step the scene and write an mp4 of it, through the renderer.

        Every ``every``-th control step is photographed as :meth:`render`
        photographs one, and the frames go to ``<path>/<camera>.mp4`` at
        ``fps``. The simulation is stepped here, so what the video shows is
        what :meth:`run` would have done.

        Args:
            path: A directory for the video; made if it does not exist.
            steps: How many control steps to run.
            camera: As :meth:`render` takes it.
            fps: Frame rate of the file.
            every: Photograph one control step in this many.

        Returns:
            The directory written to.
        """
        from robosimstudio.core.render.video import VideoSet

        spec = self._camera_for(camera)
        with VideoSet(path, fps=int(fps)) as videos:
            for i in range(int(steps)):
                self.run(1)
                if i % max(1, int(every)) == 0:
                    videos.append_frames({spec.name: self.render(spec)})
        return str(path)

    def _camera_for(self, camera: CameraSpec | str | None) -> CameraSpec:
        """The camera :meth:`render` photographs through."""
        if isinstance(camera, CameraSpec):
            return camera
        if isinstance(camera, str) and camera:
            for spec in self.cameras:
                if spec.name == camera:
                    return spec
            raise KeyError(f"no camera named {camera!r}; the scene declares {[c.name for c in self.cameras]}")
        if self.cameras:
            return self.cameras[0]
        # A three-quarter view of the work surface, in world coordinates.
        z = float(getattr(self.scene.handles, "surface_z", 0.0) or 0.0)
        return CameraSpec(
            name="view",
            width=960,
            height=540,
            fov_y=45.0,
            eye=(-0.4, -0.9, z + 0.75),
            look_at=(0.6, 0.0, z + 0.05),
            up=(0.0, 0.0, 1.0),
            near=0.01,
            far=8.0,
        )

    def _scene_facts(self, specs: list[CameraSpec]) -> dict[str, Any]:
        """Everything a renderer may ask this scene for, by the name it asks under.

        A renderer is handed a model and a camera list, so anything it needs
        *about* the compiled scene has to arrive as a keyword. Which facts a
        backend takes is the backend's own statement -- ``RendererBase.wants``
        -- and this is the other half: one provider per name, each a closure so
        nothing is computed for a backend that did not ask.
        """

        def cell() -> Any:
            from robosimstudio.scene.registry import find_cell

            return find_cell(self.scene.handles.scene_name)

        handles = self.scene.handles
        return {
            "shape_labels": handles.segmentation_labels,
            "finishes": lambda: handles.finishes,
            "deformables": lambda: handles.deformables,
            "deformable_finishes": lambda: handles.deformable_finishes,
            "deformable_colors": self._deformable_colors,
            "live_shapes": lambda: handles.live_shapes,
            "surface_z": lambda: handles.surface_z,
            "cell": cell,
        }

    def _deformable_colors(self) -> dict[str, tuple[float, float, float]]:
        """``deformable name -> display RGB``, off each spec's own ``color``, else its material's.

        A particle body's spec says ``color=None`` for "the material's" (rice is
        rice-coloured unless the scene says otherwise), so the entity's resolved
        material is the second place to look; a body with neither is left out
        and the renderer's default applies.
        """
        out: dict[str, tuple[float, float, float]] = {}
        for name in self.scene.handles.deformables:
            entity = self.scene.by_name.get(name)
            color = getattr(getattr(entity, "spec", None), "color", None)
            if color is None:
                color = getattr(getattr(entity, "material", None), "color", None)
            if color is not None:
                out[name] = tuple(float(v) for v in color)
        return out

    @property
    def render_skip(self) -> tuple[str, ...]:
        """Entity names a visual-matching composite supplies from a photograph.

        An entity whose camera has a :attr:`~robosimstudio.core.cameras.CameraSpec.background`
        may legitimately be *drawn by the photograph* rather than by simulation
        -- SimplerEnv's sink is physics-only and comes from the picture.

        Read off a typed field when the request has one. A typed field means
        the next visual-matching suite declares it rather than smuggling it.
        """
        return tuple(str(name) for name in getattr(self.request, "render_skip", ()))

    @property
    def look(self) -> RenderSpec:
        """How this scene is lit, for whichever backend is asked.

        The scene's own statement if it made one, else the house look. Read here
        rather than at each call site because the three display paths -- the
        studio's live camera panel, its video export, and the offline re-render
        -- each built a renderer without passing one, so a task that *did* state
        its lighting was honoured in the observation rig and ignored in every
        picture a person actually looked at. Resolving it on the scene means
        those three need no argument rather than needing the right one.
        """
        return look_for(getattr(self.request, "render", None))

    def _observable_rig(self, backend: str) -> tuple[CameraSpec, ...]:
        """The rig, narrowed to the channels this backend can actually produce.

        A camera declares what it wants photographed and a backend declares what
        it can photograph, and the two do not have to agree: the path tracer has
        no segmentation, so a tool-sorting agentview -- which asks for rgb,
        depth and seg -- makes :class:`RendererBase` raise, and one camera
        wanting one channel one backend lacks costs the whole rig.

        Refusing is right for a renderer somebody built by hand, and wrong here.
        This one is built from a *name*, on the way to constructing an
        environment, and there is no second chance to negotiate -- so it narrows,
        which is what lets a run collected with segmentation be re-rendered by a
        path tracer at all.

        And it says what it dropped, out loud rather than under ``verbose``,
        because this changes what a policy sees: a channel that quietly stopped
        appearing in ``obs["cameras"]`` is indistinguishable from a backend
        rendering it badly. Whether the loss matters is the caller's to judge,
        and it is not always a loss: ``seg`` is the channel this actually drops
        today, and a non-privileged graph policy may not read segmentation
        anyway -- it is on that contract's list of ground truth.
        """
        from robosimstudio.solvers.render import supported_channels

        supported = set(supported_channels(backend))
        rig, dropped, empty = [], {}, []
        for cam in self.cameras:
            kept = tuple(c for c in cam.channels if c in supported)
            if kept == tuple(cam.channels):
                rig.append(cam)
                continue
            dropped[cam.name] = sorted(set(cam.channels) - supported)
            if kept:
                rig.append(replace(cam, channels=kept))
            else:
                # Nothing left to photograph, so it leaves the rig rather than
                # staying on as an entry that renders no channels -- a camera a
                # backend cannot produce anything for is not a camera, and the
                # path tracer refuses to author a render product for one.
                empty.append(cam.name)
        if dropped and not self._narrowed_reported:
            # Once per environment, not once per rebuild: an episode that needs a
            # different model rebuilds the rig, and a benchmark run over fifty of
            # them would otherwise say this fifty times.
            self._narrowed_reported = True
            asked = sorted({c for names in dropped.values() for c in names})
            gone = f"; {sorted(empty)} produce nothing at all and are not rendered" if empty else ""
            print(
                f"[sim] the {backend!r} rig produces {sorted(supported)}, so {asked} is missing from "
                f"obs['cameras'] for {sorted(dropped)}{gone}"
            )
        return tuple(rig)

    def _say_fallback(self, backend: str, why: str) -> None:
        """Say once per ``Sim`` that pictures are colliders only, and why."""
        from robosimstudio.solvers.render import ENV, FALLBACK

        if backend == FALLBACK and not self._fallback_said:
            self._fallback_said = True
            # No override to offer on a CPU: there is nothing there rtx could run on.
            quiet = why.startswith(ENV) or why == "asked for" or self.platform.is_cpu
            hint = "" if quiet else f" ({ENV}=rtx to override)"
            print(f"[sim] rendering with {FALLBACK}, the colliders alone: {why}{hint}")

    def _make_renderer(self, backend: str) -> RendererBase:
        """The observation rig, which this object owns and closes.

        The scene's own lighting is passed unless the caller stated one: on a
        visual-matching task the rig is part of the observation rather than a
        display setting, so a policy evaluated here sees the reference's own
        exposure and not the backend's default.
        """
        # ``auto`` is resolved here, on this scene's platform; a name is taken
        # as given, so ``rtx`` without ovrtx raises rather than quietly
        # becoming a different picture.
        backend, why = self.platform.renderer_for(backend)
        self._say_fallback(backend, why)
        kwargs = dict(self.renderer_kwargs)
        render = kwargs.pop("render", None)
        kwargs.setdefault("cameras", self._observable_rig(backend))
        return self.make_renderer(backend, render=render, **kwargs)

    def _register_obs_providers(self) -> None:
        """What every scene can be asked. ``Env`` adds the instruction on top."""
        self.add_obs_provider("proprio", lambda sim: sim._observe_proprio())
        self.add_obs_provider("objects", lambda sim: sim._observe_objects())
        if any(e.carries_particles for e in self.scene.by_name.values()):
            self.add_obs_provider("deformables", lambda sim: sim._observe_deformables())
        self.add_obs_provider("step", lambda sim: sim.step_count)
        self.add_obs_provider("sim_time", lambda sim: sim.sim_time)
        if self.obs_renderer is not None or self.renderer is not None:
            self.add_obs_provider("cameras", lambda sim: sim.frames)
        if self.obs_renderer is not None and self.renderer is not None:
            self.add_obs_provider("video_cameras", lambda sim: sim.video_frames)

    def reset_scene(self) -> None:
        """Put every entity, and the solver under them, back where the scene says.

        **Here rather than on ``Env``, and the warm-up is why.** Nothing below is
        about a task: an entity re-places the coordinates it owns, the
        articulation goes back to what this run started from, and the rigid
        solver's warm-start buffers are cleared. ``EnvBase.reset`` calls this
        twice on purpose -- before the warm-up and after -- because
        ``warmup_and_capture`` steps real physics; while it lived on ``Env`` a
        bare ``Sim`` inherited the no-op stub and got neither call, so a cube
        released 200 mm over a bench came back from ``build()`` already 5.6 mm
        down with ``sim_time`` reading 0.0.

        The entities alone are not the whole of it, and the gap is what made
        consecutive episodes in one process drift. Measured on
        a tool-sorting scene, six scripted rollouts of episode 0 through one
        reused ``Env``: lengths 785, 786, 787, 787, 788, 789 where three
        separate processes each gave 785. Two things were carrying over.

        **The articulation.** An entity restores the coordinates it owns -- an
        object its free joint -- but a robot whose spec leaves
        :attr:`~robosimstudio.specs.EmbodimentSpec.reset_joint_q` off
        writes only its servo *targets*, deliberately, and no shipped embodiment
        turns it on. So the arm and the fingers began each episode wherever the
        last one abandoned them: 3.4e-2 rad away, the fingers left open by a
        release where episode 0 started them nearly shut. Restoring
        :attr:`_start_joint_q` -- the whole vector, before the entities run, so
        each of them re-places its own coordinates for *this* episode on top --
        gives that back without teleporting anything on a single-episode run,
        since the vector is the pose that run itself started from.

        **The rigid solver's own buffers.** MuJoCo warm-starts every solve from
        the last one's accelerations and holds its applied forces, none of which
        lives in a Newton :class:`~newton.State`: ``qacc_warmstart`` came into
        each episode still holding the previous one's value (max 2.3 after a
        rollout, against 0 in a fresh process). Newton's own
        :meth:`~newton.solvers.SolverBase.reset` clears exactly those buffers,
        and ``flags=0`` asks for that and nothing else -- the joint coordinates
        are this method's to restore, not the solver's to overwrite with the
        model defaults.

        Zeroing ``joint_qd`` moved ahead of the kinematics for the same reason:
        ``eval_fk`` derives ``body_qd`` *from* ``joint_qd``, so zeroing after it
        left every body carrying the previous episode's momentum inside a state
        whose joint rates read zero.
        """
        import newton

        self._refuse_layout(self.episode)
        if self._start_joint_q is not None:
            self.pipeline.state_0.joint_q.assign(self._start_joint_q)
            self.pipeline.state_1.joint_q.assign(self._start_joint_q)
        # Before the entities, not after, and the difference is not cosmetic. A
        # host engine's reset returns its island to the rest it captured, and an
        # entity's reset is where this episode's state is placed on top -- so a
        # deformable an episode MOVES (a surface-preparation foam, drawn somewhere new
        # every episode) was placed and then restored straight back to the
        # nominal. Invisible from Newton, which kept the placed particles until
        # the bridge's next tick wrote the engine's own nodes over them:
        # measured at 56 mm of disagreement between the two while every
        # particle-side assertion passed.
        #
        # Nothing here depends on the other order. The reset clears solver
        # buffers -- warm starts, contact anchors -- and the state is not final
        # at this point either way: the kinematics below still have to run.
        self.pipeline.reset_solvers(flags=0)
        for entity in self.scene.entities:
            entity.reset(self.pipeline, self.episode)
        state = self.pipeline.state_0
        # A deformable-only scene has no rigid state to restore; the entities'
        # own reset above is the whole of it. Nor has one whose bodies have no
        # joint -- a static box (``add_link``) beside a sheet, no robot -- whose
        # ``joint_qd`` is ``None``: what follows restores joint coordinates and
        # the bodies they move, which a jointless body is not in a scene with a
        # robot either (``c5b_drape.py``).
        if not self.model.body_count or not self.model.joint_count:
            self.pipeline.mark_dirty()
            return
        # After the entities, which have just re-placed every object for this
        # episode: what the episode does not run is now moved out of the way
        # and held there. Before the kinematics, so ``body_q`` agrees.
        self._park_inactive()
        self.pipeline.state_0.joint_qd.zero_()
        self.pipeline.state_1.joint_qd.zero_()
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        import warp as wp

        wp.copy(self.pipeline.state_1.body_q, state.body_q)
        # Velocities as well as poses: ``state_1`` was the only half of the
        # double buffer still holding the previous episode's ``body_qd`` after
        # everything else matched, and a buffer that is half restored is the
        # thing this method exists to not leave behind.
        wp.copy(self.pipeline.state_1.body_qd, state.body_qd)
        wp.copy(self.pipeline.state_1.joint_q, state.joint_q)
        self.pipeline.mark_dirty()

    def _refuse_layout(self, episode, scene=None) -> None:
        """Refuse ``episode`` if it stands a static object anywhere but where this model holds it.

        Every entity is asked (:meth:`~robosimstudio.domains.rigid.entity.EntityRigid.move_refusal`)
        before anything moves. :meth:`reset_scene` placed entity by entity and a static object's own
        reset refused only when its turn came, so a free cube declared before a static block had gone
        to the refused episode's x = 0.70 when the block said no (``e3_partial_reset.py``).
        ``Env.reset`` asks this too, before it takes the episode up, of the new ``scene`` it built
        for the episode when that needs another model (this one's when omitted).

        Raises:
            ValueError: The first object the layout moves, where, and the way out.
        """
        for entity in (self.scene if scene is None else scene).entities:
            ask = getattr(entity, "move_refusal", None)
            refusal = ask(episode) if callable(ask) else ""
            if refusal:
                raise ValueError(f"{refusal}. {entity.move_remedy}")

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Restore the start state, then record what "start" turned out to be.

        Both captures are facts about a *scene*: the articulation a run began at,
        and the poses things settled into. ``Env.reset`` picks an episode and
        then calls this.
        """
        obs, info = super().reset()
        # counted, so a reader that sampled step_count before can tell a reset from a pause, however many
        # steps followed it (the live page's takes and routines count their steps across one)
        self.reset_count += 1
        self._warned_nonfinite = False  # the finite flags were just cleared too
        if self._start_joint_q is None and self.model.joint_count:
            # After the base class has warmed up *and* re-posed, so this is the
            # articulation the first episode is actually about to be stepped
            # from -- which is what every later episode has to be given back.
            # A scene with no joint has no ``joint_q`` at all, hence the guard:
            # a deformable-only one, or a static box beside a sheet.
            self._start_joint_q = np.asarray(self.pipeline.snapshot.joint_q, dtype=np.float32).copy()
        self.settled_body_q = self._body_q().copy()
        return obs, info

    # ------------------------------------------------------------------- the loop
    def run(self, count: int = 1) -> Sim:
        """Advance ``count`` control steps with no action. Returns ``self``, so it chains.

        Separate from :meth:`step`, which takes an *action*: a scene with no arm
        has nothing to command, and ``sim.step(180)`` reading 180 as an action
        vector is the kind of mistake a shared name invites. The per-step
        bookkeeping is the same as :meth:`step`'s -- the step count and
        :attr:`post_step_hooks` -- so a hook a scene installs runs here too.

        **Says so once if the state stops being finite.** A diverged run looks
        exactly like a fast one -- it steps, it returns, and every number after it
        is ``nan`` -- and the caller most likely to hit it is the one this method
        is for: somebody stepping a scene they assembled themselves, with no
        evaluator watching. The check is the sticky flag the step's own read-back
        already set, so it costs no extra copy, and it prints rather than raises
        because probing an unstable configuration on purpose is a real thing to
        do. ``sim.state_is_finite()`` is the same answer as a boolean.
        """
        for _ in range(int(count)):
            self._step_once()
        if not self._warned_nonfinite and not self.state_is_finite():
            self._warned_nonfinite = True
            print(
                f"[sim] the state stopped being finite by step {self.step_count}; every reading from here "
                "is nan. Usually a solver that cannot hold this arrangement -- check "
                "sim.pipeline.program.problems(), and GENERIC's docstring for what it cannot hold."
            )
        return self

    def sync(self) -> Sim:
        """Wait for the device. Returns ``self``.

        Needed before *reading*, not before stepping: the substep block is
        queued, and a host read of ``body_q`` that skips this gets whatever the
        device had finished writing, which is usually right and occasionally a
        step behind.
        """
        self.pipeline.sync()
        return self

    @property
    def time(self) -> float:
        """Simulated seconds since the build. ``sim_time`` under a shorter name."""
        return float(self.sim_time)

    def deformables(self) -> dict[str, Any]:
        """Every entity carrying a particle probe, by name: :func:`scene_deformables` of this scene."""
        return scene_deformables(self.scene)

    def particle_probes(self) -> dict[str, Any]:
        """Those entities' probes, which is what a measure is handed."""
        return {name: e.probe for name, e in self.deformables().items()}

    def make_context(self) -> WorldState:
        """Gather this step's read-back, once.

        The same :class:`~robosimstudio.scene.state.WorldState` a benchmark
        measure is handed, which is the point: a measurement written against a
        bare scene is a measurement a task can reuse.
        """
        return WorldState(
            # Restricted to the episode's active set, so a goal, a tool or a
            # describer enumerating objects never finds a parked one. The
            # whole table when the episode names none, which is the same
            # object ``scene.handles`` is.
            handles=self.handles,
            state=self.state,
            step=self.step_count,
            sim_time=self.sim_time,
            body_q=self._body_q(),
            body_qd=self._body_qd(),
            settled_body_q=self.settled_body_q,
            joint_target_q=self._joint_target_q(),
            contacts=restrict_contacts(self.contacts, self._parked_shapes()),
            active=self.active,
            # Every entity that carries a probe, not every entity in one *bin*.
            # The bin a spec's ``stage`` puts it in is a build-order fact;
            # whether it has particles to measure is a fact about the thing.
            particles=self.particle_probes(),
            # Cloths first, extras after: both are "what the scene's own
            # entities published", keyed by entity name.
            entities=dict(self.scene.by_name),
        )

    def read(self) -> WorldState:
        """:meth:`make_context`, synced first. What a caller outside a step loop wants."""
        self.sync()
        return self.make_context()

    # -------------------------------------------------------------- observation
    def _observe_proprio(self) -> dict[str, np.ndarray]:
        return {} if self.scene.embodiment is None else self.scene.embodiment.observe(self.state)

    def _body_q(self) -> np.ndarray:
        """Rigid poses, or an empty ``[0, 7]`` for a scene with no bodies.

        A deformable-only scene -- a material rig -- leaves ``state.body_q`` as
        ``None`` rather than an empty array, and every consumer here wants an
        array it can index into and get nothing from.
        """
        if not self.model.body_count:
            return np.zeros((0, 7), dtype=np.float64)
        return self.pipeline.snapshot.body_q

    def _body_qd(self) -> np.ndarray:
        """Rigid velocities, or an empty ``[0, 6]``."""
        if not self.model.body_count:
            return np.zeros((0, 6), dtype=np.float64)
        return self.pipeline.snapshot.body_qd

    def _joint_target_q(self) -> np.ndarray:
        """Servo targets, or an empty array for a scene with no articulation."""
        targets = self.pipeline.control.joint_target_q
        if targets is None:
            return np.zeros(0, dtype=np.float64)
        return self.pipeline.snapshot.joint_target_q

    def _observe_objects(self) -> dict[str, np.ndarray]:
        body_q = self._body_q()
        return {
            name: np.asarray(handle.pose(body_q), dtype=np.float32)
            for name, handle in self.scene.handles.objects.items()
        }

    def _observe_deformables(self) -> dict[str, dict[str, float]]:
        """What each particle-carrying entity reports about itself.

        Keyed on :meth:`deformables` -- the entities that actually carry a probe
        -- and not on the ``deformables`` **bin**, which is where a spec's
        ``stage`` put it.

        ``observe`` is reached duck-typed because carrying particles and
        reporting about yourself are different claims, and an entity may make
        the first without the second.
        """
        readback = self.pipeline.snapshot
        seen = {}
        for name, entity in self.deformables().items():
            observe = getattr(entity, "observe", None)
            if callable(observe):
                seen[name] = observe(self.state, self.step_count, readback=readback)
        return seen

    # ------------------------------------------------------------------ access
    @property
    def handles(self):
        """Where everything in the scene is -- for the objects this episode runs.

        :meth:`~robosimstudio.scene.handles.SceneHandles.restricted` to the
        episode's :attr:`~robosimstudio.specs.Episode.active` set, which for
        every episode that names none is ``scene.handles`` itself. The full
        table stays on ``scene.handles`` for what is about the *model* -- the
        renderer's finishes and labels, a camera's parent body -- and this is
        what a reader asking "which objects are there" gets. Cached per active
        set, since the observation asks every step.
        """
        active = self.active
        if self._restricted is None or self._restricted[0] != active:
            self._restricted = (active, self.scene.handles.restricted(active))
        return self._restricted[1]

    @property
    def active(self) -> frozenset[str] | None:
        """Which entities the current episode runs, or ``None`` for every one."""
        return getattr(self.episode, "active", None)

    @property
    def parked(self) -> tuple[str, ...]:
        """The objects the current episode parks -- floated out of the way, held, touching nothing."""
        return self._parked

    def _parked_shapes(self) -> list[int]:
        """Every shape of every parked object, for the contact view to drop."""
        if not self._parked:
            return []
        objects = self.scene.handles.objects
        return [int(s) for name in self._parked if name in objects for s in objects[name].shapes]

    def _park_inactive(self) -> None:
        """Move what this episode does not run to its park pose and hold it there.

        The serial half of the fleet's ``park``,
        on this scene's one world, through the same
        :class:`~robosimstudio.scene.parking.GravcompParking` -- so a superset
        base run serially behaves as it does in a wave, and the reference the
        fleet is measured against is the same mechanism and not a different
        one. Called from :meth:`reset_scene` after every entity has re-placed
        itself and before the kinematics, so ``body_q`` agrees with the joint
        coordinates written here.

        Nothing to do for an episode that names no active set, which is every
        episode of every document written before superset instances existed:
        no pose is written, no table is built, and the gravcomp array is never
        touched.
        """
        names = parked_objects(self.scene, self.active)
        was = self._parked
        self._parked = tuple(names)
        if not names and not was:
            return
        parking = self._make_parking()
        if names:
            poses = scene_park_poses(self.scene)
            q = np.asarray(self.pipeline.state_0.joint_q.numpy(), dtype=np.float32).copy()
            for name in names:
                entity = self.scene.objects[name]
                start = int(entity.coord_start)
                q[start : start + 7] = poses[name].astype(np.float32)
            self.pipeline.state_0.joint_q.assign(q)
            self.pipeline.state_1.joint_q.assign(q)
        parking.set(0, [int(self.scene.objects[name].body) for name in names])
        parking.apply()

    def _make_parking(self) -> GravcompParking:
        if self._parking is None:
            solver = self.scene.program.mujoco_solver if self.scene.program is not None else None
            if solver is None:
                raise RuntimeError(
                    f"scene {self.request.full_name!r} runs an episode with an active set, but its physics has no "
                    "MuJoCo island to hold a parked body against gravity with; parking needs one"
                )
            self._parking = GravcompParking(solver, self.model, self.scene.worlds)
        return self._parking

    def gravcomp(self, name: str) -> float:
        """The solver's gravity-compensation row for one object, read back. 1.0 while parked."""
        return self._make_parking().row(0, int(self.scene.objects[name].body))

    @property
    def clearance(self):
        """How close things are, at any configuration, without stepping.

        Built on first ask and kept, because building one walks the model's
        collision filtering and that is not work to repeat per question.

        **The distinction from ``contacts`` is the useful one.** A contact probe
        reports what the solver found *this* step: it answers "what am I
        touching". This answers "what would I touch", at a joint configuration
        the arm is not in and has no intention of entering -- which is the
        question an agent has before it commits to a motion, and the one that
        was previously unanswerable without moving and looking.
        """
        if self._clearance is None:
            from robosimstudio.core.probes.clearance import ClearanceProbe

            self._clearance = ClearanceProbe(self.scene.index_model)
        return self._clearance

    @property
    def cameras(self) -> tuple[CameraSpec, ...]:
        """The rig, with every eye-in-hand camera bound to the body it rides."""
        return self._cameras

    def arm(self, name: str = ""):
        """One arm's IK, defaulting to the first."""
        iks = self.scene.embodiment.iks if self.scene.embodiment else {}
        if not iks:
            raise RuntimeError(f"scene {self.request.full_name!r} has no arms")
        return iks[name] if name else next(iter(iks.values()))

    def gripper(self, name: str = ""):
        grippers = self.scene.embodiment.grippers if self.scene.embodiment else {}
        if not grippers:
            raise RuntimeError(f"scene {self.request.full_name!r} has no grippers")
        return grippers[name] if name else next(iter(grippers.values()))

    def suction(self, name: str = ""):
        """One arm's vacuum cup (:class:`~robosimstudio.core.suction.Suction`), defaulting to the first."""
        suctions = self.scene.embodiment.suctions if self.scene.embodiment else {}
        if not suctions:
            raise RuntimeError(f"scene {self.request.full_name!r} has no suction hand")
        if name and name not in suctions:
            raise KeyError(f"no suction hand on arm {name!r}; the cups are on {sorted(suctions)}")
        return suctions[name] if name else next(iter(suctions.values()))

    def __getitem__(self, name: str) -> Ref:
        """The ref for one thing, by name.

        Available for things the *scene's own import* declared as well as things
        you added -- a merged MJCF export brings its own bodies, and a caller who
        wants one of them has a name and nothing else.
        """
        ref = self.refs.get(name)
        if ref is not None:
            return ref
        made = _ref_for(self.scene, name)
        if made is None:
            raise KeyError(f"nothing called {name!r} in this scene; there is {sorted(self.names())}")
        self.refs[name] = made
        return made

    def names(self) -> list[str]:
        """Every name in the scene, sorted: every name ``sim[name]`` answers.

        The handle tables as well as the entities, because :meth:`__getitem__`
        resolves through both -- a region, or a body a merged import brought, has
        a handle and no entity.
        """
        scene = self.scene
        handles = scene.handles
        found = set().union(handles.objects, handles.deformables, handles.statics, handles.regions)
        return sorted(set(self.refs) | set(scene.by_name) | found)

    def __contains__(self, name: str) -> bool:
        return name in self.names()

    def close(self) -> None:
        """Release the observation rig, if this object built one."""
        if self.obs_renderer is not None:
            self.obs_renderer.close()
            self.obs_renderer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self._still_renderer is not None:
            self._still_renderer.close()
            self._still_renderer = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}({getattr(self.request, 'full_name', '?')!r}, {len(self.names())} things, t={self.time:.3f}s)"


def scene_deformables(scene: Any) -> dict[str, Any]:
    """Every entity of ``scene`` carrying a particle probe, by name, in build order: what "a deformable" means.

    The one rule, asked of the scene so that whatever holds one shares it: :meth:`Sim.deformables`,
    the privileged ``sim.get_particle_state`` and ``sim.attach`` (which reach it through one world
    of a fleet, a ``WorldView`` with no ``deformables`` of its own), and every studio panel that
    lists, pins, settles or reads one. Three copies of it disagreed: ``sim.attach`` with no entity
    took the task's first ``deformables``-stage spec, and a cuttable is staged there, so a tofu
    declared before a towel had the hand pin the eight corners of its cage and report ``attached``
    while it held nothing.

    The deformables bin **and** the extras one, because which of the two a thing lands in is decided
    by its spec's ``stage`` -- a build-order statement -- and not by whether it has particles. A rope
    is the case that separates them: as a capsule chain it is an ``extras`` entity with no particles
    at all, and as a centerline it is an ``extras`` entity with nothing but. And not every particle
    carrier: a cuttable owns eight particles, the corners its cage writes back for a page to draw the
    cut by, with no probe to summarise them and nothing a pin on them moves.
    """
    return {n: e for n, e in getattr(scene, "by_name", {}).items() if getattr(e, "probe", None) is not None}


def _ref_for(scene: Any, name: str) -> Ref | None:
    """The ref class the entity under ``name`` declares, instantiated -- or ``None``.

    ``Entity.ref`` rather than a table here, for the reason that ClassVar exists:
    which accessors a thing offers is a fact about the thing.
    """
    entity = scene.by_name.get(name)
    if entity is not None:
        return type(entity).ref(scene, name)
    if scene.handles.find(name) is not None:
        return Ref(scene, name)
    return None
