# SPDX-License-Identifier: Apache-2.0
"""The task's cameras, live in the page.

Separate from :mod:`~robosimstudio_studio.panels` because it is the one
scene panel that owns a *renderer* rather than a scene node, and the lifetime
rules that follow from that -- release on a model swap, release when the box is
unticked, release on close -- are the bulk of the code. The rest of the panels
put a gizmo in the viser scene graph and take it down again.

**This is opt-in, and that is the whole design.** Export stays deferred: what a
video wants is the trajectory you already watched, rendered afterwards, larger,
through a better backend, and without costing the loop anything while it ran
(see :class:`~robosimstudio_studio.controls.PanelVideo`). What a *live*
view is for is the other question -- whether the rig a policy is about to be
scored through is pointed at anything, whether the depth channel has the object
in it, whether a backend swap changed what the pixels say. That question is
worth frames, and only while it is being asked. So the checkbox starts off, the
budget bounds what it takes when it is on, and the readout says what it is
actually costing.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from robosimstudio.core.math import xform_from_matrix
from robosimstudio.core.render.live import RenderBudget
from robosimstudio.core.render.video import colorize_depth, colorize_labels
from robosimstudio.solvers.render import default_renderer

from .panels import Panel

__all__ = ["SOURCE_OBS", "PanelCamera"]

FRUSTUM_ROOT = "/studio/cameras"
"""Scene-tree prefix the frustums live under, so they clear in one call."""

FRUSTUM_COLOR = (90, 130, 200)
FRUSTUM_COLOR_SELECTED = (250, 170, 60)
"""Which frustum ``Set from view`` and ``Reset pose`` would act on."""

FRUSTUM_SCALE = 0.12
"""How far a frustum stands off its camera [m]. Small on purpose: these sit in
a tabletop scene a metre across, and viser's 0.3 default puts the image plane
through the bench the camera is looking at."""

SOURCE_OBS = "observation"
"""Backend entry meaning "do not build one -- show the frames the environment
already rendered for the policy".

It sits in the same dropdown as the real backends because from the page's side
it is the same question -- what am I looking through -- and splitting it into a
separate ``Source`` control had one concrete cost: on the observation source a
backend cannot be chosen, so the Backend dropdown was *disabled*, and since the
environment ships a renderer by default that is what the page opened on. The
first thing anyone saw was a greyed-out control reading ``tiled``, which reads
as "this build only supports tiled" rather than "you are looking at the policy's
own frames". One list, always live, no dead widget to interpret.
"""


class _Fixed:
    """A widget-shaped constant: the source the inset renders through, which nobody picks."""

    def __init__(self, value: str) -> None:
        self.value = value
        self.disabled = False


class PanelCamera(Panel):
    """What the task's cameras see, live in the page, through any backend.

    A live view renders on the loop's own thread, between steps, because that is
    the only place a Warp read is legal: the scene has one writer by design (the
    benchmark's pooled runner, ``run.pool``, says why that is not a limitation to
    be engineered away), and a render thread reading ``state`` while the solver
    writes it is a race whose symptom is a torn frame at best. So every frame
    this draws is time the simulation does not get, and the panel's whole job is
    to spend that honestly:

    **It prefers pixels that have already been paid for.** An environment built
    with a renderer is *already* rendering its rig -- ``Env.evaluate`` does it on
    the ``render_every`` cadence, whenever something is reading the result -- so
    the default source is those frames. Showing them costs one JPEG encode and
    no GPU work at all, and it has the property a studio actually wants: what you
    are looking at is what the policy is looking at, at the resolution and on the
    cadence it sees. ``own renderer`` is the other source, for a backend or a
    resolution the observation rig is not using -- a path-traced look at a scene
    collected through the tiled sensor, say.

    **It renders each state at most once.** The gate is the step counter, not the
    clock alone, so a paused scene costs nothing and no encode is spent on an
    image identical to the one already in the page.

    **It renders the whole rig in one launch.** ``all`` hands every selected
    camera to a single renderer, which for the tiled backend is one megakernel
    over the lot rather than one launch each, and for the path tracer is one
    tracer step over one render product per camera.

    **It bounds its own share of the loop.** :class:`~robosimstudio.core.render.live.RenderBudget`
    gates on measured cost, so a backend that costs more than its share -- a
    path tracer at full resolution, or the seconds its first frame spends
    building an OVRTX scene -- self-limits instead of stalling the session. And
    the readout says what it is actually taking, because a live view that quietly
    halves the step rate is worse than no live view.

    **It leaves the captured graph alone.** Nothing here touches
    ``pipeline.hooks`` or calls ``invalidate_graph``: a render is outside
    :meth:`~robosimstudio.core.pipeline.ScenePipeline.step` entirely, so the
    substep block stays captured and the view costs the loop only its own frames.
    What it does add is one device sync per frame (a readback synchronizes), which
    is the honest reason the gate is a budget rather than a frame rate.

    Args:
        session: The owning session.
        live: Start with the view on.
        backend: :data:`SOURCE_OBS` for the environment's own frames -- used
            when it has a renderer, because those frames are already paid for --
            else the renderer.
        fps: Target frame rate.
        max_load: Largest share of wall clock the view may spend rendering.
        width: Width for the ``own renderer`` source; 0 uses the camera's own.
        height: Height, likewise.
        channels: Which channels to show, subject to what the source produces.
        jpeg_quality: Encode quality for rgb and depth. Fixed at handle
            creation by viser, so changing it rebuilds the images.
        frustums: Draw the camera being rendered in the 3D scene, showing its
            own view. ``all`` draws the whole rig; see below.

    **Where a camera is pointed is edited by looking, not by typing.** A frustum
    in the scene carries that camera's own latest frame, so where it is and what
    it therefore sees are one picture rather than two. Click it and the browser
    flies there, which is the only honest preview of a viewpoint -- an eye and a
    look-at as six numbers are not something anyone can read. ``Set from view``
    then adopts wherever you have flown as that camera's pose, and ``Reset``
    puts the task's declared one back.

    Overrides live on this panel and not in the task: a camera moved here is a
    question being asked of this session ("would the wrist see the tab from
    over here?"), and a task's declared rig is a thing its numbers were measured
    through. Loading another task, or reloading this one, starts from declared.

    **Only the camera being rendered is drawn.** The frustum's whole claim is
    that where a camera is and what it sees are one object; a camera the view is
    not rendering has nothing to put in it but an empty box or the last frame it
    happened to catch, and a rig of four of those sits over a bench a metre
    across, through the arm, in front of the one picture somebody is looking at.
    So the selection in ``Camera`` is what gets a frustum -- ``all`` still draws
    them all, which is exactly when the clutter is the point -- and switching
    the dropdown moves the frustum rather than adding one.
    """

    def __init__(
        self,
        session,
        *,
        live: bool = True,
        backend: str = SOURCE_OBS,
        fps: float = 15.0,
        max_load: float = 0.3,
        width: int = 0,
        height: int = 0,
        channels: Sequence[str] = ("rgb",),
        jpeg_quality: int = 80,
        frustums: bool = True,
    ) -> None:
        super().__init__(session)
        self.names = [c.name for c in self.env.cameras]
        self.poses: dict[str, Any] = {}
        """Session-local pose overrides, by camera name. Empty means declared."""
        self.show_frustums = bool(frustums)
        self._frusta: dict[str, Any] = {}
        self._frustum_key: tuple | None = None
        self._frustum_pushed: dict[str, np.ndarray] = {}
        """Per frustum, the exact rgb array whose encode is already in the page.

        viser's ``handle.image`` setter JPEG-encodes synchronously on the
        caller's thread -- this thread -- every time it is assigned, whether or
        not the pixels changed. The frames dict is replaced wholesale by each
        render and its arrays are never mutated in place, so identity is a
        complete change test: assigning only when the array is a different
        object skips the encode exactly when it would reproduce the
        byte-identical JPEG already there. Measured before this existed: one
        encode per camera per control step against one render per five steps.
        """
        self._frustum_tick: int | None = None
        """Step of the last full frustum update; None forces the next one."""
        self._frustum_selected: str = ""
        self._frustum_shown: tuple[str, ...] = ()
        """Which cameras the last frustum pass drew, so a change of selection
        is due immediately rather than on the draw cadence."""
        self._frustum_force = False
        """A pose gesture landed: update the frustums now, not on the cadence."""
        self._fly_to: Any = None
        self._adopt_from: Any = None
        self._reset_pose = False
        self._clicked: str = ""
        self._last_frames: dict[str, dict[str, np.ndarray]] = {}
        """The most recent render, so a frustum keeps its picture between frames."""
        self.live = bool(live)
        self.backend = str(backend)
        self.width, self.height = int(width), int(height)
        self.channels = tuple(channels)
        self.jpeg_quality = int(jpeg_quality)
        self.budget = RenderBudget(target_fps=fps, max_load=max_load)
        # The environment's own frames when it has a rig to read, else the
        # renderer. Not a choice: a page shows what the scene is photographed
        # as, and a picture is made with one renderer.
        device = getattr(getattr(self.env, "model", None), "device", None)
        self.backends = ((SOURCE_OBS,) if self.env.renderer is not None else ()) + (default_renderer(device),)
        self.backend = self.backend if self.backend in self.backends else self.backends[0]

        self._renderer = None
        self._built: tuple | None = None
        self._model: Any = None
        self._images: dict[str, Any] = {}
        self._drawn_step = -1
        self._live = self._backend = None

    # ------------------------------------------------------------- the cameras
    def specs(self) -> list[Any]:
        """The task's cameras with this session's pose overrides applied.

        Everything that renders or draws goes through here rather than through
        ``env.cameras``, so a camera moved in the page is moved for the live
        view, the frustums and the readout at once -- a preview showing the
        declared pose while the render used the edited one would be the whole
        feature failing silently.
        """
        return [self.poses.get(c.name, c) for c in self.env.cameras]

    def _spec(self, name: str) -> Any:
        return next((c for c in self.specs() if c.name == name), None)

    def _pose_key(self) -> tuple:
        """What a renderer would have to be rebuilt for, pose-wise."""
        return tuple(sorted((n, tuple(c.eye), tuple(c.look_at)) for n, c in self.poses.items()))

    # ------------------------------------------------------------------- build
    def build(self, gui) -> None:
        """What gets touched at the top, and everything else one fold down.

        What is left at the top is what gets touched: is it on, which camera,
        through what, is it drawn in the scene, and the two buttons that aim it
        -- flying to a camera and taking the view you flew to are the whole
        gesture this panel exists for. Resolution, frame budget, channels and
        the readouts are all real, and all things you set once or read when
        something looks wrong -- so they are behind ``Detail`` rather than in
        front of the images every session.
        """
        first = self.env.cameras[0]
        self._folder = gui.add_folder("Cameras")
        with self._folder:
            self._live = gui.add_checkbox("Live render", initial_value=self.live)
            self._camera = gui.add_dropdown("Camera", options=("all", *self.names), initial_value=self.names[0])
            self._backend = _Fixed(self.backend)
            self._frustums = gui.add_checkbox("Camera gizmo", initial_value=self.show_frustums)
            self._frustums.on_update(lambda _: setattr(self, "show_frustums", bool(self._frustums.value)))
            look = gui.add_button("Look through")
            look.on_click(self._on_look)
            adopt = gui.add_button("Set from view")
            adopt.on_click(self._on_adopt)

            with gui.add_folder("Detail", expand_by_default=False):
                reset = gui.add_button("Reset pose")
                reset.on_click(lambda _: setattr(self, "_reset_pose", True))
                self._pose = gui.add_text("Pose", initial_value="", disabled=True)
                self._width = gui.add_number("Width", initial_value=self.width or first.width, min=32, step=16)
                self._height = gui.add_number("Height", initial_value=self.height or first.height, min=32, step=16)
                self._fps = gui.add_number(
                    "Target fps", initial_value=float(self.budget.target_fps), min=1.0, max=120.0
                )
                self._load = gui.add_slider(
                    "Max load", min=0.05, max=1.0, step=0.05, initial_value=self.budget.max_load
                )
                self._rgb = gui.add_checkbox("rgb", initial_value="rgb" in self.channels)
                self._depth = gui.add_checkbox("depth", initial_value="depth" in self.channels)
                self._seg = gui.add_checkbox("seg", initial_value="seg" in self.channels)
                self._rate = gui.add_text("Rendered", initial_value="off", disabled=True)
                self._state = gui.add_text("Through", initial_value=self._describe_rig(), disabled=True)
        self._sync_disabled()
        self._refresh_pose_readout()

    # --------------------------------------------------------- pose, by looking
    def _on_look(self, event) -> None:
        """Fly the clicking browser to the selected camera. viser thread."""
        self._fly_to = (self._selected_name(), getattr(event, "client", None))

    def _on_adopt(self, event) -> None:
        """Take the clicking browser's viewpoint as the selected camera's pose."""
        self._adopt_from = (self._selected_name(), getattr(event, "client", None))

    def _selected_name(self) -> str:
        """Which camera the pose buttons act on.

        The ``Camera`` dropdown, except on ``all`` -- where there is no one
        answer, so the last frustum clicked is it, and failing that the first
        camera. Aiming a rig of two while looking at both is a real thing to
        want; guessing which one silently is not.
        """
        selected = str(self._camera.value)
        if selected != "all":
            return selected
        return self._clicked or self.names[0]

    def _rendered_names(self) -> tuple[str, ...]:
        """Which cameras the view is rendering -- and so which ones are drawn.

        The ``Camera`` dropdown, read the way :meth:`_config` reads it: ``all``
        is the whole rig, anything else is that one camera and nothing else.
        """
        selected = str(self._camera.value)
        return tuple(self.names) if selected == "all" else (selected,)

    def _refresh_pose_readout(self) -> None:
        name = self._selected_name()
        spec = self._spec(name)
        if spec is None:
            self._pose.value = ""
            return
        edited = name in self.poses
        eye = ", ".join(f"{v:.2f}" for v in spec.eye)
        self._pose.value = f"{name}: ({eye})" + ("  edited" if edited else "  declared")

    def _describe_rig(self) -> str:
        cams = self.env.cameras
        return f"{len(cams)} camera{'s' if len(cams) != 1 else ''}: {', '.join(c.name for c in cams)}"

    def _sync_disabled(self) -> None:
        """Grey out what the source does not use.

        Only the resolution: the environment's rig renders at the size the task
        declared, so a Width here would silently do nothing, and a knob that
        silently does nothing is how a reader concludes the panel is broken.
        """
        own = str(self._backend.value) != SOURCE_OBS
        for widget in (self._width, self._height):
            widget.disabled = not own

    @property
    def wants_obs(self) -> bool:
        """Whether this panel is reading the environment's own camera rig.

        The rig only *runs* while something reads it -- that is what keeps an
        idle scene from rendering two cameras every fourth step for nobody --
        and :attr:`~robosimstudio_studio.session.StudioSession.wants_pixels`
        is where that is decided. So this panel has to declare itself, or the
        observation source shows whatever frames happened to be left in
        ``env.frames`` from the last policy run: an image that is stale rather
        than absent, which is the worse of the two failures.

        Read before ``build`` (the session composes panels before it builds the
        GUI) it is ``False``, which is correct -- nothing is displayed yet, and
        both widgets are ``None`` until then.
        """
        if self._backend is None:
            return False
        return self._live is not None and bool(self._live.value) and str(self._backend.value) == SOURCE_OBS

    # -------------------------------------------------------------------- loop
    def drain(self) -> None:
        """Render if one is due and the scene has moved. Main thread.

        In ``drain`` rather than ``refresh`` so the gate is checked every control
        step: a session refreshing its readouts every fourth step would cap the
        view at a quarter of the control rate, and the whole point of a
        wall-clock budget is that it does not depend on the loop's cadence.
        """
        # Pose edits first: they change what a render would produce, so servicing
        # them after one would put a frame from the old viewpoint in the page and
        # leave it there until the next gate opened.
        prof = self.session.profile
        with prof.phase("iter.drain.camera.pose"):
            self._drain_pose()
        # Then the frustums, *before* any of the early returns below. They answer
        # "where are the cameras", which is a question the live view being off or
        # a frame not being due does not suspend -- and hiding them has to take
        # effect on the drain the box was unticked, not on the next render.
        with prof.phase("iter.drain.camera.frustums"):
            self._drain_frustums()
        if not self._live.value:
            if self._renderer is not None or self._images:
                self._release()
                self._rate.value = "off"
            return

        self.budget.target_fps = float(self._fps.value)
        self.budget.max_load = float(self._load.value)
        # One read of the widgets per drain: viser writes them from its own
        # threads, so reading them twice can hand two halves of this method
        # different configurations.
        config = self._config()
        if config != self._built:
            # Before the gate, not after it: a rebuild resets the budget, so
            # switching away from a backend that renders every two seconds does
            # not make you wait two seconds to see that it worked.
            self._sync_disabled()
            if not self._prepare(config):
                return
        elif self._frame_step(config) == self._drawn_step:
            return  # this state is already in the page
        if not self.budget.due(time.perf_counter()):
            return

        started = time.perf_counter()
        try:
            with prof.phase("iter.drain.camera.frames"):
                frames = self._frames(config)
        except Exception as exc:
            self._fail(f"render failed -- {type(exc).__name__}: {exc}")
            return
        if frames:
            with prof.phase("iter.drain.camera.publish"):
                self._publish(frames)
            # Again, so this render's pixels reach the frustums now rather than
            # on the next drain. Forced past the cadence gate: these frames are
            # new by construction, and the whole point of drawing a frustum is
            # that its picture is the render that just happened.
            self._last_frames = frames
            with prof.phase("iter.drain.camera.frustums_after_render"):
                self._drain_frustums(force=True)
        self._drawn_step = self._frame_step(config)
        self.budget.record(started=started, cost=time.perf_counter() - started)

    def _drain_pose(self) -> None:
        """Service the three pose gestures. Main thread."""
        if self._reset_pose:
            self._reset_pose = False
            if self.poses.pop(self._selected_name(), None) is not None:
                self._built = None  # the renderer holds the old viewpoint
                self._frustum_force = True  # the frustum holds the old pose
            self._refresh_pose_readout()

        if self._adopt_from is not None:
            name, client = self._adopt_from
            self._adopt_from = None
            spec = self._spec(name)
            if client is not None and spec is not None:
                # viser's client camera reports exactly the pair CameraSpec wants,
                # in the same world frame, so this is a copy rather than a
                # conversion -- and the look-at is what makes it one: an eye and a
                # quaternion would need the basis argument that the frustums get
                # right only because both sides are +z-forward.
                eye = tuple(float(v) for v in client.camera.position)
                target = tuple(float(v) for v in client.camera.look_at)
                self.poses[name] = spec.with_pose(eye, target)
                self._built = None
                self._frustum_force = True
                self._refresh_pose_readout()
                self.session.status = f"{name} aimed from the browser view"

        if self._fly_to is not None:
            name, client = self._fly_to
            self._fly_to = None
            spec = self._spec(name)
            if client is not None and spec is not None:
                # Assigning both is deliberate: viser derives the orientation from
                # look_at, and setting position alone swings the target with it.
                client.camera.position = tuple(float(v) for v in spec.eye)
                client.camera.look_at = tuple(float(v) for v in spec.look_at)
                self._clicked = name
                self._refresh_pose_readout()

    def _drain_frustums(self, force: bool = False) -> None:
        """Keep the in-scene frustums matching the cameras. Main thread.

        Each carries its own latest rgb frame, so where a camera is and what it
        therefore sees are one object in the page rather than two things to hold
        side by side.

        Two economies, both measured before they were written, because this
        used to run its whole body every control step and that body was 32% of
        a tool-sorting scene's loop period:

        * **The update is on the draw cadence, not the step cadence.** A
          frustum is scenery: nobody can see it move faster than the browser
          redraws, and the arm it may be mounted on only redraws every
          ``render_every`` steps itself, so updating its pose more often than
          the robot's own mesh bought nothing and cost a ``body_q`` readback
          plus three viser messages per camera per step. A pose gesture or a
          fresh render still updates immediately -- ``force`` -- because those
          are the moments a person is looking at the frustum in particular.
        * **The image is assigned only when it is a different array.** The
          setter encodes a JPEG on this thread every time it is assigned, and
          between renders the frame in ``_last_frames`` is the same object it
          was -- see :attr:`_frustum_pushed`.

        The hide-on-untick path stays per-step: unticking the box must take
        effect on the drain it happened, not up to ``render_every`` steps
        later.
        """
        if not self.show_frustums:
            if self._frusta:
                self._clear_frustums()
                self._frustum_tick = None
            return
        selected = self._selected_name()
        shown = self._rendered_names()
        due = (
            force
            or self._frustum_force
            or self._frustum_tick is None
            or selected != self._frustum_selected
            or shown != self._frustum_shown
            or not self._frusta
        )
        self._frustum_force = False
        step = int(self.env.step_count)
        if not due:
            every = max(1, int(self.session.render_every))
            due = step != self._frustum_tick and step % every == 0
        if not due:
            return
        self._frustum_tick = step
        self._frustum_selected = selected
        self._frustum_shown = shown
        specs = self.specs()
        body_q = self.env.state.body_q.numpy() if self.env.state.body_q is not None else None
        for spec in specs:
            if spec.name not in shown:
                # Hidden rather than removed: coming back costs the JPEG encode
                # of its last frame again, and a dropdown is flicked back and
                # forth. The pose is left stale too -- nothing can see it.
                handle = self._frusta.get(spec.name)
                if handle is not None:
                    handle.visible = False
                continue
            matrix = spec.pose_on_body(body_q)
            xform = xform_from_matrix(matrix)
            position = tuple(float(v) for v in xform[:3])
            # xform_from_matrix returns xyzw; viser wants wxyz. Both sides are
            # already +z-forward/+y-down, so nothing else is converted.
            wxyz = (float(xform[6]), float(xform[3]), float(xform[4]), float(xform[5]))
            image = self._last_frames.get(spec.name, {}).get("rgb")
            handle = self._frusta.get(spec.name)
            if handle is None:
                handle = self.server.scene.add_camera_frustum(
                    f"{FRUSTUM_ROOT}/{spec.name}",
                    fov=math.radians(float(spec.fov_y_from_intrinsics)),
                    aspect=float(spec.width) / float(max(1, spec.height)),
                    scale=FRUSTUM_SCALE,
                    color=FRUSTUM_COLOR_SELECTED if spec.name == selected else FRUSTUM_COLOR,
                    image=np.ascontiguousarray(image) if image is not None else None,
                    position=position,
                    wxyz=wxyz,
                )
                handle.on_click(self._on_frustum_click(spec.name))
                self._frusta[spec.name] = handle
                if image is not None:
                    self._frustum_pushed[spec.name] = image
                continue
            handle.visible = True
            handle.position, handle.wxyz = position, wxyz
            handle.color = FRUSTUM_COLOR_SELECTED if spec.name == selected else FRUSTUM_COLOR
            if image is not None and image is not self._frustum_pushed.get(spec.name):
                handle.image = np.ascontiguousarray(image)
                self._frustum_pushed[spec.name] = image

    def _on_frustum_click(self, name: str):
        """A click on a frustum selects that camera and looks through it."""

        def _clicked(event) -> None:
            self._clicked = name
            self._fly_to = (name, getattr(event, "client", None))

        return _clicked

    def _clear_frustums(self) -> None:
        for handle in self._frusta.values():
            try:
                handle.remove()
            except Exception:
                pass
        self._frusta = {}
        self._frustum_pushed = {}
        self._frustum_shown = ()

    def _frame_step(self, config: tuple) -> int:
        """A number that changes exactly when new pixels become available.

        The observation rig renders on its own ``render_every`` cadence, so
        between its renders the step counter moves while the frames do not --
        and an encode of an image already in the page spends the budget on
        nothing. A paused scene lands on the same value every drain, which is
        what makes a paused view free.
        """
        step = int(self.env.step_count)
        if config[0] == SOURCE_OBS:
            every = max(1, int(self.env.render_every))
            return step - step % every
        return step

    def _config(self) -> tuple:
        """Everything a rebuild depends on, as one comparable value."""
        selected = str(self._camera.value)
        names = tuple(self.names) if selected == "all" else (selected,)
        wanted = tuple(
            name for name, widget in (("rgb", self._rgb), ("depth", self._depth), ("seg", self._seg)) if widget.value
        )
        backend = str(self._backend.value)
        if backend == SOURCE_OBS:
            # Resolution and pose belong to the environment's rig, so a change to
            # those widgets must not rebuild anything -- and an edited pose
            # cannot reach frames the environment rendered for itself.
            return (backend, names, wanted)
        return (
            backend,
            names,
            wanted,
            max(32, int(self._width.value)),
            max(32, int(self._height.value)),
            # A renderer is built against camera poses, so an aimed camera is a
            # rebuild rather than a parameter -- which is also what makes
            # ``Set from view`` show up on the very next frame.
            self._pose_key(),
        )

    def _frames(self, config: tuple) -> dict[str, dict[str, np.ndarray]]:
        """This step's frames, from whichever backend is selected."""
        backend, names, wanted = config[:3]
        if backend == SOURCE_OBS:
            frames = self.env.frames or {}
            return {
                name: {c: array for c, array in frame.items() if c in wanted}
                for name, frame in frames.items()
                if name in names
            }
        return self._renderer.render(self.env.state)

    def _prepare(self, config: tuple) -> bool:
        """Make the selected backend current, building a renderer if it needs one."""
        self._release()
        backend, names, wanted = config[:3]
        if not wanted:
            self._state.value = "pick a channel"
            return False
        if backend == SOURCE_OBS:
            self._built, self._model = config, self.env.model
            produced = sorted({c for frame in (self.env.frames or {}).values() for c in frame})
            self._state.value = (
                f"the environment's own {self.env.renderer_name} rig, every {self.env.render_every} steps"
                + (f" -- has {', '.join(produced)}" if produced else "")
            )
            return True

        from robosimstudio.solvers.render import supported_channels

        width, height = config[3], config[4]
        supported = set(supported_channels(backend))
        channels = tuple(c for c in wanted if c in supported)
        dropped = [c for c in wanted if c not in supported]
        if not channels:
            self._fail(f"the {backend!r} backend produces none of {list(wanted)}")
            return False
        cameras = [c.scaled(width, height) for c in self.specs() if c.name in names]
        # A live path-traced frame is a preview, so it takes one tracer pass
        # rather than the six an export converges over -- the flicker that buys
        # is the right trade at 10 fps and the wrong one in a video.
        extra = {"accumulate": 1} if backend == "rtx" else {}
        # No `render=`: the scene's look is resolved by `Sim.look`, so a preview
        # is lit the way the export will be. That is deliberate rather than an
        # omission -- lighting is the one thing a preview must *not* override,
        # since checking framing against a differently-lit picture is checking
        # the wrong picture.
        try:
            self._renderer = self.env.make_renderer(backend, cameras=cameras, channels=channels, **extra)
        except Exception as exc:
            # No ovrtx, no EGL, no display: a bad afternoon, not a dead session.
            self._fail(f"{backend} would not build -- {type(exc).__name__}: {exc}")
            return False
        self._built, self._model = config, self.env.model
        self.budget.reset()
        note = f" (no {', '.join(dropped)})" if dropped else ""
        self._state.value = f"{backend} {width}x{height} {' '.join(channels)}{note}"
        return True

    def _publish(self, frames: dict[str, dict[str, np.ndarray]]) -> None:
        """Push one render into the page: an image per camera and channel."""
        for name, frame in frames.items():
            for channel in ("rgb", "depth", "seg"):
                array = frame.get(channel)
                if array is None:
                    continue
                if channel == "rgb":
                    image = np.ascontiguousarray(array)
                elif channel == "depth":
                    image = colorize_depth(array)
                else:
                    image = colorize_labels(array)
                key = f"{name} {channel}"
                handle = self._images.get(key)
                if handle is not None:
                    handle.image = image
                    continue
                with self._folder:
                    # png for labels: jpeg's ringing invents label colours at
                    # every boundary, which is the one thing a seg view is for.
                    self._images[key] = self.server.gui.add_image(
                        image,
                        label=key,
                        format="png" if channel == "seg" else "jpeg",
                        jpeg_quality=self.jpeg_quality,
                    )

    def refresh(self) -> None:
        if not self._live.value or self._built is None:
            return
        budget = self.budget
        if budget.fps:
            # One line rather than two. The share of wall clock is the number
            # worth reading -- the frame rate on its own does not say whether
            # the view is the reason the scene got slower.
            self._rate.value = (
                f"{budget.fps:.1f} fps  |  {budget.cost * 1000:.0f} ms  |  "
                f"{budget.load * 100:.0f}% of wall clock (cap {budget.max_load * 100:.0f}%)"
            )
        else:
            self._rate.value = f"measuring ({budget.frames} frames)"

    # -------------------------------------------------------------- lifecycle
    def _fail(self, message: str) -> None:
        """Say what went wrong and switch the view off rather than retry it.

        A backend that will not build fails the same way every step, so retrying
        once a frame turns one message into a spinning page and a scrolling
        console. Ticking the box again is the retry.
        """
        self._release()
        self._state.value = message[:200]
        self._live.value = False
        self.session.status = message[:200]
        print(f"[cameras] live render off: {message}")

    def _release(self) -> None:
        """Give back the images and the renderer. Idempotent.

        Not the frustums: they show where the cameras *are*, which a backend
        switch does not change, and ``_prepare`` releases on every config change
        -- so clearing them here would make them blink on each one.
        """
        for handle in self._images.values():
            try:
                handle.remove()
            except Exception:
                pass
        self._images = {}
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._built, self._model, self._drawn_step = None, None, -1
        self.budget.reset()

    def on_reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        del obs, info
        # An episode whose model key differs recompiles the scene, and a renderer
        # holds what it was built against -- BVHs, uploaded meshes, a USD stage.
        # Rebuild when that actually happened, rather than on every reset, since
        # a visual backend's rebuild reloads every mesh off disk.
        if self._built is not None and self.env.model is not self._model:
            self._release()
            # A recompiled scene may have moved the bodies its cameras ride on,
            # and a frustum holding the old model's pose is a wrong picture
            # rather than a stale one.
            self._clear_frustums()
        self._drawn_step = -1

    def close(self) -> None:
        """Release the renderer -- it holds a GL context, a stage or device buffers."""
        super().close()
        self._release()
        self._clear_frustums()
