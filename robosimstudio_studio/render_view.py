# SPDX-License-Identifier: Apache-2.0
"""The viewport itself, path-traced.

viser's client is a rasterizer and will never path-trace. What it *can* do is
draw an image behind the scene, so the page's photorealism comes from the
server: OVRTX renders the browser's own camera and that frame becomes the
viewport, with the studio's gizmos, handles and frustums drawn over it as
ordinary three.js nodes.

Three measurements decide this panel's shape, all taken on the wiping scene:

* **The frame arrives untouched.** A screenshot of the live page against the
  renderer's own PNG is MAE 0.057/255, 98.9 % of pixels bit-exact, brightness
  ratio 0.9997 -- because viser draws a background through a bare
  ``ShaderMaterial`` whose fragment shader is ``gl_FragColor = vec4(color.rgb,
  1.0)``, with no tone-mapping and no colour-space conversion in it.
* **The cost is convergence passes, not pixels.** At ``accumulate=1`` a
  half-scale frame renders in ~25 ms (40 fps) and a full-scale one in ~44 ms; at
  the ``preview`` default of six passes those are 69 ms and 107 ms. So the
  viewport never has to fall back to the raster scene while you move -- it just
  converges less, and sharpens when you stop. ``accumulate`` is read per render,
  which makes it a runtime dial; resolution is not, because the film back is
  authored once and changing it means a new renderer (~1.8 s).
* **The depth channel is not worth sending.** It encodes as a lossless
  three-channel PNG and cost 13 ms a frame, and the measured behaviour is that
  it does not occlude anything -- the overlays draw on top regardless, which is
  what this panel wants. Dropping it took the push to 1.9 ms and a live gizmo
  drag from 12 fps to 15-18.

Two things that are easy to get wrong, and were:

* **Size every render from ``client.camera.image_width/height``.** The canvas
  element is wider than the 3D viewport, because viser's GUI panel floats over
  it -- 1700 against 1404 on a 1700x1000 window. Render at the canvas size and
  the background quad is pillarboxed, which reads as a scene that will not line
  up rather than as a sizing bug.
* **Hide the model, keep the furniture -- except what the frame already has.**
  ``set_global_visibility(False)`` takes the gizmos with it, and a gizmo you
  cannot see is one you cannot aim at. So the pass goes by name, keeping
  ``/studio/`` -- with two exceptions, both raster stand-ins for something the
  path-traced frame contains: the stage slab, which stands in for the hidden
  bench and otherwise covers the mat and both arms, and the drawn room, which
  is 50 nodes of floor, wall and slat wall across the whole frame. The room is
  the one to watch: it arrived after this panel did, under the prefix that means
  "keep it", and drew the raster room over the rendered one. Measured with a
  browser attached, the pass now hides 50 scenery nodes, the stage and 45 model
  nodes, and keeps 8 gizmo nodes and 2 cursors.

The page runs with no path tracer at all -- ``[studio]`` carries viser alone --
so ``auto`` falls back to ``raster`` when ``ovrtx`` will not import, and the
readout says which mode is live and why.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import time
from typing import Any

import numpy as np

from robosimstudio.core.render.live import RenderBudget

from .panels import Panel

__all__ = ["PanelRenderView"]

RTX, RASTER = "rtx", "raster"
"""The two states of the viewport: the path-traced frame, or viser's own scene.

In the page it is one checkbox, **World model**: on is the path tracer, off is
the fast scene. **It starts off** (:attr:`StudioConfig.world_model`). A page
should open at the speed it can always sustain, and a browser waits about a
minute for its first path-traced frame, because the renderer is built per
client; a page that spends its first minute looking unloaded reads as broken.
"""

BUSY_SECONDS = 0.8
"""How long after an interaction the viewport keeps rendering at moving quality.

Long enough to cover the gap between two drag events rather than flickering
between one pass and six in the middle of a gesture."""

KEPT = "/studio"
"""The prefix whose nodes stay drawn over the path-traced frame."""

DROPPED = ("/studio/stage", "/studio/scenery")
"""...except these. See the module docstring.

Both are raster stand-ins for something the path-traced frame already contains,
and with no depth channel an overlay draws on top -- so left visible, the
stand-in stage slab covers the mat and the drawn room covers the rendered one.
The room is the newer of the two and the easier to miss: it arrived with the
scenery work, under the ``/studio`` prefix that otherwise means "furniture, keep
it", and measured 49 nodes of floor, walls and slat wall across the whole frame.
"""


def _tracer_available() -> bool:
    """Whether the path tracer would import, without importing it.

    ``find_spec`` rather than a ``try: import``, because importing ovrtx builds
    CUDA state this panel may never need: a page opened in raster mode should
    cost nothing a page without the extra would not.
    """
    try:
        return importlib.util.find_spec("ovrtx") is not None
    except (ImportError, ValueError):
        return False


class PanelRenderView(Panel):
    """OVRTX renders the browser's own camera; the frame becomes the viewport.

    Owns one renderer per resolution and re-aims it at each connected client's
    camera in turn, so two browsers each get their own view -- at the cost of one
    render apiece, which is why the budget is measured rather than assumed.
    """

    def __init__(
        self,
        session,
        *,
        world_model: bool = False,
        scale: float = 0.5,
        passes: tuple[int, int] = (1, 6),
        fps: float = 30.0,
        max_load: float = 0.9,
    ) -> None:
        super().__init__(session)
        self.mode = RTX if world_model and _tracer_available() else RASTER
        self.scale = float(scale)
        self.passes = (max(1, int(passes[0])), max(1, int(passes[1])))
        # A viewport is allowed almost the whole loop: unlike the camera panel's
        # inset, it *is* what the page shows, so a frame it skips is a frame
        # nobody sees.
        self.budget = RenderBudget(target_fps=fps, max_load=max_load)
        self._renderer: Any = None
        self._built: tuple | None = None
        self._hidden: list[Any] = []
        self._frames = 0
        self._rate = 0.0
        self._cost = 0.0
        self._window = (time.time(), 0)
        self._cam_key: dict[int, tuple] = {}
        self._touched = 0.0
        self._listening = False
        """Whether :meth:`_on_click` is on the server's click list; :meth:`close` takes it off."""
        self._note = ""
        self.clean = False
        self.locked = ""
        self._mode_request: str | None = None
        self._scale_request: float | None = None
        self._passes_request: tuple[int, int] | None = None
        self._clean_request: bool | None = None
        self._locked_request: str | None = None
        self._state: Any = None
        self._readout: Any = None

    # ------------------------------------------------------------------ build
    def build(self, gui) -> None:
        """Register widgets. Main thread."""
        # Open by default, alone among the studio's folds: this is the switch
        # between two quite different pages, and one that has to be hunted for
        # behind a collapsed heading is one nobody knows they have.
        with gui.add_folder("View", expand_by_default=True):
            # One switch rather than a list of backend names: what somebody
            # wants to say here is "show me the world, not the wireframe", and
            # the two names for that were `rtx` and `raster`, which are answers
            # to a question about renderers. Off is plain viser, and plain viser
            # is the fast one -- see the rates in this module's docstring.
            self._state = gui.add_checkbox(
                "World model", initial_value=self.mode == RTX, disabled=not _tracer_available()
            )
            self._state.on_update(lambda _: setattr(self, "_mode_request", RTX if self._state.value else RASTER))
            slider = gui.add_slider("Scale", min=0.15, max=1.0, step=0.05, initial_value=self.scale)
            slider.on_update(lambda _: setattr(self, "_scale_request", float(slider.value)))
            moving = gui.add_number("Passes moving", initial_value=self.passes[0], min=1, max=24, step=1)
            still = gui.add_number("Passes still", initial_value=self.passes[1], min=1, max=24, step=1)
            for widget in (moving, still):
                widget.on_update(lambda _: setattr(self, "_passes_request", (int(moving.value), int(still.value))))
            # Locking to a task camera is the only way to see the picture the
            # *renderer* will produce: a free-flying browser camera is square
            # pixels and a centred principal point, which is all viser's
            # perspective camera can express, so a rig with a calibrated K is
            # approximated rather than reproduced until you lock to it.
            names = ("free", *(c.name for c in self.env.cameras))
            locked = gui.add_dropdown("Lock to", options=names, initial_value="free")
            locked.on_update(
                lambda _: setattr(self, "_locked_request", "" if locked.value == "free" else str(locked.value))
            )
            clean = gui.add_checkbox("Clean", initial_value=self.clean)
            clean.on_update(lambda _: setattr(self, "_clean_request", bool(clean.value)))
            self._readout = gui.add_text("State", initial_value="", disabled=True)
        # A click is the studio's own signal that somebody is working, and the
        # session's copy is consumed by its drain before this panel's runs -- so
        # this panel listens for itself. viser accumulates scene click callbacks
        # rather than replacing them, so the session's picker is untouched --
        # and for the same reason it is added once and taken off in close():
        # the server outlives this panel, and a callback left behind on every
        # adopt would keep each old panel (and the scene it holds) alive.
        # Callback-safe: it assigns a float and nothing else.
        if not self._listening:
            self.server.scene.on_click()(self._on_click)
            self._listening = True

    def _on_click(self, event) -> None:
        """Somebody is working. Viser thread -- assign only."""
        self._touched = time.time()

    # ------------------------------------------------------------------ drain
    def drain(self) -> None:
        """Render the viewport. Main thread, once per control step."""
        self._service_requests()
        if self.mode != RTX:
            self._restore()
            return
        now = time.time()
        if self._interacting():
            self._touched = now
        if not self.budget.due(now):
            return
        clients = list(self.server.get_clients().values())
        if not clients:
            return
        if not self._ensure(clients[0]):
            return
        started = time.time()
        self._renderer.accumulate = self.passes[0] if now - self._touched < BUSY_SECONDS else self.passes[1]
        for client in clients:
            if not self._push(client):
                return
        self._hide_model()
        cost = time.time() - started
        self.budget.record(started=started, cost=cost)
        self._frames += 1
        self._cost = cost
        # Delivered frames per wall second, not 1/cost: the loop steps the sim
        # between renders, so what the renderer can do and what the page gets
        # are different numbers, and the second one is the one you are watching.
        since, count = self._window
        count += 1
        if time.time() - since >= 2.0:
            self._rate = count / (time.time() - since)
            self._window = (time.time(), 0)
        else:
            self._window = (since, count)

    def _service_requests(self) -> None:
        """Act on what the callbacks assigned."""
        if self._mode_request is not None:
            wanted, self._mode_request = self._mode_request, None
            if wanted != self.mode:
                self.mode = wanted
                self._release()
                if self.mode != RTX:
                    self._restore()
        if self._scale_request is not None:
            scale, self._scale_request = self._scale_request, None
            if abs(scale - self.scale) > 1e-6:
                # The film back is authored once, so a new scale is a new
                # renderer -- about 1.8 s, which is why this is a slider you
                # let go of rather than one you sweep.
                self.scale = scale
                self._release()
        if self._passes_request is not None:
            self.passes, self._passes_request = self._passes_request, None
        if self._clean_request is not None:
            clean, self._clean_request = self._clean_request, None
            if clean != self.clean:
                # Restore before re-hiding: what is kept changes, so the set
                # this panel is holding down is no longer the set it took.
                self._restore()
                self.clean = clean
                if self.mode == RTX:
                    self._hide_model()
        if self._locked_request is not None:
            locked, self._locked_request = self._locked_request, None
            if locked != self.locked:
                # A task camera has its own resolution and film back, so the
                # renderer it needs is not the one the viewport was using.
                self.locked = locked
                self._release()

    def _interacting(self) -> bool:
        """Whether somebody is moving the camera or dragging something.

        Three signals, any of which means *moving quality*: a client's camera
        moved since the last drain, a gizmo is mid-gesture, or a click landed
        inside :data:`BUSY_SECONDS`. The camera check doubles as the thing that
        keeps the render aimed where the browser is looking.
        """
        moved = False
        for client in self.server.get_clients().values():
            key = self._camera_key(client)
            if self._cam_key.get(client.client_id) != key:
                self._cam_key[client.client_id] = key
                moved = True
        return moved or self._dragging()

    def _dragging(self) -> bool:
        """Whether any panel's gizmo is mid-gesture.

        Asked of the panels rather than tracked here: the handle sets belong to
        the object and cloth panels, the cursor to each arm panel, and this one
        has no business owning a second copy of either.
        """
        for panel in getattr(self.session, "panels", ()) or ():
            if getattr(panel, "dragging", False):
                return True
            handles = getattr(getattr(panel, "handles", None), "handles", ())
            if any(getattr(handle, "dragging", False) for handle in handles):
                return True
        return False

    @staticmethod
    def _camera_key(client) -> tuple:
        try:
            camera = client.camera
            return (
                *np.round(np.asarray(camera.position), 5),
                *np.round(np.asarray(camera.look_at), 5),
                round(float(camera.fov), 6),
                int(camera.image_width),
                int(camera.image_height),
            )
        except Exception:
            return ()

    # --------------------------------------------------------------- renderer
    def _locked_camera(self):
        """The task camera the viewport is locked to, or ``None`` when free."""
        if not self.locked:
            return None
        return next((c for c in self.env.cameras if c.name == self.locked), None)

    def _ensure(self, client) -> bool:
        """Build a renderer for this viewport's size, if it needs one."""
        locked = self._locked_camera()
        try:
            if locked is not None:
                # The camera's own shape, not the browser's: locking to a rig
                # camera is asking for *its* picture, letterboxed into whatever
                # window you happen to have open.
                width = max(64, int(locked.width * self.scale))
                height = max(64, int(locked.height * self.scale))
            else:
                width = max(64, int(int(client.camera.image_width) * self.scale))
                height = max(64, int(int(client.camera.image_height) * self.scale))
        except Exception:
            return False
        key = (width, height, self.env.model, self.locked)
        if self._built == key and self._renderer is not None:
            return True
        self._release()
        spec = self._spec(client, width, height)
        if spec is None:
            return False
        try:
            # rgb alone: the depth channel costs 13 ms a frame to encode and
            # composites nothing. accumulate is set per frame in `drain`.
            self._renderer = self.env.make_renderer("rtx", cameras=[spec], channels=["rgb"], accumulate=self.passes[0])
            self._renderer.render(self.env.state)
        except Exception as exc:  # no ovrtx, no GPU, a contended card
            self.mode = RASTER
            self._note = f"rtx would not build -- {type(exc).__name__}: {exc}"
            self._release()
            self._restore()
            return False
        self._built, self._note = key, ""
        self.budget.reset()
        self._hide_model()
        return True

    def _spec(self, client, width: int, height: int):
        """This client's camera as a :class:`CameraSpec` the tracer can render.

        Square pixels and a centred principal point, because that is the only
        frustum viser's ``(fov_y, aspect)`` camera can express -- ``spec.scaled``
        would scale ``fx`` by width and ``fy`` by height independently and leave
        the two views disagreeing by a few per cent.

        Locked to a rig camera this does not apply: that picture is the
        renderer's own, calibrated ``K``, film back and all, and the browser is
        only where it is displayed.
        """
        locked = self._locked_camera()
        if locked is not None:
            return locked.scaled(width, height)
        base = next(iter(self.env.cameras), None)
        if base is None:
            self._note = "this scene declares no camera to borrow a spec from"
            return None
        camera = client.camera
        eye = np.asarray(camera.position, dtype=np.float64)
        look = np.asarray(camera.look_at, dtype=np.float64)
        up = np.asarray(camera.up_direction, dtype=np.float64)
        return dataclasses.replace(
            base,
            name="viewport",
            width=width,
            height=height,
            fx=None,
            fy=None,
            cx=None,
            cy=None,
            fov_y=float(np.rad2deg(float(camera.fov))),
            xform=None,
            parent_body=-1,
            parent_frame="",
            eye=tuple(float(v) for v in eye),
            look_at=tuple(float(v) for v in look),
            up=tuple(float(v) for v in up),
            channels=("rgb",),
        )

    def _push(self, client) -> bool:
        """Render this client's camera and make the frame its viewport."""
        width, height = self._built[0], self._built[1]
        spec = self._spec(client, width, height)
        if spec is None:
            return False
        self._renderer.cameras[0] = spec
        try:
            frames = self._renderer.render(self.env.state)
        except Exception as exc:
            self.mode = RASTER
            self._note = f"rtx stopped rendering -- {type(exc).__name__}: {exc}"
            self._release()
            self._restore()
            return False
        name = "viewport" if self._locked_camera() is None else self.locked
        rgb = np.ascontiguousarray(frames[name]["rgb"][..., :3]).astype(np.uint8)
        if self._locked_camera() is not None:
            rgb = self._letterbox(rgb, client)
        client.scene.set_background_image(rgb, format="jpeg", jpeg_quality=80)
        return True

    @staticmethod
    def _letterbox(rgb: np.ndarray, client) -> np.ndarray:
        """Pad a frame to the viewport's shape rather than letting it stretch.

        viser draws a background across the whole frustum, so an image of
        another aspect is *stretched*, not letterboxed -- which would quietly
        make a locked 4:3 rig camera lie about what it sees in a 16:9 window.
        Padding is the honest version: the picture keeps its shape and the
        window keeps its size.
        """
        try:
            want = float(client.camera.image_width) / float(client.camera.image_height)
        except Exception:
            return rgb
        height, width = rgb.shape[:2]
        have = width / height
        if abs(have - want) < 1e-3:
            return rgb
        if have > want:  # too wide for the window: pad top and bottom
            full = int(round(width / want))
            pad = np.zeros((full, width, 3), dtype=rgb.dtype)
            top = (full - height) // 2
            pad[top : top + height] = rgb
            return pad
        full = int(round(height * want))  # too tall: pad left and right
        pad = np.zeros((height, full, 3), dtype=rgb.dtype)
        left = (full - width) // 2
        pad[:, left : left + width] = rgb
        return pad

    # ------------------------------------------------------------ visibility
    def _hide_model(self) -> None:
        """Hide the raster scene, keep the studio's furniture.

        By name, through viser's own node registry -- the one private surface
        this panel reaches for, because there is no public way to ask what is in
        the scene. A viser that stops carrying the registry leaves the page
        showing its raster scene over the frame, which is ugly and not broken.

        Run every frame rather than on a cadence. Nodes come back visible:
        ``log_instances`` recreates a batch when its instance count changes, and
        the camera panel re-adds its frustums on a schedule of its own. On a
        cadence that showed up as a frustum flashing over a *clean* frame
        between passes, and a dict scan beside a 30 ms render is not worth a bug.
        """
        registry = getattr(self.server.scene, "_handle_from_node_name", None)
        if not registry:
            return
        for name, handle in list(registry.items()):
            if not name or (not self.clean and name.startswith(KEPT) and not name.startswith(DROPPED)):
                continue
            try:
                if handle.visible:
                    handle.visible = False
                    self._hidden.append(handle)
            except Exception:
                pass

    def _restore(self) -> None:
        """Give the raster scene back, and take the frame down. Idempotent."""
        for handle in self._hidden:
            try:
                handle.visible = True
            except Exception:
                pass
        self._hidden = []
        for client in self.server.get_clients().values():
            try:
                client.scene.set_background_image(None)
            except Exception:
                pass

    # ------------------------------------------------------------- readouts
    def refresh(self) -> None:
        """Update read-only text. Main thread."""
        if self._readout is None:
            return
        if self.mode != RTX:
            reason = self._note or ("no ovrtx installed" if not _tracer_available() else "switched off")
            self._readout.value = f"off -- plain viser, {reason}"
            return
        if self._built is None:
            waiting = "waiting for a browser" if not self.server.get_clients() else "building"
            self._readout.value = f"on -- {waiting}"
            return
        passes = self.passes[0] if time.time() - self._touched < BUSY_SECONDS else self.passes[1]
        where = f" on {self.locked}" if self.locked else ""
        clean = ", clean" if self.clean else ""
        self._readout.value = (
            f"on -- {self._built[0]}x{self._built[1]} {passes}p{where}{clean} -- "
            f"{self._rate:.1f} fps, {self._cost * 1000:.0f} ms/frame"
        )

    def on_reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        """A reset may have rebuilt the model, which the renderer's stage is of."""
        if self._built is not None and self._built[2] is not self.env.model:
            self._release()

    def _release(self) -> None:
        """Give back the renderer. Idempotent.

        Explicitly, rather than on garbage collection: ovrtx logs
        ``Renderer destroyed with 1 active binding(s)`` and leaks its step
        result outputs when a renderer goes without being closed, and a page
        that swaps tasks builds one per swap.
        """
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
        self._built = None
        self._frames = 0
        self._window = (time.time(), 0)
        self.budget.reset()

    def close(self) -> None:
        """Release the renderer and give the raster scene back. Main thread."""
        if self._listening:
            self.server.scene.remove_click_callback(self._on_click)
            self._listening = False
        self._release()
        self._restore()
        super().close()
