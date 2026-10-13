# SPDX-License-Identifier: Apache-2.0
"""The browser half of a take: a headless Chromium on the GPU, a visible cursor, and the gestures.

A take is filmed through the studio's own page, so what a reader sees is what
they would do: the pointer goes to the gizmo's arrow, presses, drags, lets go.
Nothing here writes to the session; every change reaches the scene the way a
person's would, through viser's websocket.

**On the GPU, or not at all.** Headless Chromium falls back to SwiftShader
without flags, and a page drawn in software steps at 2-5 frames a second and
looks it. With ANGLE on Vulkan it draws on the NVIDIA card, and :func:`launch`
refuses to go on unless the WebGL renderer string says so. ANGLE takes the first
Vulkan device the loader lists whatever ``CUDA_VISIBLE_DEVICES`` says, so the
browser is also shown only the take's card, through the pass-through layer in
``pickgpu.c``.

**One eased pointer step per film frame.** Every gesture moves the pointer, then
calls ``advance()`` -- which steps the scene by one frame's worth of control
steps and photographs the page (:mod:`.lockstep`) -- so the cursor and the scene
move together whatever the wall clock does. What a person would not see is done
between frames: walking the pointer along an arrow until it lights up, reading
where an arrow is now, landing a slider on its value with the arrow keys.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HELPER_JS = (HERE / "page.js").read_text(encoding="utf-8")

GPU_ARGS = (
    "--use-angle=vulkan",
    "--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan",
    "--ignore-gpu-blocklist",
    "--enable-gpu-rasterization",
)
"""ANGLE on Vulkan: ``ANGLE (NVIDIA, Vulkan 1.4.312 (NVIDIA ... RTX PRO 6000 ...``. Without
these the renderer is SwiftShader."""

CURSOR_JS = """(() => {
  if (document.getElementById('rss-finger')) return;
  const d = document.createElement('div');
  d.id = 'rss-finger';
  d.style.cssText = 'position:fixed;left:-100px;top:-100px;width:26px;height:26px;margin:-13px 0 0 -13px;' +
    'border-radius:50%;background:rgba(30,30,30,0.28);border:2px solid rgba(255,255,255,0.9);' +
    'box-shadow:0 1px 4px rgba(0,0,0,0.4);pointer-events:none;z-index:99999;';
  document.body.appendChild(d);
  addEventListener('pointermove', e => { d.style.left = e.clientX + 'px'; d.style.top = e.clientY + 'px'; }, true);
  addEventListener('pointerdown', () => { d.style.transform = 'scale(0.72)'; d.style.background = 'rgba(30,30,30,0.5)'; }, true);
  addEventListener('pointerup', () => { d.style.transform = ''; d.style.background = 'rgba(30,30,30,0.28)'; }, true);
})()"""
"""A grey finger dot that shrinks while pressed (``~/website_ws/demo/tools/record_preview.py``'s),
without its CSS transitions: a transition runs on the wall clock, and a film frame here
is whatever wall time the physics took, so the press would land in a different frame
on every take."""

QUIET_CSS = ".mantine-Notifications-root, .mantine-Notification-root { display: none !important; }"
"""viser's toasts (a software-WebGL warning, a reconnect notice) are not part of the page."""


def ease(t: float) -> float:
    """Smoothstep: a hand starts and stops gently."""
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


# =============================================================================
# A browser on the take's GPU
# =============================================================================


def _gpu_uuid() -> str:
    """The UUID of the card ``CUDA_VISIBLE_DEVICES`` names first, or ``""`` for the default."""
    want = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if not want:
        return ""
    if want.startswith("GPU-"):
        return want
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True, check=False
    ).stdout
    for line in out.splitlines():
        index, uuid = (part.strip() for part in line.split(",", 1))
        if index == want:
            return uuid
    return ""


def _layer(out: Path) -> Path | None:
    """``pickgpu.c`` built once into ``out`` with its manifest; ``None`` when there is no compiler."""
    source = HERE / "pickgpu.c"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    lib = out / f"libpickgpu-{digest}.so"
    if not lib.is_file():
        cc = shutil.which("cc") or shutil.which("gcc")
        if cc is None:
            return None
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run([cc, "-shared", "-fPIC", "-O2", "-o", str(lib), str(source)], check=True)
    manifest = {
        "file_format_version": "1.1.2",
        "layer": {
            "name": "VK_LAYER_RSS_pick_gpu",
            "type": "GLOBAL",
            "library_path": f"./{lib.name}",
            "api_version": "1.4.0",
            "implementation_version": "1",
            "description": "Show the application only the GPU named by RSS_VK_UUID",
            "functions": {"vkNegotiateLoaderLayerInterfaceVersion": "vkNegotiateLoaderLayerInterfaceVersion"},
            "disable_environment": {"RSS_PICK_GPU_OFF": "1"},
        },
    }
    (out / "pickgpu.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return out


def browser_env(scratch: Path, log: Callable[[str], None]) -> dict[str, str]:
    """The environment that shows Chromium's Vulkan only the take's card."""
    env = dict(os.environ)
    uuid = _gpu_uuid()
    if not uuid:
        return env
    layer = _layer(scratch / ".vkpick")
    if layer is None:
        log("no C compiler for the GPU-pinning layer: the browser draws on the first card the loader lists")
        return env
    env["VK_ADD_IMPLICIT_LAYER_PATH"] = str(layer)
    env["RSS_VK_UUID"] = uuid
    return env


def launch(pw, *, viewport: tuple[int, int], dpr: float, software: bool, scratch: Path, log) -> tuple[Any, Any]:
    """``(browser, context)``: headless Chromium on the GPU, at ``viewport`` CSS pixels and ``dpr``."""
    if software:
        browser = pw.chromium.launch()
    else:
        browser = pw.chromium.launch(args=list(GPU_ARGS), env=browser_env(scratch, log))
    context = browser.new_context(viewport={"width": viewport[0], "height": viewport[1]}, device_scale_factor=dpr)
    return browser, context


# =============================================================================
# The page
# =============================================================================


class Page:
    """One studio page, driven like a person drives it.

    Args:
        page: Playwright's page.
        cdp: A DevTools session on it, for the screenshots.
        advance: Steps the scene one film frame and photographs it. Every eased
            pointer step calls it once.
        sync: ``sync(gizmo)`` returns once the server holds the pose the browser
            shows for that gizmo -- what makes a drag's frame show the scene the
            pointer asked for, not the one 50 ms of viser's throttle earlier.
        where: ``where(gizmo)``, the gizmo's position as the server holds it [m].
        fps: Film frames a second, which is what a gesture's seconds are counted in.
        size: The film frame, in pixels; the screenshot is scaled to it.
    """

    def __init__(self, page, cdp, *, advance: Callable[[], None], sync: Callable[[str], None], where, fps: int, size):
        self.page = page
        self.cdp = cdp
        self.advance = advance
        self.sync = sync
        self.where = where
        self.fps = fps
        self.size = tuple(size)
        vw = page.viewport_size
        self.mouse = (vw["width"] * 0.62, vw["height"] * 0.82)
        self.pressed = False

    # ------------------------------------------------------------------ setup
    def open(self, url: str, timeout: float = 60.0) -> None:
        self.page.goto(url, wait_until="load", timeout=timeout * 1000)
        self.page.wait_for_selector("canvas", timeout=timeout * 1000)
        for _ in range(int(timeout * 10)):
            if self.page.evaluate(HELPER_JS):
                break
            self.page.wait_for_timeout(100)
        else:
            raise RuntimeError("viser never exposed window.__viserMutable")
        self.page.add_style_tag(content=QUIET_CSS)

    def rearm(self) -> None:
        """Re-install ``window.__rss``: harmless, and needed if viser remounted its canvas."""
        self.page.evaluate(HELPER_JS)

    def show_cursor(self) -> None:
        self.page.evaluate(CURSOR_JS)
        self.page.mouse.move(*self.mouse)

    def js(self, method: str, *args):
        """``window.__rss.<method>(*args)``."""
        return self.page.evaluate(f"(a) => window.__rss.{method}(...a)", list(args))

    def wait_frame(self, stamp: str, index: int, timeout: float = 10.0) -> float:
        """Milliseconds until the page shows frame ``index``; raises on a timeout."""
        ms = self.page.evaluate(
            "(a) => window.__rss.waitFrame(a[0], a[1], a[2])", [stamp, int(index), float(timeout * 1000)]
        )
        if ms < 0:
            raise TimeoutError(f"the page did not show frame {index} within {timeout:.0f} s")
        return float(ms)

    def capture(self):
        """The page as a ``PIL.Image`` at the film's size, scaled down with Lanczos.

        ``Page.captureScreenshot`` rather than the screencast: one picture of the
        frame just presented, when asked for, instead of a stream to be matched
        to sim time afterwards.
        """
        from PIL import Image

        shot = self.cdp.send("Page.captureScreenshot", {"format": "png", "optimizeForSpeed": True})
        image = Image.open(io.BytesIO(base64.b64decode(shot["data"]))).convert("RGB")
        if image.size != self.size:
            image = image.resize(self.size, Image.Resampling.LANCZOS)
        return image

    def frames(self, seconds: float) -> int:
        return max(1, round(seconds * self.fps))

    # ------------------------------------------------------------ the pointer
    def _move(self, x: float, y: float) -> None:
        self.page.mouse.move(x, y)
        self.mouse = (x, y)

    def glide(self, x: float, y: float, seconds: float, *, gizmo: str = "") -> None:
        """Ease the pointer to ``(x, y)`` over ``seconds``, one step per frame.

        With ``gizmo`` the button is down on that gizmo, and each step waits for
        the server to hold the new pose before the frame is stepped.
        """
        x0, y0 = self.mouse
        n = self.frames(seconds)
        for i in range(1, n + 1):
            k = ease(i / n)
            self._move(x0 + (x - x0) * k, y0 + (y - y0) * k)
            if gizmo:
                self.sync(gizmo)
            self.advance()

    def down(self) -> None:
        self.page.mouse.down()
        self.pressed = True

    def up(self) -> None:
        self.page.mouse.up()
        self.pressed = False

    def click(self, x: float, y: float, *, approach: float = 0.6) -> None:
        """Glide there, press for a frame, let go."""
        self.glide(x, y, approach)
        self.down()
        self.advance()
        self.up()
        self.advance()

    # ------------------------------------------------------------- the gizmo
    def hovered(self, gizmo: str) -> int:
        return int(self.js("hoveredArrow", gizmo))

    def _settle_hover(self) -> None:
        """Two animation frames: drei repaints a hovered cone on React's next commit."""
        self.page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")

    def grip(self, gizmo: str, index: int, arrow: dict, approach: float) -> tuple[float, float] | None:
        """Put the pointer on arrow ``index`` and see it light up; ``None`` if it never does.

        The hit cylinder is a few pixels wide and the rotation arcs and plane
        sliders sit a few pixels from it, so a press one pixel off grabs a
        rotation. The glide there is filmed; the walk along the arrow that
        finds the pixel drei agrees is on it is not.
        """
        (ox, oy), (tx, ty) = arrow["origin"], arrow["tip"]
        left, top, width, height = self.js("canvasRect")
        mx, my = (ox + tx) / 2, (oy + ty) / 2
        if not (left <= mx <= left + width and top <= my <= top + height):
            raise RuntimeError(
                f"{gizmo}: arrow {index} is outside the 3D view at ({mx:.0f}, {my:.0f}); frame the camera on it"
            )
        nx, ny = -(ty - oy), tx - ox
        norm = math.hypot(nx, ny) or 1.0
        candidates = [
            (ox + (tx - ox) * k + nx / norm * side, oy + (ty - oy) * k + ny / norm * side)
            # Middle out, then towards the cone: an arm's gizmo is depth tested, and a
            # hand in front of the arrow's middle can leave only its far end to press.
            for k in (0.6, 0.55, 0.65, 0.5, 0.7, 0.45, 0.75, 0.4, 0.8, 0.85, 0.9, 0.35, 0.3)
            for side in (0, -2, 2, -4, 4)
        ]
        self.glide(*candidates[0], approach)
        for x, y in candidates:
            self._move(x, y)
            self._settle_hover()
            if self.hovered(gizmo) == index:
                return x, y
        return None

    def drag_arrow(self, gizmo: str, axis, metres: float, *, seconds: float, approach: float) -> dict:
        """Drag the gizmo's arrow most along ``axis`` by ``metres`` along it, and let go.

        The end pixel is :js:`axisTarget`: the axis point ``metres`` past where
        the press landed, projected -- exact, where a pixels-per-metre measured
        at the arrow overshoots under perspective. Returns what was done, where
        the server held the gizmo at the press and at the release, and how far
        the camera moved, which a drag must not.
        """
        axes = self.js("gizmoAxes", gizmo)
        if not axes:
            raise RuntimeError(f"no arrows on {gizmo}: is it visible?")

        def along(i: int) -> float:
            return sum(a * b for a, b in zip(axes[i]["world_dir"], axis, strict=True))

        index = max(range(len(axes)), key=lambda i: abs(along(i)))
        sign = 1.0 if along(index) >= 0 else -1.0
        at = self.grip(gizmo, index, axes[index], approach)
        if at is None:
            raise RuntimeError(f"{gizmo}: arrow {index} never showed the hover colour")
        target = self.js("axisTarget", gizmo, index, at[0], at[1], metres * sign)
        beyond = self.js("axisTarget", gizmo, index, at[0], at[1], metres * sign + 0.01)
        unit = axes[index]["world_dir"]
        start = self.where(gizmo)
        before = self.js("camera")
        self.advance()  # the lit cone, before the press
        self.down()
        self.advance()
        self.glide(target[0], target[1], seconds, gizmo=gizmo)
        landed = self._land(
            gizmo, unit, start, metres * sign, ((beyond[0] - target[0]) / 0.01, (beyond[1] - target[1]) / 0.01)
        )
        self.up()
        self.sync(gizmo)
        released = self.where(gizmo)
        self.advance()
        after = self.js("camera")
        drift = max(abs(a - b) for a, b in zip(before["eye"] + before["at"], after["eye"] + after["at"], strict=True))
        return {
            "arrow": index,
            "world_dir": [round(v, 4) for v in axes[index]["world_dir"]],
            "press_px": [round(v, 1) for v in at],
            "release_px": [round(v, 1) for v in target],
            "asked_m": round(metres, 4),
            "landing_miss_mm": landed,
            "camera_drift_m": drift,
            "start": list(start),
            "released": list(released),
        }

    def _land(self, gizmo: str, unit, start, metres: float, px_per_m) -> list[float]:
        """Nudge the held pointer until the server has the gizmo ``metres`` along ``unit`` from ``start``.

        Between frames, a fraction of a pixel at a time. The glide ends on the
        pixel :js:`axisTarget` computed, and what that leaves -- drei measures
        along its own drag plane, and a pixel at half a metre is 1.4 mm -- read
        98.5 mm for an asked 100. Returns the miss before each nudge [mm].
        """
        misses = []
        for _ in range(5):
            now = self.where(gizmo)
            miss = metres - sum((b - a) * u for a, b, u in zip(start, now, unit, strict=True))
            misses.append(round(miss * 1000, 2))
            if abs(miss) < 2e-4:
                break
            self._move(self.mouse[0] + px_per_m[0] * miss, self.mouse[1] + px_per_m[1] * miss)
            self.sync(gizmo)
        return misses

    # ----------------------------------------------------------- the sidebar
    def _rect(self, method: str, *args, scroll: tuple[str, str] | None = None) -> dict:
        if scroll is not None:
            self.js("scrollIntoView", *scroll)
            self._settle_hover()
        rect = self.js(method, *args)
        if rect is None:
            raise RuntimeError(f"nothing on the page for {method}{args}")
        return rect

    def press_button(self, text: str, within: str = "", *, approach: float = 0.6) -> None:
        rect = self._rect("button", text, within, scroll=(text, within))
        self.click(rect["cx"], rect["cy"], approach=approach)

    def toggle(self, label: str, within: str = "", *, approach: float = 0.6) -> None:
        rect = self._rect("checkbox", label, within, scroll=(label, within))
        self.click(rect["cx"], rect["cy"], approach=approach)

    def fold(self, title: str, open_: bool = True, *, approach: float = 0.6, filmed: bool = True) -> None:
        """Open or close the sidebar folder titled ``title`` (no click if it already is)."""
        head = self.js("folderHeader", title)
        if head is None:
            raise RuntimeError(f"no folder titled {title!r}")
        if bool(head["open"]) == open_:
            return
        if filmed:
            self.click(head["x"] + min(60.0, head["w"] / 2), head["cy"], approach=approach)
        else:
            self.page.mouse.click(head["x"] + min(60.0, head["w"] / 2), head["cy"])
            self.page.mouse.move(*self.mouse)

    def select(self, label: str, option: str, within: str = "", *, approach: float = 0.6) -> str:
        """Pick ``option`` in the dropdown ``label``; the value it shows afterwards."""
        field = self._rect("select", label, within, scroll=(label, within))
        self.click(field["x"] + field["w"] * 0.6, field["cy"], approach=approach)
        rect = None
        for _ in range(20):
            rect = self.js("option", option)
            if rect is not None:
                break
            self._settle_hover()
        if rect is None:
            raise RuntimeError(f"{label}: no option {option!r}")
        self.click(rect["x"] + rect["w"] * 0.3, rect["cy"], approach=0.45)
        return str((self.js("select", label, within) or {}).get("value", ""))

    def slider(self, label: str, value: float, seconds: float, within: str = "", *, approach: float = 0.5) -> float:
        """Drag the slider ``label`` to ``value``, in the slider's own units; the value it reads after.

        A drag stops on a pixel and the slider on a step, so the value is landed
        exactly with the arrow keys afterwards, the thumb still focused. No frame
        passes while it is.
        """
        s = self._rect("slider", label, within, scroll=(label, within))
        track, thumb = s["track"], s["thumb"] or s["track"]
        lo, hi = float(s["min"]), float(s["max"])
        frac = (float(value) - lo) / (hi - lo) if hi > lo else 0.0
        y = track["cy"]
        self.glide(thumb["cx"], y, approach)
        self.down()
        self.advance()
        # The top of the range is reached by running past the track's end, where the thumb clamps.
        self.glide(track["x"] + track["w"] * min(max(frac, 0.0), 1.0) + (6 if frac >= 0.999 else 0), y, seconds)
        self.up()
        best = math.inf
        for _ in range(60):
            now = float(self.js("slider", label, within)["value"])
            if abs(now - value) >= best:  # a step past it: one back, and that is as close as it goes
                self.page.keyboard.press("ArrowLeft" if now > value else "ArrowRight")
                break
            best = abs(now - value)
            if best < 1e-9:
                break
            self.page.keyboard.press("ArrowRight" if now < value else "ArrowLeft")
        self.advance()
        return float(self.js("slider", label, within)["value"])

    def scroll(self, dy: float, seconds: float) -> None:
        """Wheel the sidebar by ``dy`` pixels, a share per frame, the pointer over it."""
        panel = self.js("panelRect")
        if panel is not None:
            self.glide(panel["cx"], panel["cy"], 0.3)
        n = self.frames(seconds)
        for _ in range(n):
            self.page.mouse.wheel(0, dy / n)
            self.advance()

    def rest(self, seconds: float = 0.6) -> None:
        """Take the pointer off the sidebar to the 3D view's lower right, out of what is being watched."""
        left, top, width, height = self.js("canvasRect")
        self.glide(left + width * 0.85, top + height * 0.85, seconds)

    def text(self, label: str, within: str = "") -> str:
        return str(self.js("text", label, within) or "")
