# SPDX-License-Identifier: Apache-2.0
"""The studio page stepped one film frame at a time, and the hooks that put an example on it.

**An example runs unchanged.** :func:`install` patches two methods of
:class:`~robosimstudio_studio.session.StudioSession` and nothing else: the
constructor, so the page serves on a free port with no tunnel and writes under
the take's directory, and :meth:`run`, so instead of serving until the tab
closes the session hands itself to the film. That covers ``rss.view`` and a
script that opens a ``StudioSession`` itself (``06_teleop/01_force_feedback.py``),
and whatever ``page=`` hook the example registered has run by then.

**Lockstep, not real time.** A page left to itself paces on the wall clock and
draws when a wall-clock interval allows, so a slow scene films in slow motion
and a fast one drops frames. Here pacing and the in-between draws are off, and
each film frame is:

1. exactly the control steps that belong to it -- ``floor(k * hz / fps) -
   floor((k - 1) * hz / fps)``, so a 500 Hz scene at 15 fps takes 33, 33, 34, ...
   and never drifts from sim time;
2. one draw of that state, the read-only text refreshed with it;
3. a stamp node moved to ``x = k`` and the server flushed;
4. in the browser, a wait until the stamp shows ``k`` -- viser applies a batch in
   order, so the state before it has landed -- then a render and two animation
   frames;
5. a screenshot.

A clip is then exactly the scene's own time at the film's rate, whatever the
physics cost: a 1x clip of a scene that steps at a fifth of real time.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable
from fractions import Fraction
from typing import Any

STAMP = "/rss_record/stamp"
"""The node whose x is the index of the frame the server last presented."""


def install(film: Callable[[Any], None], *, port: int, out: str, overrides: dict[str, Any]) -> None:
    """Make every ``StudioSession`` this process opens a filmed one.

    Args:
        film: Called with the session in place of serving it, after the
            example's own ``page=`` hook. The session is closed afterwards
            whatever happens.
        port: Where the page serves; the browser is pointed there.
        out: The session's ``out``: runs, videos and traces land under the take.
        overrides: ``StudioConfig`` fields the take sets on top of the example's.
    """
    from robosimstudio_studio import session as studio

    cls = studio.StudioSession
    opened = cls.__init__

    def __init__(self, task, *, episode: int = 0, config=None) -> None:
        # No tunnel (rss.view asks for one, and the request blocks until viser's
        # share service answers), a port of the recorder's choosing, and the
        # example's own settings otherwise.
        config = dataclasses.replace(config or studio.StudioConfig(), share=False, port=port, out=out, **overrides)
        opened(self, task, episode=episode, config=config)

    def run(self, *, steps: int = 0, warmup: int = 0) -> None:
        del steps, warmup
        self.begin_run()
        try:
            film(self)
        finally:
            self.close()

    cls.__init__ = __init__
    cls.run = run


def control_hz(session) -> float:
    return float(session.env.request.physics.control_hz)


class Lockstep:
    """Steps a session film frame by film frame, and presents each frame to the browser.

    Args:
        session: The studio session, begun.
        fps: Film frames per second of *sim* time.
    """

    def __init__(self, session, fps: int) -> None:
        self.session = session
        self.fps = int(fps)
        self.hz = control_hz(session)
        self.per_frame = Fraction(self.hz).limit_denominator(100000) / self.fps
        """Control steps per film frame; a fraction, so the count never drifts."""
        self.frame = -1
        """The last frame presented; ``-1`` before the first."""
        self.stepped = 0
        """Control steps taken since frame 0's clock started."""
        self.stamp = session.server.scene.add_frame(STAMP, show_axes=False, position=(-1.0, 0.0, -10.0))
        self.quiet()

    def quiet(self) -> None:
        """No pacing and no draws between frames.

        Asserted before every frame rather than once: a scene load (``adopt``)
        resets ``draw_interval`` from the new scene's control rate.
        """
        s = self.session
        s.speed = 0.0
        s.render_every = 1 << 30
        s.draw_interval = math.inf

    def step(self, n: int) -> None:
        """Exactly ``n`` control steps; a pass that services a request and steps nothing does not count."""
        done = 0
        while done < n:
            if self.session.tick():
                done += 1

    def advance(self) -> int:
        """Step the control steps that belong to the next frame. Returns its index."""
        self.quiet()
        k = self.frame + 1
        want = math.floor(k * self.per_frame)  # frame k shows the scene at k / fps
        self.step(want - self.stepped)
        self.stepped = want
        return k

    def present(self, index: int) -> None:
        """Draw the state once, refresh the read-only text with it, and stamp it ``index``."""
        s = self.session
        for panel in s.panels:
            panel.refresh()
        s.refresh_holding()
        s.draw()
        self.stamp.position = (float(index), 0.0, -10.0)
        s.server.flush()
        self.frame = index

    @property
    def sim_time(self) -> float:
        """Seconds of sim time since frame 0."""
        return self.stepped / self.hz


def wait_for(check: Callable[[], bool], timeout: float, poll: float = 0.002) -> bool:
    """Poll ``check`` until it holds or ``timeout`` seconds pass."""
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if check():
            return True
        time.sleep(poll)
    return check()


# =============================================================================
# A take, filmed
# =============================================================================


class Film:
    """One take: the session it was handed, the browser on it, the frames and what they show.

    Called with the session in place of :meth:`StudioSession.run` (see
    :func:`install`). It opens the page, frames the camera, lays out the
    sidebar, plays the take's beats one film frame at a time, and judges the
    expectations at the marks the beats left. What it filmed is kept even when
    a beat raises, so a failed take can be looked at.

    Args:
        take: The :class:`recording.takes.Take`.
        port: Where the page serves.
        scratch: Where the browser's GPU layer is built.
        software: Allow a renderer that is not NVIDIA.
        probe: Stop after the first frame, with a full-size screenshot and the
            gizmos' arrows, for framing a camera.
        log: ``log(str)``.
    """

    def __init__(self, take, *, port: int, scratch, software: bool, probe: bool, log) -> None:
        self.take = take
        self.port = port
        self.scratch = scratch
        self.software = software
        self.probe = probe
        self.log = log
        self.session: Any = None
        self.sim: Any = None
        self.lock: Lockstep | None = None
        self.page: Any = None
        self.frames: list = []
        """``PIL.Image`` per frame, at the take's size."""
        self.times: list[float] = []
        self.captions: list[str] = []
        self.latency: list[float] = []
        """Milliseconds each frame waited for the page to show it."""
        self.spans: list[dict] = []
        self.marks: dict[str, dict] = {}
        self.caption = "start"
        self.renderer = ""
        self.camera: dict = {}
        self.unsynced = 0
        """Drag frames whose gizmo pose the server never confirmed (each waited a second)."""
        self.error = ""
        self.screenshot: bytes = b""

    # ---------------------------------------------------------------- the run
    def __call__(self, session) -> None:
        import traceback

        self.session, self.sim = session, session.env
        try:
            self._shoot()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

    def _shoot(self) -> None:
        from playwright.sync_api import sync_playwright

        from . import page as browser_page

        take = self.take
        self.lock = Lockstep(self.session, take.fps)
        # Untimed and unfilmed: whatever the scene does before anyone looks at it.
        self.lock.step(round(take.warmup * self.lock.hz))
        with sync_playwright() as pw:
            browser, context = browser_page.launch(
                pw, viewport=take.viewport, dpr=take.dpr, software=self.software, scratch=self.scratch, log=self.log
            )
            try:
                tab = context.new_page()
                self.page = browser_page.Page(
                    tab,
                    context.new_cdp_session(tab),
                    advance=self.advance,
                    sync=self.sync,
                    where=self.where,
                    fps=take.fps,
                    size=take.size,
                )
                # fixedDpr: viser lowers its pixel density on its own when it
                # measures a slow frame, and every frame here is slow.
                self.page.open(f"http://127.0.0.1:{self.port}/?fixedDpr={take.dpr:g}")
                self.renderer = str(self.page.js("renderer"))
                self.log(f"renderer: {self.renderer}")
                if "NVIDIA" not in self.renderer and not self.software:
                    raise RuntimeError(f"the page is drawn by {self.renderer!r}, not the GPU (--software to allow)")
                # A page= hook that aims the camera does so as the client
                # connects, on viser's thread: let it, then frame over it.
                time.sleep(0.5)
                self.prime(1)
                if take.camera is not None:
                    self.aim(*take.camera.resolve(self))
                for title in take.layout.collapse:
                    if self.page.js("folderHeader", title) is not None:
                        self.page.fold(title, False, filmed=False)
                for title in take.layout.expand:
                    self.page.fold(title, True, filmed=False)
                self.camera = self.page.js("camera")
                self.prime(2)
                if self.probe:
                    self.screenshot = tab.screenshot()
                    return
                if take.cursor:
                    self.page.show_cursor()
                self.mark("start")
                self.advance()
                for beat in take.beats:
                    self.play(beat)
                self.caption = "end"
                self.hold(take.tail)
                self.mark("end")
            finally:
                browser.close()

    def prime(self, n: int) -> None:
        """Present the state ``n`` times unfilmed, so what the camera and the folds did has landed."""
        for i in range(n):
            self.lock.present(-2 - i)
            self.page.wait_frame(STAMP, -2 - i)
        self.lock.frame = -1

    def play(self, beat) -> None:
        start, t0 = len(self.frames), self.lock.sim_time
        self.caption = beat.caption()
        self.log(f"  {t0:6.2f} s  {self.caption}")
        detail = beat.play(self) or {}
        if beat.mark:
            self.mark(beat.mark)
        self.spans.append(
            {
                "beat": type(beat).__name__,
                "caption": self.caption,
                "frames": [start, len(self.frames) - 1],
                "sim_s": [round(t0, 3), round(self.lock.sim_time, 3)],
                **({"mark": beat.mark} if beat.mark else {}),
                **detail,
            }
        )

    # ------------------------------------------------------------- the clock
    def advance(self) -> None:
        """One film frame: its control steps, presented, waited for, photographed."""
        k = self.lock.advance()
        self.lock.present(k)
        self.latency.append(self.page.wait_frame(STAMP, k))
        self.frames.append(self.page.capture())
        self.times.append(self.lock.sim_time)
        self.captions.append(self.caption)

    def hold(self, seconds: float, until=None) -> None:
        """Let the scene run ``seconds``, filmed; stop early once ``until(film)`` holds."""
        for _ in range(max(1, round(seconds * self.take.fps))):
            self.advance()
            if until is not None and until(self):
                break

    # ------------------------------------------------------------ the camera
    def aim(self, eye, look_at, fov: float) -> None:
        """Put every client's camera there, and check the page agrees to the millimetre."""
        import numpy as np

        for client in self.session.server.get_clients().values():
            client.camera.up_direction = (0.0, 0.0, 1.0)
            client.camera.fov = math.radians(fov)
            client.camera.position = tuple(float(v) for v in eye)
            client.camera.look_at = tuple(float(v) for v in look_at)
        self.session.server.flush()

        def there() -> bool:
            cam = self.page.js("camera")
            return np.allclose(cam["eye"], eye, atol=1e-3) and np.allclose(cam["at"], look_at, atol=1e-3)

        if not wait_for(there, 10.0, poll=0.05):
            raise RuntimeError(f"the camera did not go to {eye} -> {look_at}: {self.page.js('camera')}")

    def pixel(self, xyz) -> tuple[float, float]:
        x, y, _ = self.page.js("projectWorld", [float(v) for v in xyz])
        return float(x), float(y)

    # ------------------------------------------------------------ the scene
    def _points(self, name: str):
        import numpy as np

        ref = self.sim[name]
        try:
            if len(ref.particles):
                return np.asarray(ref.particle_q, dtype=np.float64).reshape(-1, 3)
        except Exception:
            pass
        return np.asarray(ref.position, dtype=np.float64).reshape(1, 3)

    def position(self, name: str) -> tuple[float, float, float]:
        """A body's origin, or a deformable's centroid [m]."""
        return tuple(float(v) for v in self._points(name).mean(axis=0))

    def _half_height(self, name: str) -> float:
        spec = self.sim[name].spec
        half = getattr(spec, "half", None)
        if half is not None:
            return float(half[2])
        return float(getattr(spec, "radius", 0.0) or 0.0) + float(getattr(spec, "half_height", 0.0) or 0.0)

    def top(self, name: str) -> float:
        points = self._points(name)
        if len(points) > 1:
            return float(points[:, 2].max())
        return float(points[0, 2]) + self._half_height(name)

    def height(self, name: str) -> float:
        points = self._points(name)
        if len(points) > 1:
            return float(points[:, 2].max() - points[:, 2].min())
        return 2.0 * self._half_height(name)

    def corner(self, name: str, sx: int, sy: int, inset: float) -> tuple[float, float, float]:
        import numpy as np

        q = self._points(name)
        x = q[:, 0].min() + inset if sx < 0 else q[:, 0].max() - inset
        y = q[:, 1].min() + inset if sy < 0 else q[:, 1].max() - inset
        return tuple(float(v) for v in q[int(np.argmin(np.hypot(q[:, 0] - x, q[:, 1] - y)))])

    def along(self, name: str, frac: float) -> tuple[float, float, float]:
        q = self._points(name)
        return tuple(float(v) for v in q[round(min(max(frac, 0.0), 1.0) * (len(q) - 1))])

    def bench(self, point) -> tuple[float, float, float]:
        """A work-surface point in world coordinates."""
        from robosimstudio.scene.frames import in_surface
        from robosimstudio.scene.registry import get_cell

        name = getattr(getattr(self.sim, "handles", None), "scene_name", "") or ""
        cell = get_cell(name) if name else None
        return tuple(float(v) for v in in_surface(cell, (*point, 0.0, 0.0, 0.0, 1.0))[:3])

    def hand(self, arm: str) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.session.arms[arm].ik.ee_pose(self.sim.state)[0])

    def held(self) -> list[str]:
        return list(self.session.held())

    def mark(self, name: str) -> None:
        """What the expectations will ask about, as it is now."""
        names = {n for e in self.take.expect for n in e.things()}
        self.marks[name] = {
            "frame": len(self.frames) - 1,
            "sim_s": round(self.lock.sim_time, 3),
            "things": {n: self.position(n) for n in names},
            "heights": {n: self.height(n) for n in names},
            "hands": {arm: self.hand(arm) for arm in self.session.arms},
        }

    # ------------------------------------------------------------ the gizmos
    def _handles(self):
        for panel in self.session.panels:
            yield from getattr(getattr(panel, "handles", None), "handles", None) or ()

    def gizmo(self, spec: str) -> str:
        """An arm's cursor, the last handle a click put down, or a scene path as it is."""
        if spec in self.session.arms:
            return f"/studio/{spec}/cursor"
        if spec == "handle":
            live = [h for h in self._handles() if h.live]
            if not live:
                raise RuntimeError("no handle is down: click something first")
            return max(live, key=lambda h: h.seq).gizmo.name
        return spec

    def where(self, name: str):
        """Where the *server* has the gizmo: what the scene is being driven to."""
        for arm, panel in self.session.arms.items():
            if name == f"/studio/{arm}/cursor":
                return tuple(float(v) for v in panel.cursor_pos)
        for handle in self._handles():
            if handle.gizmo.name == name:
                return tuple(float(v) for v in handle.position)
        raise KeyError(f"no gizmo {name} on this page")

    def sync(self, name: str) -> None:
        """Wait (unfilmed) until the server holds the pose the browser shows for ``name``.

        viser sends a drag at most every 50 ms, trailing the last move; a frame
        stepped before it lands would show the scene chasing the pose before.
        """

        def agreed() -> bool:
            shown = self.page.js("nodeWorld", name)
            return shown is not None and math.dist(shown, self.where(name)) < 1e-6

        if not wait_for(agreed, 1.0):
            self.unsynced += 1

    def settle(self, spec: str, timeout: float = 1.5) -> None:
        """Before an arm's drag: let the hand catch up with its gizmo, filmed.

        Done when the panel says ``on target``, or when the hand has stopped:
        a hand carrying something sits under its target -- 6 to 11 mm with the
        rigid example's cube in it, past the panel's tolerance -- and waiting
        for ``on target`` there waited out the whole timeout before each drag.
        """
        panel = self.session.arms.get(spec)
        if panel is None or panel.status == "on target":
            return
        trail = [self.hand(spec)]

        def still(film) -> bool:
            trail.append(film.hand(spec))
            moving = len(trail) < 4 or math.dist(trail[-1], trail[-4]) > 5e-4
            return panel.status == "on target" or not moving

        self.hold(timeout, until=still)

    def drag(self, name: str, unit, metres: float, *, seconds: float, approach: float) -> dict:
        """One arrow drag, and how far the server had the gizmo go when the button came up.

        Read at the release, not after: once let go, an arm's panel puts its
        gizmo back on the hand when the hand is within tolerance, and a hand
        stopped 2 mm short by what it is closing on reads as a 2 mm short drag.
        """
        done = self.page.drag_arrow(name, unit, metres, seconds=seconds, approach=approach)
        moved = [b - a for a, b in zip(done.pop("start"), done.pop("released"), strict=True)]
        along = sum(m * u for m, u in zip(moved, unit, strict=True))
        done.update(gizmo=name, moved_m=[round(v, 5) for v in moved], along_m=round(along, 5))
        self.log(
            f"           {name}: asked {metres * 1000:+.1f} mm, moved {along * 1000:+.1f} mm along "
            f"{[round(u, 2) for u in unit]} (misses {done['landing_miss_mm']} mm while landing), "
            f"camera drift {done['camera_drift_m'] * 1000:.2f} mm"
        )
        return done

    # ---------------------------------------------------------- the verdict
    def judge(self) -> list[dict]:
        out = []
        for expectation in self.take.expect:
            try:
                passed, reading = expectation.judge(self)
            except KeyError as exc:
                passed, reading = False, f"no mark {exc} to judge {type(expectation).__name__} at"
            out.append({"expect": type(expectation).__name__, "passed": bool(passed), "reading": reading})
        return out
