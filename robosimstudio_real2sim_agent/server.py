# SPDX-License-Identifier: Apache-2.0
"""A studio page that rebuilds a scene file under itself, and a socket to drive it.

The seam a coding agent needs and the library did not have: a viser page
that stays open while the scene under it is replaced. Everything used here is
the studio's own public surface -- ``StudioSession.adopt`` swaps a built
``Sim`` into the page, ``tick`` is one loop iteration driven from outside,
``begin_run`` resets the clock it paces against, ``set_paused`` keeps the
page's checkbox truthful -- and ``SimBus`` marshals a request from any thread
onto the one that owns the simulator.

One process, one page, one file. The file is executed on every save
(:mod:`.scenefile` is the contract), the scene it binds is built, and the
result is swapped into the open page: the browser never reloads and the
camera stays where it was left. A file that does not build leaves the previous
scene up, with the traceback in the page's ``Live`` folder and in ``status``;
one that builds and then fails to go into the page (the old scene is already
given back by then) leaves a bare scene up instead, with the same report. The
modules the file imports from its own directory are imported afresh on every
rebuild, so an edit to a helper beside it lands with the next save.
Saves are noticed by polling the file's stamp once per loop pass and waiting
for it to hold still for :data:`STABLE_SECONDS`, because editors write in two
steps and the compile step catches whatever slips through.

The loop is the studio's own ``tick``, driven here rather than by
``StudioSession.run`` because ``run`` closes the page when it returns and
because a command has to land *between* iterations: ``adopt`` rebuilds the
panels, and a panel rebuilt inside its own drain would return into a loop over
handles that no longer exist. So every command arrives on the socket's
thread, is put on the bus, and runs on the main thread between two passes.
An exception out of a pass -- a panel's drain, a rebuild -- is reported in the
``Live`` folder and in ``status`` and the loop goes on; only ``quit`` stops it.

While a routine plays it steps the scene itself, so the studio's own stepping
is held for as long as it plays (``LiveSession.steps_held``), whatever the
Pause checkbox says; ``run N`` is refused while it plays, and a rebuild or a
reset under a ``run N`` ends that run with a reply saying so.

The page is served on every interface (viser binds ``0.0.0.0``), not only on
``localhost``: anyone who can reach the port can drive it. On a shared network
firewall the port, or reach it over an SSH tunnel (``ssh -L 8080:localhost:8080
<box>``). :attr:`LiveServer.url` names the port viser actually took, which is
the next one up when the one asked for is busy.
"""

from __future__ import annotations

import gc
import importlib
import inspect
import io
import json
import math
import os
import queue
import signal
import sys
import sysconfig
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np

from robosimstudio.core.bus import BusClosed, BusTimeout, SimBus
from robosimstudio_studio.session import StudioConfig

from . import checks, frames, stills
from .page import LiveSession
from .scenefile import Loaded, fallback_scene, run_scene_file, stamp
from .wire import ControlSocket, ready_line

__all__ = ["REFRESH_SECONDS", "STABLE_SECONDS", "LiveServer"]

STABLE_SECONDS = 0.3
"""How long a saved file has to stop changing before it is executed."""
REFRESH_SECONDS = 0.5
"""How often the ``Live`` folder's readouts are rewritten."""
BOARD_LIMIT = 8
"""Pictures kept on the board; the oldest goes when another comes. The run directory keeps every one."""


class LiveServer:
    """One page, one scene file, one socket. Everything but the accept loop runs on the main thread."""

    def __init__(
        self,
        *,
        scene_path: Path | str,
        sock_path: Path | str,
        run_dir: Path | str,
        config: StudioConfig,
        watch: bool = True,
        hold: bool = False,
    ) -> None:
        self.hold = bool(hold)
        """Land every rebuild paused rather than running."""
        self.path = Path(scene_path).resolve()
        self.sock_path = Path(sock_path)
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.watch_enabled = watch
        self.url = f"http://localhost:{config.port}"
        """The page, on the port viser actually bound -- rewritten once the server is up."""
        self.share_url: str | None = None
        # As long as the client's longest wait (ctl.TIMEOUTS).
        self.bus = SimBus(timeout=600.0)
        self.socket = ControlSocket(self.sock_path, self._on_request)
        self.session: LiveSession | None = None
        self.ns: dict = {}
        """The scene file's names. ``eval`` runs in this same namespace, so what it binds the routines see."""
        self.generation = 0
        """Rebuilds that have gone into the page; ``run N`` ends when it changes under it."""
        self.scene: Any = None
        self.sim: Any = None
        self.last_error = ""
        self.build_error = ""
        """The last build's failure alone; ``last_error`` also holds a routine's, a click's or a take's."""
        self.loaded_at = 0.0
        self.load_seconds = 0.0
        self.run_budget = 0
        self.resets = 0
        """Resets taken since the page came up: ``ctl run N`` reads a change in it as its run ended."""
        """Steps left before the page pauses again; -1 is no limit."""
        self.script: list = []
        """Generators an ``eval`` returned, advanced one ``next()`` per loop pass."""
        self.recorder: checks.Recorder | None = None
        """Records the script playing, when ``eval --record`` asked for it; finished when the queue empties."""
        self._baseline_modules: set[str] = set()
        """What was imported before the scene file first ran; never evicted on a rebuild."""
        self._tick_error = ""
        self.ref_image: np.ndarray | None = None
        self.ref_label = ""
        self.board: dict[str, np.ndarray] = {}
        """Pictures the agent put on the page, by label: what it is looking at, shown to the person watching."""
        self.clicks: queue.Queue = queue.Queue()
        """Button presses from viser's thread, taken on the main thread between two loop passes."""
        self.snap_n = 0
        self.cmp_n = 0
        self.last_take: dict = {}
        self.last_keys: list[str] = []
        """The keys of the last still comparison; a take that ends is held against the clip with them."""
        self._loaded_stamp = None
        self._pending = None
        self._pending_at = 0.0
        self._refreshed = 0.0
        self._shown: dict[str, str] = {}

    # ------------------------------------------------------------ the file
    def load(self) -> Loaded:
        """Execute the scene file and build what it declares. Raises on any failure."""
        self.evict_helpers()
        return run_scene_file(self.path)

    def evict_helpers(self) -> list[str]:
        """Forget the modules the scene file imported from its own directory, so they are read again.

        Only modules imported since the server started (anything already
        loaded then -- the library, even when the file sits at the repository
        root -- is never touched), and only those whose file is under the scene
        file's directory: a helper beside ``scene.py``, or a package of them.
        """
        root = self.path.parent
        # installed packages are never helpers, even when the virtualenv sits under the scene file's directory
        installed = {Path(p).resolve() for key in ("purelib", "platlib") if (p := sysconfig.get_paths().get(key))}
        gone = []
        for name, module in list(sys.modules.items()):
            if name in self._baseline_modules or name.partition(".")[0] in self._baseline_modules:
                continue
            file = getattr(module, "__file__", None)
            if not file:
                continue
            resolved = Path(file).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            if any(part in ("site-packages", "dist-packages") for part in resolved.parts) or any(
                resolved.is_relative_to(lib) for lib in installed
            ):
                continue
            del sys.modules[name]
            gone.append(name)
        if gone:
            importlib.invalidate_caches()
        return gone

    def swap(self, loaded: Loaded) -> bool:
        """Put a built scene into the page. False when it would not go in; a bare scene stands in then."""
        old = self.sim
        # before adopt() closes the old sim: a take's last frame is read off it
        self._finish_recorder("the scene was rebuilt")
        self.script.clear()
        self.run_budget = 0
        try:
            self._adopt(loaded)
            ok = True
        except Exception as exc:
            # adopt() gave the old scene back before it failed, so there is no
            # previous scene to keep: serve a bare one, and say why.
            error = traceback.format_exc()[-3000:]
            print(
                f"[live] {self.path.name} built but would not go into the page:\n{error}", file=sys.stderr, flush=True
            )
            try:
                scene = fallback_scene()
                self._adopt(Loaded(ns={}, scene=scene, sim=scene.build(control_mode="joint_position"), seconds=0.0))
            except Exception:
                error += (
                    "\n\nand the bare scene that should have replaced it failed too:\n" + traceback.format_exc()[-2000:]
                )
                print(f"[live] the bare scene failed too:\n{error}", file=sys.stderr, flush=True)
            self.last_error = self.build_error = error
            self.session.status = f"rebuild failed going into the page: {type(exc).__name__}: {exc}"
            ok = False
        del old
        gc.collect()
        self.refresh_live(force=True)
        return ok

    def _adopt(self, loaded: Loaded) -> None:
        # the file's names first: adopt() rebuilds the page, and its buttons are the routines the file binds
        self.ns, self.scene, self.sim = loaded.ns, loaded.scene, loaded.sim
        self.generation += 1
        self.session.adopt(loaded.sim, label="live")
        self.session.begin_run()
        self.last_error = self.build_error = ""
        self.loaded_at = time.time()
        self.load_seconds = loaded.seconds

    def reload(self, why: str = "requested") -> dict:
        """Rebuild from the file; the page is touched only when the build succeeded."""
        self._loaded_stamp, self._pending = stamp(self.path), None
        try:
            loaded = self.load()
        except (Exception, SystemExit) as exc:
            self.last_error = self.build_error = traceback.format_exc()[-3000:]
            self.session.status = f"reload failed: {type(exc).__name__}: {exc}"
            print(
                f"[live] reload ({why}) failed:\n{self.last_error}",
                file=sys.stderr,
                flush=True,
            )
            self.refresh_live(force=True)
            return {"ok": False, "error": self.last_error}
        if not self.swap(loaded):
            return {"ok": False, "error": self.last_error}
        self.session.status = f"loaded {self.path.name} in {loaded.seconds:.1f} s"
        names = loaded.sim.names()
        print(
            f"[live] loaded {self.path.name} in {loaded.seconds:.1f} s ({why}): {', '.join(names)}",
            flush=True,
        )
        return {"ok": True, "seconds": round(loaded.seconds, 2), "names": names}

    def watch(self) -> None:
        cur = stamp(self.path)
        if cur is None or cur == self._loaded_stamp:
            return
        now = time.perf_counter()
        if cur != self._pending:
            self._pending, self._pending_at = cur, now
            return
        if now - self._pending_at < STABLE_SECONDS:
            return
        self.reload("saved")

    # ------------------------------------------------------------ the loop
    def tick_once(self) -> None:
        try:
            self.bus.pump(budget=8)
            self._take_clicks()
            if self.watch_enabled:
                self.watch()
            if self.script:
                self._play()
            if self.session.tick() and self.run_budget > 0:
                self.run_budget -= 1
                if self.run_budget == 0:
                    self.session.set_paused(True)
            self._tick_error = ""
        except Exception as exc:
            # A panel's drain or a rebuild that raised: the page stays up and
            # says so. Printed once per distinct failure, since the same one
            # may come round every pass; and the pass that failed is not paced,
            # so give the loop a breath rather than spin on it.
            error = traceback.format_exc()[-3000:]
            if error != self._tick_error:
                self._tick_error = error
                self.last_error = error
                self.session.status = f"the loop failed: {type(exc).__name__}: {exc}"
                print(f"[live] the loop failed; still serving:\n{error}", file=sys.stderr, flush=True)
                self.refresh_live(force=True)
            time.sleep(0.05)
        if time.perf_counter() - self._refreshed >= REFRESH_SECONDS:
            self.refresh_live()

    def _take_clicks(self) -> None:
        while True:
            try:
                what, name = self.clicks.get_nowait()
            except queue.Empty:
                return
            if what == "play":
                reply = self.cmd_play(name)
                if not reply.get("ok"):
                    self.last_error = str(reply.get("error", ""))
                    self.refresh_live(force=True)

    def routines(self) -> dict[str, Any]:
        """The routines the scene file offers, by the name their button carries: ``routine`` (one) and
        ``routines`` (a dict of name -> callable, its keys named as ``str`` names them)."""
        found: dict[str, Any] = {}
        if callable(self.ns.get("routine")):
            found["routine"] = self.ns["routine"]
        many = self.ns.get("routines")
        if isinstance(many, dict):
            for k, v in many.items():
                if callable(v):
                    found.setdefault(str(k), v)
        return found

    def routine_names(self) -> list[str]:
        return list(self.routines())

    def start_script(self, gen, *, record: str = "", every: int = 4, camera: str = "") -> None:
        """Queue a generator to play one step per loop pass, filming it when asked.

        The page's own stepping is held while anything is queued
        (``LiveSession.steps_held``) and the Pause checkbox is left as it was,
        so when the queue empties the page carries on as it was before.
        """
        if record:
            # the new take films the new routine alone: what was queued ahead of it is dropped, not filmed into it
            self.script.clear()
            self._finish_recorder("another recording started")
            recorder = checks.Recorder(self, record, camera=camera, every=every)
            try:
                recorder.capture()  # frame 0: the state before anything happened
            except Exception:
                recorder.close()
                raise
            self.recorder = recorder
        self.script.append(gen)
        self.run_budget = 0

    def stop_scripts(self, why: str) -> None:
        """Drop every queued script and finish the take filming them. Main thread."""
        self.script.clear()
        if self.run_budget > 0:
            # a `run N` under way ends held, as it would have at its last step, not running on unbounded
            self.session.set_paused(True)
        self.run_budget = 0
        self._finish_recorder(why)

    def _play(self) -> None:
        """One step of the script at the head of the queue. It steps the sim itself;
        the studio's own stepping is held meanwhile, so nothing doubles it."""
        gen = self.script[0]
        try:
            next(gen)
        except StopIteration:
            self.script.pop(0)
            print("[live] script finished", flush=True)
        except Exception:
            self.script.pop(0)
            gen.close()
            self.last_error = traceback.format_exc()[-3000:]
            print(f"[live] script failed:\n{self.last_error}", file=sys.stderr, flush=True)
        else:
            if self.recorder is not None:
                try:
                    self.recorder.maybe()
                except Exception:
                    # the recording failed, not the routine: drop the take and let the routine play on
                    recorder, self.recorder = self.recorder, None
                    recorder.close()
                    print(
                        f"[live] the recording failed:\n{traceback.format_exc()[-3000:]}", file=sys.stderr, flush=True
                    )
            return
        if not self.script:
            self._finish_recorder("the script ended")

    def _hold_take_against_clip(self, take: str) -> None:
        """Every take is judged: held against the clip's timeline with the last comparison's keys.

        The strip and its verdict go on the board and into the log, so the
        before and the after are in front of the agent and the person watching
        without anyone asking. No timeline or no keys yet: one line says so.
        """
        if not self._timeline().get("anchors"):
            print(f"[live] take {take!r} not held against a clip: no ref/timeline.json yet", flush=True)
            return
        if not self.last_keys:
            print(
                f"[live] take {take!r} not held against the clip: no `compare --key` yet to say which things to follow",
                flush=True,
            )
            return
        reply = self.cmd_compare(keys=self.last_keys, take=take, note="held against the clip as the take ended")
        if reply.get("ok"):
            print(f"[live] take {take!r} against the clip: {reply['verdict']}\n{reply['report']}", flush=True)
        else:
            print(f"[live] take {take!r} could not be held against the clip: {reply.get('error')}", flush=True)

    def _finish_recorder(self, why: str) -> None:
        if self.recorder is None:
            return
        recorder, self.recorder = self.recorder, None
        try:
            summary = recorder.finish()
            self.last_take = summary
            checks.log_append(
                self.run_dir,
                {"cmd": "record", "scene": checks.scene_hash(self.path), **summary},
            )
            print(
                f"[live] take {summary['take']!r} written ({why}): {summary['frames']} frames, {summary['seconds']:.1f} s",
                flush=True,
            )
            self._hold_take_against_clip(summary["take"])
        except Exception:
            self.last_error = traceback.format_exc()[-3000:]
            print(
                f"[live] the take could not be written:\n{self.last_error}",
                file=sys.stderr,
                flush=True,
            )

    def refresh_live(self, force: bool = False) -> None:
        self._refreshed = time.perf_counter()
        session = self.session
        if session is None or self.sim is None or not session.host_widgets:
            return
        try:
            names = self.sim.names()
            age = time.time() - self.loaded_at
            state = "**playing a script**" if self.script else ("**paused**" if session.paused else "**running**")
            things = f"{len(names)} thing" + ("" if len(names) == 1 else "s")
            tail = "\n".join(self.last_error.rstrip().splitlines()[-8:])
            text = {
                "scene": f"**{self.path.name}** · {things} · built in {self.load_seconds:.1f} s, {age:.0f} s ago",
                "sim": f"step {self.sim.step_count} · t = {self.sim.time:.2f} s · {state}",
                "error": f"**last error** (`ctl status` has all of it)\n```\n{tail}\n```" if self.last_error else "",
            }
            for key, value in text.items():
                if force or self._shown.get(key) != value:
                    session.host_widgets[key].content = value
                    self._shown[key] = value
        except Exception:  # a widget mid-teardown; the next refresh will do
            pass

    def serve(self) -> None:
        """Open the page, take the socket, print the ready line and run the loop until ``quit``."""
        self.open_page()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: setattr(self.session, "quit_requested", True))
        try:
            # Inside the try, so a socket that cannot be had -- another live
            # server on it -- closes the page and exits before the ready line.
            self.socket.start()
            if self.config.share:
                # Newton's viewer asked for the tunnel at construction; asking again
                # returns the same URL once it is up, without blocking.
                self.share_url = self.session.server.request_share_url(verbose=False) or None
            share = self.share_url or ("unavailable" if self.config.share else None)
            print(ready_line(self.url, self.sock_path, self.path, share), flush=True)
            while self.session.viewer.is_running() and not self.session.quit_requested:
                self.tick_once()
        finally:
            try:
                # a take still filming is written out, as a reset would write it, before the page goes
                self.stop_scripts("the server stopped")
            except Exception:
                print(f"[live] the take could not be finished:\n{traceback.format_exc()[-3000:]}", file=sys.stderr)
            self.socket.close()
            self.bus.close()
            self.session.close()
            print("[live] stopped", flush=True)

    def open_page(self) -> None:
        """Build the file (a bare scene if it does not build) and open the page over it. No socket, no loop."""
        # The scene file's own directory, so what it imports by name beside
        # itself resolves -- and so `eval` can import it back the same way.
        sys.path.insert(0, str(self.path.parent))
        self._baseline_modules = set(sys.modules)
        # Stamped before the build, not after: a save made while the first
        # build runs is a change the watcher must still see.
        self._loaded_stamp = stamp(self.path)
        try:
            loaded = self.load()
            error = ""
        except (Exception, SystemExit):
            error = traceback.format_exc()[-3000:]
            print(
                f"[live] {self.path.name} did not build; serving a bare scene instead:\n{error}",
                file=sys.stderr,
            )
            scene = fallback_scene()
            loaded = Loaded(
                ns={},
                scene=scene,
                sim=scene.build(control_mode="joint_position"),
                seconds=0.0,
            )
        self.ns, self.scene, self.sim = loaded.ns, loaded.scene, loaded.sim
        self.session = LiveSession(loaded.sim, config=self.config, host=self)
        # viser takes the next port up when the one asked for is busy
        self.url = f"http://localhost:{self.session.server.get_port()}"
        self.generation += 1
        self.last_error = self.build_error = error
        self.loaded_at = time.time()
        self.load_seconds = loaded.seconds
        self.session.begin_run()
        self.refresh_live(force=True)

    # ------------------------------------------------------------ the socket
    def _on_request(self, request: dict) -> dict:
        """The socket thread's half: put the command on the bus and wait for the main thread's answer."""
        try:
            return self.bus.call(self.dispatch, request)
        except BusTimeout as exc:
            return {"ok": False, "error": f"server busy: {exc}"}
        except BusClosed:
            return {"ok": False, "error": "server shutting down"}

    def dispatch(self, request: dict) -> dict:
        """Run one command on the main thread, between two loop iterations."""
        name = str(request.get("cmd", ""))
        handler = getattr(self, f"cmd_{name}", None)
        if handler is None:
            return {"ok": False, "error": f"unknown command {name!r}"}
        try:
            return handler(**{k: v for k, v in request.items() if k != "cmd"})
        except Exception:
            return {"ok": False, "error": traceback.format_exc()[-3000:]}

    # ------------------------------------------------------------ commands
    def cmd_status(self) -> dict:
        sim, scene = self.sim, self.scene
        names = sim.names()
        positions = {}
        for name in names:
            try:
                positions[name] = [round(float(v), 4) for v in sim[name].position]
            except Exception:
                positions[name] = "no body"
        current = stamp(self.path)
        return {
            "ok": True,
            "url": self.url,
            "share_url": self.share_url,
            "pid": os.getpid(),
            "run_dir": str(self.run_dir),
            "file": str(self.path),
            "modified": time.strftime("%H:%M:%S", time.localtime(current[0] / 1e9)) if current else "missing",
            "loaded": time.strftime("%H:%M:%S", time.localtime(self.loaded_at)),
            "load_seconds": round(self.load_seconds, 2),
            "last_error": self.last_error,
            "cell": scene.cell,
            "names": names,
            "cameras": [c.name for c in sim.cameras],
            "step": int(sim.step_count),
            "time": round(float(sim.time), 3),
            "paused": bool(self.session.paused),
            "playing": len(self.script),
            "generation": self.generation,
            "resets": self.resets,
            "finite": bool(sim.state_is_finite()),
            "routines": self.routine_names(),
            "board": list(self.board),
            "recording": self.recorder.name if self.recorder is not None else "",
            "last_take": self.last_take.get("take", ""),
            "positions": positions,
        }

    def cmd_reload(self) -> dict:
        return self.reload("requested")

    def cmd_run(self, steps: int = 1) -> dict:
        if self.script:
            return {
                "ok": False,
                "error": "a routine is playing and steps the scene itself; wait for it to end, or `reset` to stop it",
            }
        self.run_budget = max(1, int(steps))
        self.session.set_paused(False)
        return {
            "ok": True,
            "step": int(self.sim.step_count),
            "budget": self.run_budget,
            "generation": self.generation,
            "resets": self.resets,
        }

    def cmd_pause(self) -> dict:
        self.run_budget = 0
        self.session.set_paused(True)
        return {"ok": True, "paused": True}

    def cmd_resume(self) -> dict:
        self.run_budget = -1
        self.session.set_paused(False)
        reply: dict = {"ok": True, "paused": False}
        if self.script:
            reply["note"] = "a routine is playing; the page runs on its own once it ends"
        return reply

    def cmd_reset(self) -> dict:
        # The page's two Reset buttons land in the same place: the loop's
        # reset_now(), which LiveSession extends to stop the scripts and the take.
        self.session.reset_requested = True
        return {"ok": True}

    def cmd_quit(self) -> dict:
        self.session.quit_requested = True
        return {"ok": True}

    def cmd_eval(self, code: str = "", record: str = "", every: int = 4, camera: str = "") -> dict:
        import robosimstudio as rss

        # The scene file's own namespace, not a copy: a name eval binds is one
        # the file's routines read, so a later `play` sees it. The next save
        # starts the namespace over. The page's names never shadow the file's.
        ns = self.ns
        ns.update(sim=self.sim, scene=self.scene)
        for name, value in (
            ("session", self.session),
            ("server", self.session.server),
            ("np", np),
            ("rss", rss),
            ("live", self),
        ):
            ns.setdefault(name, value)
        out = io.StringIO()
        value = None
        with redirect_stdout(out):
            try:
                expression = compile(code, "<eval>", "eval")
            except SyntaxError:
                exec(compile(code, "<eval>", "exec"), ns)
            else:
                value = eval(expression, ns)
        playing = False
        if inspect.isgenerator(value):
            # A script: played one step per loop pass so the page draws it,
            # instead of freezing for as long as it takes. The page is held
            # while it plays (the script does the stepping) and let go after
            # if it was running before.
            self.start_script(value, record=record, every=every, camera=camera)
            playing, value = True, None
        elif record:
            return {
                "ok": False,
                "error": "--record needs the code to return a generator (a routine the page plays); nothing was recorded",
            }
        reply = {
            "ok": True,
            "value": None if value is None else repr(value),
            "stdout": out.getvalue(),
            "playing": playing,
        }
        if record:
            reply["recording"] = str(self.run_dir / f"{record}.mp4")
        return reply

    def cmd_snapshot(
        self,
        camera: str = "",
        out: str = "",
        beside: str = "",
        size: str = "",
        eye: str = "",
        look_at: str = "",
        browser: bool = False,
        note: str = "",
    ) -> dict:
        self.snap_n = checks.next_number(self.run_dir, "snap", self.snap_n)
        path = Path(out) if out else self.run_dir / f"snap_{self.snap_n:03d}.png"
        if browser:
            rgb = stills.browser_still(self.session.server, size)
            if rgb is None:
                return {
                    "ok": False,
                    "error": "no browser is connected to the page; open it or drop --browser",
                }
            name = "browser"
        else:
            cam = stills.camera_for(self.sim, self.scene, name=camera, size=size, eye=eye, look_at=look_at)
            rgb = stills.render_still(self.sim, cam)
            name = cam.name
        stills.save_png(rgb, path)
        reply = {
            "ok": True,
            "png": str(path),
            "camera": name,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "mean": round(float(rgb.mean()), 1),
        }
        if beside:
            pair = stills.save_png(
                stills.side_by_side(rgb, beside),
                path.with_name(path.stem + "_vs_ref.png"),
            )
            reply["beside"] = str(pair)
        if note:
            reply["logged"] = checks.log_append(
                self.run_dir,
                {
                    "cmd": "snapshot",
                    "scene": checks.scene_hash(self.path),
                    "note": note,
                    "png": str(path),
                },
            )
        return reply

    def cmd_ref(
        self,
        op: str = "show",
        path: str = "",
        at: str = "0,end",
        every: float = 0.0,
        out: str = "",
        width: int = 0,
    ) -> dict:
        if op == "frames":
            return self.cut_frames(path, at=at, every=every, out=out, width=width)
        if op == "hide":
            self.session.hide_reference()
            self.ref_image, self.ref_label = None, ""
            return {"ok": True, "shown": False}
        self.ref_image, self.ref_label = stills.load_reference(path)
        self.session.show_reference(self.ref_image, self.ref_label)
        return {
            "ok": True,
            "shown": True,
            "size": [int(self.ref_image.shape[1]), int(self.ref_image.shape[0])],
        }

    # ------------------------------------------------------------ the clip, the numbers, the gate
    def ref_dir(self) -> Path:
        return self.path.parent / "ref"

    def cut_frames(
        self,
        clip: str,
        *,
        at: str = "0,end",
        every: float = 0.0,
        out: str = "",
        width: int = 0,
    ) -> dict:
        """Stills out of the clip at the times asked, a contact sheet, and the timeline they anchor."""
        clip_path = Path(clip)
        if not clip_path.is_file():
            return {"ok": False, "error": f"no clip at {clip}"}
        out_dir = Path(out) if out else self.ref_dir()
        meta = frames.probe(clip_path)
        if meta["duration"] <= 0:
            # ffmpeg read "Duration: N/A" (a raw .h264 stream, some webm): "end" would be 0 and no time can be sought
            return {
                "ok": False,
                "error": f"{clip} has no duration ffmpeg can read; re-mux it (ffmpeg -i {clip} -c copy out.mp4)",
            }
        times = frames.parse_times(at, meta["duration"] - 1.0 / (meta["fps"] or 30.0))  # "end" is the last frame
        # a still is named to the hundredth, so the anchor is that hundredth: two times inside one name are one moment
        floored: dict[float, tuple[str, str]] = {}  # t -> (label, the time as asked)
        for t, label in times:
            moment = math.floor(t * 100 + 1e-6) / 100
            if moment not in floored:
                floored[moment] = (label, f"{t:g}")
                continue
            kept, asked = floored[moment]
            if label and kept and label != kept:
                return {
                    "ok": False,
                    "error": f"{asked}:{kept} and {t:g}:{label} fall in one still ({frames._stem(moment)}.jpg); "
                    "mark them at least 0.01 s apart",
                }
            floored[moment] = (kept or label, asked)
        times = [(t, label) for t, (label, _) in floored.items()]
        stills_cut = frames.extract(clip_path, [t for t, _ in times], out_dir, width=width)
        anchors = [{"t": t, "file": p.name, "label": label} for (t, label), p in zip(times, stills_cut, strict=True)]
        spacing = float(every) if every else max(0.5, round(meta["duration"] / 24.0, 1))
        sheet, tiles = frames.contact_sheet(clip_path, spacing, out_dir / "contact.jpg")
        timeline = frames.write_timeline(out_dir, meta, anchors)
        return {
            "ok": True,
            "clip": meta["clip"],
            "fps": meta["fps"],
            "size": f"{meta['width']}x{meta['height']}",
            "duration": meta["duration"],
            "frames": [str(p) for p in stills_cut],
            "contact": f"{sheet} ({tiles} tiles, every {spacing} s)",
            "timeline": str(timeline),
            "report": "\n".join(f"{a['t']:6.2f} s  {a['file']}  {a['label']}" for a in anchors),
        }

    def _keys(self, keys: list | tuple) -> list:
        from .compare import parse_key

        return [parse_key(str(k)) for k in keys]

    def _timeline(self, path: str = "") -> dict:
        if path:
            return json.loads(Path(path).read_text())
        return frames.read_timeline(self.ref_dir())

    def _timeline_dir(self, path: str = "") -> Path:
        """Where a timeline's stills are: beside the timeline file it was read from."""
        return Path(path).parent if path else self.ref_dir()

    def _default_ref(self, ref: str, timeline: dict, timeline_path: str = "") -> str:
        if ref:
            return ref
        anchors = timeline.get("anchors") or []
        if anchors:
            first = Path(anchors[0]["file"])
            return str(first if first.is_absolute() else self._timeline_dir(timeline_path) / first)
        return ""

    def cmd_compare(
        self,
        ref: str = "",
        camera: str = "",
        keys: list | tuple = (),
        size: str = "",
        out: str = "",
        note: str = "",
        take: str = "",
        timeline: str = "",
    ) -> dict:
        parsed = self._keys(keys)
        self.cmp_n = checks.next_number(self.run_dir, "cmp", self.cmp_n)
        base = Path(out) if out else self.run_dir / f"cmp_{self.cmp_n:03d}.png"
        line = self._timeline(timeline)
        if take:
            if not line:
                return {
                    "ok": False,
                    "error": "no timeline: `ref frames <clip> --at 0,end` writes ref/timeline.json, or give --timeline",
                }
            if not parsed:
                return {
                    "ok": False,
                    "error": "a take is compared thing by thing: give at least one --key",
                }
            try:
                result = checks.compare_take(self.run_dir, take, line, parsed, self._timeline_dir(timeline))
            except Exception as exc:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            strip = checks.write_strip(base, result.pop("tiles"))
            verdict = checks.motion_summary(result["motion"])
            self.cmd_show(strip, label=f"take {take} against the clip: {verdict}")
            entry = {
                "cmd": "compare",
                "scene": checks.scene_hash(self.path),
                "score": result["score"],
                "note": note,
                "take": take,
                "strip": strip,
            }
            n = checks.log_append(self.run_dir, entry)
            (base.with_suffix(".json")).write_text(
                json.dumps(
                    {k: v for k, v in result.items() if k != "lines"},
                    default=checks._plain,
                    indent=1,
                )
            )
            return {
                "ok": True,
                "take": take,
                "score": result["score"],
                "strip": strip,
                "verdict": verdict,
                "report": "\n".join(result["lines"]),
                "logged": n,
                "anchors": result["anchors"],
                "motion": result["motion"],
            }
        ref = self._default_ref(ref, line, timeline)
        if not ref:
            return {
                "ok": False,
                "error": "no photograph: give --ref, or `ref frames <clip>` so the timeline's first anchor stands in",
            }
        if not Path(ref).is_file():
            return {"ok": False, "error": f"no photograph at {ref}"}
        rgb, seg, names, cam = checks.render_with_seg(self.sim, self.scene, camera=camera, size=size)
        try:
            result = checks.compare_still(rgb, seg, names, ref, parsed)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        pictures = checks.write_compare_pictures(base, rgb, result)
        self.cmd_show(pictures["beside"], label=f"compare {self.cmp_n}: render | {Path(ref).name}")
        self.cmd_show(pictures["look"], label=f"look {self.cmp_n}: what the photo has that the render lacks")
        if parsed:
            self.last_keys = [str(k) for k in keys]
        readings = {k: v for k, v in result.items() if k not in ("pairs", "fitted", "lines")}
        base.with_suffix(".json").write_text(
            json.dumps(
                {"ref": ref, "camera": cam.name, **readings},
                default=checks._plain,
                indent=1,
            )
        )
        n = checks.log_append(
            self.run_dir,
            {
                "cmd": "compare",
                "scene": checks.scene_hash(self.path),
                "score": result["score"],
                "note": note,
                "ref": Path(ref).name,
                "look": pictures["look"],
            },
        )
        return {
            "ok": True,
            "camera": cam.name,
            "ref": ref,
            "score": result["score"],
            "report": "\n".join(result["lines"]),
            **pictures,
            "logged": n,
            "objects": readings["objects"],
            "readings": {k: v for k, v in readings.items() if k != "objects"},
        }

    def cmd_check(
        self,
        camera: str = "",
        keys: list | tuple = (),
        objects: list | tuple = (),
        steps: int = 60,
        settle_mm: float = 2.0,
        penetration_mm: float = 1.0,
        timeline: str = "",
        note: str = "",
    ) -> dict:
        if self.script:
            return {
                "ok": False,
                "error": "a routine is playing and steps the scene itself; wait for it to end, or `reset` to stop it",
            }
        parsed = self._keys(keys)
        line = self._timeline(timeline)
        was_paused = self.session.paused
        self.session.set_paused(True)
        try:
            result = checks.gate(
                self,
                camera=camera,
                keys=parsed,
                objects=list(objects),
                steps=int(steps),
                settle_mm=float(settle_mm),
                penetration_mm=float(penetration_mm),
                timeline=line,
            )
        except Exception as exc:
            # the rest check may already have stepped the scene: reset it, as a passing check does
            self.session.reset_requested = True
            self.session.set_paused(was_paused)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        # the rest check stepped the scene: reset it to its start (what `eval` or `run N` moved is not kept),
        # and resume if it was running
        self.session.reset_requested = True
        self.session.set_paused(was_paused)
        (self.run_dir / "check.json").write_text(json.dumps(result, default=checks._plain, indent=1))
        n = checks.log_append(
            self.run_dir,
            {
                "cmd": "check",
                "scene": checks.scene_hash(self.path),
                "passed": result["passed"],
                "note": note,
            },
        )
        return {
            "ok": True,
            "passed": result["passed"],
            "report": result["report"],
            "json": str(self.run_dir / "check.json"),
            "logged": n,
            "checks": result["checks"],
            "drift_mm": result["drift_mm"],
        }

    def cmd_play(self, name: str = "", record: str = "", every: int = 4, camera: str = "") -> dict:
        """Play a routine the scene file binds, the way its button does."""
        found = self.routines()
        names = list(found)
        if not names:
            return {
                "ok": False,
                "error": "the scene file binds no `routine` (a callable taking sim, returning a generator) and no `routines` dict",
            }
        name = name or names[0]
        if name not in names:
            return {"ok": False, "error": f"no routine called {name!r}; the scene file binds {', '.join(names)}"}
        fn = found[name]
        try:
            # a routine that asks for the page's session gets it: the arm panels write the gripper
            # every tick, and a routine closing a grip has to hold them off while it plays
            wants_session = "session" in inspect.signature(fn).parameters
            gen = fn(self.sim, session=self.session) if wants_session else fn(self.sim)
        except Exception:
            return {"ok": False, "error": traceback.format_exc()[-3000:]}
        if not inspect.isgenerator(gen):
            return {
                "ok": False,
                "error": f"routine {name!r} returned {type(gen).__name__}, not a generator: it must yield once per control step",
            }
        self.start_script(gen, record=record, every=every, camera=camera)
        reply = {"ok": True, "playing": name}
        if record:
            reply["recording"] = str(self.run_dir / f"{record}.mp4")
        return reply

    def cmd_show(self, path: str = "", label: str = "", clear: bool = False) -> dict:
        """Put a picture on the page for the person watching: what the agent is looking at, as it looks at it."""
        if clear:
            self.board.clear()
            self.session.clear()
            return {"ok": True, "board": []}
        if not path:
            return {"ok": False, "error": "show needs a picture, or --clear"}
        image, name = stills.load_reference(path)
        label = label or name
        self.board.pop(label, None)  # a picture shown again is the newest, not kept in its old place
        self.board[label] = image
        self.session.show(label, image)
        while len(self.board) > BOARD_LIMIT:
            oldest = next(iter(self.board))
            del self.board[oldest]
            self.session.clear(oldest)
        return {"ok": True, "board": list(self.board)}

    def cmd_log(self, last: int = 0) -> dict:
        rows = checks.log_read(self.run_dir)
        if last:
            rows = rows[-int(last) :]
        return {
            "ok": True,
            "count": len(rows),
            "report": checks.log_report(rows),
            "entries": rows,
        }

    def cmd_catalog(self, kinds: list | tuple = ()) -> dict:
        """The names the registries know, as this process registered them.

        Server-side rather than in ``ctl`` because the content roots register
        cells and robots into the process that imported them, and because the
        client is standard-library only. ``assets`` runs the registries'
        discovery, which executes the task roots and takes seconds; naming the
        kinds wanted skips it.
        """
        from robosimstudio.assetlib.registry import list_assets
        from robosimstudio.robots.registry import list_embodiments, list_grippers
        from robosimstudio.scene.registry import list_cells

        listers = {
            "robots": list_embodiments,
            "grippers": list_grippers,
            "cells": list_cells,
            "assets": list_assets,
        }
        wanted = [str(k) for k in kinds] or list(listers)
        unknown = [k for k in wanted if k not in listers]
        if unknown:
            return {
                "ok": False,
                "error": f"no catalog named {', '.join(unknown)}; one of {', '.join(listers)}",
            }
        reply: dict = {"ok": True}
        for kind in wanted:
            reply[kind] = sorted(str(name) for name in listers[kind]())
        return reply
