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
rebuild, and what they and the file registered is taken back first
(:class:`.scenefile.SceneRegistrations`), so an edit to a helper beside it --
an entity class, a cell -- lands with the next save.
Saves are noticed by polling the stamps of the file and of its helpers --
every ``.py`` under its folder that is the scene's own code, two folders down
(:func:`.scenefile.helper_files`) -- every :data:`SCAN_SECONDS`, taken
before each build, and waiting for them to hold still for
:data:`STABLE_SECONDS`, because editors write in two steps and the compile
step catches whatever slips through. The helpers' cached bytecode is
hash-checked, so an edit within the second of the last one is read from the
source.

The loop is the studio's own ``tick``, driven here rather than by
``StudioSession.run`` because ``run`` closes the page when it returns and
because a command has to land *between* iterations: ``adopt`` rebuilds the
panels, and a panel rebuilt inside its own drain would return into a loop over
handles that no longer exist. So every command arrives on the socket's
thread, is put on the bus, and runs on the main thread between two passes.
An exception out of a pass -- a panel's drain, a rebuild -- is reported in the
``Live`` folder and in ``status`` and the loop goes on; only ``quit`` stops it.

A routine plays one ``next()`` per loop pass, whatever the Pause checkbox
says, by its kind. One that takes the page's
``session`` drives through the arm panels (``robosimstudio_studio.routine``'s
``reach``, ``grip``, ``wait``): a pass whose ``next()`` only yielded is
stepped by the page, one that stepped the scene itself is not. Any other
routine drives the scene itself: the page never steps it
(``LiveSession.steps_held``), and the arm panels are suspended while it plays
-- they would write the gripper and the joints under it -- and given back as
found when it ends; one that yields :data:`IDLE_LIMIT` times running without
stepping is stopped, saying why. A generator ``eval`` returns is the second
kind: to drive through the page, bind the routine in the scene file with a
``session`` parameter and ``play`` it. ``run N`` is refused while a routine
plays, and a rebuild or a reset under a ``run N`` ends that run with a reply
saying so. A ``run N`` that takes its steps, is paused, or is cut short by a
routine, a reset or a failing loop pass ends held; one cut short by a rebuild,
``resume`` or another run leaves the page to that. How it ended is in ``status``'s
``last_run`` (and ``runs_ended``, by serial).

The page is served on every interface (viser binds ``0.0.0.0``), not only on
``localhost``: anyone who can reach the port can drive it. On a shared network
firewall the port, or reach it over an SSH tunnel (``ssh -L 8080:localhost:8080
<box>``). :attr:`LiveServer.url` names the port viser actually took, which is
the next one up when the one asked for is busy.
"""

from __future__ import annotations

import ast
import gc
import importlib
import importlib.util
import inspect
import io
import json
import os
import queue
import signal
import sys
import threading
import time
import traceback
from collections import deque
from contextlib import contextmanager, nullcontext, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np

from robosimstudio.core.bus import BusClosed, BusTimeout, SimBus
from robosimstudio_studio.session import StudioConfig

from . import checks, frames, stills
from .page import LiveSession
from .scenefile import (
    Loaded,
    ScenePaths,
    SceneRegistrations,
    fallback_loaded,
    hash_checked,
    helper_files,
    run_scene_file,
    stamp,
)
from .wire import ControlSocket, ready_line

__all__ = ["REFRESH_SECONDS", "SCAN_SECONDS", "STABLE_SECONDS", "LiveServer"]

STABLE_SECONDS = 0.3
"""How long a saved file has to stop changing before it is executed."""
SCAN_SECONDS = 0.2
"""How often the loop looks for a save."""
ROUTINE_WAIT = 5.0
"""Seconds a ``status`` waits out one step of a routine before it answers with the busy status instead."""
REFRESH_SECONDS = 0.5
"""How often the ``Live`` folder's readouts are rewritten."""
BOARD_LIMIT = 8
"""Pictures kept on the board; the oldest goes when another comes. The run directory keeps every one."""
RECENT = 32
"""Routines ended and takes written that ``status`` keeps, by serial and by name, for a client waiting on one."""
ROUTINE_PLAYING = "a routine is playing (`status` lists it under `playing`); let it end, or `reset` to stop it"
"""What ``run`` and ``check`` answer while a routine plays."""
IDLE_LIMIT = 300
"""Passes in a row a routine that drives the scene itself may yield without stepping it before it is stopped."""
SPEED_WINDOW = 5.0
"""Seconds of wall clock the ``speed`` reading in ``status`` is taken over."""
BUSY = {
    "reload": "rebuilding the scene file",
    "snapshot": "rendering a still",
    "compare": "rendering the scene and comparing it with the photograph",
    "check": "running the gate (it steps the scene, then resets it)",
    "eval": "running eval code",
    "play": "starting a routine",
    "ref": "cutting stills out of the clip",
    "catalog": "listing the registries",
}
"""What the main thread is doing, by command, for a ``status`` asked meanwhile."""
READS_THE_SCENE = ("snapshot", "compare", "check", "eval", "play", "run", "reset")
"""The commands that act on the scene in the page: their reply says when it is not the file's (a save that did
not build, a bare scene standing in, a save not built yet), and they are refused while no scene is in."""
MISSING_STAMPS = object()
"""The stamps not given: the status takes them itself."""
FIRST_RENDER = (
    "; the first path-traced picture in a process compiles the renderer's shaders, "
    "which can take minutes -- later ones take seconds"
)


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
        # absolute before serve.main changes the process into the scene file's directory
        self.sock_path = Path(sock_path).absolute()
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.watch_enabled = watch
        self.url = f"http://localhost:{config.port}"
        """The page, on the port viser actually bound -- rewritten once the server is up."""
        self.share_url: str | None = None
        # A command waits on the main thread as long as its client said it would (`_on_request`); this is the
        # wait for one that said nothing, and for a full `status`.
        self.bus = SimBus(timeout=900.0)
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
        """Steps left of the ``run N`` under way; 0 when none is."""
        self.runs = 0
        """``run N`` commands taken, the serial of the last one."""
        self.run_target = 0
        self.runs_ended: dict[int, dict] = {}
        """How each of the last :data:`RECENT` runs ended, by serial: ``serial``, ``how`` (``done`` or why not),
        ``step``, ``target``, and ``finite`` when it ran to its end or was paused. What ``run N --wait`` waits on, its
        own and not the newest."""
        self._walked: tuple[float, list[Path]] = (-1.0, [])
        """The last walk of the scene's folder for its helpers, and when: a status reuses it for a second."""
        self.scene_id = ""
        """The hash of the files the scene in the page was built from (:func:`.checks.scene_hash`), taken as the
        build read them; the log ties each reading to it."""
        self.resets = 0
        """Resets taken since the page came up: ``ctl run N`` reads a change in it as its run ended."""
        self.script: list = []
        """Generators an ``eval`` returned, advanced one ``next()`` per loop pass."""
        self.head_stepped = False
        """Whether the head script's last ``next()`` stepped the scene."""
        self._head_ready: int | None = None
        """``id`` of the head script once its arm panels are set as its kind asks."""
        self._arms_before: dict[str, bool] = {}
        """Which arm panels were suspended when the head script started: they are put back so when it ends."""
        self.recorder: checks.Recorder | None = None
        """Films the script ``play --record`` or ``eval --record`` started; finished when that script ends."""
        self.scene_paths = ScenePaths(self.path.parent)
        """What the scene's own code is, for the modules a rebuild imports afresh and the registrations it takes back."""
        self.registrations = SceneRegistrations(self.scene_paths)
        """What the file, its helpers and ``eval`` registered since the last rebuild; taken back before the next."""
        self.busy_state: tuple[str, float, bool] = ("", 0.0, False)
        """What the main thread is doing that a ``status`` would otherwise wait out (empty between commands), since
        when, and whether it is one step of a routine -- the page at work, not held up. One tuple, written whole,
        because the socket thread reads it while the main thread changes it."""
        self.log_path = _log_path()
        """The file this process's output goes to, read once: it does not change."""
        self._motions: dict[str, dict] = {}
        """A take's motion, worked out as it was written, for the comparison that follows it."""
        self._status_lock = threading.Lock()
        self._status_pending: dict | None = None
        """The one ``status`` queued for the main thread, which every caller meanwhile waits on."""
        self.rendered = False
        """Whether a still has been rendered in this process; the first one compiles the path tracer."""
        self.script_meta: dict[int, dict] = {}
        """Per playing script (by ``id``): its serial, its name, and where it started."""
        self.scripts_started = 0
        self.last_script: dict = {}
        """How the last script to end ended: ``serial``, ``name``, ``how``, ``error``, ``steps``, ``seconds``."""
        self._speed: deque = deque()
        """``(wall, sim time)`` samples over the last :data:`SPEED_WINDOW` seconds."""
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
        self.ended: dict[int, dict] = {}
        """How each of the last :data:`RECENT` scripts ended, by serial: a client waits on its own, not the newest."""
        self.takes_written: list[str] = []
        """The last :data:`RECENT` takes written, by name."""
        self.last_keys: list[str] = []
        """The keys of the last still comparison; a take that ends outlines its things with them."""
        self.serving = "scene"
        """What the page shows: ``scene`` (the file's, or the one before a save that did not build), ``stand-in``
        (the bare scene standing in for one that did not go in) or ``nothing`` (the stand-in would not go in either:
        no scene is in the page, and the commands that read one are refused until a save goes in)."""
        self._loaded_stamp = None
        """The stamps the scene now up was built from, taken before its build: a save meanwhile is still seen."""
        self._pending = None
        self._pending_at = 0.0
        self._scanned = 0.0
        self._refreshed = 0.0
        self._shown: dict[str, str] = {}

    # ------------------------------------------------------------ the file
    def load(self, files: list[Path] | None = None) -> Loaded:
        """Execute the scene file and build what it declares. Raises on any failure.

        What the last run registered is taken back first, so an edited entity
        class, cell or gripper is registered as edited rather than kept as it
        was (a registry keeps the first of two from the same source) or refused
        (two that differ). A file that does not build gives them back, and
        puts back the helper modules the scene before it was built from, since
        that scene stays up. ``files`` are the helpers, when the caller has
        just walked for them.
        """
        taken = self.registrations.undo()
        before: dict = {}
        try:
            before = self.evict_helpers(files)
            return run_scene_file(self.path)
        except BaseException:
            # the scene before it stays up: what this run registered goes, and what that scene registered comes back;
            # so do the helper modules it was built from, in place of those this run imported
            self.registrations.undo()
            self.registrations.redo(taken)
            for name in self._scene_modules():
                del sys.modules[name]
            sys.modules.update(before)
            raise

    def evict_helpers(self, files: list[Path] | None = None) -> dict[str, Any]:
        """Forget the modules the scene file imported from its own directory, so they are read again.

        Only modules imported since the server started (anything already
        loaded then -- the library, even when the file sits at the repository
        root -- is never touched), and only those whose file is under the scene
        file's directory: a helper beside ``scene.py``, or a package of them.
        The helpers' cached bytecode is made hash-checked. Returns the modules
        forgotten, by name, for a build that fails to put back.
        """
        gone = self._scene_modules()
        files = set(helper_files(self.path) if files is None else files)
        for name, module in gone.items():
            del sys.modules[name]
            files.add(Path(module.__file__))
        # Default bytecode is trusted on the source's whole-second mtime and size: an edit within the second of the
        # one it was compiled from, to the same length, would import the old code. A helper's is made hash-checked
        # instead -- the import checks it against the text, and rewrites it so when it does not match -- once per
        # file that has bytecode, imported now or not (one whose import failed left its own).
        for file in files:
            if file.suffix == ".py":
                hash_checked(file)
        if gone:
            importlib.invalidate_caches()
        return gone

    def _scene_modules(self) -> dict[str, Any]:
        """The imported modules that are the scene's own code (:class:`.scenefile.ScenePaths`), by name."""
        return {
            name: module
            for name, module in list(sys.modules.items())
            if (file := getattr(module, "__file__", None)) and self.scene_paths.is_scene(file, name)
        }

    def swap(self, loaded: Loaded) -> bool:
        """Put a built scene into the page. False when it would not go in: the page then holds nothing of it, and
        the caller, once it lets go of ``loaded`` too, puts a bare scene in its place (:meth:`stand_in`)."""
        old = self.sim
        # before adopt() closes the old sim: a take's last frame is read off it
        self._finish_recorder("the scene was rebuilt")
        self._end_scripts("stopped: the scene was rebuilt")
        self._end_run("the scene was rebuilt", hold=False)
        try:
            self._adopt(loaded)
            self.serving = "scene"
            ok = True
        except Exception as exc:
            # adopt() closed the old scene before it failed, so there is no previous scene to keep
            error = traceback.format_exc()[-3000:]
            print(
                f"[live] {self.path.name} built but would not go into the page:\n{error}", file=sys.stderr, flush=True
            )
            self.last_error = self.build_error = error
            self.session.status = f"rebuild failed going into the page: {type(exc).__name__}: {exc}"
            ok = False
        if not ok:
            self._let_go(loaded.sim)
        del old
        gc.collect()
        if ok:
            self.refresh_live(force=True)
        return ok

    def _let_go(self, failed: Any) -> None:
        """Give back what a scene that would not go into the page holds, before a bare one is built in its place:
        the page's panels, environment and renderers over it, the viewer's model, and the server's names for it.
        Its model and state go with the last name, which the caller drops."""
        for give_back in (self.session.teardown, lambda: self.session.viewer.set_model(None), failed.close):
            try:
                give_back()
            except Exception:
                pass
        self.ns = self.scene = self.sim = None
        self.serving = "nothing"

    def stand_in(self) -> None:
        """A bare scene in the page in place of one that would not go in, keeping why as the error; when the bare
        scene will not go in either, the page serves nothing, and says so."""
        error = self.last_error
        gc.collect()  # the failed scene's model and state, now nothing names them
        try:
            self._adopt(fallback_loaded())
            self.serving = "stand-in"
        except Exception:
            self.serving = "nothing"
            error += (
                "\n\nand the bare scene that should have replaced it failed too:\n" + traceback.format_exc()[-2000:]
            )
            print(f"[live] the bare scene failed too:\n{error}", file=sys.stderr, flush=True)
        self.last_error = self.build_error = error
        self.refresh_live(force=True)

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
        # stamped before the build: a save made while it runs, to the file or a helper, is a change still to build
        files = helper_files(self.path)
        self._walked = (time.perf_counter(), files)
        self._loaded_stamp, self._pending = self._stamps(files), None
        scene_id = checks.scene_hash(self.path, files)
        try:
            loaded = self.load(files)
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
            del loaded  # the last name for the scene that would not go in: the stand-in is built without it
            self.stand_in()
            self.scene_id = "bare"
            return {"ok": False, "error": self.last_error}
        self.scene_id = scene_id
        self.session.status = f"loaded {self.path.name} in {loaded.seconds:.1f} s"
        names = loaded.sim.names()
        print(
            f"[live] loaded {self.path.name} in {loaded.seconds:.1f} s ({why}): {', '.join(names)}",
            flush=True,
        )
        return {"ok": True, "seconds": round(loaded.seconds, 2), "names": names}

    def _stamps(self, files: list[Path] | None = None) -> tuple | None:
        """The scene file's stamp, then each helper's with its path: a save to any, a helper made or one deleted, is
        a rebuild. None when the scene file is gone. ``files`` are the helpers when the caller has just walked for
        them; else the walk of the last second is used -- every stamp is read afresh, and a new helper the scene
        imports comes with a save to the scene file, which is."""
        first = stamp(self.path)
        if first is None:
            return None
        if files is None:
            at, files = self._walked
            if time.perf_counter() - at > 1.0:
                files = helper_files(self.path)
                self._walked = (time.perf_counter(), files)
        return (first, tuple((str(f), stamp(f)) for f in files))

    def watch(self) -> None:
        now = time.perf_counter()
        if now - self._scanned < SCAN_SECONDS:
            return
        self._scanned = now
        cur = self._stamps()
        if cur is None or cur == self._loaded_stamp:
            return
        if cur != self._pending:
            self._pending, self._pending_at = cur, now
            return
        if now - self._pending_at < STABLE_SECONDS:
            return
        with self._busy("rebuilding the scene file after a save"):
            self.reload("saved")

    # ------------------------------------------------------------ the loop
    def tick_once(self) -> None:
        try:
            self.bus.pump(budget=8)
            self._take_clicks()
            if self.watch_enabled:
                self.watch()
            head = self.script[0] if self.script else None
            paged = False
            if head is not None:
                serial = (self.script_meta.get(id(head)) or {}).get("serial", 0)
                sim = self.sim
                self._credit(head, serial, self._play())  # what its next() took, the last one included
                paged = self.page_steps_script() and self.script[0] is head
                before, mark = int(sim.step_count), checks.reset_mark(sim)
            if self.run_budget > 0 and self.session.paused:
                self._end_run("paused")  # the Pause checkbox, ticked under a run
            if self.session.tick() and self.run_budget > 0:
                if self.run_budget == 1:
                    self._end_run("done")
                else:
                    self.run_budget -= 1
            if paged and self.sim is sim and self.script and self.script[0] is head:
                # the page's step, taken for it -- unless a reset in that tick ended it first
                self._credit(head, serial, checks.steps_since(sim, before, mark))
            self._tick_error = ""
        except Exception as exc:
            # A panel's drain or a rebuild that raised: the page stays up and
            # says so. Printed once per distinct failure, since the same one
            # may come round every pass; and the pass that failed is not paced,
            # so give the loop a breath rather than spin on it.
            error = traceback.format_exc()[-3000:]
            self._end_run(f"the loop failed: {type(exc).__name__}: {exc} (`./ctl status` has the traceback)")
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
                reply = self.dispatch({"cmd": "play", "name": name})  # as `./ctl play` would: refused on an empty page
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
                    found.setdefault(str(k), v)  # a `routines` entry named "routine" is the bound `routine`'s name
        return found

    def routine_names(self) -> list[str]:
        return list(self.routines())

    def start_script(
        self,
        gen,
        *,
        name: str = "eval",
        drives_page: bool = False,
        record: str = "",
        every: int = 4,
        camera: str = "",
    ) -> int:
        """Queue a generator to play one step per loop pass, filming it when asked.

        It plays whatever the Pause checkbox says, which is left as it was, so
        when the queue empties the page carries on as it was before: held while
        a script that steps the scene itself plays (``LiveSession.steps_held``),
        stepping once per yield for one that does not (``LiveSession.tick``).
        ``drives_page`` is True for a routine that took the page's ``session``
        (it drives through the arm panels) and False for anything else (it
        drives the scene itself). Returns the script's serial, which
        ``status``'s ``last_script`` names once it has ended.
        """
        # an earlier routine's failure is in its last_script; this one starts clean -- what ending or
        # finishing the ones before it raises below is new, and stays
        if self.last_error not in (self.build_error, self._tick_error):
            self.last_error = self.build_error or self._tick_error
        if record:
            # Made and tried first: a camera that is not there, a frame that will not film, or a video the encoder
            # cannot make, fails before what plays, and its take, are touched.
            recorder = None
            try:
                recorder = checks.Recorder(self, record, camera=camera, every=every)
                recorder.probe()
            except Exception:
                if recorder is not None:
                    recorder.close()
                self._close(gen)  # it will never play
                raise
            # The new take films the new routine alone: what was queued ahead of it is dropped, not filmed into it.
            # The take filming that is written first, its last frame the scene as they left it; then their cleanup
            # runs; then the new take starts, from the scene the new routine starts from.
            self._finish_recorder("another recording started")
            self._end_scripts("dropped: another recording started")
            try:
                recorder.start()  # its step count from here, and frame 0 (the probe's, if the scene has not stepped)
            except Exception:
                recorder.close()
                self._close(gen)
                raise
            self.recorder = recorder
            self.last_take = {}  # filming now: until this one is written, no take is the last one
            self.takes_written = [t for t in self.takes_written if t != record]  # nor is an older one of its name
        self.script.append(gen)
        self.scripts_started += 1
        self.script_meta[id(gen)] = {
            "serial": self.scripts_started,
            "name": name,
            "drives_page": bool(drives_page),
            "idle": 0,
            "steps": 0,
            "filmed": bool(record),
            "wall": time.perf_counter(),
        }
        self._end_run("cut short: a routine started")
        return self.scripts_started

    def stop_scripts(self, why: str) -> None:
        """Drop every queued script and finish the take filming them. Main thread."""
        # the take first, as a rebuild writes it: its last frame is the scene as the routine left it, before the
        # routine's own cleanup runs
        self._finish_recorder(why)
        self._end_scripts(f"stopped: {why}")
        self._end_run(f"cut short: {why}")  # held, as it would have been at its last step, not running on unbounded

    @property
    def last_run(self) -> dict:
        """How the last ``run N`` ended (:attr:`runs_ended`'s newest), or ``{}``."""
        ended = self.runs_ended.copy()
        return ended[max(ended)] if ended else {}

    def _end_run(self, how: str, *, hold: bool = True) -> None:
        """End the ``run N`` under way, if one is: :attr:`runs_ended` says how, for the client waiting on it, and with
        ``hold`` the page is held, as it would have been at the run's last step. Main thread; never raises, since the
        loop's own failure handler ends a run."""
        if self.run_budget <= 0:
            return
        sim, finite = self.sim, None
        if how in ("done", "paused"):
            try:
                finite = bool(sim.state_is_finite())
            except Exception:
                finite = None
        ended = {"serial": self.runs, "how": how, "step": int(sim.step_count), "target": self.run_target}
        self.runs_ended = {k: v for k, v in self.runs_ended.items() if k > self.runs - RECENT}
        self.runs_ended[self.runs] = {**ended, "finite": finite}
        self.run_budget = 0
        if hold:
            try:
                self.session.set_paused(True)
            except Exception:
                pass  # the flag the loop reads is set first; only the checkbox, mid-teardown, would not take it

    def page_steps_script(self) -> bool:
        """Whether the page steps the playing script this pass: one that drives through the arm panels, whose
        last ``next()`` only yielded. A script queued that has not played yet is not stepped for."""
        if not self.script or self._head_ready != id(self.script[0]):
            return False
        meta = self.script_meta.get(id(self.script[0])) or {}
        return meta.get("drives_page") is True and not self.head_stepped

    def _arms(self) -> dict:
        return getattr(getattr(self, "session", None), "arms", None) or {}

    def _suspend_arms(self) -> None:
        """Hold the arm panels off a script that drives the scene itself."""
        for panel in self._arms().values():
            if not getattr(panel, "suspended", True):
                panel.suspend()

    def _resume_arms(self) -> None:
        """Put the arm panels back as they were when the script started: held off or live, whatever the page or the
        routine did with them meanwhile."""
        before, self._arms_before = self._arms_before, {}
        for name, panel in self._arms().items():
            was, now = before.get(name), bool(getattr(panel, "suspended", False))
            try:
                if was is False and now:
                    panel.resume()
                elif was is True and not now:
                    panel.suspend()  # one the routine took back that was held off before it
            except Exception:  # a panel of a scene since rebuilt
                pass

    def _end_scripts(self, how: str) -> None:
        """End every queued script, the head first, each reported ``how``."""
        while self.script:
            gen = self.script.pop(0)
            self._ended(gen, how, self._close(gen))
        self._next_script()

    def _close(self, gen) -> str:
        """Close a script, returning what its own ``finally`` raised, if anything; whatever it raises, it is gone."""
        close = getattr(gen, "close", None)  # a generator; a plain iterator has nothing to close
        if close is None:
            return ""
        try:
            close()
        except (Exception, SystemExit):  # a cleanup that calls exit() fails, as a routine that does; the page goes on
            error = traceback.format_exc()[-3000:]
            self.last_error = error
            print(f"[live] a script raised as it was stopped:\n{error}", file=sys.stderr, flush=True)
            return error
        return ""

    def _ended(self, gen, how: str, error: str = "") -> None:
        meta = self.script_meta.pop(id(gen), None) or {"serial": 0, "name": "", "steps": 0, "wall": time.perf_counter()}
        self.last_script = {
            "serial": meta["serial"],
            "name": meta["name"],
            "how": how,
            "error": error,
            "steps": int(meta["steps"]),
            "wall_seconds": round(time.perf_counter() - meta["wall"], 1),
        }
        self.ended[meta["serial"]] = self.last_script
        while len(self.ended) > RECENT:
            del self.ended[min(self.ended)]

    def _next_script(self) -> None:
        """The head script is gone: give the arms back, and let the next one set them as it needs."""
        self.head_stepped, self._head_ready = False, None
        self._resume_arms()

    def _credit(self, gen, serial: int, steps: int) -> None:
        """Steps a pass took -- the routine's own and the page's for it -- to the script that was playing."""
        meta = self.script_meta.get(id(gen))
        if meta is not None:
            meta["steps"] += steps
        elif serial in self.ended:
            self.ended[serial]["steps"] += steps  # it ended this pass, with these steps in it

    def _play(self) -> int:
        """One step of the script at the head of the queue; returns the steps its ``next()`` took.

        Whether the page steps it is the script's kind (:meth:`start_script`,
        :meth:`page_steps_script`): a routine that drives through the arm
        panels is stepped by the page on a pass whose ``next()`` only yielded;
        one that drives the scene itself never is, and is stopped once it has
        yielded :data:`IDLE_LIMIT` times running without stepping, which is a
        routine waiting on a page that will not step for it.
        """
        gen = self.script[0]
        meta = self.script_meta.get(id(gen)) or {}
        if self._head_ready != id(gen):
            self._head_ready = id(gen)
            self._arms_before = {name: bool(getattr(p, "suspended", True)) for name, p in self._arms().items()}
            if not meta.get("drives_page"):
                self._suspend_arms()
        before, mark = int(self.sim.step_count), checks.reset_mark(self.sim)
        try:
            with self._busy("playing one step of a routine", routine=True):
                next(gen)
        except StopIteration:
            self.script.pop(0)
            self._ended(gen, "finished")
            self._next_script()
            print("[live] script finished", flush=True)
        except (Exception, SystemExit):  # a routine that calls exit() fails; the page goes on
            self.script.pop(0)
            failure = traceback.format_exc()[-3000:]
            self._close(gen)
            self.last_error = failure
            self._ended(gen, "failed", failure)
            self._next_script()
            print(f"[live] script failed:\n{failure}", file=sys.stderr, flush=True)
        else:
            took = checks.steps_since(self.sim, before, mark)
            self.head_stepped = took > 0
            if not meta.get("drives_page"):
                meta["idle"] = 0 if self.head_stepped else meta.get("idle", 0) + 1
            if meta.get("idle", 0) >= IDLE_LIMIT:
                self.script.pop(0)
                self._close(gen)
                self.last_error = (
                    f"{meta.get('name') or 'the routine'} yielded {IDLE_LIMIT} times running without stepping the "
                    "scene, and the page does not step a routine that drives the scene itself: step it between "
                    "yields (sim.run(1)), or give the routine a `session` parameter and play it so the page "
                    "steps it through the arm panels"
                )
                self._ended(gen, "failed", self.last_error)
                self._next_script()
                print(f"[live] script stopped: {self.last_error}", file=sys.stderr, flush=True)
            elif self.recorder is not None:
                try:
                    self.recorder.maybe()
                except Exception:
                    # the recording failed, not the routine: drop the take and let the routine play on, saying so
                    recorder, self.recorder = self.recorder, None
                    recorder.close()
                    self.last_error = f"the recording of {recorder.name!r} failed:\n{traceback.format_exc()[-3000:]}"
                    print(f"[live] {self.last_error}", file=sys.stderr, flush=True)
            if self.script and self.script[0] is gen:
                return took
        took = checks.steps_since(self.sim, before, mark)  # the last next() of an ended script stepped too
        if meta.get("filmed"):
            self._finish_recorder("the script ended")  # a routine queued behind it is not filmed into its take
        return took

    def _show_take(self, take: str, moved: str, motion: dict | None = None) -> None:
        """Every take is put beside the clip as it ends, on the board and in the log, and what moved in it is said.

        The strip -- the clip's anchors above, the take's frames at the same
        point of the action below -- is for the eye of the agent and of the
        person watching; nothing here says whether the take did what the clip
        did, because nothing that keys colours can tell. With no timeline the
        take's own motion is said, off its recorded positions.
        """
        if not self._timeline().get("anchors"):
            print(f"[live] take {take!r} (no ref/timeline.json to put it beside): {moved}", flush=True)
            return
        if motion is not None:
            self._motions[take] = motion
        reply = self.cmd_compare(keys=self.last_keys, take=take, note="the take beside the clip as it ended")
        if reply.get("ok"):
            print(f"[live] take {take!r} beside the clip: {reply['strip']}\n{reply['report']}", flush=True)
        else:
            print(f"[live] take {take!r} could not be put beside the clip: {reply.get('error')}", flush=True)

    def _finish_recorder(self, why: str) -> None:
        if self.recorder is None:
            return
        with self._busy("writing the take and putting it beside the clip"):
            self._write_take(why)

    def _write_take(self, why: str) -> None:
        # `recording` names the take until it is written and beside the clip: a client waiting on it reads that
        try:
            self._write_take_files(why)
        finally:
            self.recorder = None

    def _write_take_files(self, why: str) -> None:
        recorder = self.recorder
        try:
            summary = recorder.finish()
            self.last_take = summary
            self.takes_written = [t for t in self.takes_written if t != summary["take"]][-(RECENT - 1) :]
            self.takes_written.append(summary["take"])
            motion = checks.take_motion(recorder.frames)
            moved = checks.motion_line(motion)
            checks.log_append(
                self.run_dir,
                {"cmd": "record", "scene": self.scene_id, "note": moved, **summary},
            )
            print(
                f"[live] take {summary['take']!r} written ({why}): {summary['frames']} frames, {summary['seconds']:.1f} s",
                flush=True,
            )
        except Exception:
            self.last_error = traceback.format_exc()[-3000:]
            print(
                f"[live] the take could not be written:\n{self.last_error}",
                file=sys.stderr,
                flush=True,
            )
            return
        try:
            self._show_take(summary["take"], moved, motion)
        except Exception:
            # written; only putting it beside the clip failed (a timeline.json that will not parse, say)
            self.last_error = (
                f"take {summary['take']!r} was written, but putting it beside the clip failed:\n"
                + traceback.format_exc()[-3000:]
            )
            print(f"[live] {self.last_error}", file=sys.stderr, flush=True)

    def speed(self) -> float | None:
        """Simulated seconds per wall second over the last :data:`SPEED_WINDOW` s; None before there are two samples."""
        samples = list(self._speed)  # a copy: the main thread clears the deque on a reset while a status reads it
        if len(samples) < 2:
            return None
        (w0, t0), (w1, t1) = samples[0], samples[-1]
        return round((t1 - t0) / (w1 - w0), 3) if w1 > w0 else None

    def _sample_speed(self) -> None:
        now, t = time.perf_counter(), float(self.sim.time)
        if self._speed and t < self._speed[-1][1]:
            self._speed.clear()  # a reset or a rebuild: the clock started over
        self._speed.append((now, t))
        while len(self._speed) > 2 and now - self._speed[0][0] > SPEED_WINDOW:
            self._speed.popleft()

    def refresh_live(self, force: bool = False) -> None:
        self._refreshed = time.perf_counter()
        if self.sim is not None:
            try:
                self._sample_speed()
            except Exception:  # a sim mid-swap
                pass
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
            from robosimstudio import registry

            registry.unwatch(self.registrations)
            print("[live] stopped", flush=True)

    def open_page(self) -> None:
        """Build the file (a bare scene if it does not build) and open the page over it. No socket, no loop."""
        # The scene file's own directory, so what it imports by name beside
        # itself resolves -- and so `eval` can import it back the same way.
        sys.path.insert(0, str(self.path.parent))
        self.scene_paths.set_baseline(set(sys.modules))
        from robosimstudio import registry

        registry.watch(self.registrations)
        # Stamped before the build, not after: a save made while the first
        # build runs is a change the watcher must still see.
        files = helper_files(self.path)
        self._walked = (time.perf_counter(), files)
        self._loaded_stamp = self._stamps(files)
        scene_id = checks.scene_hash(self.path, files)
        try:
            loaded = self.load(files)
            error = ""
        except (Exception, SystemExit):
            error = traceback.format_exc()[-3000:]
            print(
                f"[live] {self.path.name} did not build; serving a bare scene instead:\n{error}",
                file=sys.stderr,
            )
            loaded = fallback_loaded()
        self.ns, self.scene, self.sim = loaded.ns, loaded.scene, loaded.sim
        refused = ""
        try:
            self.session = LiveSession(loaded.sim, config=self.config, host=self)
        except Exception:
            if error:
                raise  # the bare scene itself would not go in: nothing to serve
            refused = traceback.format_exc()[-3000:]
        if refused:
            # built, but would not go into the page: the bare scene stands in, as it does after a rebuild
            error = refused
            print(
                f"[live] {self.path.name} built but would not go into the page:\n{error}", file=sys.stderr, flush=True
            )
            # What the failed scene holds goes before the stand-in is built, out here where no traceback names it
            # any more: its renderers (the page's teardown has closed what it opened over it), then, with the
            # server's names and this one, its model and state.
            try:
                loaded.sim.close()
            except Exception:
                pass
            self.ns = self.scene = self.sim = loaded = None
            gc.collect()
            loaded = fallback_loaded()
            self.ns, self.scene, self.sim = loaded.ns, loaded.scene, loaded.sim
            self.session = LiveSession(loaded.sim, config=self.config, host=self)
        self.serving = "stand-in" if error else "scene"
        self.scene_id = "bare" if error else scene_id
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
        """The socket thread's half: put the command on the bus and wait for the main thread's answer.

        ``status`` is the exception: while the main thread is busy -- a
        rebuild, a render, a gate -- it is answered here, at once, with what
        the page is doing and for how long, instead of waiting it out. Any
        other command is started only within the time the client said it would
        wait (``timeout``): one the page cannot start by then is withdrawn and
        answered as not run, and one it started runs to its end.
        """
        waits = float(request.pop("timeout", 0) or 0)
        if request.get("cmd") == "status":
            # a waiting client's poll is answered here, off the main thread, reading nothing off the scene
            return self.light_status() if request.get("light") else self._status_now()
        # a command the page cannot start before the client stops waiting is withdrawn, not run behind its back;
        # one it started runs to its end, and its reply says so (a client that sent no wait gets the bus's)
        limit = max(1.0, waits - 2.0) if waits else self.bus.timeout
        try:
            return self.bus.call_starting_within(limit, self.dispatch, request)
        except BusTimeout:
            busy, _, routine = self.busy_state
            then = (
                "send it again: it goes in between two of the routine's steps"
                if routine
                else "`./ctl status --wait`, then send it again"
            )
            return {
                "ok": False,
                "error": f"the page was {busy or 'busy'} past the wait; `{request.get('cmd')}` was not run -- {then}",
            }
        except BusClosed:
            return {"ok": False, "error": "server shutting down"}

    def _status_now(self) -> dict:
        """A status off the main thread, or, while it is busy, the busy one; one is ever queued at a time.

        Every caller waits on the one request in flight rather than queueing
        its own, so a client polling through a five-minute render leaves one
        status for the main thread to answer after it, not six hundred.
        """
        with self._status_lock:
            pending = self._status_pending
            if pending is None:
                pending = self._status_pending = {"done": threading.Event()}

                def ask(box: dict = pending) -> None:
                    try:
                        box["reply"] = self.bus.call(self.dispatch, {"cmd": "status"})
                    except (BusTimeout, BusClosed) as exc:
                        box["reply"] = {"ok": False, "error": f"server busy: {exc}"}
                    except Exception as exc:  # whatever else: every caller waiting on this one gets a reply
                        box["reply"] = {"ok": False, "error": f"status failed: {type(exc).__name__}: {exc}"}
                    finally:
                        with self._status_lock:
                            if self._status_pending is box:
                                self._status_pending = None
                        box["done"].set()

                threading.Thread(target=ask, name="live-status", daemon=True).start()
        started = time.perf_counter()
        while not pending["done"].wait(0.25):
            busy, _since, routine = self.busy_state
            # a routine's step is the page at work: waited out, but not past what a client waits for its reply
            if busy and (not routine or time.perf_counter() - started > ROUTINE_WAIT):
                return self.busy_status()
        return pending["reply"]

    def busy_status(self, stamps: Any = MISSING_STAMPS) -> dict:
        """What ``status`` says while the main thread is busy, and what every ``status`` starts from.

        Read from the socket thread while the main thread changes these, so
        each is read once: a dict or list copied whole (one step under the
        GIL), the recorder taken into a local before it is asked its name.
        """
        ended, recorder, take = self.ended.copy(), self.recorder, self.last_take
        busy, since, routine = self.busy_state
        return {
            "ok": True,
            "busy": busy,
            "busy_seconds": round(time.perf_counter() - since, 1) if busy else 0.0,
            "busy_routine": routine,
            "url": self.url,
            "share_url": self.share_url,
            "pid": os.getpid(),
            "file": str(self.path),
            "run_dir": str(self.run_dir),
            "pending_save": self._pending_save(stamps),
            "serving": "" if self.serving == "scene" else self.serving,  # said only when it is not the scene
            "server_log": self.log_path,
            "generation": self.generation,
            "resets": self.resets,
            "playing": len(self.script),
            "last_script": self.last_script,
            "last_run": self.last_run,
            "runs_ended": {str(k): v for k, v in self.runs_ended.copy().items()},
            "ended": {str(k): v for k, v in ended.items()},
            "recording": recorder.name if recorder is not None else "",
            "last_take": take.get("take", ""),
            "takes_written": list(self.takes_written),
            "last_error": self.last_error,
        }

    def light_status(self, stamps: Any = MISSING_STAMPS) -> dict:
        """The status a waiting client polls: the busy one, with where the scene is in time; nothing read off it."""
        sim = self.sim
        return {
            **self.busy_status(stamps),
            "step": int(getattr(sim, "step_count", 0)),
            "time": round(float(getattr(sim, "time", 0.0)), 3),
            "paused": bool(getattr(self.session, "paused", False)),
            "speed": self.speed(),
        }

    def _pending_save(self, stamps: Any = MISSING_STAMPS) -> bool:
        """A save the watcher will build -- to the scene file or a helper beside it -- never on a page that does not
        watch, or for a scene file that is gone. ``stamps`` when the caller has just taken them."""
        current = self._stamps() if stamps is MISSING_STAMPS else stamps
        return bool(self.watch_enabled) and current is not None and current != self._loaded_stamp

    @contextmanager
    def _busy(self, what: str, routine: bool = False):
        outer = self.busy_state[0]
        if not outer:
            self.busy_state = (what, time.perf_counter(), routine)
        try:
            yield
        finally:
            if not outer:
                self.busy_state = ("", 0.0, False)

    def dispatch(self, request: dict) -> dict:
        """Run one command on the main thread, between two loop iterations."""
        name = str(request.get("cmd", ""))
        handler = getattr(self, f"cmd_{name}", None)
        if handler is None:
            return {"ok": False, "error": f"unknown command {name!r}"}
        last = self.build_error.strip().splitlines()[-1] if self.build_error.strip() else ""
        if self.serving == "nothing" and name in READS_THE_SCENE:
            return {
                "ok": False,
                "error": f"no scene is in the page: {self.path.name} would not go in, nor the bare scene in its place "
                f"({last}); save a fix to go in, or `./ctl quit` and `./ctl up` -- `./ctl status` has the traceback",
            }
        what = BUSY.get(name, "")
        # what path-traces: a snapshot not taken off the browser, a still compare, a take's first frame
        renders = (
            (name == "snapshot" and not request.get("browser"))
            or (name == "compare" and not request.get("take"))
            or (name in ("play", "eval") and bool(request.get("record")))
        )
        if what and renders and not self.rendered:
            what += FIRST_RENDER
        try:
            with self._busy(what) if what else nullcontext():
                reply = handler(**{k: v for k, v in request.items() if k != "cmd"})
        except (Exception, SystemExit):  # `eval "sys.exit()"` fails; the page goes on
            reply = {"ok": False, "error": traceback.format_exc()[-3000:]}
        if renders and reply.get("ok"):
            self.rendered = True
        if name in READS_THE_SCENE and self._pending_save():
            reply["pending"] = "a save is not built yet: this is the scene before it -- `./ctl status --wait` first"
        if self.build_error and name in READS_THE_SCENE:
            # the page still shows the scene before the save that failed: say so, beside whatever this answered
            if self.serving == "stand-in":
                reply["stale"] = (
                    f"{self.path.name} is not up ({last}); a bare scene stands in for it, not your scene -- "
                    "`./ctl status` has the traceback"
                )
            else:
                reply["stale"] = (
                    f"the last save of {self.path.name} did not build ({last}); "
                    "this is the scene before it -- `./ctl status` has the traceback"
                )
        return reply

    # ------------------------------------------------------------ commands
    def cmd_status(self) -> dict:
        if self.serving == "nothing":
            return self.light_status()  # no scene to read: what failed is in last_error, and `serving` says so
        sim, scene = self.sim, self.scene
        names = sim.names()
        positions = {}
        for name in names:
            try:
                positions[name] = [round(float(v), 4) for v in sim[name].position]
            except Exception:
                positions[name] = "no body"
        current = self._stamps()
        # the last save to scene.py or a helper: a save to either rebuilds
        newest = max([current[0][0], *(s[0] for _, s in current[1] if s)]) if current else 0
        return {
            **self.light_status(current),
            "modified": time.strftime("%H:%M:%S", time.localtime(newest / 1e9)) if current else "missing",
            "loaded": time.strftime("%H:%M:%S", time.localtime(self.loaded_at)),
            "load_seconds": round(self.load_seconds, 2),
            "cell": scene.cell,
            "names": names,
            "cameras": [c.name for c in sim.cameras],
            "finite": bool(sim.state_is_finite()),
            "routines": self.routine_names(),
            "board": list(self.board),
            "positions": positions,
        }

    def cmd_reload(self) -> dict:
        return self.reload("requested")

    def cmd_run(self, steps: int = 1) -> dict:
        if self.script:
            return {
                "ok": False,
                "error": ROUTINE_PLAYING,
            }
        if int(steps) < 1:
            return {"ok": False, "error": f"run needs at least one step, not {steps}; `pause` holds the page"}
        self._end_run("another run started", hold=False)
        self.runs += 1
        self.run_budget = int(steps)
        self.run_target = int(self.sim.step_count) + self.run_budget
        self.session.set_paused(False)
        return {"ok": True, "run": self.runs, "step": int(self.sim.step_count), "target": self.run_target}

    def cmd_pause(self) -> dict:
        self._end_run("paused")
        self.session.set_paused(True)
        return {"ok": True, "paused": True}

    def cmd_resume(self) -> dict:
        self._end_run("resumed: the page runs on", hold=False)
        self.session.set_paused(False)
        reply: dict = {"ok": True, "paused": False}
        if self.script:
            reply["note"] = "a routine is playing; Pause and Resume take effect for the page once it ends"
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
        # the value is the last line's, when that line is an expression: a heredoc that defines a routine and
        # ends by calling it plays it, as one line calling it does
        tree = ast.parse(code, "<eval>")
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        with redirect_stdout(out):
            if tree.body:
                exec(compile(tree, "<eval>", "exec"), ns)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), "<eval>", "eval"), ns)
        playing = False
        serial = 0
        if inspect.isgenerator(value):
            # A script: played one step per loop pass so the page draws it,
            # instead of freezing for as long as it takes. It drives the scene
            # itself (sim.run between yields), with the arm panels held off; a
            # routine that drives through them is bound with a `session` and
            # played. The page carries on as it was after.
            serial = self.start_script(value, name="eval", record=record, every=every, camera=camera)
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
        if serial:
            reply["serial"] = serial
        if record:
            reply["recording"] = str(self.run_dir / f"{record}.mp4")
            reply["take"] = record
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
                    "scene": self.scene_id,
                    "note": note,
                    "png": reply.get("beside") or str(path),  # the pair, when there is one: what was judged
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
        return frames.cut(clip, at=at, every=every, out_dir=Path(out) if out else self.ref_dir(), width=width)

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
            try:
                # the last still compare's keys when none are given, as at the end of a take
                parsed = parsed or self._keys(self.last_keys)
                result = checks.compare_take(
                    self.run_dir, take, line, parsed, self._timeline_dir(timeline), moved=self._motions.pop(take, None)
                )
            except Exception as exc:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            strip = checks.write_strip(base, result.pop("tiles"))
            self.cmd_show(strip, label=f"take {take}: the clip above, the take below")
            n = checks.log_append(
                self.run_dir,
                {"cmd": "compare", "scene": self.scene_id, "note": note, "take": take, "strip": strip},
            )
            (base.with_suffix(".json")).write_text(
                json.dumps({k: v for k, v in result.items() if k != "lines"}, default=checks._plain, indent=1)
            )
            return {
                "ok": True,
                "take": take,
                "strip": strip,
                "report": "\n".join(result["lines"]),
                "logged": n,
                "anchors": result["anchors"],
                "moved": result["moved"],
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
        self.last_keys = [str(k) for k in keys]  # a compare with none leaves a take unoutlined
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
                "scene": self.scene_id,
                "note": note,
                "ref": Path(ref).name,
                "look": pictures["look"],
            },
        )
        return {
            "ok": True,
            "camera": cam.name,
            "ref": ref,
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
                "error": ROUTINE_PLAYING,
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
        except (Exception, SystemExit) as exc:
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
                "scene": self.scene_id,
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
            # a routine that asks for the page's session gets it, to drive the arm panels (reach, grip)
            # or anything else on the page; one that steps the scene itself has the panels held off for it
            wants_session = "session" in inspect.signature(fn).parameters
            gen = fn(self.sim, session=self.session) if wants_session else fn(self.sim)
        except Exception:
            return {"ok": False, "error": traceback.format_exc()[-3000:]}
        if not inspect.isgenerator(gen):
            return {
                "ok": False,
                "error": f"routine {name!r} returned {type(gen).__name__}, not a generator: it yields as it goes (REFERENCE, Acting)",
            }
        serial = self.start_script(gen, name=name, drives_page=wants_session, record=record, every=every, camera=camera)
        reply = {"ok": True, "playing": name, "serial": serial}
        if record:
            reply["recording"] = str(self.run_dir / f"{record}.mp4")
            reply["take"] = record
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


def _log_path() -> str:
    """The file this process's output goes to (``up`` sends it to ``run/serve.log`` or ``--log``), or ``""``."""
    try:
        target = os.readlink("/proc/self/fd/1")
    except OSError:
        return ""
    return target if target.startswith("/") and Path(target).is_file() else ""
