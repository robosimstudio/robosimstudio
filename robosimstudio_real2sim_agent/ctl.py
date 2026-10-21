# SPDX-License-Identifier: Apache-2.0
"""Drive the live page from a shell (standard library only, so it starts in milliseconds).

    ./ctl up                           start a page on scene.py, detached; prints its URL
    ./ctl status                       what is loaded, where things are, the last error
    ./ctl reload                       rebuild scene.py now (saving the file does this too)
    ./ctl snapshot --beside ref.png    render a PNG of the current state, the reference beside it
    ./ctl run 120                      step the page 120 control steps, then hold
    ./ctl eval "sim['cube'].position"  read or drive the live scene
    ./ctl play --wait                  play the scene's routine and return when it ends, with how it ended
    ./ctl status --wait                return once a rebuild or a render under way is done
    ./ctl catalog robots cells         the names rss.robot() and Scene(cell=) accept
    ./ctl ref show ref/photo.jpg       put a picture in the page's sidebar
    ./ctl quit                         stop the server

Commands other than ``up`` send one JSON line over ``<run-dir>/ctl.sock`` and
print the reply (raw with ``--json``). ``run-dir`` is ``./run`` under the
current directory, where ``up`` also starts the server, so the two agree;
``--run-dir`` or ``--sock`` overrides it, and the ``examples/live/ctl`` shim
pins it beside the scene file.

``up`` starts ``python -m robosimstudio_real2sim_agent.serve`` detached, in a
session of its own (an agent's shell may exit at once), logging to
``<run-dir>/serve.log``, and prints the URL once the ready line appears;
``quit`` stops it. If a cold first build outlasts the wait, the server keeps
running and the log is named.

A command that outlasts its timeout does not leave the agent guessing: the
page answers ``status`` at once even while it is busy, so the timeout says
what it is doing and for how long. ``--wait`` on ``play``, ``eval`` and
``status`` blocks until the routine ends or the page is free, printing the
page's log as it goes and a line every ten seconds, so an agent whose shell
returns only when a command does never has to poll.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from .wire import DEFAULT_RUN_DIR, NoReply, NoServer, parse_ready, request, socket_path

__all__ = [
    "CATALOGS",
    "NotReady",
    "ServerDied",
    "build_parser",
    "build_request",
    "main",
    "serve_argv",
    "show",
    "spawn_server",
    "wait_ready",
]

TIMEOUTS = {
    "reload": 330.0,
    "snapshot": 600.0,
    "eval": 300.0,
    "play": 300.0,
    "catalog": 120.0,
    "compare": 600.0,
    "check": 600.0,
    "ref": 600.0,
}
"""Seconds to wait for a reply, per command; :data:`DEFAULT_TIMEOUT` for the rest.

``play`` waits as long as ``eval``: with ``--record`` both render the take's
first frame before they answer. ``snapshot`` waits as long as ``compare``: the
first path-traced picture in a process compiles the renderer, for minutes."""
DEFAULT_TIMEOUT = 30.0
WAIT_LIMIT = 3600.0
"""How long ``--wait`` waits by default [s]."""
SAY_EVERY = 10.0
"""Seconds between the progress lines ``--wait`` prints while nothing else is printed."""
START_HINT = "python -m robosimstudio_real2sim_agent up    (from the directory holding scene.py)"
SERVE_FLAGS = ("hold", "no_watch", "share", "dark", "hide_scenery", "camera_view")
"""``up``'s switches that are the server's, spelled as argparse attributes."""
CATALOGS = ("robots", "grippers", "cells", "assets")
"""What ``catalog`` can list; ``assets`` runs the registries' discovery, which takes seconds."""
VERBOSE = ("readings", "anchors", "entries", "checks", "moved", "objects", "drift_mm")
"""Reply fields printed only under ``--json``: the raw numbers behind a ``report``."""


class ServerDied(RuntimeError):
    """The server exited before it printed the ready line; the message is the log's tail."""


class NotReady(TimeoutError):
    """The server is still running but had not printed the ready line by the deadline."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m robosimstudio_real2sim_agent",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument(
        "--run-dir",
        default=DEFAULT_RUN_DIR,
        help="the page's socket, log and snapshots (default run/)",
    )
    parser.add_argument("--sock", default="", help="the control socket (default <run-dir>/ctl.sock)")
    parser.add_argument("--json", action="store_true", help="print the raw reply")
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS, so a subcommand's own --json adds to the global one rather than
    # resetting it: `--json status` and `status --json` both print the raw reply.
    common.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="print the raw reply",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser(
        "up",
        parents=[common],
        help="start a page on a scene file, detached, and print its URL",
    )
    up.add_argument("--port", type=int, default=8080)
    up.add_argument("--scene", default="scene.py", help="the file to execute and watch")
    up.add_argument("--share", action="store_true", help="ask viser for a public URL as well")
    # Kept, not documented: the page has one shape, and these are the knobs a
    # person who has read the server's source may turn.
    up.add_argument("--hold", action="store_true", help=argparse.SUPPRESS)
    up.add_argument("--no-watch", action="store_true", help=argparse.SUPPRESS)
    up.add_argument("--dark", action="store_true", help=argparse.SUPPRESS)
    up.add_argument("--hide-scenery", action="store_true", help=argparse.SUPPRESS)
    up.add_argument("--camera-view", action="store_true", help=argparse.SUPPRESS)
    up.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="seconds to wait for the page (default 300)",
    )
    up.add_argument(
        "--log",
        default="",
        help="where the server's output goes (default <run-dir>/serve.log)",
    )
    st = sub.add_parser(
        "status",
        parents=[common],
        help="what is loaded, where things are, the last error; what the page is busy with, if it is",
    )
    _wait_flags(st, "return once the page is not busy and every save has been built")
    sub.add_parser("reload", parents=[common], help="rebuild the scene file now")
    snap = sub.add_parser("snapshot", parents=[common], help="render a PNG of the current state")
    snap.add_argument(
        "--camera",
        default="",
        help="a camera the scene declares; default: the first one, or a 3/4 view",
    )
    snap.add_argument("--out", default="", help="where to write (default <run-dir>/snap_NNN.png)")
    snap.add_argument("--beside", default="", help="a reference image to put beside it (_vs_ref.png)")
    snap.add_argument("--size", default="", help="WxH")
    snap.add_argument(
        "--eye",
        default="",
        help="x,y,z world metres: render from here instead of a declared camera",
    )
    snap.add_argument("--look-at", default="", help="x,y,z world metres, with --eye")
    snap.add_argument(
        "--browser",
        action="store_true",
        help="what the connected browser shows, from its own viewpoint",
    )
    snap.add_argument(
        "--note",
        default="",
        help="one line for the log: what this snapshot is checking",
    )
    keyed = argparse.ArgumentParser(add_help=False)
    keyed.add_argument(
        "--camera",
        default="",
        help="a camera the scene declares; default: the first one",
    )
    keyed.add_argument(
        "--key",
        action="append",
        default=[],
        metavar="NAME=HOW",
        help="how to find a thing in the photo: case=red, case=#d01a10, socket=@1240,410 (a pixel of the photo "
        "you looked at), case=red@980,640, gasket=mask:ref/gasket.png (not black = the thing); repeatable. "
        "Optional: without keys, compare still writes the pair and the look sheet",
    )
    cmp = sub.add_parser(
        "compare",
        parents=[common, keyed],
        help="the scene beside a photo: the pair and a look sheet, and per keyed thing its size, place and overlap",
    )
    cmp.add_argument(
        "--ref",
        default="",
        help="the photograph (default: the timeline's first anchor)",
    )
    cmp.add_argument("--size", default="", help="WxH")
    cmp.add_argument("--out", default="", help="where to write (default <run-dir>/cmp_NNN*.png)")
    cmp.add_argument("--note", default="", help="one line for the log: what this edit changed")
    cmp.add_argument(
        "--take",
        default="",
        help="a take recorded with --record: the clip's anchors above the take's frames, and what moved in the take",
    )
    cmp.add_argument(
        "--timeline",
        default="",
        help="the timeline.json to use with --take (default ref/timeline.json)",
    )
    chk = sub.add_parser(
        "check",
        parents=[common, keyed],
        help="what a photo cannot tell: builds, finite, camera, things present, overlap, drift; PASS or FAIL with the number",
    )
    chk.add_argument(
        "--object",
        action="append",
        default=[],
        dest="objects",
        metavar="NAME",
        help="a thing that must be in the scene; repeatable (default: the keyed ones)",
    )
    chk.add_argument(
        "--steps",
        type=int,
        default=60,
        help="control steps the scene is left alone for (default 60)",
    )
    chk.add_argument(
        "--settle-mm",
        type=float,
        default=2.0,
        help="how far a thing may drift left alone (default 2)",
    )
    chk.add_argument(
        "--penetration-mm",
        type=float,
        default=1.0,
        help="the deepest overlap allowed (default 1)",
    )
    chk.add_argument(
        "--timeline",
        default="",
        help="the clip's timeline.json (default ref/timeline.json)",
    )
    chk.add_argument("--note", default="")
    lg = sub.add_parser(
        "log",
        parents=[common],
        help="every compare, check, take and snapshot --note so far, with what it left (PASS/FAIL, the picture) "
        "and the scene.py it measured",
    )
    lg.add_argument("--last", type=int, default=0, help="only the last N")
    run = sub.add_parser("run", parents=[common], help="step N control steps, then hold")
    run.add_argument("steps", type=int)
    run.add_argument("--no-wait", action="store_true")
    sub.add_parser("pause", parents=[common])
    sub.add_parser("resume", parents=[common], help="run until paused again")
    sub.add_parser("reset", parents=[common], help="put the scene back to its start")
    ev = sub.add_parser(
        "eval",
        parents=[common],
        help="python against the live scene: sim, scene, session, np, rss, and the scene file's names",
    )
    ev.add_argument("code", nargs="*", help="an expression or statements; - reads stdin")
    ev.add_argument(
        "--record",
        default="",
        metavar="NAME",
        help="record the script this plays: <run-dir>/NAME.mp4, .json, _seg.npz",
    )
    ev.add_argument(
        "--every",
        type=int,
        default=4,
        help="with --record: a frame per this many control steps (default 4)",
    )
    ev.add_argument(
        "--camera",
        default="",
        help="with --record: the camera to record through (default: the first)",
    )
    _wait_flags(ev, "when the code returns a routine: return when it ends, saying how, its output streamed")
    cat = sub.add_parser("catalog", parents=[common], help="the names the registries know, by kind")
    cat.add_argument("kinds", nargs="*", help=f"any of {', '.join(CATALOGS)}; none means all of them")
    ref = sub.add_parser(
        "ref",
        parents=[common],
        help="a reference picture in the sidebar, or stills out of a clip",
    )
    ref.add_argument("op", choices=("show", "hide", "frames"))
    ref.add_argument(
        "path",
        nargs="?",
        default="",
        help="a picture to show, or the clip to cut frames from",
    )
    ref.add_argument(
        "--at",
        default="0,end",
        help="frames: seconds to cut, with labels: 0:closed,3.7:lid_up,end (default 0,end)",
    )
    ref.add_argument(
        "--every",
        type=float,
        default=0.0,
        help="frames: the contact sheet's spacing in seconds (default: about 24 tiles)",
    )
    ref.add_argument(
        "--out",
        default="",
        help="frames: where the stills and timeline.json go (default: ref/ beside scene.py)",
    )
    ref.add_argument(
        "--width",
        type=int,
        default=0,
        help="frames: scale the stills to this width (default: the clip's)",
    )
    play = sub.add_parser(
        "play", parents=[common], help="play a routine the scene file binds, as its button on the page does"
    )
    play.add_argument("name", nargs="?", default="", help="which, when `routines` binds several (default: the first)")
    play.add_argument("--record", default="", metavar="NAME", help="film it: <run-dir>/NAME.mp4, .json, _seg.npz")
    play.add_argument("--every", type=int, default=4, help="with --record: a frame per this many control steps")
    play.add_argument("--camera", default="", help="with --record: the camera to record through")
    _wait_flags(play, "return when the routine ends, saying how, its output streamed")
    show = sub.add_parser(
        "show", parents=[common], help="put a picture on the page for the person watching, under a label"
    )
    show.add_argument("path", nargs="?", default="", help="a PNG or JPEG; a label replaces the picture that had it")
    show.add_argument("--label", default="", help="the caption (default: the file's name)")
    show.add_argument("--clear", action="store_true", help="take every picture off the page instead")
    sub.add_parser("quit", parents=[common], help="stop the server")
    return parser


def _wait_flags(sub: argparse.ArgumentParser, what: str) -> None:
    sub.add_argument("--wait", action="store_true", help=what)
    sub.add_argument(
        "--wait-limit",
        type=float,
        default=WAIT_LIMIT,
        metavar="S",
        help=f"with --wait: give up after S seconds (default {WAIT_LIMIT:.0f})",
    )


def _absolute(path: str) -> str:
    return str(Path(path).expanduser().resolve()) if path else ""


def _keys(keys: list[str]) -> list[str]:
    """The keys as given, a mask key's file made absolute: the page reads it from its own directory."""
    out = []
    for key in keys:
        name, sep, how = key.partition("=")
        masked = sep and how.strip().startswith("mask:")
        out.append(f"{name}=mask:{_absolute(how.strip()[len('mask:') :].strip())}" if masked else key)
    return out


def build_request(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[dict, float]:
    """The JSON object a parsed command line sends, and how long to wait for its answer."""
    req: dict = {"cmd": args.cmd}
    if args.cmd == "snapshot":
        req.update(
            camera=args.camera,
            out=_absolute(args.out),
            beside=_absolute(args.beside),
            size=args.size,
            eye=args.eye,
            look_at=args.look_at,
            browser=args.browser,
            note=args.note,
        )
    elif args.cmd == "compare":
        req.update(
            ref=_absolute(args.ref),
            camera=args.camera,
            keys=_keys(list(args.key)),
            size=args.size,
            out=_absolute(args.out),
            note=args.note,
            take=args.take,
            timeline=_absolute(args.timeline),
        )
    elif args.cmd == "check":
        req.update(
            camera=args.camera,
            keys=_keys(list(args.key)),
            objects=list(args.objects),
            steps=args.steps,
            settle_mm=args.settle_mm,
            penetration_mm=args.penetration_mm,
            timeline=_absolute(args.timeline),
            note=args.note,
        )
    elif args.cmd == "log":
        req["last"] = args.last
    elif args.cmd == "play":
        req.update(name=args.name, record=args.record, every=args.every, camera=args.camera)
    elif args.cmd == "show":
        if not args.path and not args.clear:
            parser.error("show needs a picture, or --clear")
        req.update(path=_absolute(args.path), label=args.label, clear=args.clear)
    elif args.cmd == "run":
        req["steps"] = args.steps
    elif args.cmd == "eval":
        code = " ".join(args.code)
        if code.strip() == "-" or not code:
            code = sys.stdin.read()
        req["code"] = code
        if args.record:
            req.update(record=args.record, every=args.every, camera=args.camera)
    elif args.cmd == "catalog":
        unknown = [k for k in args.kinds if k not in CATALOGS]
        if unknown:
            parser.error(f"no catalog named {', '.join(unknown)}; one of {', '.join(CATALOGS)}")
        req["kinds"] = list(args.kinds)
    elif args.cmd == "ref":
        if args.op in ("show", "frames") and not args.path:
            parser.error(f"ref {args.op} needs a path")
        req.update(op=args.op, path=_absolute(args.path))
        if args.op == "frames":
            req.update(
                at=args.at,
                every=args.every,
                out=_absolute(args.out) if args.out else "",
                width=args.width,
            )
    return req, TIMEOUTS.get(args.cmd, DEFAULT_TIMEOUT)


def wait_for_steps(sock: Path, reply: dict, steps: int) -> dict:
    """After ``run N``: poll until the page has stepped N times or holds again.

    A rebuild (a save) or a reset under the run ends it: the step count starts
    over, so the target would never be reached, and the reply says what
    happened instead of waiting out the deadline. A reset is read off the
    page's reset count, not the step count, which may have climbed back past
    ``start`` by the next poll.
    """
    start = int(reply["step"])
    target = start + int(steps)
    deadline = time.time() + 30.0 + steps / 5.0
    busy = ""
    while time.time() < deadline:
        time.sleep(0.2)
        status = request(sock, {"cmd": "status"}, DEFAULT_TIMEOUT)
        if not status.get("ok"):
            return status
        busy = status.get("busy", "")
        if busy:
            continue  # a rebuild or a render under way: the next poll says how the run fared
        if status.get("generation", reply.get("generation")) != reply.get("generation"):
            return {"ok": False, "error": "the scene was rebuilt during the run; the run ended", "step": status["step"]}
        if status.get("resets", reply.get("resets")) != reply.get("resets") or status["step"] < start:
            return {"ok": False, "error": "the scene was reset during the run; the run ended", "step": status["step"]}
        if status["step"] >= target or (status["paused"] and status["playing"] == 0):
            return {
                "ok": True,
                "step": status["step"],
                "time": status["time"],
                "finite": status["finite"],
            }
    if busy:
        return {"ok": False, "error": f"the page was still {busy} when the wait ran out; how the run went is unknown"}
    return reply


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def busy_note(sock: Path) -> str:
    """What the page is busy with, as a clause to follow a timeout, or ``""``."""
    try:
        status = request(sock, {"cmd": "status"}, 5.0)
    except (NoServer, NoReply):
        return ""
    if not status.get("busy"):
        return ""
    return (
        f"; the page is busy {status['busy']} ({status.get('busy_seconds', 0):.0f} s so far) and carries on -- "
        "`status --wait` returns when it is free"
    )


def _ref_dir_without_a_page(run_dir: Path) -> Path | None:
    """Where the page would put a clip's stills, ``ref/`` beside ``scene.py``, found without asking it.

    The run directory sits beside ``scene.py`` (the ``ctl`` shim pins it
    there); failing that, the directory ``ctl`` was run from holds the file.
    """
    for folder in (run_dir.resolve().parent, Path.cwd()):
        if (folder / "scene.py").is_file():
            return folder / "ref"
    return None


def server_log(sock: Path, run_dir: Path) -> Path:
    """The log the page writes to, as the page says (``up --log`` sends it elsewhere), else ``<run-dir>/serve.log``.

    A page that cannot tell (no ``/proc``) leaves the note ``up`` wrote beside
    it, ``serve.log.path``.
    """
    try:
        named = request(sock, {"cmd": "status"}, 5.0).get("log")
    except (NoServer, NoReply):
        named = ""
    if not named:
        try:  # "<pid> <path>", trusted only for the page that pid still is
            pid, _, path = (run_dir / "serve.log.path").read_text().strip().partition(" ")
            named = path if pid == str(request(sock, {"cmd": "status"}, 5.0).get("pid")) else ""
        except (OSError, NoServer, NoReply):
            named = ""
    return Path(named) if named else run_dir / "serve.log"


def wait_until_free(sock: Path, limit: float) -> dict:
    """Poll ``status`` until the page is not busy and every save has been built; the last status."""
    start = said = time.monotonic()
    while True:
        status = request(sock, {"cmd": "status"}, DEFAULT_TIMEOUT)
        if not status.get("ok"):
            return status
        doing = status.get("busy") or ("a save not built yet" if status.get("pending_save") else "")
        if not doing:
            return status
        now = time.monotonic()
        if now - start > limit:
            return {"ok": False, "error": f"still {doing} after {limit:.0f} s; the page carries on"}
        if now - said >= SAY_EVERY:
            _say(f"waiting: {doing} ({status.get('busy_seconds', now - start):.0f} s)")
            said = now
        time.sleep(0.5)


def _log_size(log: Path) -> int:
    try:
        return log.stat().st_size
    except OSError:
        return 0


def wait_for_script(sock: Path, reply: dict, log: Path, offset: int, limit: float) -> dict:
    """After ``play`` or ``eval`` started a routine: stream the page's log until it ends, then say how.

    The routine is known by the serial the page gave it; ``status`` names the
    last one to end. Its printed output, a failure's traceback and a take's
    summary land in the page's log, which is printed as it grows.
    """
    serial = int(reply.get("serial") or 0)
    take = reply.get("take") or (Path(reply["recording"]).stem if reply.get("recording") else "")
    start = said = time.monotonic()
    status: dict = {}

    def stream() -> None:
        nonlocal offset, said
        size = _log_size(log)
        if size > offset:
            with open(log, "rb") as f:
                f.seek(offset)
                chunk = f.read(size - offset)
            offset = size
            text = chunk.decode(errors="replace").rstrip("\n")
            if text:
                print(text, flush=True)
            said = time.monotonic()

    while True:
        stream()
        status = request(sock, {"cmd": "status"}, DEFAULT_TIMEOUT)
        if not status.get("ok"):
            return status
        ended = status.get("last_script") or {}
        # ended; and a take it filmed is no longer filming, nor being written and put beside the clip
        filming = take and (status.get("busy") or status.get("recording") == take)
        if int(ended.get("serial") or 0) >= serial and not filming:
            break
        now = time.monotonic()
        if now - start > limit:
            return {"ok": False, "error": f"the routine is still playing after {limit:.0f} s; it carries on"}
        if now - said >= SAY_EVERY:
            where = f"step {status['step']}, t = {status['time']:.1f} s" if "step" in status else status.get("busy", "")
            _say(f"playing: {where} ({now - start:.0f} s)")
            said = now
        time.sleep(0.5)
    stream()  # what the page printed while the last status was on its way
    how = str(ended.get("how", ""))
    out = {
        "ok": how == "finished",
        "routine": ended.get("name", ""),
        "ended": how,
        "steps": ended.get("steps"),
        "wall_seconds": ended.get("wall_seconds"),
        "speed": status.get("speed"),
        "take": (status.get("last_take") if status.get("last_take") == take else f"{take} (not written; see the log)")
        if take
        else "",
        "recording": reply.get("recording", ""),
    }
    if how != "finished":
        out["error"] = (ended.get("error") or how).rstrip()
    return out


# ------------------------------------------------------------------ up
def serve_argv(args: argparse.Namespace, sock: Path) -> list[str]:
    """The command line ``up`` starts the server with: this interpreter, the serve module, the same socket."""
    argv = [
        sys.executable,
        "-m",
        "robosimstudio_real2sim_agent.serve",
        "--port",
        str(args.port),
        "--scene",
        str(args.scene),
        "--run-dir",
        str(args.run_dir),
        "--sock",
        str(sock),
    ]
    for flag in SERVE_FLAGS:
        if getattr(args, flag, False):
            argv.append("--" + flag.replace("_", "-"))
    return argv


def spawn_server(argv: list[str], log_path: Path) -> subprocess.Popen:
    """Start the server in a session of its own, its output to ``log_path`` (truncated first)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as log:
        return subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def wait_ready(log_path: Path, alive: Callable[[], bool], deadline: float, poll: float = 0.5) -> dict:
    """Poll the log for the ready line until ``deadline`` (a ``time.monotonic`` value).

    The line is looked for before the process is, so a server that printed it
    and then died still counts as having come up -- that is a later failure
    with its own message, not a start-up one.
    """
    while True:
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        ready = parse_ready(text)
        if ready is not None:
            return ready
        if not alive():
            raise ServerDied(text[-3000:])
        if time.monotonic() >= deadline:
            raise NotReady(text[-3000:])
        time.sleep(poll)


def _differs_from_running(args: argparse.Namespace, status: dict) -> str:
    """Why the page already up is not the one ``up`` was asked for, or ``""``.

    ``up`` on a running page is a no-op that names it -- unless the scene file
    or ``--share`` asked for differs, which a no-op would silently ignore.
    """
    running = status.get("file", "")
    if running and Path(running).resolve() != Path(args.scene).resolve():
        return (
            f"a page is already up on {running}, not {Path(args.scene).resolve()}; "
            "`quit` it first, or pick another --run-dir"
        )
    if args.share and not status.get("share_url"):
        return "a page is already up with no public URL; `quit` it and run `up --share` again"
    return ""


def cmd_up(args: argparse.Namespace, sock: Path) -> int:
    scene = Path(args.scene)
    if not scene.is_file():
        print(
            f"ctl: no scene file at {scene}; the server would serve a bare scene in its place",
            file=sys.stderr,
        )
        return 2
    try:
        status = request(sock, {"cmd": "status"}, 5.0)
    except NoServer:
        if sock.exists() and not sock.is_socket():
            print(f"ctl: {sock} exists and is not a socket; name another --sock", file=sys.stderr)
            return 2
        try:
            sock.unlink(missing_ok=True)  # a socket file with nobody behind it
        except OSError as exc:
            print(f"ctl: a dead socket at {sock} cannot be removed ({exc}); name another --sock", file=sys.stderr)
            return 2
    except NoReply:
        print(
            f"ctl: a server at {sock} is not answering; `quit` it, or pick another --run-dir",
            file=sys.stderr,
        )
        return 1
    else:
        conflict = _differs_from_running(args, status)
        if conflict:
            print(f"ctl: {conflict}", file=sys.stderr)
            return 1
        reply = {
            "ok": True,
            "already": True,
            "url": status.get("url", ""),
            "share": status.get("share_url"),
            "socket": str(sock),
            "pid": status.get("pid"),
        }
        return show(reply, args.json)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)  # before anything is written into it
    log_path = Path(args.log) if args.log else run_dir / "serve.log"
    proc = spawn_server(serve_argv(args, sock), log_path)
    (run_dir / "serve.pid").write_text(f"{proc.pid}\n")
    (run_dir / "serve.log.path").write_text(f"{proc.pid} {log_path.resolve()}\n")  # for a page with no /proc
    try:
        ready = wait_ready(log_path, lambda: proc.poll() is None, time.monotonic() + args.timeout)
    except ServerDied as exc:
        print(
            f"ctl: the server exited {proc.returncode} before the page was up; {log_path} ends:\n{exc}",
            file=sys.stderr,
        )
        return 1
    except NotReady as exc:
        print(
            f"ctl: no page after {args.timeout:.0f} s; the server (pid {proc.pid}) is still running, "
            f"and {log_path} ends:\n{exc}",
            file=sys.stderr,
        )
        return 1
    reply = {
        "ok": True,
        "url": ready["url"],
        "share": ready.get("share"),
        "socket": ready.get("sock") or str(sock),
        "pid": proc.pid,
        "log": str(log_path),
    }
    return show(reply, args.json)


# ------------------------------------------------------------------ output
def _fmt(value) -> str:
    if isinstance(value, list):
        return ", ".join(_fmt(v) for v in value) if value else "(none)"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, str) and "\n" in value:
        return "\n    " + value.rstrip().replace("\n", "\n    ")
    return str(value)


def show(reply: dict, as_json: bool) -> int:
    if as_json:
        print(json.dumps(reply, indent=1))
        return 0 if reply.get("ok") else 1
    if not reply.get("ok"):
        print(f"error: {reply.get('error', '?')}", file=sys.stderr)
        return 1
    for key, value in reply.items():
        if key == "ok" or value in ("", None) or key in VERBOSE:
            continue
        if isinstance(value, dict):
            for sub, item in value.items():
                print(f"{key}.{sub}: {_fmt(item)}")
        else:
            print(f"{key}: {_fmt(value)}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sock = socket_path(args.run_dir, args.sock)
    if args.cmd == "up":
        return cmd_up(args, sock)
    req, timeout = build_request(args, parser)
    waiting = getattr(args, "wait", False)
    log = server_log(sock, Path(args.run_dir)) if waiting and args.cmd in ("play", "eval") else Path()
    offset = _log_size(log)
    try:
        if args.cmd == "status" and waiting:
            reply = wait_until_free(sock, args.wait_limit)
        else:
            reply = request(sock, req, timeout)
        if args.cmd == "run" and reply.get("ok") and not args.no_wait:
            reply = wait_for_steps(sock, reply, args.steps)
        if args.cmd in ("play", "eval") and waiting and reply.get("ok") and reply.get("serial"):
            started = reply
            reply = wait_for_script(sock, reply, log, offset, args.wait_limit)
            # what the command itself answered stays: an eval's printed output, the note that the scene is stale
            reply.update({k: started[k] for k in ("stdout", "value", "stale") if started.get(k)})
    except NoServer:
        if args.cmd == "ref" and req.get("op") == "frames":
            # nothing about cutting stills needs the page; the stills go where the page would put them
            from . import frames

            out = req["out"] or _ref_dir_without_a_page(Path(args.run_dir))
            if out is None:
                print("ctl: no page and no scene.py beside the run directory or here; give --out", file=sys.stderr)
                return 2
            return show(
                frames.cut(req["path"], at=req["at"], every=req["every"], out_dir=out, width=req["width"]), args.json
            )
        print(f"ctl: no server at {sock}\nstart one with:  {START_HINT}", file=sys.stderr)
        return 2
    except NoReply as exc:
        print(f"ctl: {exc}{busy_note(sock)}", file=sys.stderr)
        return 1
    return show(reply, args.json)


if __name__ == "__main__":
    sys.exit(main())
