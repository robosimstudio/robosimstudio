# SPDX-License-Identifier: Apache-2.0
"""Serve the live page on a scene file, in the foreground.

    python -m robosimstudio_real2sim_agent.serve --scene examples/live/scene.py --run-dir examples/live/run --port 8080

The page's log is this terminal; ``python -m robosimstudio_real2sim_agent up`` is the
same server started detached, and ``python -m robosimstudio_real2sim_agent`` is the
command line onto either. ``--run-dir`` (default ``run/`` under the working directory)
is where the socket is: ``./ctl`` in ``examples/live/`` looks in
``examples/live/run``. ``scripts/live.py`` is this, listed with the other doors. All
this file does is turn a command line into a :class:`~robosimstudio_studio.StudioConfig`
and a :class:`~.server.LiveServer`.

``--camera-view`` is the studio's *live camera view* -- a scene camera
rendered into the sidebar as the loop runs -- and is off here by default
because every frame it draws is time the rebuild loop does not get; the word
"live" on that switch is the studio's, not this package's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robosimstudio_studio.session import StudioConfig

from .wire import DEFAULT_RUN_DIR, socket_path

__all__ = ["config_from_args", "create_parser", "main"]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m robosimstudio_real2sim_agent.serve",
        description="a studio page that rebuilds a scene file under itself",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--scene", default="scene.py", help="the file to execute and watch")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, help="snapshots, the studio's output and the socket")
    parser.add_argument("--sock", default="", help="control socket (default <run-dir>/ctl.sock)")
    parser.add_argument("--no-watch", action="store_true", help="rebuild only on `ctl reload`")
    parser.add_argument("--hold", action="store_true", help="land every rebuild paused (default: running)")
    parser.add_argument("--share", action="store_true", help="a public URL through viser's tunnel as well")
    parser.add_argument("--dark", action="store_true")
    parser.add_argument("--hide-scenery", action="store_true", help="do not draw the cell's furniture")
    parser.add_argument("--camera-view", action="store_true", help="the studio's live camera view in the sidebar")
    return parser


def config_from_args(args: argparse.Namespace) -> StudioConfig:
    return StudioConfig(
        port=args.port,
        share=args.share,
        panels="all",
        render_every=1,
        live=args.camera_view,
        show_scenery=not args.hide_scenery,
        dark_mode=args.dark,
        out=str(Path(args.run_dir) / "studio"),
    )


def main(argv=None) -> int:
    from .server import LiveServer

    args = create_parser().parse_args(argv)
    LiveServer(
        scene_path=Path(args.scene),
        sock_path=socket_path(args.run_dir, args.sock),
        run_dir=Path(args.run_dir),
        config=config_from_args(args),
        watch=not args.no_watch,
        hold=args.hold,
    ).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
