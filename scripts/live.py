# SPDX-License-Identifier: Apache-2.0
"""Open the live page on a scene file and rebuild it on every save.

    uv run python scripts/live.py --scene examples/live/scene.py --run-dir examples/live/run --port 8080

The page in the foreground, its log in this terminal, on any file that binds
``scene`` to an ``rss.Scene``. ``python -m robosimstudio_real2sim_agent up`` is the same
server started detached and ``python -m robosimstudio_real2sim_agent`` (``./ctl`` in
``examples/live/``) is the command line onto either; the docs' real-to-sim
procedure page is the contract. All this file does is stand with the other doors.
"""

from robosimstudio_real2sim_agent.serve import main

if __name__ == "__main__":
    raise SystemExit(main())
