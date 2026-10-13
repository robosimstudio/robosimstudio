# SPDX-License-Identifier: Apache-2.0
"""A grasp the way a person makes one on the page: the arm panel, driven without a browser.

The examples promise a gesture -- drag the hand down onto the thing, press
**Close**, drag it up -- and the only honest check of that promise goes through
the code the gesture goes through. So nothing here writes a joint target or a
gripper command: it moves the arm panel's cursor (what the gizmo's callback
writes), asks for a grip (what the button writes) and lets
:meth:`~robosimstudio_studio.session.StudioSession.tick` do the rest, IK, the
setpoint law and the servo included. Nothing is pinned, so what comes up with the
hand came up because the fingers held it.

Importing this module imports nothing of the library.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["Grasp", "drag_to_grasp", "go", "open_session", "tick"]


def open_session(sim, port: int, page: Callable | None = None):
    """A studio session on ``sim`` with the page an example opens, and no browser, pacing or tunnel."""
    from robosimstudio_studio.session import StudioConfig, StudioSession

    session = StudioSession(sim, config=StudioConfig(port=port, panels="drive", speed="max", live=False))
    if page is not None:
        page(session)
    session.begin_run()
    return session


def tick(session, steps: int) -> None:
    """Step the page ``steps`` control steps; a pass that did not step does not count."""
    done = 0
    while done < steps:
        if session.tick():
            done += 1


def go(session, arm: str, target, *, steps: int = 600, tolerance: float = 0.004, speed: float | None = None) -> float:
    """Put the arm's cursor at ``target`` (world [m]) and step until the tool centre is there.

    Args:
        speed: Walk the cursor there at this speed [m/s], held as a drag until it
            arrives, the way a mouse moves the gizmo. ``None`` puts it there in one
            jump. The difference matters on a scene whose arm has no device IK
            (a coupled rope): there the page commands each cursor update directly,
            with no setpoint law to limit the rate, so a jump is a flick.

    Returns the distance left [m] -- under ``tolerance`` when the hand arrived.
    """
    panel = session.arms[arm]
    target = np.asarray(target, dtype=np.float64)
    if speed:
        start = np.array(panel.cursor_pos, dtype=np.float64)
        stride = float(speed) / float(session.env.request.physics.control_hz)
        legs = max(1, int(np.ceil(np.linalg.norm(target - start) / stride)))
        for k in range(1, legs + 1):
            panel.cursor_pos, panel.dragging = start + (target - start) * (k / legs), k < legs
            tick(session, 1)
    panel.cursor_pos, panel.dragging = target, False
    for _ in range(steps):
        tick(session, 1)
        if _miss(panel, session, target) < tolerance:
            break
    return _miss(panel, session, target)


def _miss(panel, session, target) -> float:
    return float(np.linalg.norm(panel.ik.ee_pose(session.env.state)[0] - target))


@dataclass(frozen=True)
class Grasp:
    """What a lift did to the thing in the hand, measured from the points it was handed."""

    rise: float
    """How far the points' centroid rose from where it lay before the hand came down [m]."""
    drift: float
    """The worst change of the centroid's offset from the tool centre, from the close to the end of the hold [m]."""
    width: float
    """The jaw opening at the end of the hold [m]: the thing's own width when the fingers are on it."""
    reached: float
    """How far the tool centre ended from the lift target [m]."""
    finite: bool


def drag_to_grasp(
    session,
    arm: str,
    points: Callable[[], np.ndarray],
    *,
    at=None,
    hover: float = 0.0,
    lift: float = 0.10,
    close: float = 1.0,
    hold: float = 2.0,
    speed: float | None = None,
    grip: float = 1.0,
) -> Grasp:
    """Down onto it, **Close**, up, and hold: the gesture the examples ask for.

    Args:
        points: The thing's tracked points in world [m], ``(N, 3)`` or ``(3,)`` --
            a rigid body's position, a sponge's nodes, the particles a pinch took.
        at: Where the tool centre goes to close [m, world]; the points' centroid
            when not given.
        hover: Go to ``at`` raised by this much first, then straight down.
        lift: How far the hand rises after the close [m].
        close: Seconds of stepping with the grip asked for, before the lift.
        hold: Seconds held at the top, over which ``drift`` is measured.
        speed: Drag the cursor at this speed [m/s] rather than jumping it (see :func:`go`).
        grip: What the hand closes to: 1.0 is the **Close** button, anything less the
            **Grip amount** slider.
    """
    hz = float(session.env.request.physics.control_hz)

    def centroid() -> np.ndarray:
        return np.asarray(points(), dtype=np.float64).reshape(-1, 3).mean(axis=0)

    def hand() -> np.ndarray:
        return session.arms[arm].ik.ee_pose(session.env.state)[0]

    before = centroid()
    target = before if at is None else np.asarray(at, dtype=np.float64)
    if hover:
        go(session, arm, target + np.array([0.0, 0.0, hover]), speed=speed)
    go(session, arm, target, speed=speed)
    session.arms[arm].grip_request = float(grip)
    tick(session, round(close * hz))
    offset = centroid() - hand()
    top = hand() + np.array([0.0, 0.0, lift])
    go(session, arm, top, speed=speed)
    drift = 0.0
    for _ in range(round(hold * hz)):
        tick(session, 1)
        drift = max(drift, float(np.linalg.norm(centroid() - hand() - offset)))
    gripper = session.env.gripper(arm)
    return Grasp(
        rise=float(centroid()[2] - before[2]),
        drift=drift,
        width=float(gripper.width(session.env.state)) if gripper is not None else float("nan"),
        reached=float(np.linalg.norm(hand() - top)),
        finite=bool(session.env.state_is_finite()),
    )
