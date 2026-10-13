# SPDX-License-Identifier: Apache-2.0
"""A routine on the page: a script that drives the arms the way a person does, from a Run button.

A demo that is only a scene asks the reader to do the whole job by hand before
they have seen it done. A routine is the job written down: a function of the
built scene and the page that returns a generator, yielding once per control
step. :func:`robosimstudio.view` given ``routines=`` puts a **Run** button per
routine on the page, and the page takes each step as it always does.

**It drives through the arm panels**, not around them. :func:`reach` glides an
arm's gizmo to a pose at a hand's pace and waits for the hand to get there;
:func:`grip` presses the Grip buttons. So the gizmo moves on screen exactly as
it does under a mouse, the arm tracks it through the same IK and setpoint law,
and what a hand ends up holding is what its fingers hold: nothing here welds,
pins or teleports anything. A person may grab the gizmo mid-routine, and the
routine will fight them for it; **Stop** ends it where it is.

A routine that does not take the page (``fn(sim)``) drives the scene itself,
the way a policy does: the arm panels are suspended while it plays, and given
the hands back, as found, when it ends.

    def routine(sim, session):
        yield from reach(session, "right", above_cube)
        yield from reach(session, "right", at_cube)
        yield from grip(session, "right", 1.0)
        yield from reach(session, "right", above_cube)

    rss.view(sim, routines={"Pick up the cube": routine})
"""

from __future__ import annotations

import inspect
import math
import traceback
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from .panels import Panel

__all__ = ["PanelRoutine", "RoutineError", "grip", "hand", "reach", "wait"]


class RoutineError(RuntimeError):
    """A routine could not do what it said: a hand that never arrived, a grip that slipped. It stops the routine."""


def _hz(session) -> float:
    return max(1.0, float(session.env.request.physics.control_hz))


def hand(session, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """The arm's tool centre now: world position [m] and quaternion (xyzw)."""
    position, quat = session.arms[arm].ik.ee_pose(session.env.state)
    return np.asarray(position, dtype=np.float64), np.asarray(quat, dtype=np.float64)


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(a, b))
    if d < 0.0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + t * (b - a)
        return q / np.linalg.norm(q)
    angle = math.acos(min(1.0, d))
    return (math.sin((1.0 - t) * angle) * a + math.sin(t * angle) * b) / math.sin(angle)


def reach(
    session,
    arm: str,
    position,
    quat=None,
    *,
    speed: float = 0.15,
    turn: float = 1.0,
    tolerance: float = 0.006,
    settle: float = 4.0,
) -> Iterator[None]:
    """Glide the arm's gizmo to a pose, then wait for the hand to get there.

    The gizmo moves at ``speed`` [m/s] and turns at ``turn`` [rad/s], eased at
    both ends, the way a hand on a mouse moves it -- slower than the setpoint
    law's 0.31 m/s, so the arm keeps up and a held thing is not jerked. Then
    the drag ends and the hand has ``settle`` seconds to come within
    ``tolerance`` [m] -- 6 mm, because a YAM hand at rest sits 3 to 4 mm off
    its target in free air, measured -- and a hand that does not raises :class:`RoutineError`,
    because a routine carrying on from somewhere it is not is a routine
    reporting a job it did not do.

    Args:
        position: World position of the tool centre [m].
        quat: World orientation (xyzw); the cursor's own when not given.
    """
    panel = session.arms[arm]
    dt = 1.0 / _hz(session)
    start = np.array(panel.cursor_pos, dtype=np.float64)
    q0 = np.array(panel.cursor_quat, dtype=np.float64)
    q0 /= np.linalg.norm(q0)
    target = np.asarray(position, dtype=np.float64)
    q1 = q0 if quat is None else np.asarray(quat, dtype=np.float64) / np.linalg.norm(quat)
    angle = 2.0 * math.acos(min(1.0, abs(float(np.dot(q0, q1)))))
    seconds = max(float(np.linalg.norm(target - start)) / speed, angle / turn)
    steps = max(1, math.ceil(seconds / dt))
    for k in range(1, steps + 1):
        s = k / steps
        f = s * s * (3.0 - 2.0 * s)  # eased in and out
        panel.aim(start + (target - start) * f, _slerp(q0, q1, f), moving=True)
        yield
    panel.aim(target, q1, moving=False)
    miss = float("inf")
    for _ in range(max(1, math.ceil(settle / dt))):
        miss = float(np.linalg.norm(hand(session, arm)[0] - target))
        if miss < tolerance:
            return
        yield
    raise RoutineError(f"the {arm} hand stopped {miss * 1000:.0f} mm short of where it was sent")


def grip(session, arm: str, value: float, seconds: float = 0.6) -> Iterator[None]:
    """Press the arm's Grip control: 1 shut, 0 open, between for a partial grip. Then let the fingers move."""
    session.arms[arm].grip_request = float(value)
    yield from wait(session, seconds)


def wait(session, seconds: float) -> Iterator[None]:
    """Let the page step for ``seconds`` of simulated time."""
    for _ in range(max(1, math.ceil(seconds * _hz(session)))):
        yield


class PanelRoutine(Panel):
    """The Run buttons: one per routine, a Stop, and a line saying what is playing.

    Obeys the panel contract: the buttons only assign, and :meth:`drain` -- on
    the main thread, once per step -- starts, advances and stops. Drained after
    the arm panels, so a cursor a routine moves is driven on the next step,
    exactly as a gizmo dragged between two steps is. A paused page does not
    drain it, so pausing pauses the routine; a reset ends it, because the scene
    it was half way through is gone.
    """

    def __init__(self, session, routines: dict[str, Callable]) -> None:
        super().__init__(session)
        self.routines = dict(routines)
        self.requested: str | None = None
        self.stop_requested = False
        self.playing: str | None = None
        self.gen: Iterator[None] | None = None
        self.steps = 0
        self.suspended: list[Any] = []
        self._status = None

    def build(self, gui) -> None:
        with gui.add_folder("Routine", expand_by_default=True):
            for name in self.routines:
                button = gui.add_button(name)
                button.on_click(lambda _, name=name: setattr(self, "requested", name))
            stop = gui.add_button("Stop")
            stop.on_click(lambda _: setattr(self, "stop_requested", True))
            self._status = gui.add_text("Playing", initial_value="nothing", disabled=True)

    def drain(self) -> None:
        if self.stop_requested:
            self.stop_requested = False
            self._end("stopped")
        if self.requested is not None:
            name, self.requested = self.requested, None
            self._start(name)
        if self.gen is None:
            return
        try:
            next(self.gen)
        except StopIteration:
            self._end("done")
            return
        except Exception as exc:
            traceback.print_exc()
            self._end(f"stopped: {exc}")
            return
        self.steps += 1
        if self.steps % 10 == 1:
            self._say(f"{self.playing}, {self.steps / _hz(self.session):.1f} s")

    def on_reset(self, obs, info) -> None:
        self._end("reset")

    def _start(self, name: str) -> None:
        self._end("")
        fn = self.routines[name]
        drives_the_page = "session" in inspect.signature(fn).parameters
        if not drives_the_page:
            self.suspended = [panel for panel in self.session.arms.values() if not panel.suspended]
            for panel in self.suspended:
                panel.suspend()
        try:
            gen = fn(self.session.env, session=self.session) if drives_the_page else fn(self.session.env)
        except Exception as exc:
            traceback.print_exc()
            self._end(f"did not start: {exc}")
            return
        if not inspect.isgenerator(gen):
            self._end(f"did not start: {name} returned {type(gen).__name__}, not a generator")
            return
        self.gen, self.playing, self.steps = gen, name, 0
        self._say(f"{name}, 0.0 s")

    def _end(self, how: str) -> None:
        if self.gen is not None:
            self.gen.close()
            print(f"[studio] routine {self.playing}: {how}", flush=True)
            self._say(f"{self.playing}: {how}" if how else "nothing")
        elif how and how != "reset":
            self._say(how)
        self.gen, self.playing = None, None
        for panel in self.suspended:
            panel.resume()
        self.suspended = []

    def _say(self, text: str) -> None:
        if self._status is not None:
            self._status.value = text
