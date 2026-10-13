# SPDX-License-Identifier: Apache-2.0
"""What happens in a take, beat by beat, and what must be true at the end of it.

Three vocabularies, each a frozen dataclass that says what it is and carries the
few lines that do it:

* **targets** are points in the scene, resolved when the beat runs rather than
  when the take is written -- ``At("cube")`` is wherever the cube is now;
* **beats** are what the person on the page does: wait, glide, click, drag an
  arrow, press a button, move a slider;
* **expectations** are what the physics must have done, read off the scene at
  named marks. A take whose expectation fails is not written into
  ``examples/``: a slipped grasp must never ship.

A beat plays against the film (:class:`recording.lockstep.Film`): its page, its
clock and its scene. Its seconds are the clip's, which at a take's ``speed`` of 1
are the scene's too. Nothing here imports the library or a browser, so the take
table can be read by a test with neither.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def _axis(axis) -> tuple[float, float, float]:
    return AXES[axis] if isinstance(axis, str) else tuple(float(v) for v in axis)


def _add(a, b) -> tuple[float, float, float]:
    return tuple(float(x) + float(y) for x, y in zip(a, b, strict=True))


# =============================================================================
# Targets: points in the scene, resolved when they are needed
# =============================================================================


class Target:
    """A world point [m] the film resolves when a beat needs it."""

    def resolve(self, film) -> tuple[float, float, float]:
        raise NotImplementedError


@dataclass(frozen=True)
class At(Target):
    """A thing's position (a body's origin, a deformable's centroid), plus an offset [m]."""

    name: str
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0

    def resolve(self, film):
        return _add(film.position(self.name), (self.dx, self.dy, self.dz))


@dataclass(frozen=True)
class Top(Target):
    """The top of a thing, over its middle, plus ``dz`` [m]."""

    name: str
    dz: float = 0.0

    def resolve(self, film):
        x, y, _ = film.position(self.name)
        return (x, y, film.top(self.name) + self.dz)


@dataclass(frozen=True)
class Corner(Target):
    """The particle of a sheet nearest one corner of its footprint, ``inset`` [m] in.

    ``sx`` and ``sy`` are ``-1`` or ``+1``: which end of x and of y.
    """

    name: str
    sx: int = -1
    sy: int = -1
    inset: float = 0.01

    def resolve(self, film):
        return film.corner(self.name, self.sx, self.sy, self.inset)


@dataclass(frozen=True)
class Segment(Target):
    """The particle a fraction ``frac`` of the way along a rod, in its own order."""

    name: str
    frac: float = 0.5

    def resolve(self, film):
        return film.along(self.name, self.frac)


@dataclass(frozen=True)
class Bench(Target):
    """A point in the cell's work-surface frame: ``(0, 0, 0.2)`` is 200 mm over the middle of the bench."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def resolve(self, film):
        return film.bench((self.x, self.y, self.z))


@dataclass(frozen=True)
class Hand(Target):
    """An arm's tool centre."""

    arm: str = "right"

    def resolve(self, film):
        return film.hand(self.arm)


@dataclass(frozen=True)
class Highest(Target):
    """The highest point of a thing: a particle on a deformable's top, or the top link of a heap."""

    name: str = ""

    def resolve(self, film):
        return film.highest(self.name)


@dataclass(frozen=True)
class Patch(Target):
    """The middle of the points a thing had within ``radius`` [m] of ``near`` when first asked, followed after.

    What a hand closes on: the strip of a towel's edge under the jaws, the capsule
    of a cable between them, the link a handle took. Its centroid rises with the
    hand where the whole thing's barely moves -- a 350 mm towel lifted by one
    edge, a cable tied to nothing lifted by its middle.
    """

    name: str = ""
    near: Any = None
    radius: float = 0.02
    skin: float = 0.0
    """Above 0: only the points within this of the thing's top, and ``radius`` across the bench."""
    lowest: bool = False
    """The patch's lowest point rather than its middle: the bottom of a dent."""

    def resolve(self, film):
        return film.patch(self.name, self.near, self.radius, skin=self.skin, lowest=self.lowest)


@dataclass(frozen=True)
class Nearest(Target):
    """The tool centre of the hand nearest a thing."""

    name: str = ""

    def resolve(self, film):
        return film.hand(film.nearest_arm(self.name))


@dataclass(frozen=True)
class Mid(Target):
    """Halfway between two targets: a camera aimed here keeps a hand and what it is going for in one view."""

    a: Any = None
    b: Any = None

    def resolve(self, film):
        return tuple((p + q) / 2.0 for p, q in zip(resolve(self.a, film), resolve(self.b, film), strict=True))


def resolve(target, film) -> tuple[float, float, float]:
    """A :class:`Target`, a callable ``(film) -> xyz``, or a plain xyz."""
    if isinstance(target, Target):
        return target.resolve(film)
    if callable(target):
        return tuple(float(v) for v in target(film))
    return tuple(float(v) for v in target)


# =============================================================================
# Framing: where the camera stands, and which folds are open
# =============================================================================


@dataclass(frozen=True)
class Camera:
    """An explicit view: the eye and the point looked at [m, world, or targets]."""

    eye: Any
    look_at: Any
    fov: float = 50.0
    """Vertical field of view [deg]. viser opens at 75, which at half a metre makes a 50 mm cube a speck."""

    def resolve(self, film) -> tuple[tuple, tuple, float]:
        return resolve(self.eye, film), resolve(self.look_at, film), self.fov


@dataclass(frozen=True)
class Orbit:
    """A view of ``about`` from ``distance`` away, at an azimuth and an elevation [deg].

    Azimuth 0 looks along -x (the eye on the +x side), 90 along -y.
    """

    about: Any
    distance: float = 0.5
    azimuth: float = 0.0
    elevation: float = 30.0
    fov: float = 50.0

    def resolve(self, film) -> tuple[tuple, tuple, float]:
        at = resolve(self.about, film)
        az, el = math.radians(self.azimuth), math.radians(self.elevation)
        offset = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
        return _add(at, tuple(self.distance * v for v in offset)), at, self.fov


@dataclass(frozen=True)
class Layout:
    """The sidebar before the first frame: folds closed and folds opened, unfilmed.

    ``Simulation`` is closed by default: its Rate readout is wall clock, and a
    clip in lockstep would show the recorder's speed rather than the scene's.
    """

    collapse: tuple[str, ...] = ("Simulation",)
    expand: tuple[str, ...] = ()


# =============================================================================
# Beats
# =============================================================================


@dataclass(frozen=True, kw_only=True)
class Beat:
    """One thing the person does. ``label`` captions the contact sheet; ``mark`` names the moment it ends."""

    label: str = ""
    mark: str = ""

    def play(self, film) -> dict | None:
        raise NotImplementedError

    def caption(self) -> str:
        return self.label or type(self).__name__.lower()


@dataclass(frozen=True)
class Wait(Beat):
    """Let the scene run for ``seconds`` of clip, or until ``until(film)`` holds."""

    seconds: float = 1.0
    until: Callable[[Any], bool] | None = None

    def play(self, film):
        film.hold(self.seconds, until=self.until)


@dataclass(frozen=True)
class Glide(Beat):
    """Move the pointer to a world point's pixel, pressing nothing."""

    to: Any = None
    seconds: float = 0.6

    def play(self, film):
        x, y = film.pixel(resolve(self.to, film))
        film.page.glide(x, y, self.seconds)


@dataclass(frozen=True)
class Click(Beat):
    """Click a world point's pixel: on a body or a patch of cloth, it comes up on a handle."""

    target: Any = None
    approach: float = 0.7

    def play(self, film):
        x, y = film.pixel(resolve(self.target, film))
        before = film.held()
        film.page.click(x, y, approach=self.approach)
        film.hold(1.0, until=lambda f: f.held() != before)
        return {"clicked_px": [round(x, 1), round(y, 1)], "holding": film.held()}


@dataclass(frozen=True)
class Drag(Beat):
    """Drag a gizmo's arrow along ``axis`` by ``metres``, or as far as ``to`` along it.

    ``gizmo`` is an arm (``"right"``), ``"handle"`` for the last handle a click
    put down, or a scene path. An arm's drag first waits for the panel to say
    ``on target``, so each drag starts from where the hand is.
    """

    gizmo: str = "right"
    axis: Any = "z"
    metres: float = 0.0
    to: Any = None
    seconds: float = 1.0
    approach: float = 0.5

    def play(self, film):
        name = film.gizmo(self.gizmo)
        film.settle(self.gizmo)
        unit = _axis(self.axis)
        metres = self.metres
        if self.to is not None:
            here, there = film.where(name), resolve(self.to, film)
            metres = sum((b - a) * u for a, b, u in zip(here, there, unit, strict=True))
        return film.drag(name, unit, metres, seconds=self.seconds, approach=self.approach)


@dataclass(frozen=True)
class DragTo(Beat):
    """Take a gizmo to a point in axis drags: up first going up, down last going down.

    An axis already within ``tolerance`` [m] is not dragged; a second pass takes up
    what a leaning arrow put on the others, past 5 mm.
    """

    gizmo: str = "right"
    to: Any = None
    seconds_per_m: float = 6.0
    tolerance: float = 0.003

    def play(self, film):
        name = film.gizmo(self.gizmo)
        film.settle(self.gizmo)
        there = resolve(self.to, film)
        delta = [b - a for a, b in zip(film.where(name), there, strict=True)]
        order = ("z", "x", "y") if delta[2] > 0 else ("x", "y", "z")
        done = []
        # A second pass for what the first left: an arm's arrows are its tool's axes,
        # and a Panda's at home lean 7 degrees off vertical, so 206 mm down its z
        # arrow is also 26 mm along x.
        for tolerance in (self.tolerance, max(self.tolerance, 0.005)):
            for axis in order:
                film.settle(self.gizmo)
                i = "xyz".index(axis)
                d = there[i] - film.where(name)[i]
                if abs(d) < tolerance:
                    continue
                seconds = max(0.5, abs(d) * self.seconds_per_m)
                done.append(film.drag(name, AXES[axis], d, seconds=seconds, approach=0.45))
        return {"drags": done}


@dataclass(frozen=True)
class Grasp(Beat):
    """The gesture the examples promise: over it, straight down onto it, **Close**, drag up.

    ``arm`` empty takes the hand nearest the thing. ``at`` is where the tool
    centre closes (the thing's position by default); a hand not already over
    it is first taken ``clearance`` [m] above that point. Leaves the marks
    ``closed`` and ``lifted``.
    """

    thing: str = ""
    arm: str = ""
    at: Any = None
    clearance: float = 0.10
    lift: float = 0.12
    seconds_per_m: float = 9.0

    def play(self, film):
        arm = self.arm or film.nearest_arm(self.thing)
        name = film.gizmo(arm)
        film.settle(arm)
        there = resolve(self.at if self.at is not None else At(self.thing), film)
        here = film.where(name)
        done = []
        if math.dist(here[:2], there[:2]) > 0.01:
            over = (there[0], there[1], max(here[2], there[2] + self.clearance))
            done.append(DragTo(arm, over, seconds_per_m=self.seconds_per_m).play(film))
        done.append(Drag(arm, "z", to=there, seconds=1.2).play(film))
        Press(text="Close", within=f"Arm: {arm}").play(film)
        film.hold(0.8)
        film.mark("closed")
        done.append(Drag(arm, "z", self.lift, seconds=1.2).play(film))
        film.hold(0.6)
        film.mark("lifted")
        return {"arm": arm, "drags": done}


@dataclass(frozen=True)
class Press(Beat):
    """Click the button reading ``text``, inside the folder whose title starts with ``within``."""

    text: str = ""
    within: str = ""

    def play(self, film):
        film.page.press_button(self.text, self.within)


@dataclass(frozen=True)
class Toggle(Beat):
    """Click the checkbox labelled ``text``."""

    text: str = ""
    within: str = ""

    def play(self, film):
        film.page.toggle(self.text, self.within)


@dataclass(frozen=True)
class Select(Beat):
    """Pick ``option`` in the dropdown labelled ``text``."""

    text: str = ""
    option: str = ""
    within: str = ""

    def play(self, film):
        return {"shown": film.page.select(self.text, self.option, self.within)}


@dataclass(frozen=True)
class Fold(Beat):
    """Open (or close) the folder titled ``title``, on camera."""

    title: str = ""
    open: bool = True

    def play(self, film):
        film.page.fold(self.title, self.open)


@dataclass(frozen=True)
class Slider(Beat):
    """Drag the slider labelled ``text`` to ``value``, in the slider's own units, over ``seconds``.

    ``value=None`` drags it back to where it was before this take first moved it.
    """

    text: str = ""
    value: float | None = None
    seconds: float = 1.0
    within: str = ""

    def play(self, film):
        now = film.page.js("slider", self.text, self.within)
        if now is None:
            raise RuntimeError(f"no slider labelled {self.text!r}")
        start = film.sliders.setdefault((self.text, self.within), float(now["value"]))
        value = start if self.value is None else self.value
        return {"from": float(now["value"]), "value": film.page.slider(self.text, value, self.seconds, self.within)}


@dataclass(frozen=True)
class Scroll(Beat):
    """Wheel the sidebar by ``pixels`` (positive is down) over ``seconds``."""

    pixels: float = 200.0
    seconds: float = 0.5

    def play(self, film):
        film.page.scroll(self.pixels, self.seconds)


@dataclass(frozen=True)
class Run(Beat):
    """Press a routine's **Run** button and film it to its end, at most ``timeout`` seconds of scene.

    ``routine`` is the button's text; the page's first routine when empty. The
    end is the panel's own: its generator gone, its ``Playing`` line reading
    ``<name>: done`` -- or ``stopped: ...``, which the timeline keeps and an
    expectation can refuse.
    """

    routine: str = ""
    timeout: float = 240.0

    def play(self, film):
        panel = film.routine()
        name = self.routine or next(iter(panel.routines))
        film.page.press_button(name, "Routine")
        film.page.rest()
        seen = []

        def ended(f) -> bool:
            if panel.gen is not None:
                seen.append(True)
            return bool(seen) and panel.gen is None and panel.requested is None

        frames = round(self.timeout * film.take.fps / film.take.speed)
        film.hold(frames / film.take.fps, until=ended)
        film.ended = film.page.text("Playing", "Routine")
        return {"routine": name, "ended": film.ended, "started": bool(seen)}


@dataclass(frozen=True)
class Policy(Beat):
    """Press **Run** in the task page's Policy folder and film the run to its end, at most ``timeout`` s of scene.

    The end is the runner's own: ``running`` true once and false again, which
    includes the settle steps the page gives a finished graph before it scores.
    """

    timeout: float = 90.0

    def play(self, film):
        driver = film.session.driver
        if driver is None:
            raise RuntimeError("this page has no Policy folder: it is not showing a task")
        film.page.press_button("Run", "Policy")
        film.page.rest()
        seen = []

        def ended(f) -> bool:
            if driver.running:
                seen.append(True)
            return bool(seen) and not driver.running and not driver.start_requested

        film.hold(self.timeout / film.take.speed, until=ended)
        return {"started": bool(seen), "policy": film.page.text("Policy", "Policy"), "steps": driver.steps}


@dataclass(frozen=True)
class Do(Beat):
    """``fn(film)`` off camera, between two frames: for what no gesture on the page does."""

    fn: Callable[[Any], Any] | None = None

    def play(self, film):
        out = self.fn(film) if self.fn is not None else None
        return out if isinstance(out, dict) else None


# =============================================================================
# Expectations
# =============================================================================


@dataclass(frozen=True)
class Expectation:
    """What the physics must have done. ``judge`` returns ``(passed, reading)``."""

    def things(self) -> tuple[str, ...]:
        return ()

    def judge(self, film) -> tuple[bool, str]:
        raise NotImplementedError


@dataclass(frozen=True)
class Rose(Expectation):
    """``name`` was at least ``by`` [m] higher at mark ``at`` than at mark ``since``.

    ``name`` is a thing (a body's origin, a deformable's centroid) or a
    :class:`Target` -- ``Segment("cable", 0.5)`` for the middle of a span
    whose ends are tied down, which is where a lift shows.
    """

    name: str | Target
    by: float
    at: str = "end"
    since: str = "start"

    def things(self):
        return (self.name,)

    def judge(self, film):
        key = str(self.name)
        rise = film.marks[self.at]["things"][key][2] - film.marks[self.since]["things"][key][2]
        return (
            rise >= self.by,
            f"{key} rose {rise * 1000:.1f} mm from {self.since} to {self.at} (need {self.by * 1000:.0f})",
        )


@dataclass(frozen=True)
class Sank(Expectation):
    """``name`` was at least ``by`` [m] lower at mark ``at`` than at ``since``, and back within ``back`` [m] at the end.

    A press: ``Patch("sponge", Top("sponge"))`` is the foam under the fingertips,
    which dents where the sponge's overall height grows as it bulges beside them.
    """

    name: str | Target
    by: float
    at: str = "pressed"
    since: str = "start"
    back: float = 0.001

    def things(self):
        return (self.name,)

    def judge(self, film):
        key = str(self.name)
        z = {m: film.marks[m]["things"][key][2] for m in (self.since, self.at, "end")}
        dent, left = z[self.since] - z[self.at], abs(z["end"] - z[self.since])
        ok = dent >= self.by and left <= self.back
        return (
            ok,
            f"{key} sank {dent * 1000:.1f} mm at {self.at} (need {self.by * 1000:.1f}) and ended {left * 1000:.1f} mm off (need {self.back * 1000:.1f})",
        )


@dataclass(frozen=True)
class Held(Expectation):
    """``name`` kept its place in ``arm``'s hand from mark ``since`` to mark ``at``, within ``tolerance`` [m]."""

    name: str
    arm: str = "right"
    since: str = "closed"
    at: str = "end"
    tolerance: float = 0.005

    def things(self):
        return (self.name,)

    def judge(self, film):
        def offset(mark):
            m = film.marks[mark]
            return [a - b for a, b in zip(m["things"][self.name], m["hands"][self.arm], strict=True)]

        slip = math.dist(offset(self.at), offset(self.since))
        return (
            slip <= self.tolerance,
            f"{self.name} slipped {slip * 1000:.1f} mm in the {self.arm} hand from {self.since} to {self.at}",
        )


@dataclass(frozen=True)
class Squashed(Expectation):
    """``name``'s height was at least ``by`` [m] less at mark ``at`` than at ``since``."""

    name: str
    by: float
    at: str = "end"
    since: str = "start"

    def things(self):
        return (self.name,)

    def judge(self, film):
        h0 = film.marks[self.since]["heights"][self.name]
        h1 = film.marks[self.at]["heights"][self.name]
        return (
            h0 - h1 >= self.by,
            f"{self.name} height {h0 * 1000:.1f} -> {h1 * 1000:.1f} mm (need -{self.by * 1000:.0f})",
        )


@dataclass(frozen=True)
class Check(Expectation):
    """``fn(film)`` returns ``(passed, reading)`` or a bool; ``message`` says what it means."""

    fn: Callable[[Any], Any]
    message: str = ""
    needs: tuple[str, ...] = field(default=())

    def things(self):
        return self.needs

    def judge(self, film):
        out = self.fn(film)
        passed, reading = out if isinstance(out, tuple) else (bool(out), "")
        return bool(passed), " ".join(p for p in (self.message, str(reading)) if p)
