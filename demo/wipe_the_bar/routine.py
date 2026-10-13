# SPDX-License-Identifier: Apache-2.0
"""The job, from the robot's home pose to its home pose again: the take ``wipe_the_bar.py`` runs from its Run button.

The left hand steadies the bar at its far elbow. The right hand takes the
sponge off the mat, leans 35 degrees along the bar, presses the foam onto the
crown until the stripe reads a press, strokes it along the bar until the marker
is gone, puts the sponge back where it lay, and both hands go home. It is the
``surface_prep`` task's scripted take, written against the page: every move is a
drag of an arm's gizmo (:func:`~robosimstudio_studio.routine.reach`, or
:meth:`~robosimstudio_studio.panels.PanelArm.aim` for the millimetre steps of a
press) and every grip a press of its Grip control.

The grasp is physics; the hold after it is a latch. **Close** squeezes the foam
by contact alone, and once the jaws have settled on it -- both pads pressing,
the foam squeezed -- ``bar.Latch`` pins the foam under the pads where it is, until
**Open**. A grasp that does not meet that test is not rescued: the take stops
and says what it read.

Every pose is built from what the scene measures -- the bar's welded pose, the
sponge's own centroid, the hand's tool centre -- and every constant below
carries the reading it came from.
"""

from __future__ import annotations

import math

import bar
import numpy as np

from robosimstudio.core.math import quat_multiply, quat_to_mat
from robosimstudio_studio.routine import RoutineError, grip, hand, reach, wait

TILT_DEG = 35.0
"""How far the working hand leans along the bar for the wipe [deg]. Upright, the
wrist runs out of orientation at the far end of the stripe."""
PAD_OFFSET = (0.001, 0.0, 0.007)
"""Where the YAM's pads touch a held thing, in the tool frame [m]: measured off
MuJoCo's contact table on a working grasp. They lead the tool centre by 7 mm."""
CLEARANCE = 0.040
"""The sponge's underside over the crown while it crosses to and from the bar [m]."""
PRESS_N = 4.0
"""What a press steers to, read off the stripe [N]: the reference take's press. At 1.5 N
four strokes cleaned 0.89 of the stripe, at 3 N 0.94, at 4 N 0.96."""
STROKE_SPEED = 0.06
"""Along the bar during a stroke [m/s]: removal is pressure times sliding speed."""
STROKE_CYCLES = 4
"""Strokes before the take stops."""
CLEANED = 0.96
"""The share of the stripe cleaned at which the take stops stroking."""
OVERRUN_NEAR, OVERRUN_FAR = 0.025, 0.045
"""How far a stroke runs past the stripe's ends [m]: the leaning hand's patch trails its centre."""
HAND_CLEAR = 0.025
"""A stroke never comes within this of the steadying hand's pads [m]: a jaw's half-width
and a margin. At 35 mm the strokes stopped short and the stripe's last 20 mm kept 0.88-1.0
of its ink."""
FOLLOW_GAIN, FOLLOW_RATE, FOLLOW_SPAN = 4.0e-5, 0.004, (-0.015, 0.008)
"""How a stroke holds its press: the hand goes down (or up) 0.04 mm a step per newton
short (or over), at most 4 mm/s, and at most this far from the seat [m]."""
SUPPORT_STANDOFF, SUPPORT_ABOVE_AXIS = 0.050, 0.004
"""The steadying hold: from 50 mm over the far elbow, down until the pads straddle the tube 4 mm over its axis."""
SUPPORT_GAP = 2 * bar.LAYOUT.tube_radius + 0.004
"""[m] the left jaws close to the tube's diameter and 4 mm: a steadying hand, not a clamp on a welded bar."""


def wipe_quat(tilt_deg: float = 0.0) -> np.ndarray:
    """Top-down, jaws closing across the bench's x, leaning ``tilt_deg`` about x so the hand tips toward +y."""
    down = quat_multiply((0.0, 0.0, -math.sin(math.pi / 4), math.cos(math.pi / 4)), (1.0, 0.0, 0.0, 0.0))
    t = math.radians(tilt_deg)
    return np.asarray(quat_multiply((math.sin(t / 2), 0.0, 0.0, math.cos(t / 2)), down), dtype=np.float64)


class Take:
    """One run of the job, as a generator of control steps, and what it read on the way.

    ``phase`` names what the take is doing, and ``log`` keeps a line per beat, so
    a test (or a person watching the terminal) can hold each part to a number.
    """

    def __init__(self, sim, session) -> None:
        self.sim, self.session = sim, session
        self.stripe = sim["stripe"].entity
        self.sponge = sim["sponge"]
        self.latch = bar.latch(sim)
        self.phase = "idle"
        self.log: list[str] = []
        self.z = self.seat = 0.0
        """The hand's height on the bar [m, world], and where the press first seated it."""

    def _say(self, text: str) -> None:
        self.log.append(text)
        print(f"[wipe] {text}", flush=True)

    def points(self) -> np.ndarray:
        return np.asarray(self.sponge.particle_q, dtype=np.float64)

    def _aim(self, arm: str, position, quat, steps: int):
        """Hold the gizmo at a pose for ``steps`` control steps: a press, a millimetre at a time."""
        panel = self.session.arms[arm]
        panel.aim(np.asarray(position, dtype=np.float64), np.asarray(quat, dtype=np.float64), moving=True)
        for _ in range(int(steps)):
            yield

    def _press(self, xy, z_start: float, quat, *, max_travel: float, label: str):
        """Lower the hand a millimetre at a time until the stripe reads :data:`PRESS_N`.

        Returns the seat height, or ``None`` when the travel ran out with nothing
        pressing: the sponge is no longer between the jaws.
        """
        z, spent, force = float(z_start), 0.0, 0.0
        while spent < max_travel:
            z -= 0.001
            spent += 0.001
            yield from self._aim("right", (xy[0], xy[1], z), quat, steps=6)
            force = self.stripe.held_press
            if force >= PRESS_N:
                self._say(f"{label}: seated {spent * 1e3:.0f} mm down at {force:.2f} N")
                return z
        self._say(f"{label}: {spent * 1e3:.0f} mm down and nothing pressing; the sponge is out of the jaws")
        yield from self._aim("right", (xy[0], xy[1], z_start), quat, steps=30)
        return None

    def _stroke(self, a, b, quat, label: str):
        """Slide from ``a`` to ``b`` along the bar, raising or lowering the hand a little each step to hold the press.

        The arm sags differently along its reach, so one seat height pressed
        0.7 N at one end and 11 N at the other; a person feels the press and
        follows it, and so does this.
        """
        panel = self.session.arms["right"]
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        hz = float(self.sim.request.physics.control_hz)
        steps = max(1, math.ceil(float(np.linalg.norm(b[:2] - a[:2])) / STROKE_SPEED * hz))
        rate = FOLLOW_RATE / hz
        for k in range(1, steps + 1):
            p = a + (b - a) * k / steps
            short = PRESS_N - self.stripe.held_press
            low, high = self.seat + FOLLOW_SPAN[0], self.seat + FOLLOW_SPAN[1]
            self.z = float(np.clip(self.z - np.clip(FOLLOW_GAIN * short, -rate, rate), low, high))
            panel.aim((p[0], p[1], self.z), quat, moving=True)
            yield
        self._say(f"{label}: cleaned {self.stripe.cleaned:.2f}, press {self.stripe.held_press:.2f} N")

    def steps(self):
        sim, session, L = self.sim, self.session, bar.LAYOUT
        pose = np.asarray(sim["bar"].pose, dtype=np.float64)
        axis, along = pose[:3], quat_to_mat(pose[3:7]) @ np.array([0.0, 0.0, 1.0])  # +y: the bar's own axis
        crown = float(axis[2]) + L.tube_radius
        wq, uq = wipe_quat(TILT_DEG), wipe_quat(0.0)  # leaning for the wipe; upright for the pick and the straddle
        homes = {arm: session.arms[arm].ik.home_pose for arm in ("right", "left")}

        # --- the left hand steadies the bar at its far elbow -------------------
        self.phase = "steady"
        elbow = axis + along * L.support_u
        yield from reach(session, "left", elbow + np.array([0.0, 0.0, SUPPORT_STANDOFF]), uq, speed=0.20)
        yield from reach(session, "left", elbow + np.array([0.0, 0.0, SUPPORT_ABOVE_AXIS]), uq, speed=0.05)
        yield from grip(session, "left", sim.gripper("left").command_for_width(SUPPORT_GAP), seconds=0.7)
        self._say("left hand on the far elbow")

        # --- the right hand takes the sponge off the mat -----------------------
        self.phase = "pick"
        lying = self.points()
        centroid = lying.mean(axis=0)
        pick = centroid - quat_to_mat(uq) @ np.asarray(PAD_OFFSET)  # the pads onto the block, not the tool centre
        yield from reach(session, "right", pick + np.array([0.0, 0.0, 0.10]), uq, speed=0.20)
        yield from reach(session, "right", pick, uq, speed=0.05)
        yield from grip(session, "right", 1.0, seconds=0.5)  # Close: the jaws squeeze to their 8 N cap
        for _ in range(90):  # the latch engages once the grasp is settled, pressed and squeezing
            if self.latch.latched:
                break
            yield
        r = self.latch.readings
        if not self.latch.latched:
            raise RoutineError(
                f"the jaws did not grip the sponge: opening {r.get('opening', 0) * 1e3:.1f} mm, "
                f"pads {[round(f, 2) for f in r.get('press', [])]} N"
            )
        self._say(
            f"sponge gripped: jaws {r['opening'] * 1e3:.1f} mm apart on {r['free'] * 1e3:.0f} mm of foam, "
            f"pads {r['press'][0]:.1f} N and {r['press'][1]:.1f} N; latched {len(self.latch.nodes)} nodes"
        )

        # --- to the near end of the stripe, leaning -----------------------------
        self.phase = "carry"
        tcp, _ = hand(session, "right")
        drop = float(tcp[2]) - float(self.points()[:, 2].min())  # the sponge's underside below the tool centre
        clear_z = crown + drop + CLEARANCE
        half = 0.5 * L.sponge_size[1]
        u0 = max(-L.stripe_half + half - OVERRUN_NEAR, -L.straight_half + half + 0.005)
        u1 = min(L.stripe_half - half + OVERRUN_FAR, L.support_u - half - HAND_CLEAR)
        # Up off the mat, upright, then over the near block at the crossing height,
        # leaning. Over the pick itself the crossing height is out of reach (the IK
        # missed it by 10 mm upright, 21 mm leaning), and a straight line from 60 mm
        # up to the stripe swept the foam into the near elbow and out of the jaws.
        before_block = axis + along * -(L.leg_u + L.block_inset[1] + L.block_half[1] + 0.020)
        gate = np.array([0.5 * (pick[0] + axis[0]), before_block[1], clear_z])
        start = axis + along * u0
        yield from reach(session, "right", pick + np.array([0.0, 0.0, 0.10]), uq, speed=0.10)
        yield from reach(session, "right", gate, wq, speed=0.10)
        yield from reach(session, "right", [start[0], start[1], clear_z], wq, speed=0.10)
        yield from wait(session, 0.2)

        # --- press, then stroke the stripe off ----------------------------------
        self.phase = "wipe"
        far = axis + along * u1
        for cycle in range(STROKE_CYCLES):
            # One way only, the low edge trailing. Stroked back with the hand still
            # leaning, the low edge leads and digs into the crown: the press went from
            # 0.6 N to 7.6 N in two thirds of a second, measured before the latch.
            seat = yield from self._press(start, clear_z, wq, max_travel=CLEARANCE + 0.015, label=f"press {cycle}")
            if seat is None:
                break
            self.z = self.seat = seat
            yield from self._stroke(start, far, wq, f"stroke {cycle}")
            tcp, _ = hand(session, "right")
            yield from reach(session, "right", [tcp[0], tcp[1], clear_z], wq, speed=0.06, tolerance=0.02)
            if self.stripe.cleaned >= CLEANED:
                break
            yield from reach(session, "right", [start[0], start[1], clear_z], wq, speed=0.15)

        # --- the sponge back where it lay, and both hands home ------------------
        self.phase = "return"
        yield from reach(session, "right", gate, wq, speed=0.10)
        yield from reach(session, "right", pick + np.array([0.0, 0.0, 0.10]), uq, speed=0.10)
        yield from reach(session, "right", pick + np.array([0.0, 0.0, 0.002]), uq, speed=0.05)
        yield from grip(session, "right", 0.0, seconds=0.8)
        self.phase = "release"
        yield from reach(session, "right", pick + np.array([0.0, 0.0, 0.10]), uq, speed=0.10)
        yield from grip(session, "left", 0.0, seconds=0.5)
        yield from reach(session, "left", elbow + np.array([0.0, 0.0, SUPPORT_STANDOFF]), uq, speed=0.10)
        self.phase = "home"
        for arm in ("right", "left"):
            yield from reach(session, arm, *homes[arm], speed=0.20)
        moved = float(np.linalg.norm(self.points().mean(axis=0)[:2] - centroid[:2]))
        self._say(f"done: cleaned {self.stripe.cleaned:.3f}, the sponge {moved * 1e3:.0f} mm from where it lay")
        self.phase = "done"


def routine(sim, session):
    """``routines=`` on :func:`robosimstudio.view`: the whole job, one yield per control step."""
    return Take(sim, session).steps()
