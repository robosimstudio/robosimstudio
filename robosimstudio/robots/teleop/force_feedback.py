# SPDX-License-Identifier: Apache-2.0
"""Force feedback on a leader arm, against a scene this library simulates.

The leader is the *action* and the simulated arm it drags is the *state*. When
the simulated arm cannot reach where your hand is -- a jaw closed on something
solid, a link against a post, a hand pressed on the bench -- the gap between
the two is the force you should feel, and this renders it back onto the
leader's own servos. Free space stays free: below the deadband no motor is
energized at all, so the arm is as loose in your hand as it is unpowered.

Everything here is the feedback algorithm and the servo writes that render it.
The scene, the physics and the page are whatever the caller built;
:class:`Teleoperation` is handed a page and a set of leaders and owns the
hardware from the first torque-off to the last::

    from robosimstudio.robots.teleop import force_feedback as ffb
    from robosimstudio.robots.teleop.lerobot_controller import parse_leader_ports, resolve_leaders

    leaders = resolve_leaders(parse_leader_ports("right:/dev/ch340-<serial>"), "my_leader.json")
    studio = StudioSession(sim, config=StudioConfig(panels="drive"))
    with ffb.Teleoperation(studio, leaders, feedback=True) as teleop:
        studio.run()
    raise SystemExit(teleop.exit_code)

Two clocks, and that is the point
---------------------------------
A leader loop that ran at the page's step rate would render contact at whatever
rate the scene happened to be stepping -- which moves with the scene, the
render budget and the browser. So each leader arm gets **its own thread on its
own clock** (100 Hz by default), and the page keeps its own (the scene's
``control_hz``, 60 Hz for the library's default arrangement).

The split is forced anyway by a threading rule: Warp may only be touched on the
thread that steps the scene. So the halves divide exactly along that line, and
neither reaches into the other:

* **Leader thread** (:class:`LeaderArmLoop`) -- serial only, no Warp. Reads
  ``Present_Position``, converts to radians and a grip command, publishes it;
  takes the latest follower sample, computes the correction, writes
  ``Goal_Position``, ``Torque_Limit`` and ``Torque_Enable``. One thread per
  arm, because a bimanual leader is two independent buses and one arm's glitch
  must not stall the other's.
* **Main thread** (:class:`StudioBridge`) -- Warp only, no serial. Hung off the
  page's ``each_step``, which runs after its panels and *before* the step: it
  applies the leader poses as one ``joint_position`` action, then reads back
  the measured joints and the measured jaw (:func:`jaw_fraction`) and
  publishes them.

Two slots under one lock per arm, and each bus is touched by exactly one thread.

The gap is taken against the pose the arm was chasing
------------------------------------------------------
Each follower sample travels with the leader pose it was *chasing*: the action
the page applied one step before it read the arm back. The mismatch is the
difference between those two -- how far the arm got against what it was told --
rather than between the sample and wherever the hand has moved to by the time
the leader thread looks. In contact the two are the same number. In free
motion they are not: the second one contains every delay in the exchange, and
a page that steps late then reads as something in the way. See
:class:`LeaderArmLoop`'s ``compare`` for the measurement, and for the switch
back to the other form.

Why the measured jaw and not the gripper command
------------------------------------------------
A hand commanded shut on a 70 mm part stalls part way; the same command on air
reaches full travel. Reporting the *command* as the follower's state makes
those indistinguishable -- the gripper would feel weightless on exactly the
grasp you most want to feel. So the grip slot is a measurement: how far the
hand's **driven** finger joints have travelled from open
(:func:`jaw_fraction`), which is the same comparison every arm slot makes --
a driven joint against the target it was given.

The driven joints rather than the gap between the pads
(:meth:`~robosimstudio.solvers.Gripper.closed_fraction`), and that is a
measured choice. The YAM's hand drives one finger and carries the other on an
equality constraint, and at the teleoperation gains a moving wrist shakes the
carried one: with the jaw command held still and the fifth arm joint swung
0.4 rad in free space, the two fingers came 7.9 mm out of step (0.4 mm at the
asset's own gains), which the gap reads as 9% of the stroke -- over the grip
slot's deadband, so a wrist motion through empty space energized the jaw
motor. The driven finger moved 2.7% in the same pass. A jaw closed on
something stalls the driven finger too, so the grasp is still reported; what
is no longer reported is something pushing on the carried finger alone.

Staleness is a safety gate, not a corner case
----------------------------------------------
``each_step`` does not run while the page is paused or rebuilding a scene --
"paused is paused". The follower sample then stops advancing, and rendering
force against a stale mismatch would push the operator's hand toward where the
simulated arm *used to be*. So a sample older than the stale limit disarms
every motor and drops the arm back through the re-arm gate below. Hitting
pause, a stalled scene and a closed page are all the same event here.

Which is also the re-arm rule: a joint that was suspended may not energize
again until its physical and simulated positions have held *continuously*
inside the release deadband for the neutral dwell (0.5 s), and even then only
a later above-deadband sample arms it. One frame that happens to look aligned
while a scene is still settling is not alignment.

Safety
------
Torque is off at rest and off on the way out. Startup refuses to arm against
servos that came up with a nonzero torque limit; a motor is energized only by
acknowledged per-servo writes (a lost broadcast must not enable torque against
a stale goal); every disarm at a transition is verified by reading
``Torque_Enable`` back, and the state that says "this joint is live" is
committed only once every write for it has succeeded -- so a failed torque-off
is retried next tick instead of being forgotten. A fault on either thread tears
the whole session down through the same path. None of that makes an arm safe to
stand in: keep clear of it and keep a power switch in reach.

``lerobot`` and its serial driver are imported where a bus is first opened, so
this module -- the signal, the gates, the layout -- imports and tests on a
machine with neither a GPU nor an arm plugged into it.
"""

from __future__ import annotations

import json
import math
import signal
import statistics
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robosimstudio.arms import arm_slots

from .lerobot_controller import (
    ENCODER_RESOLUTION,
    ArmCalibration,
    LeaderError,
    LeaderMismatch,
    LeaderPort,
    apply_arm_actions,
    check_leaders,
    open_leader,
)

__all__ = [
    "ArmLayout",
    "ComplianceConfig",
    "JointDecision",
    "JointGate",
    "LeaderArmLoop",
    "LeaderMismatch",
    "LeaderServos",
    "Proposal",
    "Sample",
    "ServoFault",
    "StudioBridge",
    "Teleoperation",
    "Trace",
    "correction",
    "jaw_fraction",
    "preflight",
    "propose",
]

TORQUE_OFF_SETTLE_S = 0.25
"""Time given to a torque-off before its register is read back."""

COMMAND_GAP_S = 0.03
"""Pacing between per-servo commands on one bus, so a chain of them is not a burst."""

EEPROM_WRITE_SETTLE_S = 0.02
"""Time given to a servo after one of its EEPROM registers is rewritten."""

MAX_BUS_GLITCHES = 3
"""Consecutive transient bus errors ridden through before the loop faults."""

COMPARISONS = ("applied", "live")
"""What a follower sample is compared against. See :class:`LeaderArmLoop`'s ``compare``."""


def _say(message: str) -> None:
    """The operator's console. Flushed: these lines are read while the arm is live."""
    print(message, flush=True)


def _warn(message: str) -> None:
    """The same console's error stream, for a fault and what was done about it."""
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------- trace
class Trace:
    """Per-tick timing and signal, buffered in RAM and written on the way out.

    **Nothing here may cost the control loop anything.** Force feedback that
    feels choppy is a timing complaint, so an instrument that wrote to disk on
    the leader thread would be measuring its own stalls: a 100 Hz loop has 10 ms
    a tick and one page-cache miss can eat it. So a sample is a tuple appended
    to a plain list -- no formatting, no locks, no I/O -- and the whole buffer
    is serialised once in :meth:`write`, after the arm is already slack.

    One list per thread and neither reads the other's, so there is nothing to
    synchronise: each leader thread appends only to its own ``ticks``, and the
    page thread appends only to ``publishes``.

    ``scripts/analyze_leader_trace.py`` reads the file this writes.

    Args:
        limit: Samples kept per stream, oldest dropped. An hour at 100 Hz is
            360k rows, which is fine in memory and slow to analyse; the default
            keeps the most recent ~10 minutes of a session, which is what a
            complaint is ever about.
    """

    TICK_FIELDS = (
        "t",
        "period",
        "late",
        "phase",
        "read_s",
        "propose_s",
        "render_s",
        "age",
        "stale_limit",
        "n_active",
        "n_arm",
        "n_release",
        "worst_mismatch",
        "worst_effective",
        "glitch",
        "worst_slot",
        "worst_speed",
        "worst_deadband",
        "max_speed",
    )
    """What one leader tick records. ``phase`` is 0 armed, 1 suspended, 2 no-feedback.

    ``render_s`` on a suspended tick is the verified torque-off, not a
    correction being rendered -- the same wall time, a different thing, and
    ``phase`` is what separates them.
    """

    PUBLISH_FIELDS = ("t", "gap", "step", "pump_s")
    """What one page publish records."""

    def __init__(self, limit: int = 60_000) -> None:
        self.limit = int(limit)
        self.ticks: dict[str, list[tuple]] = {}
        self.publishes: list[tuple] = []
        self.events: list[tuple[float, str, str]] = []
        self.joints: dict[str, list[str]] = {}
        self.started = time.monotonic()

    def arm(self, name: str, motors: list[str]) -> list[tuple]:
        """Register one arm and hand back the list its thread appends to."""
        self.joints[name] = list(motors)
        return self.ticks.setdefault(name, [])

    @staticmethod
    def add(rows: list[tuple], row: tuple, limit: int) -> None:
        """Append one sample, dropping the oldest once the cap is reached."""
        rows.append(row)
        if len(rows) > limit:
            del rows[: len(rows) // 4]  # amortised, so this is not a per-tick memmove

    def event(self, at: float, name: str, what: str) -> None:
        """A transition worth seeing beside the timing: suspend, re-arm, glitch.

        ``at`` is a raw ``time.monotonic()``; it is stored relative to the
        trace's own start so an event lines up with the tick rows rather than
        needing the reader to subtract a boot time.
        """
        self.events.append((at - self.started, name, what))

    # ------------------------------------------------------------------ out
    def write(self, path: Path) -> str:
        """One JSON document: metadata, per-arm tick tables, publishes, events.

        Columnar rather than a row per line -- the analysis this exists for is
        "what did the period do over ten minutes", which wants arrays, and a
        column of floats is a quarter the bytes of the same numbers as repeated
        JSON keys.
        """
        doc = {
            "version": 1,
            "started_monotonic": self.started,
            "tick_fields": list(self.TICK_FIELDS),
            "publish_fields": list(self.PUBLISH_FIELDS),
            "joints": self.joints,
            "arms": {
                name: {f: [row[i] for row in rows] for i, f in enumerate(self.TICK_FIELDS)}
                for name, rows in self.ticks.items()
            },
            "publishes": {f: [row[i] for row in self.publishes] for i, f in enumerate(self.PUBLISH_FIELDS)},
            "events": [{"t": t, "arm": a, "what": w} for t, a, w in self.events],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
        return self.summary()

    def summary(self) -> str:
        """The three numbers a choppiness complaint is usually about, per arm.

        Printed at exit because the answer is often already here: a loop whose
        period p99 is four times its target is not rendering a smooth force
        however good the signal is, and that is visible without opening the file.
        """

        def q(xs: list[float], p: float) -> float:
            return xs[min(len(xs) - 1, int(p * len(xs)))] * 1e3 if xs else 0.0

        out: list[str] = []
        idx = {f: i for i, f in enumerate(self.TICK_FIELDS)}
        for name, rows in self.ticks.items():
            if not rows:
                continue
            period = sorted(r[idx["period"]] for r in rows)
            render = sorted(r[idx["render_s"]] for r in rows if r[idx["phase"]] == 0)
            downs = sorted(r[idx["render_s"]] for r in rows if r[idx["phase"]] == 1)
            age = sorted(r[idx["age"]] for r in rows if r[idx["age"]] >= 0.0)
            arms_ = sum(r[idx["n_arm"]] for r in rows)
            rel = sum(r[idx["n_release"]] for r in rows)
            susp = sum(1 for _, a, w in self.events if a == name and w == "suspend")
            out.append(
                f"  [{name}] {len(rows)} ticks\n"
                f"      loop period   p50 {q(period, 0.5):6.2f}  p95 {q(period, 0.95):6.2f}  "
                f"p99 {q(period, 0.99):6.2f}  max {q(period, 1.0):7.2f} ms\n"
                f"      render        p50 {q(render, 0.5):6.2f}  p95 {q(render, 0.95):6.2f}  "
                f"max {q(render, 1.0):7.2f} ms\n"
                f"      sample age    p50 {q(age, 0.5):6.2f}  p95 {q(age, 0.95):6.2f}  "
                f"max {q(age, 1.0):7.2f} ms\n"
                f"      torque writes {arms_} arm / {rel} release   suspends {susp}"
                + (f"  (worst torque-off {q(downs, 1.0):.0f} ms)" if downs else "")
            )
        if self.publishes:
            gaps = sorted(r[1] for r in self.publishes if r[1] >= 0.0)
            out.append(
                f"  [page] {len(self.publishes)} publishes   gap p50 {q(gaps, 0.5):6.2f}  "
                f"p95 {q(gaps, 0.95):6.2f}  max {q(gaps, 1.0):7.2f} ms"
            )
        return "\n".join(out) or "  (nothing recorded)"


# --------------------------------------------------------------------- signal
@dataclass(frozen=True)
class ComplianceConfig:
    """How a state-action mismatch becomes a correction, and how it is rendered.

    The first eight are the *signal*, and every one is in the joint's own units
    (radians for an arm joint, a fraction of the stroke for the gripper) so the
    same numbers describe both. The last three are how **this** leader's
    Feetech servos render the result: position-loop P and D registers (0-254)
    and the torque duty cap (0-1000).

    The floor, ramp, cap and servo gains are the ones a reference teleoperation
    rig for the same leader runs. The velocity terms were fitted here, from
    recorded sessions of a bimanual YAM leader driving a simulated YAM.

    Args:
        deadband: Mismatch below which nothing is rendered at all, so free
            motion feels free. With :attr:`velocity_deadband` this is the
            *floor* -- what a hand holding still must exceed.
        velocity_deadband: Seconds of the operator's own speed added to the
            deadband, so the threshold rises with how fast the hand is moving.

            **This is a latency, not a gain.** A position-controlled simulated
            arm lags its target, and it lags almost purely: fitting
            ``follower(t) = leader(t - tau)`` to a recorded session left a
            residual of 0.009 rad against a 0.053 rad signal, so the arm is a
            delay line rather than a spring. A delay line's free-space mismatch
            is exactly ``speed * tau`` -- which means a fixed deadband asks one
            question at rest and a completely different one in motion. At
            ``tau`` of 25 ms and a 0.04 deadband, any hand faster than 1.6 rad/s
            reads as contact with nothing in the scene.

            0.045 is that ``tau``, fitted from a recorded session rather than
            from the sim: a first pass used 0.025, measured with a synthetic
            leader driving the page in a tight loop, and that left out the
            serial thread, the page's 17 ms publish gap and 9-24 ms of slot
            age. Against real motion the same fit came back 35-45 ms, and the
            recording says what each value would have cost::

                k        over deadband    while moving >0.5 units/s
                25 ms        23.9%                 34.5%
                45 ms       ~19%                  ~22%
                60 ms        17.2%                 17.4%
                75 ms        16.2%                 15.0%

            Past ~45 ms the curve flattens, because what is left is not lag:
            it is per-slot static offset, which no velocity term can remove.

            Setting this to the measured ``tau`` subtracts that term instead of
            fighting it: what is left over the threshold is the part of the
            mismatch a delay does not explain, which is what contact is. Slow,
            precise work keeps the full sensitivity of :attr:`deadband`; a fast
            reach stops manufacturing force out of its own lag.

            Linear because the lag is: ``speed * tau`` has no curvature to
            match, and anything steeper masks real contact during fast motion,
            which is the one place this must not.

            0 restores a fixed deadband.
        max_deadband: Ceiling on the scaled deadband, so a fast swing into
            something solid is still felt. Without it the threshold outruns
            every mismatch a real contact can produce.

            Sized against the floor and the slope together: at 0.045 s per
            unit/s, a 0.12 ceiling binds from 1.78 units/s up, and the recorded
            session spent 27% of its armed ticks between 1 and 2 units/s. A
            ceiling that binds through the middle of normal motion is a fixed
            deadband wearing a scaled one's name, so 0.16 moves the knee out to
            2.7 units/s -- past that speed contact is the only thing that can
            still be told apart from lag anyway.
        velocity_deadband_grip: The same slope for the **grip** slot, which
            needs its own because the number it multiplies is not comparable.

            Every other slot is radians; the grip is a fraction of a stroke, and
            a hand closes its whole travel in about a third of a second -- 3 or
            more fraction/s, where an arm joint rarely passes 2 rad/s. The arm's
            0.045 applied to that reaches the ceiling on an ordinary squeeze, so
            the grip slot goes quiet exactly while closing on something, which
            is the grasp this whole loop exists to render: a hand shut on a
            solid part stalls part way and a hand shut on air does not, and the
            difference between those is the only thing the gripper has to say.

            0.020 is the grip channel's own measured lag -- fitted at 20 ms
            against a trigger worked at 0.8 Hz -- so it subtracts that slot's
            delay and no more.
        max_deadband_grip: Ceiling for the grip slot, and tighter than the
            arm's for the same reason: a grasp is a *slow* signal against a
            fast command, so a ceiling that rises with the squeeze would hide
            the stall it is meant to report.
        ramp_gain: How steeply the correction grows past the deadband.
        release_hysteresis: How far back inside the deadband a mismatch must
            fall before an energized joint lets go, so a joint sitting on the
            threshold does not chatter on and off.
        max_error: Cap on the correction, so the goal written to a servo is
            never further than this from where the operator's hand actually is.
        p_coefficient: Feetech ``P_Coefficient`` while a joint is energized.
        d_coefficient: Feetech ``D_Coefficient`` while a joint is energized.
        torque_limit: Feetech ``Torque_Limit`` while a joint is energized. The
            ceiling on how hard the arm can push back.
    """

    deadband: float = 0.04
    velocity_deadband: float = 0.045
    max_deadband: float = 0.16
    velocity_deadband_grip: float = 0.020
    max_deadband_grip: float = 0.08
    ramp_gain: float = 5.0
    release_hysteresis: float = 0.01
    max_error: float = 0.2
    p_coefficient: int = 30
    d_coefficient: int = 5
    torque_limit: int = 300

    def __post_init__(self) -> None:
        if not math.isfinite(self.deadband) or not 0.0 <= self.deadband <= 1.0:
            raise ValueError("deadband must be in [0, 1]")
        if not math.isfinite(self.ramp_gain) or not 0.0 < self.ramp_gain <= 20.0:
            raise ValueError("ramp gain must be in (0, 20]")
        if not math.isfinite(self.release_hysteresis) or not 0.0 <= self.release_hysteresis <= self.deadband:
            raise ValueError("release hysteresis must be between 0 and the deadband")
        if not math.isfinite(self.velocity_deadband) or not 0.0 <= self.velocity_deadband <= 1.0:
            raise ValueError("velocity deadband must be in [0, 1] seconds")
        if not math.isfinite(self.velocity_deadband_grip) or not 0.0 <= self.velocity_deadband_grip <= 1.0:
            raise ValueError("grip velocity deadband must be in [0, 1] seconds")
        if not math.isfinite(self.max_deadband_grip) or not self.deadband <= self.max_deadband_grip <= 1.0:
            raise ValueError("grip max deadband must be between the deadband floor and 1")
        if not math.isfinite(self.max_deadband) or not self.deadband <= self.max_deadband <= 1.0:
            raise ValueError("max deadband must be between the deadband floor and 1")
        if not math.isfinite(self.max_error) or not 0.0 < self.max_error <= 1.0:
            raise ValueError("max error must be in (0, 1]")
        if not 0 <= self.p_coefficient <= 254:
            raise ValueError("P coefficient must be in [0, 254]")
        if not 0 <= self.d_coefficient <= 254:
            raise ValueError("D coefficient must be in [0, 254]")
        if not 1 <= self.torque_limit <= 1000:
            raise ValueError("torque limit must be in [1, 1000]")

    @property
    def release_threshold(self) -> float:
        """Mismatch at which an already-energized joint lets go, at rest."""
        return self.deadband - self.release_hysteresis

    def deadband_at(self, speed: float, *, grip: bool = False) -> float:
        """The live threshold for a slot whose own coordinate is moving at ``speed``.

        ``speed`` is in the slot's units per second -- rad/s for an arm joint,
        stroke-fractions per second for the grip -- which is what lets one
        number serve both: the lag it compensates is measured in the same units
        it is expressed in.
        """
        slope = self.velocity_deadband_grip if grip else self.velocity_deadband
        ceiling = self.max_deadband_grip if grip else self.max_deadband
        if slope <= 0.0:
            return self.deadband
        return min(ceiling, self.deadband + slope * abs(speed))

    def release_at(self, speed: float, *, grip: bool = False) -> float:
        """Where an energized joint lets go, at this speed.

        Scaled with the deadband rather than held at a fixed offset below it:
        the hysteresis exists to stop a joint chattering across its own
        threshold, so it has to stay a proportion of that threshold. A 0.01
        band under a deadband that has risen to 0.12 is not hysteresis, it is a
        rounding error -- and the recorded trace that prompted this showed the
        mismatch moving 0.0255 rad in a single 10 ms tick, twice the width of
        the fixed band.
        """
        if self.deadband <= 0.0:
            return self.deadband_at(speed, grip=grip)
        return self.deadband_at(speed, grip=grip) * (self.release_threshold / self.deadband)


def correction(
    leader: float,
    follower: float,
    config: ComplianceConfig,
    *,
    limits: tuple[float, float] | None = None,
    speed: float = 0.0,
    grip: bool = False,
    reference: float | None = None,
) -> tuple[float, float, float]:
    """Deadband, ramp and cap on one joint's mismatch.

    Returns ``(target, mismatch, effective)``: where to command the servo, the
    raw gap it came from, and the correction actually applied after the ramp
    and the cap. ``target`` leads the operator's hand *toward* the simulated
    arm, which is the direction that reads as the scene pushing back.

    Args:
        leader: Where the operator's hand is.
        follower: Where the simulated arm actually is.
        config: The signal parameters.
        limits: Clamp for a joint with a bounded range -- the gripper's
            ``(0, 1)``. Arm joints pass ``None``; the cap already keeps their
            target within ``max_error`` of a real measured position.
        speed: How fast this slot is being moved [units/s], for
            :meth:`ComplianceConfig.deadband_at`. 0 is a hand at rest and gives
            the fixed-deadband behaviour.
        grip: Whether this is the grip slot, which has its own slope and ceiling.
        reference: Where the operator's hand *was* when ``follower`` was
            measured, when that is known and differs from ``leader``. The
            mismatch is taken against this and the target is still built on
            ``leader``, the hand's position now. ``None`` compares against
            ``leader``.
    """
    if not math.isfinite(leader) or not math.isfinite(follower):
        raise ValueError("feedback positions must be finite")
    if reference is None or not math.isfinite(reference):
        reference = leader
    lower = upper = 0.0
    if limits is not None:
        lower, upper = limits
        if not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper:
            raise ValueError("feedback limits must be finite and increasing")
        leader = max(lower, min(upper, leader))
        follower = max(lower, min(upper, follower))
        reference = max(lower, min(upper, reference))
    mismatch = follower - reference
    excess = max(abs(mismatch) - config.deadband_at(speed, grip=grip), 0.0)
    ramped = config.ramp_gain * excess
    effective = math.copysign(ramped, mismatch) if ramped > 0.0 else 0.0
    effective = max(-config.max_error, min(config.max_error, effective))
    target = leader + effective
    if limits is not None:
        target = max(lower, min(upper, target))
    return target, mismatch, effective


@dataclass
class JointGate:
    """One joint's live feedback state, owned by its arm's leader thread.

    Args:
        active: Whether this motor is currently energized.
        neutralized: Whether this joint has passed the re-arm dwell since it
            was last suspended. A suspended joint may not energize on the
            strength of the very sample that ends its suspension.
        neutral_since: When continuous alignment began, or ``None`` if the last
            sample broke it.
    """

    active: bool = False
    neutralized: bool = True
    neutral_since: float | None = None


@dataclass
class JointDecision:
    """What one joint's mismatch came to this tick, for rendering and readout."""

    index: int
    motor: str
    leader: float
    follower: float
    mismatch: float
    effective: float
    active: bool
    target: float
    target_ticks: int
    speed: float = 0.0
    """How fast this slot was moving, which set its threshold."""
    deadband: float = 0.0
    """The threshold that speed produced, recorded so a trace can show it."""


@dataclass
class Proposal:
    """One tick's decisions, plus the gates to commit once they are rendered.

    Two phase on purpose. The caller renders the decisions on its hardware --
    disarm ``newly_inactive`` first, re-target ``continuing``, then arm
    ``newly_active`` -- and only assigns ``gates`` after every write succeeded.
    A failed torque-off must abort the commit so the joint is still considered
    active next tick and the disarm is retried; committing early would leave
    software believing torque is off while the servo stayed live.
    """

    decisions: list[JointDecision]
    gates: list[JointGate]
    newly_active: list[int]
    newly_inactive: list[int]
    continuing: dict[str, int]
    rearmed: bool
    """True on the tick every joint finished its dwell -- worth saying out loud."""


def propose(
    config: ComplianceConfig,
    layout: ArmLayout,
    *,
    leader: list[float],
    follower: list[float],
    gates: list[JointGate],
    now: float,
    neutral_dwell_s: float,
    speeds: list[float] | None = None,
    reference: list[float] | None = None,
) -> Proposal:
    """Decide every joint's coupling from the mismatch. Pure; touches no bus.

    A joint that is not ``neutralized`` must first hold continuous alignment
    for ``neutral_dwell_s`` before it is even *eligible* to energize, and the
    dwell completing only arms it -- a later above-deadband sample is what
    actually switches torque on. Requiring the second sample is what stops a
    joint energizing in the same tick its dwell finished, off a mismatch
    measured while the scene was still settling.
    """
    was_waiting = not all(gate.neutralized for gate in gates)
    decisions: list[JointDecision] = []
    nxt: list[JointGate] = []
    for i in range(layout.width):
        speed = speeds[i] if speeds is not None else 0.0
        # The grip slot is the last one, and the only one whose units are a
        # fraction rather than radians -- which is exactly why it needs its own
        # slope. layout.limits_of marks it for the same reason.
        is_grip = i == layout.width - 1
        target, mismatch, effective = correction(
            leader[i],
            follower[i],
            config,
            limits=layout.limits_of(i),
            speed=speed,
            grip=is_grip,
            reference=reference[i] if reference is not None else None,
        )
        # Both thresholds move with the slot's own speed, so a joint that armed
        # while the hand was fast is judged on the same scale when it is asked
        # to let go -- reading ``release_threshold`` here instead would release
        # against a bar the arming never had to clear.
        release_threshold = config.release_at(speed, grip=is_grip)
        gate = gates[i]
        neutralized, neutral_since, active = gate.neutralized, gate.neutral_since, False
        if not neutralized:
            if abs(mismatch) <= release_threshold:
                if neutral_since is None:
                    neutral_since = now
                elif now - neutral_since >= neutral_dwell_s:
                    neutralized, neutral_since = True, None
            else:
                # Alignment must be *continuous*: a scene mid-settle can look
                # aligned for one frame between two that are not.
                neutral_since = None
        else:
            neutral_since = None
            active = abs(mismatch) > release_threshold if gate.active else effective != 0.0
        nxt.append(JointGate(active=active, neutralized=neutralized, neutral_since=neutral_since))
        decisions.append(
            JointDecision(
                index=i,
                motor=layout.motor_of(i),
                leader=leader[i],
                follower=follower[i],
                mismatch=mismatch,
                effective=effective,
                active=active,
                target=target,
                target_ticks=layout.ticks_of(i, target),
                speed=speed,
                deadband=config.deadband_at(speed, grip=is_grip),
            )
        )
    return Proposal(
        decisions=decisions,
        gates=nxt,
        newly_active=[i for i in range(layout.width) if nxt[i].active and not gates[i].active],
        newly_inactive=[i for i in range(layout.width) if gates[i].active and not nxt[i].active],
        continuing={
            decisions[i].motor: decisions[i].target_ticks
            for i in range(layout.width)
            if nxt[i].active and gates[i].active
        },
        rearmed=was_waiting and all(gate.neutralized for gate in nxt),
    )


# --------------------------------------------------------------------- layout
class ArmLayout:
    """One arm's ``[q_0 .. q_{n-1}, grip]`` slots, and what each one converts to.

    The same layout :func:`~robosimstudio.robots.teleop.lerobot_controller.build_action`
    emits and ``ActionJointPosition`` applies, so a leader reading, a follower
    measurement and a feedback gate all index alike and no conversion needs a
    name lookup in the loop.
    """

    def __init__(self, calibration: ArmCalibration) -> None:
        self.calibration = calibration
        self.joints = calibration.joints
        self.gripper = calibration.gripper
        self.width = len(self.joints) + 1
        """Joints plus the grip slot, which is present even with no gripper."""

    @property
    def motors(self) -> list[str]:
        """Every motor name on this arm's bus, in slot order."""
        names = [joint.name for joint in self.joints]
        if self.gripper is not None:
            names.append(self.gripper.name)
        return names

    def motor_of(self, index: int) -> str:
        """The motor behind one slot; ``""`` for the grip slot of an arm with no gripper."""
        if index < len(self.joints):
            return self.joints[index].name
        return self.gripper.name if self.gripper is not None else ""

    def limits_of(self, index: int) -> tuple[float, float] | None:
        """``(0, 1)`` for the grip slot, which is a fraction of a stroke."""
        return None if index < len(self.joints) else (0.0, 1.0)

    def ticks_of(self, index: int, value: float) -> int:
        """A target in the slot's own units, as a raw goal position.

        Wrapped to the encoder's turn the way every Feetech goal is: these
        servos count 0..4095 and a target is only ever within ``max_error`` of
        a position that was just read back, so the wrap is normalization rather
        than a range decision.
        """
        if index < len(self.joints):
            return int(round(self.joints[index].to_ticks(value))) % ENCODER_RESOLUTION
        if self.gripper is None:
            return 0
        return int(round(self.gripper.to_ticks(value))) % ENCODER_RESOLUTION

    def gates(self) -> list[JointGate]:
        """A fresh gate per slot, suspended until it proves alignment."""
        return [JointGate(active=False, neutralized=False, neutral_since=None) for _ in range(self.width)]

    def read(self, ticks: Mapping[str, float]) -> list[float]:
        """One bus reading as ``[q_0 .. q_{n-1}, grip]`` in the sim's units."""
        values = [joint.to_radians(ticks[joint.name]) for joint in self.joints]
        values.append(self.gripper.to_command(ticks[self.gripper.name]) if self.gripper is not None else 0.0)
        return values


# ------------------------------------------------------------------- hardware
class ServoFault(LeaderError, RuntimeError):
    """A safety fault on a leader bus: stop commanding and tear the session down."""


class LeaderServos:
    """One leader arm's bus: reads its encoders, renders corrections as torque.

    Owns the connection outright. Nothing else may touch this bus -- which is
    why :class:`StudioBridge` stands the studio's own ``Leader arm`` panel
    down: its ``drain()`` would read the same port from the page's thread.

    Args:
        port: The arm's serial device.
        layout: Its slots.
        config: How a correction is rendered.
        connect: ``connect(port, [calibration])`` returning something with a
            ``bus`` and a ``close()``. Defaults to
            :meth:`~robosimstudio.robots.teleop.lerobot_controller.LerobotFeetechController.connect`,
            which is where ``lerobot`` is imported; a test or a simulated
            leader passes its own.
    """

    CONFIG_REGISTERS = ("Operating_Mode", "P_Coefficient", "D_Coefficient", "I_Coefficient")
    """Written to arm feedback, restored on the way out. These live in EEPROM."""

    def __init__(
        self,
        port: str,
        layout: ArmLayout,
        config: ComplianceConfig,
        *,
        connect: Callable[..., Any] | None = None,
    ) -> None:
        self.port = port
        self.layout = layout
        self.config = config
        self.controller: Any = None
        self.saved: dict[str, dict[str, int]] = {}
        self.armed = False
        self._connect = connect

    @property
    def bus(self):
        if self.controller is None:
            raise ServoFault(f"{self.port} is not connected")
        return self.controller.bus

    def connect(self, *, configure: bool = True) -> None:
        """Open the bus, prove torque is off, and optionally arm the feedback loop.

        Args:
            configure: Whether to install the feedback gains. ``False`` is the
                no-feedback path, and it stops short deliberately: those gains
                live in servo **EEPROM**, so writing them for a run that will
                never energize a motor is wear paid for nothing, and it would
                leave a "dry run" having reconfigured the arm. Torque is still
                taken off and verified either way -- that part is not optional,
                whatever the run intends to do afterwards.

        Raises:
            LeaderUnavailable: If the bus could not be opened at all.
            ServoFault: If torque could not be proven off, or a servo came up
                with a nonzero torque limit.
        """
        leader = LeaderPort(name="", port=self.port, serial="", calibration=self.layout.calibration)
        self.controller = open_leader(leader, self._connect)
        motors = self.layout.motors
        self._torque_off(motors, passes=2)
        self._refuse_nonzero_limits(motors)
        if configure:
            self._configure(motors)

    def read_ticks(self) -> dict[str, float]:
        """Raw ``Present_Position`` per motor. ``normalize=False``: the
        calibration is measured against the counts underneath the bus's own
        normalization, not the -100..100 the servo is configured for."""
        return self.bus.sync_read("Present_Position", self.layout.motors, normalize=False, num_retry=2)

    def _configure(self, motors: list[str]) -> None:
        """Install the feedback gains, park every goal, and leave torque off.

        Configuring is not an armed state, and the ordering says so: the
        registers go in while torque is off, the goals are parked at wherever
        the arm currently *is* so a later enable cannot yank it, the torque
        limits go to zero, and only then -- after both gates are read back and
        verified shut -- does this call itself armed.
        """
        present = self.read_ticks()
        # Keep an existing snapshot: after a suspend/resume the live values are
        # the feedback gains, not the originals this has to put back.
        if not self.saved:
            self.saved = {
                motor: {
                    register: int(self.bus.read(register, motor, normalize=False, num_retry=3))
                    for register in self.CONFIG_REGISTERS
                }
                for motor in motors
            }
        desired = {
            "Operating_Mode": 0,
            "P_Coefficient": self.config.p_coefficient,
            "D_Coefficient": self.config.d_coefficient,
            "I_Coefficient": 0,
        }
        for motor in motors:
            for register, value in desired.items():
                current = int(self.bus.read(register, motor, normalize=False, num_retry=3))
                if current != value:
                    self.bus.write(register, motor, value, normalize=False, num_retry=3)
                    time.sleep(EEPROM_WRITE_SETTLE_S)
        self.bus.sync_write(
            "Goal_Position",
            {motor: int(present[motor]) for motor in motors},
            normalize=False,
            num_retry=2,
        )
        self._zero_limits(motors)
        self._torque_off(motors, passes=1)
        self._refuse_nonzero_limits(motors)
        self.armed = True

    def render(self, proposal: Proposal) -> None:
        """Disarm, re-target, then arm -- in that order, on this arm's servos.

        Letting go first is what keeps a tick from ever *adding* torque on the
        way to removing it. Continuing goals ride a single unacknowledged
        ``sync_write``, which is the fast path and safe because those motors
        are already energized against a goal a few ticks old. Arming does not:
        a lost broadcast that enabled torque against a stale goal is exactly
        the failure worth paying acknowledged writes to avoid.
        """
        decisions = proposal.decisions
        if proposal.newly_inactive:
            self._release([decisions[i].motor for i in proposal.newly_inactive])
        if proposal.continuing:
            self.bus.sync_write("Goal_Position", proposal.continuing, normalize=False, num_retry=2)
        if proposal.newly_active:
            motors = [decisions[i].motor for i in proposal.newly_active]
            for i in proposal.newly_active:
                self.bus.write(
                    "Goal_Position", decisions[i].motor, decisions[i].target_ticks, normalize=False, num_retry=3
                )
                self.bus.write(
                    "Torque_Limit", decisions[i].motor, self.config.torque_limit, normalize=False, num_retry=3
                )
            self.bus.enable_torque(motors, num_retry=3)

    def disarm(self) -> None:
        """Take torque off every motor and verify it. Safe to call repeatedly."""
        if self.controller is None:
            return
        self._torque_off(self.layout.motors, passes=1)

    def close(self) -> None:
        """Disarm, put the EEPROM registers back, and prove torque is still off."""
        if self.controller is None:
            return
        motors = self.layout.motors
        errors: list[str] = []
        try:
            self._torque_off(motors, passes=2)
        except Exception as exc:
            errors.append(f"{self.port}: torque-off failed: {exc}")
        for motor, registers in self.saved.items():
            for register, value in registers.items():
                try:
                    self.bus.write(register, motor, value, normalize=False, num_retry=3)
                except Exception as exc:
                    errors.append(f"{self.port}: could not restore {register} on {motor}: {exc}")
        try:
            # Restoring gains rewrites EEPROM, so read the torque gate back one
            # more time after it: this is the last chance to notice a servo
            # that came back up energized.
            self._torque_off(motors, passes=1)
        except Exception as exc:
            errors.append(f"{self.port}: torque-off after gain restore failed: {exc}")
        try:
            self.controller.close()
        except Exception as exc:
            errors.append(f"{self.port}: disconnect failed: {exc}")
        self.controller, self.armed = None, False
        if errors:
            raise ServoFault("; ".join(errors))

    # ------------------------------------------------------------ torque gates
    def _release(self, motors: list[str]) -> None:
        """Let go of these motors, inside the control loop, without stalling it.

        The difference from :meth:`_torque_off` is the settle-and-verify, and
        it is deliberate rather than a shortcut. ``disable_torque`` is an
        *acknowledged* write: a servo that answers has taken the command, and
        there is nothing a readback 250 ms later would add. Paying for one
        anyway would stall this arm's loop for twenty-five ticks at the exact
        moment contact is released -- which the operator feels as a hitch on
        letting go of something, the one place the arm should feel free.

        So the settle is spent only where it buys something: on a servo that
        did **not** acknowledge, where the command may genuinely be lost.
        """
        unacknowledged: list[str] = []
        for motor in motors:
            try:
                self.bus.disable_torque(motor, num_retry=3)
            except Exception:  # answered-for by the verification below
                unacknowledged.append(motor)
        # Unacknowledged is fine for the limits: torque is already gated off
        # above, and arming always rewrites the limit with an acknowledged
        # write before it enables anything.
        self.bus.sync_write("Torque_Limit", dict.fromkeys(motors, 0), normalize=False, num_retry=2)
        if unacknowledged:
            time.sleep(TORQUE_OFF_SETTLE_S)
            stuck = self._still_energized(motors)
            if stuck:
                raise ServoFault(f"{self.port}: torque is still enabled on {', '.join(stuck)}")

    def _torque_off(self, motors: list[str], *, passes: int) -> None:
        """Disable torque, zero the limits, and read both back.

        Every motor is attempted even when an earlier one does not acknowledge:
        one servo swallowing its packet must not skip the rest of the arm.
        """
        for _ in range(max(1, passes)):
            for index, motor in enumerate(motors):
                try:
                    self.bus.disable_torque(motor, num_retry=4)
                except Exception:  # verified by readback below
                    pass
                if index + 1 < len(motors):
                    time.sleep(COMMAND_GAP_S)
        self._zero_limits(motors)
        time.sleep(TORQUE_OFF_SETTLE_S)
        stuck = self._still_energized(motors)
        if stuck:
            for motor in stuck:
                try:
                    self.bus.disable_torque(motor, num_retry=4)
                except Exception:
                    pass
                time.sleep(COMMAND_GAP_S)
            time.sleep(TORQUE_OFF_SETTLE_S)
            stuck = self._still_energized(motors)
        if stuck:
            raise ServoFault(f"{self.port}: torque is still enabled on {', '.join(stuck)}")

    def _still_energized(self, motors: list[str]) -> list[str]:
        """Which motors read back as energized. An unreadable one counts as live.

        Independent per-servo reads rather than one grouped read: a grouped
        parse that dropped a motor would report it as off, which is the one
        wrong answer this check must never give.
        """
        stuck: list[str] = []
        for motor in motors:
            try:
                if int(self.bus.read("Torque_Enable", motor, normalize=False, num_retry=4)) != 0:
                    stuck.append(motor)
            except Exception:  # unreadable is not proof of off
                stuck.append(motor)
        return stuck

    def _zero_limits(self, motors: list[str]) -> None:
        """Acknowledged per-servo ``Torque_Limit`` zeroing, never a broadcast.

        A missed broadcast would leave one servo holding its feedback limit
        while this code believed the whole arm was slack.
        """
        for index, motor in enumerate(motors):
            self.bus.write("Torque_Limit", motor, 0, normalize=False, num_retry=3)
            if index + 1 < len(motors):
                time.sleep(COMMAND_GAP_S)

    def _refuse_nonzero_limits(self, motors: list[str]) -> None:
        limits = {motor: int(self.bus.read("Torque_Limit", motor, normalize=False, num_retry=4)) for motor in motors}
        nonzero = {motor: value for motor, value in limits.items() if value != 0}
        if nonzero:
            raise ServoFault(f"{self.port}: refusing to arm against nonzero torque limits: {nonzero}")


# ----------------------------------------------------------------------- loop
@dataclass
class Sample:
    """One side's latest reading, and when it was taken.

    Args:
        values: ``[q_0 .. q_{n-1}, grip]``.
        at: When it was read.
        reference: For a follower sample, the pose the simulated arm was
            *chasing* when it was measured -- the leader pose applied one page
            step earlier -- or ``None`` when there was none yet.
    """

    values: list[float]
    at: float
    reference: list[float] | None = None


class LeaderArmLoop:
    """One arm's leader thread: its own clock, its own bus, no Warp.

    Publishes what it reads and consumes what the page publishes, through two
    slots under one lock. Everything else it does -- the correction, the gates,
    the servo writes -- happens entirely on this thread.

    Args:
        name: The arm's name, for messages and the trace.
        servos: Its bus.
        config: The signal and how it is rendered.
        rate_hz: The thread's own rate.
        stale_after_s: Floor for how old a follower sample may get before the
            arm is dropped. See :meth:`stale_limit`.
        stale_factor: How many of the page's own step intervals may pass first.
        neutral_dwell_s: Continuous alignment a suspended joint must hold
            before it may energize again.
        feedback: ``False`` reads and publishes only: no proposal is formed and
            no torque register is written from the loop.
        trace: Where per-tick timing goes, when somebody asked.
        notify: Where the two operator messages go -- suspended, and live again.
        compare: What the mismatch is taken against.

            ``"applied"`` compares each follower sample with the leader pose it
            was *chasing*: the action the page applied one step before it read
            the sample back. What is left is the servo's own tracking error --
            the arm not getting where it was told -- and nothing about how long
            the sample took to arrive.

            ``"live"`` compares it with the hand's pose at the instant this
            thread looks, which is what the unmerged script this module came
            from did. Every delay between the two then reads as mismatch: the
            sample's age, and above all the page's own step interval, since a
            target that is refreshed late leaves the arm genuinely behind a hand
            that kept moving. The velocity-scaled deadband was fitted to absorb
            the *typical* delay (35-45 ms), so a late step goes straight through
            it. Stepped in virtual time through one free-space pass of 650
            leader ticks (every joint and the jaw moving, nothing in reach),
            with the page's step made irregular -- ticks on which some motor
            was energized, and the worst mismatch as a fraction of its live
            deadband, arm joints / jaw::

                page step                     "live"               "applied"
                steady 16.7 ms                  0  (0.47 / 0.93)      0  (0.17 / 0.61)
                jittered 13-23 ms              13  (0.49 / 1.22)      0  (0.14 / 0.82)
                one step in 12 late (60 ms)    54  (0.87 / 1.86)      3  (0.29 / 1.45)
                one in 6 late (70-100 ms)     178  (1.27 / 3.34)     36  (0.43 / 1.82)

            What is left under ``"applied"`` is all on the jaw and is the
            simulated finger itself: after a late step its target jumps by
            everything the hand did meanwhile, and one step later it is still
            part way there. The contact cases -- a welded wall, a jaw on a
            welded bar -- read the same either way, because a blocked arm is
            behind its target whichever pose that target is compared through.

            **Only the ``"live"`` form has been felt on hardware.** It is kept
            as a switch for exactly that reason: the default is what the
            software evidence supports, and the rig is what decides.
    """

    SPEED_SMOOTHING = 0.25
    """EMA weight on the newest speed sample: ~40 ms of memory at 100 Hz.

    A differentiated encoder is noisy, and the deadband this drives has to be
    *quieter* than the mismatch it is thresholding or it manufactures the
    chatter it exists to remove. Slow enough to ride out a single bad read,
    fast enough that stopping the hand restores full sensitivity within about
    two frames of the page.
    """

    def __init__(
        self,
        name: str,
        servos: LeaderServos,
        config: ComplianceConfig,
        *,
        rate_hz: float = 100.0,
        stale_after_s: float = 0.25,
        stale_factor: float = 3.0,
        neutral_dwell_s: float = 0.5,
        feedback: bool = True,
        trace: Trace | None = None,
        notify: Callable[[str], None] | None = None,
        compare: str = "applied",
    ) -> None:
        if not 0.0 < rate_hz <= 500.0:
            raise ValueError("the leader rate must be within (0, 500] Hz")
        if not stale_after_s > 0.0:
            raise ValueError("the stale floor must be positive")
        if compare not in COMPARISONS:
            raise ValueError(f"compare must be one of {COMPARISONS}, got {compare!r}")
        self.compare = compare
        self.name = name
        self.trace = trace
        self._rows: list[tuple] = trace.arm(name, servos.layout.motors) if trace is not None else []
        self.servos = servos
        self.config = config
        self.period = 1.0 / rate_hz
        self.stale_after_s = stale_after_s
        self.stale_factor = stale_factor
        self.neutral_dwell_s = neutral_dwell_s
        self.feedback = feedback
        self.notify = notify if notify is not None else _say
        self.layout = servos.layout
        self.gates = self.layout.gates()
        self.fault: BaseException | None = None
        self.decisions: list[JointDecision] = []
        """The last tick's readout. Assigned whole, so a reader never tears."""
        self.suspended = True
        self._lock = threading.Lock()
        self._leader: Sample | None = None
        self._follower: Sample | None = None
        self._gaps: deque[float] = deque(maxlen=8)
        """Recent intervals between the page's publishes: its measured cadence."""
        self._period_measured = 0.0
        self._late = 0.0
        self._glitch = 0
        self._speeds = [0.0] * self.layout.width
        """Per-slot |d(position)/dt|, low-pass filtered. Feeds the live deadband."""
        self._prev_leader: list[float] | None = None
        self._prev_at = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"leader-{name}", daemon=True)

    # ------------------------------------------------------------- the slots
    def publish_follower(self, values: list[float], at: float, reference: list[float] | None = None) -> None:
        """Called from the page's thread with the measured simulated arm.

        Also measures how often the page is managing to publish, which is what
        :meth:`stale_limit` reads. A gap over five seconds is not cadence -- it
        is the page having been away -- so it is not allowed to teach this that
        slow is normal.

        Args:
            values: The measured ``[q_0 .. q_{n-1}, grip]``.
            at: When it was read.
            reference: The pose this measurement was chasing; see :class:`Sample`.
        """
        with self._lock:
            if self._follower is not None:
                gap = at - self._follower.at
                if 0.0 < gap < 5.0:
                    self._gaps.append(gap)
            self._follower = Sample(values, at, reference)

    def stale_limit(self) -> float:
        """How old a follower sample may be before the arm is dropped.

        A fixed threshold cannot serve both a scene that steps in 5 ms and one
        that steps in 400 -- and heavy scenes are real: one contact-rich scene
        this was developed against spent 275 ms per step in physics alone (p95
        452 ms), so a flat 250 ms would call a perfectly healthy page stalled
        on nearly every step and thrash the arm between suspended and
        re-arming.

        So the gate asks the question it actually means -- "has the page
        stopped stepping *as it had been*" -- by scaling the measured cadence,
        with ``stale_after_s`` as a floor for a fast scene. The median rather
        than the mean, so one slow step does not raise the bar for the next
        eight.
        """
        with self._lock:
            gaps = list(self._gaps)
        if len(gaps) < 2:
            return self.stale_after_s
        return max(self.stale_after_s, self.stale_factor * statistics.median(gaps))

    def latest_leader(self) -> list[float] | None:
        """The freshest leader reading, for the page's thread to apply."""
        with self._lock:
            return None if self._leader is None else list(self._leader.values)

    def _take_follower(self) -> Sample | None:
        with self._lock:
            return self._follower

    def _put_leader(self, values: list[float], at: float) -> None:
        with self._lock:
            self._leader = Sample(values, at)

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Ask the thread to finish, then wait for it to put torque down itself.

        Joining rather than disarming from here keeps the bus single-threaded
        to the very end -- the loop's own ``finally`` is what disarms.
        """
        self._stop.set()
        if not self._thread.is_alive():
            return
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            # It did not come back, so this is the last chance to remove torque
            # and a concurrent write is the lesser risk.
            self.servos.disarm()

    def _run(self) -> None:
        glitches = 0
        deadline = time.monotonic()
        last = None
        try:
            while not self._stop.is_set():
                woke = time.monotonic()
                # The period as *measured*, not as asked for: the gap between
                # consecutive tick starts is what the operator's hand feels, and
                # it is the one number a "choppy" complaint is always about.
                self._period_measured = 0.0 if last is None else woke - last
                self._late = woke - deadline
                last = woke
                self._glitch = 0
                try:
                    self.tick(woke)
                    glitches = 0
                except ConnectionError as exc:
                    # A dropped status packet should not end a session; the arm
                    # barely moves in one period. A run of them is a real fault.
                    glitches += 1
                    self._glitch = glitches
                    if self.trace is not None:
                        self.trace.event(woke, self.name, f"glitch:{type(exc).__name__}")
                    if glitches > MAX_BUS_GLITCHES:
                        raise ServoFault(f"{self.name}: {MAX_BUS_GLITCHES} consecutive bus glitches: {exc}") from exc
                deadline += self.period
                delay = deadline - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -self.period:
                    deadline = time.monotonic()
        except BaseException as exc:  # recorded, then reported by the session
            self.fault = exc
        finally:
            try:
                self.servos.disarm()
            except BaseException as exc:
                if self.fault is None:
                    self.fault = exc

    def _update_speeds(self, leader: list[float], now: float) -> None:
        """One filtered speed per slot, in that slot's own units per second."""
        prev, prev_at = self._prev_leader, self._prev_at
        self._prev_leader, self._prev_at = list(leader), now
        if prev is None:
            return
        dt = now - prev_at
        if not 1.0e-4 < dt < 1.0:  # a stalled or resumed thread is not a velocity
            return
        a = self.SPEED_SMOOTHING
        for i in range(min(len(prev), len(leader), len(self._speeds))):
            self._speeds[i] = (1.0 - a) * self._speeds[i] + a * abs(leader[i] - prev[i]) / dt

    def tick(self, now: float) -> None:
        """Read the arm, publish it, and render whatever the mismatch asks for.

        One iteration of the thread's loop, callable directly by whoever owns
        the clock instead -- a test stepping the two halves in a fixed order
        does exactly that, and never starts the thread. ``now`` is the time
        the sample is stamped with and staleness is judged against.

        The ``t0..t3`` stamps are the trace's, and they bracket exactly the
        three things that can make a tick overrun: the bus read, the pure
        decision, and the writes that render it. Taken unconditionally --
        ``time.monotonic()`` is tens of nanoseconds and branching on whether a
        trace exists would put a different code path under the instrument than
        the one that runs without it.
        """
        t0 = time.monotonic()
        leader = self.layout.read(self.servos.read_ticks())
        self._put_leader(leader, now)
        self._update_speeds(leader, now)
        t1 = time.monotonic()
        if not self.feedback:
            self._record(now, t1 - t0, 0.0, 0.0, phase=2, age=-1.0, proposal=None)
            return
        follower = self._take_follower()
        limit = self.stale_limit()
        age = now - follower.at if follower is not None else -1.0
        if follower is None or age > limit:
            # The page is paused, loading, or gone. Rendering force against a
            # mismatch this old would push toward where the arm used to be.
            was = self.suspended
            self._suspend()
            if not was and self.trace is not None:
                self.trace.event(now, self.name, "suspend")
            self._record(now, t1 - t0, 0.0, time.monotonic() - t1, phase=1, age=age, proposal=None, limit=limit)
            return
        proposal = propose(
            self.config,
            self.layout,
            leader=leader,
            follower=follower.values,
            gates=self.gates,
            now=now,
            neutral_dwell_s=self.neutral_dwell_s,
            speeds=self._speeds,
            reference=follower.reference if self.compare == "applied" else None,
        )
        t2 = time.monotonic()
        self.servos.render(proposal)
        t3 = time.monotonic()
        # Only now: every write above succeeded. A raised torque-off leaves the
        # old gates in place so the joint is still considered active and the
        # disarm is retried next tick.
        self.gates = proposal.gates
        self.decisions = proposal.decisions
        self.suspended = False
        self._record(now, t1 - t0, t2 - t1, t3 - t2, phase=0, age=age, proposal=proposal, limit=limit)
        if proposal.rearmed:
            if self.trace is not None:
                self.trace.event(now, self.name, "rearm")
            self.notify(f"[{self.name}] alignment held; contact feedback is live again.")

    def _record(
        self,
        now: float,
        read_s: float,
        propose_s: float,
        render_s: float,
        *,
        phase: int,
        age: float,
        proposal: Proposal | None,
        limit: float = 0.0,
    ) -> None:
        """One row into this thread's own list. Never raises into the loop."""
        if self.trace is None:
            return
        n_active = 0
        worst_m = worst_e = 0.0
        worst_slot, worst_speed, worst_db = -1, 0.0, 0.0
        if proposal is not None:
            n_active = sum(1 for g in proposal.gates if g.active)
            worst_e = max((abs(d.effective) for d in proposal.decisions), default=0.0)
            if proposal.decisions:
                # *Which* slot, not just how much: the grip slot is a stroke
                # fraction judged beside six radian joints, so an aggregate
                # worst-of-seven cannot say whether the hand or the arm is
                # driving the chatter.
                worst = max(proposal.decisions, key=lambda d: abs(d.mismatch))
                worst_m, worst_slot = abs(worst.mismatch), worst.index
                worst_speed, worst_db = worst.speed, worst.deadband
        Trace.add(
            self._rows,
            (
                now - self.trace.started,
                self._period_measured,
                self._late,
                phase,
                read_s,
                propose_s,
                render_s,
                age,
                limit,
                n_active,
                len(proposal.newly_active) if proposal is not None else 0,
                len(proposal.newly_inactive) if proposal is not None else 0,
                worst_m,
                worst_e,
                self._glitch,
                worst_slot,
                worst_speed,
                worst_db,
                max(self._speeds) if self._speeds else 0.0,
            ),
            self.trace.limit,
        )

    def _suspend(self) -> None:
        """Drop every motor and put each one back behind the re-arm dwell.

        A transition, not a per-tick action. Once the arm is down there is
        nothing left to write and nothing to read: a verified torque-off sleeps
        and reads back every motor, so repeating it while a page sat paused
        would drop this loop from 100 Hz to about four and flood the bus to
        confirm, over and over, something it already knows.

        ``suspended`` is set only after the disarm succeeds, so a torque-off
        that raised is retried on the next tick rather than being remembered as
        done.
        """
        if self.suspended:
            return
        self.servos.disarm()
        self.gates = self.layout.gates()
        self.decisions = []
        self.suspended = True
        self.notify(f"[{self.name}] no simulation state for over {self.stale_limit():.2f} s; feedback suspended.")

    def status(self) -> str:
        """One line for a readout: what this arm's feedback is doing right now."""
        if self.fault is not None:
            return f"fault: {type(self.fault).__name__}"
        if not self.feedback:
            return "reading only (no motor is energized)"
        if self.suspended:
            return "suspended (waiting for the page)"
        if not all(gate.neutralized for gate in self.gates):
            return "waiting for alignment"
        active = [d for d in self.decisions if d.active]
        if not active:
            return "free"
        return "pushing back: " + ", ".join(f"{d.motor} {d.mismatch:+.2f}" for d in active)


# --------------------------------------------------------------- studio bridge
def jaw_fraction(gripper, joint_q) -> float:
    """How far a hand's *driven* finger joints are from open, 0 open .. 1 closed.

    The inverse of the map a grip command is written through
    (:meth:`~robosimstudio.solvers.Gripper.apply`): each driven joint's
    position along its own open-to-closed travel, averaged over the driven
    joints and clipped. On a mirrored pair with both fingers driven this is
    :meth:`~robosimstudio.solvers.Gripper.closed_fraction`; on a hand that
    drives one finger and carries the other it leaves the carried one out. See
    the module docstring for why that is what the grip slot compares.

    Args:
        gripper: A :class:`~robosimstudio.solvers.Gripper`.
        joint_q: The scene's joint coordinates, already on the host.
    """
    travelled = []
    for coord, open_q, closed_q in zip(gripper.coords, gripper.open_q, gripper.closed_q, strict=True):
        travel = float(closed_q) - float(open_q)
        if abs(travel) > 1.0e-9:
            travelled.append((float(joint_q[coord]) - float(open_q)) / travel)
    if not travelled:
        return 0.0
    return min(1.0, max(0.0, sum(travelled) / len(travelled)))


def preflight(env, layouts: Mapping[str, ArmLayout]) -> None:
    """:func:`~robosimstudio.robots.teleop.lerobot_controller.check_leaders`, for layouts.

    Raises:
        LeaderMismatch: Naming the arm and what the robot has instead.
    """
    check_leaders(env, {name: layout.calibration for name, layout in layouts.items()})


class StudioBridge:
    """The page's half: leader poses in as an action, measured joints back out.

    Everything here runs on the thread that steps the scene, hung off the
    page's ``each_step`` -- the only thread allowed to touch Warp.

    The page is an argument, not an import: anything with ``env`` (the live
    :class:`~robosimstudio.sim.Sim`), ``arms`` (per arm name, something with
    ``suspend()``), ``each_step(fn)`` and a ``quit_requested`` flag. The
    studio's ``StudioSession`` is one; a test passes four attributes.

    Args:
        session: The page.
        loops: One leader loop per driven arm, by the robot's own arm name.
        trace: Where publish timing goes, when somebody asked.
        clock: ``time.monotonic`` unless a test wants its own.
    """

    def __init__(
        self,
        session: Any,
        loops: Mapping[str, LeaderArmLoop],
        trace: Trace | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session = session
        self.loops = dict(loops)
        self.trace = trace
        self.clock = clock
        self.driven: list[str] = []
        self._env: Any = None
        self.fault: BaseException | None = None
        self.released = False
        self._applied: dict[str, list[float]] = {}
        """Per arm, the leader pose applied at the previous step: what the arm is chasing."""
        self._last_publish: float | None = None

    def pump(self) -> None:
        """One control step's exchange, in the order the page needs it."""
        if self.fault is not None or self.released:
            return
        for loop in self.loops.values():
            if loop.fault is not None:
                self.fault = loop.fault
                self.session.quit_requested = True
                return
        try:
            t0 = self.clock()
            self._pump()
            if self.trace is not None:
                gap = -1.0 if self._last_publish is None else t0 - self._last_publish
                self._last_publish = t0
                env = self.session.env
                Trace.add(
                    self.trace.publishes,
                    (t0 - self.trace.started, gap, int(getattr(env, "step_count", -1)), self.clock() - t0),
                    self.trace.limit,
                )
        except BaseException as exc:  # reported through the same teardown
            self.fault = exc
            self.session.quit_requested = True

    def _pump(self) -> None:
        env = self.session.env
        if env is None:
            return
        if env is not self._env:
            self._adopt(env)
        poses = {}
        for name, loop in self.loops.items():
            pose = loop.latest_leader()
            if pose is not None:
                poses[name] = pose
        apply_arm_actions(env, poses)
        # Read back *after* applying but *before* the step: joint_q still holds
        # the last completed step, so this is where the arm actually got to
        # against the target it was chasing. That lag is the whole signal.
        space = env.use_control("joint_position")
        joint_q = env.state.joint_q.numpy()
        at = self.clock()
        for index, name in enumerate(arm_slots(env)):
            loop = self.loops.get(name)
            if loop is None:
                continue
            ik, gripper = space.arms[index]
            measured = [float(joint_q[coord]) for coord in ik.arm_coords]
            measured.append(jaw_fraction(gripper, joint_q) if gripper is not None else 0.0)
            # joint_q is the state the *previous* step left, so what it was
            # chasing is the pose applied then -- not the one applied just now.
            loop.publish_follower(measured, at, self._applied.get(name))
        self._applied = {name: list(pose) for name, pose in poses.items()}

    def _adopt(self, env) -> None:
        """Take the driven arms off their gizmos, on this scene and each new one.

        Panels are rebuilt per scene load, so a page that swapped scenes would
        otherwise hand its arm back to a gizmo mid-session.

        The studio's own ``Leader arm`` panel is stood down here rather than
        left to race this: it opens a Feetech bus from the *page's* thread, and
        these ports already belong to the leader threads. It is taken out of
        the drain rather than out of the page, and its readout is made to say
        why -- a folder that silently stopped working would be a worse answer
        than one that explains itself.
        """
        self._env = env
        session = self.session
        panel = getattr(session, "leader_arm", None)
        if panel is not None:
            try:
                stand_down = getattr(panel, "stand_down", None)
                if stand_down is not None:
                    stand_down("driven by force feedback")
                else:
                    panel.close()
            except Exception:  # a dead readout must not take the session with it
                pass
            panels = getattr(session, "panels", None)
            if panels is not None and panel in panels:
                panels.remove(panel)
            session.leader_arm = None
        arms = getattr(session, "arms", None) or {}
        self.driven = [name for name in arm_slots(env) if name in self.loops and name in arms]
        for name in self.driven:
            arms[name].suspend()

    def release(self) -> None:
        """Stop driving: later calls to :meth:`pump` do nothing, and the gizmos get their arms back.

        A page keeps every ``each_step`` hook it was given, so a bridge whose
        leaders are gone has to stand down by itself -- otherwise it would go
        on applying the last pose a dead thread published. Main thread.
        """
        if self.released:
            return
        self.released = True
        arms = getattr(self.session, "arms", None) or {}
        for name in self.driven:
            resume = getattr(arms.get(name), "resume", None)
            if resume is not None:
                try:
                    resume()
                except Exception:  # the page may already be half torn down
                    pass
        self.driven = []

    def preflight(self) -> None:
        """:func:`preflight` for this bridge's own loops, against the page's scene."""
        env = self.session.env
        if env is not None:
            preflight(env, {name: loop.layout for name, loop in self.loops.items()})

    def status(self) -> dict[str, str]:
        """:meth:`LeaderArmLoop.status` per arm, for a readout on the page."""
        return {name: loop.status() for name, loop in self.loops.items()}


# -------------------------------------------------------------------- session
class Teleoperation:
    """Leaders driving a page, from the first torque-off to the last.

    A context manager, because the part that matters is the way out: every
    path from here -- the page closing, Ctrl-C, SIGTERM, a fault on a leader
    thread, an exception in the caller's own code -- goes through one teardown
    that stops each thread, takes torque off and verifies it, and puts the
    servos' EEPROM registers back::

        with Teleoperation(studio, leaders, feedback=True) as teleop:
            studio.run()
        raise SystemExit(teleop.exit_code)

    Entering checks the leaders against the robot **before** any bus is
    opened, connects each arm (torque off, verified, and with ``feedback`` the
    feedback gains installed), hooks :meth:`StudioBridge.pump` onto the page's
    ``each_step`` and starts one thread per arm. If any of that fails, whatever
    did open is closed again -- the page included -- before the error leaves:
    a :class:`LeaderMismatch`, a
    :class:`~robosimstudio.robots.teleop.lerobot_controller.LeaderUnavailable`
    or a :class:`ServoFault`, each a sentence a script can print.

    What this cannot cover is the process being killed outright (``kill -9``,
    a power cut to the computer): the servos then keep their last goal and
    torque limit until their own power is cut. Keep that switch in reach.

    Args:
        session: The page. See :class:`StudioBridge` for what it needs; it is
            also closed on the way out if it has a ``close()``.
        leaders: ``{arm name: LeaderPort}``, as
            :func:`~robosimstudio.robots.teleop.lerobot_controller.resolve_leaders`
            returns.
        feedback: ``False`` drives the simulation from the leaders and never
            energizes a motor or touches a servo's EEPROM. The run to take
            first with a new calibration.
        config: The signal and how it is rendered.
        rate_hz, stale_after_s, stale_factor, neutral_dwell_s: As on
            :class:`LeaderArmLoop`.
        trace: A path to write per-tick timing to on the way out, or ``""``.
        trace_limit: Samples kept per stream.
        connect: As on :class:`LeaderServos`.
        notify: Where operator messages go, faults included. Unstated, they
            are printed: status to stdout and faults to stderr.
        compare: As on :class:`LeaderArmLoop`: ``"applied"`` or ``"live"``.
    """

    def __init__(
        self,
        session: Any,
        leaders: Mapping[str, LeaderPort],
        *,
        feedback: bool = True,
        config: ComplianceConfig | None = None,
        rate_hz: float = 100.0,
        stale_after_s: float = 0.25,
        stale_factor: float = 3.0,
        neutral_dwell_s: float = 0.5,
        trace: str | Path = "",
        trace_limit: int = 60_000,
        connect: Callable[..., Any] | None = None,
        notify: Callable[[str], None] | None = None,
        compare: str = "applied",
    ) -> None:
        if not 0.0 < rate_hz <= 500.0:
            raise ValueError("the leader rate must be within (0, 500] Hz")
        if not stale_after_s > 0.0:
            raise ValueError("the stale floor must be positive")
        if compare not in COMPARISONS:
            raise ValueError(f"compare must be one of {COMPARISONS}, got {compare!r}")
        self.compare = compare
        self.session = session
        self.leaders = dict(leaders)
        self.feedback = bool(feedback)
        self.config = config if config is not None else ComplianceConfig()
        self.rate_hz = float(rate_hz)
        self.stale_after_s = float(stale_after_s)
        self.stale_factor = float(stale_factor)
        self.neutral_dwell_s = float(neutral_dwell_s)
        self.trace_path = Path(trace) if trace else None
        self.trace = Trace(limit=trace_limit) if trace else None
        self.notify = notify if notify is not None else _say
        self._warn = notify if notify is not None else _warn
        self._connect = connect
        self.servos: list[LeaderServos] = []
        self.loops: dict[str, LeaderArmLoop] = {}
        self.bridge: StudioBridge | None = None
        self.fatal: BaseException | None = None
        self.errors: list[str] = []
        self.exit_code = 0
        self._handlers: dict[int, Any] = {}

    # ------------------------------------------------------------------- in
    def __enter__(self) -> Teleoperation:
        layouts = {name: ArmLayout(leader.calibration) for name, leader in self.leaders.items()}
        self._catch_signals()
        try:
            env = getattr(self.session, "env", None)
            if env is not None:
                preflight(env, layouts)  # before any bus is opened
            for name, leader in self.leaders.items():
                servos = LeaderServos(leader.port, layouts[name], self.config, connect=self._connect)
                # Listed before it is connected: a connect that fails half way
                # has a bus open, and close() is what puts it back.
                self.servos.append(servos)
                servos.connect(configure=self.feedback)
                self.loops[name] = LeaderArmLoop(
                    name,
                    servos,
                    self.config,
                    rate_hz=self.rate_hz,
                    stale_after_s=self.stale_after_s,
                    stale_factor=self.stale_factor,
                    neutral_dwell_s=self.neutral_dwell_s,
                    feedback=self.feedback,
                    trace=self.trace,
                    notify=self.notify,
                    compare=self.compare,
                )
                state = "feedback gains installed" if self.feedback else "gains untouched (no feedback)"
                self.notify(
                    f"{name}: leader on {leader.port} (serial {leader.serial}), "
                    f"{len(servos.layout.motors)} motors, torque off, {state}"
                )
            self.bridge = StudioBridge(self.session, self.loops, self.trace)
            self.session.each_step(self.bridge.pump)
            for loop in self.loops.values():
                loop.start()
        except BaseException:
            self._teardown()
            self._restore_signals()
            raise
        if self.feedback:
            config = self.config
            scaled = (
                f"+{config.velocity_deadband:g}*speed capped {config.max_deadband:g}"
                if config.velocity_deadband > 0
                else ""
            )
            self.notify(
                f"Force feedback armed on {', '.join(self.loops)} at {self.rate_hz:g} Hz. "
                f"Torque stays off in free space (deadband={config.deadband}{scaled}, ramp={config.ramp_gain}, "
                f"cap={config.max_error}, P={config.p_coefficient}, D={config.d_coefficient}, "
                f"torque_limit={config.torque_limit})."
            )
        else:
            self.notify(f"Driving {', '.join(self.loops)} without feedback: no motor will be energized.")
        return self

    # ------------------------------------------------------------------ out
    def __exit__(self, exc_type, exc, tb) -> bool:
        interrupted = exc_type is not None and issubclass(exc_type, KeyboardInterrupt)
        if exc is not None and not interrupted:
            self.fatal = exc
        self._teardown()
        self._restore_signals()
        self._report(raised=exc is not None and not interrupted)
        # Ctrl-C is how a session is ended, not a failure; anything else the
        # caller's code raised is theirs to see, after the arm is slack.
        return interrupted

    def status(self) -> dict[str, str]:
        """One line per arm for a readout: free, pushing back on which motor, suspended."""
        return {name: loop.status() for name, loop in self.loops.items()}

    def _teardown(self) -> None:
        """Every arm, even if one throws: an arm left holding its feedback
        gains is exactly what this must not walk away from."""
        for loop in self.loops.values():
            try:
                loop.stop()
            except Exception as exc:
                self.errors.append(str(exc))
        if self.bridge is not None:
            self.bridge.release()
        for servos in self.servos:
            try:
                servos.close()
            except Exception as exc:
                self.errors.append(str(exc))
        close = getattr(self.session, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # a page's own run() closes it already on the normal path
                pass
        if self.trace is not None and self.trace_path is not None:
            # After every arm is slack: a trace is worth having even from a run
            # that faulted -- especially then -- but never at the cost of
            # delaying a torque-off, and its own failure is reported, not raised.
            try:
                self.notify(f"\n[trace]\n{self.trace.write(self.trace_path)}")
                self.notify(f"[trace] written to {self.trace_path}")
            except Exception as exc:
                self.errors.append(f"could not write trace to {self.trace_path}: {exc}")

    def _report(self, *, raised: bool) -> None:
        fatal = (
            self.fatal
            or (self.bridge.fault if self.bridge is not None else None)
            or next((loop.fault for loop in self.loops.values() if loop.fault), None)
        )
        self.fatal = fatal
        if fatal is None and not self.errors:
            self.notify("\nStopped; leader torque is off.")
            self.exit_code = 0
            return
        self.exit_code = 1
        if fatal is not None and not raised:
            if not isinstance(fatal, LeaderError):
                self._warn("".join(traceback.format_exception(type(fatal), fatal, fatal.__traceback__)).rstrip())
            self._warn(f"\nSAFETY STOP: {type(fatal).__name__}: {fatal}")
        for error in self.errors:
            self._warn(f"SHUTDOWN ERROR: {error}")
        self._warn("Closed after requesting torque-off on every configured motor.")

    # -------------------------------------------------------------- signals
    def _catch_signals(self) -> None:
        """Route SIGTERM and SIGHUP through the same teardown Ctrl-C takes.

        A terminated process otherwise dies between two ticks with whatever
        torque it had enabled still enabled. Only possible from the main
        thread, which is where a page's loop runs; elsewhere this is a no-op
        and the caller's own shutdown has to reach ``__exit__``.
        """
        if threading.current_thread() is not threading.main_thread():
            return

        def _interrupt(signum, frame):
            raise KeyboardInterrupt

        for name in ("SIGTERM", "SIGHUP"):
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            try:
                self._handlers[signum] = signal.signal(signum, _interrupt)
            except (ValueError, OSError):
                pass

    def _restore_signals(self) -> None:
        for signum, handler in self._handlers.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError, TypeError):
                pass
        self._handlers = {}
