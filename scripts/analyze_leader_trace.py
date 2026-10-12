# SPDX-License-Identifier: Apache-2.0
"""Read a force-feedback trace and say what stalled.

A trace is what ``examples/06_teleop/01_force_feedback.py --trace PATH`` (or
:class:`~robosimstudio.robots.teleop.force_feedback.Teleoperation` with
``trace=``) writes on the way out: per-tick timing for each leader thread, the
page's publish cadence, and the suspend and re-arm events between them.

Choppy force is a *timing* complaint, not a signal one: a correction computed
perfectly and delivered late feels worse than an approximate one delivered on
the clock. So this leads with the loop period -- what the operator's hand
actually experiences -- and then attributes every overrun to the stage that
caused it, because the three candidates want opposite fixes:

``render``
    Serial writes. Arming a joint costs four acknowledged transactions per
    motor (``Goal_Position``, ``Torque_Limit``, then ``enable_torque``, which
    is two writes in lerobot), so a tick that arms a whole arm pays ~28 round
    trips where a steady one pays a single unacknowledged broadcast. Fix by
    arming less often, not by arming faster.

``suspend``
    The staleness gate firing. One suspend is ~0.6 s of blocked thread -- the
    per-motor torque-off is paced by ``COMMAND_GAP_S`` and closed by a
    ``TORQUE_OFF_SETTLE_S`` sleep -- and then the neutral dwell before any
    joint may energize again. A handful of these in a session is the whole
    complaint on its own.

``read``
    The bus read itself. Measured at ~1.5 ms for seven motors, so this
    dominating means a bus problem, not a design one.

Gate chatter gets its own section because it is the usual cause of the first:
a mismatch sitting near the deadband arms and releases joints over and over,
and each flip is a burst of acknowledged writes inside a 10 ms budget.

Usage::

    uv run python scripts/analyze_leader_trace.py outputs/ff.json
    uv run python scripts/analyze_leader_trace.py outputs/ff.json --arm right
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PHASE = {0: "armed", 1: "suspended", 2: "no-feedback"}


def q(values: np.ndarray, p: float) -> float:
    return float(np.percentile(values, p)) if values.size else 0.0


def ms(values: np.ndarray, p: float) -> float:
    return 1e3 * q(values, p)


def bar(fraction: float, width: int = 28) -> str:
    filled = int(round(max(0.0, min(1.0, fraction)) * width))
    return "#" * filled + "." * (width - filled)


def report_arm(name: str, cols: dict, target_ms: float, joints: list[str] | None = None) -> None:
    t = np.asarray(cols["t"], dtype=float)
    period = np.asarray(cols["period"], dtype=float)
    read = np.asarray(cols["read_s"], dtype=float)
    render = np.asarray(cols["render_s"], dtype=float)
    age = np.asarray(cols["age"], dtype=float)
    phase = np.asarray(cols["phase"], dtype=int)
    n_arm = np.asarray(cols["n_arm"], dtype=int)
    n_rel = np.asarray(cols["n_release"], dtype=int)
    n_active = np.asarray(cols["n_active"], dtype=int)
    mism = np.asarray(cols["worst_mismatch"], dtype=float)

    # The first sample has no predecessor, so its period is 0 by construction.
    good = period > 0.0
    period = period[good]
    span = float(t[-1] - t[0]) if t.size > 1 else 0.0
    print(f"\n=== {name} ===")
    print(f"  {t.size} ticks over {span:.1f} s   target period {target_ms:.1f} ms")
    if not period.size:
        return

    late = period > 1.5 * target_ms * 1e-3
    print(
        f"\n  loop period   p50 {ms(period, 50):6.2f}  p95 {ms(period, 95):6.2f}  "
        f"p99 {ms(period, 99):6.2f}  max {ms(period, 100):8.2f} ms"
    )
    print(f"  ticks over 1.5x target: {late.sum()} of {period.size} ({100.0 * late.mean():.1f}%)")

    # Armed ticks only: on a suspended tick "render" is the verified torque-off,
    # which is a 0.6 s transition and would swamp every percentile here.
    armed = phase == 0
    print("\n  where an ARMED tick's time went (mean ms / share of the period):")
    total = float(np.mean(read[armed] + render[armed])) if armed.any() else 1e-12
    for label, arr in (("read (bus)", read[armed]), ("render (writes)", render[armed])):
        mean = float(np.mean(arr))
        print(f"    {label:16s} {1e3 * mean:7.3f}  {bar(mean / total)}  p99 {ms(arr, 99):7.2f}")

    down = render[phase == 1]
    if down.size:
        print(
            f"\n  SUSPEND cost  {down.size} tick(s) spent {1e3 * float(down.sum()):.0f} ms total "
            f"in verified torque-off  (worst {ms(down, 100):.0f} ms)"
        )
        print("    each one also re-arms behind the neutral dwell, so the arm is dead for longer than this")
    for label, value in (
        ("armed", phase == 0),
        ("suspended", phase == 1),
        ("no-feedback", phase == 2),
    ):
        share = float(np.mean(value))
        if share:
            print(f"  {label:14s} {100 * share:5.1f}% of ticks")

    # --- the usual culprit: gates flapping around the deadband -------------
    flips = int(n_arm.sum() + n_rel.sum())
    print(f"\n  torque transitions: {int(n_arm.sum())} arm / {int(n_rel.sum())} release")
    if span > 0:
        print(f"    = {flips / span:.1f} per second  ({'CHATTER' if flips / span > 2 else 'calm'})")
    busy = render[armed & ((n_arm + n_rel) > 0)]
    quiet = render[armed & ((n_arm + n_rel) == 0)]
    if busy.size and quiet.size:
        print(
            f"    render on a transition tick  p50 {ms(busy, 50):6.2f} ms   "
            f"on a steady tick  p50 {ms(quiet, 50):6.2f} ms"
        )
    if n_active.size:
        print(f"  joints energized: mean {n_active.mean():.2f}, max {int(n_active.max())}")

    # --- which slot, and what the live deadband did (traces v2 and later) ---
    slot = np.asarray(cols.get("worst_slot", []), dtype=float)
    wspeed = np.asarray(cols.get("worst_speed", []), dtype=float)
    wdb = np.asarray(cols.get("worst_deadband", []), dtype=float)
    names = joints or []
    if slot.size:
        live = armed & (slot >= 0)
        print("\n  which slot carried the worst mismatch (armed ticks):")
        for k in sorted({int(v) for v in slot[live]}):
            f = float((slot[live] == k).mean())
            label = names[k] if k < len(names) else f"slot {k}"
            tail = (
                "  <- the grip slot: a stroke fraction, judged under its own slope and ceiling"
                if (k == len(names) - 1 and len(names) > 1)
                else ""
            )
            print(f"    {label:14s} {100 * f:5.1f}%  {bar(f)}{tail}")
    if wdb.size and float(np.nanmax(wdb)) > 0.0:
        live = armed & (wdb > 0)
        print(
            f"\n  live deadband  p50 {q(wdb[live], 50):.4f}  p95 {q(wdb[live], 95):.4f} "
            f"(floor {float(np.min(wdb[live])):.4f})   velocity scaling is ON"
        )
        print(f"  slot speed     p50 {q(wspeed[live], 50):.3f}  p95 {q(wspeed[live], 95):.3f} units/s")
        over = float((np.asarray(cols["worst_mismatch"], dtype=float)[live] > wdb[live]).mean())
        print(f"  mismatch over the LIVE deadband: {100 * over:.1f}% of armed ticks")
    elif wdb.size:
        print("\n  velocity scaling is OFF in this run (velocity_deadband 0)")

    valid = age[age >= 0.0]
    if valid.size:
        print(
            f"\n  follower sample age  p50 {ms(valid, 50):6.2f}  p95 {ms(valid, 95):6.2f}  max {ms(valid, 100):8.2f} ms"
        )
    if mism.size:
        print(f"  worst |mismatch|     p50 {q(mism, 50):.4f}  p95 {q(mism, 95):.4f} rad")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--arm", default="", help="only this arm; empty is every arm in the file")
    ap.add_argument(
        "--rate", type=float, default=100.0, help="the leader rate the run used [Hz], for the target period"
    )
    args = ap.parse_args()

    doc = json.loads(args.trace.read_text())
    target_ms = 1e3 / args.rate
    for name, cols in doc.get("arms", {}).items():
        if args.arm and name != args.arm:
            continue
        report_arm(name, cols, target_ms, doc.get("joints", {}).get(name))

    pub = doc.get("publishes") or {}
    gaps = np.asarray([g for g in pub.get("gap", []) if g >= 0.0], dtype=float)
    if gaps.size:
        print("\n=== page ===")
        print(f"  {gaps.size} publishes")
        print(
            f"  publish gap   p50 {ms(gaps, 50):6.2f}  p95 {ms(gaps, 95):6.2f}  "
            f"p99 {ms(gaps, 99):6.2f}  max {ms(gaps, 100):8.2f} ms"
        )
        pump = np.asarray(pub.get("pump_s", []), dtype=float)
        if pump.size:
            print(f"  pump cost     p50 {ms(pump, 50):6.2f}  max {ms(pump, 100):7.2f} ms")

    events = doc.get("events") or []
    if events:
        print(f"\n=== events ({len(events)}) ===")
        counts: dict[tuple[str, str], int] = {}
        for e in events:
            key = (e["arm"], e["what"].split(":")[0])
            counts[key] = counts.get(key, 0) + 1
        for (arm, what), n in sorted(counts.items()):
            print(f"  {arm:10s} {what:12s} x{n}")
        print("\n  first few, with timestamps:")
        for e in events[:12]:
            print(f"    {e['t']:8.2f}s  {e['arm']:10s} {e['what']}")
    else:
        print("\n=== events ===\n  none -- no suspend, re-arm or bus glitch in this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
