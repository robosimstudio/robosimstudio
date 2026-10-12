# SPDX-License-Identifier: Apache-2.0
"""Which leader joints read backwards. Read-only: no torque, no writes, no EEPROM.

A sign is the one calibration term that cannot be checked by arithmetic. Scale
and zero are answerable from the file -- a direct-drive STS3215 is 4096/(2*pi)
counts per radian and a zero is a number somebody measured -- but which way a
joint counts depends on how its horn was pressed on, and the only authority is
the arm on the desk.

Comparing two *models* does not settle it either: base yaw and wrist roll are
invisible to any measure of the arm's own shape. Rotating the whole arm about
its base moves the tool without changing one link angle, so a base-yaw sign
flip is undetectable from kinematics alone and has to be read off the hardware.

So: move one joint at a time and watch its line. ``delta`` is signed, against
where that joint sat when this started:

    a joint you push AWAY from its zero should read a delta whose sign
    matches the direction the simulated arm moves on the page

Run it beside ``examples/06_teleop/00_leader_arm.py``, which drives the scene
from the same leader without energizing anything, and compare. A joint whose
number climbs while the page's arm goes the other way has the wrong ``sign`` in
the calibration; flip that joint's ``sign`` and nothing else. The two cannot
share one port at the same time, so run them in turn.

Usage::

    uv run python scripts/check_leader_signs.py --leader right:/dev/ch340-<serial>
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from robosimstudio.robots.teleop.lerobot_controller import (
    LeaderError,
    open_leader,
    parse_leader_ports,
    resolve_leaders,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--leader", required=True, help="<arm>:<port>, e.g. right:/dev/ch340-<serial>")
    ap.add_argument("--calibration", default="scripts/calibration/yam_leader.json", help="keyed by adapter serial")
    ap.add_argument("--rate", type=float, default=10.0, help="refreshes per second")
    args = ap.parse_args(argv)

    leaders = resolve_leaders(parse_leader_ports(args.leader), args.calibration)
    if len(leaders) != 1:
        raise SystemExit("--leader names one arm here: watch one bus at a time")
    ((name, leader),) = leaders.items()
    arm = leader.calibration
    motors = [j.name for j in arm.joints] + ([arm.gripper.name] if arm.gripper else [])

    controller = open_leader(leader)
    print(f"\n{name}: {leader.port} (serial {leader.serial}) -- READ ONLY, nothing is written\n")
    print("Move ONE joint at a time and watch its delta. Ctrl-C to stop.\n")
    try:
        # Torque is never enabled here, so the arm is as loose as it was.
        start = dict(controller.bus.sync_read("Present_Position", motors, normalize=False, num_retry=2))
        while True:
            ticks = controller.bus.sync_read("Present_Position", motors, normalize=False, num_retry=2)
            lines = []
            for j in arm.joints:
                now = j.to_radians(ticks[j.name])
                was = j.to_radians(start[j.name])
                d = math.degrees(now - was)
                mark = "  <-- moving" if abs(d) > 2.0 else ""
                lines.append(
                    f"  {j.name:12s} sign={j.sign:+.0f}  raw {ticks[j.name]:7.0f}  "
                    f"angle {math.degrees(now):+8.2f} deg  delta {d:+8.2f}{mark}"
                )
            if arm.gripper is not None:
                g = arm.gripper
                cmd = g.to_command(ticks[g.name])
                was = g.to_command(start[g.name])
                mark = "  <-- moving" if abs(cmd - was) > 0.03 else ""
                lines.append(
                    f"  {g.name:12s} (0=open..1=closed)  raw {ticks[g.name]:7.0f}  "
                    f"cmd {cmd:5.3f}  delta {cmd - was:+6.3f}{mark}"
                )
            sys.stdout.write("\033[H\033[J" + "\n".join(lines) + "\n")
            sys.stdout.flush()
            time.sleep(1.0 / max(1.0, args.rate))
    except KeyboardInterrupt:
        print("\nstopped; nothing was written to the arm")
    finally:
        controller.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LeaderError as exc:
        raise SystemExit(str(exc)) from None
