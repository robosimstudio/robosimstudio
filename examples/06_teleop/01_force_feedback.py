# SPDX-License-Identifier: Apache-2.0
"""Force feedback

Feel what the simulated arm runs into, on the leader arm's own servos.

Where the simulated arm cannot reach where your hand is -- its side against the
welded panel, its hand pressed on the crate -- the gap is rendered as torque on
the leader. In free space no motor is energized. Take ``--no-feedback`` first:
it drives the simulation and never energizes a motor. ``--simulate`` swaps the
arm for a scripted hand whose sweep meets the panel; a table of registers
renders nothing you can feel.

**Key features:**

* :class:`~robosimstudio.robots.teleop.force_feedback.Teleoperation` owns the servos from the first torque-off to the last
* :meth:`~robosimstudio.robots.teleop.force_feedback.Teleoperation.status` says which motor is pushing back
* :class:`~robosimstudio.robots.teleop.force_feedback.ComplianceConfig` holds the deadband, ramp, cap and servo gains
* :func:`~robosimstudio.slab` welds something to push against; a welded cylinder does not stop the arm's capsules

Needs the ``[lerobot-feetech]`` extra and a calibrated leader arm.

With feedback on, the leader's servos are powered: keep clear of the arm and keep its power switch in reach.
"""

import argparse

import robosimstudio as rss
from robosimstudio.robots.teleop import force_feedback as ffb
from robosimstudio.robots.teleop import simulated
from robosimstudio.robots.teleop.follower import YAM_FOLLOWER, YAM_TELEOP_ARM, YAM_TELEOP_GRIP
from robosimstudio.robots.teleop.lerobot_controller import LeaderError, parse_leader_ports, resolve_leaders
from robosimstudio_studio import StudioConfig, StudioSession


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Feel what the simulated arm runs into, on the leader's own servos.")
    parser.add_argument("--leader", default="", help="<arm>:<port>[,...], e.g. right:/dev/ch340-<serial>")
    parser.add_argument(
        "--calibration",
        default="scripts/calibration/local/yam_leader.json",
        help="keyed by adapter serial; the default is where scripts/calibrate_leader_arms.py writes",
    )
    parser.add_argument("--no-feedback", action="store_true", help="never energize a motor: the run to take first")
    parser.add_argument("--simulate", action="store_true", help="no arm: a scripted hand over a table of registers")
    parser.add_argument("--trace", default="", help="write per-tick timing here on the way out")
    parser.add_argument("--port", type=int, default=8080, help="the page's port")
    parser.add_argument("--share", action="store_true", help="also request a public URL for the page")
    args = parser.parse_args(argv)

    if args.simulate:
        leaders = simulated.leaders(("left", "right"))
    else:
        leaders = resolve_leaders(parse_leader_ports(args.leader), args.calibration)

    # Rigid bodies on the arm's own solver. Under the default arrangement a free
    # box rests on a penalty contact, and one this heavy sinks into the bench.
    physics = rss.replace(rss.GENERIC, backends=(("*", rss.backend("mujoco")),))
    scene = rss.Scene(cell="bench_075", name="force_feedback", physics=physics)
    scene.add(rss.robot(YAM_FOLLOWER, arm=YAM_TELEOP_ARM, gripper=YAM_TELEOP_GRIP))
    # A panel welded to the bench beside the right hand: swing into it sideways.
    scene.add(rss.slab(name="panel", half=(0.10, 0.01, 0.22), at=(0.0, -0.17, 0.22)))
    # A 20 kg crate under the left hand: press down on it.
    scene.add(rss.box(name="crate", half=(0.08, 0.08, 0.06), at=(0.10, 0.31, 0.06), mass=20.0))
    sim = scene.build(control_mode="joint_position")

    connect = simulated.connector(simulated.homes(sim, leaders)) if args.simulate else None
    studio = StudioSession(sim, config=StudioConfig(panels="drive", render_every=1, port=args.port, share=args.share))

    # Every way out of this block -- the page closing, Ctrl-C, a fault -- takes
    # torque off and puts the servos' own gains back before anything else.
    with ffb.Teleoperation(studio, leaders, feedback=not args.no_feedback, trace=args.trace, connect=connect) as teleop:
        with studio.gui.add_folder("Force feedback"):
            rows = {name: studio.gui.add_text(name, initial_value="", disabled=True) for name in teleop.loops}

        def readout() -> None:
            for name, text in teleop.status().items():
                rows[name].value = text

        studio.each_step(readout)
        studio.run()
    return teleop.exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LeaderError as exc:
        raise SystemExit(str(exc)) from None
