# SPDX-License-Identifier: Apache-2.0
"""Leader arm

Drive the simulated YAM from a real leader arm, read only.

Each control step the script reads every connected leader, converts servo ticks
to joint angles through a calibration file, and applies them as one
``joint_position`` action. Nothing is written to a servo, so nothing can push
back on your hand. ``--simulate`` swaps the arm for a scripted hand that sweeps
both arms and works the jaws, over a table of registers instead of a bus.

**Key features:**

* :func:`~robosimstudio.robots.teleop.lerobot_controller.resolve_leaders` looks each calibration up by its adapter's serial
* :meth:`~robosimstudio.robots.teleop.lerobot_controller.LerobotFeetechController.read_action` reads one arm as ``[q_0 .. q_5, grip]``
* :func:`~robosimstudio.robots.teleop.lerobot_controller.apply_arm_actions` drives only the arms that have a leader
* :data:`~robosimstudio.robots.teleop.follower.YAM_TELEOP_ARM` gives the follower servo gains that keep up with a hand

Needs the ``[lerobot-feetech]`` extra and a calibrated leader arm.
"""

import argparse

import robosimstudio as rss
from robosimstudio.robots.teleop import simulated
from robosimstudio.robots.teleop.follower import YAM_FOLLOWER, YAM_TELEOP_ARM, YAM_TELEOP_GRIP
from robosimstudio.robots.teleop.lerobot_controller import (
    LeaderError,
    apply_arm_actions,
    check_leaders,
    open_leader,
    parse_leader_ports,
    resolve_leaders,
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Drive the simulated YAM from a real leader arm, read only.")
    parser.add_argument("--leader", default="", help="<arm>:<port>[,...], e.g. right:/dev/ch340-<serial>")
    parser.add_argument("--calibration", default="scripts/calibration/yam_leader.json", help="keyed by adapter serial")
    parser.add_argument("--simulate", action="store_true", help="no arm: a scripted hand over a table of registers")
    parser.add_argument("--port", type=int, default=8080, help="the page's port")
    parser.add_argument("--share", action="store_true", help="also request a public URL for the page")
    args = parser.parse_args(argv)

    if args.simulate:
        leaders = simulated.leaders(("left", "right"))
    else:
        leaders = resolve_leaders(parse_leader_ports(args.leader), args.calibration)

    scene = rss.Scene(cell="bench_075", name="leader_arm")
    scene.add(rss.robot(YAM_FOLLOWER, arm=YAM_TELEOP_ARM, gripper=YAM_TELEOP_GRIP))
    scene.add(rss.box(name="cube", half=(0.025, 0.025, 0.025), at=(0.10, -0.25, 0.025)))
    sim = scene.build(control_mode="joint_position")

    check_leaders(sim, leaders)
    connect = simulated.connector(simulated.homes(sim, leaders)) if args.simulate else None
    controllers = {name: open_leader(leader, connect) for name, leader in leaders.items()}

    def page(studio) -> None:
        for name in controllers:
            studio.arms[name].suspend()  # the leader has this arm now, so its gizmo is hidden

        def drive() -> None:
            apply_arm_actions(studio.env, {name: c.read_action() for name, c in controllers.items()})

        studio.each_step(drive)

    try:
        rss.view(sim, page=page, port=args.port, share=args.share)
    except KeyboardInterrupt:
        pass
    finally:
        for controller in controllers.values():
            controller.close()


if __name__ == "__main__":
    try:
        main()
    except LeaderError as exc:
        raise SystemExit(str(exc)) from None
