# SPDX-License-Identifier: Apache-2.0
"""Wipe the bar

Two YAMs and a sponge at a chrome grab bar: drag the right wrist gizmo down to press, and along the bar to wipe the marker off.

The twin of a real cell; its scene, every distance in it measured off the clip,
is ``bar.py`` beside this file. The sponge is foam that SuperDex steps, and the
blue stripe wears off where the foam presses and slides, read off the engine's
own contact samples. There is no pick: ``take_hold`` poses both hands by IK
where the sponge was built, closes them and settles, and the page shows the
press, its peak pressure and a contact map. Needs the ``[superdex]`` extra
(Python 3.12).

**Key features:**

* :func:`robosimstudio.sponge` built where the hand will hold it, with ``quat=`` and ``settle_frames=0``
* :func:`robosimstudio.register_entity` for the welded bar and for a stripe that wears off
* ``sim.arm(name).solve_q_full`` and ``sim.gripper(name)`` to pose and close a hand before the page opens
* ``page=`` on :func:`robosimstudio.view` for a reading of your own, and ``studio.arms`` to hand the arms over
"""

import math
import time

import bar
import newton
import numpy as np

import robosimstudio as rss


def take_hold(sim) -> None:
    """The start state: the left hand steadying the far elbow, the sponge in the right."""
    # Both arms straight onto their holds by IK, and the targets with them.
    q = sim.state.joint_q.numpy().copy()
    for arm, (position, quat) in bar.holds(sim).items():
        ik = sim.arm(arm)
        solved = ik.solve_q_full(q, position, quat)
        miss, turn = ik.pose_errors(solved, position, quat)
        if miss > 0.005 or turn > math.radians(5.0):
            raise RuntimeError(f"the {arm} hand misses its hold by {miss * 1e3:.1f} mm, {math.degrees(turn):.1f} deg")
        q[ik.arm_coords] = solved[ik.arm_coords]
    for state in (sim.pipeline.state_0, sim.pipeline.state_1):
        state.joint_q.assign(q)
        state.joint_qd.zero_()
        newton.eval_fk(sim.model, state.joint_q, state.joint_qd, state)
    for arm in ("left", "right"):
        sim.arm(arm).hold_into(sim.state.joint_q, sim.control.joint_target_q)

    # Close both hands, ramped -- a step closed onto foam is a punch -- with the
    # slab of foam between the right jaws pinned still while they do.
    sponge = sim["sponge"]
    at = sponge.particle_q
    middle = np.abs(at[:, 0] - at[:, 0].mean()) < 0.0035
    pin = sponge.entity.bridge.pin_nodes("sponge", sponge.particles[middle], at[middle], stiffness=4000.0, damping=20.0)
    for i in range(1, 61):
        for arm, gap in bar.GRIPS.items():
            gripper = sim.gripper(arm)
            gripper.command = gripper.command_for_width(gap) * min(1.0, i / 40)
            gripper.apply(sim.control.joint_target_q)
        sim.step_teleop()
    sponge.entity.bridge.unpin(pin)

    # From here the latch carries the foam under the pads with the hand. A short settle.
    sim.post_step_hooks.append(bar.Latch(sim))
    for _ in range(30):
        sim.step_teleop()


def page(sim):
    """``page=``: the press and where it lands, and the hold taken again whenever the page resets."""
    stripe = sim["stripe"].entity

    def build(studio) -> None:
        def take() -> None:
            take_hold(sim)
            for arm in studio.arms.values():
                arm.resume()  # the gizmo onto the hand, and the grip it has onto the slider

        # The page resets the scene as it opens, and on Reset: the arms go home,
        # the hands open and the latch lets go. So the hold is taken again.
        take()

        def aim(client) -> None:  # from behind the arms, over their shoulders, at the bar
            client.camera.position, client.camera.look_at = bar.EYE, bar.LOOK

        studio.server.on_client_connect(aim)
        for client in studio.server.get_clients().values():
            aim(client)
        with studio.gui.add_folder("Sponge"):
            reading = studio.gui.add_markdown("")
            picture = studio.gui.add_image(bar.contact_map(stripe), format="png")
        last = {"step": sim.step_count, "drawn": 0.0}

        def each_step() -> None:
            if sim.step_count < last["step"]:
                take()
            last["step"] = sim.step_count
            if time.monotonic() - last["drawn"] > 0.3:
                last["drawn"] = time.monotonic()
                reading.content = f"**{stripe.press:.1f} N** · peak {stripe.peak / 1000:.1f} kPa"
                picture.image = bar.contact_map(stripe)

        studio.each_step(each_step)

    return build


def main():
    scene = bar.scene()  # two YAMs, the mat, the blocks, the bar, the sponge and the stripe
    sim = scene.build(control_mode="joint_position")
    take_hold(sim)

    rss.view(sim, page=page(sim))


if __name__ == "__main__":
    main()
