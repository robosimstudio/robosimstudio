# SPDX-License-Identifier: Apache-2.0
"""The simulated arm a YAM leader drives, and the drive that makes it worth driving.

A leader arm is a joint configuration read every tick, so the arm that follows
it is whichever shipped robot has the same joints: **``yam_bimanual``** -- two
arms of six revolute joints and a grip each, the layout a bimanual YAM leader's
calibration describes. It is used as it is registered. What changes for
teleoperation is only how hard its servos chase their targets, and that is an
override the scene applies rather than a second robot in the registry::

    import robosimstudio as rss
    from robosimstudio.robots.teleop.follower import YAM_FOLLOWER, YAM_TELEOP_ARM, YAM_TELEOP_GRIP

    scene = rss.Scene(cell="bench_075")
    scene.add(rss.robot(YAM_FOLLOWER, arm=YAM_TELEOP_ARM, gripper=YAM_TELEOP_GRIP))

An override, so the stock ``yam_bimanual`` is untouched: its own gains are the
hardware's actuator classes and other scenes are measured against them. The
same two specs fit ``yam_i2rt_arena_narrow``, which wears the same hand and
actuator classes on a different arm model; the measurements below were not
repeated there.
"""

from __future__ import annotations

from robosimstudio.specs import ActuationSpec

__all__ = ["YAM_FOLLOWER", "YAM_TELEOP_ARM", "YAM_TELEOP_GRIP"]

YAM_FOLLOWER = "yam_bimanual"
"""The registered robot a bimanual YAM leader drives."""

YAM_TELEOP_ARM = ActuationSpec(
    # A reference teleoperation rig's gains, joint for joint, because that
    # rig's force feedback is the one that feels right and the asset's own
    # gains did not.
    #
    #     the reference's model    joint1-5 kp=4000  joint6 kp=3000  kv=10
    #     this asset's MJCF        joint1-3 kp=80    joint4-6 kp=30  kv=6/2
    #
    # robosimstudio.robots.teleop.force_feedback renders ``follower - leader``
    # as contact force, so any lag between the simulated arm and its target is
    # felt as something pushing back. At the asset's own gains that lag *is*
    # the signal: measured on this asset (in the narrow-shoulder cell it was
    # tuned in -- the same links, joints, actuator classes and hand) the arm
    # was a near-pure 92 ms delay line, so a hand moving at 0.8 rad/s through
    # empty space sat over the 0.04 rad deadband 65% of the time and a brisk
    # 1.6 rad/s saturated the cap 56% of the time, with nothing in contact in
    # either measurement.
    #
    # Measured at the leader's own rendering point, paced 60 Hz, hand 0.25 rad
    # @ 0.5 Hz (0.79 rad/s peak):
    #
    #     gains           tau     free-space p95   over deadband
    #     80/30 (MJCF)     92 ms      0.0722 rad      64.8%
    #     800/300          34 ms      0.0300           0.0%
    #     2000/750         26 ms      0.0247           0.0%
    #     4000/3000        18 ms      0.0194           0.0%   <- here, = the reference
    #
    # Repeated for this file on ``yam_bimanual`` in a bare ``bench_075`` scene,
    # the same hand as a synthetic one stepped in a tight loop (no serial
    # thread, no page), so the taus are a little shorter and the shape is the
    # same:
    #
    #     80/30 (MJCF)     89.5 ms    0.0695 rad      61.3%
    #     800/300          32.0 ms    0.0334           0.0%
    #     2000/750         18.0 ms    0.0176           0.0%
    #     4000/3000        14.0 ms    0.0129           0.0%
    #
    # The same swap run inside the *reference's own loop*, changing nothing but
    # the drive parameters, moves it 4.0 ms <-> 80 ms, so this is the whole of
    # the difference between the two rigs' feel. What stiffness cannot buy back
    # is the residual ~14 ms the page costs on top (StudioBridge reads joint_q
    # *before* the step, and the leader thread takes its sample from a slot the
    # page refreshes at 58 Hz); the reference reads its state after the step in
    # the same single-threaded iteration and pays neither.
    #
    # **armature, damping and frictionloss are deliberately NOT matched.** The
    # reference model carries none of them; this one carries measured rotor
    # inertia (0.032 / 0.0018, and 0.1 on the hand). Dropping them to match
    # would buy nothing and cost stability: the same measurement ladder put
    # every one of this asset's armature, joint damping and forcerange values
    # on top of kp=4000 and still measured tau=4.0 ms with 0% deadband
    # crossings, so they are free. ActuationSpec.armature=0.0 means "keep the
    # import's own", which is what leaves them in place.
    #
    # ``unlimited`` matches the same reference, whose six arm joints are all
    # unlimited, and it is the half of this that fixed a measured fault rather
    # than a felt one. A recorded session showed one wrist joint holding a
    # 0.287 rad mismatch -- 16.5 degrees -- *while stationary*, which is this
    # asset's +-90 degree range clamping a target the leader arm's own travel
    # puts outside it. Every other joint measured 0.0000 at rest. A clamp does
    # not decay, so the leader rendered that joint as permanently in contact
    # with nothing, and it accounted for 16.5% of all the torque transitions in
    # the run. No gain, deadband or calibration removes a hard stop.
    #
    # The authority on where a leader-driven joint may go is the physical arm
    # in the operator's hand, so for this pairing the model's own range is the
    # thing that is wrong. See ActuationSpec.unlimited for what it costs: a
    # joint at its limit stops being a force the operator can feel, and
    # IK-driven front ends can reach poses the hardware cannot.
    target_ke=4000.0,
    target_kd=10.0,
    target_ke_per_dof=(4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 3000.0),
    target_kd_per_dof=(10.0, 10.0, 10.0, 10.0, 10.0, 10.0),
    unlimited=True,
)
"""Arm drive gains and range for a leader-driven YAM, matched to a reference teleoperation rig."""

YAM_TELEOP_GRIP = ActuationSpec(
    # The hand, matched to the same reference (its two fingers are kp=3000).
    #
    # Stiffening the arm and leaving the hand alone is what a first pass at this
    # did, and it is worse than doing neither: the MJCF's finger class is
    # kp=200, so against arm joints at 4000 the grip slot becomes the softest
    # thing on the robot by a factor of twenty. That matters here more than the
    # ratio suggests, because the grip slot is a *feedback slot* --
    # Gripper.closed_fraction is the seventh entry the leader compares against
    # its own hand -- and it is compared under the same 0.04 floor as the arm
    # joints despite being a fraction of a stroke rather than radians. A hand
    # that lags is therefore read as a hand meeting resistance.
    #
    # armature stays the import's 0.1, and that is load bearing rather than
    # incidental: this hand is one driven finger with the other carried by an
    # <equality>, and ActuationSpec.armature documents what happens to a geared
    # linkage driven hard with too little rotor inertia -- the drive overpowers
    # the constraint and the pads end up tens of degrees off parallel. Raising
    # kp fifteen-fold is exactly the direction that makes that worse, so the
    # armature the asset already measured is kept.
    #
    # ``unlimited`` is NOT set here, and that is the whole of what keeps this
    # hand a hand: the finger slides' ranges are what the jaws close *onto*, and
    # the <equality> that carries the second finger is written against them.
    # The arm is unlimited because a leader's travel is the authority on where
    # an arm joint may go; nothing about that argument applies to a jaw whose
    # stops are the mechanism.
    #
    # The joint physics stays the asset's own -- armature 0.1, frictionloss
    # 0.1 -- and only the drive gains are matched. Matching the reference's
    # zeros there was tried and reverted: it made the hand jitter, and the
    # measurement says which term was doing the work.
    #
    # Counting sign changes in the driven finger's velocity over a step, at
    # kp=3000 (ringing is what an operator feels as jitter, and it is invisible
    # to a lag or linkage measure -- both of which called the zeros an
    # improvement)::
    #
    #     armature  damping  frictionloss | velocity sign flips
    #       0.0       0.0        0.0      |  45     <- the jitter
    #       0.1       0.0        0.0      |  47     armature does not help
    #       0.0       0.5        0.0      |  55     viscous damping does not help
    #       0.0       0.0        0.1      |  11     <- frictionloss is the term
    #       0.1       0.5        0.1      |   9     the asset's own
    #
    # Coulomb friction has a dead zone and a viscous term does not: it is what
    # holds a light finger still against a stiff servo instead of letting it
    # hunt around the setpoint. Raising kp fifteen-fold made that matter more,
    # not less, so the one term that was arresting it is the one term not to
    # remove.
    #
    # So why does the reference not need it? Two candidate answers, both
    # measured and both wrong:
    #
    # *Its mechanism.* It drives its two fingers with an actuator each, where
    # this hand drives one finger and carries the other on an <equality>, and a
    # constraint hunting behind a stiff drive would look exactly like this. But
    # counting the flips on the driven finger and on the carried one separately
    # gives the same number in every configuration -- 62/61 zeroed, 9/9 as
    # shipped -- so the follower tracks and the ringing is in the servo.
    #
    # *Its timestep.* The reference integrates at 0.0005 s where these
    # measurements ran sim_dt 0.00104 (60 Hz, 16 substeps -- the ``GENERIC``
    # arrangement's rate), and an undamped stiff servo stable at one step is
    # not automatically stable at twice it. Re-running at 34 substeps --
    # 0.00049 s, finer than the reference's -- left the zeroed count at 66
    # against 60, marginally worse rather than better, while the shipped
    # physics stayed at 9 in both.
    #
    # What is left is the finger itself: 0.118 kg against the reference's
    # 0.0710, on a different collider set, at a contact stiffness this asset
    # sets deliberately (solref 0.004 1 against that model's 0.01 2). Those are
    # the gripper contacts, which are kept on purpose. Neither the reference's
    # +-35 N force range nor a critically damped kv changes the count
    # (62 -> 62 -> 65).
    #
    # kd is the reference's 10 rather than the asset's 14. At kp=3000 the pair
    # is far below critical damping either way, and 14 measured marginally
    # better on the ring count (9 against 11) -- two flips, which is not a
    # reason to keep this hand's servo different from the one it is matched to.
    #
    # ``effort_limit`` is the reference's force range on its fingers. It is a
    # fidelity bound, not a stability one -- it measured no change to the ring
    # count at either kd (11 -> 11, 9 -> 9) -- but at kp=3000 a full-stroke
    # error commands 3000 * 0.0475 = 142 N, four times what that rig permits
    # its own jaws, and nothing else here caps it.
    target_ke=3000.0,
    target_kd=10.0,
    effort_limit=35.0,
)
"""Hand drive gains for a leader-driven YAM. The grip slot is a feedback channel, so its lag is felt too."""
