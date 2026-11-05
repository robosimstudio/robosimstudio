# SPDX-License-Identifier: Apache-2.0
"""The rope examples' memory and twist, done by the robot's own hands through the arm panel.

``examples/03_materials/01_rope_memory`` says the right hand can carry a cable's end
round a post and leave it wrapped; ``03_materials/02_twist`` says the two hands can turn two
hanging tips and that, let go, the stiff cable unwinds while the soft one keeps
the turn. Both pages also have a graphics path (a click handle, a slider's
kinematic hold) that would pass the same numbers without a robot, so this drives
neither: it moves the arm panel's cursor the way the gizmo's arrows and ring
write it, asks for a grip the way **Close** and **Open** do, and lets
:meth:`~robosimstudio_studio.session.StudioSession.tick` run IK and the servo
(:mod:`tests.fixtures.grasp`). Nothing latches or pins a capsule
(``allow_grasp_latch=False``): what moves, the pads moved by contact.

The memory has a control. A wrap round a post can be held by the post or by the
bench instead of by the cable, so the same drive runs on the same cable built at
``dahl_eps_max=0``: the plastic cable must stay wrapped where the elastic one
springs back straight. Measured here on GPU 2 (RTX PRO 6000, Newton 1.6):
straightness 0.561 (plastic) and 0.521 (elastic) while held, 0.694 and 0.992
120 steps after the hand lifted away (0.687 and 0.998 after 480, ``mem_drive``
probe). With friction left on the bench the control failed -- the elastic cable
stayed wrapped at 0.736 and 0.776 for 0.24 and 0.024 combined -- which is
why the example zeroes the bench's and the posts' friction.

The twist is a turn about the vertical, not a roll of the wrist joint: the YAM's
tool reaches no more than about 45 degrees above level (multi-start
``solve_q_full`` over the bench), so a hand cannot point up a hanging cable, and
its wrist roll's axis lies along a level tool, across the cable, where a roll
swings the tip instead of twisting it. A level hand with its jaws across the tip
turned about the vertical twists it; the IK turns the arm's wrist-yaw and base
joints to do it, 120 degrees from 30 degrees out. Measured on the YAM's real
clamp (50 N a jaw, 2026-10-26): of the hand's 0.322 turns the soft cable takes
0.317 (0.113 in its middle third) and the stiff one 0.319 (0.113): the clamp
holds the stiff cable's twist moment too. The hand's old spring (kp 200 N/m,
about 0.6 N a jaw on this 12 mm tip) did not -- the stiff cable followed it to
53-64 degrees and slipped, keeping 0.156 turns, about 0.015 N*m, whatever the
pads' friction (mu 1000 kept as much), so what held the twist was the squeeze. Opened, the stiff reads 0.0008 within 30
steps and the soft 0.106 after 60.
"""

from __future__ import annotations

import time
import unittest
from typing import ClassVar
from unittest import mock

import numpy as np

from tests.fixtures.examples import _run
from tests.fixtures.grasp import go, open_session, tick

TIER = "medium"
"""Three builds of two rope scenes on one GPU, each driven through its page for 15-20 s of scene time."""

PORTS = {"plastic": 9460, "elastic": 9461, "twist": 9462}

BENCH = np.array([0.6, 0.0, 0.75])
"""``bench_075``'s surface frame in world [m]."""
TOOL_DOWN = np.array([0.0, 1.0, 0.0, 0.0])
"""The YAM's tool pointing down, jaws closing along world y (xyzw)."""
LEVEL = np.array([0.0, 0.70710678, 0.0, 0.70710678])
"""The YAM's tool level, pointing along world +x, jaws closing along world y (xyzw): its home orientation."""
POST = BENCH[:2] + np.array([0.0, -0.086])
"""The memory example's post, in world [m]."""
WRAP = np.radians(120.0)
"""How far round the post the hand carries the cable's end, turning with it [rad]."""
CARRY = 0.05
"""The carry's speed along its arc [m/s]."""
TURN = np.radians(120.0)
"""How far each hand turns its tip [rad], from 30 degrees out to 90 degrees in: the YAM's reach at the tips."""


def _rotz(angle: float) -> np.ndarray:
    return np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)])


def _straightness(sim, cable) -> float:
    from robosimstudio_materials.rod.measures import straightness

    return straightness(sim.state.body_q.numpy()[cable.bodies, :3])


def _ms_per_step(sim) -> float:
    sim.run(10).sync()
    t = time.perf_counter()
    sim.run(60).sync()
    return 1e3 * (time.perf_counter() - t) / 60


def _wrap(session, sim, cable) -> dict:
    """Down onto the capsule under the hand, Close, up 30 mm, round the post turning, down, Open, up, and wait."""
    from robosimstudio.core.math import quat_multiply

    arm = session.arms["right"]
    tick(session, 60)
    out = {"rest": _straightness(sim, cable)}
    lying = sim.state.body_q.numpy()[cable.bodies, :3].astype(np.float64)
    home = arm.ik.ee_pose(sim.state)[0]
    k = int(np.argmin(np.linalg.norm(lying[:, :2] - home[:2], axis=1)))
    grip_z = lying[k, 2] + 0.010  # the fingertips 7 mm over the bench, the pads across the capsule
    out["miss"] = go(session, "right", np.array([lying[k, 0], lying[k, 1], grip_z]), speed=0.10)
    arm.grip_request = 1.0
    tick(session, 60)
    out["width"] = float(sim.gripper("right").width(sim.state))
    go(session, "right", np.array([lying[k, 0], lying[k, 1], grip_z + 0.03]), speed=0.10)
    start = lying[k, :2] - POST
    shrink = WRAP * (0.02 + 0.011)  # the length the wrap lays on the post (its radius plus the cable's)
    steps = round(np.linalg.norm(start) * WRAP / CARRY * 60)
    for i in range(1, steps + 1):
        a = WRAP * i / steps
        c, s = np.cos(a), np.sin(a)
        v = np.array([c * start[0] - s * start[1], s * start[0] + c * start[1]])
        v *= 1.0 - shrink / np.linalg.norm(start) * i / steps
        arm.cursor_pos = np.array([POST[0] + v[0], POST[1] + v[1], grip_z + 0.03])
        arm.cursor_quat, arm.dragging = quat_multiply(_rotz(a), TOOL_DOWN), i < steps
        tick(session, 1)
    arm.dragging = False
    tick(session, 30)
    hand = arm.ik.ee_pose(sim.state)[0]
    out["carried"] = float(np.linalg.norm(sim.state.body_q.numpy()[cable.bodies[k], :3] - hand))
    out["held"] = _straightness(sim, cable)
    go(session, "right", np.array([hand[0], hand[1], grip_z]), speed=0.05)
    arm.grip_request = 0.0
    tick(session, 30)
    go(session, "right", np.array([hand[0], hand[1], grip_z + 0.09]), speed=0.10)
    tick(session, 120)
    out["released"] = _straightness(sim, cable)
    out["finite"] = bool(sim.state_is_finite())
    return out


class _Memory:
    """The memory example built, its page opened, and the wrap driven; the subclasses say which cable."""

    KIND = ""

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        real = rss.rod

        def rod(**kwargs):
            if cls.KIND == "elastic":
                kwargs["dahl_eps_max"] = 0.0
            return real(**kwargs)

        with mock.patch.object(rss, "rod", rod):
            cls.sim, kwargs = _run("03_materials/01_rope_memory.py")
        cls.ms_per_step = _ms_per_step(cls.sim)
        cls.session = open_session(cls.sim, PORTS[cls.KIND], kwargs.get("page"))
        cls.cable = cls.sim["cable"]
        cls.result = _wrap(cls.session, cls.sim, cls.cable)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_the_cable_is_the_kind_it_says(self) -> None:
        spec = self.cable.entity.spec
        self.assertEqual(2.0 if self.KIND == "plastic" else 0.0, spec.dahl_eps_max)
        self.assertFalse(spec.allow_grasp_latch, "a latch would carry the cable for the pads")

    def test_the_hand_closes_on_the_cable_and_carries_it(self) -> None:
        self.assertGreater(self.result["rest"], 0.99, "the cable lies straight before the hand comes")
        self.assertLess(self.result["miss"], 0.008, "the hand got down to the cable")
        self.assertAlmostEqual(0.022, self.result["width"], delta=0.003, msg="the jaws are on the 22 mm cable")
        self.assertLess(self.result["carried"], 0.02, "the capsule stayed in the jaws round the post")
        self.assertLess(self.result["held"], 0.65, f"the wrap, held: straightness {self.result['held']:.3f}")
        self.assertTrue(self.result["finite"])
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)

    def test_a_control_step_is_inside_its_budget(self) -> None:
        self.assertLess(self.ms_per_step, 60.0, f"{self.ms_per_step:.1f} ms per control step")


class TestThePlasticCableStaysWrapped(_Memory, unittest.TestCase):
    KIND = "plastic"

    def test_it_keeps_the_bend(self) -> None:
        self.assertLess(self.result["released"], 0.8, f"straightness {self.result['released']:.3f} after release")


class TestTheElasticCableSpringsBack(_Memory, unittest.TestCase):
    """The control: the same cable, drive and post, built at ``dahl_eps_max=0``."""

    KIND = "elastic"

    def test_it_springs_back_straight(self) -> None:
        self.assertGreater(self.result["released"], 0.95, f"straightness {self.result['released']:.3f} after release")


class TestTheHandsTwistTheCables(unittest.TestCase):
    """Both hands onto their tips, Close, a third of a turn each, Open: the turn enters, then only the stiff lets go."""

    ARMS: ClassVar[dict[str, str]] = {"right": "stiff_twist", "left": "soft_twist"}
    SIGN: ClassVar[dict[str, float]] = {"right": 1.0, "left": -1.0}
    """Which way each hand turns: inward, the way each arm has reach."""

    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio.core.math import quat_multiply, quat_to_mat
        from robosimstudio_materials.rod.measures import twist_turns

        cls.sim, kwargs = _run("03_materials/02_twist.py")
        cls.ms_per_step = _ms_per_step(cls.sim)
        sim = cls.sim
        session = cls.session = open_session(sim, PORTS["twist"], kwargs.get("page"))
        panels = {a: session.arms[a] for a in cls.ARMS}
        cables = {a: sim[n] for a, n in cls.ARMS.items()}

        def body_q():
            return sim.state.body_q.numpy()

        def yaw(a):
            axis = quat_to_mat(panels[a].ik.ee_pose(sim.state)[1])[:, 2]
            return float(np.arctan2(axis[1], axis[0]))

        def twist(a, every=False):
            bodies = cables[a].bodies
            return twist_turns(body_q()[bodies if every else bodies[8:17], 3:7])

        tick(session, 60)
        cls.rest = {a: twist(a) for a in cls.ARMS}
        cls.start_yaw = {a: yaw(a) for a in cls.ARMS}
        starts = {a: np.array(panels[a].cursor_pos) for a in cls.ARMS}
        axes = {a: quat_to_mat(panels[a].ik.ee_pose(sim.state)[1])[:, 2] for a in cls.ARMS}
        ahead = {a: body_q()[cables[a].bodies[-1], :3] - starts[a] for a in cls.ARMS}
        cls.to_tip = {a: float(np.dot(ahead[a], axes[a])) for a in cls.ARMS}
        cls.off_axis = {a: float(np.linalg.norm(ahead[a] - cls.to_tip[a] * axes[a])) for a in cls.ARMS}
        for k in range(1, 61):  # each cursor along its tool axis onto the tip, as that arrow drags it
            for a, p in panels.items():
                p.cursor_pos, p.dragging = starts[a] + axes[a] * cls.to_tip[a] * k / 60, k < 60
            tick(session, 1)
        for p in panels.values():
            p.dragging, p.grip_request = False, 1.0
        tick(session, 60)
        cls.width = {a: float(sim.gripper(a).width(sim.state)) for a in cls.ARMS}
        grasped = {a: yaw(a) for a in cls.ARMS}
        steps = round(np.degrees(TURN))  # a degree a step, as the ring turns
        t = time.perf_counter()
        for k in range(1, steps + 1):
            for a, p in panels.items():
                turned = cls.start_yaw[a] + cls.SIGN[a] * TURN * k / steps
                p.cursor_quat, p.dragging = quat_multiply(_rotz(turned), LEVEL), k < steps
            tick(session, 1)
        cls.ms_turning = 1e3 * (time.perf_counter() - t) / steps
        for p in panels.values():
            p.dragging = False
        tick(session, 60)
        cls.turned = {a: (yaw(a) - grasped[a]) / (2 * np.pi) for a in cls.ARMS}
        # Right-handed about the capsule's axis, which points down: a turn of +z reads negative.
        cls.taken = {a: -twist(a, every=True) for a in cls.ARMS}
        cls.held = {a: -twist(a) for a in cls.ARMS}
        for p in panels.values():
            p.grip_request = 0.0
        cls.released, done = {}, 0
        for at in (5, 15, 30, 60):
            if done == 15:  # the jaws are open: back each hand off its tip along the way it now points
                for p in panels.values():
                    p.cursor_pos = np.array(p.cursor_pos) - quat_to_mat(p.ik.ee_pose(sim.state)[1])[:, 2] * 0.06
            tick(session, at - done)
            done = at
            cls.released[at] = {a: -twist(a) for a in cls.ARMS}
        cls.finite = bool(sim.state_is_finite())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.session.teardown()

    def test_both_hang_untwisted_with_the_hands_off_them(self) -> None:
        for a, turns in self.rest.items():
            self.assertLess(abs(turns), 0.01, a)
            self.assertAlmostEqual(0.06, self.to_tip[a], delta=0.01, msg=f"{a}: the hand starts 60 mm short of its tip")
            self.assertLess(self.off_axis[a], 0.005, f"{a}: and pointing at it")

    def test_the_jaws_close_on_the_tips(self) -> None:
        for a, width in self.width.items():
            self.assertAlmostEqual(0.012, width, delta=0.002, msg=f"{a}: the jaws on the 12 mm capsule")

    def test_each_hand_turns_a_third_of_a_turn(self) -> None:
        for a, turns in self.turned.items():
            self.assertAlmostEqual(self.SIGN[a] / 3, turns, delta=0.02, msg=a)

    def test_the_soft_cable_takes_the_whole_turn(self) -> None:
        self.assertGreater(self.SIGN["left"] * self.taken["left"], 0.28, f"taken {self.taken}")
        self.assertGreater(self.SIGN["left"] * self.held["left"], 0.09, f"mid-cable {self.held}")

    def test_the_stiff_cable_takes_the_turn_in_the_clamp(self) -> None:
        """At 50 N a jaw the pads hold the stiff cable's twist moment: it takes 0.319 of the 0.322 turns."""
        self.assertGreater(self.SIGN["right"] * self.taken["right"], 0.28, f"taken {self.taken}")
        self.assertGreater(self.SIGN["right"] * self.held["right"], 0.09, f"mid-cable {self.held}")

    def test_let_go_the_stiff_one_unwinds_and_the_soft_one_keeps_it(self) -> None:
        soft = {at: abs(r["left"]) for at, r in self.released.items()}
        stiff = {at: abs(r["right"]) for at, r in self.released.items()}
        self.assertLess(stiff[30], 0.01, f"stiff {stiff}")
        self.assertGreater(soft[60], 0.08, f"soft {soft}")
        for at in (5, 15, 30, 60):
            self.assertLess(stiff[at], soft[at], at)

    def test_the_scene_steps_captured_inside_its_budget(self) -> None:
        self.assertTrue(self.finite)
        self.assertTrue(self.sim.pipeline.captured, self.sim.pipeline.capture_error)
        self.assertLess(self.ms_per_step, 60.0, f"{self.ms_per_step:.1f} ms per control step")
        self.assertLess(self.ms_turning, 80.0, f"{self.ms_turning:.1f} ms per step while both hands turn")


if __name__ == "__main__":
    unittest.main()
