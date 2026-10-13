# SPDX-License-Identifier: Apache-2.0
"""``demo/wipe_the_bar``: the whole job, from the robot's home pose, through the page's Run button.

The demo is loaded by file path, the way a reader runs it, with its folder on
``sys.path`` so that ``import bar`` and ``import routine`` find the files beside
it, and its ``main()`` runs against a studio that is not there. Then the
routine it hands the page is played the way its **Wipe the bar** button plays
it -- through the arm panels, one control step per tick -- from home to home.

The claims are physical: the arms start at home with their hands open and the
sponge on the mat; the foam presses the bar with newtons of force; the stripe is
cleaned; the sponge goes back where it lay; both arms end at home; nothing is
NaN. And the claims about the latch, which holds the sponge through the wipe
because the jaws' friction alone did not (README): nothing is pinned before the
first **Close** or after the last **Open**; it engaged only on a grasp the
contact made -- the jaws settled, both pads pressing, the foam squeezed; and it
pinned the foam where the squeeze left it, so nothing jumped when it did.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

TIER = "slow"
"""Builds the two-engine scene on a GPU and plays a minute of simulated time through the page."""

FOLDER = Path(__file__).resolve().parents[1] / "demo" / "wipe_the_bar"
SCRIPT = FOLDER / "wipe_the_bar.py"
PORT = 9341
CLEANED = 0.9
"""The share of the stripe the job must clean."""


def _load():
    """The demo as a module, its folder on the path for ``import bar`` and ``import routine``."""
    if str(FOLDER) not in sys.path:
        sys.path.insert(0, str(FOLDER))
    spec = importlib.util.spec_from_file_location("_rss_demo_wipe_the_bar", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestWipeTheBar(unittest.TestCase):
    """One take for the class: it is the scene's two engines and a minute of stepping that cost."""

    @classmethod
    def setUpClass(cls) -> None:
        if importlib.util.find_spec("superdex") is None:
            raise unittest.SkipTest("[superdex] is not installed; the sponge cannot be built")
        from robosimstudio.core.math import quat_to_mat
        from tests.fixtures.examples import _stub_studio
        from tests.fixtures.grasp import open_session, tick

        cls.module = _load()
        bar = sys.modules["bar"]
        with _stub_studio() as opened:
            cls.module.main()
        assert len(opened) == 1, f"the demo opened {len(opened)} pages, not one"
        cls.sim, cls.kwargs = opened[0]
        sim = cls.sim
        cls.session = open_session(sim, PORT, cls.kwargs.get("page"))
        routine = cls.kwargs["routines"]["Wipe the bar"]
        assert "session" in inspect.signature(routine).parameters
        cls.session.add_routines({"Wipe the bar": routine})
        panel = cls.session.panels[-1]
        tick(cls.session, 20)
        sponge, stripe, ik = sim["sponge"], sim["stripe"].entity, cls.session.arms["right"].ik
        cls.latch, bridge, gripper = bar.latch(sim), sponge.entity.bridge, sim.gripper("right")
        cls.start = {
            arm: (cls.session.arms[arm].ik.ee_pose(sim.state)[0], cls.session.arms[arm].ik.home_pose[0])
            for arm in ("right", "left")
        }
        cls.widths = {arm: sim.gripper(arm).width(sim.state) for arm in ("right", "left")}
        cls.lying = np.asarray(sponge.particle_q, dtype=np.float64)

        def pinned_in_hand() -> np.ndarray:
            """Where the latched foam nodes are now, in the hand's frame [m]."""
            pos, quat = ik.ee_pose(sim.state)
            nodes = np.asarray(sponge.particle_q, dtype=np.float64)[cls.latch.nodes]
            return (nodes - np.asarray(pos)) @ quat_to_mat(np.asarray(quat, dtype=np.float64))

        def moved() -> float:
            return float(np.linalg.norm((pinned_in_hand() - cls.latch.local).mean(axis=0)))

        cls.press = cls.peak = 0.0
        cls.trace: list[tuple[int, float, int]] = []  # (step, right grip command, pins standing)
        cls.jump = cls.lift = cls.held = None
        engaged_at, grasp_z, steps = None, None, 0
        panel.requested = "Wipe the bar"
        tick(cls.session, 1)
        while panel.gen is not None and steps < 7200:
            tick(cls.session, 1)
            steps += 1
            cls.press, cls.peak = max(cls.press, stripe.press), max(cls.peak, stripe.peak)
            cls.trace.append((steps, gripper.command, len(bridge._pins)))
            if cls.latch.latched:
                if engaged_at is None:
                    engaged_at, grasp_z = steps, float(ik.ee_pose(sim.state)[0][2])
                    cls.held = 0.0
                if steps == engaged_at + 6:  # 0.1 s after it engaged
                    cls.jump = moved()
                if cls.lift is None and float(ik.ee_pose(sim.state)[0][2]) > grasp_z + 0.010:
                    cls.lift = moved()  # 10 mm into the lift
                cls.held = max(cls.held, moved())
        cls.steps, cls.said = steps, panel._status.value if panel._status else ""
        now = np.asarray(sponge.particle_q, dtype=np.float64)
        events = [(step, what, {k: v for k, v in got.items() if k != "nodes"}) for step, what, got in cls.latch.events]
        print(
            f"[wipe_the_bar] {steps / 60.0:.1f} s, cleaned {stripe.cleaned:.3f}, press {cls.press:.2f} N, "
            f"peak {cls.peak / 1e3:.0f} kPa, sponge "
            f"{np.linalg.norm(now.mean(axis=0)[:2] - cls.lying.mean(axis=0)[:2]) * 1e3:.1f} mm from where it lay; "
            f"first close at step {next((st for st, c, _ in cls.trace if c >= 0.5), None)}; "
            f"latch {events}, jump {cls.jump}, lift {cls.lift}, held {cls.held}; {cls.said}",
            flush=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        session = getattr(cls, "session", None)
        if session is not None:
            session.teardown()

    def test_1_it_starts_at_home_with_the_hands_open_and_the_sponge_on_the_mat(self) -> None:
        for arm, (hand, home) in self.start.items():
            self.assertLess(float(np.linalg.norm(hand - home)), 0.006, f"the {arm} hand is not at home")
            self.assertGreater(self.widths[arm], 0.07, f"the {arm} hand is not open")
        mat_top = 0.75 + self.module.bar.LAYOUT.mat_top
        self.assertAlmostEqual(mat_top, float(self.lying[:, 2].min()), delta=0.003, msg="the sponge is not on the mat")

    def test_2_the_job_plays_to_the_end_and_both_arms_go_home(self) -> None:
        self.assertIn("done", self.said)
        for arm in ("right", "left"):
            ik = self.session.arms[arm].ik
            self.assertLess(float(np.linalg.norm(ik.ee_pose(self.sim.state)[0] - ik.home_pose[0])), 0.006)
        self.assertTrue(self.sim.state_is_finite())

    def test_3_the_foam_presses_the_bar(self) -> None:
        self.assertGreater(self.press, 1.0, f"the press peaked at {self.press:.2f} N")
        self.assertGreater(self.peak, 5.0e3, f"the peak pressure peaked at {self.peak:.0f} Pa")

    def test_4_the_stripe_is_clean(self) -> None:
        cleaned = self.sim["stripe"].entity.cleaned
        self.assertGreaterEqual(cleaned, CLEANED, f"{cleaned:.2f} of the stripe is clean")

    def test_5_the_sponge_goes_back_where_it_lay(self) -> None:
        now = np.asarray(self.sim["sponge"].particle_q, dtype=np.float64)
        moved = float(np.linalg.norm(now.mean(axis=0)[:2] - self.lying.mean(axis=0)[:2]))
        self.assertLess(moved, 0.02, f"the sponge is {moved * 1e3:.0f} mm from where it lay")
        self.assertAlmostEqual(float(self.lying[:, 2].min()), float(now[:, 2].min()), delta=0.003)

    def test_6_nothing_is_pinned_before_the_close_or_after_the_open(self) -> None:
        closes = [step for step, command, _ in self.trace if command >= 0.5]
        self.assertTrue(closes, "the right hand never closed")
        first, last = closes[0], closes[-1]
        for step, _, pins in self.trace:
            if step < first or step > last + 1:  # the hook lets go on the step the Open lands
                self.assertEqual(0, pins, f"something is pinned at step {step}, outside the grip")
        self.assertEqual(["engaged", "released"], [what for _, what, _ in self.latch.events])

    def test_7_it_latched_only_on_a_grasp_the_contact_made(self) -> None:
        bar = self.module.bar
        step, _, got = self.latch.events[0]
        first = next(s for s, command, _ in self.trace if command >= 0.5)
        self.assertGreaterEqual(step, first + bar.LATCH_SETTLE, "it latched before the jaws could settle")
        self.assertLessEqual(got["still"], bar.LATCH_STILL)
        self.assertGreaterEqual(min(got["press"]), bar.LATCH_PRESS, f"pads {got['press']} N")
        self.assertLessEqual(got["opening"], got["free"] - bar.LATCH_SQUEEZE, "the foam is not squeezed")

    def test_8_the_latch_pins_the_foam_where_the_squeeze_left_it(self) -> None:
        self.assertLess(self.jump, 0.001, f"the pinned foam moved {self.jump * 1e3:.2f} mm in 0.1 s")
        self.assertLess(self.lift, 0.001, f"10 mm into the lift it had moved {self.lift * 1e3:.2f} mm")

    def test_9_the_latch_holds_it_through_the_wipe(self) -> None:
        self.assertLess(self.held, 0.010, f"the latched foam moved {self.held * 1e3:.1f} mm in the hand")


if __name__ == "__main__":
    unittest.main()
