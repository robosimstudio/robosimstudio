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
NaN; and the sponge stayed in the jaws by their squeeze and friction alone,
which is read off the foam under the pads: nothing pins it.

**Skipped unless ``RSS_WIPE_THE_BAR=1``**, because three of the claims do not
hold yet and a take is two minutes of a GPU. Three takes on 2026-10-13
(``demo/wipe_the_bar/README.md``): the start, the finish at home and the press
held in all three; the stripe was 0.761, 0.724 and 0.687 clean, the foam under
the pads moved 69, 226 and 218 mm in the hand, and in two takes the sponge left
the jaws before the last stroke.
"""

from __future__ import annotations

import importlib.util
import os
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
SLIP = 0.010
"""How far the foam under the pads may move in the hand during the wipe [m]."""


def _load():
    """The demo as a module, its folder on the path for ``import bar`` and ``import routine``."""
    if str(FOLDER) not in sys.path:
        sys.path.insert(0, str(FOLDER))
    spec = importlib.util.spec_from_file_location("_rss_demo_wipe_the_bar", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(os.environ.get("RSS_WIPE_THE_BAR"), "the sponge slides in the jaws; see the module docstring")
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
        with _stub_studio() as opened:
            cls.module.main()
        assert len(opened) == 1, f"the demo opened {len(opened)} pages, not one"
        cls.sim, cls.kwargs = opened[0]
        sim = cls.sim
        cls.session = open_session(sim, PORT, cls.kwargs.get("page"))
        routine = cls.kwargs["routines"]["Wipe the bar"]
        takes = []

        def play(sim_, session):  # the page's own call, keeping the take to read
            takes.append(sys.modules["routine"].Take(sim_, session))
            return takes[-1].steps()

        assert "session" in __import__("inspect").signature(routine).parameters
        cls.session.add_routines({"Wipe the bar": play})
        panel = cls.session.panels[-1]
        tick(cls.session, 20)
        sponge, stripe, ik = sim["sponge"], sim["stripe"].entity, cls.session.arms["right"].ik
        cls.start = {
            arm: (cls.session.arms[arm].ik.ee_pose(sim.state)[0], cls.session.arms[arm].ik.home_pose[0])
            for arm in ("right", "left")
        }
        cls.widths = {arm: sim.gripper(arm).width(sim.state) for arm in ("right", "left")}
        cls.lying = np.asarray(sponge.particle_q, dtype=np.float64)

        def under_pads() -> list[np.ndarray]:
            pos, quat = ik.ee_pose(sim.state)
            rot, nodes = quat_to_mat(np.asarray(quat, dtype=np.float64)), np.asarray(sponge.particle_q)
            out = []
            for body in cls.session.arms["right"].finger_bodies:
                samples = sponge.entity.bridge.contact_points("sponge", against=int(body))
                if len(samples["pos"]):
                    near = np.linalg.norm(nodes - samples["pos"].mean(axis=0), axis=1) < 0.007
                    out.append(np.flatnonzero(near))
            return [(idx, (nodes[idx] - pos) @ rot) for idx in out]

        cls.press = cls.peak = cls.slip = 0.0
        anchors, steps = None, 0
        panel.requested = "Wipe the bar"
        tick(cls.session, 1)
        while panel.gen is not None and steps < 7200:
            tick(cls.session, 1)
            steps += 1
            phase = takes[-1].phase
            cls.press, cls.peak = max(cls.press, stripe.press), max(cls.peak, stripe.peak)
            if phase == "carry" and anchors is None:
                anchors = under_pads()
            if phase == "wipe" and anchors:
                pos, quat = ik.ee_pose(sim.state)
                rot, nodes = quat_to_mat(np.asarray(quat, dtype=np.float64)), np.asarray(sponge.particle_q)
                moved = [float(np.linalg.norm(((nodes[i] - pos) @ rot - at).mean(axis=0))) for i, at in anchors]
                cls.slip = max(cls.slip, *moved)
        cls.take, cls.steps, cls.said = takes[-1], steps, panel._status.value if panel._status else ""
        now = np.asarray(sponge.particle_q, dtype=np.float64)
        print(
            f"[wipe_the_bar] {steps / 60.0:.1f} s, cleaned {stripe.cleaned:.3f}, press {cls.press:.2f} N, "
            f"peak {cls.peak / 1e3:.0f} kPa, slip {cls.slip * 1e3:.1f} mm, sponge "
            f"{np.linalg.norm(now.mean(axis=0)[:2] - cls.lying.mean(axis=0)[:2]) * 1e3:.0f} mm from where it lay; "
            f"{cls.said}",
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
        self.assertEqual("done", self.take.phase)
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

    def test_6_the_sponge_stays_in_the_jaws_by_friction(self) -> None:
        self.assertLess(self.slip, SLIP, f"the foam under the pads moved {self.slip * 1e3:.0f} mm in the hand")


if __name__ == "__main__":
    unittest.main()
