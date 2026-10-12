# SPDX-License-Identifier: Apache-2.0
"""``examples/09_demos/00_wipe_the_bar.py``: the sponge is in the hand, and pressing and sliding it wipes the bar.

The example is loaded by file path, the way a reader runs it, with its folder on
``sys.path`` so that ``import bar`` finds the scene beside it, and its ``main()``
runs against a studio that is not there. Then the right hand is driven the way
the page's wrist gizmo drives it -- the same rate-limited setpoint and the same
device IK, every control step -- down onto the bar and halfway along it.

The claims are physical rather than about the page: the hands are where the
example says, the foam presses the bar with newtons of force, the marker is gone
where the sponge went and untouched where it did not, the sponge is still in the
hand, and nothing is NaN. The last test resets the scene the way the page's
Reset does and checks the example's page takes hold again.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

TIER = "medium"
"""Builds the two-engine scene on a GPU and steps it for about twenty seconds."""

FOLDER = Path(__file__).resolve().parents[1] / "examples" / "09_demos"
EXAMPLE = FOLDER / "00_wipe_the_bar.py"


def _load():
    """The example as a module, its folder on the path for ``import bar``."""
    if str(FOLDER) not in sys.path:
        sys.path.insert(0, str(FOLDER))
    spec = importlib.util.spec_from_file_location("_rss_demo_wipe_the_bar", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _stub_view():
    """``rss.view`` replaced for the duration; what it was handed is yielded."""
    import robosimstudio as rss

    opened: list[tuple] = []
    real = rss.view
    rss.view = lambda sim, **kwargs: opened.append((sim, kwargs))
    try:
        yield opened
    finally:
        rss.view = real


class _Arm:
    """A studio arm panel, as far as the page uses one."""

    def __init__(self) -> None:
        self.resumed = 0

    def resume(self) -> None:
        self.resumed += 1


class _Server:
    def __init__(self) -> None:
        self.on_connect: list = []

    def on_client_connect(self, fn):
        self.on_connect.append(fn)
        return fn

    def get_clients(self) -> dict:
        return {}


class _Studio:
    """What ``page=`` is handed: viser's gui, the server, the arm panels and the per-step seam."""

    def __init__(self) -> None:
        from tests.fixtures.viser_stubs import _StubGui, _StubScene

        self.gui, self.scene, self.server = _StubGui(), _StubScene(), _Server()
        self.arms = {"left": _Arm(), "right": _Arm()}
        self.steps: list = []

    def each_step(self, fn) -> None:
        self.steps.append(fn)

    def tick(self) -> None:
        for fn in self.steps:
            fn()


class TestWipeTheBar(unittest.TestCase):
    """One build for the class: it is the scene's two engines that cost, not the assertions."""

    @classmethod
    def setUpClass(cls) -> None:
        if importlib.util.find_spec("superdex") is None:
            raise unittest.SkipTest("[superdex] is not installed; the sponge cannot be built")
        cls.module = _load()
        cls.bar = sys.modules["bar"]
        with _stub_view() as opened:
            cls.module.main()
        assert len(opened) == 1, f"the example opened {len(opened)} pages, not one"
        cls.sim, cls.kwargs = opened[0]
        cls.stripe = cls.sim["stripe"].entity

    @classmethod
    def tearDownClass(cls) -> None:
        sim = getattr(cls, "sim", None)
        if sim is not None:
            sim.close()

    def _tcp(self, arm: str = "right") -> np.ndarray:
        return np.asarray(self.sim.arm(arm).ee_pose(self.sim.state)[0], dtype=np.float64)

    def _sponge_in_hand(self) -> np.ndarray:
        return self.sim["sponge"].particle_q.mean(axis=0) - self._tcp()

    def test_1_the_page_opens_on_the_sponge_in_the_right_hand(self) -> None:
        sim, bar = self.sim, self.bar
        self.assertTrue(callable(self.kwargs.get("page")), "the example adds its readout through page=")
        for arm, (position, _) in bar.holds(sim).items():
            miss = np.linalg.norm(self._tcp(arm) - position)
            self.assertLess(miss, 0.004, f"the {arm} hand is {miss * 1e3:.1f} mm off its hold")
        self.assertLess(np.linalg.norm(self._sponge_in_hand()), 0.02, "the sponge is not between the right jaws")
        self.assertAlmostEqual(bar.GRIP_GAP, sim.gripper("right").width(sim.state), delta=0.003)
        self.assertAlmostEqual(bar.SUPPORT_GAP, sim.gripper("left").width(sim.state), delta=0.003)
        self.assertEqual(0.0, self.stripe.press, "the sponge starts clear of the bar")
        self.assertTrue(np.all(self.stripe.ink == 1.0), "the stripe starts whole")
        self.assertTrue(sim.state_is_finite())

    def test_2_pressing_and_sliding_the_hand_wipes_where_the_sponge_went(self) -> None:
        from robosimstudio.solvers import OscSetpoint

        sim, stripe = self.sim, self.stripe
        ik = sim.arm("right")
        law = OscSetpoint(control_dt=1.0 / float(sim.request.physics.control_hz))
        start, quat = ik.ee_pose(sim.state)
        law.reset(start, quat)
        held = self._sponge_in_hand()
        forces, peaks = [], []

        def drive(target: np.ndarray, steps: int) -> None:
            for _ in range(steps):
                position, now = ik.ee_pose(sim.state)
                ik.set_target(*law.advance(position, now, target, quat))
                ik.solve_into(sim.state.joint_q, sim.control.joint_target_q)
                sim.step_teleop()
                forces.append(stripe.press)
                peaks.append(stripe.peak)

        pressed = np.asarray(start) - np.array([0.0, 0.0, self.bar.HOVER + 0.008])  # 8 mm past first touch
        drive(pressed, 90)
        drive(pressed + np.array([0.0, 0.09, 0.0]), 150)  # along the bar, toward the far end: half the stripe

        self.assertTrue(sim.state_is_finite(), "the scene diverged under the press")
        self.assertGreater(max(forces), 1.5, f"the press peaked at {max(forces):.2f} N")
        self.assertGreater(max(peaks), 5.0e3, f"the peak pressure peaked at {max(peaks):.0f} Pa")
        u = stripe.cells[:, 0]
        wiped = float(stripe.ink[u < -0.02].mean())
        untouched = float(stripe.ink[u > 0.06].min())
        self.assertLess(wiped, 0.7, f"the near half the sponge crossed kept {wiped:.2f} of its ink")
        self.assertEqual(1.0, untouched, "ink went where the sponge never was")
        slip = float(np.linalg.norm(self._sponge_in_hand() - held))
        self.assertLess(slip, 0.01, f"the sponge moved {slip * 1e3:.1f} mm in the hand")

    def test_3_the_page_takes_hold_again_after_a_reset(self) -> None:
        sim, bar = self.sim, self.bar
        studio = _Studio()
        sim.reset()  # what the page does as it opens: the arms home, the hands open, the latch gone
        self.kwargs["page"](studio)
        self.assertEqual([1, 1], [arm.resumed for arm in studio.arms.values()], "the arms were not handed back")
        self.assertLess(np.linalg.norm(self._tcp() - bar.holds(sim)["right"][0]), 0.004)
        self.assertLess(np.linalg.norm(self._sponge_in_hand()), 0.02, "the hold was not taken again")
        self.assertTrue(np.all(self.stripe.ink == 1.0), "a reset puts the stripe back")
        studio.tick()
        self.assertRegex(studio.gui.markdown[-1].content, r"N\*\* · peak .* kPa")
        picture = studio.gui.images[-1].image
        self.assertEqual((np.uint8, 3), (picture.dtype, picture.shape[2]))

        sim.reset()  # the Reset button: the next control step's readout takes hold again
        studio.tick()
        self.assertEqual([2, 2], [arm.resumed for arm in studio.arms.values()])
        self.assertLess(np.linalg.norm(self._sponge_in_hand()), 0.02)
        self.assertTrue(sim.state_is_finite())


if __name__ == "__main__":
    unittest.main()
