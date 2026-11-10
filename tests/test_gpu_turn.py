# SPDX-License-Identifier: Apache-2.0
"""Pool workers take turns on the GPU: physics shared, a render alone, and nobody wedged.

Threads with their own file descriptors stand in for processes here, which ``flock``
cannot tell apart: a lock belongs to the open file, not to the process. No device.
"""

from __future__ import annotations

import inspect
import os
import shutil
import tempfile
import threading
import time
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

from robosimstudio.core.gpu_turn import OFF, TURN_ENV, GpuTurn, gpu_turn
from robosimstudio_bench.envs import fleet as fleet_module
from robosimstudio_bench.envs.fleet import FleetEnv
from robosimstudio_bench.run import pool

TIER = "fast"
"""No GPU. Gates every commit."""


class _Turns(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.mkdtemp(prefix="gpu-turn-test-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def turn(self, **kwargs) -> GpuTurn:
        turn = GpuTurn(self.dir, "cuda:0", **kwargs)
        self.addCleanup(turn.close)
        return turn


class TestTurns(_Turns):
    def test_physics_shares(self):
        with self.turn(patience=0.2).stepping(), self.turn(patience=0.2).stepping():
            pass  # Neither waited out the other's patience.

    def test_a_render_waits_for_the_step_running(self):
        render = self.turn()
        entered = threading.Event()

        def run() -> None:
            with render.rendering():
                entered.set()

        with self.turn().stepping():
            thread = threading.Thread(target=run)
            thread.start()
            self.assertFalse(entered.wait(0.1))
        self.assertTrue(entered.wait(5.0))
        thread.join()

    def test_a_step_waits_for_the_render_running(self):
        step = self.turn()
        entered = threading.Event()

        def run() -> None:
            with step.stepping():
                entered.set()

        with self.turn().rendering():
            thread = threading.Thread(target=run)
            thread.start()
            self.assertFalse(entered.wait(0.1))
        self.assertTrue(entered.wait(5.0))
        thread.join()

    def test_two_renders_take_turns(self):
        second = self.turn()
        entered = threading.Event()

        def run() -> None:
            with second.rendering():
                entered.set()

        with self.turn().rendering():
            thread = threading.Thread(target=run)
            thread.start()
            self.assertFalse(entered.wait(0.1))
        self.assertTrue(entered.wait(5.0))
        thread.join()

    def test_steppers_that_overlap_cannot_starve_a_render(self):
        """The gate: a stepper arriving while a render waits queues behind it."""
        stop = threading.Event()
        rendered = threading.Event()

        def stepper() -> None:
            turn = GpuTurn(self.dir, "cuda:0")
            while not stop.is_set():
                with turn.stepping():
                    time.sleep(0.005)
            turn.close()

        threads = [threading.Thread(target=stepper) for _ in range(4)]
        for thread in threads:
            thread.start()
        time.sleep(0.05)
        with self.turn(patience=5.0).rendering():
            rendered.set()
        stop.set()
        for thread in threads:
            thread.join()
        self.assertTrue(rendered.is_set())

    def test_a_turn_gives_the_gpu_back_when_the_body_raises(self):
        with self.assertRaises(ValueError), self.turn().rendering():
            raise ValueError("a tracer failed")
        with self.turn(patience=0.2).rendering():
            pass

    def test_a_process_that_dies_holding_it_gives_it_back(self):
        """Its file descriptors close, and the locks with them."""
        holder = GpuTurn(self.dir, "cuda:0")
        context = holder.rendering()
        context.__enter__()
        holder.close()
        late = self.turn(patience=0.2)
        with late.rendering():
            pass
        self.assertFalse(late.given_up)

    def test_devices_do_not_share_a_turn(self):
        other = GpuTurn(self.dir, "cuda:1", patience=0.2)
        self.addCleanup(other.close)
        with self.turn().rendering(), other.rendering():
            pass
        self.assertFalse(other.given_up)


class TestPatience(_Turns):
    def test_a_wedged_render_does_not_wedge_the_others(self):
        waiter = self.turn(patience=0.05)
        ran = []
        with self.turn().rendering():
            with waiter.stepping():
                ran.append("step")
            with waiter.rendering():
                ran.append("render")
        self.assertEqual(ran, ["step", "render"])
        self.assertTrue(waiter.given_up)

    def test_giving_up_is_for_good(self):
        waiter = self.turn(patience=0.05)
        with self.turn().rendering(), waiter.stepping():
            pass
        started = time.monotonic()
        with self.turn().rendering(), waiter.stepping():
            pass
        self.assertLess(time.monotonic() - started, 0.04)

    def test_a_waiter_that_gave_up_holds_nothing(self):
        with self.turn().rendering():
            waiter = self.turn(patience=0.05)
            with waiter.stepping():
                pass
        with self.turn(patience=0.2).rendering():
            pass


class TestArranged(_Turns):
    def test_no_directory_is_no_turn(self):
        with mock.patch.dict(os.environ):
            os.environ.pop(TURN_ENV, None)
            self.assertIsNone(gpu_turn("cuda:0"))

    def test_off_is_no_turn(self):
        with mock.patch.dict(os.environ, {TURN_ENV: OFF}):
            self.assertIsNone(gpu_turn("cuda:0"))

    def test_a_directory_is_a_turn(self):
        with mock.patch.dict(os.environ, {TURN_ENV: self.dir}):
            turn = gpu_turn("cuda:0")
        self.addCleanup(turn.close)
        self.assertIsInstance(turn, GpuTurn)

    def test_a_directory_that_is_gone_is_no_turn_and_no_error(self):
        """A replacement worker started as its pool ends."""
        with mock.patch.dict(os.environ, {TURN_ENV: os.path.join(self.dir, "gone")}), mock.patch("builtins.print"):
            self.assertIsNone(gpu_turn("cuda:0"))

    def test_the_pool_arranges_one_for_two_workers_or_more_and_leaves_a_callers_alone(self):
        """Read off the source: running a pool spawns processes."""
        source = inspect.getsource(pool.run_pooled)
        self.assertIn("workers > 1 and not os.environ.get(TURN_ENV)", source)


class _Recorder:
    def __init__(self) -> None:
        self.events: list[str] = []

    def stepping(self):
        self.events.append("stepping")
        return nullcontext()

    def rendering(self):
        self.events.append("rendering")
        return nullcontext()


def _fleet(turn) -> FleetEnv:
    env = FleetEnv.__new__(FleetEnv)
    env._turn = turn
    env._rendered = False
    env.step_count = 0
    env._frames = None
    env._frames_at = -1
    env.renderer = SimpleNamespace(render_worlds=lambda state, worlds: ["frame"])
    env.pipeline = SimpleNamespace(state_0="state", step=lambda: None)
    env.model = SimpleNamespace(device="cuda:0")
    env._control_stage = lambda: None
    return env


class TestFleetTakesTurns(unittest.TestCase):
    def test_a_step_holds_its_turn_through_the_work(self):
        turn = _Recorder()
        env = _fleet(turn)
        order: list[str] = []
        env._control_stage = lambda: order.append("control")
        env.pipeline.step = lambda: order.append("physics")
        with mock.patch.object(fleet_module.wp, "synchronize_device", lambda device: order.append("sync")):
            env.step()
        self.assertEqual(turn.events, ["stepping"])
        self.assertEqual(order, ["control", "physics", "sync"])
        self.assertEqual(env.step_count, 1)

    def test_no_turn_adds_no_synchronize(self):
        env = _fleet(None)
        with mock.patch.object(fleet_module.wp, "synchronize_device", side_effect=AssertionError("synchronized")):
            env.step()
        self.assertEqual(env.step_count, 1)

    def test_the_first_look_takes_no_turn(self):
        """It builds pipelines, at 12-30 s, in every worker at once."""
        turn = _Recorder()
        env = _fleet(turn)
        env.render_worlds([0])
        self.assertEqual(turn.events, [])
        env.render_worlds([0])
        env.render_worlds(None)
        self.assertEqual(turn.events, ["rendering", "rendering"])

    def test_a_look_nobody_needs_to_render_takes_no_turn(self):
        turn = _Recorder()
        env = _fleet(turn)
        env.render_worlds(None)
        env.render_worlds(None)
        self.assertEqual(turn.events, [], "the second is the cached picture of this step")

    def test_no_turn_renders_as_it_always_did(self):
        env = _fleet(None)
        self.assertEqual(env.render_worlds([0]), ["frame"])
        self.assertEqual(env.render_worlds([0]), ["frame"])


if __name__ == "__main__":
    unittest.main()
