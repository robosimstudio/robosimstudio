# SPDX-License-Identifier: Apache-2.0
"""Regressions for the policy agent's fleet waits, servo law and policy close.

- A marshalled call or rendezvous ask that timed out is withdrawn: it never
  runs late, and never clears a park its thread has since entered elsewhere.
- The stepper's unreachable verdict is asked once per stall window.
- The serial servo and the fleet stepper share one orientation law.
- A graph policy's close puts back the env's per-step rendering and removes
  the staged copy of its graph.

No GPU.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from robosimstudio.core.bus import BusTimeout
from robosimstudio_policy_agent import servo
from robosimstudio_policy_agent.fleet import stepper
from robosimstudio_policy_agent.fleet.barrier import ParkTable, Rendezvous
from robosimstudio_policy_agent.fleet.world_bus import ServiceQueue

TIER = "fast"
"""No GPU. Gates every commit."""


class ServiceQueueTimeoutTest(unittest.TestCase):
    def test_timed_out_request_is_withdrawn(self) -> None:
        parks = ParkTable(1)
        queue = ServiceQueue(parks=parks)
        ran: list[int] = []
        with self.assertRaises(BusTimeout):
            queue.post(0, lambda: ran.append(1), (), {}, timeout=0.02)
        # The same thread is now in another fleet wait; the stale request must
        # neither run nor release it.
        with parks.parked(0):
            self.assertEqual(queue.pump(), 0)
            self.assertEqual(ran, [])
            self.assertTrue(parks.parked_mask()[0])

    def test_late_answer_leaves_the_next_park_alone(self) -> None:
        """A request already running when its poster gives up still finishes,
        but its completion must not unpark the wait the poster is in by then."""
        parks = ParkTable(1)
        queue = ServiceQueue(parks=parks)
        started = threading.Event()
        release = threading.Event()

        def slow() -> str:
            started.set()
            release.wait(2.0)
            return "late"

        pumper = threading.Thread(target=lambda: (time.sleep(0.01), queue.pump()))
        pumper.start()
        try:
            with self.assertRaises(BusTimeout):
                queue.post(0, slow, (), {}, timeout=0.1)
            self.assertTrue(started.is_set())
            with parks.parked(0):
                release.set()
                pumper.join(2.0)
                self.assertFalse(pumper.is_alive())
                self.assertTrue(parks.parked_mask()[0])
        finally:
            release.set()
            pumper.join(2.0)


class RendezvousTimeoutTest(unittest.TestCase):
    def test_timed_out_ask_is_withdrawn(self) -> None:
        parks = ParkTable(1)
        rendezvous = Rendezvous(parks=parks)
        calls: list[list] = []
        rendezvous.register("solve", lambda group: calls.append(group) or [None] * len(group))
        with parks.parked(0), self.assertRaises(BusTimeout):
            rendezvous.post(0, "solve", "stale", timeout=0.02)
        self.assertFalse(rendezvous.pending())
        with parks.parked(0):
            self.assertEqual(rendezvous.dispatch(), 0)
            self.assertEqual(calls, [])
            self.assertTrue(parks.parked_mask()[0])

    def test_late_settle_leaves_the_next_park_alone(self) -> None:
        parks = ParkTable(1)
        rendezvous = Rendezvous(parks=parks)
        started = threading.Event()
        release = threading.Event()

        def adapter(group):
            started.set()
            release.wait(2.0)
            return ["late"] * len(group)

        rendezvous.register("solve", adapter)
        fleet = threading.Thread(target=lambda: (time.sleep(0.01), rendezvous.dispatch()))
        fleet.start()
        try:
            with parks.parked(0), self.assertRaises(BusTimeout):
                rendezvous.post(0, "solve", "slow", timeout=0.1)
            self.assertTrue(started.is_set())
            with parks.parked(0):
                release.set()
                fleet.join(2.0)
                self.assertFalse(fleet.is_alive())
                self.assertTrue(parks.parked_mask()[0])
        finally:
            release.set()
            fleet.join(2.0)


class SharedOrientationLawTest(unittest.TestCase):
    def test_stepper_uses_the_servo_helpers(self) -> None:
        self.assertIs(stepper.solver_goal, servo.solver_goal)
        self.assertIs(stepper.rotation_error, servo.rotation_error)
        self.assertFalse(hasattr(stepper, "_solver_goal"))
        self.assertFalse(hasattr(stepper.StepService, "_rotation_error"))

    def test_servo_methods_wrap_the_helpers(self) -> None:
        rng = np.random.default_rng(3)
        for honours_roll in (False, True):
            holder = SimpleNamespace(honours_roll=honours_roll)
            for _ in range(10):
                a = rng.normal(size=4)
                b = rng.normal(size=4)
                a /= np.linalg.norm(a)
                b /= np.linalg.norm(b)
                np.testing.assert_allclose(
                    servo.CartesianServo._solver_goal(holder, a, b), servo.solver_goal(a, b, honours_roll)
                )
                self.assertEqual(
                    servo.CartesianServo._rotation_error(holder, a, b), servo.rotation_error(a, b, honours_roll)
                )
        # Exactly opposed approach axes still give a unit quaternion.
        flipped = servo.solver_goal((0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0), False)
        self.assertAlmostEqual(float(np.linalg.norm(flipped)), 1.0)
        self.assertAlmostEqual(servo.rotation_error((0.0, 0.0, 0.0, 1.0), flipped, False), np.pi)


class ReachVerdictCadenceTest(unittest.TestCase):
    def test_judged_once_per_stall_window(self) -> None:
        service = object.__new__(stepper.StepService)
        service.arm_names = ["arm"]
        service.cadence = 1
        service.step = 0
        service.fleet = SimpleNamespace(
            scene=SimpleNamespace(embodiment=SimpleNamespace(arms={"arm": SimpleNamespace(ee_body=0)}))
        )
        service._offsets = lambda name: ()
        service._release_arm = lambda world, name: None
        judged: list[int] = []
        service._beyond_reach = lambda world, name, program, target, snapshot: judged.append(program.stalled) or False
        program = stepper._Program(
            "servo",
            goal=(np.array([1.0, 0.0, 0.0]), "arm", None),
            deadline=1000,
            tolerance=0.001,
            stall_steps=12,
        )
        snapshot = SimpleNamespace(body=lambda world: None)
        with mock.patch.object(stepper, "tcp_pose", lambda *a: (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))):
            for tick in range(60):
                service.step = tick
                done, _error, _result = service._finished(0, 0, program, snapshot)
                self.assertFalse(done)
        # Stalled from the second readback: windows close at 12, 24, 36, 48.
        self.assertEqual(judged, [12, 24, 36, 48])


class GraphPolicyCloseTest(unittest.TestCase):
    def _policy(self, env, graph: str):
        from robosimstudio_policy_agent import graph_solution as gs

        with (
            mock.patch.object(gs.SolutionGraph, "_new_connector", lambda self: None),
            mock.patch.object(gs, "solution_param", lambda *a, **k: 0.0, create=True),
        ):
            return gs.SolutionGraph(env, graph=graph)

    @staticmethod
    def _env():
        task = SimpleNamespace(tools=None, graph_path=lambda g: {}[g], full_name="x")
        return SimpleNamespace(renderer=object(), obs_render_enabled=True, task=task)

    def test_on_demand_render_flag_restored(self) -> None:
        root = Path(__file__).resolve().parents[1]
        env = self._env()
        policy = self._policy(env, str(root / "examples/08_policies/pick_place_camera"))
        self.assertTrue(policy._on_demand_observations)
        self.assertFalse(env.obs_render_enabled)
        policy.close()
        self.assertTrue(env.obs_render_enabled)

    def test_staged_graph_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            graph = Path(tmp) / "g"
            graph.mkdir()
            (graph / "workflow.json").write_text(
                json.dumps({"nodes": {"r": {"type": "router", "script": "scripts/route_perception.py"}}})
            )
            policy = self._policy(self._env(), str(graph))
            staged = Path(policy.graph_path)
            self.assertNotEqual(staged, graph)
            self.assertTrue(staged.exists())
            policy.close()
            self.assertFalse(staged.exists())
            self.assertTrue(graph.exists())


if __name__ == "__main__":
    unittest.main()
