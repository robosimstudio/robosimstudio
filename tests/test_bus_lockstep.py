# SPDX-License-Identifier: Apache-2.0
"""The lockstep clock: nothing steps while a graph thinks.

``--clock lockstep`` makes a single-world graph run take strict turns with the
simulation, the way a fleet's barrier already does. The rollout's hold step is
taken only while the graph's own thread is *parked* on the simulator -- a queued
bus request, or a ``wait_for_step`` -- or after it has finished, so a node that
spends seconds in a model or a planner costs no control steps and a
deterministic graph takes the same number of steps on any machine.

These drive :class:`~robosimstudio.core.bus.SimBus` and
:meth:`~robosimstudio_policy_agent.graph_solution.SolutionGraph._lockstep_turn`
against a counting stand-in for the environment, so the contract is checked
without a scene. No GPU: the bus is plain threading.
"""

from __future__ import annotations

import random
import threading
import time
import types
import unittest
from typing import Any

from robosimstudio.core.bus import BusClosed, BusTimeout, SimBus, SimSnapshot

TIER = "fast"
"""No GPU. Gates every commit."""


class _Facade:
    """What the turn reads off the connector's facade."""

    _current_done = False


class _Env:
    """Counts steps and remembers a termination request."""

    def __init__(self) -> None:
        self.step_count = 0
        self.termination: str | None = None

    def request_termination(self, reason: str) -> None:
        self.termination = reason


class _Rig:
    """One single-world graph run, minus the scene.

    ``rollout`` is the runner's loop: ask the policy for its turn, take one
    hold step, publish. A worker request named ``step`` advances the same
    counter from inside the owner thread, as a streamed waypoint does.
    """

    def __init__(self, *, timeout: float = 5.0, hold_between_requests: bool = True) -> None:
        from robosimstudio_policy_agent.graph_solution import SolutionGraph

        self.env = _Env()
        self.bus = SimBus(timeout=timeout)
        self.holds = 0
        policy = SolutionGraph.__new__(SolutionGraph)
        policy.env = self.env
        policy.bus = self.bus
        policy.trace = {}
        policy.hold_between_requests = hold_between_requests
        policy.connector = types.SimpleNamespace(facade=_Facade())
        policy._thread = None
        self.policy = policy

    def step(self) -> int:
        """One control step, owner thread: count it and publish it."""
        self.env.step_count += 1
        self.bus.publish(SimSnapshot(step=self.env.step_count))
        return self.env.step_count

    def start(self, body) -> threading.Thread:
        """Run ``body(bus, rig)`` as the graph thread, adopted and departed as SolutionGraph does."""

        def run() -> None:
            self.bus.adopt()
            try:
                body(self.bus, self)
            finally:
                self.bus.depart()

        thread = threading.Thread(target=run, daemon=True)
        self.policy._thread = thread
        thread.start()
        return thread

    def rollout(self, *, until_departed: bool = True, max_turns: int = 10_000) -> None:
        for _ in range(max_turns):
            self.policy._lockstep_turn()
            if self.env.termination is not None:
                return
            if until_departed and self.bus.parked() == "departed":
                return
            self.holds += 1
            self.step()


class SimBusParks(unittest.TestCase):
    """The primitive: when is the graph's thread parked, and who says so."""

    def test_a_thread_in_host_code_is_not_parked(self) -> None:
        bus = SimBus(timeout=5.0)
        release = threading.Event()

        def graph() -> None:
            bus.adopt()
            release.wait(5.0)  # host code: a model, a planner
            bus.depart()

        thread = threading.Thread(target=graph, daemon=True)
        thread.start()
        try:
            self.assertEqual(bus.await_parked(0.2), "", "still thinking at the deadline")
        finally:
            release.set()
            thread.join(5.0)
        self.assertEqual(bus.parked(), "departed")

    def test_a_queued_request_parks_and_taking_it_unparks(self) -> None:
        bus = SimBus(timeout=5.0)
        bus.owner_thread = -1
        seen: list[str] = []

        def graph() -> None:
            bus.adopt()
            bus.call(lambda: seen.append(bus.parked()))
            bus.depart()

        thread = threading.Thread(target=graph, daemon=True)
        thread.start()
        self.assertEqual(bus.await_parked(5.0), "request")
        self.assertTrue(bus.service_principal())
        thread.join(5.0)
        # Read on the owner while the request ran: taken, so no longer parked --
        # before its worker was woken.
        self.assertEqual(seen, [""])

    def test_a_step_wait_parks_until_the_step_is_published(self) -> None:
        bus = SimBus(timeout=5.0)
        woke = threading.Event()

        def graph() -> None:
            bus.adopt()
            bus.wait_for_step(after=0)
            woke.set()
            time.sleep(5.0)  # host code again, until the test ends

        thread = threading.Thread(target=graph, daemon=True)
        thread.start()
        self.assertEqual(bus.await_parked(5.0), "wait")
        bus.publish(SimSnapshot(step=1))
        # Unparked by the publish itself, not by the worker getting round to it.
        self.assertEqual(bus.parked(), "")
        self.assertTrue(woke.wait(5.0))
        bus.close()

    def test_a_helper_threads_requests_are_answered_while_the_graph_thinks(self) -> None:
        """An observation poller must not starve behind a graph in host code."""
        bus = SimBus(timeout=5.0)
        release = threading.Event()
        answered: list[int] = []

        def graph() -> None:
            bus.adopt()
            release.wait(5.0)
            bus.depart()

        def poller() -> None:
            answered.append(bus.call(lambda: 7))

        principal = threading.Thread(target=graph, daemon=True)
        principal.start()
        while bus._principal is None:
            time.sleep(0.001)
        helper = threading.Thread(target=poller, daemon=True)
        helper.start()
        self.assertEqual(bus.await_parked(0.5), "", "a helper's request does not park the graph")
        helper.join(5.0)
        self.assertEqual(answered, [7])
        release.set()
        principal.join(5.0)

    def test_a_helper_threads_wait_does_not_release_the_clock(self) -> None:
        bus = SimBus(timeout=5.0)
        release = threading.Event()

        def graph() -> None:
            bus.adopt()
            release.wait(5.0)
            bus.depart()

        principal = threading.Thread(target=graph, daemon=True)
        principal.start()
        while bus._principal is None:
            time.sleep(0.001)
        helper = threading.Thread(target=lambda: bus.wait_for_step(after=0, timeout=2.0), daemon=True)
        helper.start()
        with self.assertLogs("robosimstudio.core.bus", level="WARNING") as said:
            self.assertEqual(bus.await_parked(0.3), "")

            # The owner is lockstep now, so a helper's wait is worth a word: it
            # is answered only when the graph's own thread next parks.
            def wait_briefly() -> None:
                with self.assertRaises(BusTimeout):
                    bus.wait_for_step(after=0, timeout=0.2)

            second = threading.Thread(target=wait_briefly, daemon=True)
            second.start()
            second.join(5.0)
        self.assertEqual(len(said.output), 1, "said once, not per wait")
        self.assertIn("not the graph's own thread", said.output[0])
        release.set()
        bus.publish(SimSnapshot(step=1))
        principal.join(5.0)
        helper.join(5.0)

    def test_wait_for_step_wakes_on_publish_and_times_out(self) -> None:
        bus = SimBus(timeout=5.0)
        got: list[int] = []
        waiter = threading.Thread(target=lambda: got.append(bus.wait_for_step(after=0).step), daemon=True)
        waiter.start()
        time.sleep(0.05)
        t0 = time.perf_counter()
        bus.publish(SimSnapshot(step=1))
        waiter.join(5.0)
        self.assertEqual(got, [1])
        self.assertLess(time.perf_counter() - t0, 1.0)
        with self.assertRaises(BusTimeout) as caught:
            bus.wait_for_step(after=1, timeout=0.05)
        self.assertNotIsInstance(caught.exception, BusClosed)

    def test_close_releases_a_step_wait_as_closed(self) -> None:
        bus = SimBus(timeout=5.0)
        caught: list[BaseException] = []

        def waiter() -> None:
            try:
                bus.wait_for_step(after=0)
            except BaseException as exc:
                caught.append(exc)

        thread = threading.Thread(target=waiter, daemon=True)
        thread.start()
        time.sleep(0.05)
        bus.close()
        thread.join(5.0)
        self.assertEqual(len(caught), 1)
        self.assertIsInstance(caught[0], BusClosed)
        self.assertEqual(bus.parked(), "closed")


class LockstepTurns(unittest.TestCase):
    """The policy's half: one rollout iteration's turn, over the bus."""

    def test_a_worker_sleeping_in_host_code_advances_no_steps(self) -> None:
        rig = _Rig(timeout=10.0)
        observed: dict[str, int] = {}

        def graph(bus: SimBus, rig: _Rig) -> None:
            observed["before"] = bus.snapshot.step
            time.sleep(0.4)  # perception, a VLM, a planner
            observed["after"] = bus.snapshot.step
            bus.wait_for_step(after=bus.snapshot.step)

        rig.start(graph)
        rig.rollout()
        self.assertEqual(observed["before"], observed["after"], "a step was taken while the graph was thinking")
        self.assertEqual(rig.env.step_count, 1, "the one step was the one its wait asked for")

    def test_a_parked_worker_lets_steps_happen(self) -> None:
        rig = _Rig()

        def graph(bus: SimBus, rig: _Rig) -> None:
            start = bus.snapshot.step
            while bus.snapshot.step < start + 30:  # robot.wait_steps(30)
                bus.wait_for_step(after=bus.snapshot.step)

        rig.start(graph)
        rig.rollout()
        self.assertEqual(rig.env.step_count, 30)
        self.assertEqual(rig.holds, 30)

    def test_one_hold_step_follows_each_serviced_request(self) -> None:
        """A streamed trajectory: a waypoint step, then a hold, per request."""
        rig = _Rig()

        def graph(bus: SimBus, rig: _Rig) -> None:
            for _ in range(25):
                bus.call(rig.step)
                time.sleep(random.uniform(0.0, 0.004))  # the next waypoint's host work

        rig.start(graph)
        rig.rollout()
        # 25 waypoint steps and a hold after each of the first 24; the last
        # one's falls to the settle, since the graph has departed by then.
        self.assertEqual(rig.env.step_count, 49)
        self.assertEqual(rig.holds, 24)

    def test_without_the_hold_only_the_graph_steps(self) -> None:
        rig = _Rig(hold_between_requests=False)

        def graph(bus: SimBus, rig: _Rig) -> None:
            for _ in range(25):
                bus.call(rig.step)
            bus.wait_for_step(after=bus.snapshot.step)

        rig.start(graph)
        rig.rollout()
        self.assertEqual(rig.env.step_count, 26)
        self.assertEqual(rig.holds, 1)

    def test_step_counts_do_not_depend_on_host_speed(self) -> None:
        """The determinism claim: the same graph, timed differently, takes the same steps."""

        def graph(bus: SimBus, rig: _Rig) -> None:
            for i in range(40):
                if i % 3 == 0:
                    bus.call(rig.step)  # a waypoint
                elif i % 3 == 1:
                    bus.call(lambda: bus.snapshot.step)  # a read
                else:
                    bus.wait_for_step(after=bus.snapshot.step)  # a settle
                time.sleep(random.uniform(0.0, 0.003))

        counts = []
        for _ in range(4):
            rig = _Rig()
            rig.start(graph)
            rig.rollout()
            counts.append((rig.env.step_count, rig.holds))
        self.assertEqual(len(set(counts)), 1, counts)

    def test_a_finished_graph_reads_finished_on_the_turn_it_departs(self) -> None:
        rig = _Rig()
        rig.start(lambda bus, rig: bus.call(rig.step))
        rig.policy._lockstep_turn()  # services the request
        rig.policy._lockstep_turn()  # its hold is owed, and the graph departs
        self.assertFalse(rig.policy._thread.is_alive(), "joined, so `finished` is already true")

    def test_a_graph_that_never_parks_ends_its_episode(self) -> None:
        """The timeout path: --bus-timeout is a wall-clock watchdog on host code."""
        rig = _Rig(timeout=0.3)
        release = threading.Event()
        rig.start(lambda bus, rig: release.wait(10.0))
        t0 = time.perf_counter()
        try:
            rig.rollout()
        finally:
            release.set()
        self.assertLess(time.perf_counter() - t0, 5.0)
        self.assertEqual(rig.env.step_count, 0)
        self.assertIsNotNone(rig.env.termination)
        self.assertIn("held the lockstep clock", rig.env.termination)
        self.assertIn("lockstep_stall", rig.policy.trace)

    def test_a_request_still_times_out_when_nobody_pumps(self) -> None:
        bus = SimBus(timeout=0.05)
        bus.owner_thread = -1
        with self.assertRaises(BusTimeout):
            bus.call(lambda: None)
        self.assertEqual(bus.parked(), "", "a withdrawn request no longer parks its thread")


class RuntimeChoices(unittest.TestCase):
    """The flags, the job fields and what each runner records."""

    def test_job_fields_default_to_upstream_and_validate(self) -> None:
        from robosimstudio.specs import JobSpec

        job = JobSpec(name="j", task="t")
        self.assertEqual((job.clock, job.obs_render, job.render_every), ("free", "per-step", 1))
        for bad in ({"clock": "fast"}, {"obs_render": "never"}, {"render_every": 0}):
            with self.assertRaises(ValueError):
                JobSpec(name="j", task="t", **bad)

    def test_the_command_line_reaches_the_job(self) -> None:
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "scripts" / "bench_run.py"
        spec = importlib.util.spec_from_file_location("bench_run_for_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        argv = "--task s/v --solve graph --clock lockstep --obs-render on-demand --render-every 3"
        args = module.create_parser().parse_args(argv.split())
        job = module.job_from_args(args)
        self.assertEqual((job.clock, job.obs_render, job.render_every), ("lockstep", "on-demand", 3))
        defaults = module.job_from_args(module.create_parser().parse_args(["--task", "s/v"]))
        self.assertEqual((defaults.clock, defaults.obs_render, defaults.render_every), ("free", "per-step", 1))

    def test_provenance_says_what_ran(self) -> None:
        from robosimstudio.specs import JobSpec
        from robosimstudio_bench.run.jobs import FLEET_RUNTIME, runtime_provenance

        job = JobSpec(name="j", task="t", clock="lockstep", obs_render="on-demand", render_every=2)
        self.assertEqual(runtime_provenance(job), {"clock": "lockstep", "obs_render": "on-demand", "render_every": 2})
        self.assertEqual(FLEET_RUNTIME["clock"], "fleet-barrier")

    def test_video_renders_at_the_videos_own_cadence(self) -> None:
        from robosimstudio_bench.run.jobs import video_cadence

        self.assertEqual(video_cadence(60.0, 30), 2)
        self.assertEqual(video_cadence(60.0, 60), 1)
        self.assertEqual(video_cadence(50.0, 30), 1)
        self.assertEqual(video_cadence(60.0, 30, record_every=4), 4)
        self.assertEqual(video_cadence(240.0, 30, record_every=3), 6)

    def test_apply_runtime_configures_the_policy_and_the_video(self) -> None:
        from robosimstudio.specs import JobSpec, RecordSpec
        from robosimstudio_bench.run.jobs import apply_runtime

        seen: dict[str, Any] = {}
        policy = types.SimpleNamespace(configure_runtime=lambda **kw: seen.update(kw))
        env = types.SimpleNamespace(pipeline=types.SimpleNamespace(control_hz=60.0), video_every=0)
        job = JobSpec(name="j", task="t", clock="lockstep", obs_render="on-demand", record=RecordSpec(enabled=True))
        apply_runtime(job, env, policy)
        self.assertEqual(seen, {"clock": "lockstep", "obs_render": "on-demand"})
        self.assertEqual(env.video_every, 2)
        quiet = types.SimpleNamespace(pipeline=env.pipeline, video_every=0)
        apply_runtime(JobSpec(name="j", task="t", record=RecordSpec(enabled=False)), quiet, policy)
        self.assertEqual(quiet.video_every, 0, "no recording, no video renders")

    def test_a_policy_that_cannot_ask_for_a_render_is_refused_on_demand(self) -> None:
        from robosimstudio_bench.solutions.base import SolutionIdle

        policy = SolutionIdle(env=None)
        policy.configure_runtime(clock="lockstep", obs_render="per-step")  # in lockstep by construction
        with self.assertRaises(ValueError):
            policy.configure_runtime(obs_render="on-demand")

    def test_on_demand_turns_per_step_rendering_off_and_close_puts_it_back(self) -> None:
        from robosimstudio_policy_agent.graph_solution import SolutionGraph

        env = types.SimpleNamespace(renderer=object(), obs_renderer=None, obs_render_enabled=True)
        policy = SolutionGraph.__new__(SolutionGraph)
        policy.env = env
        policy._declares_on_demand = False
        policy._on_demand_observations = False
        policy._restore_obs_render = None
        policy.configure_runtime(clock="lockstep", obs_render="on-demand")
        self.assertEqual(policy.clock, "lockstep")
        self.assertTrue(policy._on_demand_observations, "the connector hides get_observation from the poller too")
        self.assertFalse(env.obs_render_enabled)
        policy.configure_runtime(obs_render="per-step")
        self.assertTrue(env.obs_render_enabled)
        with self.assertRaises(ValueError):
            policy.configure_runtime(clock="fast")

    def test_an_on_demand_video_frame_goes_where_the_recorder_reads(self) -> None:
        from robosimstudio_bench.envs.env import Env

        class Rig:
            def __init__(self, tag: str) -> None:
                self.tag = tag

            def render(self, state) -> dict:
                return {"cam": {"rgb": self.tag}}

        one = types.SimpleNamespace(renderer=Rig("video"), obs_renderer=None, state=None, frames={}, video_frames={})
        Env._render_video(one)
        self.assertEqual(one.frames, {"cam": {"rgb": "video"}}, "one rig serves both: obs['cameras'] carries it")
        two = types.SimpleNamespace(
            renderer=Rig("video"), obs_renderer=Rig("obs"), state=None, frames={"cam": "kept"}, video_frames={}
        )
        Env._render_video(two)
        self.assertEqual(two.video_frames, {"cam": {"rgb": "video"}})
        self.assertEqual(two.frames, {"cam": "kept"}, "a video frame never stands in for an observation")


if __name__ == "__main__":
    unittest.main()
