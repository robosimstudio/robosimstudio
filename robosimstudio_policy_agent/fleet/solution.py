# SPDX-License-Identifier: Apache-2.0
"""A graph runtime per world, all of them serviced by one stepping loop.

:class:`~robosimstudio_policy_agent.graph_solution.SolutionGraph` gives one
environment one graph: a thread running ``gap.execute``, a bus, a connector, and
a rollout that pumps the bus once per step. This is the same arrangement ``W``
times over -- ``W`` threads, ``W`` connectors, ``W`` per-world buses -- with one
fleet loop underneath them all.

Almost nothing about the *graph* changes, which is the point. Each world gets its
own connector built by the same :func:`~robosimstudio_policy_agent.connector.make_connector`,
carrying the same ``arm_id`` forwarding and the same servo overrides a graph
depends on; each world's node scripts read and command through a facade bound to
that world; and worlds are invisible above the tool boundary. What changes is
underneath: a motion is submitted once and run on device rather than marshalled a
step at a time, so a wave's threads spend their time parked rather than taking
turns with the simulation.

Three things this class owns that the single-world policy also owns, and for the
same reasons:

* **A fresh connector per world per episode.** The runtime caches its tool
  registry and drains skill-bundle registrations into it on every ``execute``,
  so reusing one across episodes fails on the second with a name collision -- at
  graph startup, which reads like a broken scene rather than a stale registry.
  Per world it is also what isolates suite tools' per-episode state.
* **A settle countdown before termination.** A graph finishing is not the episode
  ending: the last motion is still settling, and the evaluator gets those steps
  to see it.
* **The trace.** Per world, into that world's *episode* directory, so a batched
  run's ``traces/`` reads exactly as a serial run's does.
"""

from __future__ import annotations

import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from robosimstudio.arms import arm_order
from robosimstudio.core.bus import BusTimeout
from robosimstudio_bench.envs.world_view import WorldView
from robosimstudio_bench.solutions.fleet import FleetSolution

from ..graphs import stage_graph
from ..perception_tools import SEGMENT, segment_many
from ..skills import pinned_registries
from .barrier import Barrier, ParkTable, Rendezvous, StallTimeout
from .ik_backend import IKBackendFleet, solve_group
from .intents import IntentTable
from .obs_format import RENDER
from .snapshots import SnapshotTable
from .stepper import StepService
from .world_bus import ServiceQueue, StepWaiter, WorldBus
from .world_facade import WorldFacade

__all__ = ["SolutionGraphBatched"]


class _FleetServo:
    """What the connector's pose verbs call, turned into a batched motion.

    The connector overrides ``go_to_pose`` and ``go_to_pose_cartesian`` to drive
    a servo rather than solve-once-and-lunge, because the reference benchmark's
    motion is the one the tasks were authored against. That override is worth
    keeping verbatim; what it drives changes. Here ``move`` submits a
    :class:`~robosimstudio_policy_agent.fleet.intents.ServoToPose` and waits,
    and the feedback loop it used to run per step happens on device for every
    servoing world at once.

    **The signature and the return value are the single-world servo's**
    (:meth:`~robosimstudio_policy_agent.servo.CartesianServo.move`), keyword
    for keyword. The connector's verbs are written against that servo and are
    not told which one they hold: ``move_cartesian_until_contact`` passes
    ``stop_on_stall``, ``go_to_pose`` reads ``outcome`` off what comes back.
    A narrower signature here turned the first into a ``TypeError`` in every
    world of every batched run, and a ``None`` return made the second report
    ``reached: False`` on every move that physically arrived.
    ```tests/test_fleet_servo_contract.py``` holds the two to each other.

    Args:
        facade: This world's :class:`~robosimstudio_policy_agent.fleet.world_facade.WorldFacade`.
        honours_roll: Whether this arm's solver closes the wrist roll, so
            arrival is measured the way the move was solved -- the roll-free
            solver aligns the approach axis and is tested on that alone.
    """

    def __init__(self, facade, *, honours_roll: bool = False) -> None:
        self.facade = facade
        self.honours_roll = bool(honours_roll)

    def move(
        self,
        pose: dict,
        *,
        arm_id: int = 0,
        max_steps: int = 0,
        speed_scale: float = 1.0,
        pos_tolerance: float | None = None,
        stop_on_stall: bool = False,
        stall_steps: int = 12,
        stall_epsilon: float = 0.0002,
    ) -> dict[str, float | str]:
        from robosimstudio.core.math import quat_wxyz_to_xyzw
        from robosimstudio.solvers.ik.solver import POS_TOLERANCE, ROT_TOLERANCE_DEG

        position = np.array(
            [float(pose["position"]["x"]), float(pose["position"]["y"]), float(pose["position"]["z"])],
            dtype=np.float64,
        )
        rotation = quat_wxyz_to_xyzw([pose["rotation"][k] for k in ("w", "x", "y", "z")])
        # Both tolerances the fleet CAN honour, and both were being dropped.
        #
        # ``pos_tolerance``: ServoToPose carried the field and the stepper read
        # it; it was simply never plumbed, so a fine leg asking for 1.5 mm
        # silently got the 4 mm default.
        #
        # ``speed_scale``: this used to warn once per process and run at full
        # speed, on the argument that raising would turn tool hanging's graph
        # into an instant fleet failure. Both halves of that were right and the
        # conclusion was not -- a run that produced a number at full speed where
        # the graph asked for 0.40, with nothing in the row saying so, is the
        # hidden divergence a batched run must not have. The stepper honours it
        # now; see :meth:`StepService._lead`.
        scale = float(speed_scale)
        if not 0.0 < scale <= 1.0:
            raise ValueError(f"speed_scale must be in (0, 1], got {speed_scale}")
        result = self.facade.servo_to_pose(
            position,
            rotation,
            arm_id=arm_id,
            max_steps=max_steps,
            pos_tolerance=POS_TOLERANCE if pos_tolerance is None else float(pos_tolerance),
            rot_tolerance=float(np.radians(ROT_TOLERANCE_DEG)),
            honours_roll=self.honours_roll,
            speed_scale=scale,
            stop_on_stall=bool(stop_on_stall),
            stall_steps=int(stall_steps),
            stall_epsilon=float(stall_epsilon),
        )
        if not isinstance(result, dict):
            # A motion superseded or cancelled before it could end lands here;
            # the connector reads a dict, so hand it the one the serial servo
            # hands over when nothing arrived.
            return {"outcome": "timeout", "remaining_distance": -1.0, "steps": 0}
        return result


class _World:
    """One world's graph runtime and everything bound to it."""

    __slots__ = (
        "bus",
        "connector",
        "error",
        "facade",
        "result",
        "settle_left",
        "spans",
        "thread",
        "trace",
        "trace_dir",
        "view",
    )

    def __init__(self) -> None:
        self.bus: WorldBus | None = None
        self.facade: WorldFacade | None = None
        self.view: WorldView | None = None
        """This world's env surface, kept so the driver can run the hooks a
        policy installs on it -- see :meth:`SolutionGraphBatched.on_step`."""
        self.connector: Any = None
        self.thread: threading.Thread | None = None
        self.result: Any = None
        self.error: BaseException | None = None
        self.settle_left: int = -1
        self.trace: dict[str, Any] = {}
        self.trace_dir: Path | None = None
        self.spans: Any = None
        """This world's tool lifecycle markers, or ``None`` when nothing captures.

        One per world rather than one for the wave: a span is a *graph's* tool
        beginning and ending, and twelve graphs run at once. Sharing would
        interleave twelve episodes' calls into one tape and resolve each span
        against whichever world happened to be between them."""

    @property
    def finished(self) -> bool:
        return self.thread is not None and not self.thread.is_alive()


class SolutionGraphBatched(FleetSolution):
    """Runs one graph per world against a batched fleet.

    Args:
        fleet: The batched environment.
        graph: Which graph to run -- a task's shipped graph name, or a path.
        perception: Which body a graph's perception subgraphs should run, as the
            single-world policy means it. Handed to the graph as an input, and a
            graph that routes on it -- every peel variant does -- fails on its
            first node without one.
        privileged: Register the ground-truth ``sim.*`` tools.
        suite_tools: Register the suite's own tools beside them. False
            withholds a task's scripted plans while keeping the generic set --
            see :func:`~robosimstudio_policy_agent.connector.make_connector`.
        settle_steps: Control steps between a world's graph finishing and its
            episode being asked to end.
        trace_dir: Where per-episode traces go. A scratch directory when omitted,
            for the reason the single-world policy documents: a shipped graph is
            package source, and writing traces beside it fills the installed tree.
        cadence: Control steps between the fleet's readbacks. Zero takes the
            fleet's own ``eval_every``, which is what the runner paces its
            ticks by -- the two have to agree, because a capped servo advances
            its setpoint once per readback for every step the readback stands
            for, and a settle counts down by the same number.
        bus_timeout: How long a node waits on a motion or a marshalled call [s].
        motion: ``"osc"`` or ``"plan"``, as the single-world policy means them.
    """

    def __init__(
        self,
        fleet,
        *,
        graph: str = "gt",
        perception: str = "oracle",
        privileged: bool = True,
        suite_tools: bool = True,
        settle_steps: int = 60,
        trace_dir: str = "",
        cadence: int = 0,
        bus_timeout: float = 120.0,
        motion: str = "",
        roll: str = "auto",
        skills: Any = None,
        stall_timeout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(fleet, **kwargs)
        self.graph = graph
        self.perception = str(perception or "")
        """The ``perception`` workflow input, as the single-world policy passes it.
        Which body a graph's perception subgraphs run -- see the single-world
        policy's own attribute for what the values mean. It used to land in
        ``kwargs`` unread, so a graph with a perception router ran its oracle
        body on every world of a ``--perception vision`` run."""
        # The registry set every world executes with, resolved from the
        # checkout rather than the cwd -- see robosimstudio_policy_agent.skills.
        self.skills = pinned_registries(skills)
        self.privileged = bool(privileged)
        self.suite_tools = bool(suite_tools)
        self.settle_steps = int(settle_steps)
        self.bus_timeout = float(bus_timeout)
        self.motion = motion
        self.roll = str(roll)
        """The roll contract, as :class:`~robosimstudio_policy_agent.ik.IKBackend` takes it."""
        self.num_worlds = int(fleet.num_worlds)

        # A scratch directory is this policy's own and :meth:`close` removes it;
        # one the caller named is theirs and is kept.
        self._scratch_trace_dir: str | None = None
        if not trace_dir:
            import tempfile

            trace_dir = self._scratch_trace_dir = tempfile.mkdtemp(prefix="robosimstudio-fleet-gap-")
        self.trace_dir = Path(trace_dir)
        self.graph_path = self._resolve_graph(fleet, graph, registries=self.skills)
        # Likewise a staged copy of the graph (one whose scripts resolve from
        # outside it): this policy's scratch, removed at close. An unstaged
        # graph is its source directory and is never touched.
        from ..graph_solution import SolutionGraph

        source = SolutionGraph._graph_source(fleet, graph)
        self._staged_graph_dir: Path | None = Path(self.graph_path) if Path(self.graph_path) != Path(source) else None
        self.task_tools = self._task_tools(fleet)
        # Work arm first, as a runtime ``arm_id`` counts. Resolved once for the
        # fleet: role order is a fact about the pairing, not about a world.
        self.arm_names = list(arm_order(fleet))

        self.parks = ParkTable(self.num_worlds)
        self.rendezvous = Rendezvous(parks=self.parks)
        self.intents = IntentTable(num_worlds=self.num_worlds, num_arms=max(1, len(self.arm_names)), parks=self.parks)
        self.snapshots = SnapshotTable(self.num_worlds, handles=fleet.handles)
        self.service = ServiceQueue(parks=self.parks)
        self.waiter = StepWaiter(parks=self.parks)
        self.stepper = StepService(
            fleet,
            intents=self.intents,
            snapshots=self.snapshots,
            service=self.service,
            waiter=self.waiter,
            arm_names=self.arm_names,
            cadence=int(cadence) or int(getattr(fleet, "eval_every", 1) or 1),
        )
        self.worlds: list[_World] = [_World() for _ in range(self.num_worlds)]
        self._requests: dict[int, str] = {}
        self._live = np.zeros(self.num_worlds, dtype=bool)
        self._stalled: set[int] = set()
        # What this wave answers as a group rather than one world at a time.
        # Registered here, on the driver, rather than by each world's backend:
        # the rendezvous is the fleet's and a world should not be able to decide
        # what the fleet batches.
        for kind in (IKBackendFleet.SOLVE, IKBackendFleet.PLAN_LINEAR, IKBackendFleet.PLAN_JOINT):
            self.rendezvous.register(kind, solve_group)
        if getattr(fleet, "cameras", ()):
            self.rendezvous.register(RENDER, self._render_group)
            self.rendezvous.register(SEGMENT, self._segment_group)
        self.barrier = Barrier(
            self.parks,
            expect=self._expect_parked,
            rendezvous=self.rendezvous,
            service=self.service,
            on_stall=self._retire_stalled,
            timeout=stall_timeout or self.bus_timeout,
        )

    # ------------------------------------------------------------------ set-up
    @staticmethod
    def _resolve_graph(fleet, graph: str, *, registries: Any = None) -> str:
        """A graph name to a directory, the way the single-world policy does it.

        Literally the same way: both call ``graph_dir``. And both hand the result
        to :func:`~robosimstudio_policy_agent.graphs.stage_graph`, which is where a script
        the graph does not ship resolves from the bundle its subgraph's ``skill``
        names, else from its collection's ``_shared/scripts/``, else from
        ``robosimstudio_policy_agent/nodes/``. A name that is neither shipped nor
        an existing directory raises there too, rather than scoring every world
        zero on a workflow that cannot load.
        """
        from ..graph_solution import SolutionGraph

        return stage_graph(SolutionGraph._graph_source(fleet, graph), registries=registries)

    @staticmethod
    def _task_tools(fleet):
        """The suite's own tool registrar, ``(registry, env, bus) -> None``.

        Named ``tools`` on the task spec, the same attribute the single-world
        policy reads. A suite's tools are how a graph asks the task what it is
        supposed to be doing next, so a missing registrar is not a degraded run:
        every node that routes on one fails immediately.
        """
        return getattr(fleet.task, "tools", None)

    # --------------------------------------------------------------- lifecycle
    def reset_wave(self, slots: list) -> None:
        """Start a graph for every assigned world.

        Every world is torn down first, then rebuilt: a bus that has been closed
        cannot be reused, and a connector cannot outlive an episode.
        """
        super().reset_wave(slots)
        self.close_worlds()
        self.waiter.reopen()
        self.rendezvous.reopen()
        self.parks.reset()
        self.stepper.reset()
        self._stalled = set()
        self._requests = {}
        self._live = np.array([not slot.parked for slot in slots], dtype=bool)
        self.snapshots.publish(
            self.fleet.snapshot(), infos=[{} for _ in range(self.num_worlds)], obs=self._observations()
        )
        for slot in slots:
            if slot.parked:
                continue
            self._begin_world(slot)

    def _begin_world(self, slot) -> None:
        world = int(slot.world)
        state = self.worlds[world]
        state.bus = WorldBus(
            world,
            snapshots=self.snapshots,
            service=self.service,
            waiter=self.waiter,
            timeout=self.bus_timeout,
            parks=self.parks,
            rendezvous=self.rendezvous,
        )
        state.facade = WorldFacade(
            world,
            fleet=self.fleet,
            bus=state.bus,
            intents=self.intents,
            arm_names=self.arm_names,
            max_steps=self.fleet.max_steps,
        )
        state.spans = self._new_spans(state)
        state.spans.reset(int(slot.episode.episode_id))
        state.connector = self._new_connector(state)
        # Build the tool registry here, on this thread, before the graph thread
        # exists. The runtime's ``tool_registry`` is a lazy property, so leaving
        # it unbuilt makes ``gap.execute`` its first accessor -- and registering
        # the privileged tools reads device state directly, unmarshalled
        # (``SimTools._compute_home_ee_quat``, whose own docstring states the
        # invariant: "constructed on the main thread before the graph starts").
        # The single-world policy pre-builds it in its ``reset`` for the same
        # reason; here it also keeps W graph threads off Warp while the fleet
        # is capturing its control stage, where a touch fails the capture.
        _ = state.connector.tool_registry
        state.trace_dir = self.trace_dir / f"{slot.episode.episode_id:04d}"
        state.result = state.error = None
        state.settle_left = -1
        state.trace = {}
        state.thread = threading.Thread(
            target=self._run_graph,
            args=(world, slot.episode.instruction),
            name=f"gap-graph-w{world}",
            daemon=True,
        )
        state.thread.start()

    def _new_spans(self, state: _World):
        """This world's tool span recorder.

        Always built, as the single-world graph policy builds its one: the rows
        are a list and cost nothing until something asks for them, and a run
        that turns capture on mid-flight would otherwise have nothing to hand
        over for the episodes already running.

        Stepped by this world's own bus, so a marker lands on the wave step the
        tool actually ran at -- which is what makes a span resolve to endpoints
        the tape has points for.
        """
        from robosimstudio.record.capture import ToolSpanRecorder

        return ToolSpanRecorder(lambda bus=state.bus: bus.snapshot.step)

    def raw_spans(self, world: int) -> list:
        """One world's tool lifecycle rows, for the recorder. Empty when not capturing."""
        state = self.worlds[int(world)] if 0 <= int(world) < self.num_worlds else None
        recorder = getattr(state, "spans", None) if state is not None else None
        return list(recorder.rows()) if recorder is not None else []

    def _new_connector(self, state: _World):
        """This world's connector: the shipped one, over this world's facade."""
        from ..connector import make_connector

        world = state.facade.world
        view = WorldView(self.fleet, world)
        state.view = view
        # The backend resolves the roll contract exactly as the single-world
        # connector does, and the fleet servo reads its answer.
        ik = IKBackendFleet(self.fleet, world, state.bus, roll=self.roll)
        honour_roll = bool(ik.honours_roll)
        # What the backend was told, so the servo intents this facade submits
        # solve the same orientation the boundary solves do.
        state.facade.honours_roll = honour_roll
        return make_connector(
            view,
            bus=state.bus,
            privileged=self.privileged,
            task_tools=self.task_tools,
            suite_tools=self.suite_tools,
            motion=self.motion,
            facade=state.facade,
            ik=ik,
            # No servo: a Cartesian move is a ServoToPose motion the fleet runs
            # batched, installed by the connector's pose verbs through the
            # facade. Handing the single-world CartesianServo here would put a
            # host solve back on the step path, once per world per step.
            servo=_FleetServo(state.facade, honours_roll=honour_roll),
            skills=self.skills,
            capture_spans=state.spans,
        )

    def _run_graph(self, world: int, instruction: str) -> None:
        state = self.worlds[world]
        # This thread is the world, as far as the barrier is concerned. Said
        # here, from the thread itself and before anything runs on it, because
        # the graph runtime starts helper threads of its own once it is going --
        # an observation poller that calls the fleet on its own schedule -- and
        # the barrier must not wait for those or be released by them.
        self.parks.adopt(world)
        # The ``_World`` is reused by the next wave, and a thread that outlives
        # :meth:`close_worlds`' join must not write its end into that wave's
        # episode. Only the thread the world is still running records anything.
        me = threading.current_thread()
        try:
            from gap import execute

            inputs = {"instruction": instruction}
            if self.perception:
                inputs["perception"] = self.perception
            result = execute(
                self.graph_path,
                connector=state.connector,
                skills=self.skills or None,
                # Both inputs the runtime resolves ``$ref: in.*`` against. A
                # graph that declares no perception router never reads the
                # second, which is what makes it additive; a graph that does --
                # every peel variant -- died on its first node without it,
                # because a wave passed only the first. Built above rather than
                # inline so an unset perception is absent rather than empty.
                inputs=inputs,
                trace_dir=str(state.trace_dir) if state.trace_dir else None,
            )
            if state.thread is me:
                state.result = result
        except BaseException as exc:
            if state.thread is me:
                state.error = exc
                state.trace["traceback"] = traceback.format_exc()

    # -------------------------------------------------------------------- step
    def act(self, live: np.ndarray) -> None:
        """Hold the fleet until every live world is parked, then install their motions.

        The hold is why a batched episode lands where a serial one does, and it
        is the one thing this method does that the pre-barrier version did not.
        See :mod:`.barrier` for why a thinking world must stop the clock;
        :meth:`_expect_parked` and :meth:`_retire_stalled` are this class's half
        of it, and they are the only two things the loop needs to know about
        worlds.
        """
        self._live = np.asarray(live, dtype=bool)
        self.barrier.quiesce()
        self.stepper.install(self._live)

    def _expect_parked(self) -> np.ndarray:
        """Per world, whether it owes the barrier a park before the fleet may step.

        A world owes one when it is live, its graph thread is still running, and
        it has not been retired. A finished graph owes nothing -- its world is
        only being stepped out its settle -- and neither does a world the
        watchdog has given up on, which is what stops one wedged thread from
        holding a wave open forever.
        """
        expected = np.zeros(self.num_worlds, dtype=bool)
        for world in range(self.num_worlds):
            if not self._live[world] or world in self._stalled:
                continue
            thread = self.worlds[world].thread
            expected[world] = thread is not None and thread.is_alive()
        return expected

    def _retire_stalled(self, worlds: list[int]) -> None:
        """End the episodes of whichever worlds have held the barrier too long.

        The world, not the wave. A wedged thread is one episode's failure, and
        failing its twenty-three neighbours with it would throw away a whole
        wave's measurement for one bad node -- so this cancels that world's
        motions, asks the runner to close its episode, and takes it out of
        :meth:`_expect_parked` so the fleet can step again even if the thread
        never wakes.
        """
        for held in worlds:
            world = int(held)
            self._stalled.add(world)
            where = self._where(world)
            reason = f"held the fleet for {self.barrier.watch.timeout:.0f} s without parking{where}"
            print(f"[fleet] world {world} {reason}; retiring it and stepping on")
            self.stepper.retire(world, reason)
            self.rendezvous.drop_world(world, StallTimeout(f"world {world} {reason}"))
            self._requests[world] = f"stalled: {reason}"

    def _where(self, world: int) -> str:
        """Where a stalled world's thread actually is, for the message.

        A stall is the one failure this design adds, so it should not be
        reported as a bare timeout: the frame names the script, and the script
        names the node, which is what turns "the wave hung" into "this node
        blocks on something the fleet does not know about".
        """
        import sys
        import traceback

        thread = self.worlds[world].thread
        ident = getattr(thread, "ident", None)
        frame = sys._current_frames().get(ident) if ident is not None else None
        if frame is None:
            return ""
        stack = traceback.extract_stack(frame)
        return f" (in {stack[-1].filename.rsplit('/', 1)[-1]}:{stack[-1].lineno} {stack[-1].name})" if stack else ""

    def on_step(self, snapshot) -> None:
        """Publish, retire finished motions, and count down the settles.

        ``snapshot`` is ``None`` off cadence ticks, and nothing happens then --
        which is the whole economy: between ticks the fleet steps and the host
        does not look.
        """
        self.service.pump()
        if snapshot is None:
            return
        # Each live world's after-the-step hooks, the way ``Env.step`` and the
        # scripted wave driver run them. A hook is something a policy installed
        # mid-episode and expects once per control step with its env -- a grasp
        # pin re-asserted against the pad, which is the only thing holding a
        # sponge the fingers do not grip. No driver ran them for a graph wave,
        # so a privileged graph's ``sim.attach`` latched once and let go.
        for world in range(self.num_worlds):
            view = self.worlds[world].view
            if not self._live[world] or view is None:
                continue
            for hook in list(view.post_step_hooks):
                hook(view)
        self.stepper.settle(snapshot, infos=self._infos(snapshot))
        for world in range(self.num_worlds):
            if not self._live[world]:
                continue
            state = self.worlds[world]
            if state.thread is None:
                continue
            if state.finished and state.settle_left < 0:
                state.settle_left = self.settle_steps
                self._record_result(world)
            if state.settle_left >= 0:
                # Ticks arrive every ``eval_every`` control steps, so that is
                # what one tick counts down by; the policy's own cadence is 1.
                state.settle_left -= max(1, int(getattr(self.fleet, "eval_every", 1)))
                if state.settle_left < 0:
                    reason = "graph finished" if state.error is None else f"graph failed: {state.error}"
                    self._requests[world] = reason

    def termination_requests(self) -> dict[int, str]:
        return dict(self._requests)

    def retire_world(self, world: int, reason: str = "the episode ended") -> None:
        """Take a scored world out of service, the moment the runner scores it.

        What the serial path gets for free when its rollout stops stepping: the
        bus closes, the graph thread's next call raises, and the graph ends.
        Here the fleet keeps stepping for the other worlds, and a finished
        world that is not retired keeps running -- its thread off the barrier
        (a finished slot owes it no park), its installed motion still driving
        the arm, its calls still serviced -- against a fleet it no longer
        belongs to. Cancels its motions, closes its facade and bus, and drops
        its calls, so whatever it was in the middle of ends with the episode.
        """
        world = int(world)
        state = self.worlds[world] if 0 <= world < self.num_worlds else None
        if state is None or state.thread is None:
            return
        self.stepper.retire(world, reason)
        if state.facade is not None:
            state.facade.close()
        elif state.bus is not None:
            state.bus.close()

    def _render_group(self, asks: list[tuple[int, Any]]) -> list[Any]:
        """Photograph every world that asked, in one go. Fleet thread, at the barrier.

        This is the case the rendezvous exists for. A render is the most
        expensive thing a node can ask for and the one most likely to be asked
        for by every world at once -- a wave of the same graph reaches its
        perception node together -- so answering them one at a time would pay
        the rig's cost per world *and* serialise the wave behind it.

        Only the worlds that asked: the tiled tracer covers the fleet in one
        launch either way, but the path tracer is a pass per world and a wave
        where three worlds out of sixteen are looking should pay for three.
        """
        worlds = [world for world, _payload in asks]
        frames = self.fleet.render_worlds(worlds)
        return [frames[world] if world < len(frames) else {} for world in worlds]

    @staticmethod
    def _segment_group(asks: list[tuple[int, Any]]) -> list[Any]:
        """Segment for every world that is looking, in one forward pass.

        The other half of the render group, and the more expensive half. A wave
        running one graph reaches its perception node together, so this is asked
        by every world at once -- and answered one at a time it would hold the
        barrier for the model's full cost, once per world.

        Nothing about a world's answer changes: :func:`segment_many` is the one
        implementation and a single call is the batch of one.
        """
        return segment_many([payload for _world, payload in asks])

    def _observations(self) -> list[dict]:
        from .obs_format import world_observations

        return world_observations(self.fleet, self.fleet.snapshot())

    def _infos(self, snapshot) -> list[dict]:
        """Per world, what a graph's ``sim.check_success`` should answer from.

        Read off each world's slot, which the runner has just scored -- the
        evaluator is authoritative here exactly as it is in a serial run, and a
        graph may read its verdict but not write one.
        """
        infos = []
        for world in range(self.num_worlds):
            slot = self.slots[world] if world < len(self.slots) else None
            info = dict(getattr(slot, "info", {}) or {})
            info.setdefault("success", bool(getattr(slot, "success", False)))
            info.setdefault("reward", float(getattr(slot, "reward", 0.0)))
            infos.append(info)
        return infos

    # ------------------------------------------------------------------- trace
    def _record_result(self, world: int) -> None:
        """Fill in a world's trace the moment its graph stops running."""
        from ..graph_solution import _summarise

        state = self.worlds[world]
        slot = self.slots[world] if world < len(self.slots) else None
        state.trace["graph_result"] = _summarise(state.result)
        state.trace["steps"] = int(self.stepper.step)
        if state.error is not None:
            state.trace["error"] = f"{type(state.error).__name__}: {state.error}"
        failure = getattr(state.result, "error", None)
        if failure is not None and bool(getattr(slot, "success", False)):
            # Not a failure: the evaluator declared success and the world's
            # episode ended under whatever node was still running. The env is
            # authoritative, so this is a cancellation -- calling it an error
            # would make every successful run look broken.
            state.trace["graph_result"] = {
                **state.trace.get("graph_result", {}),
                "cancelled_at_episode_end": f"{type(failure).__name__}: {failure}",
            }
        elif failure is not None:
            state.trace["graph_error"] = f"{type(failure).__name__}: {failure}"

    def write_trace(self, world: int) -> Path | None:
        """Write one world's ``graph_result.json`` beside the runtime's trace."""
        import json

        state = self.worlds[world]
        if not state.trace or state.trace_dir is None:
            return None
        state.trace_dir.mkdir(parents=True, exist_ok=True)
        path = state.trace_dir / "graph_result.json"
        path.write_text(json.dumps(state.trace, indent=2, default=str))
        return path

    def provenance(self) -> dict[str, Any]:
        """What the barrier cost this run, and whom it gave up on.

        ``held_s`` beside the summary's ``utilization`` is the pair worth
        reading: utilization says how much of the fleet was working, this says
        how much of the wave was nobody working at all because one world was
        thinking. A wave whose worlds are in step spends almost nothing here; a
        wave that has diverged spends most of itself here, and that is the
        number that says whether widening it further is worth anything.
        """
        return {**self.barrier.stats(), "stalled_worlds": sorted(self._stalled)}

    def row_extras(self, world: int) -> dict[str, Any]:
        state = self.worlds[world]
        extras: dict[str, Any] = {}
        if state.trace.get("graph_error"):
            extras["graph_error"] = str(state.trace["graph_error"])[:400]
        # How the world's steps were spent: moving, or holding still while its
        # graph thought. A serial episode's length is nearly all motion; a
        # batched one keeps stepping through every SAM3 call and plan, so the
        # two lengths are not the same number and this is what makes them
        # comparable (idle = the fleet's step count at this tick minus busy).
        busy = getattr(self.stepper, "busy_steps", None)
        if busy is not None and 0 <= world < len(busy):
            extras["busy_steps"] = int(busy[world])
            extras["idle_steps"] = int(max(0, int(self.stepper.step) - int(busy[world])))
        if world in self._stalled:
            # A stall is a property of *this* episode and belongs in its row.
            # Left only in the wave's provenance it would read as a fleet-wide
            # problem, when what happened is that one node blocked on something
            # the barrier could not see.
            extras["fleet_stalled"] = True
        return extras

    def status(self, world: int) -> str:
        state = self.worlds[world]
        if state.thread is None:
            return "idle"
        if not state.finished:
            return "running"
        return "failed" if state.error is not None else "finished"

    # ------------------------------------------------------------------- close
    def close_worlds(self) -> None:
        """Release every world's graph thread and write what it produced.

        Everything that can be blocking a graph thread is released *first*, for
        the whole fleet, before any thread is joined. A thread parked on the
        shared step waiter cannot be woken by its own world's bus closing, so
        joining first means waiting out one timeout per world for threads that
        were never going to wake.
        """
        # Everything that can be holding a graph thread is released first, and
        # the park table is emptied with them: a thread released by a close is
        # running, not parked, and a stale park would let the next wave's
        # barrier close over a thread that is still winding down.
        self.intents.cancel_all(BusTimeout("the wave ended"))
        self.rendezvous.close()
        self.waiter.close()
        for world, state in enumerate(self.worlds):
            if state.thread is None:
                continue
            self.stepper.retire(world, "the wave ended")
            if state.facade is not None:
                # The facade, not just the bus: a thread that outlives the join
                # below must find its motions refused, not queued for the next
                # wave's world of the same index.
                state.facade.close()
            elif state.bus is not None:
                state.bus.close()
            state.thread.join(timeout=5.0)
            if not state.trace:
                self._record_result(world)
            self.write_trace(world)
            state.thread = None
            state.connector = None

    def close(self) -> None:
        self.close_worlds()
        self.intents.cancel_all(BusTimeout("the run closed"))
        self.rendezvous.close()
        self.service.close()
        self.waiter.close()
        import shutil

        scratch = getattr(self, "_scratch_trace_dir", None)
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        staged = getattr(self, "_staged_graph_dir", None)
        if staged is not None:
            shutil.rmtree(staged, ignore_errors=True)
            self._staged_graph_dir = None
