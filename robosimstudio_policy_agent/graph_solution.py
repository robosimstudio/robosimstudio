# SPDX-License-Identifier: Apache-2.0
"""Running a graph as a policy, in process.

The graph runs on its own thread and drives the environment through the
connector's tools; the environment's own loop stays where it is, calling
:meth:`SolutionGraph.get_action` once per control step. That call is what *pumps*
the bus, so the graph's pending requests are serviced between steps and the
simulation keeps a single writer.

Which means the action this policy returns is usually "hold": the graph is not
producing actions, it is producing tool calls, and the tools produce the motion.
The policy's job is to keep time moving and to notice when the graph is finished.

When it is, the policy asks for termination -- and only for termination. A graph
reporting success is a *claim*; the evaluator's verdict is the measurement, and
the settle countdown between the two exists because a graph typically declares
done at the moment it releases an object, before that object has landed.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from robosimstudio.core.bus import SimBus
from robosimstudio.record.capture import ToolSpanRecorder
from robosimstudio_bench.solutions.base import SolutionBase, solution_param

from .facade import GapEnvFacade
from .graphs import stage_graph, subgraph_skills
from .skills import pinned_registries


class _OnDemandConnector:
    """Expose the connector's tools without advertising a background poller.

    gap starts a 10 Hz observation stream when it discovers ``get_observation``
    on the connector object, and for a graph that reads on demand that stream is
    pure cost: on an RTX rig it path traces every camera ten times a second
    behind a policy that asked for nothing. Shadowing the NAME at class level
    hides it from that discovery while the tool registry keeps the connector's
    own bound method, so an explicit ``robot.get_observation`` still renders
    current RGB-D.

    Applied only when the workflow declares ``meta.observation_mode:
    on_demand``; everything else falls through ``__getattr__`` untouched.
    """

    get_observation = None

    def __init__(self, connector):
        self._connector = connector

    def __getattr__(self, name):
        return getattr(self._connector, name)


def _accepts_node_callbacks(execute: Any) -> bool:
    """Whether the installed ``gap.execute`` takes the per-node callbacks.

    A version question rather than a feature flag: they were added upstream
    after the rev this repository pins, and a runtime that does not have them
    raises ``TypeError`` before the first node runs. Read once per episode off
    the signature, which is cheap and cannot disagree with the callee.
    """
    import inspect

    try:
        params = inspect.signature(execute).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "on_node_start" in params and "on_node_end" in params


__all__ = ["SolutionGraph", "trace_at_episode_end"]


class SolutionGraph(SolutionBase):
    """Drives an environment with a GaP workflow, in process.

    Args:
        env: The environment.
        graph: Path to a workflow directory, or a graph name the task ships
            (``"gt"`` resolves through ``TaskSpec.graphs``).
        privileged: Register the ground-truth ``sim.*`` tools.
        settle_steps: Control steps to keep stepping after the graph finishes,
            so the evaluator scores a settled scene rather than one mid-release.
        pump_budget: Bus requests serviced per control step.
        bus_timeout: Seconds a graph worker waits for the owner thread to
            service a request; 0 keeps :class:`~robosimstudio.core.bus.SimBus`'s
            own default. The watchdog is *wall* clock, so it is sensitive to what
            else the machine is doing: a pool of environments sharing cores can
            leave a request unpumped for 30 s with nothing wrong, and the node
            error that follows would be recorded as the graph's failure rather
            than the scheduler's. Raise it when workers share cores.
        task_tools: ``(registry, env, bus) -> None`` for a suite's own tools.
            Defaults to the task's own ``TaskSpec.tools``, so a suite that
            ships privileged tools beside its graph gets them without every
            entry point having to know they exist.
        suite_tools: Whether those suite tools are registered at all. False
            withholds them while keeping the generic privileged set -- see
            :func:`~robosimstudio_policy_agent.connector.make_connector`.
        motion: ``"osc"`` (the default, when empty) or ``"plan"``. How the pose
            verbs move; see
            :func:`~robosimstudio_policy_agent.connector.make_connector`. A graph
            moving the reference's way by default is what makes its number
            comparable with the scripted routine's on the same scene.
    """

    def __init__(
        self,
        env,
        *,
        graph: str = "gt",
        privileged: bool = True,
        perception: str = "oracle",
        settle_steps: int = 60,
        pump_budget: int = 64,
        bus_timeout: float = 0.0,
        task_tools: Any = None,
        suite_tools: bool = True,
        skills: Any = None,
        trace_dir: str = "",
        roll: str = "auto",
        motion: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(env, **kwargs)
        # The registry set this run executes with, resolved from the checkout
        # rather than the cwd (robosimstudio_policy_agent.skills): the same list stages
        # the graph, decides which in-process stand-ins stand down, and goes to
        # gap.execute, so the three cannot disagree. An explicit ``skills`` or
        # $GAP_SKILLS_PATH overrides it wholesale, as gap means them.
        self.skills = pinned_registries(skills)
        self.source_graph = self._graph_source(env, graph)
        self.graph_path = stage_graph(self.source_graph, registries=self.skills)
        # A staged copy is a scratch directory of this policy's own, removed at
        # close as ``check_graph`` removes its; an unstaged graph is the source.
        self._staged_graph_dir = Path(self.graph_path) if Path(self.graph_path) != Path(self.source_graph) else None
        workflow_path = Path(self.graph_path) / "workflow.json"
        try:
            workflow_doc = json.loads(workflow_path.read_text())
            self._on_demand_observations = workflow_doc.get("meta", {}).get("observation_mode") == "on_demand"
            self._recording_skills = subgraph_skills(workflow_doc)
        except (OSError, ValueError, TypeError):
            self._recording_skills = {}
            self._on_demand_observations = False
        # A GRAPH THAT SAYS IT READS ON DEMAND STOPS PAYING FOR PER-STEP FRAMES.
        # Such a graph moves through blocking tools for hundreds of control
        # ticks between visual reads, and path tracing those intermediate states
        # is both wasteful and semantically misleading -- they are not
        # observations the policy consumed. The facade renders synchronously
        # inside ``get_observation`` (see ``_observation(refresh_cameras=True)``),
        # so every perception node still gets current RGB-D from the requested
        # backend while motion-only ticks do no render work.
        #
        # Driven by the workflow's own ``meta.observation_mode`` rather than by
        # "does this env have a renderer", which is what it was written as. The
        # difference matters: as written it would stale ``env.frames`` for EVERY
        # graph policy in the tree, and every recorded video with it -- a change
        # to eight other contributors' evidence, made from inside a cable
        # commit. The two graphs that want it declare it, and the same flag
        # already gates ``_OnDemandConnector`` below.
        self._restore_obs_render: bool | None = None
        """The env's own ``obs_render_enabled``, put back at close: the env
        outlives this policy for any caller that reuses it."""
        if self._on_demand_observations and env.renderer is not None:
            self._restore_obs_render = env.obs_render_enabled
            env.obs_render_enabled = False
        self.privileged = bool(privileged)
        self.perception = str(perception)
        """Which body a graph's perception subgraphs should run.

        Handed to the workflow as a top-level input, so a graph carrying both an
        ``oracle`` body (privileged ``sim.*`` ground truth) and a ``vision`` one
        (RGB-D through a detector) picks between them with a router rather than
        by shipping two copies of every unchanged node script. The value is only
        meaningful to a graph that declares such a router; one that does not
        simply ignores it.

        ``--perception vision --no-privileged`` is the pair that means something:
        it withholds the ground-truth tools *and* asks for the body that does not
        want them, which is the only combination that actually checks a graph
        runs on perception alone.
        """
        self.roll = str(roll or "auto")
        self.motion = str(motion)
        self.task_tools = task_tools if task_tools is not None else getattr(env.task, "tools", None)
        self.suite_tools = bool(suite_tools)
        # A caller's flag wins; otherwise the task's own document is asked, and
        # SimBus's default is the answer for a task that says nothing. Some
        # plans are legitimately expensive -- cloth_fold's clearance search is
        # documented at ~13 s a crease against a 30 s watchdog -- and the task
        # that knows that is better placed to say so than every person who ever
        # runs it. Without this the shirt graph scores 0.333 under
        # `--bus-timeout 180` and 0.000 without it, on the same code: the plan
        # node is killed mid-solve and the trace reads as a failed fold rather
        # than an unfinished one.
        self.bus_timeout = float(bus_timeout) or float(solution_param(env, "bus-timeout", 0.0) or 0.0)
        self.bus = self._new_bus()
        self.tool_spans = ToolSpanRecorder(lambda: self.bus.snapshot.step)
        self.connector = self._new_connector()
        self.settle_steps = int(settle_steps)
        self.pump_budget = int(pump_budget)
        # gap defaults its trace to the *workflow* directory, which for a
        # shipped graph is package source -- a run would write dag_trace.json
        # and node_data/ into the installed tree. Default to a scratch
        # directory instead; a caller that wants the trace kept passes one.
        # The scratch directory is this policy's own, and :meth:`close` removes
        # it; a directory the caller named is theirs and is left alone.
        self._scratch_trace_dir = None if trace_dir else tempfile.mkdtemp(prefix="robosimstudio-gap-")
        self.trace_dir = trace_dir or self._scratch_trace_dir
        self.episode_trace_dir: Path | None = None
        """Where *this* episode's trace goes -- a subdirectory per episode.

        The runtime writes ``dag_trace.json`` and ``node_data/`` into whatever
        directory it is handed, so handing it the same one every episode leaves
        a multi-episode run holding only the last episode's trace. It cannot be
        varied through ``solution_kwargs`` either: those are part of a pool
        worker's environment key, so a per-episode value would rebuild the scene
        between episodes. The policy numbers them itself.
        """

        self._thread: threading.Thread | None = None
        self._generation = 0
        """Which episode's graph thread may write results. Bumped on every
        reset, so a previous thread that outlived its join -- a node stuck in a
        call the closed bus could not release -- has its late writes dropped
        rather than landing in the next episode's result, error and trace."""
        self._result: Any = None
        self._error: BaseException | None = None
        self._settle_left = -1
        self.trace: dict[str, Any] = {}
        self._recording_node = ""

    def _new_bus(self) -> SimBus:
        """A bus, at this policy's watchdog setting."""
        return SimBus(timeout=self.bus_timeout) if self.bus_timeout > 0 else SimBus()

    def _new_connector(self):
        """A connector, and with it a fresh tool registry.

        The runtime's ``Connector`` caches its ``tool_registry``, and
        ``gap.execute`` *drains the skill bundles' registrations into it* on
        every call. Reusing one connector across episodes therefore fails on the
        second with a name collision (``'curobo.plan_to_grasp_poses' already
        registered``) -- and fails at graph startup, so the episode dies in a
        few dozen steps with no motion at all, which reads like a broken scene
        rather than a stale registry.
        """
        from .connector import make_connector

        return make_connector(
            self.env,
            bus=self.bus,
            privileged=self.privileged,
            task_tools=self.task_tools,
            suite_tools=self.suite_tools,
            roll=self.roll,
            motion=self.motion,
            capture_spans=self.tool_spans,
            skills=self.skills,
        )

    @staticmethod
    def _resolve_graph(env, graph: str, *, registries: Any = None) -> str:
        """A graph name to a directory the runtime can be handed.

        :meth:`_graph_source` finds the directory; this stages it through
        :func:`~robosimstudio_policy_agent.graphs.stage_graph` under *registries* -- a
        script the workflow names that is not beside the graph resolves from
        the bundle its subgraph's ``skill`` names, else from its collection's
        ``_shared/scripts/``, else from ``robosimstudio_policy_agent/nodes/``, and
        the runtime is handed a directory in
        which every script exists. The rule is stated once, there.
        """
        return stage_graph(SolutionGraph._graph_source(env, graph), registries=registries)

    @staticmethod
    def _graph_source(env, graph: str) -> str:
        """A graph name to its source directory: a task's shipped graph, or a path.

        A name the task ships resolves against its suite package. Anything else
        is passed through as a path -- but **only if that path exists**, which is
        the whole of this method's history.

        It used to pass anything through, and the failure that produced is the
        worst kind a benchmark can have: a mistyped or misremembered name became
        a nonexistent path, the workflow failed to load, and the run reported
        ``0/1 success (0%)`` -- indistinguishable at a glance from a policy that
        tried the task and lost. The reason to catch it here is that the mistake
        is *easy and invited*: every shipped graph is registered under the name
        ``gt`` while the directories differ, so ``cable_route/port_easy`` lives in
        ``graphs/weave`` and ``cloth_fold/shirt`` in ``graphs/gt_shirt``, and
        ``--graph weave`` reads like the obvious thing to type.

        ``check_graph_claims`` does not cover this. It refuses a *shipped* graph
        on a pairing whose binding does not claim it, which is a different
        mistake -- a name with no workflow behind it never reaches that check.

        This is the *source* directory -- what a run's provenance names and
        what the parity tooling compares -- not the staged one the runtime is
        handed; :meth:`_resolve_graph` is the pair.
        """
        try:
            relative = env.task.graph_path(graph)
        except KeyError as exc:
            from pathlib import Path

            if Path(graph).is_dir():
                return str(graph)
            shipped = [name for name, _ in getattr(env.task, "graphs", ())]
            raise ValueError(
                f"--graph {graph!r} is neither a graph {env.task.full_name!r} ships "
                f"({shipped or 'none'}) nor a directory that exists. Every shipped graph is "
                f"registered as its *name*, not its directory -- 'gt' is usually the answer even "
                f"where the folder is called something else. Refusing rather than running, "
                f"because a workflow that fails to load scores zero and reads like a task the "
                f"policy failed."
            ) from exc
        from robosimstudio_bench.solutions.base import graph_dir

        return graph_dir(env.task.suite, relative)

    # ---------------------------------------------------------------- lifecycle
    def reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        del obs
        # Release the previous episode's graph thread, then start clean: a new
        # bus (the old one is closed and cannot be reused) and a new connector
        # (see _new_connector).
        if self._thread is not None:
            self.bus.close()
            self._thread.join(timeout=5.0)
            self.bus = self._new_bus()
            self.connector = self._new_connector()

        # The episode that just ended, before its trace is cleared.
        self.write_trace()
        self._generation += 1
        self._result, self._error = None, None
        self._settle_left = -1
        self.trace = {}
        self.tool_spans.reset(self.env.episode.episode_id)
        self._recording_node = ""
        self.episode_trace_dir = Path(self.trace_dir) / f"{self.env.episode.episode_id:04d}"
        # Publish a first snapshot so a node that reads before the first step
        # sees the reset state rather than an empty one.
        self.connector.facade.publish_step(0.0, False, False, dict(info))
        # The registry is built here, on this thread, before the graph thread
        # exists, as the batched policy does: it is a lazy property, so leaving
        # it unbuilt makes ``gap.execute`` its first accessor, and registering
        # the privileged tools reads device state directly
        # (``SimTools._compute_home_ee_quat`` -- "constructed on the main
        # thread before the graph starts").
        _ = self.connector.tool_registry
        self._thread = threading.Thread(target=self._run_graph, args=(self._generation,), name="gap-graph", daemon=True)
        self._thread.start()

    def _run_graph(self, generation: int) -> None:
        """Run the graph; what it leaves behind is kept only while *generation* is current."""
        try:
            from gap import execute

            # Both go in as graph *inputs* (addressable as
            # ``{"$ref": "in.instruction"}`` and ``{"$ref": "in.perception"}``),
            # not as keywords: gap.execute takes graph, connector, skills,
            # inputs, trace_dir. A graph that declares no perception router
            # never reads the second one, which is what makes this additive --
            # every shipped graph runs unchanged.
            kwargs: dict[str, Any] = {
                "connector": (_OnDemandConnector(self.connector) if self._on_demand_observations else self.connector),
                # An empty pin means "nothing resolved"; gap reads None as
                # "resolve for yourself", which is the same answer said its way.
                "skills": self.skills or None,
                "inputs": {
                    "instruction": self.env.episode.instruction,
                    "perception": self.perception,
                },
                "trace_dir": str(self.episode_trace_dir) if self.episode_trace_dir else None,
            }
            # The node callbacks label a recording and nothing else, and they
            # arrived from a gap newer than the one this repository pins. Asked
            # of the signature rather than assumed, because handing them to the
            # pinned `execute` is a ``TypeError`` that kills the graph thread on
            # its first step -- which is every graph, not only the ones that
            # wanted the label.
            if _accepts_node_callbacks(execute):
                kwargs["on_node_start"] = lambda name, ok: self._on_graph_node_start(name, ok, generation)
                kwargs["on_node_end"] = lambda name, ok: self._on_graph_node_end(name, ok, generation)
            result = execute(self.graph_path, **kwargs)
            if generation == self._generation:
                self._result = result
        except BaseException as exc:
            if generation == self._generation:
                self._error = exc
                self.trace["traceback"] = traceback.format_exc()

    @property
    def finished(self) -> bool:
        return self._thread is not None and not self._thread.is_alive()

    def _on_graph_node_start(self, name: str, _success: Any, generation: int | None = None) -> None:
        if generation is None or generation == self._generation:
            self._recording_node = str(name)

    def _on_graph_node_end(self, name: str, _success: Any, generation: int | None = None) -> None:
        if (generation is None or generation == self._generation) and self._recording_node == str(name):
            self._recording_node = ""

    def recording_stage(self) -> str:
        """Exact active subgraph, skill, and node for video overlays."""
        node = self._recording_node
        if not node:
            return "Graph: planning" if not self.finished else "Graph: complete"
        scope, _, inner = node.partition(".")
        skill = self._recording_skills.get(scope, "")
        lines = [f"Subgraph: {scope}"]
        if skill:
            lines.append(f"Skill: {skill}")
        if inner:
            lines.append(f"Node: {inner}")
        return "\n".join(lines)

    # -------------------------------------------------------------------- step
    @property
    def control_mode(self) -> str:
        """``joint_position`` -- what this policy's *holds* are, always.

        The graph's motion happens inside ``robot.*`` tools on a worker thread,
        which select the joint controller themselves
        (:attr:`~robosimstudio_policy_agent.facade.GapEnvFacade.JOINT_MODE`). What
        :meth:`get_action` returns is the hold *between* those moves, and it is a
        joint vector -- so it has to be tagged, not left to whichever mode happens
        to be active. Untagged, a session that ran a Cartesian routine first would
        hold in ``osc_pose`` against a setpoint left over from the routine until
        the graph's first move switched modes: bounded by the lead clamp, and
        still the arm drifting for no reason anybody could see.
        """
        return GapEnvFacade.JOINT_MODE

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        """Pump the bus, then hold. The graph's tools produce the motion.

        "Hold" means re-issuing the last *commanded* joint targets, not the
        measured ones. The rollout steps once per iteration regardless of what
        the graph is doing, so an idle action here lands between the waypoints a
        node is streaming and relaxes the servo on every other step -- see
        :meth:`~robosimstudio_policy_agent.facade.GapEnvFacade.hold_action`.
        """
        del obs
        self.bus.pump(budget=self.pump_budget)

        if self.finished and self._settle_left < 0:
            self._settle_left = self.settle_steps
            self.trace["graph_result"] = _summarise(self._result)
            self.trace["steps"] = self.env.step_count
            if self._error is not None:
                self.trace["error"] = f"{type(self._error).__name__}: {self._error}"
            # gap.execute never raises on workflow failure -- the exception
            # lands in result.error. Surfacing its chained traceback is the
            # difference between "the graph failed" and knowing which node did
            # and why; the cause chain is where the real exception lives.
            failure = getattr(self._result, "error", None)
            if failure is not None and self.env.last_result.success:
                # Not a failure: the evaluator declared success and the
                # environment terminated, closing the bus under whatever
                # trailing node was still running (typically a go-home after
                # the object has already landed). The env is authoritative, so
                # this is a cancellation, and calling it an error would make
                # every successful run look broken.
                self.trace["graph_result"] = {
                    **self.trace.get("graph_result", {}),
                    "cancelled_at_episode_end": f"{type(failure).__name__}: {failure}",
                }
            elif failure is not None:
                self.trace["graph_error"] = f"{type(failure).__name__}: {failure}"
                self.trace["graph_traceback"] = "".join(
                    traceback.format_exception(type(failure), failure, failure.__traceback__)
                )

        if self._settle_left >= 0:
            self._settle_left -= 1
            if self._settle_left < 0:
                # A workflow that failed inside gap.execute never raises here;
                # its error is on the result. Reporting "graph finished" for
                # it made the recorded termination reason of an aborted
                # episode indistinguishable from a clean run.
                failure = self._error
                if failure is None and self.trace.get("graph_error"):
                    failure = self.trace["graph_error"]
                reason = "graph finished" if failure is None else f"graph failed: {failure}"
                self.env.request_termination(reason)

        return self.connector.facade.hold_action()

    def row_extras(self) -> dict[str, Any]:
        """What ended the graph, on the results row rather than only in a trace.

        `SolutionGraphBatched` has carried this since it was written and the
        single-world policy never did, so a measurement pass -- which does not
        record, and therefore never writes a trace file -- reported
        `close_kind: policy_request` with nothing beside it. Measured, that cost
        a six-agent audit to answer "which of three termination paths fired"
        for six conditions whose every episode ended that way; the string was
        sitting in `self.trace` the whole time.
        """
        extras: dict[str, Any] = {}
        if self.trace.get("graph_error"):
            extras["graph_error"] = str(self.trace["graph_error"])[:400]
        result = self.trace.get("graph_result")
        if isinstance(result, dict) and result.get("exit"):
            extras["graph_exit"] = str(result["exit"])[:120]
        return extras

    def on_step(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        """Publish the step so worker threads see fresh state, and hand the
        graph's trace to the environment's info so a recorder can write it."""
        del reward
        from robosimstudio.core.bus import SimSnapshot

        self.trace = trace_at_episode_end(
            self.trace,
            close_kind=str(info.get("close_kind", "")),
            finished=self.finished,
            steps=self.env.step_count,
        )
        if self.trace:
            info["graph_result"] = dict(self.trace)

        self.bus.publish(
            SimSnapshot(
                step=self.env.step_count,
                sim_time=self.env.sim_time,
                obs=obs,
                body_q=np.asarray(self.env.state.body_q.numpy(), dtype=np.float64),
                info=dict(info),
                frames=dict(self.env.frames),
            )
        )

    def write_trace(self) -> Path | None:
        """Write this episode's ``graph_result.json`` beside the runtime's trace.

        What makes a failed run diagnosable rather than merely unsuccessful, and
        written by the policy rather than by whoever called it so that a pooled
        run keeps it too: only the results *row* crosses a worker's queue, so an
        entry point reading ``info["graph_result"]`` gets nothing from a pool.

        Returns:
            The file, or ``None`` when there was no trace to write.
        """
        if not self.trace or self.episode_trace_dir is None:
            return None
        try:
            self.episode_trace_dir.mkdir(parents=True, exist_ok=True)
            path = self.episode_trace_dir / "graph_result.json"
            path.write_text(json.dumps(self.trace, indent=2, default=str))
            return path
        except OSError as exc:
            # A trace is evidence about a run, not the run. Losing it must not
            # take the episode with it.
            print(f"[graph] could not write the trace to {self.episode_trace_dir}: {exc}")
            return None

    def close(self) -> None:
        self.write_trace()
        self.bus.close()
        if self._thread is not None and self._thread.is_alive():
            # The bus close releases every waiting worker, so the graph thread
            # unwinds on its own; a daemon thread that does not is not worth
            # blocking a run over.
            self._thread.join(timeout=5.0)
        # Nothing reads the scratch traces once the policy is closed: nobody
        # was told where they are. A caller's own trace_dir is kept.
        self._generation = getattr(self, "_generation", 0) + 1
        scratch = getattr(self, "_scratch_trace_dir", None)
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
        staged = getattr(self, "_staged_graph_dir", None)
        if staged is not None:
            shutil.rmtree(staged, ignore_errors=True)
            self._staged_graph_dir = None
        restore = getattr(self, "_restore_obs_render", None)
        if restore is not None:
            env = self.env
            env.obs_render_enabled = restore
            self._restore_obs_render = None

    def raw_spans(self) -> list[dict[str, Any]]:
        """Tool begin/end markers for the recorder's raw capture tape."""
        return self.tool_spans.rows()

    @property
    def name(self) -> str:
        # The source directory, not the staged one: a run's provenance should
        # name the graph a reader can open, and a scratch path that is gone by
        # the time anyone reads run_meta.json names nothing.
        return f"graph:{self.source_graph}"


def trace_at_episode_end(
    trace: dict[str, Any],
    *,
    close_kind: str,
    finished: bool,
    steps: int,
) -> dict[str, Any]:
    """Complete the trace when the episode ends with the graph still running.

    A graph that is still executing when the episode ends never reaches the
    ``finished`` branch in :meth:`SolutionGraph.get_action`, which is where the
    trace is normally filled in -- and the episode that ends that way is the
    **successful** one. The evaluator is authoritative and terminates the moment
    the object is in its region, typically while the graph is still running its
    trailing go-home. So the one run whose trace is most worth having was the
    one run that recorded none: a directory of failures with traces beside a
    success with nothing.

    A trace that already has content is left alone -- the graph got there first
    and its own result is the better record.

    Args:
        trace: What the policy has recorded so far.
        close_kind: The environment's reason for ending, empty mid-episode.
        finished: Whether the graph thread has exited.
        steps: Control steps taken.

    Returns:
        The trace to report. The same object when there is nothing to add.
    """
    if trace or not close_kind or finished:
        return trace
    return {
        "graph_result": {"still_running_at_episode_end": True, "close_kind": close_kind},
        "steps": steps,
    }


def _summarise(result: Any) -> dict[str, Any]:
    """A JSON-safe view of whatever the runtime returned."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, list, dict))}
    return {"result": str(result)}
