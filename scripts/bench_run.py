# SPDX-License-Identifier: Apache-2.0
"""Run a policy over a task's episodes, and write a run directory.

The headless door: evaluation, data collection and graph runs are one command,
because they are one thing -- a policy driving episodes through the same
:func:`~robosimstudio_bench.run.rollout`, with the results written the same
way. What changes between them is which policy, how many episodes and whether
the episodes are kept, and none of those is a different program. The interactive
door is the studio: ``rss.view(env)``.

    # one of your tasks (a folder of .task files) and its graph, 16 episodes over four processes
    python scripts/bench_run.py --task-root my_tasks --task push/cube --solve graph --graph pick \\
        --episodes 0:16 --workers 4

    # the same sixteen episodes as sixteen worlds in one model
    python scripts/bench_run.py --task-root my_tasks --task push/cube --solve graph --graph pick \\
        --episodes 0:16 --num-worlds 16

    # without the simulator's own answers: no sim.* tool is registered
    python scripts/bench_run.py --task-root my_tasks --task push/cube --solve graph --graph pick --no-privileged

    # a job file's whole sweep
    python scripts/bench_run.py --jobs my_jobs.py --workers 4

**A pairing is a first-class part of a run.** ``--scene`` and ``--embodiment``
stage the task somewhere else or on another robot; both have to be pairings the
task *declares*, and ``Env`` raises with the admissible list otherwise. What
they do not do is imply the shipped policy still applies -- a graph is written
against one robot in one place, so running it elsewhere needs
``--allow-undeclared-graph`` and is recorded as the experiment it is.

**Serial or pooled is a scheduling choice, not a different measurement.**
``--workers 1`` (the default) runs the episodes in this process, which is where a
traceback is a traceback and a debugger works. Above one, they are spread over
worker processes -- one environment each, because a graph policy holds a runtime
thread and a bus per environment and replicated worlds in one model would step
them all at the pace of whichever was mid-plan; see
:mod:`robosimstudio_bench.run.pool`. Either way every rollout is the same
code and the report below is the same report.

Two things to know before reading the number it prints.

**Episode ids are not automatically distinct scenes.** Every shipped suite
declares ``EpisodeSpec(count=1, sampler=fixed)``, and ``make_episode`` keeps
sampling past ``count`` rather than refusing -- so ``--episodes 0:32`` on a
shipped variant is *32 trials of one layout*, which for a graph is a real
experiment (the runtime's thread timing is not pinned, so the rollouts genuinely
differ) but measures run-to-run variance, not generalisation. For layout spread,
give the variant a sampler that varies -- ``grid_xy_quat``, ``move_near``,
``compose`` -- in a job file that registers its own suite, and pass that file so
the workers import it too.

**Do not oversubscribe the CPU.** Each worker runs a physics loop and, for a
graph, a runtime thread whose tool calls are serviced once per control step. Take
the cores away and episodes stretch, some truncate that would not have, and the
bus watchdog -- 30 s of wall clock -- starts firing. ``--workers`` wants sizing
against cores; ``--bus-timeout`` is the release valve when it cannot be.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from dataclasses import replace
from pathlib import Path

from robosimstudio.roots import add_task_root
from robosimstudio.specs import CaptureSpec, JobSpec, RecordSpec
from robosimstudio_bench.run import RunnerEval, RunSummary, load_jobs, run_pooled
from robosimstudio_bench.solutions.declared import ROUTINES_ABSENT, ROUTINES_MODULE, routines_installed
from robosimstudio_bench.taskbuild.registry import list_tasks

# Shorthands, so the common cases do not need a dotted path. A solution is the
# code that solves a task: one of its graphs, or `idle`, which drives nothing and
# is the floor. A tree that has them adds the task's scripted routine and a
# policy served over the wire; a release that leaves those modules out offers
# neither, rather than a name that fails at import.
SOLUTIONS = {
    "idle": "robosimstudio_bench.solutions.base:SolutionIdle",
    "graph": "robosimstudio_policy_agent.graph_solution:SolutionGraph",
}
if routines_installed():
    SOLUTIONS["routine"] = f"{ROUTINES_MODULE}:SolutionRoutine"
if importlib.util.find_spec("robosimstudio_bench.solutions.vla") is not None:
    SOLUTIONS["vla"] = "robosimstudio_bench.solutions.vla:SolutionVLA"
SOLUTION_ALIASES = {"scripted": "routine"} if "routine" in SOLUTIONS else {}
"""The spelling `--solve routine` had; honoured for one release."""


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--task", default="", help="suite/variant; --list prints them, --task-root adds a folder of your own"
    )
    parser.add_argument(
        "--documents",
        action="append",
        default=[],
        metavar="A,B,C",
        help=(
            "several task documents to run in ONE job, comma-separated and repeatable, or @file for "
            "one per line. They must all resolve to the same model -- in practice all declaring the "
            "same (:fleet-base X) -- and with --num-worlds above one they share a wave: thirty "
            "instance documents of one superset run as three waves of twelve instead of thirty "
            "processes. --episodes then selects which episodes OF EACH (default 0:1, since a paper "
            "instance document declares one). Each world is scored on its own document's goal over "
            "its own objects and its row carries its own task, so the results are the same rows the "
            "documents produce one at a time. Needs --num-worlds above 1: the serial runner has no "
            "wave to share, so a job naming several documents with --num-worlds 1 is refused"
        ),
    )
    parser.add_argument(
        "--task-root",
        action="append",
        default=[],
        metavar="DIR",
        help=(
            "a directory holding task directories; repeatable. Tasks in it register beside this "
            "repository's own, so running somebody else's benchmark is a path rather than an install. "
            "Also settable as ROBOSIMSTUDIO_TASK_ROOTS, which is what worker processes inherit"
        ),
    )
    parser.add_argument(
        "--solve", "--policy", dest="solve", default="idle", help=f"{', '.join(SOLUTIONS)}, or a dotted path"
    )
    parser.add_argument(
        "--episodes",
        default="0:1",
        help=(
            "which episodes: a half-open range '0:16', a strided one '0:100:4' (every fourth id), "
            "or a list '0,4,9'. A benchmark instance set is often strided through its base document, "
            "and naming the ids keeps a run to the instances rather than everything between them"
        ),
    )
    parser.add_argument(
        "--scene",
        default="",
        help=(
            "stage the task in another registered place; empty is its own. Must be a pairing the "
            "task declares -- Env raises with the admissible list otherwise"
        ),
    )
    parser.add_argument(
        "--embodiment",
        default="",
        help=(
            "run the task on another registered robot; empty is its own. Same rule as --scene: "
            "the pairing has to be declared"
        ),
    )
    parser.add_argument(
        "--gripper",
        default="",
        help=(
            "put another registered hand on every arm; empty is the robot's own. Same rule again, and "
            "only a hand that carries a file of its own can go on an arm that did not ship with it"
        ),
    )
    parser.add_argument("--jobs", default="", help="a Python job file declaring JOBS: list[JobSpec]")
    parser.add_argument(
        "--renderer",
        default="",
        help="rtx (the renderer) or tiled (colliders only); observer/video renderer; empty is fastest",
    )
    parser.add_argument(
        "--obs-renderer",
        default="",
        dest="obs_renderer",
        help="renderer for obs['cameras'] (perception); defaults to --renderer when omitted",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help=(
            "keep the episodes: video, ground-truth state and every step, under the run directory. "
            "Off by default because a measurement pass does not need them and they are the "
            "expensive part -- turn it on to collect data, and for anything render_offline or "
            "export_lerobot will be pointed at"
        ),
    )
    parser.add_argument("--record-depth", action="store_true", help="also write a colourised depth preview")
    parser.add_argument(
        "--record-cameras",
        default="",
        help=(
            "with --record, comma-separated camera names to write as mp4; empty writes every camera. A path "
            "tracer charges per camera per frame, and eight recorded runs of four cameras each put this box at "
            "a load that changes what the policy does -- one camera at a lower rate does not"
        ),
    )
    parser.add_argument(
        "--record-every", type=int, default=1, help="with --record, keep one step in this many (1 keeps all)"
    )
    parser.add_argument(
        "--record-fps",
        type=int,
        default=30,
        help="mp4 frame rate; frames are sampled against SIMULATION time, so this is also the "
        "playback speed. Match the task's control rate (60 for these suites) for one frame "
        "per control step -- the finest distinct motion there is, since the sim only advances "
        "the visible state once per control step.",
    )
    parser.add_argument(
        "--capture-state",
        action="store_true",
        help=(
            "record a full-rate raw simulator tape, static name/index manifest, contacts and "
            "graph motion spans. The capture contains no scene descriptions or narration"
        ),
    )
    parser.add_argument(
        "--record-from-motion",
        action="store_true",
        help="with --record, omit leading idle frames and begin when the commanded robot action first changes",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help=(
            "control steps before truncation; 0 uses the task's own budget, which is the right "
            "answer now that tasks size it per variant -- a one-goal compartment pick and an "
            "eight-goal sort do not want the same ceiling, and a tool posted through a hole wants "
            "several times either. A graph needs far more than a scripted policy: every robot.* "
            "tool blocks until the arm converges, so one 'go to pose' is tens to hundreds of steps "
            "and a pick-and-place is thousands"
        ),
    )
    parser.add_argument("--out", default="outputs", help="where run directories go")
    parser.add_argument("--name", default="", help="run directory name; defaults to task+policy")
    parser.add_argument("--device", default="cuda", help="the device when --devices is not given")
    parser.add_argument("--list", action="store_true", help="list the registered tasks and exit")

    group = parser.add_argument_group("many at once")
    group.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "processes to run, each holding one environment. 1 keeps the episodes in this process, "
            "which is the debuggable path. Above one, size against cores rather than episodes: a "
            "graph worker starved of CPU does not merely run slower, it scores differently"
        ),
    )
    group.add_argument(
        "--num-worlds",
        type=int,
        default=1,
        help=(
            "episodes to run at once in one replicated model. 1 is the serial runner. Above one, "
            "the job runs in waves of this many and one step advances all of them -- which is where "
            "the throughput is, since a wave costs little more than a single world does"
        ),
    )
    group.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help=(
            "control steps between scoring passes in a batched run. 1 scores every step, which is "
            "what a run compared against a serial baseline wants; higher trades that many steps of "
            "latency in noticing an episode ended for less host work"
        ),
    )
    group.add_argument(
        "--view",
        action="store_true",
        help=(
            "serve a live grid of the wave in a browser. The worlds are superimposed in the "
            "simulation and spread out only at draw time, so watching costs the frames and "
            "nothing else. Served from this process, so it needs --workers 1"
        ),
    )
    group.add_argument("--view-port", type=int, default=8080, help="port for --view")
    group.add_argument("--devices", default="", help="comma-separated Warp devices to spread workers over")
    group.add_argument(
        "--stagger",
        type=float,
        default=4.0,
        help="seconds between worker starts, so N cold Warp caches do not collide",
    )
    group.add_argument(
        "--threads-per-worker",
        type=int,
        default=0,
        help="OMP_NUM_THREADS per worker; 0 divides the visible cores among them",
    )

    if "vla" in SOLUTIONS:
        group = parser.add_argument_group("vla policy")
        group.add_argument("--vla-host", default="127.0.0.1", help="where the inference server is")
        group.add_argument("--vla-port", type=int, default=8765, help="its port")
        group.add_argument(
            "--vla-name",
            default="",
            help=(
                "the checkpoint's published name (rt-1-x, octo-base, ...). Recorded as the run's "
                "policy, so the results can be paired against the published table -- a run named "
                "anything else measures fine and correlates with nothing"
            ),
        )
        group.add_argument(
            "--vla-camera",
            default="",
            help="which camera's frames to send; empty is the task's first, the evaluation camera",
        )

    group = parser.add_argument_group("graph policy")
    group.add_argument("--graph", default="gt", help="a graph the task ships, or a path to a workflow directory")
    group.add_argument(
        "--skills",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "a skill registry checkout (a directory with skills/ and/or scripts/), repeatable. A full "
            "override of $GAP_SKILLS_PATH and of the [tool.gap] table in pyproject.toml, which names "
            "the open-robot-skills clone beside this checkout -- so only for a checkout that lives "
            "somewhere else. Recorded in the run's policy kwargs, where an environment variable is not"
        ),
    )
    group.add_argument(
        "--roll",
        default="auto",
        choices=("auto", "free", "held"),
        help=(
            "the roll contract every solve and arrival test is held to. 'auto' (default) reads the "
            "robot's own declaration (EmbodimentSpec.honour_roll, off everywhere but the UR5e); "
            "'free' aligns the tool axis and leaves the wrist to the solver, which is what every "
            "shipped graph was measured against; 'held' solves the whole orientation -- a graph that "
            "chose the direction the jaws close along asks for it per call with orientation='lock'."
        ),
    )
    group.add_argument(
        "--motion",
        default="osc",
        help=(
            "how robot.go_to_pose moves. 'osc' (default) is the reference benchmark's: a feedback "
            "servo with OSC_POSE's own law, bounded Cartesian speed and a taper into the target -- "
            "the motion these tasks were authored against. 'plan' is the runtime's own: one IK "
            "solve, then a joint move, a step input on a stiff position servo. Faster in wall "
            "clock, for debugging a graph's logic rather than its motion. 'curobo' keeps the "
            "servo's per-step solver and sends the *plans* to a collision-aware planner, which "
            "needs the embodiment to declare a curobo_config -- panda and yam_tool_hanging do, "
            "and anything else prints and falls back. Any dotted path to a planner class is also "
            "accepted."
        ),
    )
    group.add_argument(
        "--no-privileged",
        action="store_true",
        help=(
            "withhold the sim.* tools -- the generic set and a suite's own -- to check a graph "
            "runs on perception alone. Pair it with '--perception vision'; withholding the tools "
            "while still asking for the body that wants them only measures how a graph fails"
        ),
    )
    group.add_argument(
        "--no-suite-tools",
        action="store_true",
        help=(
            "skip the suite's own tool registrar (TaskSpec.tools) entirely -- every tool it "
            "registers, whatever its name or tags -- while keeping the generic privileged set. "
            "The generic set is the same catalogue on every task -- a pose, a bound, an OBB, a "
            "relation -- and withholding it is what '--no-privileged' is for. A suite's own tools "
            "are written for one task and routinely hand over its answer: where to pick and place, "
            "the insertion schedule, the free end of a film. Use this to measure what a policy can "
            "do rather than what the suite already knows"
        ),
    )
    group.add_argument(
        "--perception",
        default="oracle",
        choices=("oracle", "vision"),
        help=(
            "which body a graph's perception subgraphs run. 'oracle' reads the simulator's own "
            "answer and is right for building a graph or generating data; 'vision' goes through "
            "RGB-D and a detector, which is what has to work on a real rig. Both declare the same "
            "outputs, so the rest of the graph cannot tell which ran -- and a graph that ships "
            "only one body ignores this. Passed to the workflow as an input, not a flag the "
            "runtime knows about"
        ),
    )
    group.add_argument(
        "--allow-undeclared-graph",
        action="store_true",
        help=(
            "run a shipped graph on a pairing whose binding does not claim it. Refused by "
            "default: a shipped graph is written against one robot in one place, so this is an "
            "experiment rather than a measurement, and the flag is what labels it as one in the "
            "run's provenance"
        ),
    )
    group.add_argument(
        "--settle-steps",
        type=int,
        default=60,
        help="steps to keep stepping after the graph finishes, so the evaluator scores a settled scene",
    )
    group.add_argument(
        "--bus-timeout",
        type=float,
        default=0.0,
        help=(
            "seconds a graph worker waits for the sim thread to service a request; 0 keeps the "
            "SimBus default of 30. Raise it when workers share cores, or contention alone will "
            "fail episodes that were converging"
        ),
    )
    return parser


def parse_episodes(text: str) -> tuple[tuple[int, int], tuple[int, ...]]:
    """``a:b``, ``a:b:s`` or ``a,b,c`` -> the bracketing span and the explicit id list.

    The list is empty for a plain range and for a strided or listed selection
    that happens to be contiguous, so a job that named every id of a range is
    the same job as one that named the range. Ids are de-duplicated and sorted:
    an episode is run once whatever order it was asked in, and the rows sort
    the way a serial run's do.

    Raises:
        SystemExit: On an empty selection or a malformed spec.
    """
    text = text.strip()
    try:
        if "," in text:
            ids = tuple(sorted({int(v) for v in text.split(",") if v.strip()}))
        elif text.count(":") == 2:
            first, last, stride = (int(v) for v in text.split(":"))
            ids = tuple(range(first, last, max(1, stride)))
        else:
            first_text, _, last_text = text.partition(":")
            first = int(first_text)
            last = int(last_text) if last_text else first + 1
            if last <= first:
                raise SystemExit(f"[run] --episodes {text!r} names no episode")
            return (first, last), ()
    except ValueError as exc:
        raise SystemExit(f"[run] --episodes {text!r}: {exc}") from None
    if not ids:
        raise SystemExit(f"[run] --episodes {text!r} names no episode")
    contiguous = ids == tuple(range(ids[0], ids[-1] + 1))
    return (ids[0], ids[-1] + 1), (() if contiguous else ids)


def parse_documents(values: list[str]) -> tuple[str, ...]:
    """``--documents`` as a flat tuple: comma-separated, repeatable, ``@file`` expanded.

    Order is the order given and duplicates are dropped, because a document
    named twice would be one world running it twice in the same wave -- which
    is a legitimate thing to ask for and not a legitimate thing to ask for by
    accident. Use ``--episodes`` to run several episodes of each.
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        token = token.strip()
        if not token or token.startswith("#"):
            return
        if token.startswith("@"):
            for line in Path(token[1:]).read_text().splitlines():
                add(line)
            return
        if token not in seen:
            seen.add(token)
            out.append(token)

    for value in values or ():
        for token in str(value).split(","):
            add(token)
    return tuple(out)


def job_from_args(args: argparse.Namespace) -> JobSpec:
    """The command line as one job. A job file is the same thing, in Python."""
    episodes, episode_list = parse_episodes(args.episodes)
    documents = parse_documents(getattr(args, "documents", []))
    if documents:
        # The job's identity is its first document unless the caller named a
        # task outright, so a mixed run's directory and provenance say what it
        # was rather than the parser's default.
        if not args.task:
            args.task = documents[0]
        # ``--episodes`` already defaults to ``0:1``, which is episode 0 of
        # each document -- exactly right for a paper instance document, which
        # declares one. Nothing to adjust; noted because the reading of the
        # flag changes (episodes OF EACH, not of the job) and that is the whole
        # difference.
    args.solve = SOLUTION_ALIASES.get(args.solve, args.solve)
    policy = SOLUTIONS.get(args.solve, args.solve)
    kwargs: list[tuple[str, object]] = []
    if policy == SOLUTIONS["graph"]:
        # The trace directory is deliberately not per episode: solution_kwargs is
        # part of a pool worker's environment key, so varying it between
        # episodes would rebuild the scene each time. SolutionGraph numbers the
        # episodes underneath it.
        kwargs = [
            ("graph", args.graph),
            ("privileged", not args.no_privileged),
            ("suite_tools", not args.no_suite_tools),
            ("perception", args.perception),
            ("settle_steps", args.settle_steps),
            ("roll", args.roll),
            ("motion", args.motion),
        ]
        if args.bus_timeout > 0:
            kwargs.append(("bus_timeout", args.bus_timeout))
        if args.skills:
            # A tuple: solution_kwargs is part of a pool worker's environment key.
            kwargs.append(("skills", tuple(str(Path(p).resolve()) for p in args.skills)))
    elif policy == SOLUTIONS.get("vla"):
        kwargs = [("host", args.vla_host), ("port", args.vla_port), ("policy", args.vla_name or "remote")]
        if args.vla_camera:
            kwargs.append(("camera", args.vla_camera))
    # A path-valued --graph would put slashes in the run directory's name and
    # nest it somewhere nobody looks; its basename identifies it as well.
    graph_label = Path(str(args.graph).rstrip("/")).name if "/" in str(args.graph) else args.graph
    suffix = (
        f"graph_{graph_label}"
        if policy == SOLUTIONS["graph"]
        else f"vla_{args.vla_name or 'remote'}"
        if policy == SOLUTIONS.get("vla")
        else args.solve
    )
    # The pairing goes in the default run name too. A task name alone stopped
    # identifying a run the moment one task could be staged with several robots,
    # and two runs of one task on two robots landing in one directory is a
    # measurement quietly overwriting another.
    pairing = "".join(f"_{v}" for v in (args.scene, args.embodiment, args.gripper) if v)
    return JobSpec(
        name=args.name or f"{args.task.replace('/', '_')}{pairing}_{suffix}",
        task=args.task,
        documents=documents,
        scene=args.scene,
        embodiment=args.embodiment,
        gripper=args.gripper,
        solution=policy,
        solution_kwargs=tuple(kwargs),
        episodes=episodes,
        episode_list=episode_list,
        renderer=args.renderer,
        obs_renderer=args.obs_renderer,
        record=RecordSpec(
            enabled=args.record,
            cameras=tuple(c.strip() for c in args.record_cameras.split(",") if c.strip()),
            every=max(1, int(args.record_every)),
            record_depth=args.record_depth,
            video_fps=args.record_fps,
            start_on_motion=args.record_from_motion,
        ),
        capture=CaptureSpec(enabled=args.capture_state),
        max_steps=args.max_steps,
        allow_undeclared_graph=args.allow_undeclared_graph,
        device=args.device,
        num_worlds=args.num_worlds,
        eval_every=args.eval_every,
    )


def check_graph_claims(jobs: list[JobSpec]) -> None:
    """Refuse a shipped graph on a pairing whose binding does not claim it.

    The studio gates its graph menu the same way and for the same reason
    (:class:`~robosimstudio_studio.controls.GraphControls`): a shipped
    graph is written against one robot in one place, so offering it on a pairing
    the binding does not claim invites a run that fails for reasons nothing can
    explain -- and whose zero then reads as the graph's, not the pairing's.

    Checked here rather than in :class:`~robosimstudio_bench.run.RunnerEval`
    because it is advice about a *claim* rather than an invariant of the model.
    The hard check is :meth:`Env.__init__`'s, which refuses an undeclared
    ``(scene, embodiment)`` outright; this one refuses a *policy* on a declared
    pairing, and ``--allow-undeclared-graph`` is how a person says they are
    exploring rather than measuring.

    A ``--graph`` that is a path rather than a shipped name is left alone: the
    binding cannot have an opinion about a directory it has never heard of.

    Raises:
        SystemExit: Naming the graph, the pairing and what the binding does claim.
    """
    from robosimstudio_bench.taskbuild.matrix import declared_key
    from robosimstudio_bench.taskbuild.registry import get_task

    for job in jobs:
        graph = dict(job.solution_kwargs).get("graph", "")
        if not graph or job.allow_undeclared_graph:
            continue
        task = get_task(job.task)
        try:
            task.graph_path(str(graph))
        except KeyError:
            continue  # a path, not a shipped name
        # Raises for an undeclared pairing, which is the right error and a
        # better one than the compile would give -- it arrives before a GPU is
        # touched and it lists what is on offer.
        # The declared spelling first, as ``Env`` takes it: naming the hand the
        # robot already wears is the binding that left it unstated, and
        # ``binding_for`` compares raw keys.
        want = declared_key(task, job.scene, job.embodiment, job.gripper) or (job.scene, job.embodiment, job.gripper)
        binding = task.binding_for(*want)
        if graph in binding.graphs:
            continue
        pairing = f"scene={binding.scene or task.scene!r} embodiment={binding.embodiment or task.embodiment!r}" + (
            f" gripper={binding.gripper!r}" if binding.gripper else ""
        )
        claimed = sorted(binding.graphs) or "no graphs"
        raise SystemExit(
            f"[run] {job.task} on {pairing} does not claim graph {graph!r}; that binding claims {claimed}"
            + (f" ({binding.notes})" if binding.notes else "")
            + ".\n[run] Pass --allow-undeclared-graph to run it anyway -- the result is an experiment, "
            "and is stamped as one in the run's provenance."
        )


def with_trace_dir(job: JobSpec, root: str) -> JobSpec:
    """Point a graph job's traces at its own run directory.

    Otherwise ``SolutionGraph`` writes them to a scratch directory, which is the
    right default for a library and the wrong one for a run somebody will want to
    read afterwards. Only for a graph, and only when the caller did not name one:
    a job file that set ``trace_dir`` meant it.
    """
    kwargs = dict(job.solution_kwargs)
    if "trace_dir" in kwargs or (job.solution != SOLUTIONS["graph"] and "graph" not in kwargs):
        return job
    kwargs["trace_dir"] = str(Path(root) / job.name / "traces")
    return replace(job, solution_kwargs=tuple(kwargs.items()))


def check_capture_schedules(jobs: list[JobSpec]) -> None:
    """Reject a capture schedule a wave cannot honour, before building a model.

    A wave captures on its scoring cadence, because that is when it reads the
    state a point is made of. So ``--eval-every`` above one and capture are a
    tape sparser than it claims to be: a span that resolves to "step 40" would
    have been recorded at 39 or 42, and the self-learning loop that reads these
    wants exact endpoints (see :class:`~robosimstudio.specs.run.CaptureSpec`).
    Refused rather than silently resampled.
    """
    unsupported = [job.name for job in jobs if job.capture.enabled and job.num_worlds > 1 and job.eval_every > 1]
    if unsupported:
        raise SystemExit(
            "[run] raw point/span capture in a wave records on the scoring cadence, so it needs "
            "--eval-every 1; a sparser tape would resolve spans to steps it never recorded. "
            f"Job(s) asking for both: {', '.join(unsupported)}"
        )


def check_document_waves(jobs: list[JobSpec]) -> None:
    """Reject ``--documents`` on the serial runner, which has no wave to share.

    Several documents in one job is a statement about a *wave*: one model, one
    world each, each scored on its own goal. The serial runner builds one
    environment from one document and runs its episodes through it, so the only
    thing it could do with a second document is ignore it -- and a run that
    silently measured document one thirty times, labelled as thirty documents,
    is the exact class of number nothing downstream can tell apart from a real
    one. Pass them one at a time there, or raise ``--num-worlds``.
    """
    serial = [job.name for job in jobs if getattr(job, "documents", ()) and job.num_worlds <= 1]
    if serial:
        raise SystemExit(
            "[run] --documents shares a WAVE between documents, so it needs --num-worlds above 1. "
            "The serial runner builds one model from one document and would ignore the rest. "
            f"Job(s) asking for both: {', '.join(serial)}"
        )


def report(summary: RunSummary) -> None:
    """Print what the run measured: the rate, its interval, and what it cost.

    One report for a serial run and a pooled one, because they are one
    measurement -- and printing a bare rate for a serial run was how the numbers
    that need reading together, the interval and the close-kind histogram, came
    to be seen only when somebody happened to use the pool.
    """
    lo, hi = summary.interval
    kinds = ", ".join(f"{kind} {count}" for kind, count in summary.close_kinds.items())
    hz = float(summary.builds[0]["control_hz"]) if summary.builds else 0.0
    print(
        f"[run] {summary.job.name}: {summary.successes}/{summary.episodes} success "
        f"({summary.success_rate * 100:.0f}%, 95% CI {lo * 100:.0f}-{hi * 100:.0f}%), "
        f"mean length {summary.mean_length:.0f} steps"
    )
    print(f"[run]   ended: {kinds}")
    workers = f"{summary.workers} workers" if summary.workers > 1 else "one process"
    tail = f", {summary.pool_wall_s / 60:.1f} min all in" if getattr(summary, "pool_wall_s", 0.0) else ""
    print(
        f"[run]   {summary.wall_s / 60:.1f} min of episodes on {workers} ({summary.episodes_per_min:.2f} ep/min){tail}"
    )
    print(
        f"[run]   {summary.steps} control steps, {summary.steps_per_s:.0f}/s end to end, "
        f"{summary.steps_per_s_stepping:.0f}/s while stepping"
        + (f" ({summary.steps_per_s_stepping / hz:.1f}x realtime)" if hz else "")
    )
    # An environment that fell back to eager is several times slower and its rows
    # look no different, so the count is printed rather than assumed.
    captured = summary.captured
    print(f"[run]   cuda graph: {captured['captured']} captured, {captured['eager']} eager")
    if summary.unrun:
        print(f"[run]   WARNING {len(summary.unrun)} episodes never ran: {list(summary.unrun)[:16]}")
    print(f"[run]   -> {summary.root}")


def main() -> int:
    args = create_parser().parse_args()
    # A run that configures no logging discards every logger.info in the tree
    # before it reaches a handler, because the root logger's default level is
    # WARNING. That is not a cosmetic loss: a graph's own measurements go out
    # that way -- `[reach] err_mm=... tol_mm=...`, `[grip] status=... width_mm=...`
    # -- and a measurement nobody can read is not instrumentation. The scripts
    # used to work around it by logging at WARNING instead, which put the
    # knowledge of *this* command inside a skill that has no business knowing
    # about it; the workaround is gone and this is where it belonged.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Before any registry is touched: a root declared after the first lookup
    # would be a root nothing scanned, and the failure would be a task that
    # "does not exist" rather than a root that arrived too late.
    for root in args.task_root:
        add_task_root(root)
    if args.list:
        for name in list_tasks():
            print(name)
        return 0

    if args.solve in ("routine", "scripted") and "routine" not in SOLUTIONS:
        print(ROUTINES_ABSENT, file=sys.stderr)
        return 2
    if not args.jobs and not args.task and not args.documents:
        print("name a task: --task SUITE/VARIANT (--list prints them; --task-root DIR adds your own)", file=sys.stderr)
        return 2
    jobs = load_jobs(args.jobs) if args.jobs else [job_from_args(args)]
    jobs = [with_trace_dir(job, args.out) for job in jobs]
    # Before anything is built: a job file's sweep is checked here too, since a
    # sweep across pairings is exactly where an unclaimed graph goes unnoticed.
    check_graph_claims(jobs)
    check_capture_schedules(jobs)
    check_document_waves(jobs)
    if not args.record and not args.jobs:
        print("[run] not recording episodes (pass --record to keep them)")

    if args.workers > 1:
        if args.view:
            # The pool runs each job in a worker process with no page of its own;
            # watching a wave is the in-process fleet runner's, under --workers 1.
            raise SystemExit("[run] --view serves the wave from this process, so it needs --workers 1")
        summaries = list(
            run_pooled(
                jobs,
                root=args.out,
                workers=args.workers,
                devices=tuple(d.strip() for d in args.devices.split(",") if d.strip()),
                stagger=args.stagger,
                threads_per_worker=args.threads_per_worker,
                # The workers are fresh interpreters: a job file that registers
                # its own suite has to be imported in each of them, or the task
                # will not resolve.
                preload=args.jobs,
            ).values()
        )
    else:
        summaries = []
        for job in jobs:
            # num_worlds picks the runner, and 1 is literally the code path this
            # has always taken: a batched run is a different scheduler over the
            # same measurement, not a different measurement.
            if job.num_worlds > 1:
                from robosimstudio_bench.run.fleet import RunnerFleet

                runner = RunnerFleet(job, root=args.out)
                if args.view:
                    runner.view_port = args.view_port
            else:
                runner = RunnerEval(job, root=args.out)
            runner.run()
            summaries.append(runner.summary)

    for summary in summaries:
        report(summary)
    return 0 if all(s.successes or not s.episodes for s in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
