# SPDX-License-Identifier: Apache-2.0
"""The rollout, and the runner that repeats it over a job's episodes.

Both are pure library. There is no CLI here: entry points are the examples,
which compose these. That keeps every knob a Python value in a job file rather
than a flag whose parsing is a second, undocumented configuration format.

:func:`rollout` is the one place an episode's shape is defined -- reset, loop,
close -- so a scripted run, a graph run and a VLA run are the same code path and
their results are comparable by construction. The external-termination path is
forwarded here rather than in the policies, so every policy gets it for free and
none of them can accidentally claim success.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robosimstudio.record import EpisodeRecorder, EpisodeStore, RunMeta
from robosimstudio.specs import JobSpec, RecordSpec
from robosimstudio_bench.envs.env import Env

from ..solutions import SolutionBase, load_solution
from .jobs import job_task
from .summary import RunSummary

__all__ = ["RolloutResult", "RunnerEval", "rollout"]


@dataclass
class RolloutResult:
    """One episode's outcome.

    Attributes:
        success: What the evaluator said.
        reward: Its last reward.
        steps: Control steps taken.
        close_kind: Why the episode ended -- ``success``, ``truncated``,
            ``policy_request``, ``diverged``.
        info: The final step's info dict.
        row: The results row, as written to ``episode_results.jsonl``.
    """

    success: bool = False
    reward: float = 0.0
    steps: int = 0
    close_kind: str = ""
    info: dict[str, Any] = field(default_factory=dict)
    row: dict[str, Any] = field(default_factory=dict)


def rollout(
    env: Env,
    policy: SolutionBase,
    *,
    episode_id: int = 0,
    recorder: EpisodeRecorder | None = None,
    max_steps: int = 0,
) -> RolloutResult:
    """Run one episode to termination.

    Args:
        env: The environment.
        policy: What drives it.
        episode_id: Which episode.
        recorder: Where to record it, or ``None``.
        max_steps: Overrides the environment's own when non-zero.
    Returns:
        The episode's outcome and its results row.
    """
    wall0 = time.perf_counter()
    obs, info = env.reset(episode_id)
    if recorder is not None:
        recorder.begin(env.episode, env)
    policy.reset(obs, info)

    def record_step(current, action, observed, earned, details):
        recording_stage = getattr(policy, "recording_stage", None)
        if recording_stage is None:
            recorder.step(current, action, observed, earned, details)
        else:
            recorder.step(
                current,
                action,
                observed,
                earned,
                details,
                stage=recording_stage(),
            )

    if recorder is not None:
        env.add_step_observer(record_step)

    limit = max_steps or env.max_steps or 10_000
    reward = 0.0
    terminated = truncated = False
    try:
        for _ in range(limit):
            action = policy.get_action(obs)
            # The policy names the action space it emits into; empty keeps the
            # environment's own. Nested tool steps are recorded by the same
            # completed-step listener as this outer step.
            obs, reward, terminated, truncated, info = env.step(action, mode=policy.control_mode)
            policy.on_step(obs, reward, info)
            if terminated or truncated:
                break
        else:
            truncated = True
            info.setdefault("close_kind", "truncated")
    finally:
        if recorder is not None:
            env.remove_step_observer(record_step)

    extra = policy.row_extras()
    if recorder is not None and hasattr(recorder, "record_spans") and hasattr(policy, "raw_spans"):
        recorder.record_spans(policy.raw_spans())
    row = (
        recorder.end(env, info, extra=extra)
        if recorder is not None
        else {
            "task": env.task.full_name,
            "episode_id": env.episode.episode_id,
            "seed": env.episode.seed,
            "success": bool(info.get("success", False)),
            "reward": float(reward),
            "episode_length": env.step_count,
            "close_kind": info.get("close_kind", "unknown"),
            # WHY the policy asked out, not just that it did. The env already
            # computes this -- `EnvBase` stamps `termination_reason` next to
            # `close_kind` -- and the recorder writes it; a measurement pass is
            # not recording, so without this line a whole campaign reports
            # `policy_request` and nothing else. Measured: six conditions ended
            # 100% on `policy_request` and the reason each one gave up took a
            # six-agent audit of the logs to recover.
            "termination_reason": str(info.get("termination_reason", "")),
            "metrics": info.get("metrics", {}),
            # The recorder stamps this on the rows it writes; a measurement pass
            # is not recording and would otherwise report no episode timings at
            # all, which is half of what a rate is read against.
            "wall_s": round(time.perf_counter() - wall0, 3),
            **extra,
        }
    )
    return RolloutResult(
        success=bool(info.get("success", False)),
        reward=float(reward),
        steps=env.step_count,
        close_kind=str(info.get("close_kind", "unknown")),
        info=info,
        row=row,
    )


class RunnerEval:
    """Runs a job's episodes and writes the run directory.

    Args:
        job: What to run.
        root: Where run directories go. The run lands in ``root/<job.name>``.
        verbose: Print one line per episode.
    """

    def __init__(self, job: JobSpec, *, root: str | Path = "outputs", verbose: bool = True) -> None:
        # One environment from one document: a second document could only be
        # ignored, and its episodes would read as unrun rather than as refused.
        if getattr(job, "documents", ()):
            raise ValueError(
                f"job {job.name!r} names documents, which share a WAVE; the serial runner builds one model "
                "from job.task. Use RunnerFleet (num_worlds above 1), or one job per document."
            )
        self.job = job
        self.root = Path(root) / job.name
        self.verbose = bool(verbose)
        self.store = EpisodeStore(self.root)
        self.results: list[RolloutResult] = []
        self.summary = RunSummary(job=job, root=self.root, workers=1)
        """What the run measured, in the shape a pooled run reports too."""

    def run(self) -> list[RolloutResult]:
        """Run every episode, appending results as they finish."""
        job = self.job
        started = datetime.now(UTC).isoformat()
        wall0 = time.perf_counter()

        env = Env(
            job_task(job),
            scene=job.scene,
            embodiment=job.embodiment,
            gripper=job.gripper,
            renderer=job.renderer,
            obs_renderer=job.obs_renderer,
            device=job.device,
            max_steps=job.max_steps,
            control_mode=job.control_mode,
        )
        policy = load_solution(job.solution, env, **dict(job.solution_kwargs))
        recorder = (
            EpisodeRecorder(self.root, job.record, task=env.task, capture=job.capture)
            if job.record.enabled or job.capture.enabled
            else None
        )

        # Warm before timing, exactly as a pool worker does: the first reset is
        # where kernel compilation and the CUDA-graph capture happen, so doing it
        # here keeps the first episode timed like the others -- and lets the run
        # report *whether* the capture took, which is the failure that otherwise
        # hides behind perfectly ordinary rows several times too slow.
        env.reset()
        self.summary.builds = [
            {
                "task": job.task,
                "device": str(env.model.device),
                "captured": bool(env.pipeline.captured),
                # ``captured`` here is momentary: a task whose reset ends on
                # invalidate_graph reports False with a re-capture pending, so
                # the step mix below is what says how the episodes actually ran.
                "capture_error": env.pipeline.capture_error,
                "substeps": int(env.pipeline.substeps),
                "control_hz": float(env.pipeline.control_hz),
                "build_s": round(time.perf_counter() - wall0, 2),
            }
        ]
        episodes0 = time.perf_counter()

        try:
            for episode_id in job.episode_ids:
                result = rollout(
                    env,
                    policy,
                    episode_id=episode_id,
                    recorder=recorder,
                    max_steps=job.max_steps,
                )
                row = {**result.row, "solution": policy.name, "renderer": job.renderer, "device": job.device}
                self.store.append_result(row)
                self.results.append(result)
                self.summary.rows.append(row)
                if self.verbose:
                    print(
                        f"[eval] {job.task} episode {episode_id}: "
                        f"{'SUCCESS' if result.success else 'failure':>7} "
                        f"reward {result.reward:.2f} steps {result.steps} ({result.close_kind})"
                    )
        finally:
            self.summary.wall_s = time.perf_counter() - episodes0
            step_mix = {
                "steps_replayed": int(env.pipeline.steps_replayed),
                "steps_eager": int(env.pipeline.steps_eager),
                "recapture_failures": int(env.pipeline.recapture_failures),
                "capture_error": env.pipeline.capture_error,
            }
            policy.close()
            if recorder is not None:
                recorder.close()
            env.close()

        successes = sum(1 for r in self.results if r.success)
        lo, hi = self.summary.interval
        self.store.write_run_meta(
            RunMeta(
                run_name=job.name,
                task=job.task,
                scene=env.task.scene,
                embodiment=env.task.embodiment,
                gripper=env.task.gripper,
                solution=policy.name,
                renderer=job.renderer,
                device=job.device,
                started=started,
                finished=datetime.now(UTC).isoformat(),
                episodes=len(self.results),
                successes=successes,
                # The same keys a pooled run writes, so a run directory reads the
                # same way whether one process produced it or eight did.
                provenance={
                    "pooled": False,
                    "workers": 1,
                    # Whether this pairing's binding claims the graph that ran.
                    # A number from a pairing nobody has claimed is an
                    # experiment, and the artifact is where that has to be
                    # legible -- see JobSpec.allow_undeclared_graph.
                    "undeclared_graph": bool(job.allow_undeclared_graph),
                    "wall_s": round(self.summary.wall_s, 2),
                    "run_wall_s": round(time.perf_counter() - wall0, 2),
                    "episodes_per_min": round(self.summary.episodes_per_min, 2),
                    "success_rate": round(self.summary.success_rate, 4),
                    "wilson95": [round(lo, 4), round(hi, 4)],
                    "close_kinds": self.summary.close_kinds,
                    "unrun_episodes": list(self.summary.unrun),
                    "control_steps": self.summary.steps,
                    "steps_per_s": round(self.summary.steps_per_s, 1),
                    "steps_per_s_stepping": round(self.summary.steps_per_s_stepping, 1),
                    "cuda_graph": self.summary.captured,
                    "cuda_graph_steps": step_mix,
                    "builds": self.summary.builds,
                    "suite_provenance": _suite_provenance(job.task),
                },
            )
        )
        if self.verbose:
            n = len(self.results)
            print(f"[eval] {job.name}: {successes}/{n} success ({successes / n * 100 if n else 0:.0f}%) -> {self.root}")
        return self.results


def _suite_provenance(task: str) -> str:
    from robosimstudio_bench.taskbuild.registry import get_suite

    try:
        return get_suite(task.partition("/")[0]).provenance
    except KeyError:
        return ""


def run_jobs(
    jobs: list[JobSpec], *, root: str | Path = "outputs", verbose: bool = True
) -> dict[str, list[RolloutResult]]:
    """Run several jobs in order, returning each one's results."""
    return {job.name: RunnerEval(job, root=root, verbose=verbose).run() for job in jobs}


def default_record(**overrides) -> RecordSpec:
    """A record spec with the usual settings, for a job file that wants one line."""
    return RecordSpec(**overrides)
