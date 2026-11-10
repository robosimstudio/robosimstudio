# SPDX-License-Identifier: Apache-2.0
"""Loading job files.

A job file is a Python module declaring ``JOBS: list[JobSpec]``. Python-first
means an evaluation sweep is a list comprehension rather than a templated YAML
document, and that the thing which validates a job is the dataclass rather than
a schema maintained beside it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from robosimstudio.specs import JobSpec, TaskSpec

__all__ = [
    "FLEET_RUNTIME",
    "apply_runtime",
    "job_task",
    "load_jobs",
    "runtime_provenance",
    "video_cadence",
]

FLEET_RUNTIME: dict[str, Any] = {"clock": "fleet-barrier", "obs_render": "on-demand", "render_every": None}
"""What a batched run (``num_worlds`` above 1) runs under, whatever its job asked.

``JobSpec.clock`` is a no-op there, not an override: the fleet steps only when
every live world's graph thread is parked (:mod:`robosimstudio_policy_agent.fleet.barrier`),
which is lockstep for each world already, and a world's cameras render only when
its graph looks. Recorded as what ran rather than as what was asked, so a
batched run's provenance cannot claim a free clock it never had."""


def job_task(job: JobSpec, name: str | TaskSpec = "") -> TaskSpec:
    """The document a job runs, resolved and with ``job.seed`` applied.

    Every runner -- serial, pooled, batched -- resolves its documents here, so
    the seed override is applied in one place and an episode's ``seed`` (which
    its row records) is the one the job asked for.

    Args:
        job: The job.
        name: A document to resolve instead of ``job.task`` -- one of
            ``job.documents`` for a mixed job -- or an already-resolved spec.
    """
    from robosimstudio_bench.taskbuild.episodes import with_base_seed
    from robosimstudio_bench.taskbuild.registry import get_task

    target = name or job.task
    task = get_task(target) if isinstance(target, str) else target
    return with_base_seed(task, job.seed)


def load_jobs(path: str | Path) -> list[JobSpec]:
    """Load ``JOBS`` from a Python job file.

    Args:
        path: The file. It is imported under its own name, so it may define
            helpers and samplers alongside the jobs.

    Returns:
        The declared jobs.
    """
    file = Path(path).expanduser().resolve()
    if not file.exists():
        raise FileNotFoundError(f"no job file at {file}")
    spec = importlib.util.spec_from_file_location(f"robosimstudio_job_{file.stem}", file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import job file {file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    jobs = getattr(module, "JOBS", None)
    if jobs is None:
        raise AttributeError(f"job file {file} declares no JOBS list")
    for job in jobs:
        if not isinstance(job, JobSpec):
            raise TypeError(f"job file {file} declares a {type(job).__name__}, expected JobSpec")
    return list(jobs)


def apply_runtime(job: JobSpec, env: Any, policy: Any) -> None:
    """Put a job's clock and observation render onto a built environment and policy.

    The one place a serial runner and a pool worker do it, so the two cannot
    disagree about what ``--clock`` and ``--obs-render`` mean.

    The clock and the observation mode are the *policy's* to apply -- only a
    graph has a thread to keep in step and a way to ask for a render -- so they
    go through :meth:`~robosimstudio_bench.solutions.base.SolutionBase.configure_runtime`,
    which refuses on-demand observations for a policy that cannot ask. The
    environment's part is the video: with observations rendered on demand, a
    recording still wants its frames, at the video's own cadence
    (:attr:`~robosimstudio_bench.envs.env.Env.video_every`) rather than every step.
    """
    configure = getattr(policy, "configure_runtime", None)
    if configure is not None:
        configure(clock=job.clock, obs_render=job.obs_render)
    elif job.obs_render != "per-step":
        raise ValueError(
            f"{type(policy).__name__} cannot ask for a render, so --obs-render {job.obs_render} is refused"
        )
    if job.record.enabled and job.record.record_video:
        env.video_every = video_cadence(float(env.pipeline.control_hz), job.record.video_fps, job.record.every)


def video_cadence(control_hz: float, video_fps: int, record_every: int = 1) -> int:
    """Control steps between video renders, for a run whose observations render on demand.

    As often as the mp4 takes a frame and no more often: the recorder samples
    on the simulation clock at ``video_fps`` and only on steps it records, so a
    render every ``floor(control_hz / video_fps)`` steps, rounded to a multiple
    of ``record_every``, is never more than that many steps behind the frame it
    feeds.
    """
    every = max(1, int(record_every))
    per_frame = max(1, int(float(control_hz) // max(1, int(video_fps))))
    return every * max(1, per_frame // every)


def runtime_provenance(job: JobSpec) -> dict[str, Any]:
    """The clock and the observation render a serial run used, for ``run_meta.json``.

    Recorded because they change what a step count means: under ``free`` the
    episode's length includes whatever the host spent thinking, and under
    ``lockstep`` it does not, so two rows are comparable only if both say which.
    """
    return {"clock": job.clock, "obs_render": job.obs_render, "render_every": int(job.render_every)}
