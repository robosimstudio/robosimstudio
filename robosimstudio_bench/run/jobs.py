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

from robosimstudio.specs import JobSpec, TaskSpec

__all__ = ["job_task", "load_jobs"]


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
