# SPDX-License-Identifier: Apache-2.0
"""Tasks: the document language, goals, ``Env`` and the run harness.

``import robosimstudio_bench as bench`` gives the few names a script needs, each
imported on first use so the import itself costs nothing::

    task = bench.task(DOCUMENT)                    # a task written as a string
    env = bench.Env(task)                          # an Env is a Sim: rss.view(env) opens it
    result = bench.rollout(env, policy)            # one episode, scored: result.success, result.reward

What is here, by subpackage: the ``.task`` language
(:mod:`~robosimstudio_bench.taskdoc`), discovery and lowering
(:mod:`~robosimstudio_bench.taskbuild`), the spec-driven ``Env`` and its fleet
twin (:mod:`~robosimstudio_bench.envs`), what a policy is
(:mod:`~robosimstudio_bench.solutions`), the run harness
(:mod:`~robosimstudio_bench.run`), and the goal half of the asset system
(:mod:`~robosimstudio_bench.goals`): binding, goal trees, evaluation -- the
questions a benchmark asks *of* an asset, as against what an asset *is*, which
stays in :mod:`~robosimstudio.assetlib`.

It stands **beside** the library rather than inside it, like
:mod:`robosimstudio_policy_agent`, and names nothing the library does not
already provide. The library reaches it in two places only: the studio's task
panels, and the asset library's content loader, which looks for task folders
when this package is installed. A scene assembles, builds and steps without it.

Importing :mod:`robosimstudio_bench.taskbuild` installs task-folder discovery on
the library's cell, robot and kind registries, so a scene that names a cell a
task folder declares needs that import first.
"""

from __future__ import annotations

import importlib

_FLAT = {
    "task": "robosimstudio_bench.taskbuild.text",
    "get_task": "robosimstudio_bench.taskbuild.registry",
    "list_tasks": "robosimstudio_bench.taskbuild.registry",
    "register_sampler": "robosimstudio_bench.taskbuild.lower",
    "register_arrangement": "robosimstudio_bench.taskbuild.arrangements",
    "register_camera_rig": "robosimstudio_bench.taskbuild.arrangements",
    "Env": "robosimstudio_bench.envs.env",
    "FleetEnv": "robosimstudio_bench.envs.fleet",
    "rollout": "robosimstudio_bench.run.runner",
    "run_jobs": "robosimstudio_bench.run.runner",
    "RunnerEval": "robosimstudio_bench.run.runner",
    "wilson_interval": "robosimstudio_bench.run.summary",
    "SolutionIdle": "robosimstudio_bench.solutions.base",
    "load_solution": "robosimstudio_bench.solutions.base",
    "add_task_root": "robosimstudio.roots",
}
"""Each flat name and the module that defines it."""

__all__ = [
    "Env",
    "FleetEnv",
    "RunnerEval",
    "SolutionIdle",
    "add_task_root",
    "get_task",
    "list_tasks",
    "load_solution",
    "register_arrangement",
    "register_camera_rig",
    "register_sampler",
    "rollout",
    "run_jobs",
    "task",
    "wilson_interval",
]


def __getattr__(name: str):
    module = _FLAT.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_FLAT})
