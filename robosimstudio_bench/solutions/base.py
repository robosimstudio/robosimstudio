# SPDX-License-Identifier: Apache-2.0
"""What a policy is, and how a job names one.

One method, ``get_action(obs) -> action``, plus optional lifecycle hooks. A
policy that finishes early calls ``env.request_termination(reason)`` -- it may
end an episode, and it may not declare it a success. That split is the whole
reason this base class is worth having rather than passing callables around.

Policies are named in job files by **dotted path** rather than by registry key.
A registry is right for suites and embodiments, which a task file refers to by
name and which ship with the package; a policy is often somebody's own class in
their own module, and making them register it first is friction with no payoff.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio_bench.taskbuild.artifacts import resolve_dotted

__all__ = [
    "SolutionBase",
    "SolutionCommon",
    "SolutionIdle",
    "graph_dir",
    "load_solution",
    "resolve_dotted",
    "solution_param",
]


class SolutionCommon:
    """What a policy is regardless of how many worlds it drives.

    The single-world and batched policies are two shapes on purpose -- driving
    one environment and driving N behind a live mask is different work, and
    they are kept as twins so "the batched form of X" is a name you can guess.
    What is *not* different is the part below: both hold the thing they drive,
    both release what they opened, and both report a name the results row is
    keyed by.

    ``name`` was one of only three method bodies identical across the whole
    fleet/single split, so it is the part that merges, and the driving methods
    stay apart because they genuinely differ.

    **What deliberately does not merge is ``__init__``.** A single-world policy
    holds ``self.env`` and a fleet policy holds ``self.fleet``, and that is not
    two names for one thing: ``env.`` is *world* scope and ``fleet.`` is *driver*
    scope. ``test_fleet.py`` enforces the distinction by scanning for ``env.``
    reads and requiring a ``WorldView`` to be able to answer each one -- so
    renaming the fleet's handle to ``env`` makes a driver-level read look like a
    tool-level one, and the suite says so.
    """

    def close(self) -> None:
        """Release whatever the policy opened. Idempotent."""

    @property
    def name(self) -> str:
        return f"{type(self).__module__}.{type(self).__qualname__}"


class SolutionBase(SolutionCommon):
    """Base class for anything that drives an environment.

    Args:
        env: The environment. Held so a policy can read handles, solve IK, or
            request termination; it must not step the environment itself.
    """

    def __init__(self, env, **kwargs: Any) -> None:
        self.env = env
        self.kwargs = kwargs

    def reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        """Called after each :meth:`Env.reset`, before the first action."""

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        """The action for this step."""
        raise NotImplementedError

    def on_step(self, obs: dict[str, Any], reward: float, info: dict[str, Any]) -> None:
        """Called after each step, for a policy that tracks progress."""

    def row_extras(self) -> dict[str, Any]:
        """Extra keys for the episode's results row, read at episode end.

        The seam through which a routine's own telemetry -- per-fling coverage,
        per-drag logs -- lands in ``episode_results.jsonl`` without the run
        being recorded. Keys land at the row's top level; keep them JSON-able
        and distinctively named.
        """
        return {}

    def raw_spans(self) -> list[dict[str, Any]]:
        """Raw state-mutating tool intervals for an episode recorder.

        Policies that do not invoke tools return none.  The rows are lifecycle
        metadata only; simulator state is recorded by the runner's owner
        thread and joined to these step references offline.
        """
        return []

    def recording_stage(self) -> str:
        """Short label for video overlays; empty disables the overlay."""
        return ""

    @property
    def control_mode(self) -> str:
        """Which action space :meth:`get_action` emits into, or empty for the env's.

        A fact about the checkpoint or the script, not about the scene -- a VLA
        trained on LIBERO emits normalized Cartesian deltas whatever task it is
        pointed at, and a planner emits joint targets. Declaring it here is what
        lets the rollout tag each step, so one variant serves both without a
        second copy of it existing to hold the other mode.
        """
        return ""


class SolutionIdle(SolutionBase):
    """Holds the scene where it is. The baseline every task should fail."""

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        del obs
        return self.env.idle_action()


def solution_param(env, name: str, default=None):
    """One ``(:policy-params ...)`` entry from the task's own document.

    A scripted routine takes inputs that are not statements about the world --
    which way to yaw the grasp, how many steps to lift for, which of two layouts
    this variant is -- and the document declares them, in a section fenced off
    from the goal precisely because they steer the *artifact* rather than the
    score.

    Every one of them was also written into the routine as a module constant, so
    each pair was two numbers kept in step by a test. This is the other half of
    that arrangement finally reading: the document is asked first and the
    constant becomes the documented default, which is what it should have been
    all along -- a value for a caller with no document, not a second source of
    truth beside one.

    Args:
        env: The live environment, which knows the document it was built from.
        name: The parameter, spelled as the document spells it (hyphens).
        default: What to answer when there is no document or no such entry.
            Not every task is defined by one, and a routine that refused to run
            without a document would be a routine nobody could reuse.
    """
    doc = getattr(env, "doc", None)
    params = dict(getattr(doc, "policy_params", {}) or {}) if doc is not None else {}
    return params.get(name, default)


def graph_dir(suite: str, relative: str) -> str:
    """A shipped graph directory, from the suite it belongs to.

    Delegates to :func:`robosimstudio_bench.taskbuild.artifacts.graph_dir`, which is
    where it lives now that a suite records the directory it was loaded from.
    Kept as a name here because the graph policies call it and a policy should
    not have to know which package answers.
    """
    from robosimstudio_bench.taskbuild.artifacts import graph_dir as _graph_dir

    return _graph_dir(suite, relative)


def load_solution(path: str, env, **kwargs: Any):
    """Instantiate a policy from a dotted path.

    Args:
        path: ``"package.module:ClassName"`` or ``"package.module.ClassName"``.
            Empty gives :class:`SolutionIdle`.
        env: The environment to bind it to.
        **kwargs: Constructor arguments.
    """
    if not path:
        return SolutionIdle(env, **kwargs)
    return resolve_dotted(path)(env, **kwargs)
