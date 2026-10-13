# SPDX-License-Identifier: Apache-2.0
"""Scoring: the evaluator protocol, and the context it is handed.

Two rules make a benchmark number mean something, and both are structural here
rather than a convention anyone has to remember:

* **The environment scores, not the policy.** ``info["success"]`` comes from the
  evaluator the task declared. A policy may *request termination* -- a graph
  reporting it has finished, a VLA emitting a stop token -- and the environment
  will honour it, but the success bit is not the policy's to write.
* **Each task owns its evaluator.** There is no term-composition framework here
  because success conditions are not compositional in practice: "the tape is
  peeled" is not "put_on plus grasp_lift with different thresholds", it is its
  own thing that reads its own task's state. :mod:`~robosimstudio.assetlib.vocabulary.measures`
  offers reusable *pieces*; a suite writes the evaluator that uses them.

:class:`WorldState` exists so the expensive parts happen once per step. Reading
``body_q`` back from device is the single largest cost in scoring, and a naive
evaluator does it once per object; this does it once per step and hands the
array around.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from robosimstudio.scene.state import WorldState

__all__ = [
    "EvalResult",
    "Evaluator",
    "EvaluatorNone",
    "WorldState",
    "diverged_result",
    "evaluator_for",
    "finite_reward",
]


@dataclass
class EvalResult:
    """One step's verdict.

    Attributes:
        success: Whether the task's goal is satisfied *now*. An evaluator that
            latches (a goal that cannot be undone) latches internally and
            reports the latched value here.
        reward: A scalar in [0, 1]. For a multi-goal task the fraction of goal
            predicates satisfied, which is what makes partial progress legible
            without turning the benchmark into a reward-shaping exercise.
        metrics: Everything else worth recording -- per-predicate flags,
            distances, fractions. Written into the episode row, never used to
            decide success.
    """

    success: bool = False
    reward: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)


def diverged_result() -> EvalResult:
    """The verdict for a world whose physics has diverged.

    One definition for the serial :meth:`~robosimstudio_bench.envs.env.Env.evaluate`
    and the batched :func:`~robosimstudio_bench.run.fleet.score_world`, so the same
    broken episode is scored the same whichever runner stepped it: no success,
    no reward, and the evaluator never consulted on NaN state.
    """
    return EvalResult(success=False, reward=0.0, metrics={"diverged": True})


def finite_reward(value: Any) -> float:
    """``value`` as a float, with NaN and infinities read as 0.0.

    A non-finite reward is a measurement that failed, not a score; written into
    a row it poisons every mean taken over the run.
    """
    number = float(value)
    return number if math.isfinite(number) else 0.0


@runtime_checkable
class Evaluator(Protocol):
    """What a task's evaluator must implement.

    Deliberately two methods. A benchmark evaluator is stateful -- consecutive
    grasp counts, latched flags, monotone progress -- and pretending otherwise
    means every task reimplements that state in the environment.
    """

    def reset(self, ctx: WorldState) -> None:
        """Clear per-episode state. Called after the scene is at its start pose."""
        ...

    def __call__(self, ctx: WorldState) -> EvalResult:
        """Score one step."""
        ...


class EvaluatorNone:
    """Scores nothing. For a scene being driven rather than evaluated."""

    def reset(self, ctx: WorldState) -> None:
        del ctx

    def __call__(self, ctx: WorldState) -> EvalResult:
        del ctx
        return EvalResult()


def evaluator_for(task: Any, episode: Any = None) -> Any:
    """A fresh evaluator for one task, built for one episode when the factory can take one.

    ``task.evaluator`` has always been called as ``evaluator(task)``, and every
    hand-written evaluator class still is. A document's factory
    (:func:`~robosimstudio_bench.goals.evaluate.evaluator_factory`) also takes
    ``episode=``, and masks the goal to the episode's
    :attr:`~robosimstudio.specs.Episode.active` set -- a four-object instance
    of an eight-object superset scores four of four, not four of eight. This
    is the one call site that knows which shape it is holding, so the runner,
    the serial environment and a world's view all build theirs here rather
    than each guessing at a signature.

    ``EvaluatorNone`` for a task that declares none.
    """
    import inspect

    factory = getattr(task, "evaluator", None)
    if factory is None:
        return EvaluatorNone()
    if episode is not None:
        try:
            parameters = inspect.signature(factory).parameters
        except (TypeError, ValueError):
            parameters = {}
        takes = "episode" in parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        if takes:
            return factory(task, episode=episode)
    return factory(task)
