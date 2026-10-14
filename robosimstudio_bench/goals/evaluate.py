# SPDX-License-Identifier: Apache-2.0
"""A document, in, and the thing that scores an episode, out.

The top of the asset layer, and the shortest module in it -- which is the point.
Every hard part already has a home: :mod:`~robosimstudio_bench.goals.bind` turns a
clause into a measure, :mod:`~robosimstudio_bench.goals.goal` composes clauses into a
tree, :mod:`~robosimstudio_bench.goals.reward` scores separately from the
goal. This wires the three together and hands the result to the toolkit's
lowering step, which asked for exactly one function.

That is the layer law paying off rather than being asserted. The toolkit does not
import this; it takes a callable, and this is what fills it in.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.registry import ASSETS
from robosimstudio_bench.taskdoc.ast import Condition, TaskDoc
from robosimstudio_bench.taskdoc.parse import parse_condition

from .bind import bind
from .goal import build_goal, mask_condition, maskable_objects
from .reward import Scored, build_reward

__all__ = ["catalog", "evaluator_factory", "make_evaluator"]


def catalog(name: str) -> AssetRecord | None:
    """One asset, or ``None``.

    ``None`` rather than raising, because a document may legitimately name a type
    with no asset behind it -- an imported problem's ``hammer_0 - hammer`` still
    has a body and a position. Whether that is acceptable is the typechecker's
    call, made against the whole document; by the time binding runs it has been
    made.
    """
    try:
        return ASSETS.get(name)
    except KeyError:
        return None


def make_evaluator(doc: TaskDoc, *, task: str = "", active: Iterable[str] | None = None) -> Scored:
    """Build the evaluator for a checked document.

    Args:
        doc: A document that has typechecked.
        task: The task's name, for task-scoped handler overrides.
        active: The objects this episode runs, or ``None`` for every one. When
            given, every goal, observe and reward clause about an object outside it is
            dropped before the tree is built
            (:func:`~robosimstudio_bench.goals.goal.mask_condition`), so an
            instance of a superset is scored on its own objects and the
            parked ones are neither counted against it nor measured at their
            park poses.

    Raises:
        ValueError: On a goal the combinator set does not cover, or one that
            every active object is absent from.
        BindError: On a clause no handler implements for its operands.
    """
    if doc.goal is None:
        raise ValueError(f"task {doc.name!r} states no goal, so there is nothing to score it with")
    names = frozenset(str(n) for n in active) if active is not None else None
    objects = maskable_objects(doc)
    goal = doc.goal if names is None else mask_condition(doc.goal, names, objects)
    if goal is None:
        raise ValueError(
            f"task {doc.name!r}: every goal clause is about an object outside this episode's active set "
            f"{sorted(names or ())}, so there is nothing to score it with"
        )
    observe = (
        doc.observe
        if names is None
        else tuple(c for c in (mask_condition(c, names, objects) for c in doc.observe) if c is not None)
    )

    def leaf(condition: Condition):
        return bind(condition, doc, catalog, task=task or doc.name)

    def node(form: Any, path: str):
        # A reward clause and an observe entry arrive as raw parsed forms rather
        # than conditions, because `_plain` reads those sections. They are parsed
        # here with the goal's own parser, so a quantity cannot mean one thing in
        # a goal and something else in a reward. Masked like the goal, so a
        # reward about a parked object is not measured at its park pose; and
        # labelled under its own path, so its verdicts do not land on the goal's.
        condition = form if isinstance(form, Condition) else parse_condition(form)
        if names is not None:
            condition = mask_condition(condition, names, objects)
        return None if condition is None else build_goal(condition, leaf, path=path)

    return Scored(
        goal=build_goal(goal, leaf),
        reward=build_reward(
            doc.reward,
            lambda form: node(form, "reward"),
            make_mask=lambda form: node(form, "reward.gated_by"),
        ),
        observe=tuple(build_goal(c, leaf, path=f"observe.{i}") for i, c in enumerate(observe)),
        doc=doc,
        # The binder this goal was built with, so verifying `:init` measures a
        # constraint the way the goal would rather than through a second binder
        # free to disagree with the first.
        leaf=leaf,
        active=names,
    )


def evaluator_factory(doc: TaskDoc, *, task: str = "") -> Callable[..., Scored]:
    """A factory in the shape ``TaskSpec.evaluator`` has always held.

    The runtime constructs its evaluator as ``task.evaluator(task)`` and does so
    **per environment**, deliberately: a shared instance would carry one world's
    latched ``once`` into another's, which is a bug that only appears under fleet
    execution and looks like a task succeeding before it starts.

    So this returns a factory that builds a fresh tree each call -- and builds one
    *now*, discarded, so that a goal nothing can measure fails while the task is
    being compiled rather than when somebody constructs their third environment.
    The eager build is the whole reason this is not simply a lambda.

    The factory also takes ``episode=`` -- and it is
    :func:`~robosimstudio_bench.goals.context.evaluator_for` that passes one,
    for every caller -- so a tree is built for the episode's
    :attr:`~robosimstudio.specs.Episode.active` set. An episode that names
    none gets the whole goal, exactly as before.

    Args:
        doc: A document that has typechecked.
        task: The task's name, for task-scoped handler overrides.
    """
    make_evaluator(doc, task=task)

    def build(spec=None, *, episode=None) -> Scored:
        del spec  # the tree is bound to the document, not to the spec
        return make_evaluator(doc, task=task, active=getattr(episode, "active", None))

    return build
