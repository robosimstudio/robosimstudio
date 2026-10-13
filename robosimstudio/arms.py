# SPDX-License-Identifier: Apache-2.0
"""Which arm a runtime ``arm_id`` means, and which slot that arm owns.

**These are two different numbers, and conflating them is a bug in each
direction.** They coincided on every robot this stack shipped, which is why one
name did both jobs until a pairing arrived where they differ.

``arm_id`` is what a *graph* says. A node calls ``robot.go_to_pose(arm_id=0)``
meaning "the hand that does the task", because that is the only thing a
task-level policy can mean without naming a robot. So the order it counts in is
by **role**: the work arm, then the support arm, then whatever else the robot
has.

The **slot** is where that arm's numbers live in an action vector, and it is
fixed by :attr:`~robosimstudio.specs.EmbodimentSpec.arms` -- "the arms,
in the order their slices appear in an action vector", which
:meth:`Env._make_controllers` builds the action spaces from. It is a property of
the model's layout and nothing may reorder it.

Before this module the facade used declaration order for both. That is correct
exactly when the work arm is declared first, which every shipped binding happened
to satisfy: ``yam_i2rt`` lists ``(right, left)`` and claims ``work_arm="right"``;
``yam_bimanual`` lists ``(left, right)`` and tape claims ``work_arm="left"``. The
exception was the one cross-embodiment pairing declared at the time -- the
since-retired cable pick-and-place variant on ``yam_bimanual``, which claimed
``work_arm="right"`` against a robot that declares ``left`` first. There,
``arm_id=0`` resolved to the **idle** hand, and every ``robot.*`` call in a
graph would have driven it. Its successor row, the cable-routing variant on the
same robot, keeps that case in the benchmark's matrix.

The suites that already worked around this did so by computing the index
themselves -- ``list(env.handles.arms).index(self._arm_name())`` appears in four
``tools.py`` files -- which agreed with the facade only because both were
declaration order. :func:`arm_id_of` is that computation, done once and in terms
of the role ordering, so the tools and the facade agree by construction rather
than by coincidence.
"""

from __future__ import annotations

__all__ = ["arm_id_of", "arm_name", "arm_order", "arm_role", "arm_slot", "arm_slots"]


def arm_slots(env) -> tuple[str, ...]:
    """Arm names in **action-vector** order: what the model's layout says.

    ``embodiment.iks`` preserves the spec's declaration order, and
    :meth:`Env._make_controllers` builds every action space from exactly this
    sequence -- so slot ``i`` belongs to ``arm_slots(env)[i]`` and no reordering
    here is safe.
    """
    embodiment = getattr(env.scene, "embodiment", None)
    return tuple(embodiment.iks) if embodiment is not None else ()


def arm_order(env) -> tuple[str, ...]:
    """Arm names in **role** order: work first, then support, then the rest.

    The order a runtime ``arm_id`` counts in. Falls back to declaration order for
    a scene whose binding named no roles, which keeps a robot with one arm --
    where the distinction cannot arise -- reading exactly as it always did.
    """
    slots = arm_slots(env)
    handles = getattr(env, "handles", None)
    if handles is None or len(slots) < 2:
        return slots
    ordered = [name for name in (handles.work_arm, handles.support_arm) if name in slots]
    # dict.fromkeys rather than a set: the tail has to stay in declaration order,
    # and a set would make the second arm of a three-armed robot arbitrary.
    return tuple(dict.fromkeys([*ordered, *slots]))


def arm_name(env, arm_id: int) -> str:
    """The arm one runtime ``arm_id`` means.

    Clamps rather than raising, which is the runtime's own convention: its
    single-arm tools pass ``arm_id=0`` unconditionally and a bimanual graph that
    over-counts should drive the last arm rather than end the episode.
    """
    order = arm_order(env)
    if not order:
        # A bare Sim has no task; its request carries the same name.
        owner = getattr(env, "task", None) or getattr(env, "request", None)
        raise RuntimeError(f"scene {getattr(owner, 'full_name', '?')!r} has no arms")
    return order[min(max(int(arm_id), 0), len(order) - 1)]


def arm_id_of(env, name: str) -> int:
    """The runtime ``arm_id`` for a named arm -- the inverse of :func:`arm_name`.

    What a suite tool returns when it has resolved an arm by *role* and has to
    hand a graph a number for it.
    """
    order = arm_order(env)
    if name not in order:
        raise KeyError(f"no arm {name!r}; have {list(order)}")
    return order.index(name)


def arm_slot(env, name: str) -> int:
    """The action-vector slot a named arm owns.

    Every write into an action vector goes through this rather than through an
    ``arm_id``. The two agree on every robot whose work arm is declared first and
    silently disagree on the rest, and a disagreement here writes one hand's
    joint targets into the other hand's slice.
    """
    slots = arm_slots(env)
    if name not in slots:
        raise KeyError(f"no arm {name!r}; have {list(slots)}")
    return slots.index(name)


def arm_role(env, name: str) -> str:
    """``"work"``, ``"support"``, or ``""`` for an arm the binding did not name."""
    handles = getattr(env, "handles", None)
    if handles is None:
        return ""
    if name and name == handles.work_arm:
        return "work"
    if name and name == handles.support_arm:
        return "support"
    return ""
