# SPDX-License-Identifier: Apache-2.0
"""Which (task, scene, robot, hand) combinations are admissible, and why not.

The pairing question, which needs all three axes at once: a task\'s declared
requirements, the scene it would stand in, and what the robot library says about
the hardware. It lives here rather than with the robot registry because only one
of its three inputs is a fact about hardware -- :func:`_admits_gripper` answers
that half and is imported.

A refusal always carries its reason, because the whole point is to say *why* a
combination is not offered rather than to leave it silently missing.
"""

from __future__ import annotations

from robosimstudio.robots.registry import (
    _admits_gripper,
    attachable_grippers,
    get_embodiment,
    get_gripper,
    hands_of,
    list_embodiments,
    resolve_gripper,
)
from robosimstudio.scene.registry import get_cell, list_cells
from robosimstudio.specs import Binding

from .registry import SUITES

__all__ = [
    "admits",
    "candidate_pairings",
    "candidate_triples",
    "declared_key",
    "list_pairs",
    "pairing_key",
]


def pairing_key(task, binding: Binding) -> tuple[str, str, str]:
    """A binding's key with an unstated hand resolved to the robot's own.

    :meth:`Binding.key` leaves ``""`` alone because a spec may not reach for the
    registry. This is the spelling two bindings have to agree on to be the same
    pairing -- one that says nothing about the hand and one that names the hand
    the robot already wears are the same combination, and a matrix that listed
    both would be reporting one row twice.
    """
    scene, embodiment, gripper = binding.key(task)
    if gripper or not embodiment:
        return (scene, embodiment, gripper)
    own = hands_of(get_embodiment(embodiment))
    return (scene, embodiment, own[0] if len(own) == 1 else "")


def declared_key(task, scene: str = "", embodiment: str = "", gripper: str = "") -> tuple[str, str, str] | None:
    """The declared binding's own ``(scene, embodiment, gripper)`` for a request, or ``None``.

    The raw :meth:`Binding.key` of the binding whose :func:`pairing_key` equals
    the request's, so naming the hand the robot already wears finds the binding
    that left it unstated -- the one :func:`admits` calls the native pairing.
    Callers swap the request for this key before :meth:`TaskSpec.binding_for`,
    which compares raw keys; otherwise the declared binding's graphs and mount
    are lost to a bare one synthesised for the other spelling, or the run paths
    refuse what :func:`admits` admitted. Empty arguments mean the task's own, as
    they do for ``binding_for``.
    """
    declared = task.bindings or (task.native_binding,)
    want = (scene or task.scene, embodiment or task.embodiment, gripper or task.gripper)
    for binding in declared:
        if binding.key(task) == want:
            return want
    try:
        asked = pairing_key(task, Binding(scene=want[0], embodiment=want[1], gripper=want[2]))
    except LookupError:
        # A robot the registry does not know; ``binding_for`` names it better.
        return None
    for binding in declared:
        if pairing_key(task, binding) == asked:
            return binding.key(task)
    return None


def list_pairs() -> list[tuple[str, str, str, str, bool, tuple[str, ...]]]:
    """Every declared ``(task, scene, embodiment, gripper, is_native, graphs)``.

    The compatibility matrix, read off the bindings the tasks themselves
    declare. ``is_native`` marks the pairing a task was authored against -- the
    one its reproduction claims and its shipped graphs are about -- and
    ``graphs`` is what is known to work on that pairing specifically, which is
    empty for a pairing that compiles and can be teleoperated but has no policy.
    """
    out: list[tuple[str, str, str, str, bool, tuple[str, ...]]] = []
    for _, suite in SUITES:
        for task in suite.variants:
            native = pairing_key(task, task.native_binding)
            for binding in task.bindings or (task.native_binding,):
                scene, embodiment, gripper = binding.key(task)
                key = pairing_key(task, binding)
                out.append((task.full_name, scene, embodiment, gripper, key == native, binding.graphs))
    return sorted(out)


def admits(task, scene: str, embodiment: str, gripper: str = "") -> tuple[bool, str]:
    """Whether a pairing is worth trying, and why not when it is not.

    Declared bindings win outright: a task that states a pairing has already
    answered this, and one that states none is admissible on its native pairing
    by construction. Everything else is checked against
    :class:`~robosimstudio.specs.TaskRequirements`, which holds only
    what can be read off the specs -- arm count, a raised surface, a gripper
    span. Reach, contact budgets and whether a routine's waypoints still solve
    are *not* here: they are measurements, and a predicate that guessed at them
    would reject working pairings while admitting broken ones. The matrix tool
    finds those out by compiling.

    What a *robot* says about itself is checked too, and the two are not the same
    kind of claim: :attr:`~robosimstudio.specs.EmbodimentSpec.unbuildable` and
    :attr:`~robosimstudio.specs.EmbodimentSpec.monolithic` are facts about the
    files on disk, not measurements, so they refuse here rather than being
    discovered by a compile whose message is about a label.

    ``gripper`` is the fourth axis and ``""`` means the robot's own hand. The
    three rules it adds are all about whether the hand can *get onto* the arm --
    which is a fact about the two specs, so it belongs here rather than being
    found out by a compile that fails in the importer.

    Returns:
        ``(admissible, reason)``; ``reason`` is empty when it is.
    """
    if (scene, embodiment, gripper) in task.binding_keys:
        return True, ""
    if embodiment:
        # The same pairing spelled the other way -- ``""`` against the hand the
        # robot already wears. Compared through :func:`pairing_key` so naming the
        # robot's own hand is not refused by a rule a declared pairing outranks.
        asked = pairing_key(task, Binding(scene=scene, embodiment=embodiment, gripper=gripper))
        if asked in {pairing_key(task, b) for b in task.bindings or (task.native_binding,)}:
            return True, ""
    need = task.requires
    if need.monolithic:
        # Both axes: the robot is inside the arena this task imports, so another
        # embodiment would be built *beside* it rather than instead of it.
        return False, "scene and robot arrive in one import; needs the robot-stripped variant"
    if not embodiment:
        # A task with no robot -- a material rig. The arm and hand rules have
        # nothing to apply to, so only the scene's own requirement is left.
        # Reached when a menu offers this task on a scene it did not declare.
        if need.arms:
            return False, f"needs {need.arms} arm(s) and names no robot"
        if need.needs_surface and not get_cell(scene).has_surface:
            return False, f"needs a raised work surface; {scene!r} is bare ground"
        return True, ""
    spec = get_embodiment(embodiment)
    if spec.unbuildable:
        # Ahead of every other robot-side rule because it outranks all of them:
        # arm count, work surface and gripper span all pass for a robot whose
        # scene never finishes compiling, and what the caller then gets is a
        # traceback about a label rather than a sentence about the pairing.
        return False, spec.unbuildable
    if spec.monolithic:
        # The other direction of the same rule: not "this task's scene has a
        # robot in it" but "this robot's asset has a scene in it". Standing one
        # in a foreign cell imports the room it came from.
        return False, f"{embodiment!r} arrives with its own scene; it can only stand where that scene already is"
    if len(spec.arms) < need.arms:
        return False, f"needs {need.arms} arm(s), {embodiment!r} has {len(spec.arms)}"
    if need.needs_surface and not get_cell(scene).has_surface:
        return False, f"needs a raised work surface; {scene!r} is bare ground"
    ok, why = _admits_gripper(spec, gripper)
    if not ok:
        return False, why
    if need.min_gripper_span:
        # Off the hand, through its own effective_span, rather than restating
        # the derivation here -- a rule spelled in two places is a rule that
        # will disagree once, and this copy and the compiler's were the two.
        # The *selected* hand, not the widest the robot could wear: once the
        # gripper is an axis, "opens wide enough" is a question about the
        # combination being compiled.
        hands = (
            [get_gripper(gripper)]
            if gripper
            else [h for h in (resolve_gripper(spec, a) for a in spec.arms) if h is not None]
        )
        if not hands:
            return False, f"needs a {need.min_gripper_span:.3f} m span; {embodiment!r} declares no gripper"
        widest = max(h.effective_span for h in hands)
        if widest < need.min_gripper_span:
            name = gripper or embodiment
            return False, f"needs a {need.min_gripper_span:.3f} m span, {name!r} opens to {widest:.3f} m"
    return True, ""


def candidate_pairings() -> list[tuple[str, str, str, str, bool, str]]:
    """Every ``(task, scene, embodiment, gripper, is_native, reason)`` worth trying.

    The full product filtered by :func:`admits`, with the rejected ones carried
    along and their reason attached -- a matrix that silently omitted them would
    read as "these were fine" rather than "these were ruled out, here is why".

    The gripper axis takes ``""`` plus every *attachable* hand rather than every
    registered one. A descriptive hand is not a value this axis can hold: it is
    already covered by ``""`` on the robot that contains it, and on every other
    robot it is a row whose only content is the same refusal.
    """
    out: list[tuple[str, str, str, str, bool, str]] = []
    scenes, embodiments = list_cells(), list_embodiments()
    grippers = ["", *attachable_grippers()]
    for _, suite in SUITES:
        for task in suite.variants:
            native = pairing_key(task, task.native_binding)
            for scene in scenes:
                for embodiment in embodiments:
                    for gripper in grippers:
                        ok, why = admits(task, scene, embodiment, gripper)
                        key = pairing_key(task, Binding(scene=scene, embodiment=embodiment, gripper=gripper))
                        out.append((task.full_name, scene, embodiment, gripper, key == native, "" if ok else why))
    return sorted(out)


def candidate_triples() -> list[tuple[str, str, str, bool, str]]:
    """:func:`candidate_pairings` restricted to each robot's own hand.

    The three-axis view, kept because "which robots can host this task" is a
    question people ask independently of which hand is on them.
    """
    return [(t, s, e, native, why) for t, s, e, g, native, why in candidate_pairings() if not g]
