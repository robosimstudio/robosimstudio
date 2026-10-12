# SPDX-License-Identifier: Apache-2.0
"""Which robots and which hands exist, and which of them go together.

Two registries and the rules for pairing them. A hand is its own registry rather
than a field on an embodiment, because it is the axis the shipped robots differ
along: ``panda`` and ``panda_robotiq`` are one Franka with two hands,
``yam_bimanual`` and ``yam_i2rt`` one YAM with two. Naming a
:class:`~robosimstudio.specs.GripperSpec` from an
:class:`~robosimstudio.specs.ArmSpec` is what makes that a pairing rather than a
fifth robot with ten constants copied into it.

Whether a *task* admits a robot is not decided here -- that needs the task's own
requirements and its scene, and lives with them. What lives here is the half
that is a fact about the hardware: :func:`hands_of`, :func:`attachable_grippers`
and :func:`_admits_gripper` answer "can this hand go on this arm at all", which
is true or false before any task is named.
"""

from __future__ import annotations

from robosimstudio.registry import Registry
from robosimstudio.specs import ArmSpec, EmbodimentSpec, GripperSpec

__all__ = [
    "EMBODIMENTS",
    "GRIPPERS",
    "attachable_grippers",
    "get_embodiment",
    "get_gripper",
    "hands_of",
    "list_embodiments",
    "list_grippers",
    "register_embodiment",
    "register_gripper",
    "resolve_gripper",
]

EMBODIMENTS: Registry[EmbodimentSpec] = Registry(
    "embodiment",
    ("robosimstudio.robots.embodiments",),
    hint="A robot shipped as content registers when the benchmark tier is imported: where it is installed, "
    "import its taskbuild package and try again. Without it, the robots listed are all there are.",
)
GRIPPERS: Registry[GripperSpec] = Registry("gripper", ("robosimstudio.robots.grippers",))


def register_embodiment(spec: EmbodimentSpec) -> EmbodimentSpec:
    """Register an embodiment spec."""
    return EMBODIMENTS.register(spec.name, spec, origin=getattr(spec, "__module__", ""))


def register_gripper(spec: GripperSpec) -> GripperSpec:
    """Register a gripper spec. Call at module scope in the hand's own module."""
    return GRIPPERS.register(spec.name, spec, origin=getattr(spec, "__module__", ""))


def resolve_gripper(spec: EmbodimentSpec, arm: ArmSpec) -> GripperSpec | None:
    """The hand on one arm, looked up and templated, or ``None`` for a bare arm.

    Arm first, then the embodiment's default, so a bimanual robot names its hand
    once and an arm that differs says so.

    The lookup is lazy on purpose: an embodiment module names a gripper as a
    *string* at import and never triggers the gripper registry's own load, so the
    two packages have no import-order relationship to get wrong.
    """
    key = arm.gripper or spec.gripper
    return GRIPPERS.get(key).for_arm(arm.name) if key else None


def get_embodiment(name: str) -> EmbodimentSpec:
    return EMBODIMENTS.get(name)


def get_gripper(name: str) -> GripperSpec:
    return GRIPPERS.get(name)


def list_embodiments() -> list[str]:
    """Every registered embodiment name, sorted."""
    return EMBODIMENTS.names()


def list_grippers() -> list[str]:
    """Every registered gripper name, sorted."""
    return GRIPPERS.names()


def hands_of(spec: EmbodimentSpec) -> list[str]:
    """The registry keys of the hands this robot declares, in arm order.

    Deduplicated, because a bimanual robot with one kind of hand names it once
    and both arms resolve to the same entry. Empty for a bare robot.
    """
    out: list[str] = []
    for arm in spec.arms:
        key = arm.gripper or spec.gripper
        if key and key not in out:
            out.append(key)
    return out


def attachable_grippers() -> list[str]:
    """Every registered hand that carries a file of its own, sorted.

    The gripper axis's *values*: a descriptive hand can only be indexed inside
    the arm import that already contains it, so pointing one at another robot
    would match nothing. A hand with an asset can be welded onto any imported
    wrist, which is what makes it a thing to vary rather than a thing to look up.
    """
    return sorted(name for name in GRIPPERS.names() if GRIPPERS.get(name).asset is not None)


def _admits_gripper(spec: EmbodimentSpec, gripper: str) -> tuple[bool, str]:
    """Whether this hand can be put on this robot at all.

    Three ways it cannot, each of which would otherwise surface as a build
    failure whose message is about labels rather than about the pairing:

    * a *descriptive* hand indexes joints and bodies that only exist inside the
      arm import that already contains it, so on any other robot it matches
      nothing;
    * a robot that arrives inside its task's scene export is never imported here,
      so there is no wrist for a weld to bolt onto;
    * a robot whose asset declares two arms as two articulation roots cannot take
      a weld at all -- Newton accepts only the most recently added articulation
      as a parent, so one arm would get its hand and the other would not.
    """
    if not gripper:
        return True, ""
    own = hands_of(spec)
    if gripper in own:
        return True, ""
    hand = get_gripper(gripper)
    if hand.asset is None:
        return False, f"hand {gripper!r} indexes another asset's labels and carries no file to attach"
    if spec.asset is None:
        return False, f"{spec.name!r} arrives inside its scene's import; there is no wrist to weld {gripper!r} onto"
    if len(spec.arms) > 1:
        return False, f"{spec.name!r} is two articulations in one asset; a hand cannot be welded onto one of them"
    wrists = {a.ee_body_name for a in spec.arms if a.ee_body_name}
    if hand.mount_body and hand.mount_body not in wrists:
        # A mount transform is measured against one flange, and naming the body
        # is how a hand says which. Read off the specs, so it belongs here: the
        # alternative is a compile that dies on `'wrist_3_link' matched 0
        # labels`, which is a true statement about labels and tells a reader
        # nothing about the pairing.
        return False, (
            f"hand {gripper!r} bolts to a body called {hand.mount_body!r}, measured against that "
            f"flange; {spec.name!r}'s wrist is {sorted(wrists) or 'unnamed'}"
        )
    return True, ""
