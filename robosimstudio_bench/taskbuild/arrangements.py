# SPDX-License-Identifier: Apache-2.0
"""Named physics arrangements and camera rigs, by versioned name.

A document stages the two of them by *name* -- ``(:physics cloth_vbd@ref)``,
``(:cameras agentview@v1)`` -- and says nothing about what either is, which is
the point: a solver's contact stiffnesses and a camera's field of view are
calibrated facts about a setup, not statements a task is in a position to make.

Registered here, they resolve the way an import profile does -- and this is
deliberately the same shape as :data:`~staging.PROFILES`, down to the refusals,
because it is the same problem. **Content registers; the toolkit only holds the
table.** ``tasks/<name>/staging.py`` populates both at import, and the direction
stays content-to-library, which is what lets lowering keep its claim to being a
pure function of a document plus a registry.

An arrangement is an :class:`~robosimstudio.specs.physics.ArrangementSpec` and
nothing else: the rates, a backend per physics domain, the coupler and the
collision policy, all data. There is no solver factory beside it any more --
the composer builds the solvers from the spec and the assets' declarations, and
what it built is the manifest a repro bundle carries. A name therefore means
exactly what its registration says, and a registration can be diffed.

The caller-supplied dicts do not go away, and should not. They are the injection
seam that keeps the structural tests registry-free, and they are how a loader
supplies the one thing a registration cannot: a rig *aimed* at a layout the
document has no coordinates for. A supplied name wins over a registered one.
"""

from __future__ import annotations

import operator

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.registry import Registry
from robosimstudio.specs import ArrangementSpec

__all__ = [
    "CAMERA_RIGS",
    "PHYSICS",
    "get_arrangement",
    "get_camera_rig",
    "list_camera_rigs",
    "list_physics",
    "register_arrangement",
    "register_camera_rig",
]


PHYSICS: Registry[ArrangementSpec] = Registry("physics arrangement", same=operator.eq)
"""Registered physics arrangements, by versioned name."""

CAMERA_RIGS: Registry[tuple[CameraSpec, ...]] = Registry("camera rig", same=operator.eq)
"""Registered camera rigs, by versioned name."""


def _check_name(name: str, what: str) -> None:
    """Refuse an unversioned name, for the reason a profile refuses one.

    An arrangement revised in place re-stages every task that named it, and does
    it silently -- the documents do not change, so nothing in a diff says the
    physics under them did.
    """
    if "@" not in name:
        raise ValueError(
            f"{what} {name!r} carries no version. Write {name}@v1 -- "
            f"a {what} revised in place restages every task that named it, silently."
        )


def register_arrangement(name: str, spec: ArrangementSpec) -> ArrangementSpec:
    """Register one arrangement under a versioned name.

    Returns the registered spec, so a module can keep saying
    ``X = register_arrangement("x@v1", ArrangementSpec(...))`` and have ``X``
    be the spec its readers want.

    Raises:
        ValueError: On an unversioned name, or a *different* arrangement under a
            taken name. A redundant copy is fine: two tasks that each ship the
            cell they stand in each declare its arrangement, and refusing the
            second would force them to share a directory -- which is the thing
            task self-containment is trying to do without.
    """
    _check_name(name, "arrangement")
    if not isinstance(spec, ArrangementSpec):
        raise TypeError(f"arrangement {name!r} must be an ArrangementSpec, got {type(spec).__name__}")
    return PHYSICS.register(name, spec, origin=_origin())


def register_camera_rig(name: str, rig: tuple[CameraSpec, ...]) -> tuple[CameraSpec, ...]:
    """Register one camera rig. Refuses an unversioned name, or a *different* rig under a taken one."""
    _check_name(name, "camera rig")
    return CAMERA_RIGS.register(name, tuple(rig), origin=_origin())


def _origin() -> str:
    """The module that called a register function, for the clash message."""
    import inspect

    frame = inspect.currentframe()
    caller = frame.f_back.f_back if frame and frame.f_back else None
    return caller.f_globals.get("__name__", "") if caller else ""


def get_arrangement(name: str) -> ArrangementSpec:
    return PHYSICS.get(name)


def get_camera_rig(name: str) -> tuple[CameraSpec, ...]:
    return CAMERA_RIGS.get(name)


def list_physics() -> list[str]:
    return PHYSICS.names()


def list_camera_rigs() -> list[str]:
    return CAMERA_RIGS.names()


# ---------------------------------------------------------------------------
# What the library ships to name
# ---------------------------------------------------------------------------
#
# These are the stock entries: ``GENERIC`` at the rates a rigid or a soft
# scene wants. A foreign-engine arrangement is a suite's own to declare --
# ``surface_prep`` registers its SuperDex one beside its documents.


def _install_defaults() -> None:
    from dataclasses import replace

    from robosimstudio.specs.physics import GENERIC, MUJOCO_500HZ

    register_arrangement("rigid_mujoco@v1", replace(GENERIC, substeps=4))
    """A rigid-only scene: ``GENERIC`` at the four substeps its docstring says a rigid scene should state."""
    register_arrangement("mujoco_500hz@v1", MUJOCO_500HZ)
    """The arm and every free rigid body in one MuJoCo island: ``rss.MUJOCO_500HZ``, under its task-language name.

    The bundle a hand can hold a rigid object in. ``GENERIC`` (and
    ``rigid_mujoco@v1``) puts free bodies on VBD behind a proxy coupler, which
    pushes and cannot pinch. The rates and contact law are the tool-sorting
    benchmark's own, measured there."""
    register_arrangement("cloth_vbd@v1", GENERIC)
    register_arrangement("rod_vbd@v1", GENERIC)
    """A sheet or a cable on VBD beside a MuJoCo arm, exactly as ``GENERIC`` composes them."""


_install_defaults()
