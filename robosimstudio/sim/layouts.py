# SPDX-License-Identifier: Apache-2.0
"""Seeded spatial variants of a scene, in the work-surface frame.

A Real2Sim loop that has one layout of the bar and the sponge is a loop that
overfits that layout. The paper's sponge-wiping study therefore built thirty
spatial variants of the same scene and improved the policy across them. The
benchmark tier already does this with a document's ``:sampler``; a scene
assembled by hand had no equivalent, so every caller rebuilt the scene in a
loop and restated every ``at=``.

:func:`layouts` is that missing loop: it clones the scene and jitters named
things in the same frame ``at=`` uses. Each clone is an independent
:class:`~robosimstudio.sim.scene.Scene`, so it can be built without rebinding
the original's refs -- which is the contract :meth:`Scene.build` already states.

Nothing here steps physics. A variant is a *declaration*, the way the original
was; :meth:`~robosimstudio.sim.scene.Scene.build` is still the moment it becomes
a sim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from robosimstudio.core.math import quat_from_axis_angle, quat_multiply

from .scene import Robot, Scene

__all__ = ["clone", "layouts", "place"]

_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def clone(scene: Scene) -> Scene:
    """An independent copy of ``scene``: same cell, physics and things; its own refs.

    Building the copy does not rebind the original's handles. That is the
    reason this exists rather than ``build()`` twice on one ``Scene``, which
    :meth:`Scene.build` documents as following the newer sim.
    """
    copy = Scene(
        cell=scene.cell,
        physics=scene.physics,
        action=scene.action,
        name=scene.name,
        binding=scene.binding,
        render=scene.render,
    )
    if scene._robot is not None:
        copy.add(scene._robot)
    for spec in scene._specs:
        copy.add(spec)
    for spec in scene.regions:
        copy.add(spec)
    for spec in scene.cameras:
        copy.add(spec)
    return copy


def place(scene: Scene, name: str, *, at: Sequence[float] | None = None, yaw: float | None = None) -> Scene:
    """Move one thing, in the work-surface frame, and return ``scene`` so it chains.

    Takes effect on the next :meth:`Scene.build`. A sim already built from this
    scene is not moved: that would be writing a pose onto a live body, which is
    the shortcut the live page forbids.

    Args:
        scene: The scene that holds the thing.
        name: What :meth:`Scene.add` named it.
        at: A position ``(x, y, z)`` or a pose ``(x, y, z, qx, qy, qz, qw)`` in
            the cell's work-surface frame, the same numbers a constructor's
            ``at=`` takes. Omitted, the translation is left alone.
        yaw: Extra rotation about the surface normal [rad], composed on the
            thing's current (or ``at=``) orientation. Refused together with a
            seven-number ``at=``, which already states a rotation.

    Raises:
        KeyError: If nothing in the scene is called ``name``.
        TypeError: If neither ``at`` nor ``yaw`` is given.
        ValueError: If the thing has no work-surface pose to amend, if ``at``
            is the wrong length, or if ``yaw`` is given with a seven-number
            ``at``.
    """
    if at is None and yaw is None:
        raise TypeError(f"place({name!r}) needs at= or yaw=")
    spec = _lookup(scene, name)
    values = None if at is None else tuple(float(v) for v in at)
    if values is not None and len(values) not in (3, 7):
        raise ValueError(
            f"place({name!r}): at= is a position (x, y, z) or a pose (x, y, z, qx, qy, qz, qw) "
            f"in the work-surface frame, and it was given {len(values)} numbers"
        )
    if values is not None and len(values) == 7 and yaw is not None:
        raise ValueError(f"place({name!r}): at= as a pose already has a rotation; do not also pass yaw=")
    _replace(scene, name, _placed(spec, name, at=values, yaw=yaw))
    return scene


def layouts(
    scene: Scene,
    n: int,
    *,
    seed: int = 0,
    xy: float | Mapping[str, float] = 0.0,
    yaw: float | Mapping[str, float] = 0.0,
    z: float | Mapping[str, float] = 0.0,
    names: Sequence[str] | None = None,
) -> tuple[Scene, ...]:
    """``n`` independent clones of ``scene``, each with named things jittered in the work-surface frame.

    A half-width of ``0.05`` draws uniformly from ``[-0.05, 0.05]`` metres (or
    radians, for ``yaw``) around the thing's current pose. A mapping names the
    things it applies to; a scalar applies to every eligible thing, or to
    ``names`` when that is given. Positions that would leave a bounded work
    surface are clipped to it, so a wide draw still sits on the bench.

    The original scene is not modified. Layout ``k`` of ``layouts(..., seed=s)``
    is the same scene every time.

    Args:
        scene: The scene to vary. It is not built; each clone is a declaration.
        n: How many layouts to draw. At least one.
        seed: Passed to :func:`numpy.random.default_rng`.
        xy: Half-width of the draw in x and in y [m].
        yaw: Half-width of the heading draw about the surface normal [rad].
        z: Half-width of the height draw [m]. Leave at zero to keep things on
            the work surface.
        names: Restrict which things move. Default is every free body that
            states a work-surface pose (not a welded static, not a robot
            standing at its own mount).

    Raises:
        ValueError: If ``n < 1``, if a named thing is not in the scene or
            cannot be moved, or if a mapping names one.
    """
    count = int(n)
    if count < 1:
        raise ValueError(f"layouts needs n >= 1, got {n}")
    movable = _movable(scene)
    if names is None:
        chosen = tuple(movable)
    else:
        chosen = tuple(names)
        missing = [name for name in chosen if name not in scene]
        if missing:
            raise ValueError(f"layouts: nothing called {missing} in this scene; there is {sorted(scene._refs)}")
        unmoved = [name for name in chosen if name not in movable]
        if unmoved:
            raise ValueError(
                f"layouts: {unmoved} cannot be moved in the work-surface frame; "
                "a welded static, a camera, a region, or a robot with no at= has no pose to jitter"
            )
    for label, widths in (("xy", xy), ("yaw", yaw), ("z", z)):
        _check_mapping(scene, widths, label, movable)
    rng = np.random.default_rng(int(seed))
    out: list[Scene] = []
    clip = _clipper(scene)
    for index in range(count):
        variant = clone(scene)
        variant.name = f"{scene.name}_{index:03d}"
        for name in chosen:
            dx = rng.uniform(-w, w) if (w := _width(xy, name, chosen)) else 0.0
            dy = rng.uniform(-w, w) if (w := _width(xy, name, chosen)) else 0.0
            dz = rng.uniform(-w, w) if (w := _width(z, name, chosen)) else 0.0
            dyaw = rng.uniform(-w, w) if (w := _width(yaw, name, chosen)) else 0.0
            _replace(variant, name, _shifted(_lookup(variant, name), dx, dy, dz, dyaw, clip_xy=clip))
        out.append(variant)
    return tuple(out)


def _check_mapping(scene: Scene, widths: float | Mapping[str, float], label: str, movable: tuple[str, ...]) -> None:
    """Refuse a mapping that names something the scene does not hold, or cannot move."""
    if not isinstance(widths, Mapping):
        return
    missing = [name for name in widths if name not in scene]
    if missing:
        raise ValueError(f"layouts: {label} names {missing}, and the scene has {sorted(scene._refs)}")
    unmoved = [name for name in widths if name not in movable]
    if unmoved:
        raise ValueError(f"layouts: {label} names {unmoved}, which cannot be moved in the work-surface frame")


def _width(spec: float | Mapping[str, float], name: str, chosen: Sequence[str]) -> float:
    """The half-width that applies to ``name``: a mapping's own entry, or the scalar when ``name`` is chosen."""
    if isinstance(spec, Mapping):
        return float(spec.get(name, 0.0))
    return float(spec) if name in chosen else 0.0


def _movable(scene: Scene) -> tuple[str, ...]:
    """Names :func:`layouts` will jitter when ``names`` is omitted, in add order."""
    names: list[str] = []
    if scene._robot is not None and _can_move(scene._robot):
        names.append(scene._robot.label)
    for spec in scene._specs:
        if _can_move(spec):
            names.append(spec.name)
    return tuple(names)


def _can_move(spec: Any) -> bool:
    """Whether ``spec`` states a work-surface pose a draw can amend.

    A robot without ``at=`` stands at its own mount, which is not a pose this
    module should invent. A welded static and a ``world_xform`` body are
    fixtures of the place, not things a spatial variant of the *task* moves.
    """
    if isinstance(spec, Robot):
        return spec.at is not None
    if getattr(spec, "static", False) or getattr(spec, "world_xform", False):
        return False
    return hasattr(spec, "pose") and hasattr(spec, "quat")


def _lookup(scene: Scene, name: str) -> Any:
    if scene._robot is not None and scene._robot.label == name:
        return scene._robot
    for spec in scene._specs:
        if getattr(spec, "name", "") == name:
            return spec
    if name not in scene:
        raise KeyError(f"nothing called {name!r} in this scene; there is {sorted(scene._refs)}")
    raise ValueError(
        f"{name!r} cannot be moved in the work-surface frame; "
        "a camera, a region or a welded static has no pose to amend"
    )


def _replace(scene: Scene, name: str, spec: Any) -> None:
    if scene._robot is not None and scene._robot.label == name:
        scene._robot = spec
        return
    for index, current in enumerate(scene._specs):
        if getattr(current, "name", "") == name:
            scene._specs[index] = spec
            return
    raise KeyError(f"nothing called {name!r} in this scene; there is {sorted(scene._refs)}")


def _placed(spec: Any, name: str, *, at: tuple[float, ...] | None, yaw: float | None) -> Any:
    """``spec`` with ``at`` and/or ``yaw`` applied, or a refusal."""
    if isinstance(spec, Robot):
        current = spec.at or (0.0, 0.0, 0.0, *_IDENTITY_QUAT)
        if at is None:
            x, y, z, *quat = current
        elif len(at) == 3:
            x, y, z = at
            quat = list(current[3:])
        else:
            x, y, z = at[:3]
            quat = list(at[3:])
        if yaw:
            quat = list(_yawed(quat, yaw))
        return replace(spec, at=(float(x), float(y), float(z), *[float(v) for v in quat]))
    if not (hasattr(spec, "pose") and hasattr(spec, "quat")):
        raise ValueError(
            f"{name!r} cannot be moved in the work-surface frame; "
            "a camera, a region or a welded static has no pose to amend"
        )
    pose = tuple(float(v) for v in spec.pose)
    quat = tuple(float(v) for v in spec.quat)
    if at is None:
        x, y, z = pose[:3]
    elif len(at) == 3:
        x, y, z = at
    else:
        x, y, z = at[:3]
        quat = at[3:]
    if yaw:
        quat = tuple(float(v) for v in _yawed(quat, yaw))
    return replace(spec, pose=(x, y, z), quat=quat)


def _shifted(
    spec: Any,
    dx: float,
    dy: float,
    dz: float,
    dyaw: float,
    *,
    clip_xy,
) -> Any:
    """``spec`` translated by ``(dx, dy, dz)`` and yawed by ``dyaw``, clipped in xy."""
    if isinstance(spec, Robot):
        assert spec.at is not None
        x, y, z = spec.at[0] + dx, spec.at[1] + dy, spec.at[2] + dz
        x, y = clip_xy(x, y)
        quat = _yawed(spec.at[3:], dyaw)
        return replace(spec, at=(float(x), float(y), float(z), *[float(v) for v in quat]))
    x, y, z = float(spec.pose[0]) + dx, float(spec.pose[1]) + dy, float(spec.pose[2]) + dz
    x, y = clip_xy(x, y)
    quat = _yawed(spec.quat, dyaw)
    return replace(spec, pose=(x, y, z), quat=tuple(float(v) for v in quat))


def _yawed(quat: Sequence[float], yaw: float) -> np.ndarray:
    """``quat`` with a rotation of ``yaw`` rad about +z composed on the left.

    Left composition is a yaw in the parent (work-surface) frame, which is the
    heading a layout varies: the thing turns on the bench, it does not roll
    about its own axis.
    """
    if abs(float(yaw)) < 1.0e-15:
        return np.asarray(quat, dtype=np.float64)
    return quat_multiply(quat_from_axis_angle((0.0, 0.0, 1.0), float(yaw)), quat)


def _clipper(scene: Scene):
    """An ``(x, y) -> (x, y)`` that keeps a point on the cell's work surface.

    Unbounded cells (``void``, a cell that states no ``half``) pass through.
    A missing cell name is treated the same way: there is no surface to leave.
    """
    if not scene.cell:
        return lambda x, y: (x, y)
    from robosimstudio.scene.registry import get_cell

    try:
        cell = get_cell(scene.cell)
    except KeyError:
        return lambda x, y: (x, y)
    if cell.half is None:
        return lambda x, y: (x, y)
    hx, hy = float(cell.half[0]), float(cell.half[1])

    def clip(x: float, y: float) -> tuple[float, float]:
        return max(-hx, min(hx, x)), max(-hy, min(hy, y))

    return clip
