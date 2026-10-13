# SPDX-License-Identifier: Apache-2.0
"""Unprivileged ``geometry.*`` tools: a box in, grasp poses out.

These are the shared geometry helpers a graph reaches for between *seeing* a
thing and *moving* to it. They are pure functions of their arguments -- no
simulator, no environment, no bus -- which is what makes them unprivileged in
the structural sense rather than the polite one: there is nothing here for a
graph to learn that its caller did not already hand it. A box fitted to a point
cloud and a box read out of the simulator produce the same poses, and that
substitutability is the whole reason ``sim.get_object_obb`` returns a
``gap_core.types.OrientedBoundingBox`` rather than something of its own.

They are registered beside ``robot.*`` and *before* the privileged branch, for
the same reason the robot tools are: a graph rewritten to fit its own box still
runs under ``--no-privileged`` and could still cross the bridge. Registering
them inside the branch would make every perception graph privileged-only, which
is the opposite of the point -- the shipped grocery ``gap`` graph fits its box
from SAM3 masks and then calls straight into here.

**Why this module exists at all.** The reference stack supplies ``geometry.*``
from its own tool bundle (``open-robot-skills``, imported by gap as
``gap_skills.tools.*``), which is a separate distribution this package does not
depend on and cannot resolve. Three shipped graphs called into it, so they
failed at the first grasp node with "Tool 'geometry.top_down_grasp_candidates'
not found" on a checkout that had installed everything this project declares.
What is reimplemented here is the contract those graphs were written against,
stated in ``tool_sort/graphs/gt/scripts/apply_grasp_yaw.py`` and honoured
exactly: the primary pose points straight down world -Z and **does not encode
the object's yaw**, because the fan's real consumer is a goalset planner
choosing among candidates, and a graph that takes ``poses.0`` alone is expected
to decide the yaw itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robosimstudio.core.math import quat_to_mat
from robosimstudio.solvers import top_down_grasp_quat

__all__ = ["GeometryTools", "reference_bundle_supplies_geometry", "register_geometry_tools"]

#: The tool frame pointing straight down: 180 degrees about x, which carries a
#: hand's own approach axis (local +z) onto world -z. xyzw, as everything below
#: newton speaks. This is a *convention*, not a measurement of any robot -- a
#: ``geometry.*`` tool has no embodiment to ask, which is exactly why the graph
#: treats this rotation as its fallback and prefers the robot-aware
#: ``sim.get_object_obb`` ``grasp_quat`` when it has one.
DOWN_XYZW = (1.0, 0.0, 0.0, 0.0)

#: How many yaws the fan carries. The reference emits 24, and the number is not
#: arbitrary: a parallel jaw is symmetric under a 180-degree flip, so the fan
#: spans ``[0, pi)`` rather than a full turn, and 24 of them land 7.5 degrees
#: apart.
DEFAULT_CANDIDATES = 24


def reference_bundle_supplies_geometry(registries: Sequence[str | Path] | None = None) -> bool:
    """Whether a resolvable skill registry ships the real ``geometry`` bundle.

    Args:
        registries: The registry roots the run executes with, when the caller
            has them (:func:`~robosimstudio_policy_agent.skills.pinned_registries`);
            ``None`` resolves gap's own set. Handing over the run's list is
            what keeps this decision and ``gap.execute``'s registry the same.

    The stand-in below exists because that bundle usually cannot be resolved
    (see the module docstring). When it *can* -- a checkout on
    ``$GAP_SKILLS_PATH``, in ``[tool.gap].registries``, or discovered beside the
    gap one -- registering the stand-in is worse than useless: the names are the
    same, ``register_callable`` refuses a duplicate, and the bundle loses the
    race because this package registers first. The whole graph then dies at load
    with "Tool name collision", which is how the perception graphs failed on a
    checkout that had done nothing wrong but install the reference stack.

    So the reference wins where it exists. It is the thing this module is a copy
    of, it carries the rest of ``geometry.*`` besides the one name reimplemented
    here, and a copy that shadows its original is a bug waiting for the two to
    drift.
    """
    return _registry_ships("geometry", registries)


def _registry_ships(bundle: str, registries: Sequence[str | Path] | None) -> bool:
    """Whether one of *registries* (or gap's own set) holds ``scripts/<bundle>``."""
    if registries is not None:
        return any((Path(root) / "tools" / bundle).is_dir() for root in registries)
    try:
        from gap.skills.registries import resolve_registries
    except ImportError:
        return False
    try:
        resolved = resolve_registries()
    except Exception:  # a misconfigured registry is not our error to raise
        return False
    return any((spec.path / "tools" / bundle).is_dir() for spec in resolved)


def register_geometry_tools(registry, *, registries: Sequence[str | Path] | None = None) -> None:
    """Register the unprivileged geometry tools on a GaP tool registry.

    A no-op where the reference bundle is resolvable -- see
    :func:`reference_bundle_supplies_geometry`.

    Args:
        registry: A ``gap_core.tools.ToolRegistry``.
        registries: The run's registry roots, or ``None`` to resolve gap's.
    """
    if reference_bundle_supplies_geometry(registries):
        return
    tools = GeometryTools()
    for name, fn, summary in tools.entries():
        registry.register_callable(name, fn, summary=summary, tags=("geometry",))


class GeometryTools:
    """The geometry implementations. Stateless, and deliberately so."""

    def entries(self) -> list[tuple[str, Any, str]]:
        return [
            (
                "geometry.top_down_grasp_candidates",
                self.top_down_grasp_candidates,
                "Top-down grasp poses over an oriented bounding box, as a yaw fan.",
            ),
        ]

    # ------------------------------------------------------------------ tools
    def top_down_grasp_candidates(
        self,
        obb: dict[str, Any],
        z_offset: float = 0.0,
        count: int = DEFAULT_CANDIDATES,
    ) -> dict[str, Any]:
        """A fan of straight-down grasp poses over one box.

        The gripper always points straight down along world -Z, and **yaw is not
        derived from the OBB**: candidate ``k`` is the box's top face yawed by
        ``k * pi / count`` about world z, so ``poses[0]`` is the unyawed pose and
        the rest are the alternatives a goalset planner would choose among. A
        caller that takes ``poses[0]`` alone is choosing to decide yaw itself.

        The z is a **fingertip** height, measured off the top face rather than
        off the centre, because that is the surface a descent has to clear: for a
        box lying at an angle the top is the centre plus the box's own extent
        projected onto world z, which is not any single half-extent. ``z_offset``
        is then how far *into* the object the fingertips are asked to go, so the
        graphs pass a negative number.

        Nothing here knows what the object is resting on. A caller that can find
        out -- ``sim.get_object_obb`` reports ``bottom_z`` -- is expected to clamp
        this z against it, which is what keeps a 13.6 mm syringe from being
        grasped 30 mm below its own top face and through the bench.

        Args:
            obb: An ``OrientedBoundingBox``: ``center``, ``extent`` and
                ``orientation``. ``extent`` is a **half**-extent, gap's
                convention and this package's. ``orientation`` is wxyz.
            z_offset: Added to the top face's world z [m]. Negative descends
                into the object.
            count: How many yaws to emit over ``[0, pi)``.

        Returns:
            ``{"candidates": {"poses": [...], ...}}`` -- ``poses`` a list of
            ``Se3Pose``, longest-lived field of the three and the one the graphs
            subscript.
        """
        centre, half, rot = _read_obb(obb)
        # The support of a rotated box along world z: sum of |R[2, i]| * half[i].
        # Reading half[2] instead is right only for an axis-aligned box, and
        # wrong by up to the box's diagonal for anything that has been knocked
        # over -- which is the normal state of a tool in a cluttered bin.
        top_z = float(centre[2] + float(np.abs(rot[2, :]) @ half))
        z = top_z + float(z_offset)

        n = max(1, int(count))
        poses = []
        for k in range(n):
            # top_down_grasp_quat wraps into [-90, 90) on the way, which is the
            # same symmetry the fan is built on and keeps the last wrist joint
            # off its limit; composing here rather than writing the product out
            # keeps one yaw convention in the package.
            q = top_down_grasp_quat(DOWN_XYZW, np.pi * k / n)
            poses.append(
                {
                    "position": {"x": float(centre[0]), "y": float(centre[1]), "z": z},
                    "rotation": {"w": float(q[3]), "x": float(q[0]), "y": float(q[1]), "z": float(q[2])},
                }
            )

        return {
            "candidates": {
                "poses": poses,
                "count": n,
                # What the z was measured from, so a caller clamping against a
                # support surface can say by how much it moved.
                "top_z": top_z,
                "approach": {"x": 0.0, "y": 0.0, "z": -1.0},
            }
        }


# --------------------------------------------------------------------- helpers
def _read_obb(obb: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, half-extent and rotation matrix from an ``OrientedBoundingBox``.

    Accepts the mapping form gap's types serialise to and the plain sequence a
    hand-written caller reaches for, because a tool that raises ``TypeError`` on
    ``[0, 0, 1]`` teaches nothing about which of the two it wanted.
    """
    if not isinstance(obb, dict):
        raise ValueError(f"obb must be an OrientedBoundingBox mapping, got {type(obb).__name__}")
    try:
        centre = _vec(obb["center"])
        half = _vec(obb["extent"])
    except KeyError as exc:
        raise ValueError(f"obb is missing {exc.args[0]!r}; expected center, extent and orientation") from exc
    # "rotation" is accepted beside "orientation" because the two spellings are
    # both in circulation in graph scripts, and a silently identity-rotated box
    # grasps a knocked-over tool at the wrong height rather than failing.
    quat = obb.get("orientation", obb.get("rotation"))
    return centre, half, _rot(quat)


def _vec(v: Any) -> np.ndarray:
    """A ``Vec3`` as xyz, from the mapping form or a sequence."""
    if isinstance(v, dict):
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])], dtype=np.float64)
    if isinstance(v, Sequence) and len(v) >= 3:
        return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)
    raise ValueError(f"expected a Vec3 mapping or a 3-sequence, got {v!r}")


def _rot(q: Any) -> np.ndarray:
    """Rotation matrix of a wxyz ``Quaternion``; identity when unstated."""
    if q is None:
        return np.eye(3)
    if isinstance(q, dict):
        q_xyzw = [float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])]
    elif isinstance(q, Sequence) and len(q) >= 4:
        # wxyz, as gap states quaternions everywhere.
        q_xyzw = [float(q[1]), float(q[2]), float(q[3]), float(q[0])]
    else:
        raise ValueError(f"expected a wxyz Quaternion mapping or 4-sequence, got {q!r}")
    return quat_to_mat(q_xyzw)
