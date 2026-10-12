# SPDX-License-Identifier: Apache-2.0
"""What an inverse-kinematics solver is here: numpy in, numpy out, a batch of worlds at once.

One protocol for the three solver families this stack keeps -- the batched
PyRoKi solve, the damped-least-squares reference, and Newton's on-device
Levenberg-Marquardt -- so the graph-policy runtime's IK and the fleet that
fuses a wave's asks both drive
any of them through one door. A solver answers a **batch**: ``W`` seeds, ``W``
targets, ``W`` solutions with their errors, and ``W = 1`` is a batch of one.
Nothing here knows an ``Se3Pose`` dict, an ``arm_id`` or a bus.

The **roll** is the one contract every solver honours: ``"free"`` aligns the
tool's +z with the target's and leaves the wrist roll to the solver, which is
what every shipped graph on every embodiment but the UR5e was authored and
measured against; ``"held"`` solves the whole orientation. Full-pose authority
is opt-in per call, never a side effect of the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

from robosimstudio.core.math import quat_to_mat

__all__ = [
    "POS_TOLERANCE",
    "ROT_TOLERANCE_DEG",
    "IKSolver",
    "LineRefusal",
    "Roll",
    "approach_axes",
    "plan_line",
    "pose_errors",
]

Roll = Literal["free", "held"]

POS_TOLERANCE = 0.004
"""Servo arrival ball [m]: 4 mm, the distance an OSC setpoint is called reached at."""

ROT_TOLERANCE_DEG = 3.0
"""Servo arrival cone [deg], about the approach axis or the whole rotation as the roll contract says."""


class IKSolver(Protocol):
    """A batched inverse-kinematics solver over one arm.

    Attributes:
        kinematics: The arm it solves for -- coordinates, limits, tool frame and
            forward kinematics (:class:`~robosimstudio.solvers.ik.arm.ArmKinematics`).
        capabilities: What this family can do: ``"batched"`` (a wave in one
            call), ``"captured"`` (runs inside a CUDA-graph-captured control
            step), ``"needs_urdf"`` (drives only an arm that ships a URDF).
    """

    kinematics: Any
    capabilities: frozenset[str]

    def solve(
        self,
        seed: np.ndarray,
        position: np.ndarray,
        quat_xyzw: np.ndarray,
        *,
        roll: Roll = "free",
        active: np.ndarray | None = None,
        iterations: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve ``W`` targets from ``W`` seeds.

        Args:
            seed: Each world's arm coordinates to start from ``[W, n]``.
            position: Each world's tool-centre target ``[W, 3]`` [m].
            quat_xyzw: Each world's target orientation ``[W, 4]``; under
                ``roll="free"`` only its +z is asked for.
            roll: The roll contract.
            active: Which worlds are asking ``[W]``; an inactive world comes
                back exactly as it went in.
            iterations: A budget, in the solver's own units; ``0`` is its default.

        Returns:
            ``(q [W, n], err [W, 2])``: the solutions and, per world, the
            position error [m] and the rotation error [rad] -- the whole
            rotation under ``"held"``, the angle between approach axes under
            ``"free"`` -- measured by forward kinematics, never assumed.
        """
        ...


@dataclass(frozen=True)
class LineRefusal:
    """Why a straight line could not be solved, for the caller to report.

    Attributes:
        world: Which world's line.
        waypoint: The first waypoint that missed, 1-based, of ``waypoints``.
        waypoints: How many the line was cut into.
        position_error: That waypoint's position miss [m].
        rotation_error: Its rotation miss [rad].
        binding: ``"the position"``, ``"the held rotation"`` or ``"both position and the held rotation"``.
    """

    world: int
    waypoint: int
    waypoints: int
    position_error: float
    rotation_error: float
    binding: str

    def text(self, *, tolerance: float, rot_tolerance: float) -> str:
        return (
            f"at waypoint {self.waypoint} of {self.waypoints} the solver was "
            f"{self.position_error * 1000:.0f} mm and {np.degrees(self.rotation_error):.0f} deg off "
            f"(tolerances {tolerance * 1000:.0f} mm and {np.degrees(rot_tolerance):.0f} deg), "
            f"so {self.binding} is what could not be met"
        )


def approach_axes(quat_xyzw: np.ndarray) -> np.ndarray:
    """Each orientation's +z in the world ``[W, 3]``: the whole of what a roll-free solve is asked for."""
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1, 4)
    return np.stack([quat_to_mat(row)[:, 2] for row in q])


def pose_errors(
    position: np.ndarray, quat_xyzw: np.ndarray, target_position: np.ndarray, target_quat: np.ndarray, *, roll: Roll
) -> np.ndarray:
    """``[W, 2]`` of position error [m] and rotation error [rad] under the roll contract.

    The rotation is the whole rotation under ``"held"`` and the angle between
    approach axes under ``"free"``, which is the error a roll-free solver was
    asked to close and the only one it is fair to hold it to.
    """
    pos = np.asarray(position, dtype=np.float64).reshape(-1, 3)
    want = np.asarray(target_position, dtype=np.float64).reshape(-1, 3)
    distance = np.linalg.norm(pos - want, axis=1)
    got = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1, 4)
    goal = np.asarray(target_quat, dtype=np.float64).reshape(-1, 4)
    if roll == "held":
        dots = np.abs(np.einsum("wi,wi->w", got, goal)) / np.maximum(
            np.linalg.norm(got, axis=1) * np.linalg.norm(goal, axis=1), 1.0e-12
        )
        angle = 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))
    else:
        cos = np.einsum("wi,wi->w", approach_axes(got), approach_axes(goal))
        angle = np.arccos(np.clip(cos, -1.0, 1.0))
    return np.stack([distance, angle], axis=1)


def plan_line(
    solver: IKSolver,
    seed: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    quat_xyzw: np.ndarray,
    *,
    roll: Roll = "free",
    waypoints: int | np.ndarray = 40,
    tolerance: float | np.ndarray = 0.03,
    rot_tolerance: float = np.radians(5.0),
    iterations: int = 0,
    active: np.ndarray | None = None,
) -> tuple[np.ndarray, list[LineRefusal | None]]:
    """A straight line in Cartesian space, solved at each waypoint, for ``W`` worlds at once.

    **Not collision aware.** It goes where it is told. The interpolation is in
    Cartesian space and each waypoint is seeded from the previous solution,
    which keeps a chain in one joint-space branch -- interpolating between
    branches is what swings an elbow through the table between two perfectly
    good endpoints. The orientation is the *end* pose's throughout, never
    slerped: a leg that turns the wrist while it travels turns a held object in
    the jaws, and every caller asks for a pose it wants to arrive in.

    Args:
        solver: Any :class:`IKSolver`.
        seed: ``[W, n]`` arm coordinates the first waypoint is seeded from.
        start, end: ``[W, 3]`` tool-centre positions [m].
        quat_xyzw: ``[W, 4]`` the orientation held along the leg.
        roll: The roll contract every waypoint is solved and gated under.
        waypoints: How many solves the line is cut into; per world when an array.
        tolerance: Position error above which a waypoint is out of reach [m]; per world when an array.
        rot_tolerance: Rotation error above which a waypoint is out of reach
            [rad]; consulted under ``"held"`` only.
        iterations: The solver's budget per waypoint.
        active: Which worlds are asking ``[W]``.

    Returns:
        ``(path [W, K, n], refusals)`` with ``K`` the longest leg: every world's
        configurations (a shorter leg's later rows repeat its last), and per
        world ``None`` or the :class:`LineRefusal` naming the first waypoint
        that missed and which constraint was binding. A refused world's later
        rows repeat its last solution; the refusal is the thing to read.
    """
    seed = np.asarray(seed, dtype=np.float64).reshape(-1, solver.kinematics.dof)
    n_worlds = seed.shape[0]
    first = np.asarray(start, dtype=np.float64).reshape(n_worlds, 3)
    last = np.asarray(end, dtype=np.float64).reshape(n_worlds, 3)
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(n_worlds, 4)
    # Per world, so a wave's legs may differ in length and in what they accept:
    # a short leg simply stops being active early.
    steps = np.maximum(1, np.broadcast_to(np.asarray(waypoints, dtype=np.int64), (n_worlds,)).copy())
    tolerances = np.broadcast_to(np.asarray(tolerance, dtype=np.float64), (n_worlds,)).copy()
    live = np.ones(n_worlds, dtype=bool) if active is None else np.asarray(active, dtype=bool).copy()
    longest = int(steps.max())
    path = np.repeat(seed[:, None, :], longest, axis=1)
    refusals: list[LineRefusal | None] = [None] * n_worlds
    current = seed
    for i in range(1, longest + 1):
        asking = live & (i <= steps)
        if not asking.any():
            break
        point = first + (last - first) * (i / steps).reshape(-1, 1)
        current, err = solver.solve(current, point, quat, roll=roll, active=asking, iterations=iterations)
        path[:, i - 1, :] = current
        for w in np.flatnonzero(asking):
            miss_p = err[w, 0] > tolerances[w]
            miss_r = roll == "held" and err[w, 1] > rot_tolerance
            if miss_p or miss_r:
                binding = (
                    "both position and the held rotation"
                    if miss_p and miss_r
                    else "the held rotation"
                    if miss_r
                    else "the position"
                )
                refusals[w] = LineRefusal(int(w), i, int(steps[w]), float(err[w, 0]), float(err[w, 1]), binding)
                live[w] = False
    return path, refusals
