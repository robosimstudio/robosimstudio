# SPDX-License-Identifier: Apache-2.0
"""Boundary-time IK for one world of a fleet: the graph runtime's backend, fused across the wave.

A wave runs one graph per world on its own thread, and every boundary solve --
a pose, a straight line, a joint-space ease -- reaches the fleet's thread at
the barrier as one of a *group*: at the barrier every world that is going to
ask has asked, so the group is complete, which is the thing a solver that
fuses them needs and cannot get any other way. This module is the fused half:
the backend each world holds (:class:`IKBackendFleet`, an
:class:`~robosimstudio_policy_agent.ik.IKBackend` bound to one world), the rendezvous
adapter the barrier calls (:func:`solve_group`), and the three fusers that
answer a group with one batched call to the solver
(:meth:`~robosimstudio.solvers.ik.solver.IKSolver.solve` over ``W`` rows).

The arithmetic is the single-world backend's exactly -- the same solver, the
same gates, the same refusal texts -- and what the fusion changes is the
number of calls: a wave's forty-waypoint lines are forty batched solves rather
than four hundred and eighty, and a wave's pose solves are one. Both the solver
and its fallback (:data:`~robosimstudio_policy_agent.ik.SOLVER`,
:data:`~robosimstudio_policy_agent.ik.FALLBACK`) batch.

Measured before the fusers, on twelve worlds: the boundary solve was 254.8 ms
of a 255.7 ms wave step, and the graph threads spent that time parked. See
:mod:`robosimstudio.core.streams` for why the group takes the stream guard once.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.arms import arm_name
from robosimstudio.core.streams import reading
from robosimstudio.solvers.ik.fleet_fk import FleetArmFK
from robosimstudio.solvers.ik.solver import plan_line
from robosimstudio_policy_agent.ik import IKBackend, build_solver
from robosimstudio_policy_agent.types import se3, se3_arrays

__all__ = ["IKBackendFleet", "solve_group"]


def solve_group(asks: list[tuple[int, tuple]]) -> list[Any]:
    """Answer a whole wave's boundary solves. Fleet thread, at the barrier.

    Grouped by kind and arm, each group handed to its fuser; a group the fuser
    declines -- one member, an arm the fleet cannot batch -- is answered one at
    a time with the single-world body, which is the same arithmetic. The stream
    guard is taken once for the group: it is the same lock either way and this
    is the thread that actually moves the stream.

    A solve that finds nothing returns ``None``, which is an answer. Only a
    raised exception becomes one world's node error, and it is that world's
    alone -- an unreachable target is not the group's problem.
    """
    answers: list[Any | None] = [None] * len(asks)
    with reading():
        for (kind, arm_id), members in _by_arm(asks).items():
            fuse = _FUSERS.get(kind)
            fused = fuse(asks, members, arm_id) if fuse is not None else None
            for slot in members:
                if fused is not None and slot in fused:
                    answers[slot] = fused[slot]
                    continue
                _world, payload = asks[slot]
                _backend, _kind, fn, args = payload
                try:
                    answers[slot] = fn(*args)
                except BaseException as exc:
                    answers[slot] = exc
    return answers


def _by_arm(asks: list[tuple[int, tuple]]) -> dict[tuple[str, int], list[int]]:
    """Ask positions grouped by what they are and which arm they are about.

    One arm at a time because that is what a solver is: two arms of one robot
    are disjoint chains with their own coordinates and limits. Keyed on the
    kind as well, because a pose solve is one search and a straight line is a
    chain of them seeded end to end.
    """
    groups: dict[tuple[str, int], list[int]] = {}
    for slot, (_world, payload) in enumerate(asks):
        key = ("", -1)
        try:
            _backend, kind, _fn, args = payload
            where = _ARM_ARG.get(kind)
            if where is not None:
                key = (str(kind), int(args[where]))
        except (TypeError, ValueError, IndexError):
            key = ("", -1)
        groups.setdefault(key, []).append(slot)
    return groups


def _fleet_joint_q(fleet: Any) -> np.ndarray:
    """The wave's joint coordinates on the host, once for the group."""
    host = getattr(fleet, "host_joint_q", None)
    if callable(host):
        return host()
    return fleet.pipeline.state_0.joint_q.numpy()


def _arm_fk(fleet: Any, arm_id: int, arm: Any) -> FleetArmFK:
    """This arm's batched forward kinematics over the replicated model, built once per fleet."""
    cache = getattr(fleet, "_boundary_arm_fk", None)
    if cache is None:
        cache = {}
        fleet._boundary_arm_fk = cache
    got = cache.get(arm_id)
    if got is None:
        got = cache[arm_id] = FleetArmFK(
            fleet.model,
            fleet.worlds,
            arm_coords=arm.arm_coords,
            ee_body=arm.ee_body,
            ee_offset=arm.ee_offset,
            ee_offset_rot=arm.ee_offset_rot,
        )
    return got


def _group_solver(first: IKBackendFleet, arm_id: int):
    """The wave's solver for one arm: the family's, over the fleet's batched kinematics, built once per fleet.

    One per (arm, family) rather than per world, because a fused call is one
    problem over every world's row -- and because the PyRoKi family pays a JIT
    per batch shape, which a per-world solver would pay ``W`` times over.

    Built by the single-world backend's rule (:func:`~robosimstudio_policy_agent.ik.build_solver`):
    a family that cannot serve the arm falls back to the damped-least-squares
    one, once per fleet and said once, and the backend asking takes on the
    contract of the family in hand. It did not, once: this built the family
    asked for and nothing else, so on a robot that ships no URDF every
    boundary solve that reached the wave's own solver -- ``motion.plan_joint``,
    the one verb the planner does not answer -- raised ``ValueError`` where the
    serial connector had quietly solved with newton. Recorded on the fleet
    beside the solver, because the fusers ask for the solver by the family
    *requested* and every member must adopt the family *served*.
    """
    fleet = first.fleet
    cache = getattr(fleet, "_boundary_solvers", None)
    if cache is None:
        cache = {}
        fleet._boundary_solvers = cache
    families = getattr(fleet, "_boundary_families", None)
    if families is None:
        families = {}
        fleet._boundary_families = families
    key = (int(arm_id), first.family)
    if key not in cache:
        name = arm_name(fleet, arm_id)
        arm = first._arm(arm_id)
        fk = _arm_fk(fleet, arm_id, arm)
        fk.seed_from(_fleet_joint_q(fleet))
        cache[key], families[key] = build_solver(fleet, name, first.family, fallback=first.fallback, fk=fk.poses)
    first._adopt(families[key])
    return cache[key]


def _members_agree(asks: list[tuple[int, tuple]], members: list[int]) -> IKBackendFleet | None:
    """The group's first backend, when every member shares its family and roll contract; else ``None``."""
    first = asks[members[0]][1][0]
    if first is None:
        return None
    for slot in members:
        backend = asks[slot][1][0]
        if backend is None or backend.family != first.family or bool(backend.honours_roll) != bool(first.honours_roll):
            return None
    return first


def _agreed(asks: list[tuple[int, tuple]], members: list[int], arm_id: int) -> IKBackendFleet | None:
    """:func:`_members_agree`, once every member holds the arm's solver.

    Resolved first because the wave's solver is where a family that cannot
    serve the arm falls back, and a backend takes on the fallback's roll
    contract as it resolves -- so the agreement check has to see the contract
    the members will solve under, not the one they were configured with. A
    member whose solver cannot be built declines the group; the loop answers
    it one at a time and the raise is that world's alone.
    """
    for slot in members:
        backend = asks[slot][1][0]
        if backend is None or getattr(getattr(backend, "fleet", None), "worlds", None) is None:
            return None
        try:
            backend.solver_for(arm_id)
        except (ValueError, ImportError):
            return None
    return _members_agree(asks, members)


def _solve_arm_together(asks: list[tuple[int, tuple]], members: list[int], arm_id: int) -> dict[int, Any] | None:
    """One arm's pose solves as one batched solve, or ``None`` to solve them one by one.

    Declining is not failing: a group of one has nothing to fuse, and a group
    that cannot be set up is a group the loop answers.
    """
    if arm_id < 0 or len(members) < 2:
        return None
    first = _agreed(asks, members, arm_id)
    if first is None:
        return None
    try:
        solver = _group_solver(first, arm_id)
        fk = _arm_fk(first.fleet, arm_id, first._arm(arm_id))
        fk.seed_from(_fleet_joint_q(first.fleet))
        count = first.fleet.worlds.num_worlds
        seeds = fk.live_arm_coords()
        goals = np.zeros((count, 3))
        quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (count, 1))
        active = np.zeros(count, dtype=bool)
        for slot in members:
            world = int(asks[slot][0])
            backend, _kind, _fn, args = asks[slot][1]
            pose, _arm_id, seed_joints = args
            seeds[world] = backend._seed(arm_id, seed_joints)[0][0]
            goals[world], quats[world] = se3_arrays(pose)
            active[world] = True
        roll = first._contract()
        solved, err = solver.solve(seeds, goals, quats, roll=roll, active=active, iterations=first.iterations)
    except BaseException:
        return None
    out: dict[int, Any] = {}
    for slot in members:
        world = int(asks[slot][0])
        backend = asks[slot][1][0]
        distance, rotation = float(err[world, 0]), float(err[world, 1])
        backend.last_solve_error = (distance, rotation if roll == "held" else None)
        if distance > backend.tolerance or (roll == "held" and rotation > backend.rot_tolerance):
            out[slot] = None
        else:
            out[slot] = [float(v) for v in solved[world]]
    return out


def _plan_arm_together(asks: list[tuple[int, tuple]], members: list[int], arm_id: int) -> dict[int, Any] | None:
    """One arm's straight lines, walked waypoint by waypoint for the whole group.

    A leg is a chain seeded end to end, so what is shared across worlds is the
    waypoint *index*: at each index every world solves its own point from its
    own previous solution, together. Worlds may differ in their endpoints,
    their waypoint counts, their tolerances and their failures; a world whose
    waypoint is out of reach leaves the group there with its refusal recorded.
    """
    if arm_id < 0 or len(members) < 2:
        return None
    first = _agreed(asks, members, arm_id)
    if first is None:
        return None
    legs = []
    for slot in members:
        backend, _kind, _fn, args = asks[slot][1]
        if len(args) < 7 or bool(args[5]) != bool(asks[members[0]][1][3][5]):
            return None  # a held-rotation leg and a free one are different contracts
        legs.append((slot, backend, args))
    try:
        solver = _group_solver(first, arm_id)
        fk = _arm_fk(first.fleet, arm_id, first._arm(arm_id))
        fk.seed_from(_fleet_joint_q(first.fleet))
        count = first.fleet.worlds.num_worlds
        seeds = fk.live_arm_coords()
        starts, ends = np.zeros((count, 3)), np.zeros((count, 3))
        quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (count, 1))
        steps = np.ones(count, dtype=np.int64)
        tolerances = np.full(count, np.inf)
        active = np.zeros(count, dtype=bool)
        for slot, backend, args in legs:
            world = int(asks[slot][0])
            start_pose, end_pose, _a, seed_joints, num_waypoints, _hold_rotation, tolerance = args
            seeds[world] = backend._seed(arm_id, seed_joints)[0][0]
            starts[world], _ = se3_arrays(start_pose)
            ends[world], quats[world] = se3_arrays(end_pose)
            steps[world] = max(1, int(num_waypoints) or backend.waypoints)
            tolerances[world] = float(backend.tolerance if tolerance is None else tolerance)
            active[world] = True
            backend.last_refusal = ""
            backend.last_position_error = None
        roll = first._contract(bool(legs[0][2][5]))
        path, refusals = plan_line(
            solver,
            seeds,
            starts,
            ends,
            quats,
            roll=roll,
            waypoints=steps,
            tolerance=tolerances,
            rot_tolerance=first.rot_tolerance,
            iterations=first.iterations,
            active=active,
        )
    except BaseException:
        return None
    out: dict[int, Any] = {}
    for slot, backend, _args in legs:
        world = int(asks[slot][0])
        refusal = refusals[world]
        if refusal is not None:
            backend.last_position_error = float(refusal.position_error)
            backend.last_refusal = (
                "orientation='lock' holds the end pose's whole rotation along the leg; "
                + refusal.text(tolerance=float(tolerances[world]), rot_tolerance=first.rot_tolerance)
                if roll == "held"
                else refusal.text(tolerance=float(tolerances[world]), rot_tolerance=first.rot_tolerance)
            )
            out[slot] = None
        else:
            rows = path[world, : int(steps[world]), :]
            out[slot] = {"waypoints": [{"positions": [float(v) for v in row]} for row in rows]}
    return out


def _plan_joint_arm_together(asks: list[tuple[int, tuple]], members: list[int], arm_id: int) -> dict[int, Any] | None:
    """One arm's joint-space eases, solved for the whole group at once. Never refuses."""
    if arm_id < 0 or len(members) < 2:
        return None
    first = _agreed(asks, members, arm_id)
    if first is None:
        return None
    try:
        solver = _group_solver(first, arm_id)
        fk = _arm_fk(first.fleet, arm_id, first._arm(arm_id))
        fk.seed_from(_fleet_joint_q(first.fleet))
        count = first.fleet.worlds.num_worlds
        seeds = fk.live_arm_coords()
        goals = np.zeros((count, 3))
        quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (count, 1))
        active = np.zeros(count, dtype=bool)
        holds = []
        for slot in members:
            world = int(asks[slot][0])
            backend, _kind, _fn, args = asks[slot][1]
            pose, _arm_id, seed_joints, _num_waypoints, _hold_rotation = args
            seeds[world] = backend._seed(arm_id, seed_joints)[0][0]
            goals[world], quats[world] = se3_arrays(pose)
            active[world] = True
            holds.append(bool(args[4]))
        if len(set(holds)) != 1:
            return None
        roll = first._contract(holds[0])
        solved, err = solver.solve(seeds, goals, quats, roll=roll, active=active, iterations=first.iterations)
    except BaseException:
        return None
    out: dict[int, Any] = {}
    for slot in members:
        world = int(asks[slot][0])
        backend, _kind, _fn, args = asks[slot][1]
        count_steps = max(1, int(args[3]) or backend.waypoints)
        start, end = seeds[world], solved[world]
        waypoints = []
        for i in range(1, count_steps + 1):
            t = i / count_steps
            s = t * t * (3.0 - 2.0 * t)
            waypoints.append({"positions": [float(v) for v in (start + s * (end - start))]})
        out[slot] = {
            "waypoints": waypoints,
            "position_error_m": float(err[world, 0]),
            "rotation_error_rad": float(err[world, 1]) if roll == "held" else 0.0,
            "joint_config": [float(v) for v in end],
        }
    return out


class _WorldRow:
    """A wave's solver bound to one world: that world's row expanded to the wave, the rest held where they are.

    What a single world's solve or chained line calls when the solver it was
    handed is the group's -- built over every world's kinematics, so it wants
    ``W`` rows even when one world asks. The other rows are the wave's live
    coordinates with their ``active`` flag down, which the solver leaves
    untouched, and only this world's row comes back.
    """

    def __init__(self, solver, fleet, fk, world: int) -> None:
        self.solver = solver
        self.fleet = fleet
        self.fk = fk
        self.world = int(world)
        self.kinematics = solver.kinematics

    def solve(self, seed, position, quat_xyzw, *, roll="free", active=None, iterations=0):
        count = self.fleet.worlds.num_worlds
        self.fk.seed_from(_fleet_joint_q(self.fleet))
        seeds = self.fk.live_arm_coords()
        seeds[self.world] = np.asarray(seed, dtype=np.float64).reshape(-1)[: seeds.shape[1]]
        goals = np.tile(np.asarray(position, dtype=np.float64).reshape(1, 3), (count, 1))
        quats = np.tile(np.asarray(quat_xyzw, dtype=np.float64).reshape(1, 4), (count, 1))
        asking = np.zeros(count, dtype=bool)
        asking[self.world] = active is None or bool(np.asarray(active).reshape(-1)[0])
        with reading():
            solved, err = self.solver.solve(seeds, goals, quats, roll=roll, active=asking, iterations=iterations)
        return solved[self.world : self.world + 1], err[self.world : self.world + 1]


class IKBackendFleet(IKBackend):
    """The boundary-time backend, bound to one world of a fleet.

    Args:
        fleet: The batched environment.
        world: Which world this backend answers for.
        bus: That world's bus, which marshals a solve onto the fleet's thread
            -- or joins a rendezvous group when the bus can batch the kind.
        roll: The roll contract.
        tolerance, waypoints, iterations: The single-world defaults.
    """

    SOLVE = "ik.solve"
    """The group a single-pose solve joins."""
    PLAN_LINEAR = "ik.plan_linear"
    """The group a straight-line plan joins: forty chained solves, a different shape from a pose."""
    PLAN_JOINT = "ik.plan_joint"
    """The group a joint-space ease joins: one solve, asked for at a different moment by a different verb."""

    def __init__(
        self,
        fleet: Any,
        world: int,
        bus: Any,
        *,
        roll: str = "auto",
        tolerance: float = 0.03,
        waypoints: int = 40,
        iterations: int = 80,
    ) -> None:
        self.fleet = fleet
        self.world = int(world)
        super().__init__(fleet, bus, roll=roll, tolerance=tolerance, waypoints=waypoints, iterations=iterations)

    # ------------------------------------------------------------- the world's
    def _arm(self, arm_id: int):
        """This arm's solver on the fleet's **template** model, whose indices are the embodiment's."""
        embodiment = self.fleet.scene.embodiment
        iks = embodiment.iks if embodiment is not None else {}
        if not iks:
            raise RuntimeError(f"task {self.fleet.task.full_name!r} has no arms")
        return iks[arm_name(self.fleet, arm_id)]

    def _build_solver(self, name: str):
        """The wave's solver for this arm (shared across worlds), or the single-world one when there is no wave."""
        arm_id = next(i for i in range(len(self.fleet.handles.arms)) if arm_name(self.fleet, i) == name)
        if getattr(self.fleet, "worlds", None) is not None:
            return _group_solver(self, arm_id)
        return super()._build_solver(name)

    def _seed(self, arm_id: int, seed_joints: Any | None) -> tuple[np.ndarray, np.ndarray]:
        """This world's arm coordinates, from the wave's host copy, with the caller's seed written over them."""
        ik = self._arm(arm_id)
        joint_q = _fleet_joint_q(self.fleet)
        view = self.fleet.worlds.coord_view(joint_q, self.world)
        template = np.asarray(view, dtype=np.float64).copy()
        row = template[list(ik.arm_coords)].copy()
        if seed_joints is not None:
            values = np.asarray(seed_joints, dtype=np.float64).reshape(-1)
            row[: len(values)] = values[: len(row)]
        return row.reshape(1, -1), template

    def _bound(self, solver, template):
        """The wave's shared solver as this world sees it: its row in, its row out, the other worlds masked off."""
        if getattr(self.fleet, "worlds", None) is None or self.fleet.worlds.num_worlds == 1:
            return super()._bound(solver, template)
        arm_id = self._arm_id_of(solver)
        return _WorldRow(solver, self.fleet, _arm_fk(self.fleet, arm_id, self._arm(arm_id)), self.world)

    # The solve bodies, under the stream guard. Newton's Levenberg-Marquardt
    # solver moves the device's process-global current stream for the duration
    # and a wave has one script thread per world asking; the guard is here, on
    # the fleet's side of the marshalling, rather than on the worker's -- with
    # it on the worker's side both worlds of gear_mesh/gearmesh-easy-pair-01 parked at
    # ``gearboard.reachable`` and the fleet ran at 44 % CPU stepping nobody.
    def _solve(self, *args, **kwargs):
        with reading():
            return super()._solve(*args, **kwargs)

    def _plan_line(self, *args, **kwargs):
        with reading():
            return super()._plan_line(*args, **kwargs)

    def _plan_joint(self, *args, **kwargs):
        with reading():
            return super()._plan_joint(*args, **kwargs)

    def _arm_id_of(self, solver) -> int:
        cache = getattr(self.fleet, "_boundary_solvers", {})
        for (arm_id, _family), held in cache.items():
            if held is solver:
                return int(arm_id)
        return 0

    def current_pose(self, arm_id: int = 0) -> dict:
        """Where **this world's** tool centre is now, off the published snapshot: no device read."""
        arm = self._arm(arm_id)
        # The world's snapshot is a per-world VIEW, indexed like the template
        # model -- as every other reader of ``snapshot.body(world)`` indexes it.
        # A fleet row here read past the view's end on every world but 0
        # (IndexError: index 24 ... size 19, wrenches_level1 world 1).
        body_q = self.bus.snapshot.body_q
        xform = np.asarray(body_q[int(arm.ee_body)], dtype=np.float64)
        from robosimstudio.core.math import quat_multiply, transform_point

        return se3(transform_point(xform, arm.ee_offset), quat_multiply(xform[3:7], arm.ee_offset_rot))

    # ------------------------------------------------------------- rendezvous
    def solve_ik(
        self, target_world_pose: dict, *, arm_id: int = 0, seed_joints: Any | None = None, tcp_offset: Any | None = None
    ):
        del tcp_offset
        return self._ask(self.SOLVE, self._solve, (target_world_pose, arm_id, seed_joints))

    def _own_plan_linear(
        self,
        start_world_pose: dict,
        end_world_pose: dict,
        *,
        arm_id: int = 0,
        tcp_offset: Any | None = None,
        seed_joints: Any | None = None,
        num_waypoints: int = 0,
        hold_rotation: bool = False,
        tolerance: float | None = None,
    ):
        del tcp_offset
        return self._ask(
            self.PLAN_LINEAR,
            self._plan_line,
            (start_world_pose, end_world_pose, arm_id, seed_joints, num_waypoints, hold_rotation, tolerance),
        )

    def plan_joint(
        self,
        target_world_pose: dict,
        *,
        arm_id: int = 0,
        tcp_offset: Any | None = None,
        seed_joints: Any | None = None,
        num_waypoints: int = 0,
        hold_rotation: bool = False,
    ) -> dict:
        del tcp_offset
        return self._ask(
            self.PLAN_JOINT, self._plan_joint, (target_world_pose, arm_id, seed_joints, num_waypoints, hold_rotation)
        )

    def _ask(self, kind: str, fn: Any, args: tuple) -> Any:
        """Join this kind's group, or marshal the old way when nothing serves it."""
        batchable = getattr(self.bus, "batchable", None)
        if callable(batchable) and batchable(kind):
            return self.bus.batched(kind, (self, kind, fn, args))
        return self.bus.call(fn, *args)


_ARM_ARG = {IKBackendFleet.SOLVE: 1, IKBackendFleet.PLAN_LINEAR: 2, IKBackendFleet.PLAN_JOINT: 1}
"""Where each kind's ask carries its ``arm_id``."""

_FUSERS = {
    IKBackendFleet.SOLVE: _solve_arm_together,
    IKBackendFleet.PLAN_LINEAR: _plan_arm_together,
    IKBackendFleet.PLAN_JOINT: _plan_joint_arm_together,
}
