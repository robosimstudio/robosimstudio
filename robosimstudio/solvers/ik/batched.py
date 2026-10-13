# SPDX-License-Identifier: Apache-2.0
"""One inverse-kinematics solve for every world at once.

:class:`~robosimstudio.solvers.ik.arm_ik.ArmIK` solves one arm on one world
with ``n_problems=1``. Newton's LM solver has always been batched; this is the
same solver told how many worlds there are, plus the two kernels that move
coordinates between the fleet's ``joint_q``/``joint_target_q`` layout and the
solver's ``[worlds, coords]`` one.

Three things make that more than a parameter change.

**The IK model is not the fleet model.** The solver evaluates forward kinematics
on a *single-world* model and applies it to ``n_problems`` independent joint
vectors -- so what it wants is one world's kinematics, not the replicated scene.
More usefully, it need not be the *whole* scene either: the LM path refuses
models with exotic joint types or more than 64 coordinates
(:meth:`ArmIK._lm_hostility`), and both limits come from things the arm does not
have. A cable, a D6-jointed connector and a bin of loose tools all push a scene
past them while the arm stays six or seven revolute joints. So this class takes
an IK model, a map from its coordinates to the fleet's, and solves in whichever
of the two is smaller.

**Arms are solved one after another, into one buffer.** Each arm gets its own
solver with its own ``joint_dof_mask``, exactly as the single-world path does, and
they run in sequence over a shared ``[worlds, coords]`` seed array. Two arms of one
robot are disjoint kinematic chains, so a masked solve leaves the other arm's
coordinates untouched and the order does not matter. One solver carrying both
arms' objectives would be a single larger least-squares problem -- coupling two
independent arms through one damping parameter, and needing the idle arm's
current pose supplied as an objective just to hold it still.

**A world that is not being driven must not be written.** Every world is in the
fleet's ``joint_target_q``, and a world whose script is thinking between motions
holds its last targets. That hold is free -- the array simply is not touched --
so the scatter is gated per world by an enable flag, and a disabled world's
targets come out bit-identical rather than approximately unchanged.

Seeding comes from the live ``joint_q`` every solve, for the reason the
single-world path documents: an arm that is blocked has physics and IK
disagreeing, and re-seeding from where the arm actually is keeps the two from
drifting apart. The whole thing captures into one CUDA graph, lazily, keyed on
the array pair -- the pattern and the fallback are
:meth:`ArmIK.solve_into`'s.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import newton
import newton.ik as ik
import numpy as np
import warp as wp

from robosimstudio.core.streams import capturing

log = logging.getLogger(__name__)

__all__ = ["BatchedArmIK", "RobotIKRig"]


@wp.kernel
def _gather_seeds(
    joint_q: wp.array[wp.float32],
    coord_map: wp.array[wp.int32],
    defaults: wp.array[wp.float32],
    coords_per_world: int,
    seeds: wp.array2d[wp.float32],
):
    """Fleet ``joint_q`` to solver seeds: one row per world.

    ``coord_map[i] < 0`` marks an IK-model coordinate the fleet model has no
    counterpart for. It takes the IK model's own rest value, which is what keeps
    a reduced IK model (one built without the scene's loose objects) seeded
    consistently rather than from zero.
    """
    world, i = wp.tid()
    source = coord_map[i]
    if source >= 0:
        seeds[world, i] = joint_q[world * coords_per_world + source]
    else:
        seeds[world, i] = defaults[i]


@wp.kernel
def _scatter_targets(
    solution: wp.array2d[wp.float32],
    ik_coords: wp.array[wp.int32],
    scene_coords: wp.array[wp.int32],
    enable: wp.array[wp.int32],
    coords_per_world: int,
    joint_target_q: wp.array[wp.float32],
):
    """One arm's solved coordinates into the fleet's targets, per enabled world."""
    world, k = wp.tid()
    if enable[world] == 0:
        return
    joint_target_q[world * coords_per_world + scene_coords[k]] = solution[world, ik_coords[k]]


class BatchedArmIK:
    """One arm's LM solver, over every world at once.

    Args:
        model: The *IK* model -- single-world kinematics the solver evaluates.
        num_worlds: How many independent problems to solve.
        arm_coords: This arm's ``joint_q`` indices **in the IK model**.
        arm_dofs: This arm's ``joint_qd`` indices in the IK model, for the mask.
        scene_coords: Where this arm's coordinates live in the *fleet* model,
            one per entry of ``arm_coords`` and in the same order. This is the
            only place the two index spaces meet.
        ee_body: Body in the IK model whose frame the target is about.
        ee_offset: Tool-centre point in ``ee_body``'s frame [m].
        ee_offset_rot: Rotation from ``ee_body``'s frame to the target frame, xyzw.
        home_pose: ``(position, quat_xyzw)`` to point every world at before
            anything has set a target, so an unsteered world solves to its rest
            pose rather than to the origin.
        iterations: LM iterations per solve. 8, as the single-world path uses.
        rotation_weight: Orientation residual weight relative to position.
    """

    def __init__(
        self,
        model: newton.Model,
        *,
        num_worlds: int,
        arm_coords: Sequence[int],
        arm_dofs: Sequence[int],
        scene_coords: Sequence[int],
        ee_body: int,
        home_pose: tuple[Sequence[float], Sequence[float]],
        ee_offset: Sequence[float] = (0.0, 0.0, 0.0),
        ee_offset_rot: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        iterations: int = 8,
        rotation_weight: float = 0.6,
    ) -> None:
        if len(arm_coords) != len(scene_coords):
            raise ValueError(
                f"arm_coords ({len(arm_coords)}) and scene_coords ({len(scene_coords)}) must pair one to one"
            )
        self.model = model
        self.num_worlds = int(num_worlds)
        self.arm_coords = [int(c) for c in arm_coords]
        self.arm_dofs = [int(d) for d in arm_dofs]
        self.iterations = int(iterations)
        device = model.device

        home_pos, home_quat = np.asarray(home_pose[0], dtype=np.float32), np.asarray(home_pose[1], dtype=np.float32)
        # Targets live in arrays the objectives are *constructed on*, so a solve
        # inside a captured graph reads whatever was written into them from
        # outside it. The single-world path keeps scratch buffers for the same
        # reason; here they are the objectives' own storage and there is no copy.
        self.targets_pos = wp.array(np.tile(home_pos, (self.num_worlds, 1)), dtype=wp.vec3, device=device)
        self.targets_rot = wp.array(np.tile(home_quat, (self.num_worlds, 1)), dtype=wp.vec4, device=device)
        self.enable = wp.zeros(self.num_worlds, dtype=wp.int32, device=device)

        self.pos_objective = ik.IKObjectivePosition(
            link_index=int(ee_body),
            link_offset=wp.vec3(*[float(v) for v in ee_offset]),
            target_positions=self.targets_pos,
        )
        self.rot_objective = ik.IKObjectiveRotation(
            link_index=int(ee_body),
            link_offset_rotation=wp.quat(*[float(v) for v in ee_offset_rot]),
            target_rotations=self.targets_rot,
            weight=float(rotation_weight),
        )
        self.limit_objective = ik.IKObjectiveJointLimit(
            joint_limit_lower=model.joint_limit_lower,
            joint_limit_upper=model.joint_limit_upper,
        )
        mask = np.zeros(model.joint_dof_count, dtype=bool)
        mask[self.arm_dofs] = True
        # AUTODIFF, NOT ANALYTIC, AND THE DIFFERENCE IS PER WORLD. Measured on
        # a gear-pair meshing scene with three problems seeded identically and aimed at one
        # target (Newton 1.5.0.dev0): the analytic Jacobian solved problem 0 to
        # the target and problems 1 and 2 to a second, identical, wrong answer
        # -- q1 0.017 rad, tool centre 18.8 mm off in y -- with or without the
        # dof mask, with or without the limit objective. Autodiff agrees across
        # problems to the last digit. So a wave's world 0 arrived and every
        # other world stopped 19 mm short of every target; on a 1.5 mm fine leg
        # that is a graph that never seats a gear. Whatever the analytic path
        # reads for problems past the first, it is not that problem's state.
        self.solver = ik.IKSolver(
            model=model,
            n_problems=self.num_worlds,
            objectives=[self.pos_objective, self.rot_objective, self.limit_objective],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.AUTODIFF,
            joint_dof_mask=wp.array(mask, dtype=wp.bool, device=device),
        )

        self.ik_coords_wp = wp.array(np.asarray(self.arm_coords, dtype=np.int32), dtype=wp.int32, device=device)
        self.scene_coords_wp = wp.array(np.asarray(scene_coords, dtype=np.int32), dtype=wp.int32, device=device)
        # Host mirrors, so setting one world's target does not read the device
        # back to leave the others alone.
        self._pos_host = np.tile(home_pos, (self.num_worlds, 1))
        self._rot_host = np.tile(home_quat, (self.num_worlds, 1))
        self._enable_host = np.zeros(self.num_worlds, dtype=np.int32)

    # ------------------------------------------------------------------ targets
    def set_target(self, world: int, position: Sequence[float], quat_xyzw: Sequence[float]) -> None:
        """Aim one world's target. Call :meth:`flush` once the batch is set."""
        self._pos_host[int(world)] = position
        self._rot_host[int(world)] = quat_xyzw

    def set_targets(self, positions: np.ndarray, quats_xyzw: np.ndarray) -> None:
        """Aim every world at once, from ``[worlds, 3]`` and ``[worlds, 4]``."""
        self._pos_host[:] = positions
        self._rot_host[:] = quats_xyzw

    def set_enabled(self, enabled: np.ndarray) -> None:
        """Which worlds this arm writes targets for. The rest hold, exactly."""
        self._enable_host[:] = np.asarray(enabled, dtype=np.int32).reshape(-1)

    def enable_world(self, world: int, enabled: bool = True) -> None:
        """Start or stop driving one world, leaving the others as they are.

        The per-world form, because worlds start and stop Cartesian moves
        independently: one graph beginning a servo must not switch off another
        that is mid-move.
        """
        self._enable_host[int(world)] = 1 if enabled else 0

    def enabled(self) -> np.ndarray:
        """Which worlds this arm is driving, as a copy the caller may keep."""
        return self._enable_host.copy()

    def flush(self) -> None:
        """Push host target and enable mirrors to the arrays the graph reads."""
        self.targets_pos.assign(self._pos_host.astype(np.float32))
        self.targets_rot.assign(self._rot_host.astype(np.float32))
        self.enable.assign(self._enable_host)

    @property
    def active(self) -> bool:
        """Whether any world is being driven by this arm right now."""
        return bool(self._enable_host.any())


class RobotIKRig:
    """Every arm of one robot, solved for every world, into the fleet's targets.

    Args:
        ik_model: Single-world model the solver runs its kinematics on. May be
            the fleet's own template, or a reduced build holding only what the
            arm needs -- see this module's docstring for when the difference
            matters.
        num_worlds: Worlds in the fleet.
        coords_per_world: The fleet model's ``joint_q`` stride.
        coord_map: For each IK-model coordinate, the *template* coordinate it
            corresponds to, or -1 where there is none. ``None`` means the two
            index spaces are the same, which is the common case.
        iterations: LM iterations per solve.
    """

    def __init__(
        self,
        ik_model: newton.Model,
        *,
        num_worlds: int,
        coords_per_world: int,
        coord_map: Sequence[int] | None = None,
        iterations: int = 8,
    ) -> None:
        self.model = ik_model
        self.num_worlds = int(num_worlds)
        self.coords_per_world = int(coords_per_world)
        self.iterations = int(iterations)
        self.arms: dict[str, BatchedArmIK] = {}
        device = ik_model.device
        n_ik = int(ik_model.joint_coord_count)

        mapping = np.arange(n_ik, dtype=np.int32) if coord_map is None else np.asarray(coord_map, dtype=np.int32)
        if mapping.size != n_ik:
            raise ValueError(f"coord_map must have {n_ik} entries, got {mapping.size}")
        self.coord_map = wp.array(mapping, dtype=wp.int32, device=device)
        self.defaults = wp.array(
            np.asarray(ik_model.joint_q.numpy(), dtype=np.float32), dtype=wp.float32, device=device
        )
        self.seeds = wp.zeros((self.num_worlds, n_ik), dtype=wp.float32, device=device)
        self._graph = None
        self._graph_key: tuple[int, int] | None = None

    def add_arm(self, name: str, arm: BatchedArmIK) -> None:
        if arm.num_worlds != self.num_worlds:
            raise ValueError(f"arm {name!r} solves {arm.num_worlds} worlds, rig has {self.num_worlds}")
        self.arms[name] = arm

    # -------------------------------------------------------------------- solve
    def solve_into(self, joint_q: wp.array, joint_target_q: wp.array) -> None:
        """Solve every arm for every world, writing the enabled worlds' targets.

        Captured into one graph on the first call per array pair, and replayed
        thereafter. The capture is keyed on the pair because a graph bakes device
        pointers: handed a rebuilt scene's arrays it must recapture rather than
        write into the old ones. Failure is not fatal -- eager is the same
        arithmetic, slower -- which is the rule the substep block and the
        single-world IK solve both already follow.

        A caller assembling a *larger* graph -- the fleet captures the solve, the
        segment drivers and the hands as one stage -- should launch
        :meth:`solve` instead and capture that, rather than nesting a captured
        graph inside another one.
        """
        key = (joint_q.ptr, joint_target_q.ptr)
        if key != self._graph_key:
            self._graph_key = key
            self._graph = None
            if self.model.device.is_cuda:
                try:
                    # One eager pass first, so every kernel module is built
                    # before capture -- compilation inside a capture fails it.
                    self.solve(joint_q, joint_target_q)
                    with capturing(joint_q.device) as capture:
                        self.solve(joint_q, joint_target_q)
                    self._graph = capture.graph
                except Exception as exc:
                    self._graph = None
                    log.warning("batched IK solve capture failed (%s: %s); running eager", type(exc).__name__, exc)
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self.solve(joint_q, joint_target_q)

    def solve(self, joint_q: wp.array, joint_target_q: wp.array) -> None:
        """The batched solve as plain launches, for capturing into a larger graph.

        Gather every world's seed from live state, run each arm's masked solver
        over the shared seed buffer, then scatter the enabled worlds' arm
        coordinates into the fleet's targets. Launches only -- no synchronize, no
        host work -- so it composes into whatever graph the caller is building.
        """
        device = self.model.device
        wp.launch(
            _gather_seeds,
            dim=(self.num_worlds, int(self.model.joint_coord_count)),
            inputs=[joint_q, self.coord_map, self.defaults, self.coords_per_world],
            outputs=[self.seeds],
            device=device,
        )
        for arm in self.arms.values():
            arm.solver.step(self.seeds, self.seeds, iterations=arm.iterations)
        for arm in self.arms.values():
            wp.launch(
                _scatter_targets,
                dim=(self.num_worlds, len(arm.arm_coords)),
                inputs=[
                    self.seeds,
                    arm.ik_coords_wp,
                    arm.scene_coords_wp,
                    arm.enable,
                    self.coords_per_world,
                ],
                outputs=[joint_target_q],
                device=device,
            )

    def flush(self) -> None:
        """Push every arm's host-side targets and enables to the device."""
        for arm in self.arms.values():
            arm.flush()

    def solution(self) -> np.ndarray:
        """The last solve's joint vectors, ``[worlds, ik_coords]``. Syncs."""
        return self.seeds.numpy()

    @property
    def active(self) -> bool:
        return any(arm.active for arm in self.arms.values())
