# SPDX-License-Identifier: Apache-2.0
"""PyRoKi over jaxls: a batched full-pose solve for any arm that ships a URDF.

The default solver family. A wave of ``W`` targets is one jaxls problem --
``W`` joint variables, ``W`` pose costs -- solved on CPU JAX in a few
milliseconds (measured 0.7 ms for one Panda solve, 4.9 ms for twelve, against
22 ms per solve for the finite-difference solver it replaces). Position and
orientation are one cost with an analytic Jacobian
(:func:`pyroki.costs.pose_cost_analytic_jac`); under the roll-free contract the
orientation term is only the tool axis, as a residual differentiated by JAX.

**The URDF has to describe the arm the simulator steps.** Construction
measures the transform from the URDF's tool link to the simulator's tool
centre at two configurations and refuses a URDF whose transform drifts between
them: every solve would otherwise be silently wrong by the drift. That
self-check is the one piece kept from the wrapper this replaces; the wrapper
imported the graph runtime for its solve, and a core solver imports no runtime.

Frames: targets arrive in the world, are restated in the URDF root's frame --
the arm's base body -- and every answer is verified by URDF forward kinematics
plus the calibrated transform, pure JAX and numpy, so a worker thread can
verify without touching the simulator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robosimstudio.core.math import quat_multiply, quat_to_mat

from .arm import ArmKinematics
from .solver import Roll, pose_errors

__all__ = ["CALIBRATION_TOLERANCE_DEG", "CALIBRATION_TOLERANCE_M", "IKSolverPyroki"]

CALIBRATION_TOLERANCE_M = 1.0e-3
"""How far the tool-link-to-TCP transform may drift between the two probes [m] before the URDF is refused."""
CALIBRATION_TOLERANCE_DEG = 0.1
"""The same, in degrees of rotation."""

POS_WEIGHT = 50.0
ORI_WEIGHT = 10.0
"""The pose cost's weights, the graph runtime's own so a solve here lands where its solves landed."""
AXIS_WEIGHT = 10.0
"""The roll-free orientation term's weight: the tool axis alone."""
LIMIT_WEIGHT = 100.0
REST_WEIGHT = 0.01
"""A pull toward the seed, small: it keeps the elbow on the seed's branch and nothing more."""
LAMBDA_INITIAL = 1.0
"""jaxls's trust-region start, the runtime's own."""


def _quiet() -> None:
    """jaxls narrates every problem it analyses at INFO through loguru; a solver is not a log."""
    try:
        from loguru import logger
    except ImportError:
        return
    logger.disable("jaxls")
    logger.disable("pyroki")


def _transform(position: Any, quat_xyzw: Any) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_to_mat(np.asarray(quat_xyzw, dtype=np.float64))
    m[:3, 3] = np.asarray(position, dtype=np.float64)
    return m


def _rotvec(rot: np.ndarray) -> np.ndarray:
    """A rotation matrix as axis times angle."""
    cos = np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos))
    if angle < 1.0e-9:
        return np.zeros(3)
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    return axis / max(np.linalg.norm(axis), 1.0e-12) * angle


def _wxyz_xyz(row: np.ndarray) -> np.ndarray:
    """A pyroki ``wxyz_xyz`` row as a 4x4."""
    w, x, y, z, px, py, pz = (float(v) for v in row)
    return _transform([px, py, pz], [x, y, z, w])


class IKSolverPyroki:
    """A batched full-pose solver over one arm's URDF.

    Args:
        kinematics: The arm, with ``urdf`` and ``base_xform`` set (the URDF
            root is the base body's frame) and a forward kinematics to
            calibrate against.
        urdf: The URDF path, overriding the kinematics' own.
        joints: The URDF's actuated joints this arm's coordinates drive, in
            coordinate order; the URDF's first ``dof`` actuated joints when not
            given. Every other actuated joint (fingers) is held at zero.
        link: The URDF link the tool centre is calibrated against; the child
            of the last driven joint when not given.
    """

    capabilities = frozenset({"batched", "needs_urdf"})

    def __init__(
        self,
        kinematics: ArmKinematics,
        *,
        urdf: str = "",
        joints: tuple[str, ...] = (),
        link: str = "",
    ) -> None:
        import pyroki as pk
        import yourdfpy

        _quiet()
        self.kinematics = kinematics
        path = urdf or kinematics.urdf
        if not path:
            raise ValueError("PyRoKi solves an arm that ships a URDF; this one names none (EmbodimentSpec.urdf)")
        from robosimstudio.paths import resolve_path

        self.urdf = str(resolve_path(path))
        model = yourdfpy.URDF.load(
            self.urdf, load_meshes=False, build_collision_scene_graph=False, load_collision_meshes=False
        )
        self._robot = pk.Robot.from_urdf(model)
        actuated = list(self._robot.joints.actuated_names)
        wanted = list(joints) or actuated[: kinematics.dof]
        if len(wanted) != kinematics.dof or any(name not in actuated for name in wanted):
            raise ValueError(
                f"the URDF's actuated joints {actuated} do not cover this arm's {kinematics.dof} coordinates ({wanted})"
            )
        self._slots = np.array([actuated.index(name) for name in wanted], dtype=np.intp)
        self._n_actuated = len(actuated)
        link_names = list(self._robot.links.names)
        if link:
            self._link = link_names.index(link)
        else:
            joint_index = list(self._robot.joints.names).index(wanted[-1])
            # pyroki numbers each joint's child link as the joint's own index plus
            # one (the root is link 0), which is the frame the last driven joint moves.
            self._link = int(joint_index) + 1
        if kinematics.base_xform is not None:
            self._t_base = _transform(kinematics.base_xform[:3], kinematics.base_xform[3:7])
        else:
            # No base body to read: the arm stands on the world, its mount
            # collapsed into it. The root frame is then *measured* from motion
            # (:meth:`_root_from_motion`), and the calibration below checks it.
            self._t_base = self._root_from_motion()
        self._t_base_inv = np.linalg.inv(self._t_base)
        self._t_ht = self._calibrate()
        self._t_ht_inv = np.linalg.inv(self._t_ht)
        self._solvers: dict[tuple[int, str], Any] = {}

    # ------------------------------------------------------------- calibration
    def _root_from_motion(self) -> np.ndarray:
        """The URDF root's world pose, from how the tool moves when each joint moves.

        With ``T_i = T_root @ L_i @ T_tool`` for the model's tool pose ``T_i``
        and the URDF's tool-link pose ``L_i`` at probe ``i``, the unknown tool
        transform cancels out of ``T_i @ T_0^-1 = T_root @ (L_i @ L_0^-1) @
        T_root^-1``: the classic hand-eye problem in ``T_root`` alone. One
        probe per joint, each bending that joint from home, gives one rotation
        pair per axis; the rotation is the least-squares alignment of the two
        axis sets and the translation follows linearly. The tool-link
        calibration that runs after this is the check on it -- a wrong root
        shows up there as drift and the URDF is refused.
        """
        kin = self.kinematics
        home = np.asarray(kin.home_q, dtype=np.float64)
        probes = [home]
        for i in range(kin.dof):
            bent = home.copy()
            bent[i] += 0.4
            probes.append(bent)
        probes = np.stack(probes)
        pos, quat = kin.poses(probes)
        world = np.stack([_transform(pos[i], quat[i]) for i in range(len(probes))])
        links = self._urdf_link(probes)
        a_axes, b_axes, a_list, b_list = [], [], [], []
        for i in range(1, len(probes)):
            a = world[i] @ np.linalg.inv(world[0])
            b = links[i] @ np.linalg.inv(links[0])
            ra, rb = _rotvec(a[:3, :3]), _rotvec(b[:3, :3])
            if np.linalg.norm(ra) < 1.0e-6 or np.linalg.norm(rb) < 1.0e-6:
                continue  # a joint that moves nothing says nothing about the frame
            a_axes.append(ra / np.linalg.norm(ra))
            b_axes.append(rb / np.linalg.norm(rb))
            a_list.append(a)
            b_list.append(b)
        if len(a_axes) < 2:
            raise ValueError("the root frame cannot be measured: fewer than two joints move the tool")
        u, _, vt = np.linalg.svd(np.asarray(a_axes).T @ np.asarray(b_axes))
        d = np.sign(np.linalg.det(u @ vt))
        rot = u @ np.diag([1.0, 1.0, d]) @ vt
        # (R_a - I) t = R t_b - t_a, stacked over the pairs.
        lhs = np.concatenate([a[:3, :3] - np.eye(3) for a in a_list])
        rhs = np.concatenate([rot @ b[:3, 3] - a[:3, 3] for a, b in zip(a_list, b_list, strict=True)])
        t, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        root = np.eye(4)
        root[:3, :3] = rot
        root[:3, 3] = t
        return root

    def _cfg(self, arm_q: np.ndarray) -> np.ndarray:
        """A full URDF configuration ``[W, n_actuated]`` with the arm's coordinates in their slots."""
        rows = np.asarray(arm_q, dtype=np.float64).reshape(-1, self.kinematics.dof)
        cfg = np.zeros((rows.shape[0], self._n_actuated))
        cfg[:, self._slots] = rows
        return cfg

    def _urdf_link(self, arm_q: np.ndarray) -> np.ndarray:
        """The tool link's pose in the URDF root frame, per row: ``[W, 4, 4]``."""
        cfg = self._cfg(arm_q)
        rows = np.asarray(self._robot.forward_kinematics(cfg))[:, self._link]
        return np.stack([_wxyz_xyz(row) for row in rows])

    def _calibrate(self) -> np.ndarray:
        """``T_link->tcp``, and with it the proof that the URDF matches the model."""
        kin = self.kinematics
        home = np.asarray(kin.home_q, dtype=np.float64)
        bent = home.copy()
        for i in range(kin.dof):
            bent[i] += 0.25 if i % 2 else -0.25
        probes = np.stack([home, bent])
        pos, quat = kin.poses(probes)
        links = self._urdf_link(probes)
        transforms = [
            np.linalg.inv(links[i]) @ (self._t_base_inv @ _transform(pos[i], quat[i])) for i in range(len(probes))
        ]
        first, second = transforms
        dp = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
        cos = np.clip((np.trace(second[:3, :3].T @ first[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        dr = float(np.degrees(np.arccos(cos)))
        if dp > CALIBRATION_TOLERANCE_M or dr > CALIBRATION_TOLERANCE_DEG:
            raise ValueError(
                f"URDF/model kinematic mismatch: the tool-link->TCP transform drifted {dp * 1000:.2f} mm / "
                f"{dr:.3f} deg between two configurations. {self.urdf} does not describe this scene's arm; "
                "every solve would be off by that drift."
            )
        return first

    # ------------------------------------------------------------------- solve
    def tcp_poses(self, arm_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The tool centre in the world for every row, by URDF kinematics: ``(position [W, 3], quat_xyzw [W, 4])``."""
        from robosimstudio.core.math import mat_to_quat

        tcp = self._t_base @ self._urdf_link(arm_q) @ self._t_ht
        return tcp[:, :3, 3].copy(), np.stack([mat_to_quat(m[:3, :3]) for m in tcp])

    def _solver(self, worlds: int, roll: str):
        key = (int(worlds), roll)
        if key not in self._solvers:
            self._solvers[key] = self._build(worlds, roll)
        return self._solvers[key]

    def _build(self, worlds: int, roll: str):
        import jax
        import jax.numpy as jnp
        import jaxlie
        import jaxls
        import pyroki as pk

        robot = self._robot
        link = self._link
        t_ht = jnp.asarray(self._t_ht)
        t_ht_inv = jnp.asarray(self._t_ht_inv)
        slots = jnp.asarray(self._slots)
        n_act = self._n_actuated

        @jaxls.Cost.create_factory
        def tcp_axis_cost(vals, joint_var, target_position, target_axis, pos_weight, axis_weight):
            """Position of the tool centre, and its +z against the target's: the roll-free contract."""
            cfg = vals[joint_var]
            row = robot.forward_kinematics(cfg)[link]
            link_pose = jaxlie.SE3(row).as_matrix()
            tcp = link_pose @ t_ht
            residual_p = (tcp[:3, 3] - target_position) * pos_weight
            residual_a = jnp.cross(tcp[:3, 2], target_axis) * axis_weight
            return jnp.concatenate([residual_p, residual_a])

        def solve_one(seed_cfg, target_pos, target_wxyz, target_axis):
            """One world's problem, in the one-row shapes the batched costs take."""
            seed_cfg, target_pos, target_wxyz, target_axis = (
                x[None] for x in (seed_cfg, target_pos, target_wxyz, target_axis)
            )
            broadcast = jax.tree.map(lambda x: x[None], robot)
            joint_var = robot.joint_var_cls(jnp.arange(1))
            costs = [
                pk.costs.limit_cost(broadcast, joint_var, weight=LIMIT_WEIGHT),
                pk.costs.rest_cost(joint_var, seed_cfg, weight=REST_WEIGHT),
            ]
            if roll == "held":
                link_target = jaxlie.SE3.from_matrix(
                    jax.vmap(lambda p, q: jaxlie.SE3(jnp.concatenate([q, p])).as_matrix() @ t_ht_inv)(
                        target_pos, target_wxyz
                    )
                )
                costs.append(
                    pk.costs.pose_cost_analytic_jac(
                        broadcast,
                        joint_var,
                        link_target,
                        jnp.full((1,), link),
                        pos_weight=POS_WEIGHT,
                        ori_weight=ORI_WEIGHT,
                    )
                )
            else:
                costs.append(
                    tcp_axis_cost(
                        joint_var,
                        target_pos,
                        target_axis,
                        jnp.full((1,), POS_WEIGHT),
                        jnp.full((1,), AXIS_WEIGHT),
                    )
                )
            problem = jaxls.LeastSquaresProblem(costs, [joint_var]).analyze()
            solution = problem.solve(
                initial_vals=jaxls.VarValues.make([joint_var.with_value(seed_cfg)]),
                verbose=False,
                linear_solver="dense_cholesky",
                trust_region=jaxls.TrustRegionConfig(lambda_initial=LAMBDA_INITIAL),
            )
            return solution[joint_var][0]

        def solve(seed_cfg, target_pos, target_wxyz, target_axis):
            # Every row its own least-squares problem -- its own trust region,
            # its own termination -- rather than one problem over the wave.
            # Solved jointly, the rows shared one damping and one stopping
            # rule, so a world's answer depended on which other worlds sat
            # beside it in the wave: the same episode converged in one wave
            # and stalled short of a servo's ball in another. ``vmap`` keeps
            # each row's loop running until that row alone has converged.
            return jax.vmap(solve_one)(seed_cfg, target_pos, target_wxyz, target_axis)

        del n_act, slots, worlds
        return jax.jit(solve)

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
        import jax.numpy as jnp

        del iterations  # jaxls converges on its own termination, not a budget
        kin = self.kinematics
        seed = np.asarray(seed, dtype=np.float64).reshape(-1, kin.dof)
        n_worlds = seed.shape[0]
        position = np.asarray(position, dtype=np.float64).reshape(n_worlds, 3)
        quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(n_worlds, 4)
        # Targets in the URDF root's frame.
        base_pos = np.zeros((n_worlds, 3))
        base_wxyz = np.zeros((n_worlds, 4))
        base_axis = np.zeros((n_worlds, 3))
        base_quat_xyzw = np.zeros((n_worlds, 4))
        for w in range(n_worlds):
            local = self._t_base_inv @ _transform(position[w], quat[w])
            base_pos[w] = local[:3, 3]
            from robosimstudio.core.math import mat_to_quat

            q = mat_to_quat(local[:3, :3])
            base_quat_xyzw[w] = q
            base_wxyz[w] = [q[3], q[0], q[1], q[2]]
            base_axis[w] = local[:3, 2]
        solver = self._solver(n_worlds, "held" if roll == "held" else "free")
        cfg = np.asarray(
            solver(jnp.asarray(self._cfg(seed)), jnp.asarray(base_pos), jnp.asarray(base_wxyz), jnp.asarray(base_axis))
        )
        solved = cfg[:, self._slots]
        solved = np.clip(solved, kin.lower, kin.upper)
        if active is not None:
            live = np.asarray(active, dtype=bool)
            solved = np.where(live[:, None], solved, seed)
        pos, got = self.tcp_poses(solved)
        return solved, pose_errors(pos, got, position, quat, roll=roll)


def tcp_offset_quat(link_quat_xyzw: np.ndarray, t_ht: np.ndarray) -> np.ndarray:
    """The tool frame's orientation from a link's, through the calibrated transform (xyzw)."""
    from robosimstudio.core.math import mat_to_quat

    return quat_multiply(np.asarray(link_quat_xyzw, dtype=np.float64), mat_to_quat(t_ht[:3, :3]))
