# SPDX-License-Identifier: Apache-2.0
"""Cartesian control of one arm through masked inverse kinematics.

:class:`ArmIK` takes DOF indices explicitly rather than reading them off a scene
object, which is what lets one implementation serve a Panda, one YAM arm, and
*both* YAM arms at once over the same model -- each instance masking only its
own joints. Without the mask the solver happily "solves" a loose tool's free
joint into the target, since everything in a scene shares one ``joint_q``.

**The solver is PyRoKi** (:mod:`robosimstudio.solvers.ik.pyroki`): a full-pose
solve over the arm's URDF, built the first time a solve is asked for and used
by every entry point here -- :meth:`solve_into` for a tracked target,
:meth:`solve_q` and :meth:`solve_q_full` for a waypoint. An arm that ships no
URDF, or a process where ``pyroki`` does not import, falls back to Newton's own
solvers with one printed line and no change of interface:

* :meth:`solve_into` then tracks on device at a few LM iterations per frame,
  captured in a graph.
* :meth:`solve_q` then converges a waypoint on the host with damped least
  squares against an approach-axis objective, clipping to joint limits every
  iteration.

Nothing selects between them from outside: the URDF decides.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import newton
import newton.ik as ik
import numpy as np
import warp as wp

from robosimstudio.core.math import mat_to_quat, quat_multiply, quat_to_mat, transform_matrix
from robosimstudio.core.streams import capturing

log = logging.getLogger(__name__)

__all__ = ["ArmIK"]


class _ChainFK:
    """Host-side FK for one arm's serial chain: the tool pose in ~50 us.

    :meth:`ArmIK.fk_ee` used to run ``newton.eval_fk`` over the whole model and
    read the full ``body_q`` back -- a device round-trip measured at 2.12 ms.
    That is invisible per call and ruinous where the call actually happens:
    :meth:`ArmIK.solve_q` builds its Jacobian by finite differences, nine FK
    evaluations per iteration, and the weave's station planner was spending
    157-485 s per episode inside exactly that loop while the GPU sat idle
    between 4 us kernels. The chain is eight joints of fixed transforms; walked
    on the host it costs 51 us, 41x less, and the whole plan collapses to
    seconds. Measured 2026-08-11 on a cable-routing scene.

    Built once off the model arrays by :meth:`ArmIK._build_chain_fk`, which
    also decides whether this chain is one the walk can represent (fixed,
    revolute and prismatic joints ending at the world or at a jointless body)
    and validates the result against the device path before trusting it --
    an FK that silently disagreed with the simulator would move every plan
    in the repository at once.

    **Opt-in per solve scope, never the default** (:meth:`ArmIK.host_fk`).
    Micrometre agreement is not the same as the same *solutions*: the FD
    descent in :meth:`ArmIK.solve_q` is a fixed-point search whose basin
    choice can flip on float64-vs-float32 noise, and the wrap's keyframes
    were tuned against the device path's exact fixed points. Swapping the
    walk in globally resampled that lottery -- measured 2026-08-11 as the
    wrap's retained turns scattering to 0.18-0.96 (2/5 episodes passing)
    where the stock plan holds the 0.55-0.96 band. A planner that re-solves
    against the live scene each time -- the weave -- has no tuned fixed
    points to lose and takes the 41x; a plan that ships tuned numbers keeps
    the solver those numbers were tuned on.
    """

    __slots__ = ("ee_offset", "ee_offset_rot", "joints", "x_root")

    PRISMATIC = int(newton.JointType.PRISMATIC)
    REVOLUTE = int(newton.JointType.REVOLUTE)

    def __init__(
        self, joints: list[dict], x_root: np.ndarray, ee_offset: np.ndarray, ee_offset_rot: np.ndarray
    ) -> None:
        self.joints = joints
        self.x_root = x_root
        self.ee_offset = ee_offset
        self.ee_offset_rot = ee_offset_rot

    def ee(self, joint_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The tool-centre pose of a full ``joint_q``, matching :meth:`ArmIK.ee_pose`."""
        q = np.asarray(joint_q, dtype=np.float64)
        m = self.x_root
        for joint in self.joints:
            kind = joint["type"]
            if kind == self.REVOLUTE:
                x_j = np.eye(4)
                x_j[:3, :3] = quat_to_mat(_axis_angle_quat(joint["axis"], q[joint["q_start"]]))
            elif kind == self.PRISMATIC:
                x_j = np.eye(4)
                x_j[:3, 3] = joint["axis"] * q[joint["q_start"]]
            else:  # FIXED
                x_j = None
            step = joint["x_pj"] if x_j is None else joint["x_pj"] @ x_j
            m = m @ step @ joint["x_cj_inv"]
        quat = mat_to_quat(m[:3, :3])
        return m[:3, 3] + m[:3, :3] @ self.ee_offset, quat_multiply(quat, self.ee_offset_rot)


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    half = 0.5 * float(angle)
    s = np.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)])


@wp.kernel
def _write_arm_targets(
    joint_q_ik: wp.array2d[wp.float32],
    arm_coords: wp.array[wp.int32],
    joint_target_q: wp.array[wp.float32],
):
    """Copy the IK solution's arm coordinates into the solver's position targets."""
    i = wp.tid()
    joint_target_q[arm_coords[i]] = joint_q_ik[0, arm_coords[i]]


@wp.kernel
def _hold_arm_targets(
    joint_q: wp.array[wp.float32],
    arm_coords: wp.array[wp.int32],
    joint_target_q: wp.array[wp.float32],
):
    """Make this arm's targets its achieved coordinates. See :meth:`ArmIK.hold_into`."""
    i = wp.tid()
    joint_target_q[arm_coords[i]] = joint_q[arm_coords[i]]


@wp.kernel
def _write_coord_values(
    values: wp.array[wp.float32],
    coords: wp.array[wp.int32],
    joint_target_q: wp.array[wp.float32],
):
    """Write per-coordinate values into the solver's position targets."""
    i = wp.tid()
    joint_target_q[coords[i]] = values[i]


class ArmIK:
    """Inverse kinematics for one arm, masked off the rest of the model.

    Args:
        model: The finalized model both the IK and the simulation run on.
        arm_coords: ``joint_q`` indices of this arm's actuated joints.
        arm_dofs: ``joint_qd`` indices of the same joints, for the solve mask.
        ee_body: Body whose frame the target is expressed about.
        home_state: A state already at the home posture, used to read the
            end-effector's resting pose so :attr:`home_pose` has something to
            report before anything has been solved.
        ee_offset: Tool-centre point in ``ee_body``'s frame [m]. The arena
            YAM's measured grip point sits at (0, -0.044, 0.130) in its wrist
            frame -- off the wrist axis, where the pads actually are; a Panda
            imported with a ``gripper0_eef`` body needs no offset at all.
        ee_offset_rot: Rotation from ``ee_body``'s frame to the frame the
            orientation target is expressed in, xyzw.
        iterations: LM iterations per :meth:`solve_into`. 8 tracks a dragged
            gizmo without visible lag inside a 60 Hz frame budget.
        rotation_weight: Weight on the orientation residual relative to
            position. Below 1.0 the solver gives up orientation before
            position, which is the trade you want when a target is dragged to
            the edge of the workspace.
        urdf: The arm's URDF, as its embodiment names it. The PyRoKi solver is
            built from it; empty means the Newton solvers serve this arm.
        base_xform: World pose of the arm's base body ``(x, y, z, qx, qy, qz, qw)``,
            the URDF root's frame. Read off ``home_state`` when not given.
        base_body: The base body's index, for reading ``base_xform`` off the
            home state; -1 when unknown.
        home_q: The arm's coordinates at its home posture; read off the
            model's ``joint_q`` when not given.
    """

    def __init__(
        self,
        model: newton.Model,
        *,
        arm_coords: Sequence[int],
        arm_dofs: Sequence[int],
        ee_body: int,
        home_state: newton.State,
        ee_offset: Sequence[float] = (0.0, 0.0, 0.0),
        ee_offset_rot: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        iterations: int = 8,
        rotation_weight: float = 0.6,
        urdf: str = "",
        base_xform: Sequence[float] | None = None,
        base_body: int = -1,
        home_q: Sequence[float] | None = None,
    ) -> None:
        self.model = model
        self.arm_coords = [int(c) for c in arm_coords]
        self.arm_dofs = [int(d) for d in arm_dofs]
        self.ee_body = int(ee_body)
        self.iterations = int(iterations)
        self.ee_offset = np.asarray(ee_offset, dtype=np.float64)
        self.ee_offset_rot = np.asarray(ee_offset_rot, dtype=np.float64)

        self._home_pos, self._home_quat = self.ee_pose(home_state)
        self.urdf = str(urdf or "")
        template = np.asarray(model.joint_q.numpy(), dtype=np.float64)
        self._template = template
        self._home_q = np.asarray(home_q, dtype=np.float64) if home_q is not None else template[self.arm_coords].copy()
        if base_xform is not None:
            self._base_xform: np.ndarray | None = np.asarray(base_xform, dtype=np.float64).reshape(7)
        elif int(base_body) >= 0:
            self._base_xform = np.asarray(home_state.body_q.numpy()[int(base_body)], dtype=np.float64)
        else:
            self._base_xform = None
        self._pyroki: Any = None
        self._pyroki_tried = False
        self.solver_name = ""
        """Which solver serves this arm: ``"pyroki"``, ``"newton"`` or ``""`` before the first solve."""
        self._fk_state: newton.State | None = None
        self.fk_host_reason = ""
        """Why the host chain walk is unavailable, or empty when it validated.
        Introspection, not control flow."""
        self._chain_fk = self._build_chain_fk(home_state)
        self._fk_host_active = 0
        """Depth of open :meth:`host_fk` scopes. The walk runs only inside one."""
        self.joint_q_ik = wp.zeros((1, model.joint_coord_count), dtype=wp.float32, device=model.device)

        self.lm_hostile = self._lm_hostility(model)
        if self.lm_hostile is None:
            self.pos_objective = ik.IKObjectivePosition(
                link_index=self.ee_body,
                link_offset=wp.vec3(*[float(v) for v in self.ee_offset]),
                target_positions=wp.array([wp.vec3(*self._home_pos)], dtype=wp.vec3),
            )
            self.rot_objective = ik.IKObjectiveRotation(
                link_index=self.ee_body,
                link_offset_rotation=wp.quat(*[float(v) for v in self.ee_offset_rot]),
                target_rotations=wp.array([wp.vec4(*self._home_quat)], dtype=wp.vec4),
                weight=float(rotation_weight),
            )
            self.limit_objective = ik.IKObjectiveJointLimit(
                joint_limit_lower=model.joint_limit_lower,
                joint_limit_upper=model.joint_limit_upper,
            )
            self.solver = ik.IKSolver(
                model=model,
                n_problems=1,
                objectives=[self.pos_objective, self.rot_objective, self.limit_objective],
                lambda_initial=0.1,
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
                joint_dof_mask=self._dof_mask(),
            )
        else:
            self.pos_objective = self.rot_objective = self.limit_objective = None
            self.solver = None
            log.warning("%s; cartesian tracking runs the FD solver instead", self.lm_hostile)

        # Scratch reused every frame. The objectives' setters copy into their
        # existing buffers, so the solve can sit inside a captured graph while
        # these keep being refreshed from outside it.
        self._pos_scratch = wp.zeros(1, dtype=wp.vec3, device=model.device)
        self._rot_scratch = wp.zeros(1, dtype=wp.vec4, device=model.device)
        self._pos_host = np.zeros((1, 3), dtype=np.float32)
        self._rot_host = np.zeros((1, 4), dtype=np.float32)
        self._arm_coords_wp = wp.array(np.asarray(self.arm_coords, dtype=np.int32), dtype=wp.int32, device=model.device)
        self._arm_dofs_wp = wp.array(np.asarray(self.arm_dofs, dtype=np.int32), dtype=wp.int32, device=model.device)
        self._values_wp = wp.zeros(len(self.arm_coords), dtype=wp.float32, device=model.device)
        self._solve_graph = None
        self._solve_graph_key: tuple[int, int] | None = None

    # ------------------------------------------------------------------ pyroki
    def _pyroki_solver(self):
        """The PyRoKi solver over this arm, built on first use; ``None`` where the Newton solvers serve.

        Lazy, because the build imports JAX and loads the URDF, which a scene
        nobody solves IK on should not pay for. It falls back -- and says so,
        once -- when ``pyroki`` does not import, when the embodiment names no
        URDF, or when the URDF does not describe this model's arm (the
        solver calibrates and refuses on drift).
        """
        if self._pyroki_tried:
            return self._pyroki
        self._pyroki_tried = True
        reason = ""
        if not self.urdf:
            reason = "the embodiment names no URDF"
        else:
            try:
                from robosimstudio.solvers.ik.arm import ArmKinematics
                from robosimstudio.solvers.ik.pyroki import IKSolverPyroki

                kinematics = ArmKinematics.of(
                    self,
                    template=self._template,
                    urdf=self.urdf,
                    home_q=self._home_q,
                    base_xform=self._base_xform,
                )
                self._pyroki = IKSolverPyroki(kinematics)
            except ImportError as exc:
                reason = f"pyroki is not installed ({exc})"
            except SyntaxError as exc:
                # jaxls (pyroki's solver) uses Python 3.12 syntax; on an older
                # interpreter importing it is a SyntaxError, not an ImportError.
                reason = f"pyroki's jaxls needs Python 3.12 or newer ({type(exc).__name__}: {exc})"
                warnings.warn(f"{reason}; this arm is solved by Newton's own IK instead", stacklevel=2)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
        if self._pyroki is None:
            self.solver_name = "newton"
            if self.urdf:
                log.warning("%s; this arm is solved by Newton's own IK", reason)
            else:
                log.info("%s; this arm is solved by Newton's own IK", reason)
        else:
            self.solver_name = "pyroki"
        return self._pyroki

    @staticmethod
    def _quat_with_z(axis: np.ndarray) -> np.ndarray:
        """Any orientation (xyzw) whose +z is ``axis``: the roll-free target a full-pose solver takes."""
        z = np.asarray(axis, dtype=np.float64)
        z = z / max(np.linalg.norm(z), 1.0e-12)
        helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(helper, z)
        x = x / max(np.linalg.norm(x), 1.0e-12)
        y = np.cross(z, x)
        return mat_to_quat(np.stack([x, y, z], axis=1))

    def _dof_mask(self) -> wp.array:
        """Per-DOF mask selecting only this arm's joints."""
        mask = np.zeros(self.model.joint_dof_count, dtype=bool)
        mask[self.arm_dofs] = True
        return wp.array(mask, dtype=wp.bool, device=self.model.device)

    @staticmethod
    def _lm_hostility(model: newton.Model) -> str | None:
        """Why Newton's LM solver must not even be *built* for this model.

        Both limits are measured, not conjectural. The analytic-jacobian
        kernels specialize over the model's joint types, and a model holding
        CABLE or D6 joints sends that warp module into a >40-minute
        single-core build (killed builds cache nothing). Independently, the
        solver's ``tile_cholesky`` sizes its tile by ``joint_coord_count``:
        at 124 coords the LTO wants 246 kB of shared memory against sm_120's
        99 kB and the *compile itself* raises. Overflow scales with coords^2,
        putting the ceiling near 79 on that device; 64 leaves headroom for
        smaller cards. Returns the reason, or ``None`` when LM is safe.
        """
        simple = {
            int(newton.JointType.REVOLUTE),
            int(newton.JointType.PRISMATIC),
            int(newton.JointType.BALL),
            int(newton.JointType.FREE),
            int(newton.JointType.FIXED),
        }
        exotic = {int(t) for t in model.joint_type.numpy().tolist()} - simple
        if exotic:
            names = {int(getattr(newton.JointType, n)): n for n in ("CABLE", "D6", "DISTANCE")}
            listed = ", ".join(sorted(names.get(t, str(t)) for t in exotic))
            return f"model holds {listed} joints (LM kernel specialization is pathological)"
        if model.joint_coord_count > 64:
            return f"model has {model.joint_coord_count} joint coords (LM cholesky tile overflows shared memory)"
        return None

    @property
    def home_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Tool-centre position [m] and xyzw quaternion at the home posture."""
        return self._home_pos.copy(), self._home_quat.copy()

    # ------------------------------------------------------------- per-frame
    def set_target(self, position: Sequence[float], quat_xyzw: Sequence[float]) -> None:
        """Aim the objectives at a world pose, without solving."""
        self._pos_host[0] = position
        self._rot_host[0] = quat_xyzw
        if self.solver is None:
            return  # the FD path reads the host copies at solve time
        self._pos_scratch.assign(self._pos_host)
        self._rot_scratch.assign(self._rot_host)
        self.pos_objective.set_target_positions(self._pos_scratch)
        self.rot_objective.set_target_rotations(self._rot_scratch)

    def hold_into(self, joint_q: wp.array, joint_target_q: wp.array) -> None:
        """Make this arm's targets the configuration it is already in.

        The opposite of :meth:`solve_into`, and for one situation: something
        outside the servo is moving the arm and the servo must stop objecting.
        A position-controlled arm holds ``joint_target_q``, so a wrench hung off
        one of its links (the studio's link handles) works against a spring
        anchored wherever the target was last written, deflects the arm a little
        and lets it snap back. Retargeting to the achieved pose every step
        leaves no error to restore, so the arm goes where it is pulled and stays
        there when released.

        Not the same as writing nothing: leaving the target alone freezes it at
        a pose the arm is being dragged away from, which is the spring again.

        On device and with no readback, like :meth:`solve_into` -- a per-step
        host round-trip is what this is instead of.
        """
        wp.launch(
            _hold_arm_targets,
            dim=len(self.arm_coords),
            inputs=[joint_q, self._arm_coords_wp],
            outputs=[joint_target_q],
            device=self.model.device,
        )

    def solve_into(self, joint_q: wp.array, joint_target_q: wp.array) -> None:
        """Solve from the current configuration, writing this arm's targets.

        Stays entirely on device: reading the solution back would cost a
        synchronize per frame and rule out capturing the solve in a graph.

        And it **is** captured in a graph, lazily, on the first call per
        ``(joint_q, joint_target_q)`` pair. The solve's cost on the device is
        small; what a per-step tracking loop actually pays is the *host* time
        of issuing it -- eight LM iterations unroll into on the order of a
        hundred kernel launches, measured at 2.75 ms per step of pure launch
        overhead in the studio's loop. Replaying them as one graph is the same
        arithmetic in one launch. The pieces were already shaped for it: the
        seed is re-copied from ``joint_q`` inside the captured body, and the
        objectives' targets live in scratch buffers the graph reads by pointer
        while :meth:`set_target` refreshes them from outside -- that is what
        the scratch existed for.

        Keyed on the array pair rather than captured once, because a graph
        bakes device pointers: handed a different state or control array (a
        rebuilt scene) it must recapture rather than write into the old one.
        Capture failing is not fatal -- eager is the same arithmetic slower,
        exactly like the substep block's own capture.

        Seeding from the live ``joint_q`` rather than the previous solution
        keeps IK and physics from drifting apart when the arm is blocked -- the
        solver then works from where the arm actually is.

        On an LM-hostile model (:meth:`_lm_hostility`) the same call runs the
        FD solver on the host instead: one readback per frame, and the solve
        early-exits once converged, so warm tracking costs a couple of
        iterations. Slower than LM, but LM cannot compile there at all.
        """
        solver = self._pyroki_solver()
        if solver is not None:
            full = joint_q.numpy().astype(np.float64)
            solved, _ = solver.solve(
                full[self.arm_coords],
                self._pos_host[0].astype(np.float64),
                self._rot_host[0].astype(np.float64),
                roll="held",
            )
            full[self.arm_coords] = solved[0]
            self.joint_q_ik.assign(full.astype(np.float32).reshape(1, -1))
            wp.launch(
                _write_arm_targets,
                dim=len(self.arm_coords),
                inputs=[self.joint_q_ik, self._arm_coords_wp],
                outputs=[joint_target_q],
                device=self.model.device,
            )
            return
        if self.solver is None:
            seed = joint_q.numpy().astype(np.float64)
            solution = self.solve_q_full(
                seed,
                self._pos_host[0].astype(np.float64),
                self._rot_host[0].astype(np.float64),
                polish_passes=1,
                iterations=40,
            )
            self.joint_q_ik.assign(solution.astype(np.float32).reshape(1, -1))
            wp.launch(
                _write_arm_targets,
                dim=len(self.arm_coords),
                inputs=[self.joint_q_ik, self._arm_coords_wp],
                outputs=[joint_target_q],
                device=self.model.device,
            )
            return
        key = (joint_q.ptr, joint_target_q.ptr)
        if key != self._solve_graph_key:
            self._solve_graph_key = key
            self._solve_graph = None
            if self.model.device.is_cuda:
                try:
                    # One eager pass first, so every kernel module is built
                    # before capture -- compilation inside a capture fails it.
                    self._solve_eager(joint_q, joint_target_q)
                    with capturing(joint_q.device) as capture:
                        self._solve_eager(joint_q, joint_target_q)
                    self._solve_graph = capture.graph
                except Exception as exc:
                    self._solve_graph = None
                    log.warning("IK solve capture failed (%s: %s); running eager", type(exc).__name__, exc)
        if self._solve_graph is not None:
            wp.capture_launch(self._solve_graph)
        else:
            self._solve_eager(joint_q, joint_target_q)

    def _solve_eager(self, joint_q: wp.array, joint_target_q: wp.array) -> None:
        """The LM solve as plain launches: the body the graph is captured from."""
        wp.copy(self.joint_q_ik.reshape((self.model.joint_coord_count,)), joint_q)
        self.solver.step(self.joint_q_ik, self.joint_q_ik, iterations=self.iterations)
        wp.launch(
            _write_arm_targets,
            dim=len(self.arm_coords),
            inputs=[self.joint_q_ik, self._arm_coords_wp],
            outputs=[joint_target_q],
            device=self.model.device,
        )

    def set_joint_targets(self, arm_values: Sequence[float], joint_target_q: wp.array, *, by_dof: bool = False) -> None:
        """Write this arm's targets directly, for joint-space control.

        Touches only this arm's entries, on device -- a host-side write of the
        whole target vector would clobber whatever the other arm's IK just put
        there.

        Args:
            arm_values: One value per arm joint [m or rad].
            joint_target_q: The solver's target vector to write into.
            by_dof: Index by ``joint_qd`` rather than ``joint_q``. Newton sizes
                ``joint_target_q`` by ``joint_dof_count`` unless
                ``newton.use_coord_layout_targets`` is set, and the two differ
                for any model holding free joints (7 coords, 6 DOFs each).
        """
        self._values_wp.assign(np.asarray(arm_values, dtype=np.float32))
        wp.launch(
            _write_coord_values,
            dim=len(self.arm_coords),
            inputs=[self._values_wp, self._arm_dofs_wp if by_dof else self._arm_coords_wp],
            outputs=[joint_target_q],
            device=self.model.device,
        )

    # -------------------------------------------------------------- kinematics
    def ee_pose(self, state: newton.State) -> tuple[np.ndarray, np.ndarray]:
        """World pose of the tool-centre point: the body frame plus the offset."""
        tf = state.body_q.numpy()[self.ee_body]
        pos = np.array(tf[:3], dtype=np.float64)
        quat = np.array(tf[3:7], dtype=np.float64)
        return pos + quat_to_mat(quat) @ self.ee_offset, quat_multiply(quat, self.ee_offset_rot)

    def fk_ee(self, joint_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tool-centre pose a joint vector would put the arm at (pure FK).

        The device path, unless the caller opened a :meth:`host_fk` scope and
        construction validated the chain walk -- see :class:`_ChainFK` for
        both the speedup and the reason it is not simply the default.
        """
        if self._fk_host_active and self._chain_fk is not None:
            return self._chain_fk.ee(joint_q)
        return self._fk_ee_device(joint_q)

    @contextmanager
    def host_fk(self):
        """Run :meth:`fk_ee` -- and so :meth:`solve_q` -- on the host chain walk.

        For a planner that solves against the live scene and pays thousands of
        FK evaluations per plan: the weave's station planner went from 157-485 s
        to seconds inside this scope. A plan built from *tuned* keyframes should
        stay outside it, on the solver its numbers were tuned against; the
        class docstring carries that measurement. No-op (device path) when the
        chain did not validate, so a caller never has to check first.
        """
        self._fk_host_active += 1
        try:
            yield
        finally:
            self._fk_host_active -= 1

    def _fk_ee_device(self, joint_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The device FK: full-model ``eval_fk`` plus a ``body_q`` readback."""
        self._eval_fk(joint_q)
        return self.ee_pose(self._fk_state)

    def _build_chain_fk(self, home_state: newton.State) -> _ChainFK | None:
        """Walk the model's joint arrays from ``ee_body`` to a fixed root.

        Returns the host evaluator, or ``None`` with :attr:`fk_host_reason`
        set when the chain holds a joint kind the walk cannot represent, when
        the root's pose is not a constant of the scene, or when the walk
        disagrees with the device path -- checked at the home posture and two
        perturbed configurations, because an FK that is almost right is worse
        than a slow one.
        """
        model = self.model
        joint_type = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_x_p = model.joint_X_p.numpy()
        joint_x_c = model.joint_X_c.numpy()
        joint_axis = model.joint_axis.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()

        supported = {int(newton.JointType.FIXED), int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC)}
        child_to_joint = {int(joint_child[j]): j for j in range(model.joint_count)}
        chain: list[int] = []
        body = self.ee_body
        root_body = -1
        while body in child_to_joint:
            j = child_to_joint[body]
            if int(joint_type[j]) not in supported:
                self.fk_host_reason = f"chain joint {j} has type {int(joint_type[j])}"
                return None
            chain.append(j)
            body = int(joint_parent[j])
            root_body = body
            if body < 0:
                break
        chain.reverse()
        if not chain:
            self.fk_host_reason = "ee_body hangs from no joint"
            return None
        joints = [
            {
                "type": int(joint_type[j]),
                "x_pj": transform_matrix(joint_x_p[j]),
                "x_cj_inv": np.linalg.inv(transform_matrix(joint_x_c[j])),
                "axis": np.asarray(joint_axis[int(joint_qd_start[j])], dtype=np.float64),
                "q_start": int(joint_q_start[j]),
            }
            for j in chain
        ]
        # A root that is itself jointed never gets here (the walk follows
        # joints), so a non-world root is a jointless body: its pose is set
        # once by the compiler and constant thereafter -- readable from any
        # state, including the home state this constructor already holds.
        if root_body >= 0:
            x_root = transform_matrix(home_state.body_q.numpy()[root_body])
        else:
            x_root = np.eye(4)
        walk = _ChainFK(joints, x_root, self.ee_offset, self.ee_offset_rot)

        q_home = np.asarray(home_state.joint_q.numpy(), dtype=np.float64)
        rng = np.random.default_rng(0)
        for probe in range(3):
            q = q_home.copy()
            if probe:
                for c in self.arm_coords:
                    q[c] += rng.uniform(-1.0, 1.0)
            pos_dev, quat_dev = self._fk_ee_device(q.astype(np.float32))
            pos_host, quat_host = walk.ee(q)
            pos_err = float(np.linalg.norm(pos_dev - pos_host))
            align = abs(float(np.dot(quat_dev, quat_host))) / max(
                float(np.linalg.norm(quat_dev) * np.linalg.norm(quat_host)), 1e-12
            )
            rot_err = 2.0 * np.arccos(min(1.0, align))
            if pos_err > 1.0e-4 or rot_err > 1.0e-3:
                self.fk_host_reason = f"walk disagrees with device FK ({pos_err * 1e3:.3f} mm, {rot_err:.5f} rad)"
                log.warning("host FK fell back: %s", self.fk_host_reason)
                return None
        return walk

    def fk_bodies(self, joint_q: np.ndarray) -> np.ndarray:
        """Every body's world transform for a joint vector (pure FK).

        The same evaluation :meth:`fk_ee` runs, returning the whole
        ``body_q`` it fills rather than one link of it -- which is what lets a
        caller draw a *preview of the hand* at a candidate configuration
        (fingers, palm and all) instead of a bare frame at the tool point.

        Returns:
            ``[body_count, 7]`` of ``(x, y, z, qx, qy, qz, qw)``.
        """
        self._eval_fk(joint_q)
        return self._fk_state.body_q.numpy()

    def _eval_fk(self, joint_q: np.ndarray) -> None:
        """Fill the scratch state with the FK of ``joint_q``."""
        if self._fk_state is None:
            self._fk_state = self.model.state()
        q_wp = wp.array(np.asarray(joint_q, dtype=np.float32), dtype=wp.float32, device=self.model.device)
        newton.eval_fk(self.model, q_wp, self._fk_state.joint_qd, self._fk_state)

    def solve_q(
        self,
        seed_q: np.ndarray,
        position: Sequence[float],
        approach_world: Sequence[float] = (0.0, 0.0, -1.0),
        *,
        iterations: int = 300,
        damping: float = 0.1,
        step: float = 0.5,
        fd_eps: float = 1.0e-4,
        pos_scale: float = 100.0,
    ) -> np.ndarray:
        """Converge a waypoint: reach ``position`` with the tool +z along
        ``approach_world``, wrist roll free. Returns a full ``joint_q``.

        Damped least squares over a position error stacked with the
        axis-alignment error ``cross(axis, target)``, paired with the *angular*
        Jacobian by finite differences over Newton FK. Two things it gets right
        that the LM solver does not: the axis-pair formulation walks straight
        through the 90-degree saddle a folded home posture starts at (a full
        quaternion objective's gradient vanishes there), and clipping to joint
        limits per iteration recovers where the limit objective stalls in a
        corner ~10 cm short.

        ``pos_scale`` is load bearing, not cosmetic. It reproduces the
        conditioning of the reference implementation, which solved in
        *centimetres*: its position error dominates the unit-magnitude axis
        error about 35:1, so the solve fixes position first and aligns the
        wrist in the endgame. Run in metres with equal weights the ordering
        flips -- the first iterations chase the axis, commit the elbow to the
        wrong branch, and jam a wrist joint into its limit 5-7 cm short.

        Chain waypoints by seeding each solve from the previous solution.
        Per-frame tracking stays on :meth:`solve_into`.
        """
        target = np.asarray(position, dtype=np.float64)
        axis_target = np.asarray(approach_world, dtype=np.float64)
        axis_target = axis_target / np.linalg.norm(axis_target)
        q = np.asarray(seed_q, dtype=np.float64).copy()
        coords = self.arm_coords
        solver = self._pyroki_solver()
        if solver is not None:
            solved, _ = solver.solve(q[coords], target, self._quat_with_z(axis_target), roll="free")
            q[coords] = solved[0]
            return q.astype(np.float32)
        lower = self.model.joint_limit_lower.numpy()[self.arm_dofs]
        upper = self.model.joint_limit_upper.numpy()[self.arm_dofs]

        def pose(qv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            pos, quat = self.fk_ee(qv.astype(np.float32))
            return pos, quat_to_mat(quat)

        # Two stages: position only, then position + axis. Acting from the
        # start, the axis pull reliably picks the elbow branch that jams a
        # wrist joint into its limit short of the target. Position-only DLS
        # settles into the branch that can actually reach; turning the axis
        # term on afterwards only polishes the wrist inside that branch.
        for it in range(iterations):
            with_axis = it >= iterations // 3
            pos, rot = pose(q)
            axis_err = np.cross(rot[:, 2], axis_target) if with_axis else np.zeros(3)
            err = np.concatenate([(target - pos) * pos_scale, axis_err])
            if with_axis and np.linalg.norm(target - pos) < 1.5e-3 and np.linalg.norm(axis_err) < 3.0e-2:
                break
            jac = np.zeros((6, len(coords)))
            for k, c in enumerate(coords):
                qp = q.copy()
                qp[c] += fd_eps
                pos_p, rot_p = pose(qp)
                jac[:3, k] = (pos_p - pos) * pos_scale / fd_eps
                if with_axis:
                    w = rot_p @ rot.T  # the small rotation taking rot to rot_p
                    jac[3:, k] = np.array([w[2, 1] - w[1, 2], w[0, 2] - w[2, 0], w[1, 0] - w[0, 1]]) / (2.0 * fd_eps)
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), err)
            q[coords] += step * dq
            q[coords] = np.clip(q[coords], lower, upper)
        return q.astype(np.float32)

    def solve_q_full(
        self,
        seed_q: np.ndarray,
        position: Sequence[float],
        quat_xyzw: Sequence[float],
        *,
        polish_passes: int = 2,
        **solve_q_kwargs,
    ) -> np.ndarray:
        """Converge a waypoint on a **full pose**: position, approach axis,
        and wrist roll. Returns a full ``joint_q``.

        For the tasks an approach axis cannot carry -- a keyed connector, a
        grasp across a cable's tangent -- anywhere a free roll closes the jaws
        across the wrong dimension of the part. It exists beside the LM solver
        because the LM kernels specialize over the model's joint types, and a
        model carrying cable or D6 joints pays a pathological first-ever
        module compile; this path runs entirely on the FK kernels every scene
        has already built.

        Staged, because six-dimensional FD descent from a seed is not worth its
        failure modes (a strong rotation term yanks the solve into another
        elbow basin; a weak one parks position-perfect with the wrist half a
        turn out -- measured as a gripper whose chassis converged below its
        own pads, plowing the workpiece, on a 0.2 mm position score):

        1. :meth:`solve_q` converges position with the tool axis along the
           target frame's +z -- the proven staged solve, everything except
           roll.
        2. The roll is closed **exactly**: it is one coordinate, the last arm
           joint, whose axis is the tool axis. The residual roll error is
           measured about that axis and applied directly, clipped to the
           joint's limits; a short position polish then absorbs what the step
           cost in position.
        3. What stages 1 and 2 leave -- a degree or two of orientation and
           hundredths of a millimetre of position on an off-axis TCP -- is closed by a few
           damped least-squares steps on all six coordinates
           (:func:`~robosimstudio.solvers.ik.fleet_dls.polish_pose_batched`).
           Against stage 2's answer, what it hands back has a position error
           no larger than ``max(stage 2's, 0.1 mm)`` and a weighted 6-D error
           (0.1 m per radian) no larger -- smaller if any step it kept stayed
           within that position ceiling, else it is stage 2's answer
           unchanged. The orientation error alone is not bounded. No joint
           ends more than 0.3 rad from stage 2's answer; and since that answer
           is inside the joint limits whenever stage 2's descent took a step,
           every joint stays inside them and moves at most 0.15 rad per polish
           step. Its damping rises near a singular wrist, so neighbouring
           targets get neighbouring answers there.

        **That polish is not always cosmetic, and the cost is not always
        millimetres.** This used to say the roll turns "while moving the TCP
        nowhere", which holds only where the tool-centre point sits *on* the
        wrist axis. It does on the Panda, whose ``ee_offset`` is the zero
        vector. It does not on this stack's other arms: the YAM's TCP is 44 mm
        off-axis (``ee_offset = (0, -0.044, 0.130)``), so a roll correction
        sweeps the TCP around a 44 mm circle -- measured at 22.78 mm for 30
        degrees and **62.23 mm for 90** (44*sqrt(2)). ``ik_backend.py``'s offset
        table has the rest: 134.7 mm on yam_i2rt, 255.3 mm on droid, so the
        larger the hand, the worse this is.

        The polish is hard-capped at ``min(60, iterations)``, so on an off-axis
        hand a large roll correction can leave more position error behind than
        the polish can recover. Measured on the shirt fold: a knot at
        12.71 mm was dragged to 29.45 mm by a 26.58-degree roll correction and
        polished back only to 20.70 mm, failing a 10 mm gate. The fix there was
        not more iterations -- it was choosing a roll the whole leg can hold,
        so the correction is small. Prefer :meth:`solve_q` where roll is
        genuinely free; reach for this one when roll is a real constraint, and
        gate on **both** halves of :meth:`pose_errors` when you do.

        Orientation targets are in the **grasp-site frame** (``ee_offset_rot``
        applied), matching :meth:`ee_pose` and the LM objectives. Gate plans
        on :meth:`pose_errors`, both halves.
        """
        from robosimstudio.core.math import quat_to_mat

        target = np.asarray(position, dtype=np.float64)
        solver = self._pyroki_solver()
        if solver is not None:
            q = np.asarray(seed_q, dtype=np.float64).copy()
            solved, _ = solver.solve(q[self.arm_coords], target, np.asarray(quat_xyzw, dtype=np.float64), roll="held")
            q[self.arm_coords] = solved[0]
            return q.astype(np.float32)
        rot_target = quat_to_mat(np.asarray(quat_xyzw, dtype=np.float64))
        approach = rot_target[:, 2]
        roll_coord = self.arm_coords[-1]
        lower = self.model.joint_limit_lower.numpy()[self.arm_dofs[-1]]
        upper = self.model.joint_limit_upper.numpy()[self.arm_dofs[-1]]

        polish_kwargs = dict(solve_q_kwargs)
        polish_kwargs["iterations"] = min(60, int(polish_kwargs.get("iterations", 60)))
        q = self.solve_q(np.asarray(seed_q, dtype=np.float64), target, approach, **solve_q_kwargs)
        for _ in range(max(1, polish_passes)):
            _, quat = self.fk_ee(np.asarray(q, dtype=np.float32))
            rot = quat_to_mat(quat)
            # Roll error about the tool axis: how far the target's x-axis sits
            # from ours, measured in the plane normal to the approach.
            x_now = rot[:, 0] - np.dot(rot[:, 0], approach) * approach
            x_want = rot_target[:, 0] - np.dot(rot_target[:, 0], approach) * approach
            n_now, n_want = np.linalg.norm(x_now), np.linalg.norm(x_want)
            if n_now < 1.0e-9 or n_want < 1.0e-9:
                break
            x_now, x_want = x_now / n_now, x_want / n_want
            angle = float(np.arctan2(np.dot(np.cross(x_now, x_want), approach), np.dot(x_now, x_want)))
            if abs(angle) < 1.0e-3:
                break
            q = np.asarray(q, dtype=np.float64)
            # The last joint's positive rotation may spin the tool either way;
            # measure its sign once per pass rather than assuming.
            probe = q.copy()
            probe[roll_coord] += 1.0e-3
            _, quat_p = self.fk_ee(np.asarray(probe, dtype=np.float32))
            rot_p = quat_to_mat(quat_p)
            x_p = rot_p[:, 0] - np.dot(rot_p[:, 0], approach) * approach
            x_p = x_p / max(np.linalg.norm(x_p), 1.0e-12)
            sign = np.sign(np.dot(np.cross(x_now, x_p), approach)) or 1.0
            q[roll_coord] = np.clip(q[roll_coord] + sign * angle, lower, upper)
            q = self.solve_q(q, target, approach, **polish_kwargs)
        return np.asarray(self._polish_pose(q, target, rot_target), dtype=np.float32)

    def _polish_pose(self, joint_q: np.ndarray, target: np.ndarray, rot_target: np.ndarray) -> np.ndarray:
        """Stage 3 of :meth:`solve_q_full`: the whole 6-D residual, closed by damped least squares.

        The batched solver's own polish run on one world, so the scalar and the
        wave forms stay one arithmetic -- see
        :func:`~robosimstudio.solvers.ik.fleet_dls.polish_pose_batched` for what
        it fixes and exactly what it guarantees against the ``joint_q`` it is
        handed.
        """
        from robosimstudio.solvers.ik.fleet_dls import polish_pose_batched

        full = np.asarray(joint_q, dtype=np.float64)
        coords = list(self.arm_coords)
        ik = self

        class _OneArm:
            def poses(self, arm_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                positions, quats = [], []
                for row in np.asarray(arm_q, dtype=np.float64):
                    probe = full.copy()
                    probe[coords] = row
                    pos, quat = ik.fk_ee(np.asarray(probe, dtype=np.float32))
                    positions.append(np.asarray(pos, dtype=np.float64))
                    quats.append(np.asarray(quat, dtype=np.float64))
                return np.asarray(positions), np.asarray(quats)

        limits = self.arm_dofs
        polished = polish_pose_batched(
            _OneArm(),
            full[coords][None, :],
            target[None, :],
            rot_target[None, :, :],
            lower=self.model.joint_limit_lower.numpy()[limits].astype(np.float64),
            upper=self.model.joint_limit_upper.numpy()[limits].astype(np.float64),
        )
        out = full.copy()
        out[coords] = polished[0]
        return out

    def pose_errors(
        self, joint_q: np.ndarray, position: Sequence[float], quat_xyzw: Sequence[float]
    ) -> tuple[float, float]:
        """Position [m] and orientation [rad] error of a configuration against
        a full-pose target -- what a plan gate checks, and it must check
        **both**: a solve can score 0.2 mm of position with the wrist half a
        turn out, and on a scene with a virtual grasp that runs to a
        perfect-looking success under a hand that never gripped anything.
        """
        from robosimstudio.core.math import quat_to_mat

        pos, quat = self.fk_ee(np.asarray(joint_q, dtype=np.float32))
        rot = quat_to_mat(quat)
        rot_target = quat_to_mat(np.asarray(quat_xyzw, dtype=np.float64))
        w = rot_target @ rot.T
        angle = float(np.arccos(np.clip((np.trace(w) - 1.0) / 2.0, -1.0, 1.0)))
        return float(np.linalg.norm(pos - np.asarray(position, dtype=np.float64))), angle

    def solve_best(
        self,
        seeds: Sequence[np.ndarray],
        position: Sequence[float],
        approach_world: Sequence[float] = (0.0, 0.0, -1.0),
        **kwargs,
    ) -> tuple[np.ndarray, float]:
        """:meth:`solve_q` from every seed offered, keeping the closest result.

        The DLS lands in whichever basin its seed favours and the basins differ
        by centimetres, so a routine that plans a chain of waypoints wants the
        best of several rather than the first.

        Returns:
            The winning joint vector and its position error [m].
        """
        solutions = [self.solve_q(s, position, approach_world, **kwargs) for s in seeds]
        errors = [float(np.linalg.norm(self.fk_ee(q)[0] - np.asarray(position, dtype=np.float64))) for q in solutions]
        best = int(np.argmin(errors))
        return solutions[best], errors[best]

    def plan_cartesian(
        self,
        seed_q: np.ndarray,
        start: Sequence[float],
        end: Sequence[float],
        approach_world: Sequence[float] = (0.0, 0.0, -1.0),
        *,
        waypoints: int = 40,
        tolerance: float = 0.0,
        **kwargs,
    ) -> list[np.ndarray] | None:
        """A straight line in Cartesian space, solved at each waypoint.

        **Not collision aware.** It goes where it is told, and a caller that
        asks it to move through a bin gets a hand in a bin. That is the honest
        capability, and stating it is better than a name that implies planning.

        The interpolation is in *Cartesian* space and each waypoint is seeded
        from the previous solution, which keeps the chain in one joint-space
        branch -- interpolating between branches is what swings an elbow through
        the table between two perfectly good endpoints.

        Args:
            seed_q: Full joint vector to seed the first waypoint from.
            start: Where the tool point starts [m].
            end: Where it ends [m].
            approach_world: Tool +z at every waypoint. Wrist roll stays free;
                a caller that needs a preview of the roll the solver picked
                should run :meth:`fk_bodies` on the returned configurations.
            waypoints: How many solves the line is cut into.
            tolerance: Position error above which a waypoint counts as out of
                reach and the whole line is abandoned [m]. 0 accepts whatever
                the solver converged to.
            **kwargs: Passed through to :meth:`solve_q`, e.g. ``iterations``.

        Returns:
            One full joint vector per waypoint, or ``None`` if any waypoint was
            further out than ``tolerance``.
        """
        first, last = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
        seed = np.asarray(seed_q, dtype=np.float32)
        steps = max(1, int(waypoints))
        path: list[np.ndarray] = []
        for i in range(1, steps + 1):
            point = first + (last - first) * (i / steps)
            solution, error = self.solve_best([seed], point, approach_world, **kwargs)
            if tolerance > 0.0 and error > tolerance:
                return None
            seed = solution
            path.append(solution)
        return path
