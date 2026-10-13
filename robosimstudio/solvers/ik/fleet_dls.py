# SPDX-License-Identifier: Apache-2.0
"""A wave's boundary IK solves, run as one search instead of W searches.

:meth:`~robosimstudio.solvers.ik.arm_ik.ArmIK.solve_q_full` is what a graph's
pose verbs reach for, and on a wave it is the whole cost of a control step:
measured at twelve worlds, ``solve_q_full`` was 254.8 ms of the 255.7 ms twelve
scripted routines took per step. Everything else -- the physics, the host
island, the routines' own arithmetic -- was noise beside it.

It is expensive for a structural reason rather than a numerical one. The solve
is damped least squares over a **finite-difference** Jacobian, so each iteration
evaluates forward kinematics once for the pose and once per arm coordinate, and
each of those evaluations is a device round trip: a write, ``eval_fk``, and a
read. Eight round trips an iteration, eighty iterations, and then a roll closure
that does it again twice. The device is idle between them.

So this runs the *same search* for every world at once. Each of those FK
evaluations becomes one batched call over the fleet's replicated model
(:class:`~robosimstudio.solvers.ik.fleet_fk.FleetArmFK`), which returns
bit-identical poses -- checked, not assumed, and it has to be: the descent is a
fixed-point search whose basin choice flips on float noise, and
:class:`~robosimstudio.solvers.ik.arm_ik._ChainFK` measured what substituting a
*numerically different* FK does to it. Nothing here changes an operand. The
arithmetic below is :meth:`ArmIK.solve_q`'s, with a leading world axis and the
early exit turned from a ``break`` into a mask.

**The mask is the one thing worth reading carefully.** The scalar solve breaks
out of the loop when its pose is close enough, *before* computing that
iteration's Jacobian -- so its answer is the ``q`` it walked in with. A world
that would have broken therefore has to stop being updated at exactly the same
iteration, which is what ``active`` does. It keeps being evaluated (its column
of the batch is computed and thrown away) because dropping it would change the
shape of the batch and buy nothing: the cost is the round trip, not the rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["polish_pose_batched", "pose_errors_batched", "solve_q_batched", "solve_q_full_batched"]


def _quat_to_mats(quat: np.ndarray) -> np.ndarray:
    """``[n, 4]`` xyzw to ``[n, 3, 3]``: :func:`robosimstudio.core.math.quat_to_mat`, per row.

    Including its normalisation, which is not a detail. That function divides by
    the quaternion's own norm because "a quaternion that has drifted off the
    unit sphere (which read-back float32 poses routinely have) still yields an
    orthogonal matrix rather than one that quietly scales" -- and these
    quaternions come straight off a float32 ``body_q`` read, so they have. A
    batched form that assumed unit length differed from the scalar one in the
    fifth decimal of the first iteration, which in a fixed-point search is not
    a rounding difference but a different search.
    """
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    n = x * x + y * y + z * z + w * w
    s = np.where(n < 1.0e-12, 0.0, 2.0 / np.where(n < 1.0e-12, 1.0, n))
    out = np.empty((len(quat), 3, 3), dtype=np.float64)
    out[:, 0, 0] = 1.0 - s * (y * y + z * z)
    out[:, 0, 1] = s * (x * y - z * w)
    out[:, 0, 2] = s * (x * z + y * w)
    out[:, 1, 0] = s * (x * y + z * w)
    out[:, 1, 1] = 1.0 - s * (x * x + z * z)
    out[:, 1, 2] = s * (y * z - x * w)
    out[:, 2, 0] = s * (x * z - y * w)
    out[:, 2, 1] = s * (y * z + x * w)
    out[:, 2, 2] = 1.0 - s * (x * x + y * y)
    # A quaternion at the origin is the identity, as the scalar form returns.
    return np.where((n < 1.0e-12)[:, None, None], np.eye(3), out)


def solve_q_batched(
    fk: Any,
    seeds: np.ndarray,
    targets: np.ndarray,
    approaches: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    active: np.ndarray | None = None,
    iterations: int = 300,
    damping: float = 0.1,
    step: float = 0.5,
    fd_eps: float = 1.0e-4,
    pos_scale: float = 100.0,
) -> np.ndarray:
    """:meth:`ArmIK.solve_q` for every world at once. ``[W, n]`` in, ``[W, n]`` out.

    Args:
        fk: A :class:`~.fleet_fk.FleetArmFK`, already seeded with the pose the
            non-arm joints hold.
        seeds: Each world's starting arm coordinates ``[W, n]``.
        targets: Each world's tool-centre target ``[W, 3]`` [m].
        approaches: Each world's tool +z target ``[W, 3]``, normalised here.
        lower, upper: The arm's joint limits ``[n]``, the template's.
        active: Which worlds are actually asking ``[W]``. A world that is not
            comes back exactly as it went in -- this is a *wave*, and the worlds
            that did not ask for a solve are the common case.

    The keyword defaults are :meth:`ArmIK.solve_q`'s, including ``pos_scale``,
    which that method's docstring explains is load-bearing rather than cosmetic.
    """
    q = np.asarray(seeds, dtype=np.float64).copy()
    n_worlds, n_coords = q.shape
    target = np.asarray(targets, dtype=np.float64).reshape(n_worlds, 3)
    axis_target = np.asarray(approaches, dtype=np.float64).reshape(n_worlds, 3)
    axis_target = axis_target / np.maximum(np.linalg.norm(axis_target, axis=1, keepdims=True), 1.0e-12)
    live = np.ones(n_worlds, dtype=bool) if active is None else np.asarray(active, dtype=bool).copy()
    eye6 = np.eye(6)

    for it in range(int(iterations)):
        if not live.any():
            break
        with_axis = it >= int(iterations) // 3
        pos, quat = fk.poses(q)
        rot = _quat_to_mats(quat)

        gap = target - pos
        if with_axis:
            axis_err = np.cross(rot[:, :, 2], axis_target)
            # The scalar form breaks here, before its Jacobian: a world whose
            # pose is already close enough answers with the q it walked in with.
            close = (np.linalg.norm(gap, axis=1) < 1.5e-3) & (np.linalg.norm(axis_err, axis=1) < 3.0e-2)
            live &= ~close
            if not live.any():
                break
        else:
            axis_err = np.zeros((n_worlds, 3))
        err = np.concatenate([gap * pos_scale, axis_err], axis=1)

        jac = np.zeros((n_worlds, 6, n_coords))
        for k in range(n_coords):
            probe = q.copy()
            probe[:, k] += fd_eps
            pos_p, quat_p = fk.poses(probe)
            jac[:, :3, k] = (pos_p - pos) * pos_scale / fd_eps
            if with_axis:
                # The small rotation taking rot to rot_p, as a vector.
                turn = _quat_to_mats(quat_p) @ np.transpose(rot, (0, 2, 1))
                jac[:, 3, k] = (turn[:, 2, 1] - turn[:, 1, 2]) / (2.0 * fd_eps)
                jac[:, 4, k] = (turn[:, 0, 2] - turn[:, 2, 0]) / (2.0 * fd_eps)
                jac[:, 5, k] = (turn[:, 1, 0] - turn[:, 0, 1]) / (2.0 * fd_eps)

        gram = jac @ np.transpose(jac, (0, 2, 1)) + damping**2 * eye6
        # ``err[..., None]`` because numpy reads a 2-D right-hand side as one
        # matrix of columns, not a stack of vectors -- the stacked solve wants
        # ``[W, 6, 1]`` and gives back the same.
        dq = np.einsum("wkj,wk->wj", jac, np.linalg.solve(gram, err[:, :, None])[:, :, 0])
        moved = np.clip(q + step * dq, lower, upper)
        q = np.where(live[:, None], moved, q)
    return q


def solve_q_full_batched(
    fk: Any,
    seeds: np.ndarray,
    targets: np.ndarray,
    quats_xyzw: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    active: np.ndarray | None = None,
    polish_passes: int = 2,
    **solve_kwargs: Any,
) -> np.ndarray:
    """:meth:`ArmIK.solve_q_full` for every world at once.

    The staged solve, then the roll closed exactly and a position polish, twice
    -- and each of those three things masked per world, because each of the
    scalar form's three early exits is a world's own.
    """
    q = np.asarray(seeds, dtype=np.float64).copy()
    n_worlds = len(q)
    rot_target = _quat_to_mats(np.asarray(quats_xyzw, dtype=np.float64).reshape(n_worlds, 4))
    approach = rot_target[:, :, 2]
    live = np.ones(n_worlds, dtype=bool) if active is None else np.asarray(active, dtype=bool).copy()

    q = solve_q_batched(fk, q, targets, approach, lower=lower, upper=upper, active=live, **solve_kwargs)

    polish_kwargs = dict(solve_kwargs)
    polish_kwargs["iterations"] = min(60, int(polish_kwargs.get("iterations", 60)))
    turning = live.copy()
    for _ in range(max(1, int(polish_passes))):
        if not turning.any():
            break
        _pos, quat = fk.poses(q)
        rot = _quat_to_mats(quat)

        # The roll error about the tool axis, in the plane normal to approach.
        x_now = rot[:, :, 0] - np.einsum("wi,wi->w", rot[:, :, 0], approach)[:, None] * approach
        x_want = rot_target[:, :, 0] - np.einsum("wi,wi->w", rot_target[:, :, 0], approach)[:, None] * approach
        n_now = np.linalg.norm(x_now, axis=1)
        n_want = np.linalg.norm(x_want, axis=1)
        turning &= (n_now >= 1.0e-9) & (n_want >= 1.0e-9)
        if not turning.any():
            break
        x_now = x_now / np.maximum(n_now, 1.0e-12)[:, None]
        x_want = x_want / np.maximum(n_want, 1.0e-12)[:, None]
        angle = np.arctan2(
            np.einsum("wi,wi->w", np.cross(x_now, x_want), approach),
            np.einsum("wi,wi->w", x_now, x_want),
        )
        turning &= np.abs(angle) >= 1.0e-3
        if not turning.any():
            break

        # Which way the last joint spins the tool, measured rather than assumed.
        probe = q.copy()
        probe[:, -1] += 1.0e-3
        _pp, quat_p = fk.poses(probe)
        rot_p = _quat_to_mats(quat_p)
        x_p = rot_p[:, :, 0] - np.einsum("wi,wi->w", rot_p[:, :, 0], approach)[:, None] * approach
        x_p = x_p / np.maximum(np.linalg.norm(x_p, axis=1), 1.0e-12)[:, None]
        sign = np.sign(np.einsum("wi,wi->w", np.cross(x_now, x_p), approach))
        sign[sign == 0.0] = 1.0

        rolled = q.copy()
        rolled[:, -1] = np.clip(q[:, -1] + sign * angle, lower[-1], upper[-1])
        q = np.where(turning[:, None], rolled, q)
        q = solve_q_batched(fk, q, targets, approach, lower=lower, upper=upper, active=turning, **polish_kwargs)
    return polish_pose_batched(fk, q, targets, rot_target, lower=lower, upper=upper, active=live)


def _rotvecs(turn: np.ndarray) -> np.ndarray:
    """``[n, 3, 3]`` rotations to ``[n, 3]`` rotation vectors (the log map), per row."""
    trace = np.trace(turn, axis1=1, axis2=2)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    skew = np.stack(
        [turn[:, 2, 1] - turn[:, 1, 2], turn[:, 0, 2] - turn[:, 2, 0], turn[:, 1, 0] - turn[:, 0, 1]], axis=1
    )
    sin = np.sin(angle)
    # Small angles: the skew part over two is the vector itself to second order.
    scale = np.where(sin > 1.0e-6, angle / np.where(sin > 1.0e-6, 2.0 * sin, 1.0), 0.5)
    return skew * scale[:, None]


POLISH_STEPS = 8
"""Damped Gauss-Newton steps the full-pose polish may take, rejected ones included.
Two or three close a 1-2 degree residual; the rest is room for a step that has
to be retried stiffer."""

POLISH_ROT_WEIGHT = 0.1
"""Metres per radian in the polish's error: 0.1 m of position weighs what one
radian of orientation does."""

POLISH_MAX_STEP = 0.15
"""Radians any one joint may move in one polish step; a longer step is scaled
down whole, keeping its direction. A bound per step, whatever the arm. On the
YAM (2026-10-04) a cap of 0.1-0.3 left forty reachable targets at the uncapped
residuals, 0.0132 mm / 0.0018 degrees at worst, and 0.05 slowed them to 0.0191
mm / 0.0030 degrees."""

POLISH_MAX_MOVE = 0.3
"""Radians any one joint may move over the whole polish, from the pose it was
handed. Near a singular wrist (:data:`POLISH_SINGULAR_SIGMA`) the polish leaves
a residual it cannot close without a wrist swing, and a hand tracked through
there carries it; once out, an unbounded polish took a 1.6 degree residual down
to 0.1-0.2 degrees in one call with a 0.45-0.47 rad move. On the YAM
(2026-10-04), two of 26 such tracking chains jumped 0.48 and 0.50 rad between
targets 0.05 degrees apart unbounded, 0.038 and 0.035 rad bounded at 0.3. That
bound changed 1 of 400 answers on the static sets and none of their closures;
0.15 and 0.2 each left one of 24 reachable targets unclosed."""

POLISH_POS_SLACK = 1.0e-4
"""Metres of position error the polish may hand back even when it was handed
less: what closing an orientation is allowed to cost in position."""

POLISH_SINGULAR_SIGMA = 5.0e-3
"""The weighted Jacobian's smallest singular value below which the damping
rises. It is not a margin kept around the singular wrist alone: ordinary YAM
configurations sit just above it, and the raised damping already acts on some.

Measured on the YAM at the polish's entry (2026-10-04). Near the wrist: 1.6e-5
to 2.6e-4 with q5 within 0.002 rad of +-pi/2, 4.6e-3 (median) at 0.08 rad,
1.1e-2 (median; least 1.7e-3) at 0.2 rad. Elsewhere: 6.5e-3 at the least over forty reachable FK
targets on ``plate_rack/put_away``, but 1.95e-3 on one of the 24 targets of
``test_arm_ik_full_pose``'s recipe moved to the YAM (two of the 24 are under
it, raising the floor to 9.3e-3 and 2.3e-3; both are among the three that
neither this polish nor the fixed-floor one closes). Forty configurations
within 0.5 rad of home sit at 5.6e-3 at the least and 6.1e-3 at the tenth
percentile, on ``plate_rack/put_away``, ``surface_prep/wipe`` and
``crate_wash/flip`` alike; on the last two, 2 of 40 FK targets each entered
the polish at 4.3e-3 to 4.4e-3, got a floor of about 5e-3, and closed to
within 0.001 mm and 0.0002 degrees (the fixed floor closed them too)."""

POLISH_SINGULAR_DAMPING = 1.0e-2
"""The damping a step gets at an exact singularity, rising smoothly to it from
the floor as the smallest singular value falls from
:data:`POLISH_SINGULAR_SIGMA` to zero."""


def polish_pose_batched(
    fk: Any,
    q: np.ndarray,
    targets: np.ndarray,
    rot_targets: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    active: np.ndarray | None = None,
    steps: int = POLISH_STEPS,
    fd_eps: float = 1.0e-4,
    damping: float = 1.0e-3,
    max_step: float = POLISH_MAX_STEP,
    max_move: float = POLISH_MAX_MOVE,
    pos_slack: float = POLISH_POS_SLACK,
    singular_sigma: float = POLISH_SINGULAR_SIGMA,
    singular_damping: float = POLISH_SINGULAR_DAMPING,
) -> np.ndarray:
    """Close the whole six-dimensional pose error left by the staged full-pose solve.

    The staged solve converges position and the tool axis by descent, closes
    the roll on the last joint, and polishes position again -- and on an arm
    whose tool-centre point sits off its wrist axis it parks a degree or two
    off the target orientation however many iterations it is given (measured
    1-1.7 degrees on the YAM over 40-300 iterations, 2026-09-29), with up
    to a few hundredths of a millimetre of position (0.034 mm at most over
    forty reachable YAM targets, 2026-10-04). A residual that moves from step to step
    makes a hand driven through it every control step twitch: it shook a plate
    held by its rim out of the pads, and it is why two twins patched it locally.

    A few damped least-squares steps on all six coordinates at once, over a
    finite-difference Jacobian of the same FK, weighting
    :data:`POLISH_ROT_WEIGHT` metres per radian. A step is kept only if it
    lowers that weighted error; one that does not is retried, on the Jacobian
    already in hand, with ten times the damping it was actually taken with.
    **What comes back, per world, against the pose it was handed:**

    * its position error is at most ``max(handed, pos_slack)``, the ceiling.
      The answer is the last kept step within it, else the pose handed in, so
      an orientation the arm cannot reach -- a roll past the last joint's
      limit -- is not bought with position. Measured on the YAM (2026-10-04),
      a roll 20 degrees past joint 6 used to cost up to 3.5 mm; now at most
      0.1 mm;
    * its weighted error is no larger, and smaller if any kept step stayed
      within the ceiling. Kept steps that all went above it return the pose
      handed in, unchanged, however much they lowered the weighted error;
    * no joint ends more than ``max_move`` from where it was handed;
    * if the pose handed in is inside ``lower``/``upper``, every joint stays
      inside them and no joint moves more than ``max_step`` in any one step.
      Stage 2's answer is inside them whenever its descent took a step, since
      :meth:`ArmIK.solve_q` clips each one. A pose handed in outside them is
      not snapped in on its own: with no step kept it comes back as it went
      in, and a kept step clips each joint into the limits only as far as
      ``max_move`` allows.

    The orientation error alone is **not** bounded: a step that closes more
    weighted position than it opens in orientation is kept.

    Near a singular wrist the damping rises with the smallest singular value
    (``singular_sigma``, ``singular_damping``). At a fixed floor the polish
    moved the YAM's wrist up to 0.59 rad near q5 = +-pi/2 for a sub-degree
    residual it could not close, so targets 0.01 degrees apart got answers up
    to 0.54 rad apart; with it, at most 0.024 rad over the same 240
    configurations, and forty reachable targets away from the singularity close
    exactly as they did. The price: within 0.08 rad of q5 = +-pi/2 it closes
    52 of 182 one-degree targets to 0.1 mm / 0.05 degrees, where the fixed
    floor closed 79 by swinging the wrist.

    Per world, masked like the rest of this module: a world that is done, or
    did not ask, comes back bit-identical, and every world's answer is the one
    it gets alone.
    """
    q = np.asarray(q, dtype=np.float64).copy()
    n_worlds, n_coords = q.shape
    target = np.asarray(targets, dtype=np.float64).reshape(n_worlds, 3)
    rot_target = np.asarray(rot_targets, dtype=np.float64).reshape(n_worlds, 3, 3)
    live = np.ones(n_worlds, dtype=bool) if active is None else np.asarray(active, dtype=bool).copy()
    weight = np.array([1.0, 1.0, 1.0, POLISH_ROT_WEIGHT, POLISH_ROT_WEIGHT, POLISH_ROT_WEIGHT])
    eye6 = np.eye(6)

    def weighted_error(pos: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rot = _quat_to_mats(quat)
        gap = target - pos
        turn = _rotvecs(rot_target @ np.transpose(rot, (0, 2, 1)))
        return np.concatenate([gap, turn], axis=1) * weight, rot

    pos, quat = fk.poses(q)
    err, rot = weighted_error(pos, quat)
    # The answer: the last kept step whose position error is within the
    # ceiling. Every kept step lowers the weighted error, so it is also the
    # best such step; the path itself may pass above the ceiling, because
    # closing a reachable orientation routinely does for a step or two.
    answer = q.copy()
    near_lo, near_hi = q - float(max_move), q + float(max_move)
    ceiling = np.maximum(np.linalg.norm(err[:, :3], axis=1), float(pos_slack))
    # Levenberg-Marquardt on the damping, per world: a step that does not help
    # is retried stiffer rather than ending the polish, and one that does
    # loosens it again. A world that cannot be helped runs out of steps with
    # the pose it came in with.
    lam = np.full(n_worlds, float(damping))
    # A rejected step leaves q where it was, so its Jacobian, and the damping
    # floor read off it, are still the ones in hand: only a world that moved
    # pays the n_coords probes again.
    jac = np.zeros((n_worlds, 6, n_coords))
    floor = np.full(n_worlds, float(damping))
    stale = np.ones(n_worlds, dtype=bool)
    for _ in range(int(steps)):
        done = (np.linalg.norm(err[:, :3], axis=1) < 2.0e-5) & (
            np.linalg.norm(err[:, 3:], axis=1) < 2.0e-4 * POLISH_ROT_WEIGHT
        )
        live &= ~done
        if not live.any():
            break
        if (live & stale).any():
            fresh = np.zeros((n_worlds, 6, n_coords))
            for k in range(n_coords):
                probe = q.copy()
                probe[:, k] += fd_eps
                pos_p, quat_p = fk.poses(probe)
                fresh[:, :3, k] = (pos_p - pos) / fd_eps
                fresh[:, 3:, k] = _rotvecs(_quat_to_mats(quat_p) @ np.transpose(rot, (0, 2, 1))) / fd_eps
            fresh = fresh * weight[None, :, None]
            jac = np.where(stale[:, None, None], fresh, jac)
            # Each world's smallest singular value, read only where it is
            # asked for and finite: a stacked SVD raises on one bad matrix.
            near = np.zeros(n_worlds)
            readable = stale & live & np.isfinite(fresh).all(axis=(1, 2))
            if readable.any() and singular_sigma > 0.0:
                sigma = np.linalg.svd(fresh[readable], compute_uv=False)[:, -1]
                near[readable] = np.clip(1.0 - (sigma / float(singular_sigma)) ** 2, 0.0, 1.0)
            floor = np.where(stale, np.sqrt(float(damping) ** 2 + float(singular_damping) ** 2 * near), floor)
            stale[:] = False
        lam_step = np.maximum(lam, floor)
        gram = jac @ np.transpose(jac, (0, 2, 1)) + (lam_step**2)[:, None, None] * eye6
        dq = np.einsum("wkj,wk->wj", jac, np.linalg.solve(gram, err[:, :, None])[:, :, 0])
        longest = np.abs(dq).max(axis=1)
        dq = dq * np.minimum(1.0, float(max_step) / np.maximum(longest, 1.0e-300))[:, None]
        moved = np.clip(np.clip(q + dq, lower, upper), near_lo, near_hi)
        pos_m, quat_m = fk.poses(moved)
        err_m, rot_m = weighted_error(pos_m, quat_m)
        better = live & (np.linalg.norm(err_m, axis=1) < np.linalg.norm(err, axis=1))
        q = np.where(better[:, None], moved, q)
        pos = np.where(better[:, None], pos_m, pos)
        err = np.where(better[:, None], err_m, err)
        rot = np.where(better[:, None, None], rot_m, rot)
        # A rejected step is retried ten times stiffer than the damping it
        # actually had: near a singularity that is the floor, not ``lam``, and
        # ``lam * 10`` alone can stay under the floor and retry the same step.
        lam = np.where(better, np.maximum(lam / 10.0, float(damping)), lam_step * 10.0)
        stale = better
        within = better & (np.linalg.norm(err_m[:, :3], axis=1) <= ceiling)
        answer = np.where(within[:, None], moved, answer)
    return answer


def pose_errors_batched(
    fk: Any, q: np.ndarray, targets: np.ndarray, quats_xyzw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """:meth:`ArmIK.pose_errors` per world: position [m] and orientation [rad]."""
    q = np.asarray(q, dtype=np.float64)
    pos, quat = fk.poses(q)
    rot = _quat_to_mats(quat)
    rot_target = _quat_to_mats(np.asarray(quats_xyzw, dtype=np.float64).reshape(len(q), 4))
    turn = rot_target @ np.transpose(rot, (0, 2, 1))
    trace = np.trace(turn, axis1=1, axis2=2)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return np.linalg.norm(pos - np.asarray(targets, dtype=np.float64).reshape(len(q), 3), axis=1), angle
