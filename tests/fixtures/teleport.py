# SPDX-License-Identifier: Apache-2.0
"""Putting a built scene into a chosen state, without stepping it.

The single most useful thing in the GPU test suite, and it fits in a dozen lines.
Writing an object's free-joint coordinates directly and running forward
kinematics is what makes a test a test of the **evaluator** rather than of the
controller: the question "does this goal fire when the world is in this state"
gets asked directly, instead of being asked through a policy that may or may not
have cooperated.

It is also the mechanism behind the gauntlet's oracle-reachability stage. A goal
that no reachable state satisfies is a task bug, and teleporting into a
satisfying state finds it without a policy existing at all.

A script under ``scripts/`` that needs the same thing keeps its own copy
deliberately: ``scripts/`` is not a package and must not depend on ``tests/``, and
inverting that to put a test helper in the shipped toolkit would be worse than
two short functions that do the same obvious thing.

**Deformables are harder than rigids, and honestly so.** Writing ``particle_q``
directly can produce a self-intersecting configuration that explodes on the first
step, so :func:`teleport_particles` is paired with :func:`settle_and_check` --
which separates "the goal is unreachable" from "the teleport was unphysical".
Those are different failures and a stage that reports them identically is a stage
nobody can act on.
"""

from __future__ import annotations

import numpy as np

__all__ = ["settle_and_check", "teleport_body", "teleport_particles"]


def teleport_body(env, name: str, position) -> None:
    """Put one object where we say, without stepping anything.

    Writes only that object's free-joint coordinates, which is what makes this a
    test of the *evaluator* rather than of the controller. Both pipeline states
    are written, because a reset may hand either one to the next step and a scene
    whose two states disagree is a bug that surfaces much later and elsewhere.
    """
    import newton

    handle = env.handles.objects[name]
    for state in (env.pipeline.state_0, env.pipeline.state_1):
        q = state.joint_q.numpy()
        q[handle.coord_start : handle.coord_start + 3] = position
        state.joint_q.assign(q)
    newton.eval_fk(env.model, env.pipeline.state_0.joint_q, env.pipeline.state_0.joint_qd, env.pipeline.state_0)


def teleport_particles(env, cloth: str, positions: np.ndarray) -> None:
    """Write a deformable's particles directly, without stepping.

    ``positions`` is ``[n, 3]`` in world coordinates for this cloth's own
    particles, in the order :attr:`EntityCloth.particles` gives them.

    Both states are written for the same reason as the rigid case. Velocities are
    zeroed: a teleported sheet that kept its old momentum is not the state anyone
    meant to describe, and the resulting motion looks like a physics bug.
    """
    entity = env.scene.by_name[cloth]
    indices = np.asarray(entity.particles, dtype=np.int64)
    if len(indices) != len(positions):
        raise ValueError(f"cloth {cloth!r} has {len(indices)} particles; got {len(positions)} positions")
    for state in (env.pipeline.state_0, env.pipeline.state_1):
        q = state.particle_q.numpy()
        q[indices] = np.asarray(positions, dtype=q.dtype)
        state.particle_q.assign(q)
        qd = state.particle_qd.numpy()
        qd[indices] = 0.0
        state.particle_qd.assign(qd)


def settle_and_check(env, seconds: float = 1.0) -> dict[str, float]:
    """Step a teleported scene for a while, and report whether it stayed sane.

    A teleport can place matter in a configuration physics refuses -- layers
    interpenetrating, a sheet through a table -- and the resulting explosion is
    *not* evidence about the goal. So this reports both the settling and its
    health, and a caller asserts on them separately:

    Returns:
        ``{"steps", "max_speed", "finite"}`` -- how long it ran, the fastest
        anything was still moving at the end, and whether the state is free of
        NaNs. ``finite`` false means the teleport was unphysical and nothing
        about the goal has been learned.
    """
    steps = max(1, int(seconds * env.pipeline.control_hz))
    for _ in range(steps):
        env.step(env.idle_action())

    state = env.pipeline.state_0
    speeds: list[float] = []
    finite = True
    for array in (state.body_qd, state.particle_qd):
        if array is None:
            continue
        values = np.asarray(array.numpy(), dtype=np.float64)
        if not len(values):
            continue
        finite = finite and bool(np.all(np.isfinite(values)))
        speeds.append(float(np.max(np.linalg.norm(values[:, :3], axis=1))) if values.ndim > 1 else 0.0)
    return {"steps": float(steps), "max_speed": max(speeds) if speeds else 0.0, "finite": finite}
