# SPDX-License-Identifier: Apache-2.0
"""Forward kinematics for one arm, in every world at once.

:meth:`~robosimstudio.solvers.ik.arm_ik.ArmIK.fk_ee` answers for one joint
vector, and the way it answers is a device round trip: a host-to-device write of
the joint coordinates, ``newton.eval_fk``, and a device-to-host read of the
whole ``body_q``. :class:`_ChainFK`'s docstring measures that round trip at
2.12 ms against 4 us of actual kernel, which is invisible per call and ruinous
where the calls actually are -- :meth:`ArmIK.solve_q` builds its Jacobian by
finite differences, one FK for the pose plus one per arm coordinate, every
iteration, for up to eighty iterations.

A wave makes that worse in the one way that is fixable. Twelve worlds solving
the same waypoint is twelve times the same seven-or-eight round trips per
iteration, run one after another on the fleet's thread, while the device sits
idle between them. The arithmetic is not the problem and neither is the FK: the
*count* is.

So this evaluates one perturbation for **every world at once**. The model it
walks is the fleet's own replicated one, whose worlds are independent
articulations, so a batched ``eval_fk`` does per world exactly what the
single-world call does -- the same kernel over the same joint transforms -- and
the poses come back bit-identical. That matters more here than anywhere else in
the fleet: the FD descent is a fixed-point search, and :class:`_ChainFK`
measured what happens when a *numerically different* FK is substituted into it
(the wrap's retained turns scattering to 0.18-0.96 against a stock band of
0.55-0.96). Batching is allowed to make the same search cheaper. It is not
allowed to make it a different search.

What this is not: a faster FK. One call here costs more than one call there --
it walks W worlds of the whole scene rather than one arm. It is fewer calls.
"""

from __future__ import annotations

from typing import Any

import newton
import numpy as np
import warp as wp

__all__ = ["FleetArmFK"]


class FleetArmFK:
    """One arm's tool pose in every world, from one ``eval_fk``.

    Args:
        model: The fleet's **stepping** model -- the replicated one whose worlds
            are the wave's. Not the template: a world past the first has no rows
            in the template, and asking for them is the index error this whole
            change is written around.
        worlds: The :class:`~robosimstudio.core.worlds.WorldMap` for that model.
        arm_coords: The arm's ``joint_q`` indices **in the template**, which is
            what an embodiment's arm index carries.
        ee_body: The arm's tool body index in the template.
        ee_offset: Tool-centre point in ``ee_body``'s frame [m].
        ee_offset_rot: Rotation from ``ee_body``'s frame to the tool frame, xyzw.
    """

    def __init__(
        self,
        model: Any,
        worlds: Any,
        *,
        arm_coords: Any,
        ee_body: int,
        ee_offset: Any = (0.0, 0.0, 0.0),
        ee_offset_rot: Any = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        self.model = model
        self.worlds = worlds
        self.num_worlds = int(worlds.num_worlds)
        self.arm_coords = [int(c) for c in arm_coords]
        self.ee_offset = np.asarray(ee_offset, dtype=np.float64)
        self.ee_offset_rot = np.asarray(ee_offset_rot, dtype=np.float64)
        self._state = None
        # Where each world's arm coordinates and tool body live in the fleet's
        # arrays. Resolved once: the strides do not change for a built model,
        # and doing it per iteration would put a Python loop back inside the
        # thing this class exists to take one out of.
        self._coord_rows = np.array(
            [[worlds.coord(c, w) for c in self.arm_coords] for w in range(self.num_worlds)],
            dtype=np.intp,
        )
        self._ee_rows = np.array([worlds.body(int(ee_body), w) for w in range(self.num_worlds)], dtype=np.intp)
        self._q_host = np.zeros(int(model.joint_coord_count), dtype=np.float32)

    def seed_from(self, joint_q: np.ndarray) -> None:
        """Hold ``joint_q`` as the pose every world's non-arm joints keep.

        The arm coordinates are overwritten by every :meth:`poses` call; the
        rest of the scene is not, and it has to be *something*. This is the
        live state, so a solve sees the bench where the bench is.
        """
        self._q_host[:] = np.asarray(joint_q, dtype=np.float32).reshape(-1)

    def seed_world(self, world: int, template_q: np.ndarray) -> None:
        """Give one world its own pose for everything but the arm.

        :meth:`seed_from` gives every world the live scene, which is what a
        caller solving against the world as it stands wants. A caller solving
        from a seed it has been *carrying* -- a routine chaining waypoints, say
        -- wants that seed instead, and the two differ in more than the arm: a
        stored seed holds whichever coordinates the other arm had when it was
        taken.

        Those coordinates cannot move this arm's tool. Asserting that in general
        would mean knowing which joints lie between this arm and the root, so
        this sidesteps the question: the world is walked with exactly the pose
        the caller asked about, and nothing has to be argued.
        """
        values = np.asarray(template_q, dtype=np.float32).reshape(-1)
        width = len(self._q_host) // max(1, self.num_worlds)
        if len(values) != width:
            raise ValueError(f"a world's pose is {width} coordinates, got {len(values)}")
        start = int(world) * width
        self._q_host[start : start + width] = values

    def live_arm_coords(self) -> np.ndarray:
        """Each world's arm coordinates as the last :meth:`seed_from` left them.

        What a world that is *not* asking for a solve should hold: a batched
        search has a row for every world whether or not anybody wanted it, and
        the rows nobody wanted are better filled with where the arm actually is
        than with zeros, which would be a configuration the limits do not allow.
        """
        return np.asarray(self._q_host[self._coord_rows], dtype=np.float64)

    def poses(self, arm_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tool pose per world for ``arm_q`` ``[worlds, len(arm_coords)]``.

        Returns:
            ``(positions [worlds, 3], quats_xyzw [worlds, 4])``, float64, the
            same two things :meth:`ArmIK.ee_pose` returns for one world.
        """
        arm_q = np.asarray(arm_q, dtype=np.float32).reshape(self.num_worlds, len(self.arm_coords))
        self._q_host[self._coord_rows] = arm_q

        if self._state is None:
            self._state = self.model.state()
        q_wp = wp.array(self._q_host, dtype=wp.float32, device=self.model.device)
        newton.eval_fk(self.model, q_wp, self._state.joint_qd, self._state)
        # One read of the whole array, then slice on the host -- a device slice
        # transfers the whole thing anyway, and W reads of it would be W copies
        # of the same bytes. The rule the rest of the fleet follows.
        body_q = self._state.body_q.numpy()[self._ee_rows]

        pos = np.asarray(body_q[:, :3], dtype=np.float64)
        quat = np.asarray(body_q[:, 3:7], dtype=np.float64)
        return pos + _rotate(quat, self.ee_offset), _multiply(quat, self.ee_offset_rot)


def _rotate(quat: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``quat`` (xyzw, ``[n, 4]``) applied to one vector, per row."""
    u, w = quat[:, :3], quat[:, 3:4]
    uv = u @ v
    return 2.0 * uv[:, None] * u + (w * w - np.einsum("ij,ij->i", u, u)[:, None]) * v + 2.0 * w * np.cross(u, v)


def _multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` (xyzw), ``[n, 4]`` by one quaternion."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = (float(v) for v in b)
    out = np.empty_like(a)
    out[:, 0] = aw * bx + ax * bw + ay * bz - az * by
    out[:, 1] = aw * by - ax * bz + ay * bw + az * bx
    out[:, 2] = aw * bz + ax * by - ay * bx + az * bw
    out[:, 3] = aw * bw - ax * bx - ay * by - az * bz
    return out
