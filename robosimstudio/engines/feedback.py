# SPDX-License-Identifier: Apache-2.0
"""Contact wrenches out of a foreign engine, shaped, and fed into ``body_f``.

Two halves of one seam, and both are engine-neutral on purpose.

:class:`WrenchAccumulator` is the host half: reactions arrive one at a time, in
whatever form the engine reports them, and leave as a single ``[n_bodies, 6]``
array in Newton's convention -- linear force at the body's centre of mass,
torque beside it, both world frame. Two engines report two different things and
this is where the difference stops: SuperDex exposes a per-actor-pair *contact
force* and nothing else, so it contributes a force at a point and the moment arm
is computed here; Genesis's IPC coupler exposes a constraint reaction that is
already a force **and a pure couple**, so it contributes both and only the arm
of the force is added. One ``add`` covers each, and neither engine keeps its own
copy of the centre-of-mass bookkeeping.

:class:`WrenchFeed` is the device half: one persistent ``wp.array`` the host
rewrites each tick and one kernel that adds it into ``body_f``, launched
unconditionally so it may live inside the captured graph. That is the standard
flag-array pattern for feeding a capture -- a host-side ``if`` around a launch
works eagerly and then silently does nothing once captured.

**The shaping is load bearing, not taste**, and the argument is worth keeping
where the code is. A foreign engine's reaction at first contact is stiff, and
the composer applies it as a zero-order hold across a whole control step's
substeps -- explicit Euler on a stiff spring, which launched a 63 g cube at
8 m/s the first time this ran without it. The EMA turns the spike into a ramp
without moving the resting equilibrium (a resting cube still feels exactly its
weight), and the cap bounds the impulse any single step can hand a light body.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from .pose import compose, quat_rotate

__all__ = ["WrenchAccumulator", "WrenchFeed"]


@wp.kernel
def _add_wrench(wrench: wp.array(dtype=wp.spatial_vector), body_f: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    body_f[i] = body_f[i] + wrench[i]


class WrenchAccumulator:
    """One control step's reactions, gathered and shaped for ``body_f``.

    Args:
        model: The finalized ``newton.Model``; read once for ``body_com`` and
            ``body_mass``, which the shaping needs and which do not change.
        alpha: EMA coefficient for a *rising* force. 1.0 disables the ramp.
        dv_max: The most velocity [m/s] one control step's wrench may add to a
            body. Scales the whole wrench, torque with it.
    """

    def __init__(self, model: Any, *, alpha: float = 0.35, dv_max: float = 0.5) -> None:
        count = int(getattr(model, "body_count", 0))
        self.alpha = float(alpha)
        self.dv_max = float(dv_max)
        self._com = model.body_com.numpy().copy() if count else np.zeros((0, 3))
        self._mass = model.body_mass.numpy().copy() if count else np.zeros(0)
        self._raw = np.zeros((count, 6), dtype=np.float64)
        self._ema = np.zeros((count, 6), dtype=np.float64)
        self._out = np.zeros((count, 6), dtype=np.float32)

    @property
    def ema(self) -> np.ndarray:
        """The filter's state. Read by tests; not for writing."""
        return self._ema

    def begin(self) -> None:
        """Start a control step: forget the reactions of the last one."""
        self._raw.fill(0.0)

    def add(
        self,
        body: int,
        force: np.ndarray,
        point: np.ndarray,
        body_pose: np.ndarray,
        torque: np.ndarray | None = None,
    ) -> None:
        """One reaction on one body, in world coordinates.

        Args:
            body: Index into the model's bodies.
            force: The force **on the body**. An engine that reports the force
                on the *deformable* negates before it gets here.
            point: Where the force acts, world frame.
            body_pose: The body's world pose ``[x, y, z, qx, qy, qz, qw]``,
                needed to place its centre of mass this step.
            torque: A pure couple, if the engine reports one. A couple has no
                reference point, so it is added as it stands.
        """
        force = np.asarray(force, dtype=np.float64)
        com = body_pose[:3] + quat_rotate(body_pose[3:7], self._com[body])
        self._raw[body, :3] += force
        self._raw[body, 3:] += np.cross(np.asarray(point, dtype=np.float64)[:3] - com, force)
        if torque is not None:
            self._raw[body, 3:] += np.asarray(torque, dtype=np.float64)

    def add_at_shape(self, body: int, force: np.ndarray, body_pose: np.ndarray, local: np.ndarray, **kw) -> None:
        """:meth:`add`, for a reaction reported at a *body-local* frame."""
        self.add(body, force, compose(body_pose, local)[:3], body_pose, **kw)

    def shape(self, dt: float) -> np.ndarray:
        """This step's accumulated reactions, filtered and capped.

        Returns the internal ``[n_bodies, 6]`` float32 buffer -- stable across
        calls, so a :class:`WrenchFeed` bound to it once keeps working.
        """
        raw, ema = self._raw, self._ema
        # Asymmetric on purpose: a rising force ramps in (the anti-launch half),
        # and a force that has dropped -- above all, to zero on separation --
        # lands immediately. A symmetric filter kept pushing a body for three
        # steps after the surfaces had parted, which is how a resting cube
        # turned into a bouncing one.
        rising = np.linalg.norm(raw[:, :3], axis=1) >= np.linalg.norm(ema[:, :3], axis=1)
        ema[rising] += self.alpha * (raw[rising] - ema[rising])
        ema[~rising] = raw[~rising]
        shaped = ema.copy()
        if dt > 0.0:
            magnitude = np.linalg.norm(shaped[:, :3], axis=1)
            limit = self._mass * (self.dv_max / float(dt))
            over = magnitude > np.maximum(limit, 1.0e-12)
            if np.any(over):
                scale = np.ones_like(magnitude)
                scale[over] = limit[over] / magnitude[over]
                shaped *= scale[:, None]
        self._out[:] = shaped.astype(np.float32)
        return self._out

    def zero(self) -> np.ndarray:
        """Forget everything, filter state included. What a reset wants."""
        self._raw.fill(0.0)
        self._ema.fill(0.0)
        self._out.fill(0.0)
        return self._out


class WrenchFeed:
    """The device side: a persistent wrench array, added into ``body_f``."""

    def __init__(self, model: Any) -> None:
        self.device = model.device
        self.array = wp.zeros(int(model.body_count), dtype=wp.spatial_vector, device=model.device)

    def assign(self, wrench: np.ndarray) -> None:
        """Copy a host ``[n_bodies, 6]`` float32 array onto the device."""
        self.array.assign(wrench)

    def stage(self, state: Any, dt: float) -> None:
        """Add the harvested wrenches into ``body_f``. The ``rigid_force`` stage.

        Launched unconditionally: the composer schedules this inside the
        captured graph, and a launch guarded on the host would run once
        eagerly and then never again.
        """
        del dt
        if state.body_f is None:
            return
        wp.launch(_add_wrench, dim=self.array.shape[0], inputs=[self.array, state.body_f], device=self.device)
