# SPDX-License-Identifier: Apache-2.0
"""The engine-neutral half of a co-simulation, pinned against the code it replaces.

``robosimstudio/engines/feedback.py`` and ``engines/pose.py`` were lifted out of
the SuperDex bridge (now ``engines/superdex/bridge.py``) so that a second engine
second copy of them. A refactor that claims to change no numbers has to be
*checkable*, so the reference implementations below are transcribed verbatim from
the bridge as it stood before the lift -- a reviewer can diff them against the
extracted versions, and these tests assert the two agree bit for bit over a few
thousand randomised steps rather than on one hand-picked example.

``WrenchFeed`` is not covered here: it needs a device, and the fast tier has
none. It is left to the medium tier, where the SuperDex scenes run with an
engine in the loop.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "fast"
"""Pure numpy. No GPU, no engine, no scene."""


# --------------------------------------------------------------------------
# The reference: the SuperDex bridge before the lift, transcribed verbatim.
# --------------------------------------------------------------------------
def _ref_quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _ref_quat_rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)


def _ref_compose(body, local):
    out = np.empty(7)
    out[:3] = body[:3] + _ref_quat_rotate(body[3:7], local[:3])
    out[3:7] = _ref_quat_mul(body[3:7], local[3:7])
    return out


def _ref_lerp_pose(a, b, t):
    out = np.empty(7)
    out[:3] = (1.0 - t) * a[:3] + t * b[:3]
    qb = b[3:7] if np.dot(a[3:7], b[3:7]) >= 0.0 else -b[3:7]
    q = (1.0 - t) * a[3:7] + t * qb
    norm = float(np.linalg.norm(q))
    out[3:7] = q / norm if norm > 0.0 else b[3:7]
    return out


def _ref_shape(raw, ema, mass, alpha, dv_max, dt):
    """The EMA and the velocity cap, exactly as ``tick`` ran them. Mutates ``ema``."""
    rising = np.linalg.norm(raw[:, :3], axis=1) >= np.linalg.norm(ema[:, :3], axis=1)
    ema[rising] += alpha * (raw[rising] - ema[rising])
    ema[~rising] = raw[~rising]
    shaped = ema.copy()
    if dt > 0.0:
        magnitude = np.linalg.norm(shaped[:, :3], axis=1)
        limit = mass * (dv_max / float(dt))
        over = magnitude > np.maximum(limit, 1.0e-12)
        if np.any(over):
            scale = np.ones_like(magnitude)
            scale[over] = limit[over] / magnitude[over]
            shaped *= scale[:, None]
    return shaped.astype(np.float32)


class _Model:
    """The three arrays ``WrenchAccumulator`` reads off a finalized model."""

    class _Arr:
        def __init__(self, a):
            self._a = a

        def numpy(self):
            return self._a

    def __init__(self, com, mass):
        self.body_count = len(mass)
        self.body_com = self._Arr(com)
        self.body_mass = self._Arr(mass)


def _unit_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


class TestPoseHelpers(unittest.TestCase):
    def test_compose_matches_the_bridge_exactly(self):
        from robosimstudio.engines.pose import compose

        rng = np.random.default_rng(20260827)
        for _ in range(2000):
            body = np.concatenate([rng.normal(size=3), _unit_quat(rng)])
            local = np.concatenate([rng.normal(size=3) * 0.05, _unit_quat(rng)])
            np.testing.assert_array_equal(compose(body, local), _ref_compose(body, local))

    def test_lerp_pose_matches_the_bridge_exactly(self):
        from robosimstudio.engines.pose import lerp_pose

        rng = np.random.default_rng(11)
        for _ in range(2000):
            a = np.concatenate([rng.normal(size=3), _unit_quat(rng)])
            b = np.concatenate([rng.normal(size=3), _unit_quat(rng)])
            t = float(rng.uniform(0.0, 1.0))
            np.testing.assert_array_equal(lerp_pose(a, b, t), _ref_lerp_pose(a, b, t))

    def test_lerp_takes_the_short_way_round(self):
        """``q`` and ``-q`` are one rotation; lerping between them must not sweep."""
        from robosimstudio.engines.pose import lerp_pose

        a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        b = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        mid = lerp_pose(a, b, 0.5)
        self.assertAlmostEqual(abs(float(mid[6])), 1.0, places=12, msg="halfway is the same rotation")


class TestWrenchAccumulator(unittest.TestCase):
    """The shaping, against the bridge's own arithmetic, over a randomised run."""

    def _rig(self, bodies=4, seed=3):
        rng = np.random.default_rng(seed)
        com = rng.normal(size=(bodies, 3)) * 0.02
        mass = np.abs(rng.normal(size=bodies)) * 0.5 + 0.05
        return rng, _Model(com, mass), com, mass

    def test_it_reproduces_the_bridge_over_a_long_run(self):
        from robosimstudio.engines.feedback import WrenchAccumulator

        rng, model, com, mass = self._rig()
        alpha, dv_max, dt = 0.35, 0.5, 1.0 / 60.0
        acc = WrenchAccumulator(model, alpha=alpha, dv_max=dv_max)
        ref_ema = np.zeros((model.body_count, 6))

        for step in range(500):
            raw = np.zeros((model.body_count, 6))
            acc.begin()
            for body in range(model.body_count):
                # A contact that comes and goes, so both EMA branches and the
                # cap are all exercised: separation is the falling branch, and
                # the occasional spike is what the cap is for.
                if (step + body) % 7 < 4:
                    force = rng.normal(size=3) * (60.0 if step % 53 == 0 else 3.0)
                    pose = np.concatenate([rng.normal(size=3), _unit_quat(rng)])
                    local = np.concatenate([rng.normal(size=3) * 0.03, _unit_quat(rng)])
                    torque = rng.normal(size=3) * 0.1 if body % 2 else None
                    acc.add_at_shape(body, force, pose, local, torque=torque)
                    world = _ref_compose(pose, local)
                    com_w = pose[:3] + _ref_quat_rotate(pose[3:7], com[body])
                    raw[body, :3] += force
                    raw[body, 3:] += np.cross(world[:3] - com_w, force)
                    if torque is not None:
                        raw[body, 3:] += torque
            got = acc.shape(dt)
            want = _ref_shape(raw, ref_ema, mass, alpha, dv_max, dt)
            np.testing.assert_array_equal(got, want, err_msg=f"diverged at step {step}")

    def test_a_resting_load_keeps_its_weight(self):
        """The EMA ramps a transient; it must not move the equilibrium."""
        from robosimstudio.engines.feedback import WrenchAccumulator

        _, model, _, _ = self._rig(bodies=1)
        acc = WrenchAccumulator(model, alpha=0.35, dv_max=1e9)
        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        weight = np.array([0.0, 0.0, -0.628])
        for _ in range(200):
            acc.begin()
            acc.add(0, weight, np.zeros(3), pose)
            out = acc.shape(1.0 / 60.0)
        self.assertAlmostEqual(float(out[0, 2]), float(weight[2]), places=5)

    def test_separation_lands_immediately(self):
        """The asymmetry: a force that drops to zero does not ring on."""
        from robosimstudio.engines.feedback import WrenchAccumulator

        _, model, _, _ = self._rig(bodies=1)
        acc = WrenchAccumulator(model, alpha=0.35, dv_max=1e9)
        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        for _ in range(50):
            acc.begin()
            acc.add(0, np.array([0.0, 0.0, -5.0]), np.zeros(3), pose)
            acc.shape(1.0 / 60.0)
        acc.begin()  # separated: nothing added this step
        out = acc.shape(1.0 / 60.0)
        np.testing.assert_array_equal(out, np.zeros((1, 6), dtype=np.float32))

    def test_the_cap_bounds_one_step_of_velocity(self):
        from robosimstudio.engines.feedback import WrenchAccumulator

        _, model, _, mass = self._rig(bodies=1)
        dt, dv_max = 1.0 / 60.0, 0.5
        acc = WrenchAccumulator(model, alpha=1.0, dv_max=dv_max)
        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        acc.begin()
        acc.add(0, np.array([0.0, 0.0, -1.0e5]), np.zeros(3), pose)
        out = acc.shape(dt)
        self.assertAlmostEqual(float(np.linalg.norm(out[0, :3])), float(mass[0] * dv_max / dt), places=2)

    def test_zero_forgets_the_filter(self):
        from robosimstudio.engines.feedback import WrenchAccumulator

        _, model, _, _ = self._rig(bodies=2)
        acc = WrenchAccumulator(model)
        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        acc.begin()
        acc.add(0, np.array([0.0, 0.0, -9.0]), np.zeros(3), pose)
        acc.shape(1.0 / 60.0)
        self.assertTrue(np.any(acc.ema))
        np.testing.assert_array_equal(acc.zero(), np.zeros((2, 6), dtype=np.float32))
        self.assertFalse(np.any(acc.ema))


if __name__ == "__main__":
    unittest.main()
