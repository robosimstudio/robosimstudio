# SPDX-License-Identifier: Apache-2.0
"""The arithmetic under the batched boundary solve, against the arithmetic it copies.

:mod:`robosimstudio.solvers.ik.fleet_dls` runs
:meth:`~robosimstudio.solvers.ik.arm_ik.ArmIK.solve_q`'s search for a whole wave
at once, and its whole claim is that it is the *same* search -- the descent is a
fixed-point iteration whose basin choice flips on float noise, so an operand
that differs in the fifth decimal is not a rounding difference, it is a
different answer.

This module exists because that went wrong once, in the way it will go wrong
again. The batched rotation matrix was written as the textbook formula with its
hard-coded factor of two; the scalar one this tree has always used divides by
the quaternion's own norm instead, because -- as its docstring says -- "a
quaternion that has drifted off the unit sphere (which read-back float32 poses
routinely have) still yields an orthogonal matrix rather than one that quietly
scales". These quaternions come straight off a float32 ``body_q`` read, so they
have drifted, and the two forms disagreed by 5e-05 rad in the first iteration.

No GPU: two functions, the same inputs.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio.core.math import quat_to_mat
from robosimstudio.solvers.ik.fleet_dls import _quat_to_mats

TIER = "fast"
"""Pure numpy. Gates every commit."""


def _drifted(rng, n: int, scale: float) -> np.ndarray:
    """Unit quaternions nudged off the sphere, the way a float32 read leaves them."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1)[:, None]
    return q * (1.0 + rng.normal(scale=scale, size=(n, 1)))


class TestTheBatchedRotationIsTheScalarOne(unittest.TestCase):
    """Row for row, including the normalisation that is the whole point."""

    def test_unit_quaternions_agree(self):
        rng = np.random.default_rng(3)
        quats = _drifted(rng, 64, 0.0)

        batched = _quat_to_mats(quats)
        one_at_a_time = np.stack([quat_to_mat(q) for q in quats])

        self.assertLess(float(np.abs(batched - one_at_a_time).max()), 1e-15)

    def test_drifted_quaternions_agree(self):
        """The case that actually occurs, and the one the bug was in.

        A float32 ``body_q`` read comes back off the unit sphere by ~1e-7; this
        exaggerates it to make the failure unmistakable rather than marginal.
        """
        rng = np.random.default_rng(4)
        for scale in (1e-7, 1e-4, 1e-2):
            with self.subTest(drift=scale):
                quats = _drifted(rng, 64, scale)
                batched = _quat_to_mats(quats)
                one_at_a_time = np.stack([quat_to_mat(q) for q in quats])
                self.assertLess(float(np.abs(batched - one_at_a_time).max()), 1e-15)

    def test_a_drifted_quaternion_still_gives_an_orthogonal_matrix(self):
        """What the normalisation buys, stated directly: an un-normalised form
        returns a matrix that quietly scales, and a scaled 'rotation' inside a
        Jacobian is a wrong gradient rather than a small one."""
        rng = np.random.default_rng(5)
        mats = _quat_to_mats(_drifted(rng, 32, 1e-2))

        for mat in mats:
            np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-12)

    def test_a_zero_quaternion_is_the_identity(self):
        """The scalar form's guard, which a batched divide would turn into a nan."""
        out = _quat_to_mats(np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]))

        np.testing.assert_allclose(out[0], np.eye(3), atol=0)
        np.testing.assert_allclose(out[1], quat_to_mat([0.0, 0.0, 0.0, 1.0]), atol=0)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_a_quarter_turn_about_z_is_what_it_should_be(self):
        """One matrix checked by eye, so the whole family is anchored to
        something a reader can verify without running anything."""
        half = np.sqrt(0.5)
        mat = _quat_to_mats(np.array([[0.0, 0.0, half, half]]))[0]

        np.testing.assert_allclose(mat @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
