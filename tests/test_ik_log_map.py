# SPDX-License-Identifier: Apache-2.0
"""The rotation log map the IK solvers read their orientation errors with: ``|log R|`` is the angle, for every ``R``.

:func:`robosimstudio.core.math.mat_to_axis_angle`, which the batched full-pose
polish (:mod:`robosimstudio.solvers.ik.fleet_dls`) measures its rotation
residual with, over a stack, and PyRoKi's root-frame measurement reads each
joint's turn with, one matrix at a time. The property is the definition: the
vector's length is the rotation's angle and the exponential of the vector is
the rotation, on random rotations and on the ones a log map is easy to get
wrong at -- no turn, a tiny one, and a half turn, approached to within 1e-9 rad
and reached. A half turn read as nothing is a wrist upside down that the
polish calls closed.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio.core.math import mat_to_axis_angle

TIER = "fast"
"""Pure numpy over 3x3 matrices: no GPU."""

HARD_ANGLES = (0.0, 1.0e-8, 1.0e-4, 0.5, np.pi / 2, 3.0, np.pi - 1.0e-3, np.pi - 1.0e-6, np.pi - 1.0e-7)
HARD_ANGLES += (np.pi - 1.0e-9, np.pi)
"""Where a log map divides by ``sin`` of the angle, and so where one goes wrong."""


def _exp(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues' formula: the rotation a rotation vector stands for."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1.0e-15:
        return np.eye(3)
    x, y, z = rotvec / angle
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _rotations(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(angles [n], rotations [n, 3, 3])``: every hard angle about random axes, then random rotations."""
    rng = np.random.default_rng(seed)
    angles = np.concatenate([np.repeat(HARD_ANGLES, 4), rng.uniform(0.0, np.pi, 40)])
    axes = rng.normal(size=(len(angles), 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return angles, np.stack([_exp(axis * angle) for axis, angle in zip(axes, angles, strict=True)])


class _LogMapLaw:
    """The two claims, over whichever log map a subclass reads with."""

    def log(self, rotations: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def test_the_length_is_the_angle(self) -> None:
        angles, rotations = _rotations(0)
        lengths = np.linalg.norm(self.log(rotations), axis=1)
        for angle, length in zip(angles, lengths, strict=True):
            self.assertAlmostEqual(angle, length, delta=1.0e-6, msg=f"a turn of pi - {np.pi - angle:.3g} rad")

    def test_the_exponential_of_the_log_is_the_rotation(self) -> None:
        angles, rotations = _rotations(1)
        for angle, rotation, rotvec in zip(angles, rotations, self.log(rotations), strict=True):
            np.testing.assert_allclose(_exp(rotvec), rotation, atol=1.0e-6, err_msg=f"a turn of {angle!r} rad")


class TestTheBatchedLogMap(_LogMapLaw, unittest.TestCase):
    """Over a stack ``[n, 3, 3]``: what the full-pose polish measures the orientation it closes with."""

    def log(self, rotations: np.ndarray) -> np.ndarray:
        return mat_to_axis_angle(rotations)


class TestOneMatrixAtATime(_LogMapLaw, unittest.TestCase):
    """On one ``[3, 3]``: how far each probe turned the tool, when PyRoKi measures the arm's root frame."""

    def log(self, rotations: np.ndarray) -> np.ndarray:
        return np.stack([mat_to_axis_angle(r) for r in rotations])

    def test_one_matrix_reads_as_its_row_of_a_stack(self) -> None:
        _, rotations = _rotations(2)
        np.testing.assert_array_equal(mat_to_axis_angle(rotations), self.log(rotations))
        self.assertEqual((3,), mat_to_axis_angle(np.eye(3)).shape)


class TestARotationReadBackInFloat32(unittest.TestCase):
    """A rotation off a float32 ``body_q``: its entries carry 6e-8 of noise, which the log map must not magnify.

    Just short of where it read the axis off the symmetric part, the angle came
    from an arccos over ``2 sin``, which multiplies that noise by about
    ``pi / sin``: a turn of ``pi - 1.001e-3`` read 0.1 rad off.
    """

    def test_turns_near_a_half_turn_read_their_angle_to_the_noise(self) -> None:
        rng = np.random.default_rng(3)
        for short in (1.001e-3, 1.5e-3, 3.0e-3, 1.0e-2, 0.1, 9.99e-4, 1.0e-5):
            for _ in range(8):
                axis = rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                rotation = _exp(axis * (np.pi - short)).astype(np.float32).astype(np.float64)
                rotvec = mat_to_axis_angle(rotation)
                with self.subTest(short_of_a_half_turn=short):
                    self.assertAlmostEqual(np.pi - short, float(np.linalg.norm(rotvec)), delta=1.0e-5)
                    np.testing.assert_allclose(_exp(rotvec), rotation, atol=1.0e-5)


class TestAHalfTurnLeftOpenIsSeen(unittest.TestCase):
    """What the log map is for: the polish measures a wrist half a turn out as half a turn, not as closed."""

    def test_the_polish_error_of_a_half_turn_is_pi(self) -> None:
        half = np.diag([1.0, -1.0, -1.0])  # a half turn about x: the tool upside down
        for off in (0.0, 1.0e-7):
            with self.subTest(short_of_a_half_turn_by=off):
                turn = _exp(np.array([np.pi - off, 0.0, 0.0])) if off else half
                self.assertAlmostEqual(np.pi - off, float(np.linalg.norm(mat_to_axis_angle(turn))), places=6)


class _OneJointWrist:
    """A wrist that turns about x by its one joint, its tool at the origin: the polish's ``fk``."""

    def poses(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        half = np.asarray(q, dtype=np.float64)[:, 0] / 2.0
        quat = np.stack([np.sin(half), np.zeros_like(half), np.zeros_like(half), np.cos(half)], axis=1)  # xyzw
        return np.zeros((len(q), 3)), quat


class TestThePolishTurnsAWristUpsideDownBack(unittest.TestCase):
    """The batched full-pose polish reads its orientation error with the log map: a wrist half a turn out is moved
    toward its target rather than called closed and handed back where it was."""

    def test_a_wrist_seeded_at_or_near_a_half_turn_from_its_target_is_moved_toward_it(self) -> None:
        from robosimstudio.solvers.ik.fleet_dls import POLISH_MAX_MOVE, polish_pose_batched, pose_errors_batched

        wrist = _OneJointWrist()
        for off in (0.0, 1.0e-7):
            with self.subTest(short_of_a_half_turn_by=off):
                seed = np.array([[np.pi - off]])
                answer = polish_pose_batched(
                    wrist,
                    seed,
                    np.zeros((1, 3)),
                    np.eye(3)[None],
                    lower=np.array([-2.0 * np.pi]),
                    upper=np.array([2.0 * np.pi]),
                )
                _, angle = pose_errors_batched(wrist, answer, np.zeros((1, 3)), np.array([[0.0, 0.0, 0.0, 1.0]]))
                self.assertAlmostEqual(POLISH_MAX_MOVE, abs(float(answer[0, 0] - seed[0, 0])), places=6)
                self.assertLess(float(angle[0]), np.pi - off - 0.25, "the wrist is still upside down")


if __name__ == "__main__":
    unittest.main()
