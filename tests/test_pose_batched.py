# SPDX-License-Identifier: Apache-2.0
"""The batched pose arithmetic is the per-pose arithmetic, to the last bit that matters.

:mod:`robosimstudio.engines.pose` holds each operation twice: once for one pose,
once for a whole set. The second exists only because of how often the first is
called -- a three-world ``surface_prep`` composes and interpolates 2808 mirrors
twice a control step, and at that count the per-call overhead of a seven-element
numpy operation is most of the cost of doing it.

Two copies of an operation is a standing invitation for them to drift, so this
module is the thing that stops it: every batched form is checked against a loop
over its scalar twin, on inputs chosen to reach the branches that are easy to
get wrong when the shape changes -- the shortest-arc sign flip, which is per row
and decides whether a gripper sweeps the long way round through a deformable,
and the degenerate norm the sign flip can produce.

No GPU, no engine, no model: arithmetic against arithmetic.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio.engines.cosim import CosimRig
from robosimstudio.engines.pose import compose, compose_many, lerp_pose, lerp_poses

TIER = "fast"
"""Pure numpy. Gates every commit."""

TOL = 1e-12
"""Not exact: the batched form reassociates the same products. A pose is metres
and unit quaternions, so this is far below anything a contact could see."""


def _quats(rng, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1)[:, None]


def _poses(rng, n: int) -> np.ndarray:
    return np.hstack([rng.normal(size=(n, 3)), _quats(rng, n)])


class TestComposeMany(unittest.TestCase):
    """``body ∘ local`` for a set is the same product for each of them."""

    def test_it_matches_the_one_pose_form(self):
        rng = np.random.default_rng(11)
        bodies, locals_ = _poses(rng, 64), _poses(rng, 64)

        batched = compose_many(bodies, locals_)
        one_at_a_time = np.stack([compose(bodies[i], locals_[i]) for i in range(64)])

        self.assertLess(float(np.abs(batched - one_at_a_time).max()), TOL)

    def test_an_identity_local_leaves_the_body_pose_alone(self):
        """The case a reader checks by eye, and the one a transposed index breaks."""
        rng = np.random.default_rng(12)
        bodies = _poses(rng, 8)
        ident = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (8, 1))

        self.assertLess(float(np.abs(compose_many(bodies, ident) - bodies).max()), TOL)

    def test_an_offset_local_is_rotated_into_the_body_frame(self):
        """A quarter turn about +z sends a local +x offset to world +y.

        Stated for one body rather than derived, so this fails if the batched
        rotation silently became an inverse or a transpose.
        """
        half = np.sqrt(0.5)
        body = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, half, half]])
        local = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        out = compose_many(body, local)[0]

        np.testing.assert_allclose(out[:3], [1.0, 2.1, 3.0], atol=1e-12)

    def test_an_empty_set_composes_to_an_empty_set(self):
        self.assertEqual(compose_many(np.zeros((0, 7)), np.zeros((0, 7))).shape, (0, 7))


class TestLerpPoses(unittest.TestCase):
    """The substep sweep for a set is the sweep for each of them."""

    def test_it_matches_the_one_pose_form_at_every_substep(self):
        rng = np.random.default_rng(13)
        a, b = _poses(rng, 64), _poses(rng, 64)

        for t in (0.0, 0.25, 0.5, 1.0):
            with self.subTest(t=t):
                batched = lerp_poses(a, b, t)
                one_at_a_time = np.stack([lerp_pose(a[i], b[i], t) for i in range(64)])
                self.assertLess(float(np.abs(batched - one_at_a_time).max()), TOL)

    def test_the_sign_match_is_per_row(self):
        """A set where some rows need the flip and some do not.

        Getting this wrong is not a rounding difference: a mirror that takes the
        long way round between two substeps sweeps its contact surface through
        whatever it is resting against. Rows are built in pairs -- ``q`` and
        ``-q`` for the same rotation -- so exactly half of them need the flip and
        every row must nonetheless land on the same place.
        """
        rng = np.random.default_rng(14)
        start = _poses(rng, 32)
        ends = _quats(rng, 32)
        flipped = ends.copy()
        flipped[::2] *= -1.0  # same rotation, opposite representation

        far = np.hstack([start[:, :3] + 0.1, ends])
        far_flipped = np.hstack([start[:, :3] + 0.1, flipped])

        one = lerp_poses(start, far, 0.5)
        other = lerp_poses(start, far_flipped, 0.5)

        # Up to the quaternion double cover: the two must name one rotation.
        dots = np.abs(np.einsum("ij,ij->i", one[:, 3:7], other[:, 3:7]))
        self.assertLess(float(np.abs(dots - 1.0).max()), 1e-9)
        self.assertLess(float(np.abs(one[:, :3] - other[:, :3]).max()), TOL)

    def test_the_result_stays_a_unit_quaternion(self):
        rng = np.random.default_rng(15)
        a, b = _poses(rng, 48), _poses(rng, 48)

        for t in (0.1, 0.5, 0.9):
            with self.subTest(t=t):
                norms = np.linalg.norm(lerp_poses(a, b, t)[:, 3:7], axis=1)
                self.assertLess(float(np.abs(norms - 1.0).max()), 1e-9)

    def test_opposite_rotations_at_the_midpoint_fall_back_to_the_endpoint(self):
        """The one degenerate case: the sign match cannot save a 180-degree pair,
        whose average is the zero quaternion. The scalar form takes the endpoint
        and so must this, rather than dividing by zero."""
        a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
        b = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

        out = lerp_poses(a, b, 0.5)

        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(np.linalg.norm(out[:, 3:7], axis=1), [1.0], atol=1e-12)
        np.testing.assert_allclose(out[0], lerp_pose(a[0], b[0], 0.5), atol=TOL)

    def test_an_empty_set_interpolates_to_an_empty_set(self):
        self.assertEqual(lerp_poses(np.zeros((0, 7)), np.zeros((0, 7)), 0.5).shape, (0, 7))


class TestTheSubstepDriveUsesThem(unittest.TestCase):
    """The rig yields what it used to, now that the arithmetic is batched.

    :meth:`CosimRig.substeps_of` is the only caller that matters and its
    contract is a list of ``(mirror, pose)``, so the check is that the pairing
    survived the change to array rows -- a zip against the wrong axis would
    still be seven numbers per mirror, and every one of them wrong.
    """

    def test_each_mirror_gets_its_own_pose(self):
        class _Mirror:
            def __init__(self, body, local):
                self.body = body
                self.local = np.asarray(local, dtype=np.float64)

        ident = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        mirrors = [
            _Mirror(2, [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            _Mirror(-1, ident),  # world-fixed: posed once at build, never here
            _Mirror(5, [0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ]
        targets = {
            2: np.array([1.0, 0.0, 0.0, *ident[3:]]),
            5: np.array([0.0, 3.0, 0.0, *ident[3:]]),
        }

        rig = CosimRig([2, 5], substeps=1)
        batches = list(rig.substeps_of(mirrors, targets))

        self.assertEqual(len(batches), 1)
        _sub, posed = batches[0]
        self.assertEqual([m for m, _ in posed], [mirrors[0], mirrors[2]])
        np.testing.assert_allclose(posed[0][1][:3], [1.5, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(posed[1][1][:3], [0.0, 3.25, 0.0], atol=1e-12)

    def test_a_moved_body_moves_its_mirror_on_the_next_tick(self):
        """The layout is held across ticks; the poses are not.

        A rig that cached the *answer* rather than the shape of the question
        would freeze every collider in the scene at its first tick's pose --
        silently, and in a way that looks like a physics bug rather than a
        cache one.
        """
        ident = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        class _Mirror:
            body = 2
            local = np.array(ident)

        mirrors = [_Mirror()]
        rig = CosimRig([2], substeps=1)
        seen = []
        for x in (0.0, 1.0, 5.0):
            targets = {2: np.array([x, 0.0, 0.0, *ident[3:]])}
            rig.commit(targets)  # this tick's pose is the next tick's previous
            _sub, posed = next(iter(rig.substeps_of(mirrors, targets)))
            seen.append(float(posed[0][1][0]))

        self.assertEqual(seen, [0.0, 1.0, 5.0])

    def test_a_different_mirror_set_is_not_answered_from_the_held_plan(self):
        """A caller that hands over a fresh list gets a fresh layout.

        The genesis bridge builds its list per tick, so this is not a
        hypothetical: answering it from another list's plan would pose the
        wrong shapes, or index past the end of it.
        """
        ident = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        class _Mirror:
            def __init__(self, body, x):
                self.body = body
                self.local = np.array([x, 0.0, 0.0, *ident[3:]])

        rig = CosimRig([2, 5], substeps=1)
        targets = {b: np.array(ident) for b in (2, 5)}

        first = next(iter(rig.substeps_of([_Mirror(2, 0.5)], targets)))[1]
        second = next(iter(rig.substeps_of([_Mirror(5, 9.0), _Mirror(2, 0.25)], targets)))[1]

        self.assertEqual([float(p[0]) for _m, p in first], [0.5])
        self.assertEqual([float(p[0]) for _m, p in second], [9.0, 0.25])

    def test_a_changed_body_set_rebuilds_the_plan(self):
        """Eligibility is 'is this mirror's body in targets', so the target keys
        are part of the question and not only the mirror list."""
        ident = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        class _Mirror:
            def __init__(self, body):
                self.body = body
                self.local = np.array(ident)

        mirrors = [_Mirror(2), _Mirror(5)]
        rig = CosimRig([2, 5], substeps=1)

        both = next(iter(rig.substeps_of(mirrors, {b: np.array(ident) for b in (2, 5)})))[1]
        one = next(iter(rig.substeps_of(mirrors, {2: np.array(ident)})))[1]

        self.assertEqual([m.body for m, _p in both], [2, 5])
        self.assertEqual([m.body for m, _p in one], [2])

    def test_a_yielded_pose_is_not_overwritten_by_the_next_substep(self):
        """Rows are views into one array per substep, so a caller that holds a
        pose past the yield -- which the SuperDex bridge does, as ``prev`` --
        must not find it rewritten under it."""

        class _Mirror:
            body = 2
            local = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        rig = CosimRig([2], substeps=4)
        rig.commit({2: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])})
        held = []
        for _sub, posed in rig.substeps_of([_Mirror()], {2: np.array([4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])}):
            held.append(posed[0][1])

        np.testing.assert_allclose([float(p[0]) for p in held], [1.0, 2.0, 3.0, 4.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
