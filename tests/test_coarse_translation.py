# SPDX-License-Identifier: Apache-2.0
"""The coarse translation block's budget, what it refuses, and ``collide=False``.

The block exists because a stiff *connected* FEM's whole-object translation mode
reaches badly under VBD's particle solve: on the labelled Clawhauser volume at 20
substeps and 12 iterations that mode's residual is 192.88 N (29.41 N at 512)
against a 1.227 N weight. What is checkable in a unit is therefore not the
resting balance -- that is measured against the real volume, where the mode is
ill-conditioned, and a 4-particle test mesh has no such mode to fail -- but the
budget that bounds the block's extra motion, the shapes it must refuse, and the
one-line collision contract a non-colliding decor shape relies on.

Refusals are the contract rather than defensive code: one common translation is
the object's motion only if the volume is one face-connected piece with every
particle free and active, so a scene that breaks that must fail loudly at
construction rather than drift.
"""

from __future__ import annotations

import unittest

import newton
import numpy as np
import warp as wp

import robosimstudio as rss
from robosimstudio.domains.static.entity import EntityStatic

TIER = "medium"
"""A GPU: two kernel launches and three solver constructions over 4-8 particles."""


def tet_model(*tets: tuple[int, int, int, int], pinned: int | None = None):
    """Small stiff tet volumes on ``cuda:0``, for the block's own checks.

    Each tet gets its own four corners, at least a millimetre apart and off the
    plane of each other: a colinear or flat set is a zero-volume element, which
    Newton warns about and drops, leaving a model the block refuses for having
    no volume at all instead of for the reason under test.
    """
    corners = np.array(
        [[0.0, 0.0, 0.0], [0.0012, 0.0, 0.0], [0.0004, 0.0009, 0.0], [0.0005, 0.0004, 0.0011]]
    )
    builder = newton.ModelBuilder()
    for index in range(len(tets)):
        for corner in corners:
            position = corner + np.array([0.004 * index, 0.0, 0.8])
            builder.add_particle(pos=tuple(position), vel=(0.0, 0.0, 0.0), mass=0.001)
    if pinned is not None:
        builder.particle_flags[pinned] = builder.particle_flags[pinned] & ~int(newton.ParticleFlags.ACTIVE)
    for tet in tets:
        builder.add_tetrahedron(*tet, k_mu=5.0e5, k_lambda=5.0e6, k_damp=0.0)
    # Newton's own VBD requires particle colour groups, and it checks that inside
    # the solver's constructor -- which the block calls before its own checks, so
    # an uncoloured model would refuse for the wrong reason and never reach the
    # budget refusal this module exists to pin down.
    builder.color()
    return builder.finalize(device="cuda:0")


class TestCoarseTranslation(unittest.TestCase):
    def setUp(self):
        if not wp.is_cuda_available():
            self.skipTest("the coarse block is launched on the particle model's CUDA device")

    def test_budget_clamps_the_cumulative_substep_motion(self):
        from robosimstudio_materials.coarse_translation import _apply_block, _solve_block

        options = wp.get_module_options(_apply_block.module).copy()
        self.addCleanup(wp.set_module_options, options, module=_apply_block.module)
        wp.set_module_options(
            {"deterministic": wp.DeterministicMode.RUN_TO_RUN, "deterministic_max_records": 0},
            module=_apply_block.module,
        )
        device = "cuda:0"
        # A 10 mm proposal: 1 N of residual against a 100 N/m translation block,
        # under the qualified 5 mm budget.
        residual = wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3, device=device)
        hessian = wp.array([np.eye(3, dtype=np.float32) * 100.0], dtype=wp.mat33, device=device)
        path_length = wp.zeros(1, dtype=float, device=device)
        step_path = wp.zeros(1, dtype=float, device=device)
        shift = wp.zeros(1, dtype=wp.vec3, device=device)
        applications = wp.zeros(1, dtype=int, device=device)
        invalid = wp.zeros(1, dtype=int, device=device)
        launch = [
            residual,
            hessian,
            invalid,
            0.005,
            0.005,
            1.0,
            path_length,
            step_path,
            shift,
            applications,
        ]
        wp.launch(_solve_block, dim=1, inputs=launch, device=device)
        self.assertAlmostEqual(float(path_length.numpy()[0]), 0.005, places=6)
        # The frame reader's counter is cumulative: a substep that spends the
        # budget and the next one that spends nothing still add to it, and only
        # ``reset`` clears it.
        self.assertAlmostEqual(float(step_path.numpy()[0]), 0.005, places=6)
        self.assertEqual(int(applications.numpy()[0]), 1)
        reference = np.zeros((1, 3), dtype=np.float32)
        q = wp.array(reference, dtype=wp.vec3, device=device)
        displacement = wp.zeros_like(q)
        wp.launch(
            _apply_block,
            dim=1,
            inputs=[
                wp.zeros(1, dtype=int, device=device),
                shift,
                wp.array(reference, dtype=wp.vec3, device=device),
                q,
                displacement,
            ],
            device=device,
        )
        moved = (q.numpy() - reference)[0]
        np.testing.assert_allclose(moved, np.array([0.005, 0.0, 0.0]), rtol=0, atol=1e-7)
        np.testing.assert_array_equal(displacement.numpy()[0], moved)
        # The budget is per substep: a second iteration proposes nothing more.
        wp.launch(_solve_block, dim=1, inputs=launch, device=device)
        np.testing.assert_array_equal(shift.numpy()[0], np.zeros(3, dtype=np.float32))
        self.assertEqual(int(applications.numpy()[0]), 1)
        self.assertAlmostEqual(float(step_path.numpy()[0]), 0.005, places=6)
        # An unusable solve -- here an overflowed contact buffer -- moves nothing.
        invalid.assign(np.ones(1, dtype=np.int32))
        wp.launch(_solve_block, dim=1, inputs=launch, device=device)
        np.testing.assert_array_equal(shift.numpy()[0], np.zeros(3, dtype=np.float32))
        self.assertEqual(int(applications.numpy()[0]), 1)

    def test_the_block_refuses_what_it_cannot_represent(self):
        from robosimstudio_materials.coarse_translation import SolverVBDCoarse

        with self.assertRaisesRegex(ValueError, "one quarter of contact detection gap"):
            SolverVBDCoarse(tet_model((0, 1, 2, 3)), coarse_contact_gap=0.002, coarse_budget=0.005)
        with self.assertRaisesRegex(ValueError, "one face-connected volume"):
            SolverVBDCoarse(tet_model((0, 1, 2, 3), (4, 5, 6, 7)), coarse_contact_gap=0.02, coarse_budget=0.005)
        with self.assertRaisesRegex(ValueError, "active, unpinned"):
            SolverVBDCoarse(tet_model((0, 1, 2, 3), pinned=0), coarse_contact_gap=0.02, coarse_budget=0.005)

    def test_the_block_accepts_self_contact(self):
        """Self-contact pair forces are internal; the common-translation block
        reads only rigid-soft contacts, so it can stay enabled beside them."""
        from robosimstudio_materials.coarse_translation import SolverVBDCoarse

        solver = SolverVBDCoarse(
            tet_model((0, 1, 2, 3)),
            particle_enable_self_contact=True,
            coarse_contact_gap=0.02,
            coarse_budget=0.005,
        )
        self.assertIsNotNone(solver.coarse_block)

    def test_collide_false_clears_rigid_and_particle_collision(self):
        collision = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
        for collide in (False, True):
            with self.subTest(collide=collide):
                builder = newton.ModelBuilder()
                neighbor = builder.add_shape_box(body=-1, hx=0.1, hy=0.1, hz=0.1, label="neighbor")
                neighbor_flags = int(builder.shape_flags[neighbor])
                entity = EntityStatic(rss.slab(name="decor", half=(0.06, 0.06, 0.00025), collide=collide, visible=True))
                entity.build_into(builder, episode=None)
                self.assertEqual(entity.range.shapes, (1, 2))
                flags = int(builder.shape_flags[1])
                self.assertEqual(flags & collision, collision if collide else 0)
                self.assertEqual(int(builder.shape_flags[neighbor]), neighbor_flags)


if __name__ == "__main__":
    unittest.main()
