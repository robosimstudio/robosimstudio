# SPDX-License-Identifier: Apache-2.0
"""Core mechanisms: math, camera intrinsics, the substep guard, the re-key fix."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import newton
import numpy as np

from robosimstudio.core.builder import (
    assert_even_substeps,
    rekey_custom_attributes,
    restate_body_attribute,
)
from robosimstudio.core.cameras import CameraSpec
from robosimstudio.core.math import (
    quat_from_axis_angle,
    quat_inverse,
    quat_multiply,
    quat_slerp,
    quat_to_axis_angle,
    quat_to_mat,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    transform_matrix,
    xform_from_matrix,
)
from robosimstudio.core.probes.clearance import ClearanceProbe, _segment_segment
from robosimstudio.core.probes.particle import ParticleProbe
from robosimstudio.core.render.live import RenderBudget
from robosimstudio.core.render.video import colorize_depth, colorize_labels
from robosimstudio.core.shell import grid_surface, mid_surface, prism_slab

TIER = "fast"
"""No GPU. Gates every commit."""


class TestMath(unittest.TestCase):
    """The canonical quaternion helpers."""

    def test_wxyz_round_trip(self):
        """Converting to wxyz and back is the identity."""
        q = np.array([0.1, -0.2, 0.3, 0.9])
        q = q / np.linalg.norm(q)
        np.testing.assert_allclose(quat_wxyz_to_xyzw(quat_xyzw_to_wxyz(q)), q, atol=1e-12)

    def test_quat_multiply_matches_matrix_product(self):
        """Quaternion composition agrees with composing the rotation matrices."""
        a = quat_from_axis_angle((0, 0, 1), 0.7)
        b = quat_from_axis_angle((1, 0, 0), -0.4)
        np.testing.assert_allclose(quat_to_mat(quat_multiply(a, b)), quat_to_mat(a) @ quat_to_mat(b), atol=1e-12)

    def test_quat_inverse(self):
        """A rotation composed with its inverse is the identity."""
        q = quat_from_axis_angle((0.3, -0.5, 0.8), 2.1)
        np.testing.assert_allclose(quat_to_mat(quat_multiply(q, quat_inverse(q))), np.eye(3), atol=1e-12)

    def test_axis_angle_round_trip_takes_the_short_way(self):
        """Past half a turn the rotation vector comes back the near way round.

        A servo handed the long way round would take a joint through its limits
        to reach an orientation a few degrees away.
        """
        for angle in (0.1, 1.0, 3.0):
            back = quat_to_axis_angle(quat_from_axis_angle((0, 0, 1), angle))
            np.testing.assert_allclose(back, [0.0, 0.0, angle], atol=1e-9)
        self.assertAlmostEqual(quat_to_axis_angle(quat_from_axis_angle((0, 0, 1), 2 * np.pi - 0.3))[2], -0.3)

    def test_slerp_interpolates_the_short_arc(self):
        """Halfway between two rotations, the near way, whatever hemisphere they are in."""
        a = quat_from_axis_angle((0, 0, 1), 0.0)
        b = quat_from_axis_angle((0, 0, 1), 1.0)
        self.assertAlmostEqual(quat_to_axis_angle(quat_slerp(a, b, 0.5))[2], 0.5, places=9)
        # ``-b`` is the same rotation, so it must interpolate the same way.
        self.assertAlmostEqual(quat_to_axis_angle(quat_slerp(a, -b, 0.5))[2], 0.5, places=9)
        np.testing.assert_allclose(quat_slerp(a, b, 0.0), a, atol=1e-12)

    def test_xform_matrix_round_trip_at_180_degrees(self):
        """A 180-degree yaw survives the matrix round trip.

        The branch selection matters here: at 180 degrees ``w`` is 0, so a
        conversion that assumes ``w`` is the largest component divides by zero
        and produces a NaN pose. Half the camera rigs sit at this angle.
        """
        for angle in (0.0, np.pi / 2, np.pi, -np.pi + 1e-6):
            xform = np.append([1.0, -2.0, 3.0], quat_from_axis_angle((0, 0, 1), angle))
            back = xform_from_matrix(transform_matrix(xform))
            np.testing.assert_allclose(back[:3], xform[:3], atol=1e-12)
            # A quaternion and its negation are the same rotation.
            self.assertLess(min(np.linalg.norm(back[3:] - xform[3:]), np.linalg.norm(back[3:] + xform[3:])), 1e-9)


class TestCameraSpec(unittest.TestCase):
    """Intrinsics, and the aperture form the tiled backend renders through."""

    def test_fov_and_explicit_intrinsics_agree(self):
        """A field of view and the focal length it implies give the same K."""
        by_fov = CameraSpec(name="c", width=640, height=480, fov_y=45.0)
        f = by_fov.intrinsics[1, 1]
        by_k = CameraSpec(name="c", width=640, height=480, fx=f, fy=f)
        np.testing.assert_allclose(by_fov.intrinsics, by_k.intrinsics, atol=1e-9)

    def test_scaled_preserves_field_of_view(self):
        """Rescaling a camera scales K and leaves the field of view alone."""
        spec = CameraSpec(name="c", width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        half = spec.scaled(320, 240)
        self.assertAlmostEqual(spec.fov_y_from_intrinsics, half.fov_y_from_intrinsics, places=9)
        self.assertAlmostEqual(half.intrinsics[0, 0], 250.0, places=9)

    def test_unknown_channel_raises(self):
        """A camera asking for a channel no backend defines is an error."""
        with self.assertRaises(ValueError):
            CameraSpec(name="c", channels=("rgb", "normals"))


class TestSubstepGuard(unittest.TestCase):
    """An odd substep count leaves the frame in the wrong buffer."""

    def test_odd_substeps_raise(self):
        """Odd substep counts are rejected."""
        for n in (1, 3, 7):
            with self.assertRaises(ValueError):
                assert_even_substeps(n)

    def test_even_substeps_pass(self):
        """Even substep counts are returned unchanged."""
        self.assertEqual(assert_even_substeps(20), 20)

    def test_zero_and_negative_raise(self):
        """A non-positive substep count is rejected before the parity check."""
        for n in (0, -2):
            with self.assertRaises(ValueError):
                assert_even_substeps(n)


class TestReplaySteps(unittest.TestCase):
    """What an offline re-render replays, and what it must not take a length from.

    The bug this guards: ``privileged.npz`` carries an episode's *constants*
    beside its per-step state -- ``<cloth>/bond_index``, ``<cloth>/bond_anchor``,
    ``<cloth>/bond_row``, ``particle_subset`` -- and the slicer behind
    :meth:`EpisodeData.replay_steps` truncates to its shortest array. A 340-step
    tape peel with 45 bonds replayed as 45 steps: no error and no warning, just
    the first eighth of the episode and then the end of the clip.
    """

    def _episode(self, **privileged):
        from pathlib import Path

        from robosimstudio.record.store import EpisodeData

        return EpisodeData(path=Path("."), meta={}, steps={}, privileged=privileged)

    def test_an_episode_constant_does_not_shorten_the_replay(self):
        """Twelve steps stay twelve steps next to four bonds and a subset."""
        episode = self._episode(
            body_q=np.zeros((12, 3, 7), dtype=np.float32),
            particle_q=np.zeros((12, 5, 3), dtype=np.float32),
            tape_bond_active=np.zeros((12, 4), dtype=np.uint8),
            particle_subset=np.arange(5, dtype=np.int32),
            **{
                "tape/bond_index": np.zeros(4, dtype=np.int32),
                "tape/bond_anchor": np.zeros((4, 3), dtype=np.float32),
                "tape/bond_row": np.zeros(4, dtype=np.int32),
            },
        )
        steps = list(episode.replay_steps())
        self.assertEqual(len(steps), 12)
        # And narrowed to what a state can be written from: a per-step channel
        # no replay reads is still a length nothing should be measured against.
        self.assertEqual(sorted(steps[0]), ["body_q", "particle_q"])

    def test_a_recording_cut_mid_write_still_truncates(self):
        """The shortest *replayable* channel is still the honest length.

        Truncation is the point when a run was interrupted: the last step's
        particles may never have been written, and rendering a body pose against
        a cloth from another step would be a frame that never existed.
        """
        episode = self._episode(
            body_q=np.zeros((12, 3, 7), dtype=np.float32),
            particle_q=np.zeros((7, 5, 3), dtype=np.float32),
        )
        self.assertEqual(len(list(episode.replay_steps())), 7)


class TestRekeyCustomAttributes(unittest.TestCase):
    """The collapse_fixed_joints BODY-attribute bug, and its fix.

    ``collapse_fixed_joints`` renumbers bodies and re-keys only attributes that
    *reference* an entity; a BODY-frequency attribute keeps its pre-collapse
    keys and its values land on whichever body now holds that index. On a real
    arm that silently removes gravity compensation from half the links.
    """

    def _builder_with_gravcomp(self, values: dict[int, float]) -> newton.ModelBuilder:
        builder = newton.ModelBuilder()
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="gravcomp",
                dtype=float,
                frequency=newton.Model.AttributeFrequency.BODY,
                namespace="mujoco",
                default=0.0,
                values=dict(values),
            )
        )
        return builder

    def test_body_frequency_values_are_rekeyed(self):
        """Values follow their body through the collapse's renumbering."""
        builder = self._builder_with_gravcomp({0: 1.0, 3: 1.0, 7: 1.0})
        results = {"body_remap": {0: 0, 3: 1, 7: 2}, "joint_remap": {}, "body_merged_parent": {}}
        counts = rekey_custom_attributes(builder, results)
        self.assertEqual(counts["mujoco:gravcomp"], 3)
        self.assertEqual(builder.custom_attributes["mujoco:gravcomp"].values, {0: 1.0, 1: 1.0, 2: 1.0})

    def test_merged_body_does_not_displace_its_parent(self):
        """A body merged away hands its value up only if the parent has none."""
        builder = self._builder_with_gravcomp({0: 1.0, 1: 9.0, 2: 5.0})
        # Body 1 merges into body 0, which survives as index 0 and has its own
        # value; body 2 merges into a surviving body 3 that declares none.
        results = {
            "body_remap": {0: 0, 3: 1},
            "joint_remap": {},
            "body_merged_parent": {1: 0, 2: 3},
        }
        rekey_custom_attributes(builder, results)
        values = builder.custom_attributes["mujoco:gravcomp"].values
        self.assertEqual(values[0], 1.0, "the surviving body's own value must win over a merged child's")
        self.assertEqual(values[1], 5.0, "a merged child hands its value to a parent that declares none")

    def test_joint_frequency_values_are_rekeyed(self):
        """A JOINT-frequency attribute follows the joint remap, dropping removals."""
        builder = newton.ModelBuilder()
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="stiffness",
                dtype=float,
                frequency=newton.Model.AttributeFrequency.JOINT,
                namespace="test",
                default=0.0,
                values={0: 1.0, 4: 2.0, 9: 3.0},
            )
        )
        rekey_custom_attributes(builder, {"body_remap": {}, "joint_remap": {0: 0, 9: 1}, "body_merged_parent": {}})
        self.assertEqual(builder.custom_attributes["test:stiffness"].values, {0: 1.0, 1: 3.0})

    def test_restate_writes_every_body(self):
        """The blanket fallback sets the attribute on all surviving bodies."""
        builder = self._builder_with_gravcomp({0: 1.0})
        for _ in range(4):
            builder.add_body()
        written = restate_body_attribute(builder, "mujoco:gravcomp", 1.0)
        self.assertEqual(written, builder.body_count)
        self.assertEqual(set(builder.custom_attributes["mujoco:gravcomp"].values), set(range(builder.body_count)))

    def test_restate_on_a_missing_attribute_is_a_no_op(self):
        """Restating an attribute the import never declared writes nothing."""
        self.assertEqual(restate_body_attribute(newton.ModelBuilder(), "mujoco:gravcomp", 1.0), 0)


class TestRenderBudget(unittest.TestCase):
    """The gate a live view renders behind.

    The bug this guards: a live view paced by a target frame rate alone. Ask for
    15 fps of a backend that takes 300 ms and the loop spends all of its wall
    clock rendering, the step rate collapses, and the picture is of a scene that
    has stopped moving. The cap has to be measured, and it has to be a share of
    the clock rather than a frame count.
    """

    def test_a_cheap_frame_runs_at_the_target_rate(self):
        """When the cost is affordable, the frame rate is the only limit."""
        budget = RenderBudget(target_fps=10.0, max_load=0.5)
        budget.record(started=0.0, cost=0.004)
        budget.record(started=0.1, cost=0.004)
        self.assertAlmostEqual(budget.interval, 0.1, places=6)
        self.assertFalse(budget.due(0.15))
        self.assertTrue(budget.due(0.2))

    def test_an_expensive_frame_self_limits_to_its_share(self):
        """The load cap stretches the interval past what the target asked for.

        38 ms a frame at a 30% cap is one frame every 127 ms -- 7.9 fps, not the
        15 that was requested, and that is the whole mechanism.
        """
        budget = RenderBudget(target_fps=15.0, max_load=0.3)
        for i in range(3):
            budget.record(started=i * 0.2, cost=0.038)
        self.assertAlmostEqual(budget.interval, 0.038 / 0.3, places=6)
        self.assertAlmostEqual(1.0 / budget.interval, 7.9, places=1)

    def test_the_first_measurement_is_replaced_rather_than_blended(self):
        """A backend's warm-up frame must not throttle the view for a minute.

        A tiled sensor compiles its megakernel on the first launch, so frame one
        can cost a hundred times frame two. Folded into an EMA that would leave
        the interval seconds long well after the view became cheap.
        """
        budget = RenderBudget(target_fps=60.0, max_load=0.3)
        budget.record(started=0.0, cost=2.0)
        budget.record(started=3.0, cost=0.004)
        self.assertAlmostEqual(budget.cost, 0.004, places=6)

    def test_the_cap_waits_for_a_cost_it_can_trust(self):
        """While warming up, the frame-rate floor paces things, not the cap.

        Otherwise the one expensive warm-up frame sets an interval that has to
        elapse before the cheap frame that would correct it is allowed to run.
        """
        budget = RenderBudget(target_fps=20.0, max_load=0.3)
        budget.record(started=0.0, cost=2.0)
        self.assertAlmostEqual(budget.interval, 0.05, places=6)

    def test_the_first_frame_is_always_due(self):
        """Nothing measured yet is not a reason to withhold the first frame."""
        self.assertTrue(RenderBudget(target_fps=1.0).due(0.0))

    def test_nothing_is_reported_until_a_window_closes(self):
        """A rate over a fraction of a second is noise, so it is not published."""
        budget = RenderBudget(target_fps=10.0, max_load=0.5, window=1.0)
        for i in range(5):
            budget.record(started=i * 0.1, cost=0.02)
        self.assertEqual(budget.fps, 0.0)
        self.assertEqual(budget.frames, 5, "frames are still counted while the window fills")

    def test_load_and_fps_are_measured_over_a_closed_window(self):
        """The readout reports what the loop paid, not what was configured.

        A settled window is exact because its start is the previous window's
        closing *time* rather than a frame: 10 frames of 20 ms in the second
        second is 10 fps at 20% load, which is the number to compare against the
        cap. Only the first window is a frame long, since it has no predecessor
        to start from.
        """
        budget = RenderBudget(target_fps=10.0, max_load=0.5, window=1.0)
        for i in range(21):
            budget.record(started=i * 0.1, cost=0.02)
        self.assertAlmostEqual(budget.fps, 10.0, places=6)
        self.assertAlmostEqual(budget.load, 0.2, places=6)

    def test_a_zero_target_leaves_the_cap_in_charge(self):
        """'As often as my share allows' is a legitimate setting."""
        budget = RenderBudget(target_fps=0.0, max_load=0.25)
        for i in range(3):
            budget.record(started=i, cost=0.05)
        self.assertAlmostEqual(budget.interval, 0.2, places=6)

    def test_a_degenerate_cap_does_not_divide_by_zero(self):
        """A cap of zero is clamped rather than raising inside the loop."""
        budget = RenderBudget(target_fps=10.0, max_load=0.0)
        for i in range(3):
            budget.record(started=i, cost=0.01)
        self.assertTrue(np.isfinite(budget.interval))

    def test_reset_forgets_the_measurement(self):
        """Another backend costs what it costs; carrying a number over lies."""
        budget = RenderBudget(target_fps=10.0, max_load=0.3)
        for i in range(3):
            budget.record(started=i, cost=0.5)
        budget.reset()
        self.assertEqual(budget.cost, 0.0)
        self.assertEqual(budget.frames, 0)
        self.assertTrue(budget.due(0.0))


class TestColorizeChannels(unittest.TestCase):
    """Turning a non-rgb channel into something displayable."""

    def test_depth_misses_stay_black(self):
        """0 means a miss everywhere in this stack, including in a preview."""
        depth = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        image = colorize_depth(depth)
        self.assertEqual(image.shape, (2, 2, 3))
        np.testing.assert_array_equal(image[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(image[1, 1], [0, 0, 0])
        # Near is bright, so the closer hit outranks the further one.
        self.assertGreater(int(image[0, 1, 0]), int(image[1, 0, 0]))

    def test_labels_are_coloured_and_the_background_is_black(self):
        labels = np.array([[0, 1], [2, 3]], dtype=np.int32)
        image = colorize_labels(labels)
        self.assertEqual(image.shape, (2, 2, 3))
        np.testing.assert_array_equal(image[0, 0], [0, 0, 0])
        self.assertGreater(int(image[0, 1].sum()), 0)

    def test_adjacent_ids_are_far_apart(self):
        """Two objects lying against each other must not be two shades of one colour.

        A linear hue ramp gives consecutive ids nearly the same colour, which is
        the one thing a segmentation view exists to distinguish.
        """
        colors = colorize_labels(np.arange(1, 9, dtype=np.int32)).reshape(-1, 3).astype(np.int32)
        gaps = np.abs(np.diff(colors, axis=0)).max(axis=1)
        self.assertGreater(int(gaps.min()), 60)

    def test_the_same_id_always_gets_the_same_colour(self):
        """A label's colour cannot depend on what else is in frame."""
        alone = colorize_labels(np.array([7], dtype=np.int32))
        crowded = colorize_labels(np.array([3, 7, 11], dtype=np.int32))
        np.testing.assert_array_equal(alone[0], crowded[1])


class TestPrismSlab(unittest.TestCase):
    """Giving a surface a thickness, which is the whole of a bath towel.

    The failures this guards against are all silent ones. A slab whose tets do
    not conform simulates a crack; a slab with an inverted tet is rejected at
    build with a message about volume that says nothing about thickness; a slab
    whose mass is lumped wrong weighs the wrong amount and drapes accordingly.
    None of them look like a thickness bug when they surface.
    """

    def _slab(self, thickness=0.005, cells=(6, 5)):
        vertices, faces = grid_surface((0.06, 0.05), cells)
        return vertices, faces, *prism_slab(vertices, faces, thickness)

    def test_volume_is_area_times_thickness(self):
        """A flat sheet's slab holds exactly the volume it claims to."""
        *_, slab, tets = self._slab()
        edges = slab[tets[:, 1:]] - slab[tets[:, [0]]]
        volume = float((np.linalg.det(edges) / 6.0).sum())
        self.assertAlmostEqual(volume, 0.06 * 0.05 * 0.005, places=12)

    def test_every_tet_is_positively_oriented(self):
        """``add_tetrahedron`` rejects a non-positive volume, so none may exist.

        The input surface's winding is not something an arbitrary mesh can be
        trusted about, so this holds for a mesh wound the wrong way too.
        """
        for flip in (False, True):
            vertices, faces = grid_surface((0.06, 0.05), (6, 5))
            if flip:
                faces = faces[:, ::-1]
            slab, tets = prism_slab(vertices, faces, 0.005)
            edges = slab[tets[:, 1:]] - slab[tets[:, [0]]]
            self.assertGreater(float((np.linalg.det(edges) / 6.0).min()), 0.0, f"flipped={flip}")

    def test_tets_conform(self):
        """Neighbouring prisms must agree about the wall they share.

        Split two adjacent prisms on opposite diagonals and their tets meet
        along a crack instead of a face. Nothing raises: VBD simulates the
        cracked mesh happily and the slab quietly comes apart under load. The
        signature is an interior face belonging to one tet instead of two.
        """
        _, faces, _, tets = self._slab()
        quads = np.sort(
            np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), axis=1
        )
        _, counts = np.unique(quads, axis=0, return_counts=True)
        self.assertEqual(int(counts.max()), 2, "a face shared by more than two tets is non-manifold")
        # The boundary is the two faces plus a rim of two triangles per edge.
        boundary_edges = 2 * (6 + 5)
        self.assertEqual(int((counts == 1).sum()), 2 * len(faces) + 2 * boundary_edges)

    def test_thickness_is_the_thickness(self):
        """The surface ends up in the middle, not on one side.

        Offsetting to one side instead would bury a sheet laid on a table half
        its own thickness into the table, and every pose stated for the cloth
        would be off by that much.
        """
        vertices, _, slab, _ = self._slab(thickness=0.004)
        n = len(vertices)
        np.testing.assert_allclose(slab[n:, 2] - slab[:n, 2], 0.004, atol=1e-15)
        np.testing.assert_allclose(mid_surface(slab, n), vertices, atol=1e-15)

    def test_rejects_a_thickness_that_does_not_fit(self):
        """A surface folded tighter than its own half-thickness has no slab."""
        with self.assertRaises(ValueError):
            prism_slab(*grid_surface((0.06, 0.05), (6, 5))[:2], 0.0)

    def test_mid_surface_is_unbiased_when_the_sheet_turns_over(self):
        """The midpoint survives a fold; either face alone does not.

        Score a fold against one face and the flip enters as a full thickness
        of error -- 5 mm on a bath towel, which is the size of the tolerance a
        fold is judged against.
        """
        vertices, faces = grid_surface((0.06, 0.05), (6, 5))
        slab, _ = prism_slab(vertices, faces, 0.005)
        n = len(vertices)
        # Turn the sheet upside down about z: the faces swap places.
        flipped = slab.copy()
        flipped[:, 2] *= -1.0
        np.testing.assert_allclose(mid_surface(flipped, n)[:, 2], 0.0, atol=1e-15)
        # The bottom face alone reads half a thickness on the wrong side.
        self.assertAlmostEqual(float(flipped[:n, 2].mean()), 0.0025, places=12)


class TestCoverage(unittest.TestCase):
    """Occupancy coverage, and the bounding box it exists to replace.

    An unfolding task is *scored* on this, so the metric is the task. A measure
    a policy can satisfy without unfolding anything is not a hard task with a
    lenient gate, it is a different and much easier task wearing the name.
    """

    CELL = 0.02

    def _probe(self, points, baseline):
        """A probe over bare arrays, with no model behind it."""
        probe = ParticleProbe.__new__(ParticleProbe)
        probe.positions = np.asarray(points, dtype=np.float64)
        probe.baseline = np.asarray(baseline, dtype=np.float64)
        probe.indices = lambda name=None: np.arange(len(probe.positions))
        probe.points = lambda name=None: probe.positions
        return probe

    @staticmethod
    def _flat(side: float = 0.3, cells: int = 30):
        grid = np.stack(
            np.meshgrid(np.linspace(0, side, cells + 1), np.linspace(0, side, cells + 1), indexing="xy"), axis=-1
        ).reshape(-1, 2)
        return np.c_[grid, np.zeros(len(grid))]

    def test_flat_sheet_is_fully_covered(self):
        flat = self._flat()
        self.assertAlmostEqual(self._probe(flat, flat).coverage(cell=self.CELL), 1.0, places=6)

    def test_a_ball_covers_almost_nothing(self):
        flat = self._flat()
        rng = np.random.default_rng(0)
        ball = np.c_[
            rng.uniform(0.12, 0.18, len(flat)), rng.uniform(0.12, 0.18, len(flat)), rng.uniform(0.0, 0.06, len(flat))
        ]
        self.assertLess(self._probe(ball, flat).coverage(cell=self.CELL), 0.1)

    def test_flicking_the_corners_out_does_not_count_as_unfolding(self):
        """The case that rules the bounding box out, and the reason this exists.

        A garment lying in a ball with its four corners pulled to the extremes
        fills its bounding rectangle exactly as well as one spread flat, so a
        rectangle scores it 1.00. It has not been unfolded; it has been given
        four whiskers. A policy rewarded on the rectangle finds this, because it
        is enormously cheaper than actually flattening a garment.
        """
        flat = self._flat()
        rng = np.random.default_rng(0)
        gamed = np.c_[
            rng.uniform(0.12, 0.18, len(flat)), rng.uniform(0.12, 0.18, len(flat)), rng.uniform(0.0, 0.06, len(flat))
        ]
        gamed[:4, :2] = [(0.0, 0.0), (0.3, 0.0), (0.0, 0.3), (0.3, 0.3)]
        probe = self._probe(gamed, flat)
        lo, hi = gamed[:, :2].min(axis=0), gamed[:, :2].max(axis=0)
        box = float((hi[0] - lo[0]) * (hi[1] - lo[1]) / 0.09)
        self.assertAlmostEqual(box, 1.0, places=6, msg="the bounding box should be fooled -- that is the point")
        self.assertLess(probe.coverage(cell=self.CELL), 0.1, "occupancy must not be")

    def test_folding_in_half_halves_the_coverage(self):
        """A known answer: the same cloth over half the table."""
        flat = self._flat()
        folded = flat.copy()
        far = folded[:, 1] > 0.15
        folded[far, 1] = 0.3 - folded[far, 1]
        folded[far, 2] = 0.005
        self.assertAlmostEqual(self._probe(folded, flat).coverage(cell=self.CELL), 0.5, places=2)

    def test_empty_and_degenerate_are_zero_not_an_error(self):
        """A metric read before anything is built must not raise."""
        empty = np.zeros((0, 3))
        self.assertEqual(self._probe(empty, empty).covered_area(cell=self.CELL), 0.0)
        self.assertEqual(self._probe(empty, empty).coverage(cell=self.CELL), 0.0)

    def test_a_bad_cell_is_rejected(self):
        flat = self._flat()
        with self.assertRaises(ValueError):
            self._probe(flat, flat).covered_area(cell=0.0)


class _Arr:
    """The one thing a probe wants of a model array: ``.numpy()``."""

    def __init__(self, values, dtype=None):
        self._values = np.asarray(values, dtype=dtype)

    def numpy(self):
        return self._values


class TestSegmentDistance(unittest.TestCase):
    """The kernel under :class:`ClearanceProbe`: segment to segment, in bulk."""

    @staticmethod
    def _one(p1, q1, p2, q2) -> float:
        return float(_segment_segment(np.array([p1]), np.array([q1]), np.array([p2]), np.array([q2]))[0, 0])

    def test_parallel_segments_are_their_offset_apart(self):
        d = self._one((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.5, 0.0), (1.0, 0.5, 0.0))
        self.assertAlmostEqual(d, 0.5, places=9)

    def test_crossing_segments_touch(self):
        d = self._one((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))
        self.assertAlmostEqual(d, 0.0, places=9)

    def test_skew_segments_take_the_common_normal(self):
        """Crossing in plan, one unit apart in height."""
        d = self._one((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 1.0), (0.0, 1.0, 1.0))
        self.assertAlmostEqual(d, 1.0, places=9)

    def test_the_closest_points_can_be_endpoints(self):
        """Collinear and disjoint: the answer is the gap between the ends."""
        d = self._one((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.0, 0.0, 0.0), (4.0, 0.0, 0.0))
        self.assertAlmostEqual(d, 2.0, places=9)

    def test_degenerate_segments_are_points(self):
        """A sphere is a zero-length segment, which is why it needs no case."""
        d = self._one((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 3.0, 4.0), (0.0, 3.0, 4.0))
        self.assertAlmostEqual(d, 5.0, places=9)

    def test_it_is_symmetric(self):
        a = self._one((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.3, -0.4, 0.7), (0.9, 0.2, 0.1))
        b = self._one((0.3, -0.4, 0.7), (0.9, 0.2, 0.1), (0.0, 0.0, 0.0), (1.0, 1.0, 0.0))
        self.assertAlmostEqual(a, b, places=9)


class TestClearanceProbe(unittest.TestCase):
    """Two capsules on two bodies, and what the model's filtering does to them."""

    RADIUS = 0.05
    HALF = 0.20

    def _model(self, *, filtered=(), collide=(True, True)):
        """Two z-aligned capsules, each at its own body's origin."""
        capsule = int(newton.GeoType.CAPSULE)
        flags = [int(newton.ShapeFlags.COLLIDE_SHAPES) if on else 0 for on in collide]
        return SimpleNamespace(
            shape_count=2,
            shape_body=_Arr([0, 1], np.int32),
            shape_flags=_Arr(flags, np.int32),
            shape_type=_Arr([capsule, capsule], np.int32),
            shape_scale=_Arr([[self.RADIUS, self.HALF, 0.0]] * 2),
            # Identity local transforms: the shapes sit at their bodies.
            shape_transform=_Arr([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2),
            shape_collision_radius=_Arr([self.HALF + self.RADIUS] * 2),
            shape_collision_filter_pairs=set(filtered),
        )

    @staticmethod
    def _poses(*positions):
        return np.array([[*p, 0.0, 0.0, 0.0, 1.0] for p in positions])

    def test_side_by_side_capsules_report_the_gap_between_their_surfaces(self):
        probe = ClearanceProbe(self._model())
        # Axes 0.3 apart, two 0.05 radii between them.
        gap, a, b = probe.closest([0], [1], self._poses((0.0, 0.0, 0.0), (0.3, 0.0, 0.0)))
        self.assertAlmostEqual(gap, 0.3 - 2 * self.RADIUS, places=9)
        self.assertEqual((a, b), (0, 1))

    def test_overlapping_capsules_report_a_negative_gap(self):
        probe = ClearanceProbe(self._model())
        gap = probe.between([0], [1], self._poses((0.0, 0.0, 0.0), (0.06, 0.0, 0.0)))
        self.assertAlmostEqual(gap, 0.06 - 2 * self.RADIUS, places=9)
        self.assertLess(gap, 0.0)

    def test_end_to_end_capsules_measure_along_their_axes(self):
        """The half-heights count: the answer is not centre-to-centre."""
        probe = ClearanceProbe(self._model())
        gap = probe.between([0], [1], self._poses((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
        self.assertAlmostEqual(gap, 1.0 - 2 * self.HALF - 2 * self.RADIUS, places=9)

    def test_a_filtered_pair_is_infinitely_far_apart(self):
        """Not a collision however close it comes -- a gripper's pads meet by design."""
        probe = ClearanceProbe(self._model(filtered=[(0, 1)]))
        gap, a, b = probe.closest([0], [1], self._poses((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        self.assertEqual(gap, float("inf"))
        self.assertEqual((a, b), (-1, -1))
        self.assertEqual(probe.live_pairs([0], [1]), 0)

    def test_a_non_colliding_shape_is_not_measured(self):
        probe = ClearanceProbe(self._model(collide=(True, False)))
        self.assertEqual(probe.between([0], [1], self._poses((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))), float("inf"))
        self.assertEqual(probe.live_pairs([0], [1]), 0)

    def test_live_pairs_counts_what_the_broad_phase_would_test(self):
        self.assertEqual(ClearanceProbe(self._model()).live_pairs([0], [1]), 1)

    def test_a_box_is_bounded_rather_than_missed(self):
        """A box becomes a capsule around its longest axis -- never optimistic."""
        model = self._model()
        model.shape_type = _Arr([int(newton.GeoType.BOX), int(newton.GeoType.BOX)], np.int32)
        model.shape_scale = _Arr([[0.01, 0.02, 0.10]] * 2)
        probe = ClearanceProbe(model)
        # Long axes side by side across x: the true surface gap is 0.5 - 2*0.01,
        # and the bound thickens each by hypot(0.01, 0.02) instead.
        gap = probe.between([0], [1], self._poses((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)))
        self.assertLess(gap, 0.5 - 2 * 0.01, "the bound must never claim more room than there is")
        self.assertAlmostEqual(gap, 0.5 - 2 * float(np.hypot(0.01, 0.02)), places=9)


if __name__ == "__main__":
    unittest.main()
