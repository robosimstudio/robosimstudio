# SPDX-License-Identifier: Apache-2.0
"""The fold measures, pinned on synthetic geometry.

``cloth_fold_error`` and the
shirt's three-crease score are library vocabulary
(``robosimstudio.assetlib.vocabulary.measures``), scored here on sheets built
in the test rather than on any scene -- the number the benchmark reports has to
be right before a scene is asked to produce it.
"""

from __future__ import annotations

import types
import unittest

import numpy as np

from robosimstudio.assetlib.semantics.atoms import get_atom
from robosimstudio.assetlib.vocabulary.measures import cloth_fold_error, mirror_pairs
from robosimstudio.core.probes import ParticleProbe

TIER = "fast"
"""No GPU. Gates every commit."""


class TestFoldMetric(unittest.TestCase):
    """The fold score, pinned on synthetic layouts.

    Worth having separately from any scene: this is the number the benchmark
    reports, and whether it is right has nothing to do with whether a robot
    cooperated. Each case here is a way of scoring well without having folded
    anything, which is what a fold measure has to reject.
    """

    SIZE = 0.40
    CELLS = 21
    MID = 0.0

    def setUp(self):
        xs = np.linspace(-self.SIZE / 2, self.SIZE / 2, self.CELLS)
        gx, gy = np.meshgrid(xs, xs, indexing="ij")
        self.rest = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
        model = types.SimpleNamespace(particle_count=len(self.rest))
        self.moving = np.flatnonzero(self.rest[:, 1] > self.MID)
        self.probe = ParticleProbe(
            model,
            {
                "c": np.arange(len(self.rest)),
                "c/moving": self.moving,
                "c/base": np.flatnonzero(self.rest[:, 1] <= self.MID),
            },
        )
        self.probe.positions = self.rest.copy()
        self.probe.velocities = np.zeros_like(self.rest)
        self.probe.baseline = self.rest.copy()
        self.pairs = mirror_pairs(self.probe, "c/moving", "c/base", axis=1, origin=self.MID)
        self.ctx = types.SimpleNamespace(probe=lambda _name: self.probe)

    def score(self, positions):
        self.probe.positions = positions
        return cloth_fold_error(self.ctx, "c", self.pairs)

    def test_the_flat_start_scores_zero(self):
        """0 is "no better than the sheet you were handed", by construction."""
        score, _, fixation = self.score(self.rest.copy())
        self.assertAlmostEqual(score, 0.0, places=6)
        self.assertAlmostEqual(fixation, 0.0, places=9)

    def test_a_perfect_fold_scores_one(self):
        """Reflecting the moving half onto the standing half is the ideal."""
        folded = self.rest.copy()
        folded[self.moving, 1] = 2 * self.MID - folded[self.moving, 1]
        folded[self.moving, 2] += 0.001  # one sheet thickness up
        score, group, _ = self.score(folded)
        self.assertGreater(score, 0.99)
        self.assertLess(group, 0.002)

    def test_a_sideways_drag_scores_below_zero(self):
        """The degenerate solution: halve the extent, stack nothing.

        ``cloth_fold``'s extent ratio accepts this. The drag term is the whole
        reason the score does not.
        """
        dragged = self.rest.copy()
        dragged[:, 1] -= self.SIZE / 2
        score, _, fixation = self.score(dragged)
        self.assertLess(score, 0.0)
        self.assertGreater(fixation, 0.1)

    def test_a_crumple_scores_below_zero(self):
        """Gathered into a ball -- a small group distance for the wrong reason.

        The carried particles really are near their targets, because everything
        is near everything. What rejects it is that the standing half had to be
        dragged into the pile to get there.
        """
        rng = np.random.default_rng(0)
        score, group, fixation = self.score(rng.normal(scale=0.02, size=self.rest.shape))
        self.assertLess(score, 0.0)
        self.assertLess(group, 0.06)
        self.assertGreater(fixation, 0.1)

    def test_a_half_finished_fold_scores_below_the_gate(self):
        """Carried halfway over scores 0.75 exactly, which is why the gate is above it.

        The arithmetic is worth stating: a particle taken half the distance to
        its mirror sits at half the initial separation, so the score is 1 - 0.5.
        A gate at 0.75 would pass a fold left standing open like a tent.
        """
        half = self.rest.copy()
        half[self.moving, 1] = self.MID - (half[self.moving, 1] - self.MID) * 0.5
        score, _, _ = self.score(half)
        self.assertAlmostEqual(score, 0.75, places=6)
        self.assertLess(score, get_atom("fold_score").calibration.gate)

    def test_layer_forgiveness_forgives_exactly_the_named_separation(self):
        """A flap landing two thicknesses up needs two thicknesses forgiven.

        The mechanism behind ``FoldStage.layer_forgiveness``: the evaluator
        multiplies its per-sheet separation by the crease's own factor before
        handing it to ``cloth_fold_error``. A hem folded onto an already-folded
        stack rests higher than a flap on bare cloth, and under-forgiving that
        reads pure bookkeeping as placement error.
        """
        lift = 0.004
        folded = self.rest.copy()
        folded[self.moving, 1] = 2 * self.MID - folded[self.moving, 1]
        folded[self.moving, 2] += 2.0 * lift  # onto a stack, not onto bare cloth
        self.probe.positions = folded
        under, group, _ = cloth_fold_error(self.ctx, "c", self.pairs, layer_thickness=lift)
        exact, _, _ = cloth_fold_error(self.ctx, "c", self.pairs, layer_thickness=2.0 * lift)
        self.assertAlmostEqual(group, 2.0 * lift, places=6)
        self.assertLess(under, 0.99, "one thickness forgiven cannot absorb a two-thickness landing")
        self.assertGreater(exact, 0.99, "the full stack height forgiven, the fold is perfect again")


class TestThreeFoldMetric(unittest.TestCase):
    """The shirt's three-crease scoring, pinned on a synthetic sheet.

    A 0.4 m grid folded to the SpeedFolding template -- outer quarters in
    across y, then the hem quarter up across x -- with each crease's sets built
    by the suite's **rectangle rule**: ``moving`` and ``base`` clipped to the
    region no other fold moves. Two facts are pinned, because the shirt spec's
    correctness rests on exactly these:

    * the three per-crease scores survive each other's folds (the mirror pairs
      commute -- a y-mirror pair shares its x, so the hem fold carries both
      members together);
    * *without* the x-clip, the hem fold's legitimate motion reads as the side
      folds' drag and destroys their scores -- the failure the rectangle rule
      exists to prevent, replayed in miniature.
    """

    SIZE = 0.40
    CELLS = 21
    Y_CREASE = 0.1
    X_CREASE = -0.1
    X_LAND = 0.0
    LIFT = 0.001

    def setUp(self):
        xs = np.linspace(-self.SIZE / 2, self.SIZE / 2, self.CELLS)
        gx, gy = np.meshgrid(xs, xs, indexing="ij")
        self.rest = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
        r = self.rest
        yc, xc, xl = self.Y_CREASE, self.X_CREASE, self.X_LAND
        clipped = r[:, 0] > xc  # the region the hem fold does not move
        band = np.abs(r[:, 1]) <= yc  # the band the side folds do not move
        masks = {
            "far_side": (r[:, 1] > yc) & clipped,
            "far_body": (r[:, 1] >= 0) & (r[:, 1] <= yc) & clipped,
            "near_side": (r[:, 1] < -yc) & clipped,
            "near_body": (r[:, 1] >= -yc) & (r[:, 1] <= 0) & clipped,
            "hem_panel": (r[:, 0] < xc) & band,
            "hem_body": (r[:, 0] >= xc) & (r[:, 0] <= xl) & band,
            # The half-planes the rectangle rule replaces, kept to demonstrate
            # what they cost (see the double-charge test).
            "far_side_unclipped": r[:, 1] > yc,
            "far_body_unclipped": (r[:, 1] >= 0) & (r[:, 1] <= yc),
        }
        model = types.SimpleNamespace(particle_count=len(r))
        sets = {"c": np.arange(len(r))}
        sets.update({f"c/{name}": np.flatnonzero(mask) for name, mask in masks.items()})
        self.probe = ParticleProbe(model, sets)
        self.probe.positions = self.rest.copy()
        self.probe.velocities = np.zeros_like(self.rest)
        self.probe.baseline = self.rest.copy()
        self.ctx = types.SimpleNamespace(probe=lambda _name: self.probe)
        self.folds = (
            ("far_side", "far_body", 1, yc),
            ("near_side", "near_body", 1, -yc),
            ("hem_panel", "hem_body", 0, xc),
        )
        self.pairs = {
            moving: mirror_pairs(self.probe, f"c/{moving}", f"c/{base}", axis=axis, origin=origin)
            for moving, base, axis, origin in self.folds
        }

    def _folded_positions(self) -> np.ndarray:
        """All three creases applied as perfect reflections, in template order.

        The physical folds move **everything** past their crease -- including
        the slivers the scoring rectangles exclude -- which is the whole point:
        the scored sets are a subset of what actually moves.
        """
        r = self.rest
        p = r.copy()
        left = r[:, 1] > self.Y_CREASE
        p[left, 1] = 2 * self.Y_CREASE - p[left, 1]
        p[left, 2] += self.LIFT
        right = r[:, 1] < -self.Y_CREASE
        p[right, 1] = -2 * self.Y_CREASE - p[right, 1]
        p[right, 2] += self.LIFT
        hem = r[:, 0] < self.X_CREASE  # x untouched by the side folds
        p[hem, 0] = 2 * self.X_CREASE - p[hem, 0]
        p[hem, 2] += 2 * self.LIFT
        return p

    def test_the_three_creases_survive_each_other(self):
        """Every per-crease score stays near-perfect after all three folds."""
        self.probe.positions = self._folded_positions()
        for moving, base, _, _ in self.folds:
            score, _, fixation = cloth_fold_error(
                self.ctx, "c", self.pairs[moving], moving=moving, base=base, layer_thickness=2 * self.LIFT
            )
            self.assertGreater(score, 0.97, f"{moving}: a perfect fold sequence must score as one")
            self.assertLess(fixation, 1e-9, f"{moving}: a clipped base is out of every other fold's way")

    def test_an_unclipped_base_double_charges_the_hem_fold(self):
        """The same perfect folds, scored with half-plane sets: the first
        crease collapses, because the hem fold legitimately moved the hem-ward
        sliver of its base and the drag term charges the side fold for it."""
        unclipped = mirror_pairs(
            self.probe, "c/far_side_unclipped", "c/far_body_unclipped", axis=1, origin=self.Y_CREASE
        )
        self.probe.positions = self._folded_positions()
        score, _, fixation = cloth_fold_error(
            self.ctx,
            "c",
            unclipped,
            moving="far_side_unclipped",
            base="far_body_unclipped",
            layer_thickness=2 * self.LIFT,
        )
        self.assertGreater(fixation, 0.01, "the hem fold's motion must show up as base drift")
        self.assertLess(score, get_atom("fold_score").calibration.gate, "the double-charge must be visible at the gate")
