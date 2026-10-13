# SPDX-License-Identifier: Apache-2.0
"""A batched segmentation against a single one, on the real weights.

A stub can check the batching *contract* -- how many passes a group costs,
whose answer is whose, what a differing threshold does -- and cannot check the one thing a wave's parity bar actually rests on:
that a world's answer is the same answer it would have got alone.

It is not bit-identical and does not need to be. SAM3 is query-based: it returns
its full set of queries every time, and ``SAM3_FLOOR`` is 0.0, so a caller that
states no floor gets two hundred detections of which all but one are noise.
Batched, the model's matmuls reduce in a different order, and in bfloat16 that is
enough to reorder the noise and to move a score by one representable step.

So the claim is stated where it means something -- at a floor a graph would
actually use. Measured on three images at 0.3 and 0.5: same detection count,
top mask IoU 1.000000, top box within a pixel, top score identical in two of
three worlds and one bfloat16 ulp out in the third.

Skipped without the weights, which are a gated repository.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio_policy_agent import perception_tools as P

TIER = "medium"
"""A GPU and the SAM3 weights. Runs per PR, not per commit."""

THRESHOLD = 0.5
"""A floor a graph would use. Below any floor the answer is noise -- see above."""


def _block(seed: int) -> np.ndarray:
    """A dark frame with one red block somewhere in it."""
    frame = np.full((240, 320, 3), 30, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    y, x = int(rng.integers(40, 180)), int(rng.integers(40, 260))
    frame[y : y + 50, x : x + 50] = (220, 40, 40)
    return frame


@unittest.skipUnless(P.sam3_available(), "the SAM3 stack is not importable")
class TestABatchedAnswerIsTheAnswerAloneToo(unittest.TestCase):
    """Three worlds looking at once, against the same three looking one at a time."""

    @classmethod
    def setUpClass(cls) -> None:
        asks = [P.SegmentAsk(image=_block(s), query="red block", threshold=THRESHOLD) for s in (1, 2, 3)]
        cls.batched = P.segment_many(asks)
        cls.alone = [P.segment_many([ask])[0] for ask in asks]

    def test_every_world_finds_the_same_number_of_things(self):
        for world, (group, single) in enumerate(zip(self.batched, self.alone, strict=True)):
            with self.subTest(world=world):
                self.assertEqual(len(group["scores"]), len(single["scores"]))
                self.assertGreater(len(group["scores"]), 0, "the block was not found at all")

    def test_the_masks_are_the_same_pixels(self):
        """The mask is what a graph grounds a grasp on, so this is the real claim."""
        for world, (group, single) in enumerate(zip(self.batched, self.alone, strict=True)):
            with self.subTest(world=world):
                union = np.logical_or(group["masks"][0], single["masks"][0]).sum()
                overlap = np.logical_and(group["masks"][0], single["masks"][0]).sum()
                self.assertEqual(int(overlap), int(union), "the batched mask is not the single one")

    def test_the_boxes_agree_to_a_pixel(self):
        for world, (group, single) in enumerate(zip(self.batched, self.alone, strict=True)):
            with self.subTest(world=world):
                worst = max(abs(a - b) for a, b in zip(group["boxes"][0], single["boxes"][0], strict=True))
                self.assertLessEqual(worst, 1.0, f"{worst:.3f} px apart")

    def test_the_score_is_within_a_bfloat16_step(self):
        """Not equality: the batch reduces its matmuls in a different order.

        One step at this magnitude is ~4e-3, which is what bfloat16 can express
        there -- and it moves no mask, no box and no count.
        """
        for world, (group, single) in enumerate(zip(self.batched, self.alone, strict=True)):
            with self.subTest(world=world):
                self.assertAlmostEqual(group["scores"][0], single["scores"][0], delta=8.0e-3)


if __name__ == "__main__":
    unittest.main()
