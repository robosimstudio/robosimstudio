# SPDX-License-Identifier: Apache-2.0
"""A body chain's manifest record names every segment, not just the first.

A cable is fifty capsule bodies and ``EntityHandle.body`` is the first of
them -- the one welded in the anchor. With only that index on the tape, every
reader downstream measured a capsule that never moves: the cable "never
moved", no grasp "met" it, and its containment in a seat was the anchor's.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import newton
import numpy as np

from robosimstudio.record.capture import capture_manifest
from robosimstudio.scene.handles import EntityHandle

TIER = "fast"


def _env(objects: dict[str, EntityHandle], shape_type: np.ndarray, shape_scale: np.ndarray):
    handles = SimpleNamespace(
        objects=objects,
        statics={},
        arms={},
        work_arm="",
        support_arm="",
        deformables={},
        regions={},
        surface_z=0.75,
        robot_shapes=[],
        segmentation_names=lambda: {},
    )
    task = SimpleNamespace(
        full_name="suite/task",
        scene="bench",
        embodiment="arm",
        gripper="",
        instruction="",
        physics=SimpleNamespace(control_hz=60.0),
    )
    model = SimpleNamespace(
        shape_type=SimpleNamespace(numpy=lambda: shape_type),
        shape_scale=SimpleNamespace(numpy=lambda: shape_scale),
    )
    return SimpleNamespace(handles=handles, task=task, model=model, episode=None, doc=None)


class TestChainManifest(unittest.TestCase):
    def test_chain_records_every_body_and_its_segment_radius(self):
        shapes = np.arange(10, 14, dtype=np.int32)
        shape_type = np.full(20, int(newton.GeoType.BOX), dtype=np.int32)
        shape_type[shapes] = int(newton.GeoType.CAPSULE)
        shape_scale = np.zeros((20, 3), dtype=np.float32)
        shape_scale[shapes] = (0.0035, 0.006, 0.0)
        cable = EntityHandle(name="cable", body=5, bodies=(5, 6, 7, 8), shapes=shapes)
        box = EntityHandle(name="box", body=3, shapes=np.array([2], dtype=np.int32))

        entities = capture_manifest(_env({"cable": cable, "box": box}, shape_type, shape_scale))["entities"]

        self.assertEqual(entities["cable"]["body"], 5)
        self.assertEqual(entities["cable"]["bodies"], [5, 6, 7, 8])
        self.assertAlmostEqual(entities["cable"]["segment_radius"], 0.0035, places=6)
        # The ordinary one-body entity is unchanged: no chain, no radius.
        self.assertEqual(entities["box"]["bodies"], [])
        self.assertIsNone(entities["box"]["segment_radius"])

    def test_chain_of_non_round_segments_has_no_radius(self):
        shapes = np.arange(0, 3, dtype=np.int32)
        shape_type = np.full(3, int(newton.GeoType.BOX), dtype=np.int32)
        chain = EntityHandle(name="links", body=0, bodies=(0, 1, 2), shapes=shapes)

        record = capture_manifest(_env({"links": chain}, shape_type, np.zeros((3, 3))))["entities"]["links"]

        self.assertEqual(record["bodies"], [0, 1, 2])
        self.assertIsNone(record["segment_radius"])


if __name__ == "__main__":
    unittest.main()
