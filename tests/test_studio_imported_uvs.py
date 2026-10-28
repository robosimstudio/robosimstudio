"""An imported mesh that brought its own texture is drawn on its own UVs.

The studio box-projects UVs over a mesh only when it has none of its own that
fit its points; a SimReady asset's authored UVs are kept, so its texture lands
where the asset put it.
"""

from __future__ import annotations

import unittest

import numpy as np

from robosimstudio_studio.garments import authored_uvs

TIER = "fast"


class TestAnImportedTextureKeepsItsUVs(unittest.TestCase):
    POINTS = np.zeros((4, 3), dtype=np.float32)
    UVS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    def test_a_mesh_with_its_own_texture_keeps_its_uvs(self) -> None:
        data = {"points": self.POINTS, "uvs": self.UVS, "texture": np.zeros((2, 2, 3), dtype=np.uint8)}
        np.testing.assert_array_equal(self.UVS, authored_uvs(data))

    def test_a_mesh_without_its_own_texture_is_box_projected(self) -> None:
        self.assertIsNone(authored_uvs({"points": self.POINTS, "uvs": self.UVS, "texture": None}))
        self.assertIsNone(authored_uvs({"points": self.POINTS, "uvs": None, "texture": "albedo.png"}))

    def test_uvs_that_do_not_match_the_points_are_not_used(self) -> None:
        self.assertIsNone(authored_uvs({"points": self.POINTS, "uvs": self.UVS[:3], "texture": "albedo.png"}))


if __name__ == "__main__":
    unittest.main()
