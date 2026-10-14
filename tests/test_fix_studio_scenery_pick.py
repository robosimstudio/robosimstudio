# SPDX-License-Identifier: Apache-2.0
"""Toggling scenery re-reads the picker's shapes, so clicks hit what is drawn."""

from __future__ import annotations

import unittest
from unittest import mock

from robosimstudio_studio import session as session_mod
from robosimstudio_studio.session import StudioSession

TIER = "fast"


class SceneryInvalidatesPicker(unittest.TestCase):
    def test_apply_scenery_invalidates_the_picker(self):
        stub = mock.MagicMock()
        with (
            mock.patch.object(session_mod, "garment_visuals", return_value=()),
            mock.patch.object(session_mod, "scene_repaints_live", return_value=False),
        ):
            StudioSession.apply_scenery(stub)
        stub._set_scenery_visible.assert_called_once()
        stub.picker.invalidate.assert_called_once_with()

    def test_no_picker_is_fine(self):
        stub = mock.MagicMock()
        stub.picker = None
        with (
            mock.patch.object(session_mod, "garment_visuals", return_value=()),
            mock.patch.object(session_mod, "scene_repaints_live", return_value=False),
        ):
            StudioSession.apply_scenery(stub)


if __name__ == "__main__":
    unittest.main()
