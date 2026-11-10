# SPDX-License-Identifier: Apache-2.0
"""The RTX backend stops the driver trimming its shader cache, unless the caller chose otherwise."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from robosimstudio.solvers.render import rtx

TIER = "fast"


class DriverShaderCacheTest(unittest.TestCase):
    def test_sets_skip_cleanup_when_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(rtx.SHADER_CACHE_SKIP_CLEANUP_ENV, None)
            rtx.keep_driver_shader_cache()
            self.assertEqual(os.environ[rtx.SHADER_CACHE_SKIP_CLEANUP_ENV], "1")

    def test_a_callers_choice_wins(self) -> None:
        with mock.patch.dict(os.environ, {rtx.SHADER_CACHE_SKIP_CLEANUP_ENV: "0"}):
            rtx.keep_driver_shader_cache()
            self.assertEqual(os.environ[rtx.SHADER_CACHE_SKIP_CLEANUP_ENV], "0")


if __name__ == "__main__":
    unittest.main()
