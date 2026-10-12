# SPDX-License-Identifier: Apache-2.0
"""The tree is ruff-clean: ``ruff check`` and ``ruff format --check`` find nothing.

``pyproject.toml`` has configured ruff since the repository began, and until
2026-09-14 nothing ran it: 537 findings and 288 unformatted files had
accumulated behind a configuration everyone assumed was enforced. This is the
enforcement. What ruff may and may not say lives in ``[tool.ruff]`` -- every
ignore with its reason, every per-file exemption with its argument -- and
never here: a test that carried its own exceptions would be a second
configuration for the same tool.

Two subprocesses over the whole checkout, a few seconds each, no GPU.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

TIER = "fast"
"""Runs ruff; imports nothing of the library."""

REPO = Path(__file__).resolve().parents[1]


class TestTheTreeIsRuffClean(unittest.TestCase):
    def setUp(self) -> None:
        if importlib.util.find_spec("ruff") is None:
            self.skipTest("ruff is behind the [dev] extra: uv sync --extra dev")

    def _ruff(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "ruff", *args, "."],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )

    def test_ruff_check_finds_nothing(self) -> None:
        done = self._ruff("check", "--output-format", "concise")
        self.assertEqual(done.returncode, 0, "ruff check:\n" + done.stdout + done.stderr)

    def test_ruff_format_would_change_nothing(self) -> None:
        done = self._ruff("format", "--check")
        self.assertEqual(done.returncode, 0, "ruff format --check:\n" + done.stdout + done.stderr)


if __name__ == "__main__":
    unittest.main()
