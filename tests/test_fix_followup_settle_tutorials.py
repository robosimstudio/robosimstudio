# SPDX-License-Identifier: Apache-2.0
"""The 'your own kind' tutorial entities join the scene's one particle settle.

``on_scene_ready`` calls ``join_settle``, not ``snapshot_rest``: a tutorial thing
that only snapshots never lands on ``pipeline.particle_settles``, so a cloth or
soft body that settles after it leaves it with an unsettled rest to reset to.

No GPU: the sources are read, not imported.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

TIER = "fast"
"""Reads two example sources: no GPU."""

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = (
    ROOT / "examples" / "02_cloth" / "04_your_own_cloth.py",
    ROOT / "examples" / "03_soft" / "02_your_own_soft.py",
)


def _called(path: Path) -> set[str]:
    """The ``self.<name>(...)`` calls inside each ``on_scene_ready`` in ``path``."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == "on_scene_ready":
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                    names.add(call.func.attr)
    return names


class TestTutorialsJoinSettle(unittest.TestCase):
    def test_each_tutorial_joins_and_does_not_snapshot_alone(self) -> None:
        for path in TUTORIALS:
            with self.subTest(path=path.name):
                called = _called(path)
                self.assertIn("join_settle", called)
                self.assertNotIn("snapshot_rest", called)


if __name__ == "__main__":
    unittest.main()
