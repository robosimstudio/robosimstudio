# SPDX-License-Identifier: Apache-2.0
"""Every example folder's README shows its scripts, as their docstrings say them.

``scripts/example_readmes.py`` writes one section per numbered script into its
folder's README: the title, the clip, what it shows and the line that runs it.
A docstring edited without a rerun, a script added without one, a clip left
in ``media/`` that no page shows, or a page showing a clip that is not there is
a README that has stopped being true.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

TIER = "fast"
"""No GPU: reads files."""

REPO = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("example_readmes", REPO / "scripts" / "example_readmes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheExampleReadmes(unittest.TestCase):
    def test_every_readme_is_what_its_scripts_say(self) -> None:
        self.assertEqual(0, _tool().main(["--check"]), "run scripts/example_readmes.py")

    def test_no_page_shows_a_clip_that_is_not_there(self) -> None:
        self.assertEqual([], _tool().dangling_media())

    def test_a_page_showing_a_missing_clip_is_refused(self) -> None:
        """The check itself, on a tree of one page and one clip, so it is known to bite."""
        tool = _tool()
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "examples" / "00_x"
            (folder / "media").mkdir(parents=True)
            (folder / "media" / "here.webp").write_bytes(b"")
            (folder / "README.md").write_text("![a](media/here.webp)\n![b](media/gone.webp)\n", encoding="utf-8")
            tool.REPO, tool.EXAMPLES = Path(root), Path(root) / "examples"
            self.assertEqual(["examples/00_x/README.md -> media/gone.webp"], tool.dangling_media())

    def test_every_demo_shows_its_clip(self) -> None:
        """The demos are what a visitor from the project page looks for: each one is seen running."""
        demos = REPO / "examples" / "09_demos"
        readme = (demos / "README.md").read_text(encoding="utf-8")
        for script in sorted(demos.glob("[0-9][0-9]_*.py")):
            with self.subTest(script=script.name):
                self.assertTrue((demos / "media" / f"{script.stem}.webp").is_file(), "no clip in media/")
                self.assertIn(f"(media/{script.stem}.webp)", readme)
