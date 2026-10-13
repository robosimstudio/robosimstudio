# SPDX-License-Identifier: Apache-2.0
"""Every clip a README shows is filmed by a take, or kept for a reason that is written down.

``scripts/recording/takes.py`` is the table ``scripts/record_examples.py``
films from: each example's own page, driven by a cursor. A script added without
a take is a README whose clip nobody can remake; a take that names a script
which moved writes nothing; a clip in ``media/`` that no take films and no
reason keeps is one nobody can say the provenance of. All three are table
facts, read here without the library, a GPU or a browser.
"""

from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path

TIER = "fast"
"""Reads the take table and lists files; imports nothing of the library."""

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from recording.takes import KEPT, TAKES  # noqa: E402

NUMBERED = re.compile(r"^\d\d_\w+$")


def scripts() -> list[Path]:
    """The scripts a clip is owed to: ``examples/NN_x/NN_y.py``, and ``demo/<name>/<name>.py``."""
    found = [p for p in (REPO / "examples").glob("[0-9][0-9]_*/[0-9][0-9]_*.py") if NUMBERED.match(p.parent.name)]
    found += [p for p in (REPO / "demo").glob("*/*.py") if p.stem == p.parent.name]
    return sorted(found)


def key(path: Path) -> str:
    """How the table names a script's clip or a file in a ``media/`` folder: ``02_cloth/00_towel``."""
    folder = path.parent.parent if path.parent.name == "media" else path.parent
    return f"{folder.relative_to(REPO).as_posix().removeprefix('examples/')}/{path.stem}"


class TestTheTakeTable(unittest.TestCase):
    def test_every_take_names_a_script_that_is_there(self) -> None:
        """A pending take may name a script another branch has not landed yet, if its whole folder is new."""
        for take in TAKES:
            with self.subTest(take=take.key):
                script = REPO / take.example
                if take.pending and not (REPO / Path(take.example).parts[0]).is_dir():
                    continue
                self.assertTrue(script.is_file(), f"{take.example} is not in this tree")

    def test_every_clip_lands_in_a_media_folder(self) -> None:
        for take in TAKES:
            with self.subTest(take=take.key):
                parts = take.media.relative_to(REPO).parts
                self.assertEqual("media", parts[-2])
                self.assertEqual(".webp", take.media.suffix)
                inside_examples = parts[0] == "examples" and len(parts) == 4 and NUMBERED.match(parts[1])
                inside_demo = parts[0] == "demo" and len(parts) == 4
                self.assertTrue(
                    inside_examples or inside_demo, f"{take.media} is not in examples/NN_*/media or demo/*/media"
                )

    def test_no_two_takes_write_one_clip(self) -> None:
        keys = [take.key for take in TAKES]
        self.assertEqual(sorted(set(keys)), sorted(keys))
        self.assertFalse(set(keys) & set(KEPT), "a clip is both filmed and kept")

    def test_every_kept_clip_says_why(self) -> None:
        for name, reason in KEPT.items():
            with self.subTest(kept=name):
                self.assertGreater(len(reason.split()), 4, "a reason is a sentence")

    def test_every_script_has_a_take_or_a_reason(self) -> None:
        owed = {take.key for take in TAKES} | set(KEPT)
        for script in scripts():
            with self.subTest(script=str(script.relative_to(REPO))):
                self.assertTrue(
                    key(script) in owed, f"{key(script)}: add a take to scripts/recording/takes.py, or a KEPT reason"
                )

    def test_every_clip_on_disk_is_filmed_or_kept(self) -> None:
        owed = {take.key for take in TAKES} | set(KEPT)
        clips = sorted([*REPO.glob("examples/*/media/*.webp"), *REPO.glob("demo/*/media/*.webp")])
        self.assertTrue(clips)
        for clip in clips:
            with self.subTest(clip=str(clip.relative_to(REPO))):
                self.assertTrue(key(clip) in owed, f"{key(clip)}: no take films this clip and no reason keeps it")

    def test_reading_the_table_imports_nothing_heavy(self) -> None:
        """The table is read here and by ``--list`` on machines with no GPU and no browser."""
        heavy = ("robosimstudio", "robosimstudio_studio", "playwright", "PIL", "numpy", "viser", "warp", "newton")
        probe = (
            f"import sys; sys.path.insert(0, {str(REPO / 'scripts')!r}); import recording.takes; "
            f"print(sorted({{m.split('.')[0] for m in sys.modules}} & set({heavy!r})))"
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=True)
        self.assertEqual("[]", done.stdout.strip())


if __name__ == "__main__":
    unittest.main()
