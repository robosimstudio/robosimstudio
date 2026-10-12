# SPDX-License-Identifier: Apache-2.0
"""A robot whose module is not installed is simply not a robot here.

``EMBODIMENTS`` and ``GRIPPERS`` find their entries by scanning a package, not
by naming modules, so a robot is one file and leaving that file out is the whole
of removing it. Two of them are left out of some distributions for a reason that
has nothing to do with code -- the ``droid`` and ``ur5`` descriptions are
upstream geometry this repository may not be able to pass on -- and the claim
that makes that safe is the one checked here: with those two files gone, the
registry loads, lists what is left, and every robot it lists still finds its
hand.

What would break it is one embodiment module importing another by name, or
anything in the library importing either at module scope. The first is a real
relationship in one place -- the two composed YAMs share ``yam_look`` -- which
is why the modules left out are named here rather than "any one of them".

Run in a subprocess over a copy of the package's directory with the files
removed, because a registry loads once per process and an import blocker would
not do: the scan lists *files*, and a file that is there and refused is an
error, where a file that is absent is nothing at all.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

TIER = "fast"
"""No GPU, no scene. Gates every commit."""

REPO = Path(__file__).resolve().parents[1]
EMBODIMENTS = REPO / "robosimstudio" / "robots" / "embodiments"

LEFT_OUT = ("droid", "ur5")
"""Robot modules a distribution may omit, with the assets they import."""

REPORT = textwrap.dedent(
    """
    import json
    import sys

    import robosimstudio.robots.embodiments as package

    package.__path__[:] = [sys.argv[1]]

    from robosimstudio.robots.registry import GRIPPERS, get_embodiment, hands_of, list_embodiments, resolve_gripper

    names = list_embodiments()
    hands = {name: hands_of(get_embodiment(name)) for name in names}
    for name in names:
        for arm in get_embodiment(name).arms:
            assert resolve_gripper(get_embodiment(name), arm) is not None, (name, arm.name)
    try:
        get_embodiment("droid")
        miss = ""
    except KeyError as error:
        miss = str(error)
    print(json.dumps({"robots": names, "hands": hands, "grippers": GRIPPERS.names(), "miss": miss,
                      "loaded": sorted(m for m in sys.modules if m.startswith(package.__name__ + "."))}))
    """
).strip()


class TestARobotModuleMayBeAbsent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with TemporaryDirectory() as tmp:
            for path in sorted(EMBODIMENTS.glob("*.py")):
                if path.stem not in LEFT_OUT:
                    shutil.copy(path, tmp)
            cls.result = subprocess.run(
                [sys.executable, "-c", REPORT, tmp],
                cwd=REPO,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )
        lines = [line for line in cls.result.stdout.splitlines() if line.startswith("{")]
        cls.got = json.loads(lines[-1]) if cls.result.returncode == 0 and lines else None

    def setUp(self) -> None:
        self.assertEqual(0, self.result.returncode, f"the robots need a module left out:\n{self.result.stderr[-3000:]}")
        self.assertIsNotNone(self.got, f"the probe printed nothing:\n{self.result.stdout[-2000:]}")

    def test_the_rest_are_listed_and_the_absent_are_not(self) -> None:
        for name in ("panda", "panda_robotiq", "yam_bimanual", "yam_i2rt"):
            self.assertIn(name, self.got["robots"])
        for name in LEFT_OUT:
            self.assertNotIn(name, self.got["robots"])
            self.assertNotIn(f"robosimstudio.robots.embodiments.{name}", self.got["loaded"])

    def test_every_robot_listed_still_finds_its_hand(self) -> None:
        """A hand is looked up by name at build, so a robot whose hand lived in
        a module that left with another robot would list and then fail to build."""
        for name, hands in self.got["hands"].items():
            with self.subTest(robot=name):
                self.assertTrue(hands and all(hands), f"{name} lost a hand")
                self.assertLessEqual(set(hands), set(self.got["grippers"]))

    def test_asking_for_an_absent_robot_says_what_is_here(self) -> None:
        self.assertIn("unknown embodiment 'droid'", self.got["miss"])
        self.assertIn("panda", self.got["miss"])
        self.assertIn("where it is installed", self.got["miss"], "the hint must read right without the benchmark tier")


if __name__ == "__main__":
    unittest.main()
