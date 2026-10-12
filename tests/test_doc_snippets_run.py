# SPDX-License-Identifier: Apache-2.0
"""The snippet each page opens with actually runs, and prints what it says.

Two of these were wrong within an hour of being written. The first paired a cube
with a rope, which is precisely the combination the composer warns about -- the
cube does not fall and the snippet's own comment was a number it never produces.
The second transcribed an x from a different example. Both were caught by running
them, and neither would have been caught by anything else: a documentation test
that checks links and module names says nothing about whether the code works.

So the **opening** snippet of each page is executed. Only the opening one, and
deliberately: it is the one a reader types first and the only one that has to
stand alone. Everything further down is a fragment about a seam, and a test that
demanded every fenced block be runnable would push those toward being padded out
with imports nobody needs to read.

Viewer calls are stripped rather than the snippet being written around them.
``rss.view(sim)`` belongs in a quickstart -- it is the point -- and opening a
browser page belongs nowhere in a test suite.
"""

from __future__ import annotations

import functools
import re
import unittest
from pathlib import Path

TIER = "medium"
"""Compiles and steps a real scene, because that is what the snippets do."""

REPO = Path(__file__).resolve().parents[1]

CORE_API = REPO / "docs/core_api.md"
"""The development tree's long guide. A release ships the site's pages instead."""

README = REPO / "README.md"
"""A README may carry the hello world or only point at it; it is held to running when it carries one."""

PAGES = tuple(
    page
    for page in (
        CORE_API,
        README,
        REPO / "docs/source/getting_started.md",
        REPO / "docs/source/physics.md",
        REPO / "docs/source/studio.md",
        REPO / "docs/source/notes/conventions.md",
    )
    if page.is_file() and (page != README or "```python" in page.read_text())
)
"""Pages whose first ``python`` block is a standalone, runnable example: every
one of these that the tree has."""

API_PAGES = tuple(page for page in (CORE_API, REPO / "docs/source/getting_started.md") if page.is_file())
"""The pages that list methods rather than only demonstrating them."""

REFERENCE = REPO / "examples/live/REFERENCE.md"
"""The live page's reference; its frames section prints where a cube lands, though its first block is no scene."""

PRINTS = PAGES + tuple(page for page in (REFERENCE,) if page.is_file())
"""Pages whose ``print(cube.position)`` comments are held to the real number."""

RECEIVERS = ("sim", "scene", "cube", "rope", "towel", "chain")
"""Variable names the page's fences use for a ``Sim``, a ``Scene`` and a ``Ref``.

Named rather than inferred, because inferring the type of a variable in a
markdown fence is a static analysis this does not need: what the page calls
things is a fact about the page, and a new name here is a two-word edit."""

OPENS_A_WINDOW = ("rss.view(", "rss.draw(")
"""Calls that would serve a browser page. Stripped, not forbidden."""


def _first_python_block(page: Path) -> str:
    blocks = re.findall(r"```python\n(.*?)```", page.read_text(), flags=re.S)
    return blocks[0] if blocks else ""


@functools.cache
def _ran(page: Path) -> dict:
    """The namespace a page's opening snippet leaves, executed once per process."""
    lines = [
        line for line in _first_python_block(page).splitlines() if not any(call in line for call in OPENS_A_WINDOW)
    ]
    namespace: dict = {}
    exec(compile("\n".join(lines), f"<{page.name}>", "exec"), namespace)
    return namespace


class TestTheOpeningSnippets(unittest.TestCase):
    def test_each_page_has_one(self) -> None:
        for page in PAGES:
            with self.subTest(page=page.name):
                self.assertTrue(_first_python_block(page).strip(), f"{page.name} opens with no python block")

    def test_each_one_runs(self) -> None:
        for page in PAGES:
            with self.subTest(page=page.name):
                try:
                    _ran(page)
                except Exception as exc:  # pragma: no cover -- the failure is the point
                    self.fail(f"{page.name}'s opening snippet raised {type(exc).__name__}: {exc}")

    def test_none_of_them_composes_a_scene_the_composer_warns_about(self) -> None:
        """The failure that made this file. A snippet may not demonstrate a
        combination the library itself reports as wrong -- and the library is the
        judge, so this cannot drift from what the composer actually says."""
        for page in PAGES:
            with self.subTest(page=page.name):
                sim = _ran(page).get("sim")
                self.assertIsNotNone(sim, f"{page.name}'s snippet binds no `sim`")
                self.assertEqual([], list(sim.pipeline.program.warnings), f"{page.name}'s snippet is a warned scene")


class TestTheApiTablesAreReal(unittest.TestCase):
    """Every method the page names exists on the object it names it for.

    A documentation page listing a method that was renamed is rot of exactly the
    kind ``test_docs_are_true`` catches for *modules* and cannot catch for
    *members*: ``sim.step_count`` and ``sim.steps`` are both plausible, only one
    is real, and prose does not raise.

    Checked against a live ``Sim`` rather than the class, because half of what a
    caller reaches for -- ``controllers``, ``handles``, ``refs`` -- is set in
    ``__init__`` and invisible to ``hasattr`` on the type. That is the whole
    reason this is a medium test rather than a grep.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075", name="doc")
        cube = scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0.0, 0.0, 0.20)))
        scene.add(rss.robot("panda"))
        cls.objects = {"scene": scene, "sim": scene.build(), "cube": cube}
        cls.objects["rope"] = cls.objects["towel"] = cls.objects["chain"] = cube

    @unittest.skipUnless(API_PAGES, "no page that lists the API is in this tree")
    def test_every_named_member_exists(self) -> None:
        for page in API_PAGES:
            with self.subTest(page=page.name):
                code = "\n".join(re.findall(r"```python\n(.*?)```", page.read_text(), flags=re.S))
                missing: list[str] = []
                for name in RECEIVERS:
                    target = self.objects[name]
                    for attr in sorted(set(re.findall(rf"\b{name}\.(\w+)", code))):
                        if not hasattr(target, attr):
                            missing.append(f"{name}.{attr} ({type(target).__name__})")
                self.assertEqual([], missing, f"{page.name} names members that do not exist: {missing}")


class TestThePrintedNumbersAreTheRealOnes(unittest.TestCase):
    """The comment beside ``print(cube.position)`` says where the cube lands.

    It is the first number a reader sees and the only evidence in either page
    that the thing works, and until now nothing compared it to anything. Both
    pages carried ``(0.520, 0.000, 0.775)`` for a while -- the z was right and
    the x had been copied from a different example, where the cube is offset
    80 mm along the bench. A snippet that runs and prints a number the comment
    contradicts is worse than one that does not run, because it looks checked.

    Matched on the tuple in the comment rather than on stdout, so the page keeps
    saying what it means to a human and the test reads the same characters. A
    page that shows a reset prints the cube back where it was put, so that pose
    is the other number a comment may carry.
    """

    PATTERN = re.compile(r"print\(cube\.position\)\s*#\s*\(([-\d. ,]+)\)")

    def test_the_comment_is_where_the_cube_actually_lands(self) -> None:
        import robosimstudio as rss

        scene = rss.Scene(cell="bench_075")
        cube = scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0, 0, 0.2)))
        sim = scene.build()
        start = [float(v) for v in cube.position]
        sim.run(150).sync()
        landed = [float(v) for v in cube.position]

        checked: set[str] = set()
        for page in PRINTS:
            for match in self.PATTERN.finditer(page.read_text()):
                stated = [float(v) for v in match.group(1).split(",")]
                checked.add(page.name)
                with self.subTest(page=page.name, stated=match.group(1)):
                    self.assertEqual(3, len(stated), f"{page.name}: expected three numbers, got {match.group(1)!r}")
                    # A page prints the cube where it comes to rest, or -- after a
                    # reset -- where it was put. Any other number is a number the
                    # scene never produces.
                    off = {
                        label: max(abs(said - is_) for said, is_ in zip(stated, real, strict=True))
                        for label, real in (("at rest", landed), ("as placed", start))
                    }
                    self.assertLess(
                        min(off.values()),
                        0.005,
                        f"{page.name} prints the cube at {stated}; it rests at {[round(v, 4) for v in landed]} "
                        f"and was placed at {[round(v, 4) for v in start]}",
                    )
        self.assertTrue(checked, "no page prints the cube's position any more; the pattern is stale")
        if README in PAGES:
            # Where the README carries the hello world, it is the page every reader starts on.
            self.assertIn("README.md", checked, "the README stopped printing the cube's position; the pattern is stale")


class TestNothingInTheNamespaceIsUndiscoverable(unittest.TestCase):
    """Every flat ``rss.`` name is written down somewhere a reader will look.

    The namespace is a curated table -- 51 names chosen one at a time -- and the
    cost of curating it is that adding one is easy and mentioning it is a
    separate act nobody is prompted to do. A name reachable only by reading
    ``__init__.py`` is a name that does not exist for anybody who did not write
    it.

    Deliberately weak: it asks that the name *appears*, not that it is explained
    well. A stronger check would be a style opinion, and this one is the useful
    half -- it fires on the case that actually happens, which is a new export
    nobody documented.
    """

    @unittest.skipUnless(
        CORE_API.is_file(),
        "docs/core_api.md is not in this tree; the site's API pages render every export from the table itself",
    )
    def test_every_export_appears_in_the_docs(self) -> None:
        import robosimstudio as rss

        pages = "\n".join(page.read_text() for page in PAGES)
        missing = sorted(name for name in rss._EXPORTS if name not in pages)
        self.assertEqual(
            [],
            missing,
            f"exported by rss and mentioned in neither {[p.name for p in PAGES]}: {missing}. "
            "core_api.md section 6b is the catch-all list.",
        )


if __name__ == "__main__":
    unittest.main()
