# SPDX-License-Identifier: Apache-2.0
"""``robosimstudio.specs`` promises a flat re-export. This is the check.

The package docstring says "Every name is re-exported here, flat, so a caller
imports from ``robosimstudio.specs`` and never has to know which submodule a spec
lives in". It was false for **28 of 72** names -- ``ShapeSpec``, ``HostActor``
and the whole ``params`` vocabulary (``ui``, ``LIVE``, ``RECAPTURE``,
``REBUILD``) among them, which is exactly what a person annotating a new spec's
fields with change tiers reaches for first.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

TIER = "fast"

REPO = Path(__file__).resolve().parents[1]
CORE_API = REPO / "docs" / "core_api.md"
"""The development tree's guide to the core tier. A release documents the same namespace on its site instead."""
API_PAGES = REPO / "docs" / "source" / "api" / "robosimstudio"
"""The site's pages for the flat namespace, in a tree that has the site."""
DOCS_EXTENSION = REPO / "docs" / "source" / "_ext" / "rss_docs.py"

needs_core_api = unittest.skipUnless(
    CORE_API.is_file(), "docs/core_api.md is not in this tree; its API pages are docs/source/api/robosimstudio/"
)
"""Decorator for a claim about what that one page says."""
needs_docs_site = unittest.skipUnless(
    DOCS_EXTENSION.is_file(), "no docs site in this tree; the release documents the namespace in docstrings"
)
"""Decorator for a claim about the site's API pages."""


def _export_groups(source: str) -> list[tuple[str, list[str]]]:
    """``_EXPORTS`` as ``[(comment above the group, [flat name, ...]), ...]``, read from the source.

    The reading ``docs/source/_ext/rss_docs.py`` does to lay the namespace out
    on pages, restated because importing that module imports Sphinx: a comment
    line opens a group, and a comment directly under a comment continues it.
    """
    groups: list[tuple[str, list[str]]] = []
    inside = False
    for line in source.splitlines():
        if not inside:
            inside = line.startswith("_EXPORTS")
            continue
        if line.startswith("}"):
            break
        comment = re.match(r"^\s*#\s*(.+?)\s*$", line)
        if comment:
            if groups and not groups[-1][1]:
                groups[-1] = (f"{groups[-1][0]} {comment.group(1)}", [])
            else:
                groups.append((comment.group(1), []))
            continue
        entry = re.match(r'^\s*"(\w+)"\s*:', line)
        if entry:
            if not groups:
                groups.append(("Names", []))
            groups[-1][1].append(entry.group(1))
    return [group for group in groups if group[1]]


def _group_titles(source: str) -> dict[str, str]:
    """The extension's ``GROUP_TITLES`` -- group comment to page heading -- read without importing it."""
    for node in ast.parse(source).body:
        target = node.target if isinstance(node, ast.AnnAssign) else next(iter(getattr(node, "targets", [])), None)
        if getattr(target, "id", None) == "GROUP_TITLES":
            return ast.literal_eval(node.value)
    raise AssertionError(
        "docs/source/_ext/rss_docs.py no longer holds GROUP_TITLES; the pages are grouped some other way"
    )


class TestTheFacadeReExportsEverything(unittest.TestCase):
    """Both directions: nothing missing, and nothing named that cannot import."""

    @staticmethod
    def _declared() -> dict[str, str]:
        out: dict[str, str] = {}
        for path in sorted((REPO / "robosimstudio" / "specs").glob("*.py")):
            if path.name == "__init__.py":
                continue
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets):
                    for entry in node.value.elts:
                        out.setdefault(entry.value, path.name)
        return out

    def test_every_submodule_name_is_reachable_from_the_package(self) -> None:
        import robosimstudio.specs as specs

        declared = self._declared()
        missing = sorted(n for n in declared if n not in set(specs.__all__))
        self.assertEqual(
            missing,
            [],
            "the package docstring promises a flat re-export; these are only reachable through "
            + ", ".join(f"{n} ({declared[n]})" for n in missing),
        )

    def test_every_re_exported_name_actually_resolves(self) -> None:
        """``__all__`` listing a name it cannot import is worse than omitting it."""
        import robosimstudio.specs as specs

        unresolvable = [n for n in specs.__all__ if not hasattr(specs, n)]
        self.assertEqual(unresolvable, [], "named in __all__ and not importable")


class TestTheFlatNamespaceIsDocumented(unittest.TestCase):
    """Every ``rss.<name>`` is documented where this tree documents it.

    Two places, and a tree has one or both. The site's API pages
    (``docs/source/api/robosimstudio/``), in a tree that has the site, are generated
    from the ``_EXPORTS`` source one comment group at a time, so the claim there
    is about groups: each has a page. ``docs/core_api.md`` is the development
    tree's guide and is hand-written, so the claims there are about the page
    itself: its section 6b exists so that "nothing in it is undiscoverable" and
    lists the flat namespace one line each. It said **51 names** while
    ``_EXPORTS`` held **59**, and four of those appeared nowhere on the page. One
    of the four, ``Arrangement``, was a bare alias for ``ArrangementSpec``: a
    second spelling of one concept, in no example, task, tool, test or document.
    It is gone; the other three are documented.

    A hand-written count beside a table nobody regenerates is a claim that goes
    stale on the next commit, which is why it is checked here rather than
    counted again by the next reader.

    **The namespace is the one this checkout has.** ``_EXPORTS`` is the names
    every tree carries; a domain pack a tree may leave out (the rope) adds its
    own while it is installed, and ``dir(rss)`` is the two
    together. So the guide is held to ``dir(rss)``: a checkout with the pack
    documents the pack's names, and one without it must not.
    """

    @staticmethod
    def _names() -> list[str]:
        import robosimstudio as rss

        return [name for name in dir(rss) if name != "__version__"]

    def test_the_namespace_is_the_core_table_and_the_installed_packs(self) -> None:
        """Both answers, read off the import system: every core name is in
        ``dir(rss)`` in any tree, and a pack's names are there exactly when the
        pack is -- neither promised by a tree that lacks it nor dropped by one
        that has it."""
        import importlib.util

        import robosimstudio as rss

        names = set(self._names())
        self.assertLessEqual(set(rss._EXPORTS), names, "a core name is missing from dir(rss)")
        expected = set(rss._EXPORTS)
        for pack, (_, provided) in rss._PACKS.items():
            installed = importlib.util.find_spec(pack) is not None
            with self.subTest(pack=pack, installed=installed):
                self.assertEqual(set(provided) & names, set(provided) if installed else set())
            expected |= set(provided) if installed else set()
        self.assertEqual(names, expected, "dir(rss) holds a name neither the core table nor an installed pack provides")

    @needs_docs_site
    def test_every_core_name_is_on_an_api_page(self) -> None:
        """The site lays ``_EXPORTS`` out by its comment groups: a page says
        ``.. rss-flat-api:: <heading>``, and the extension maps a group's comment
        to that heading. A group whose heading no page asks for is a run of names
        that fell off the site without an error -- unless the index catches the
        unknown ones with ``:ungrouped:``, which it does for a comment the
        extension has no heading for and for nothing else."""
        import robosimstudio as rss

        groups = _export_groups((REPO / "robosimstudio" / "__init__.py").read_text())
        grouped = [name for _, names in groups for name in names]
        self.assertEqual(sorted(grouped), sorted(rss._EXPORTS), "the grouped reading and the table disagree")

        titles = _group_titles(DOCS_EXTENSION.read_text())
        pages = [path.read_text() for path in sorted(API_PAGES.glob("*.rst"))]
        self.assertTrue(pages, f"no API pages under {API_PAGES}")
        asked = {
            found.strip() for page in pages for found in re.findall(r"(?m)^\.\. rss-flat-api::[ \t]*(\S.*)$", page)
        }
        catches_unknown = any(re.search(r"(?m)^\.\. rss-flat-api::[ \t]*\n[ \t]+:ungrouped:", page) for page in pages)
        for comment, names in groups:
            heading = titles.get(comment)
            with self.subTest(group=comment):
                if heading is None:
                    self.assertTrue(catches_unknown, f"{names} sit under a comment no page documents")
                else:
                    self.assertIn(heading, asked, f"{names} are headed {heading!r} and no page asks for that group")
        self.assertLessEqual(asked, set(titles.values()), "a page asks for a group _EXPORTS does not have")

    @needs_core_api
    def test_the_documented_count_matches_the_namespace(self) -> None:
        doc = CORE_API.read_text()
        stated = re.search(r"The flat namespace is (\d+) names", doc)
        self.assertIsNotNone(stated, "core_api.md no longer states the namespace size")
        self.assertEqual(int(stated.group(1)), len(self._names()), "the stated count and the namespace disagree")

    @needs_core_api
    def test_every_flat_name_appears_on_the_page(self) -> None:
        doc = CORE_API.read_text()
        missing = [n for n in self._names() if not re.search(rf"`{n}`|\brss\.{n}\b", doc)]
        self.assertEqual(missing, [], "reachable as rss.<name> and documented nowhere")

    def test_every_flat_name_resolves(self) -> None:
        """PEP 562 means a typo in ``_EXPORTS`` fails only on first access.

        Walked over ``dir(rss)``, so a pack's names are resolved where the pack
        is installed and are not asked for where it is not."""
        import robosimstudio as rss

        broken = []
        for name in self._names():
            try:
                getattr(rss, name)
            except Exception as exc:
                broken.append(f"{name}: {type(exc).__name__}")
        self.assertEqual(broken, [], "named in the flat namespace and not importable")


if __name__ == "__main__":
    unittest.main()
