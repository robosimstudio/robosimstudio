# SPDX-License-Identifier: Apache-2.0
"""The rope is a pack: installed, it is whole; left out, the library is whole without it.

The ``rod`` kind -- cables, ropes, cords: the capsule chain, the Cosserat
centerline, their latch, their measures -- is one package on a domain root,
``examples/10_rope/rod/``, reached as ``robosimstudio.domains.rod``. A tree may
leave that directory out. This file is the three claims that make leaving it out
a *deletion* rather than a port:

1. **The library holds no rod code.** Everything that is the rod's lives in the
   pack and arrives by registering -- its entities, its measures, its material
   kind, its state layout, its arc-length ``point`` feature kind, its
   ``Centerline`` capability, how a foreign engine reads and builds its actor.
   What used to be the rod's and is really scenery -- the post -- lives in the
   ``static`` domain, and the pack reuses it.
2. **Without the pack, nothing is half there.** ``import robosimstudio`` works,
   every registry scan answers without it, ``rss.post`` still names an entity
   that exists, and the three flat names it provided are ``AttributeError``s
   that say the rope is not installed.
3. **With the pack, nothing moved for a caller.** The same names resolve to the
   same objects, whichever registry is asked first.

Claim 2 is measured in a subprocess whose domain roots are rebuilt without the
pack -- the real mechanism (``ROBOSIMSTUDIO_DOMAIN_ROOTS_ONLY``), not an import
hook: a registry *scan* lists directories, so a hook that refused the import of
a directory still on the path would test a state no release is ever in.

No GPU and no scene; the scenes that build and step without the pack are
``tests/test_scenes_without_the_rod.py``.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

TIER = "fast"
"""Text, AST and registries. No GPU, no scene."""

REPO = Path(__file__).resolve().parents[1]
LIBRARY = REPO / "robosimstudio"
PACK_ROOT = REPO / "examples" / "10_rope"
PACK = PACK_ROOT / "rod"
HAS_PACK = (PACK / "__init__.py").is_file()
"""Whether this checkout carries the pack. A release without it runs claims 1 and 2 only."""

ROD_NAMES = frozenset(
    {
        "Centerline",
        "CableDriveSpec",
        "CableLatchSpec",
        "EntityCable",
        "EntityPeg",
        "EntityPegRound",
        "EntityRod",
        "EntityRodSuperdex",
        "RodMaterialSpec",
        "RodParam",
        "RodSpec",
        "SpecCable",
        "_anchor_dofs",
        "_build_rod_actor",
        "_read_rod",
        "_teleport_rod",
        "_transported_frames",
        "anchor_dofs",
        "build_rod_actor",
        "hold_segment",
        "read_rod",
        "straightness",
        "teleport_rod",
        "transported_frames",
        "winding_turns",
    }
)
"""Identifiers that are the pack's. None may be defined, imported or referenced in the library."""

PACK_PATHS = ("robosimstudio.domains.rod", "10_rope/rod", "solvers.grasp.latch", "solvers/grasp/latch")
"""Spellings of a path into the pack (and of where its latch used to be). Searched as text,
docstrings included: a cross-reference to a module a release leaves out is a broken reference."""

ALLOWED_STRINGS: dict[tuple[str, str], str] = {
    ("__init__.py", "rod"): "the flat-name table: rss.rod is the pack's, and the table is how its absence is said",
    ("__init__.py", "RodSpec"): "the flat-name table",
    ("__init__.py", "RodMaterialSpec"): "the flat-name table",
    ("__init__.py", "robosimstudio.domains.rod"): "the flat-name table's key: which pack provides the three names",
    ("assetlib/semantics/library.py", "rod"): (
        "the geometric FEATURE kind -- a rigid span something hangs on or winds round (a peg, a hanger rail) -- "
        "which is not the rope and is read by threaded/encloses"
    ),
    ("domains/rigid/measures/geometry.py", "rod"): "the same feature kind, where its measures dispatch on it",
}
"""``(library file, string) -> why`` for the rod-ish strings the library still uses as values.

A ratchet in the shape ``test_no_domain_is_special`` uses: state the exemption
with its reason rather than loosening the rule. ``"rod"`` as a *value* anywhere
else is the library branching on the rope's domain or layout by name."""


def _library_files() -> list[Path]:
    return sorted(p for p in LIBRARY.rglob("*.py") if "__pycache__" not in p.parts)


def _docstring_ids(tree: ast.AST) -> set[int]:
    """Every string node that is prose: a docstring, or a bare string under a field."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            found.add(id(node.value))
    return found


class TestTheLibraryHoldsNoRodCode(unittest.TestCase):
    """Claim 1, read off the source: AST for code, text for paths."""

    def test_no_library_module_defines_imports_or_references_a_rod_name(self) -> None:
        offenders: list[str] = []
        for path in _library_files():
            where = path.relative_to(LIBRARY).as_posix()
            for node in ast.walk(ast.parse(path.read_text())):
                names: list[str] = []
                if isinstance(node, ast.Name):
                    names = [node.id]
                elif isinstance(node, ast.Attribute):
                    names = [node.attr]
                elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    names = [node.name]
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [part for alias in node.names for part in (alias.name.split(".")[-1], alias.asname or "")]
                    module = getattr(node, "module", "") or ""
                    names += [f"<import {m}>" for m in (module, *(a.name for a in node.names)) if "domains.rod" in m]
                elif isinstance(node, ast.arg):
                    names = [node.arg]
                for name in names:
                    if name in ROD_NAMES or name.startswith("<import"):
                        offenders.append(f"{where}:{getattr(node, 'lineno', 0)}: {name}")
        self.assertEqual(
            [],
            offenders,
            "the library names something that is the rod pack's. It belongs in examples/10_rope/rod/ and arrives by "
            "registering -- see the module docstring for the seams.",
        )

    def test_a_rod_string_is_a_value_only_where_it_is_stated(self) -> None:
        offenders: list[str] = []
        used: set[tuple[str, str]] = set()
        for path in _library_files():
            where = path.relative_to(LIBRARY).as_posix()
            tree = ast.parse(path.read_text())
            prose = _docstring_ids(tree)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Constant) and isinstance(node.value, str)) or id(node) in prose:
                    continue
                value = node.value
                if value == "rod" or value in ROD_NAMES or any(spelling in value for spelling in PACK_PATHS):
                    if (where, value) in ALLOWED_STRINGS:
                        used.add((where, value))
                    else:
                        offenders.append(f"{where}:{node.lineno}: {value!r}")
        self.assertEqual([], offenders, "a rod name used as a value in the library, outside the stated exemptions")
        self.assertEqual(
            sorted(ALLOWED_STRINGS),
            sorted(used),
            "an exemption that no longer applies is a line to delete, not to keep",
        )

    def test_no_library_text_points_into_the_pack(self) -> None:
        """Docstrings too. A Sphinx role or a path that names the pack is a reference a release
        without the pack cannot resolve, and it is how the next edit learns to import it."""
        offenders = []
        for path in _library_files():
            where = path.relative_to(LIBRARY).as_posix()
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                for spelling in PACK_PATHS:
                    if spelling in line and (where, "robosimstudio.domains.rod") != ("__init__.py", spelling):
                        offenders.append(f"{where}:{number}: {spelling}")
        self.assertEqual([], offenders, "library text naming a path inside the rod pack")

    def test_the_latch_is_not_a_library_file(self) -> None:
        self.assertFalse((LIBRARY / "solvers" / "grasp" / "latch.py").exists())

    def test_the_post_is_the_static_domains(self) -> None:
        """``rss.post`` names an entity the ``static`` domain defines and registers itself."""
        from robosimstudio.domains.static import EntityPost, fixture_shape_cfg, post
        from robosimstudio.domains.static.entity import EntityStatic
        from robosimstudio.scene.registry import get_entity

        self.assertEqual("post", post(name="p").entity)
        self.assertIs(get_entity("post"), EntityPost)
        self.assertEqual("robosimstudio.domains.static.posts", EntityPost.__module__)
        self.assertTrue(issubclass(EntityPost, EntityStatic))
        self.assertTrue(EntityPost.claims_body, "a post is placed per world, so a board of them batches")
        self.assertEqual("robosimstudio.domains.static.posts", fixture_shape_cfg.__module__)


REPORT = """
import importlib, importlib.util, json
import robosimstudio as rss

out = {"pack_spec": importlib.util.find_spec("robosimstudio.domains.rod") is not None, "errors": {}}
for name in ("rod", "RodSpec", "RodMaterialSpec"):
    try:
        getattr(rss, name)
        out["errors"][name] = "resolved"
    except AttributeError as exc:
        out["errors"][name] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        out["errors"][name] = f"WRONG {type(exc).__name__}: {exc}"
out["hasattr"] = [n for n in out["errors"] if hasattr(rss, n)]
out["dir"] = [n for n in dir(rss) if n in out["errors"]]
broken = []
for name in sorted(rss._EXPORTS):
    try:
        getattr(rss, name)
    except Exception as exc:
        broken.append(f"{name}: {type(exc).__name__}: {exc}")
out["broken_core_names"] = broken

from robosimstudio.specs.physics import get_backend, list_domains
out["domains"] = list_domains()
from robosimstudio.assetlib.semantics import KIND_NAMES, list_kinds
from robosimstudio.assetlib.semantics.substrates import SUBSTRATES, is_substrate
out["rod_is_substrate"] = is_substrate("rod")
out["substrates"] = SUBSTRATES.names()
out["kinds"] = list_kinds()
out["library_kinds"] = sorted(KIND_NAMES)
from robosimstudio.engines.geometry import ACTOR_KINDS, builders_for
get_backend("superdex")
out["actor_kinds"] = ACTOR_KINDS.names()
out["superdex_builds"] = builders_for("superdex")
from robosimstudio.scene.registry import ENTITIES, get_entity
out["entities"] = ENTITIES.names()
out["post"] = get_entity("post").__module__ + "." + get_entity("post").__qualname__
from robosimstudio.domains import capabilities
out["capabilities"] = capabilities.CAPABILITIES.names()
out["has_centerline"] = hasattr(capabilities, "Centerline")
from robosimstudio.assetlib.materials import MATERIAL_KINDS
out["material_kinds"] = MATERIAL_KINDS.names()
out["generic"] = [domain for domain, _ in rss.GENERIC.backends]
out["generic_backends"] = [get_backend(spec.solver).alias for _, spec in rss.GENERIC.backends]
try:
    importlib.import_module("robosimstudio.solvers.grasp.latch")
    out["latch"] = "importable"
except ModuleNotFoundError:
    out["latch"] = "absent"
print(json.dumps(out))
"""


def _run(script: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", **env},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def _is_pack(child: Path) -> bool:
    """Whether ``child`` is a pack, as :func:`robosimstudio.roots.extend_path` reads a root.

    A package or one module, under a name that is an identifier. What a moved
    pack leaves behind -- a directory still holding its ``__pycache__`` and no
    ``__init__.py`` -- is neither, and a link to it would import as an empty
    namespace package under the name the pack used to have.
    """
    if child.name.startswith(("_", ".")):
        return False
    if child.is_dir():
        return child.name.isidentifier() and (child / "__init__.py").is_file()
    return child.suffix == ".py" and child.stem.isidentifier()


def roots_without_the_pack(stack: ExitStack) -> dict[str, str]:
    """An environment whose domain roots are this checkout's, less the rod.

    The checkout's own roots are switched off and named again one by one, in
    the library's order (:data:`robosimstudio.roots.DEFAULT_DOMAIN_ROOT_NAMES`,
    each only if it is there). A root the rod is not on is used as it stands:
    the example folders, which hold ``cloth.py`` and ``soft.py``. The rod's own
    root is replaced by a scratch directory holding a link to every *other*
    pack on it -- so ``robosimstudio.domains.rod`` is as absent as it is in a
    release that never shipped the directory: not on any ``__path__``, not
    listed by a scan, not found by the import system.
    """
    from robosimstudio import roots

    kept: list[Path] = []
    for name in roots.DEFAULT_DOMAIN_ROOT_NAMES:
        root = REPO / name
        if not root.is_dir():
            continue
        if root != PACK_ROOT:
            kept.append(root)
            continue
        scratch = Path(stack.enter_context(TemporaryDirectory()))
        for child in sorted(root.iterdir()):
            if child.stem != PACK.name and _is_pack(child):
                (scratch / child.name).symlink_to(child, target_is_directory=child.is_dir())
        kept.append(scratch)
    return {
        roots.DOMAIN_ROOTS_ENV: os.pathsep.join(str(root) for root in kept),
        f"{roots.DOMAIN_ROOTS_ENV}_ONLY": "1",
    }


class TestWithoutThePack(unittest.TestCase):
    """Claim 2: a process that has never had the rod directory."""

    @classmethod
    def setUpClass(cls) -> None:
        with ExitStack() as stack:
            proc = _run(REPORT, roots_without_the_pack(stack))
        if proc.returncode != 0:
            raise AssertionError(f"the library did not come up without the rod pack:\n{proc.stderr[-4000:]}")
        cls.got = json.loads([line for line in proc.stdout.splitlines() if line.startswith("{")][-1])

    def test_the_pack_really_was_absent(self) -> None:
        self.assertFalse(self.got["pack_spec"], "the probe ran with the pack importable, so it proved nothing")
        self.assertNotIn("rod", self.got["domains"])

    def test_the_three_names_say_the_rope_is_not_installed(self) -> None:
        for name, message in self.got["errors"].items():
            with self.subTest(name=name):
                self.assertTrue(message.startswith("AttributeError: "), message)
                self.assertIn("is not installed here", message)
                self.assertIn("rope", message)
                self.assertIn(f"rss.{name}", message)
        self.assertEqual([], self.got["hasattr"], "hasattr(rss, name) must be False without the pack")
        self.assertEqual([], self.got["dir"], "dir(rss) must not list a name that does not resolve")

    def test_every_other_flat_name_still_resolves(self) -> None:
        self.assertEqual([], self.got["broken_core_names"])

    def test_every_registry_scan_answers_without_it(self) -> None:
        got = self.got
        self.assertLessEqual({"rigid", "static", "articulated"}, set(got["domains"]))
        self.assertFalse(got["rod_is_substrate"], "the arc-length layout is minted by the pack, not by the library")
        self.assertNotIn("rod", got["substrates"])
        self.assertEqual(got["library_kinds"], got["kinds"], "the feature kinds are exactly the library's own")
        self.assertNotIn("point", got["kinds"])
        self.assertIn("rod", got["kinds"], "the geometric rod FEATURE kind is not the rope and stays")
        self.assertEqual(["shell", "tet"], got["actor_kinds"])
        self.assertEqual(["shell", "tet"], got["superdex_builds"])
        self.assertEqual([], [name for name in got["entities"] if name.startswith("rod")])
        self.assertEqual([], got["capabilities"], "no pack here publishes a capability")
        self.assertFalse(got["has_centerline"])
        self.assertNotIn("rod", got["material_kinds"])
        self.assertEqual("absent", got["latch"])

    def test_the_post_and_the_default_arrangement_do_not_need_it(self) -> None:
        self.assertEqual("robosimstudio.domains.static.posts.EntityPost", self.got["post"])
        self.assertIn("post", self.got["entities"])
        self.assertNotIn("rod", self.got["generic"], "GENERIC names no backend for a domain a release may lack")
        self.assertEqual(len(self.got["generic"]), len(self.got["generic_backends"]), "every backend it names resolves")


FIRST_ASKS = {
    "the actor kinds": "from robosimstudio.engines.geometry import ACTOR_KINDS; print('rod' in ACTOR_KINDS.names())",
    "the superdex builders": (
        "from robosimstudio.engines.geometry import actor_builder; print(actor_builder('superdex', 'rod') is not None)"
    ),
    "the layouts": "from robosimstudio.assetlib.semantics.substrates import is_substrate; print(is_substrate('rod'))",
    "the feature kinds": "from robosimstudio.assetlib.semantics import is_registered_kind; print(is_registered_kind('point'))",
    "the capabilities": (
        "from robosimstudio.domains.capabilities import Centerline; print(Centerline.__module__.endswith('rod.centerline'))"
    ),
    "the flat namespace": "import robosimstudio as rss; print(rss.rod(name='r').entity == 'rod')",
}
"""One question each, asked as the first thing a fresh process does."""


@unittest.skipUnless(HAS_PACK, "this checkout does not carry the rod pack")
class TestWithThePack(unittest.TestCase):
    """Claim 3: installed, everything the pack provides is there -- whoever asks first."""

    def test_each_registry_finds_the_pack_when_it_is_asked_first(self) -> None:
        """Nothing in the library imports the pack, so each of these would answer
        differently before and after something else happened to import it --
        unless the lookup itself runs the scan. In a fresh process, so "first"
        is true of every one of them."""
        for what, script in FIRST_ASKS.items():
            with self.subTest(first_asked=what):
                proc = _run(script, {})
                self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
                self.assertEqual("True", proc.stdout.strip().splitlines()[-1], f"{what} did not find the rod pack")

    def test_the_names_are_the_packs_own_objects(self) -> None:
        import robosimstudio as rss
        from robosimstudio.domains import rod as pack

        self.assertIs(rss.rod, pack.rod)
        self.assertIs(rss.RodSpec, pack.RodSpec)
        self.assertIs(rss.RodMaterialSpec, pack.RodMaterialSpec)
        self.assertLessEqual({"rod", "RodSpec", "RodMaterialSpec"}, set(dir(rss)))

    def test_the_pack_reuses_the_static_post_rather_than_owning_it(self) -> None:
        from robosimstudio.domains import rod as pack
        from robosimstudio.domains.static import EntityPost, EntityStatic, fixture_shape_cfg
        from robosimstudio.scene.registry import get_entity

        self.assertIs(pack.EntityPeg, EntityPost, "the name the cable suites import is the static domain's class")
        self.assertIs(pack.fixture_shape_cfg, fixture_shape_cfg)
        self.assertIs(get_entity("post"), EntityPost)
        self.assertTrue(
            issubclass(pack.EntityPegRound, EntityStatic), "the spool is the pack's own fixture, on the same base"
        )

    def test_the_engine_seams_hold_the_packs_functions(self) -> None:
        from robosimstudio.domains.rod import actor

        from robosimstudio.engines.geometry import ACTOR_KINDS, actor_builder
        from robosimstudio.specs.physics import actor_consumes, get_backend

        get_backend("superdex")
        self.assertIs(ACTOR_KINDS.get("rod"), actor.read_rod)
        self.assertIs(actor_builder("superdex", "rod"), actor.build_rod_actor)
        self.assertEqual(tuple(actor.CONSUMES), actor_consumes("superdex", "rod"))
        self.assertEqual(["rod", "shell", "tet"], ACTOR_KINDS.names())

    def test_the_bridge_builds_only_what_its_bundle_ships(self) -> None:
        """The engine bundle's own table is the shell and the tet. The rod's builder
        is a function of the bridge, registered from the pack, not a method on it."""
        from robosimstudio.engines.superdex.bridge import SuperdexBridge

        self.assertEqual({"shell", "tet"}, set(SuperdexBridge.BUILDERS))
        for needed in ("actor_root", "contact_params", "scene_of"):
            self.assertTrue(
                callable(getattr(SuperdexBridge, needed, None)), f"a builder outside the bundle uses {needed}"
            )

    def test_the_latch_answers_to_the_name_content_declares(self) -> None:
        import importlib

        from robosimstudio.domains.rod import latch

        self.assertIs(importlib.import_module("robosimstudio.solvers.grasp.latch"), latch)
        self.assertEqual(PACK / "latch.py", Path(latch.__file__))

    def test_the_layout_and_its_kind_are_the_packs(self) -> None:
        from robosimstudio.assetlib.semantics import KIND_NAMES, get_kind
        from robosimstudio.assetlib.semantics.substrates import is_substrate
        from robosimstudio.domains import rod as pack

        self.assertTrue(is_substrate("rod"))
        self.assertEqual(("point",), pack.KIND_NAMES)
        self.assertNotIn("point", KIND_NAMES)
        self.assertEqual(("rod",), get_kind("point").substrates)


if __name__ == "__main__":
    unittest.main()
