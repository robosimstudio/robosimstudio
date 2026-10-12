# SPDX-License-Identifier: Apache-2.0
"""A domain pack may be one module, and the shipped ones are found wherever they lie.

``cloth`` and ``soft`` are one file each, in the example folder that runs it --
``examples/02_cloth/cloth.py`` beside ``00_towel.py``, ``examples/03_soft/soft.py``
beside ``00_sponge.py`` -- and the rope is a package, ``examples/10_rope/rod/``
beside ``00_cable.py``. They are still ``robosimstudio.domains.cloth``, ``.soft``
and ``.rod``. Four things have to hold for that to be true, and each fails
silently when it does not:

* **A root may hold a module, not only a package.** The import system and
  ``pkgutil.iter_modules`` already treat ``cloth.py`` and ``cloth/`` alike; the
  duplicate rule in :func:`robosimstudio.roots.extend_path` has to as well, or a
  stale package left on another root is shadowed without a word.
* **A root may hold example scripts beside its pack.** ``00_towel.py`` is on the
  package's ``__path__`` and is not a kind: the scan must not import it, and two
  folders each holding a ``00_...`` script of one name must not read as a clash.
* **A checkout has three default roots**, ``examples/02_cloth``,
  ``examples/03_soft`` and ``examples/10_rope``, each used only if it is there.
  A tree that ships without one must not notice.
* **A wheel has none of them.** It carries the packs merged into
  ``robosimstudio/_domain_packs`` -- each mapped pack by pack, the two modules
  as files and the rope as a directory, so no example script rides along -- and
  nothing but ``roots._Roots.packaged`` looks there: the path an editable
  checkout never exercises, which is how the vendored assets once shipped to a
  directory nothing read.

``tests/test_registry_discovery`` holds the claim these rest on (a kind on
somebody else's root is reached under our import name). This module needs only
the library, so it runs on a tree that has no benchmark tier.

Subprocesses throughout: ``__path__`` is extended when the package is first
imported, so every claim here is about what a fresh *process* sees.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent

from robosimstudio import roots

TIER = "fast"
"""No GPU. Imports the library in a subprocess."""

REPO = Path(__file__).resolve().parents[1]

KIND = dedent(
    '''
    """A kind of thing declared on somebody else's root."""
    from robosimstudio.scene.entity import Entity
    from robosimstudio.scene.registry import register_entity
    from robosimstudio.specs.physics import register_domain

    register_domain("probe_granular", state=("particles",), reads="cloth",
                    role="deformables", summary="A bed of grains.")

    @register_entity("probe_granular")
    class EntityProbeGranular(Entity):
        domain = "probe_granular"
    '''
).strip()

REPORT = dedent(
    """
    import json
    from robosimstudio.specs.physics import list_domains
    from robosimstudio.scene.registry import get_entity
    cls = get_entity("probe_granular")
    import robosimstudio.domains.probe_granular as kind
    print(json.dumps({"domains": list_domains(), "module": cls.__module__, "file": kind.__file__}))
    """
).strip()


def _run(script: str, *rootpaths: Path, cwd: Path = REPO, pythonpath: Path | None = None, only: bool = False):
    env = {
        **os.environ,
        roots.DOMAIN_ROOTS_ENV: os.pathsep.join(str(r) for r in rootpaths),
        f"{roots.DOMAIN_ROOTS_ENV}_ONLY": "1" if only else "",
        "CUDA_VISIBLE_DEVICES": "",
    }
    if pythonpath is not None:
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(pythonpath), os.environ.get("PYTHONPATH", "")) if p)
    return subprocess.run(
        [sys.executable, "-c", script], cwd=cwd, env=env, capture_output=True, text=True, timeout=600, check=False
    )


def _last_json(proc: subprocess.CompletedProcess):
    return json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1])


class TestAPackMayBeOneModule(unittest.TestCase):
    """``<root>/<name>.py`` is a kind exactly as ``<root>/<name>/__init__.py`` is."""

    def _module_root(self, stack: ExitStack, name: str = "probe_granular") -> Path:
        root = Path(stack.enter_context(TemporaryDirectory()))
        (root / f"{name}.py").write_text(KIND)
        return root

    def _package_root(self, stack: ExitStack, name: str = "probe_granular") -> Path:
        root = Path(stack.enter_context(TemporaryDirectory()))
        (root / name).mkdir()
        (root / name / "__init__.py").write_text(KIND)
        return root

    def test_a_single_module_on_a_root_registers_and_reads_as_ours(self) -> None:
        with ExitStack() as stack:
            root = self._module_root(stack)
            proc = _run(REPORT, root)
        self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
        got = _last_json(proc)
        self.assertIn("probe_granular", got["domains"], "a one-module kind on a root did not register")
        self.assertIn("cloth", got["domains"], "the shipped kinds are gone -- the path was replaced, not extended")
        self.assertEqual("robosimstudio.domains.probe_granular", got["module"])
        self.assertEqual("probe_granular.py", Path(got["file"]).name)

    def test_a_module_and_a_package_of_one_name_refuse(self) -> None:
        """The clash a move from package to module makes likely: the old directory
        still on one root, the new file on another. ``iter_modules`` would keep the
        first and never say there was a second."""
        for order in ("module first", "package first"):
            with self.subTest(order=order), ExitStack() as stack:
                module = self._module_root(stack, "probe_dup")
                package = self._package_root(stack, "probe_dup")
                pair = (module, package) if order == "module first" else (package, module)
                proc = _run("import robosimstudio.domains", *pair)
                self.assertNotEqual(0, proc.returncode, "one name as a module and as a package was accepted")
                self.assertIn("probe_dup", proc.stderr)
                self.assertIn(str(pair[1]), proc.stderr, "the message must name both roots")

    def test_an_emptied_directory_is_not_a_second_copy(self) -> None:
        """What a moved pack leaves behind in a checkout that has run it: the
        directory, holding nothing but ``__pycache__``. It is not a package -- the
        scan would not yield it -- so it must not be refused as one, and the
        module on the other root must still be the one that is imported."""
        for order in ("leftover first", "leftover last"):
            with self.subTest(order=order), ExitStack() as stack:
                module = self._module_root(stack)
                leftover = Path(stack.enter_context(TemporaryDirectory()))
                (leftover / "probe_granular" / "__pycache__").mkdir(parents=True)
                (leftover / "probe_granular" / "__pycache__" / "__init__.cpython-312.pyc").write_bytes(b"")
                pair = (leftover, module) if order == "leftover first" else (module, leftover)
                proc = _run(REPORT, *pair)
                self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
                self.assertEqual(str(module / "probe_granular.py"), _last_json(proc)["file"])


NUMBERED = dedent(
    """
    raise RuntimeError("a numbered example was imported as a domain pack")
    """
).strip()
"""An example script as the scan must see it: not at all. Importing it is the failure."""

SCANNED = dedent(
    """
    import json, sys
    from robosimstudio.specs.physics import list_domains
    domains = list_domains()
    stray = sorted(m for m in sys.modules if m.startswith("robosimstudio.domains.") and not m.rsplit(".", 1)[-1].isidentifier())
    print(json.dumps({"domains": domains, "stray": stray}))
    """
).strip()


class TestAnExampleBesideAPackIsNotAPack(unittest.TestCase):
    """A kind's folder holds its module *and* the numbered scripts that run it."""

    def _folder(self, stack: ExitStack, *scripts: str, pack: str = "") -> Path:
        root = Path(stack.enter_context(TemporaryDirectory()))
        if pack:
            (root / f"{pack}.py").write_text(KIND.replace("probe_granular", pack).replace("ProbeGranular", "Probe"))
        for name in scripts:
            (root / name).write_text(NUMBERED)
        return root

    def test_the_scan_imports_the_pack_and_not_the_scripts_beside_it(self) -> None:
        with ExitStack() as stack:
            root = self._folder(stack, "00_drop.py", "01_fold.py", pack="probe_granular")
            proc = _run(SCANNED, root)
        self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
        got = _last_json(proc)
        self.assertIn("probe_granular", got["domains"], "the pack beside the scripts did not register")
        self.assertEqual([], got["stray"], "a numbered script was imported under robosimstudio.domains")

    def test_two_folders_may_each_hold_a_script_of_one_name(self) -> None:
        """``02_cloth/00_intro.py`` and ``03_soft/00_intro.py`` are two examples,
        not one kind declared twice -- which is what the duplicate rule would say
        of two files called ``intro.py``."""
        with ExitStack() as stack:
            first = self._folder(stack, "00_intro.py", pack="probe_granular")
            second = self._folder(stack, "00_intro.py", pack="probe_gel")
            proc = _run(SCANNED, first, second)
        self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
        got = _last_json(proc)
        self.assertLessEqual({"probe_granular", "probe_gel"}, set(got["domains"]))
        self.assertEqual([], got["stray"])

    def test_the_counter_test_two_packs_of_one_name_still_refuse(self) -> None:
        """The exemption is for names no import can spell, and for nothing else."""
        with ExitStack() as stack:
            first = self._folder(stack, "00_intro.py", pack="probe_granular")
            second = self._folder(stack, "01_other.py", pack="probe_granular")
            proc = _run("import robosimstudio.domains", first, second)
        self.assertNotEqual(0, proc.returncode, "one pack name on two roots was accepted")
        self.assertIn("probe_granular", proc.stderr)

    def test_the_checkouts_own_scan_imports_no_numbered_script(self) -> None:
        """The shipped folders, as they lie: ``cloth`` and ``soft`` register and
        ``00_towel`` is in nobody's ``sys.modules``."""
        numbered = [
            path
            for name in roots.DEFAULT_DOMAIN_ROOT_NAMES
            for path in sorted((REPO / name).glob("[0-9]*.py"))
            if (REPO / name).is_dir()
        ]
        proc = _run(SCANNED)
        self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
        got = _last_json(proc)
        self.assertLessEqual({"cloth", "soft"}, set(got["domains"]))
        self.assertEqual([], got["stray"], f"imported as packs, of {[p.name for p in numbered]}")


def _packs(root: Path) -> list[Path]:
    """What a scan of ``root`` imports: identifier-named modules, and packages."""
    found = []
    for child in sorted(root.iterdir()):
        if child.name.startswith(("_", ".")):
            continue
        if child.is_dir() and (child / "__init__.py").is_file() and child.name.isidentifier():
            found.append(child)
        elif child.is_file() and child.suffix == ".py" and child.stem.isidentifier():
            found.append(child)
    return found


class TestWhereTheShippedPacksLive(unittest.TestCase):
    """The checkout's default roots: the three example folders, each if present."""

    def test_the_default_roots_are_the_three_named_and_in_that_order(self) -> None:
        self.assertEqual(("examples/02_cloth", "examples/03_soft", "examples/10_rope"), roots.DEFAULT_DOMAIN_ROOT_NAMES)
        present = [REPO / name for name in roots.DEFAULT_DOMAIN_ROOT_NAMES if (REPO / name).is_dir()]
        for folder in ("02_cloth", "03_soft", "10_rope"):
            self.assertIn(REPO / "examples" / folder, present, "a shipped kind is not where the roots look")
        self.assertEqual(present, roots.DOMAINS._checkouts())

    def test_a_default_root_that_is_not_there_is_skipped(self) -> None:
        """A tree that ships without one of the folders has one root fewer and no error."""
        probe = roots._Roots("ROBOSIMSTUDIO_PROBE_ROOTS", ("examples/02_cloth", "no_such_directory_at_all"))
        self.assertEqual([REPO / "examples" / "02_cloth"], probe._checkouts())
        self.assertEqual([], roots._Roots("ROBOSIMSTUDIO_PROBE_ROOTS", "no_such_directory_at_all")._checkouts())

    def test_no_folder_is_called_domains_under_examples(self) -> None:
        """A kind's folder is named for the kind. ``examples/domains`` was the
        catalogue's word for it, and a reader looking for cloth looks for cloth."""
        self.assertEqual([], sorted((REPO / "examples" / "domains").glob("*.py")), "a pack is still filed there")
        for name in roots.DEFAULT_DOMAIN_ROOT_NAMES:
            self.assertEqual(1, len(_packs(REPO / name)), f"{name} holds one implementation")

    def test_only_the_task_kind_looks_for_collections(self) -> None:
        """``<checkout>/*/tasks`` is a task root; ``<checkout>/*/domains`` is not a
        domain root, or every top-level directory would be searched for one and
        an example folder could be found twice over."""
        found = roots.DOMAINS._checkouts()
        self.assertEqual(len(set(found)), len(found))
        for root in found:
            self.assertIn(root.relative_to(REPO).as_posix(), roots.DEFAULT_DOMAIN_ROOT_NAMES)
        # The counter-test: the task kind does still look one level down.
        collections = [c / "tasks" for c in sorted(REPO.iterdir()) if c.name[:1] not in "._" and (c / "tasks").is_dir()]
        for nested in collections:
            self.assertIn(nested, roots.TASKS._checkouts())

    def test_cloth_and_soft_are_one_module_each(self) -> None:
        script = dedent(
            """
            import json
            import robosimstudio as rss
            import robosimstudio.domains.cloth as cloth
            import robosimstudio.domains.soft as soft
            print(json.dumps({
                "cloth": cloth.__file__, "soft": soft.__file__,
                "sheet": rss.sheet.__module__, "sponge": rss.sponge.__module__,
                "spec": rss.ClothSpec.__module__, "towel": rss.materials.TERRY_TOWELLING is cloth.TERRY_TOWELLING,
            }))
            """
        )
        proc = _run(script)
        self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
        got = _last_json(proc)
        self.assertEqual(REPO / "examples" / "02_cloth" / "cloth.py", Path(got["cloth"]))
        self.assertEqual(REPO / "examples" / "03_soft" / "soft.py", Path(got["soft"]))
        self.assertEqual("robosimstudio.domains.cloth", got["sheet"])
        self.assertEqual("robosimstudio.domains.soft", got["sponge"])
        self.assertEqual("robosimstudio.domains.cloth", got["spec"])
        self.assertTrue(got["towel"], "rss.materials no longer holds the fabrics the cloth module defines")


ALONE = dedent(
    """
    import importlib, json, sys
    from importlib.abc import MetaPathFinder

    REFUSED = "robosimstudio.domains.__OTHER__"


    class _Refuse(MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == REFUSED or name.startswith(REFUSED + "."):
                raise ImportError(name + " was imported by __KIND__, which must stand without it")
            return None


    sys.meta_path.insert(0, _Refuse())

    import robosimstudio as rss
    from robosimstudio.scene.registry import get_entity
    from robosimstudio.specs.physics import list_domains

    kind = importlib.import_module("robosimstudio.domains.__KIND__")
    spec = rss.__MAKE__(name="thing")
    print(json.dumps({
        "domains": list_domains(), "file": kind.__file__, "spec": type(spec).__name__,
        "entity": get_entity(spec.entity).__module__,
        "loaded": sorted(m for m in sys.modules if m.startswith("robosimstudio.domains.")),
    }))
    """
).strip()


class TestEachKindStandsAlone(unittest.TestCase):
    """Cloth without soft, soft without cloth, the rope without either: one folder, one kind, nothing borrowed.

    ``soft.py`` used to reach into ``cloth.py`` for ``resolve_selector``, inside
    two functions, so a tree holding the soft folder alone imported, registered,
    and then failed at the first build. What the two share is the library's now
    (:mod:`robosimstudio.domains.selection`), and this is the claim that follows:
    with the other kind's folder off the path *and* its module refused by name,
    a kind still imports, registers, and hands out its spec and its entity.
    ``tests/test_examples.py`` holds the other half on a GPU -- the examples in
    one folder build and step with the other folder absent.
    """

    PAIRS = (("soft", "03_soft", "sponge", "SoftSpec", "cloth"), ("cloth", "02_cloth", "sheet", "ClothSpec", "soft"))

    def test_neither_implementation_imports_the_other_at_any_scope(self) -> None:
        """Static, so it also sees an import inside a function nobody has called yet."""
        import ast

        for kind, folder, _, _, other in self.PAIRS:
            with self.subTest(kind=kind):
                tree = ast.parse((REPO / "examples" / folder / f"{kind}.py").read_text())
                named = [
                    ast.unparse(node)
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Import | ast.ImportFrom) and f"domains.{other}" in ast.unparse(node)
                ]
                self.assertEqual([], named, f"{kind}.py imports {other}")

    def test_a_kind_registers_and_builds_its_spec_with_the_other_refused(self) -> None:
        for kind, folder, make, spec, other in self.PAIRS:
            with self.subTest(kind=kind, without=other):
                script = ALONE.replace("__OTHER__", other).replace("__KIND__", kind).replace("__MAKE__", make)
                proc = _run(script, REPO / "examples" / folder, only=True)
                self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
                got = _last_json(proc)
                self.assertIn(kind, got["domains"])
                self.assertNotIn(other, got["domains"], "the other kind's folder was meant to be off the path")
                self.assertEqual(REPO / "examples" / folder / f"{kind}.py", Path(got["file"]))
                self.assertEqual(spec, got["spec"])
                self.assertEqual(f"robosimstudio.domains.{kind}", got["entity"])
                self.assertNotIn(f"robosimstudio.domains.{other}", got["loaded"])
                self.assertIn("robosimstudio.domains.selection", got["loaded"], "what the two kinds share")

    ROPE = REPO / "examples" / "10_rope" / "rod"

    @unittest.skipUnless(ROPE.is_dir(), "this tree does not carry the rope's folder")
    def test_the_rope_borrows_from_neither(self) -> None:
        """The third folder, a package rather than one module: no file in it names
        cloth or soft, and with both refused it registers and hands out its spec."""
        import ast

        for path in sorted(self.ROPE.rglob("*.py")):
            tree = ast.parse(path.read_text())
            named = [
                ast.unparse(node)
                for node in ast.walk(tree)
                if isinstance(node, ast.Import | ast.ImportFrom)
                and any(f"domains.{other}" in ast.unparse(node) for other in ("cloth", "soft"))
            ]
            self.assertEqual([], named, f"{path.relative_to(REPO)} imports another kind")
        for other in ("cloth", "soft"):
            with self.subTest(without=other):
                script = ALONE.replace("__OTHER__", other).replace("__KIND__", "rod").replace("__MAKE__", "rod")
                proc = _run(script, self.ROPE.parent, only=True)
                self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
                got = _last_json(proc)
                self.assertIn("rod", got["domains"])
                self.assertFalse({"cloth", "soft"} & set(got["domains"]), "only the rope's folder was on the path")
                self.assertEqual(self.ROPE / "__init__.py", Path(got["file"]))
                self.assertEqual("RodSpec", got["spec"])
                self.assertEqual("robosimstudio.domains.rod.entity", got["entity"])


INSTALLED = dedent(
    """
    import json
    import robosimstudio
    from robosimstudio import roots
    import robosimstudio as rss
    import robosimstudio.domains.cloth as cloth
    import robosimstudio.domains.soft as soft
    from robosimstudio.specs.physics import list_domains
    import robosimstudio.domains.rod as rod
    towel = rss.sheet(name="towel", material=rss.materials.TERRY_TOWELLING)
    print(json.dumps({
        "package": robosimstudio.__file__,
        "roots": [str(r) for r in roots.domain_roots()],
        "cloth": cloth.__file__, "soft": soft.__file__, "rod": rod.__file__,
        "domains": list_domains(),
        "towel": [type(towel).__name__, towel.entity], "sponge": type(rss.sponge()).__name__,
        "cable": [type(rss.rod(name="cable")).__name__, rss.rod(name="cable").entity],
    }))
    """
)


def _force_include(pyproject: Path) -> dict[str, str]:
    try:
        import tomllib
    except ModuleNotFoundError as missing:
        raise unittest.SkipTest("reading pyproject.toml takes tomllib, which Python has from 3.11") from missing

    return tomllib.loads(pyproject.read_text())["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]


class TestAnInstallFindsItsPacks(unittest.TestCase):
    """The wheel's layout, laid out by hand: the package, and every pack merged into ``_domain_packs``.

    ``pyproject.toml`` force-includes each pack onto the one target
    ``robosimstudio/_domain_packs``: the cloth and soft modules as files, and the
    rope's package as a directory, which hatch walks and writes under the
    target. This builds that directory by reading the same block and doing what
    it says -- a file copied to its target, a directory copied onto it -- beside
    a copy of the package, with no checkout above it, and asks a fresh process
    what it sees.
    """

    def _installed(self, stack: ExitStack) -> Path:
        site = Path(stack.enter_context(TemporaryDirectory())).resolve() / "site-packages"
        skip = shutil.ignore_patterns("__pycache__")
        shutil.copytree(REPO / "robosimstudio", site / "robosimstudio", ignore=skip)
        target = f"robosimstudio/{roots.DOMAINS.packaged}"
        for source, to in _force_include(REPO / "pyproject.toml").items():
            if to != target and not to.startswith(target + "/"):
                continue
            if (REPO / source).is_dir():
                shutil.copytree(REPO / source, site / to, ignore=skip, dirs_exist_ok=True)
            else:
                (site / to).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / source, site / to)
        return site

    def test_the_force_include_block_ships_every_pack_and_no_example_script(self) -> None:
        """The other half of the claim: what the copy above imitates is what the wheel is told to do.

        A root mapped whole ships whatever it holds, so it may hold no example
        script. A root that is an example folder is mapped file by file, and
        then the block has to name each pack in it -- a second module added
        beside ``cloth.py`` would register in a checkout and be missing from an
        install -- and nothing that is not one.
        """
        include = _force_include(REPO / "pyproject.toml")
        target = f"robosimstudio/{roots.DOMAINS.packaged}"
        mapped = {source: to for source, to in include.items() if to == target or to.startswith(target + "/")}
        self.assertTrue(mapped, f"nothing is force-included onto {target}")
        expected: dict[str, str] = {}
        for name in roots.DEFAULT_DOMAIN_ROOT_NAMES:
            root = REPO / name
            if not root.is_dir():
                continue
            if name in mapped:
                scripts = sorted(path.name for path in root.glob("*.py") if not path.stem.isidentifier())
                self.assertEqual([], scripts, f"{name} is packaged whole and holds example scripts")
                expected[name] = target
            else:
                expected.update({f"{name}/{pack.name}": f"{target}/{pack.name}" for pack in _packs(root)})
        self.assertEqual(expected, mapped, "the wheel's packs are not the checkout's")
        for source in include:
            self.assertTrue(
                (REPO / source).exists(),
                f"force-include names {source!r}, which this tree does not have: hatch refuses to build a wheel "
                "(or an editable install) with 'Forced include not found'. A tree that drops a root drops its line.",
            )

    def test_the_public_overlay_ships_the_same_packs(self) -> None:
        """``oss/pyproject.toml`` replaces the root's in the public tree, which
        holds the same three example folders: the same lines, pack for pack."""
        overlay = REPO / "oss" / "pyproject.toml"
        if not overlay.is_file():
            self.skipTest(
                "no oss/ overlay here: this is the public tree, and its pyproject.toml is the one checked above"
            )
        target = f"robosimstudio/{roots.DOMAINS.packaged}"
        ours = {s: t for s, t in _force_include(REPO / "pyproject.toml").items() if t.startswith(target)}
        theirs = {s: t for s, t in _force_include(overlay).items() if t.startswith(target)}
        self.assertEqual(ours, theirs)

    def test_packaged_packs_are_found_with_no_checkout(self) -> None:
        with ExitStack() as stack:
            site = self._installed(stack)
            packs = site / "robosimstudio" / "_domain_packs"
            self.assertTrue((packs / "cloth.py").is_file() and (packs / "soft.py").is_file())
            self.assertTrue((packs / "rod" / "__init__.py").is_file(), "the rope's package was not packaged whole")
            numbered = sorted(p.name for p in packs.rglob("*.py") if not p.stem.isidentifier())
            self.assertEqual([], numbered, "an example script was packaged as a domain pack")
            proc = _run(INSTALLED, cwd=site, pythonpath=site)
            self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
            got = _last_json(proc)
        self.assertEqual(site / "robosimstudio" / "__init__.py", Path(got["package"]), "the copy was not what ran")
        self.assertEqual([str(packs)], got["roots"], "an install's only domain root is its packaged one")
        self.assertEqual(packs / "cloth.py", Path(got["cloth"]))
        self.assertEqual(packs / "soft.py", Path(got["soft"]))
        self.assertEqual(packs / "rod" / "__init__.py", Path(got["rod"]))
        self.assertLessEqual({"cloth", "soft", "rod"}, set(got["domains"]))
        self.assertEqual(["ClothSpec", "cloth"], got["towel"])
        self.assertEqual("SoftSpec", got["sponge"])
        self.assertEqual(["RodSpec", "rod"], got["cable"])


if __name__ == "__main__":
    unittest.main()
