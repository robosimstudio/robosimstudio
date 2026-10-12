# SPDX-License-Identifier: Apache-2.0
"""A foreign engine declared **outside this checkout**, reached under our name.

``ROBOSIMSTUDIO_ENGINE_ROOTS`` is the mechanism behind the project's headline
claim -- somebody else's simulator composed beside the Newton islands without a
line of library code -- and it was the one root system with no test at all. Its
two siblings are both covered end to end: task roots by
``tests/test_task_roots``, domain roots by ``tests/test_registry_discovery``.

That gap nearly cost the mechanism. ``engine_roots()`` returns ``[]`` on a plain
checkout (no top-level ``engines/`` directory exists), ``ROBOSIMSTUDIO_ENGINE_ROOTS``
is set nowhere, and ``add_engine_root``/``clear_engine_roots`` have no callers --
so by caller count it reads as dead, and a sweep looking for dead extension
points would take it. It is not dead: ``robosimstudio/engines/__init__.py``
extends its own ``__path__`` from it, exactly as the domains package does, and
that line is what makes an out-of-tree bundle importable as
``robosimstudio.engines.<name>``.

An unused seam and a broken seam are indistinguishable without a test, and the
difference decides whether it should be kept or cut. This is the test.

Run in a subprocess with the variable in the environment, like its two siblings,
because ``__path__`` is extended at import time and there is no honest way to
re-do that inside a live interpreter.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from robosimstudio import roots

TIER = "fast"

BUNDLE = '''# SPDX-License-Identifier: Apache-2.0
"""A foreign engine bundle that lives outside the checkout."""

from robosimstudio.specs.physics import register_backend

BACKEND = register_backend(
    "probe_engine",
    cls="probe_engine_pkg.SolverProbe",
    supports=("particles",),
    host_engine=True,
    summary="A probe engine on a root.",
)
'''

REPORT = """
import json
import robosimstudio.engines                     # extends __path__ from the roots
from robosimstudio.specs.physics import ENGINES, get_backend

ENGINES.load()                                   # the scan the registry does lazily
info = get_backend("probe_engine")
print(json.dumps({
    "host_engine": bool(info.host_engine),
    "supports": list(info.supports),
    "cls": info.cls,
    "module": __import__("robosimstudio.engines.probe_engine", fromlist=["x"]).__name__,
}))
"""


class TestAnEngineOnARoot(unittest.TestCase):
    def _root(self, stack: ExitStack, name: str = "probe_engine") -> Path:
        root = Path(stack.enter_context(TemporaryDirectory()))
        (root / name).mkdir()
        (root / name / "__init__.py").write_text(BUNDLE)
        return root

    def _run(self, script: str, *rootpaths: Path) -> subprocess.CompletedProcess:
        env = {
            **os.environ,
            roots.ENGINE_ROOTS_ENV: os.pathsep.join(str(r) for r in rootpaths),
            "CUDA_VISIBLE_DEVICES": "",
        }
        return subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )

    def test_a_bundle_on_a_root_registers_and_is_reachable_as_ours(self) -> None:
        with ExitStack() as stack:
            proc = self._run(REPORT, self._root(stack))
        self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
        got = json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1])
        self.assertTrue(got["host_engine"], "a bundle on a root did not register as a host engine")
        self.assertEqual(["particles"], got["supports"])
        self.assertEqual(
            "robosimstudio.engines.probe_engine",
            got["module"],
            "an out-of-tree engine must be reachable under OUR import name, the way a domain is",
        )

    def test_the_shipped_bundles_survive_a_root(self) -> None:
        """Extending the path, not replacing it -- the failure the domains test names too."""
        script = "import robosimstudio.engines as e; import json; print(json.dumps(sorted(p.split('/')[-1] for p in e.__path__)))"
        with ExitStack() as stack:
            proc = self._run(script, self._root(stack))
        self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
        self.assertGreaterEqual(
            len(json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("[")][-1])),
            2,
            "the library's own engines directory is gone -- the path was replaced, not extended",
        )

    def test_two_roots_holding_one_name_refuse(self) -> None:
        """``iter_modules`` keeps the first and never imports the second, which
        hides exactly the collision the registries exist to report."""
        with ExitStack() as stack:
            first = self._root(stack, "probe_dup_engine")
            second = self._root(stack, "probe_dup_engine")
            proc = self._run("import robosimstudio.engines", first, second)
        self.assertNotEqual(0, proc.returncode, "a duplicate engine across two roots was accepted")
        self.assertIn("probe_dup_engine", proc.stderr)
        self.assertIn(str(second), proc.stderr, "the message must name both roots")


NEWTON_PROOF = """
import json
import sys
import robosimstudio.specs                       # the IR alone: no bundle, no scan yet
assert "robosimstudio.engines" not in sys.modules, "importing the IR imported an engine bundle"
from robosimstudio.specs.physics import get_backend
mujoco = get_backend("mujoco")
vbd = get_backend("vbd")
print(json.dumps({
    "mujoco_cls": mujoco.cls,
    "vbd_maximal": bool(vbd.maximal_coordinates),
    "bundle_loaded": "robosimstudio.engines.newton" in sys.modules,
    "domains_scanned": "robosimstudio.domains" in sys.modules,
}))
"""


class TestTheNewtonBundleIsFoundByTheSameScan(unittest.TestCase):
    """The Newton solvers register from ``engines/newton``, not from the IR.

    ``mujoco``, ``vbd``, ``xpbd``, ``featherstone`` and ``kinematic`` used to
    register the moment ``specs/physics.py`` was imported, which made them the
    one kind of alias no scan ever had to find. They are a bundle now, and the
    claim that matters is that nothing changed for the caller: an arm-only
    scene resolves ``"mujoco"`` with nothing imported by hand, through the same
    ``ENGINES`` scan that finds ``superdex`` and ``genesis``. A fresh subprocess
    with only the IR imported, because in a live interpreter something has
    always already loaded the bundle.
    """

    def test_an_alias_resolves_with_only_the_ir_imported(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-c", NEWTON_PROOF],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
        got = json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1])
        self.assertEqual("newton.solvers.SolverMuJoCo", got["mujoco_cls"])
        self.assertTrue(got["vbd_maximal"], "vbd's registered facts did not survive the move")
        self.assertTrue(got["bundle_loaded"], "the alias resolved without the scan importing engines/newton")
        self.assertFalse(
            got["domains_scanned"],
            "a Newton alias is found by the engine scan; the domain bundles are not its business",
        )


class TestNoBundleImportsItsEngine(unittest.TestCase):
    """An engine adapter reaches its SDK inside a function, never at import.

    The law the whole three-tier install rests on: a base checkout imports the
    library, sees every backend alias resolve, and pays for no engine wheel.
    ``tests/test_genesis_absent.py`` made this claim for one engine and made it
    with an assertion that could not fail (``'genesis' not in sys.modules or
    sys.modules['genesis'] is not None`` is true either way). This is the claim
    for **every** bundle, including any added later, against the same banned
    list ``ruff`` lints with -- read from ``pyproject.toml`` rather than copied,
    so the lint rule and the runtime law cannot drift apart.

    A subprocess per bundle, because the question is what an import *does* in a
    fresh interpreter and this one has already imported half the tree.
    """

    NEWTONS_OWN = frozenset({"pxr"})
    """Banned modules that arrive with ``newton`` itself and are not ours to avoid.

    Measured: ``import newton`` alone puts ``pxr`` in ``sys.modules`` (``import
    warp`` does not), so every module in this repository that imports the IR
    inherits it. Exempted by name rather than by widening the check, so the day
    newton stops pulling USD this list is what says the exemption can go.
    """

    def _banned(self) -> list[str]:
        try:
            import tomllib
        except ModuleNotFoundError as missing:
            raise unittest.SkipTest("reading pyproject.toml takes tomllib, which Python has from 3.11") from missing

        config = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
        banned = config["tool"]["ruff"]["lint"]["flake8-tidy-imports"]["banned-module-level-imports"]
        return [name for name in banned if name not in self.NEWTONS_OWN]

    def _bundles(self) -> list[str]:
        import robosimstudio.engines as engines

        root = Path(engines.__file__).parent
        return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "__init__.py").is_file())

    def test_the_bundles_are_found(self) -> None:
        """The counter-test: an empty sweep would pass the law vacuously."""
        found = self._bundles()
        self.assertGreaterEqual(len(found), 3, f"expected newton, superdex and genesis at least; found {found}")
        self.assertIn("newton", found)

    def test_no_bundle_imports_a_banned_module(self) -> None:
        banned = self._banned()
        self.assertIn("superdex", banned, "the list must still name the SDKs this is about")
        for bundle in self._bundles():
            with self.subTest(bundle=bundle):
                code = (
                    "import importlib, json, sys\n"
                    f"importlib.import_module('robosimstudio.engines.{bundle}')\n"
                    "try:\n"
                    f"    importlib.import_module('robosimstudio.engines.{bundle}.bridge')\n"
                    "except ModuleNotFoundError:\n"
                    "    pass\n"
                    f"print(json.dumps(sorted(m for m in {banned!r} if m in sys.modules)))\n"
                )
                proc = subprocess.run(
                    [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False
                )
                self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
                pulled = json.loads(proc.stdout.strip().splitlines()[-1])
                self.assertEqual(
                    [],
                    pulled,
                    f"robosimstudio.engines.{bundle} imports {pulled} at module scope; a base install has none of them",
                )


if __name__ == "__main__":
    unittest.main()
