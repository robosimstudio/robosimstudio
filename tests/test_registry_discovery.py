# SPDX-License-Identifier: Apache-2.0
"""Content can reach every registry that says it can.

Four registries -- ``DOMAINS``, ``MATERIALS``, ``ATOMS`` and ``HANDLERS`` --
advertised a content path and did not have one. Each scanned a fixed package or
nothing at all and carried no populator, so a domain, a fabric, an atom or a
handler declared in a task directory registered **if and only if** something else
had already imported that task. ``specs/registries.py`` stated the promise outright:
*"A task registers its own the same way, from its assets/ package."*

The usage counts are what a promise nobody could keep looks like from outside.
Against thirty-six assets and fifteen entity classes declared by content, these
four had **eight domains, seven materials, nineteen atoms and thirty handlers --
every one of them the library's own, and none from a task.**

So this module asks the question that was never asked: put a domain and a
material in a throwaway task root, execute it, and see whether the registries
have them. And separately, that each of the four now carries a populator at all
-- because a registry that scans nothing and is told nothing finds nothing, and
that failure looks exactly like an empty corpus.

Static: no scene, no GPU.
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
from textwrap import dedent

from robosimstudio import roots
from robosimstudio.assetlib.materials import MATERIALS
from robosimstudio.assetlib.semantics.atoms import ATOMS
from robosimstudio.assetlib.semantics.handlers import HANDLERS
from robosimstudio.specs.physics import DOMAINS
from robosimstudio_bench.taskbuild.discovery import scan

TIER = "fast"
"""No GPU. Gates every commit."""

DOCUMENT = dedent(
    """
    (define (problem PROBE_DISCOVERY)
      (:domain robosimstudio/probe)
      (:language a probe)
      (:objects)
      (:goal)
    )
    """
).strip()

LOADER = dedent(
    '''
    """A task directory that declares a physics domain and a fabric."""
    from robosimstudio.assetlib.materials import MaterialRecord, register_material
    from robosimstudio.domains.cloth import ClothMaterialSpec
    from robosimstudio.specs.physics import register_domain

    register_domain(
        "probe_gel",
        state=("particles",),
        reads="cloth",
        summary="A probe domain declared by content.",
    )
    register_material(
        MaterialRecord(
            name="probe_denim",
            spec=ClothMaterialSpec(),
            notes="A probe fabric declared by content.",
        )
    )
    '''
).strip()


class _Root:
    """A throwaway task root, cleaned up and unregistered afterwards."""

    def __init__(self) -> None:
        self._tmp = TemporaryDirectory()
        self.path = Path(self._tmp.name)
        directory = self.path / "probe_discovery"
        directory.mkdir(parents=True)
        (directory / f"probe_discovery{roots.TASK_SUFFIX}").write_text(DOCUMENT)
        (directory / "task.py").write_text(LOADER)

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, *exc: object) -> None:
        for registry, name in ((DOMAINS, "probe_gel"), (MATERIALS, "probe_denim")):
            registry._items.pop(name, None)
            registry._origins.pop(name, None)
        roots.clear_task_roots()
        self._tmp.cleanup()


class TestContentReachesTheseRegistries(unittest.TestCase):
    def test_a_task_directory_can_declare_a_domain_and_a_material(self) -> None:
        """The promise ``specs/registries.py`` made, executed.

        A ``register_domain`` from a task directory was reachable only by luck of
        import order before this; the domain here is declared by a root that
        nothing else has any reason to import.
        """
        with _Root() as root:
            self.assertNotIn("probe_gel", DOMAINS._items, "the probe leaked from an earlier run")
            scan(root)
            self.assertIn("probe_gel", DOMAINS._items, "a domain declared by content did not register")
            self.assertIn("probe_denim", MATERIALS._items, "a material declared by content did not register")
            self.assertEqual(("cloth",), DOMAINS.get("probe_gel").reads)


class TestEveryContentRegistryHasAPopulator(unittest.TestCase):
    """A registry that scans nothing and is told nothing finds nothing.

    That failure is invisible: it looks exactly like a corpus with none of the
    thing in it, which is the state all four of these were in.
    """

    def test_the_four_that_had_none_now_have_one(self) -> None:
        for name, registry in (
            ("DOMAINS", DOMAINS),
            ("MATERIALS", MATERIALS),
            ("ATOMS", ATOMS),
            ("HANDLERS", HANDLERS),
        ):
            with self.subTest(registry=name):
                self.assertTrue(
                    registry.discovers,
                    f"{name} scans no content and is told about none; a task's entry would never be found",
                )


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
    from robosimstudio.specs.physics import list_domains, get_domain
    from robosimstudio.scene.registry import get_entity
    cls = get_entity("probe_granular")
    print(json.dumps({"domains": list_domains(), "summary": get_domain("probe_granular").summary,
                      "module": cls.__module__, "entity": cls.__name__}))
    """
).strip()


class TestAKindMayLiveOutsideTheCheckout(unittest.TestCase):
    """A domain package on somebody else's root, reached by our own import name.

    The task roots execute a directory under a synthetic parent
    (``rstask.<slug>_<hash>``) precisely so a task never becomes a top-level
    importable name. A domain package wants the opposite: ten task directories
    write ``from robosimstudio.domains.rod import EntityCable``, and that line
    has to keep working whether the rod is in the library or on a root. So
    ``robosimstudio.domains`` extends its own ``__path__`` instead, and the
    registries that scan it -- ``DOMAINS``, ``ENTITIES``, ``MATERIALS``, every
    one walking ``pkgutil.iter_modules(__path__)`` -- pick up both in one sweep
    with no registry change at all.

    Run in a subprocess because ``__path__`` is built when the package is first
    imported, so this is a claim about what a *process* sees.
    """

    def _root(self, stack, name: str = "probe_granular") -> Path:
        root = Path(stack.enter_context(TemporaryDirectory()))
        (root / name).mkdir()
        (root / name / "__init__.py").write_text(KIND)
        return root

    def _run(self, script: str, *rootpaths: Path) -> subprocess.CompletedProcess:
        env = {
            **os.environ,
            roots.DOMAIN_ROOTS_ENV: os.pathsep.join(str(r) for r in rootpaths),
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

    def test_it_registers_and_reads_as_ours(self) -> None:
        with ExitStack() as stack:
            proc = self._run(REPORT, self._root(stack))
        self.assertEqual(0, proc.returncode, proc.stderr[-2000:])
        got = json.loads([ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1])
        self.assertIn("probe_granular", got["domains"], "a domain on a root did not register")
        self.assertIn(
            "cloth", got["domains"], "the library's own kinds are gone -- the path was replaced, not extended"
        )
        self.assertEqual("A bed of grains.", got["summary"])
        self.assertEqual(
            "robosimstudio.domains.probe_granular",
            got["module"],
            "an out-of-tree kind must be reachable under OUR import name -- that is what keeps a task's "
            "`from robosimstudio.domains.rod import ...` working after the rod moves out",
        )

    def test_two_roots_holding_one_name_refuse(self) -> None:
        """``pkgutil.iter_modules`` keeps the first and never imports the second,
        which hides exactly the collision the registries exist to report."""
        with ExitStack() as stack:
            first = self._root(stack, "probe_dup")
            second = self._root(stack, "probe_dup")
            proc = self._run("import robosimstudio.domains", first, second)
        self.assertNotEqual(0, proc.returncode, "a duplicate kind across two roots was accepted")
        self.assertIn("probe_dup", proc.stderr)
        self.assertIn(str(second), proc.stderr, "the message must name both roots")


if __name__ == "__main__":
    unittest.main()
