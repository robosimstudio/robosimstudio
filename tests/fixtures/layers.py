# SPDX-License-Identifier: Apache-2.0
"""Importing a set of modules in a subprocess with some of them refused.

Shared by the two halves of the partition's enforcement -- the package law in
``test_import_direction.py`` and the content seam in ``test_content_seam.py`` --
because both make the same kind of measurement and a second copy of an import
blocker is a second thing to get subtly wrong.

The blocker is a meta-path finder rather than a source grep, and that is the
whole point: it catches the relative spellings a shared parent package makes
possible, it catches transitive reaches, and it cannot be fooled by a string.
What it deliberately does *not* catch is a lazy import inside a function, which
is what the registries legitimately do.
"""

from __future__ import annotations

import pkgutil
import subprocess
import sys
import unittest
from pathlib import Path

__all__ = ["import_with_blocked", "modules_in"]

_BLOCKER = """
import importlib
import sys
from importlib.abc import MetaPathFinder

BLOCKED = __BLOCKED__


class _Block(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in BLOCKED):
            return None
        raise ImportError(name + " was imported by a __LAYER__ module. __WHY__")


sys.meta_path.insert(0, _Block())

MODULES = __MODULES__
for module in MODULES:
    importlib.import_module(module)
print("ok", len(MODULES))
"""

_WHY_LAYER = (
    "A package must import with everything above it absent. Where a lower mechanism needs knowledge from "
    "higher up it takes explicit data (index arrays, VisualPart lists, shape->label tables) or states a "
    "protocol, and the layer above supplies it. See DEPS for what this one may name."
)

_WHY_TASKS = (
    "The benchmark machinery must not depend on the suites at import time: it knows what a task is, not "
    "which tasks exist. Suites arrive through the registries, which import them by name when a lookup "
    "first needs one -- so name the package inside the function that resolves it, never at module scope."
)


def modules_in(package, optional: frozenset[str] = frozenset()) -> list[str]:
    """Every importable module in a layer, minus the ones needing an extra.

    Backends and bridges that need an optional dependency are excluded by name:
    they are imported lazily by whatever composes them, and a machine without
    pyrender should not fail an architectural test.
    """
    # The package's own ``__path__``, not ``Path(__file__).parent``. Two of the
    # library's packages extend theirs so their content can live outside the
    # checkout, and the in-tree directory is then *one* of the places their
    # modules are rather than the only one. Anchored to that one directory this
    # would keep passing after content moved out and quietly stop checking it,
    # which is the worst way for a law to fail: green, and blind.
    search = [str(entry) for entry in getattr(package, "__path__", ())] or [str(Path(package.__file__).parent)]
    names = [package.__name__]
    for info in pkgutil.walk_packages(search, prefix=package.__name__ + "."):
        if info.name not in optional:
            names.append(info.name)
    return names


def import_with_blocked(
    test: unittest.TestCase, modules: list[str], blocked: tuple[str, ...], layer: str, why: str
) -> None:
    """Import every module in a subprocess with every ``blocked`` prefix refused."""
    test.assertGreater(len(modules), 1, f"walk_packages found suspiciously few {layer} modules")
    script = (
        _BLOCKER.replace("__MODULES__", repr(modules))
        .replace("__BLOCKED__", repr(blocked))
        .replace("__LAYER__", layer)
        .replace("__WHY__", why)
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600, check=False)
    test.assertEqual(result.returncode, 0, f"{layer} failed to import without {blocked}:\n{result.stderr}")
