# SPDX-License-Identifier: Apache-2.0
"""The asset component: what things *are*, declared once and owned in one place.

An asset declares its substrate, the features it offers, the states it publishes
and what it needs of physics -- and carries the handlers over them. A task then
*composes* assets rather than teaching the simulator what a thing is.

**Two halves, and the split is machinery versus content:**

:mod:`~robosimstudio.assetlib.semantics`
    The vocabulary: substrates, feature kinds, the selector language, atoms,
    handlers. Knows nothing about any particular
    asset.
An asset family's own module (a tape, a shirt, a set of fixtures)
    The content: one module per asset family, declaring features and registering
    handlers in that vocabulary.

**Why it sits above the toolkit rather than inside it.** The dependency runs
``task documents (data) -> assetlib -> toolkit -> core -> newton``, and the
direction is load-bearing: this layer *compiles down* into the frozen specs the
toolkit already consumes. A resolved selector becomes the ``Selector`` callable
``ClothSpec.particle_sets`` has always held; a compiled goal tree becomes the
``Evaluator`` ``TaskSpec.evaluator`` has always held; declared sim requirements
become a solver arrangement. So the scene compiler never learns that features,
recipes or goal trees exist -- it keeps consuming exactly what it consumes
today, and this layer keeps its freedom to change without touching the runtime.

Cross-package imports are **absolute** (``from robosimstudio.scene... import``),
never relative. Relative dot-counting silently changes meaning when a module
moves between nesting depths, which is the regression the import-direction tests
already guard against under ``tasks/``; the same argument applies here and the
same rule follows.

**On the name.** ``assetlib`` and not ``assets``, for a concrete reason:
``robosimstudio/assets/`` is the path ``pyproject.toml`` force-includes the
vendored meshes into, and one
:func:`~robosimstudio.paths._candidate_roots` searches. A Python
package sitting there would collide with the data directory an installed wheel
resolves its meshes, MJCF and USD against. The data path is spoken for; the code
layer takes the other name.
"""

from __future__ import annotations

# Only these two eagerly, and each for a reason the laziness below does not
# reach: ``vocabulary`` exists to *register* -- importing it is the whole point
# -- and the populator has to be installed before any registry is first asked.
from . import vocabulary  # noqa: F401 -- registers the atom catalogue on import

_EXPORTS: dict[str, tuple[str, str]] = {
    "ASSETS": ("registry", "ASSETS"),
    "AssetRecord": ("record", "AssetRecord"),
    "Facet": ("interface", "Facet"),
    "Interface": ("interface", "Interface"),
    "Member": ("interface", "Member"),
    "SimRequirements": ("sim", "SimRequirements"),
    "asset_to_dict": ("record", "asset_to_dict"),
    "facet": ("interface", "facet"),
    "get_asset": ("registry", "get_asset"),
    "interface": ("interface", "interface"),
    "list_assets": ("registry", "list_assets"),
    "register_asset": ("registry", "register_asset"),
    "sim_to_dict": ("sim", "sim_to_dict"),
}
"""Flat name -> (module, name in it). Resolved on access, not at import.

Thirteen names, all of them about what an asset *is*: its record, its
registry, its interface, its physics requirements. The goal half -- binding,
goal trees, evaluation, the gauntlet, the catalogue -- used to be re-exported
from here too, and that is what made a ``[core]`` install impossible: nine of
those modules name the ``.task`` language at module scope, so ``rss.box`` ->
``domains.rigid.declare`` -> ``assetlib.record`` ran this file and imported a
document parser into a process that would never read a document. They live in
the benchmark tier's ``goals`` package now and are imported from there by
name; nothing here reaches upward at module scope.

:pep:`562` ``__getattr__`` stays, for the same reason the root package uses it:
``from robosimstudio.assetlib.record import AssetRecord`` costs ``record.py``
and nothing else, and a table is something a reader can see.
"""


def __getattr__(name: str):
    """Resolve one export, importing its module on first access (:pep:`562`)."""
    found = _EXPORTS.get(name)
    if found is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module, attribute = found
    value = getattr(importlib.import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = value  # the second access is a plain lookup
    return value


def __dir__() -> list[str]:
    return sorted(_EXPORTS)


__all__ = [
    "ASSETS",
    "AssetRecord",
    "Facet",
    "Interface",
    "Member",
    "SimRequirements",
    "asset_to_dict",
    "facet",
    "get_asset",
    "interface",
    "list_assets",
    "register_asset",
    "sim_to_dict",
]


def _install_content_discovery() -> None:
    """Give the three registries with no populator the one ``ASSETS`` has.

    ``MATERIALS``, ``ATOMS`` and ``HANDLERS`` were reachable from content only by
    accident of import order: each scans a fixed package or nothing at all, so a
    fabric, an atom or a handler declared in a task directory registered if and
    only if something else had already imported that task. The usage counts say
    what that was worth -- the library registers seven materials, nineteen atoms
    and thirty handlers, and content registers **none of any of them**, against
    thirty-six assets and fifteen entity classes on the surfaces that do work.

    They take ``ASSETS``' own populator, which is the right one: all four answer
    questions about things a task directory declares, and the argument that keeps
    a task-root scan off ``CELLS`` and ``EMBODIMENTS`` -- that the library ships
    the answer -- does not apply to any of them.
    """
    from .materials import MATERIALS
    from .registry import _discover
    from .semantics.atoms import ATOMS
    from .semantics.handlers import HANDLERS

    for registry in (MATERIALS, ATOMS, HANDLERS):
        registry.add_discover(_discover)


_install_content_discovery()
