# SPDX-License-Identifier: Apache-2.0
"""The asset registry: what exists, and what an agent may compose from.

The same generic registry every other axis in this stack uses -- one name-to-value
map that refuses to be written twice, loading its modules lazily on first lookup.
Three properties matter more here than elsewhere:

**A duplicate is refused, naming both origins.** Two modules declaring a
``cable`` is a real mistake with a silent failure mode, and the registry turns it
into an error that says where each came from.

**Loading is lazy and by name.** The registry names the domain packs and each
content loader as *strings* and imports them at call time, exactly as the suite
registry does -- which is what keeps the layer below free of any module-scope
dependency on this one, and what the import-direction tests measure. It is also
what lets content be absent: a loader whose package is not installed is skipped,
and a query answers with what the library declares (:data:`CONTENT_LOADERS`).

**A missing asset is a catalog gap, not an error.** :func:`get_asset` raises with
the registered names listed, because "the thing your instruction needs does not
exist yet" is the single most useful thing an authoring loop can be told. It is
the signal that turns "the task failed" into "the catalog needs a hook", and it
should read that way to whoever receives it.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

from robosimstudio.registry import Registry

from .record import AssetRecord

__all__ = ["ASSETS", "CONTENT_LOADERS", "find_asset", "get_asset", "list_assets", "register_asset"]


def _factory(build: Any) -> str:
    """A factory's name, not its identity: the same module executed twice -- under
    its synthetic package and again as ``tasks.<suite>`` by a test -- yields two
    function objects for one factory."""
    return getattr(build, "__qualname__", None) or repr(build)


def _same_record(first: AssetRecord, second: AssetRecord) -> bool:
    """Whether two declarations describe the same thing.

    Compared on what the record *says*, not on where the copy was found: two
    tasks each carrying their own declaration of the same tool have different
    ``home`` directories and therefore different absolute file paths, and they
    are still the same tool. Files are matched by filename, which is the part
    that is a fact about the asset.

    Anything that would change what may be asked of it -- substrate, features,
    states, published states, sim requirements -- is compared exactly, so a
    second declaration that quietly added a feature is still a clash. Two
    different ``build`` factories are a clash too; a declaration with one and
    a declaration without are the same tool, one of which also knows how to
    place itself, and the first registered is kept.
    """
    if (first.substrate, first.tags, first.mesh_sha) != (second.substrate, second.tags, second.mesh_sha):
        return False
    if first.build is not None and second.build is not None and _factory(first.build) != _factory(second.build):
        return False
    if (first.features, first.states, first.published, first.sim) != (
        second.features,
        second.states,
        second.published,
        second.sim,
    ):
        return False
    return {k: Path(v).name for k, v in first.files.items()} == {k: Path(v).name for k, v in second.files.items()}


CONTENT_LOADERS: tuple[str, ...] = ("robosimstudio_bench.taskbuild.discovery:load_roots",)
"""Where content comes from: ``module:function`` names, run in order on every load.

Names rather than objects, because each one sits a layer above this registry
and may not be installed at all. The benchmark tier is something this library
can be used without, so an entry is tried only when its top-level package can be
found: an install that lacks it answers an asset query with what the library
itself declares, where it used to answer with ``ImportError`` from inside a
lookup -- and the callers of a lookup catch ``KeyError``, or nothing.

A package that *is* there and fails to load is a different thing, and still
raises: see :func:`_installed` for where the line is drawn.

A second body of content is a second entry. A process that wants one without
editing this tuple installs its own populator with ``ASSETS.add_discover``.
"""


def _installed(package: str) -> bool:
    """Whether a top-level package can be found, asked without running it.

    ``find_spec`` on a top-level name consults the finders and executes none of
    the package's own code, which is what keeps two failures apart. *Not there*
    is ``None`` here, and the loader is skipped. *There and broken* gets as far
    as the import in :func:`_discover` and raises from it, with its own
    traceback -- an optional tier is not a licence to swallow a broken one, and
    a catalog that came back short because a task directory failed to execute
    is the failure nobody finds.

    A finder that refuses the name outright counts as not there. That is how
    this repository's tests stand in for an install without the benchmark tier
    -- a meta-path blocker that raises -- and it is the only way an
    ``ImportError`` can come out of a lookup that executes nothing.

    So does a bare directory of that name. A checkout switched from a tree that
    had the tier to one that does not keeps the directory, holding nothing but
    bytecode caches, and Python finds that as a namespace package: a spec with
    no ``origin``. Taking it for the package would turn every asset query into
    ``No module named ...discovery`` on exactly the tree this guard is for.
    """
    if sys.modules.get(package) is not None:
        return True
    try:
        spec = importlib.util.find_spec(package)
    except ImportError:
        return False
    return spec is not None and spec.origin is not None


def _discover() -> None:
    """Run each content loader whose package is installed, so content-declared assets register.

    The library's own declarations are already in by the time this runs:
    :data:`ASSETS` scans the domain packs first, the same scan ``MATERIALS``,
    ``ATOMS`` and ``HANDLERS`` make before they run this very hook. What is
    left for it is content, and content is optional -- see
    :data:`CONTENT_LOADERS`.

    This registry keeps the hook its siblings gave up, and the asymmetry is the
    point: the library declares **no** asset of its own and content declares all
    of them -- 302 records, every one from a task directory, when this was last
    counted (2026-10-04) -- so asking this registry anything is a question about
    content and loading it is the answer rather than a cost. ``CELLS``,
    ``EMBODIMENTS`` and ``GRIPPERS`` are the other way round -- the library
    ships all the robots and most of the benches -- so they no longer pay for a
    task-directory scan to hand back something they shipped themselves.

    Imported at call time: the discovery machinery is a layer up, and at module
    scope this would be an import that cannot complete.
    """
    for entry in CONTENT_LOADERS:
        module, _, function = entry.partition(":")
        if _installed(module.partition(".")[0]):
            getattr(importlib.import_module(module), function)()


ASSETS: Registry[AssetRecord] = Registry("asset", ("robosimstudio.domains",), same=_same_record)
ASSETS.add_discover(_discover)
"""Every registered asset.

Two sources, in this order. The domain packs are scanned first, exactly as
``MATERIALS`` scans them, so a record a pack declares -- in the library or on a
domain root -- is here whether or not any content is installed. Then the
:data:`CONTENT_LOADERS` run, and each asset module of each task directory
registers as it executes, so nothing maintains a list by hand.

Note what is *not* here: no asset of this repository's own. A library that
shipped a shirt would make everyone using it carry one. The packs in this
checkout *build* records -- ``Body.from_mesh``, ``Rod.from_length`` -- and
register none, so with no content installed this registry is empty, and says so
by listing nothing rather than by raising.
"""


def register_asset(record: AssetRecord) -> AssetRecord:
    """Register an asset under its own name.

    Call at module scope in the asset's own module. Refuses a declaration with
    structural problems rather than letting it into the catalog: an asset that
    cannot be validated is one an agent will try to compose with and fail to,
    somewhere much less obvious than here.
    """
    problems = record.problems()
    if problems:
        raise ValueError(f"asset {record.name!r} is not well formed:\n  " + "\n  ".join(problems))
    if record.home is None:
        # Where the declaring module sits, so the record's files resolve beside
        # it. Stamped here rather than declared, because a record cannot know
        # which directory somebody will keep it in -- and read off the caller's
        # frame rather than passed, so no asset module has to remember to.
        frame = inspect.stack()[1]
        object.__setattr__(record, "home", Path(frame.filename).resolve().parent)
    _resolve_beside(record)
    return ASSETS.register(record.name, record, origin=getattr(record, "__module__", ""))


def _resolve_beside(record: AssetRecord) -> None:
    """Rewrite declared files that ship beside the declaration to absolute paths.

    An asset whose geometry sits in its own directory is resolvable without any
    root having heard of it, which is what lets a task directory carry its own
    meshes and be moved. Done once, at registration, so every reader -- the
    lowering accessor, the catalog, a test walking ``files`` -- sees one path
    that works rather than each having to know the rule.

    Only when the file is actually there: a record naming a path in the vendored
    tree keeps naming it, so a same-named file happening to sit beside the
    declaration cannot silently redirect it.
    """
    if record.home is None:
        return
    home = Path(record.home)
    rewritten = {}
    for key, declared in record.files.items():
        beside = home / declared
        rewritten[key] = str(beside) if not Path(declared).is_absolute() and beside.exists() else declared
    if rewritten != record.files:
        object.__setattr__(record, "files", rewritten)


def get_asset(name: str) -> AssetRecord:
    """Look one up by name.

    Raises:
        KeyError: Listing what is registered. This is the catalog-gap signal an
            authoring agent acts on, so the message is the useful half.
    """
    return ASSETS.get(name)


def list_assets() -> list[str]:
    """Every registered asset name, sorted."""
    return ASSETS.names()


def find_asset(name: str) -> AssetRecord | None:
    """The record ``name`` declares, or ``None`` for a name nobody declared -- the page's question, not the loader's."""
    if not name:
        return None
    try:
        return ASSETS.get(name)
    except KeyError:
        return None
