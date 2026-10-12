# SPDX-License-Identifier: Apache-2.0
"""Where tasks come from: directories, not packages.

A task is a **directory**. It holds the documents that state its variants, the
assets they name, the arrangement they stage, and the solutions that answer
them -- and it is self-contained, so handing somebody a task is handing them a
directory. A **root** is a directory holding task directories, and this module
is how the runtime is told which roots to read.

**Why paths rather than an installable package.** The previous design named a
content *package* (``ROBOSIMSTUDIO_CORPUS``), which the registries imported. That
works for exactly one body of content at a time and requires it to be
installable: a colleague's three tasks in a git checkout could not be run
alongside this repository's without packaging them first. A root is a path, so
running somebody else's tasks is ``--task-root ~/their-repo/tasks`` and nothing
else.

Roots accumulate from three places, most explicit last, and all of them apply:

1. ``<checkout>/tasks`` and every ``<checkout>/<collection>/tasks`` -- what
   this repository ships, when it is there: a *collection* is a top-level
   directory owning a ``tasks/`` of its own, found by shape one level down and
   no deeper. An installed
   wheel has no such directory and gets no default root; you point it at one.
2. :data:`ROOTS_ENV`, an ``os.pathsep``-separated list.
3. :func:`add_task_root`, which is what ``--task-root`` calls.

:func:`add_task_root` also **writes back into the environment**, and that is
load-bearing rather than tidy: the worker pool spawns processes, and a spawned
process inherits ``os.environ`` and nothing else. A root added on the command
line would otherwise exist only in the parent, and every worker would come up
with a registry that had never heard of the task it was told to run.

**A task directory is one that holds at least one ``.task`` document.** Names
beginning with ``_`` or ``.`` are skipped, so a root may keep scratch or private
directories beside its tasks without either being mistaken for one.

**There is no shared area, deliberately.** Two tasks standing in the same cell
each carry their own declaration of it. The registries accept a second
*identical* declaration and refuse a conflicting one, so the redundancy is
checked rather than merely permitted -- and every task directory stays something
you can hand to somebody on its own, which a shared directory would quietly take
away.

This module imports nothing -- not from this package, not from anywhere -- so
every layer may read it without a cycle and without pointing upward.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "DEFAULT_DOMAIN_ROOT_NAMES",
    "DEFAULT_ENGINE_ROOT_NAME",
    "DEFAULT_ROOT_NAME",
    "DOMAIN_ROOTS_ENV",
    "ENGINE_ROOTS_ENV",
    "ROOTS_ENV",
    "SUITE_SUFFIX",
    "TASK_SUFFIX",
    "add_domain_root",
    "add_engine_root",
    "add_task_root",
    "clear_domain_roots",
    "clear_engine_roots",
    "clear_task_roots",
    "domain_roots",
    "engine_roots",
    "extend_path",
    "is_task_dir",
    "suite_dir",
    "task_roots",
]

ROOTS_ENV = "ROBOSIMSTUDIO_TASK_ROOTS"
"""Environment variable holding an ``os.pathsep``-separated list of roots.

Where the content lives is a deployment fact rather than a library one. With
this set, running a different body of tasks is an environment variable rather
than a fork -- and because the pool's workers inherit the environment, it is
also the channel a command-line root travels down.
"""

DEFAULT_ROOT_NAME = "tasks"
"""The directory this repository keeps its own tasks in, relative to the checkout."""

DOMAIN_ROOTS_ENV = "ROBOSIMSTUDIO_DOMAIN_ROOTS"
"""Environment variable holding an ``os.pathsep``-separated list of domain roots.

A **domain root** is a directory of packs, each declaring a kind of thing --
what it is, what builds it, what may be measured of it -- as a package or as one
module. The library's own live in ``robosimstudio/domains/``; a root here
contributes more, under the same import name, because that package extends its
``__path__`` from this list.

The same shape as the task roots and for the same reason: where content lives is
a deployment fact rather than a library one. It is a *separate* variable rather
than a second use of the task one because a task directory and a domain package
are different things -- ``is_task_dir`` wants a ``.task`` document, which a
domain package has no reason to carry, and scanning one list for both would mean
every domain needed a task to be found."""

DEFAULT_DOMAIN_ROOT_NAMES = ("examples/02_cloth", "examples/03_soft", "examples/10_rope")
"""Where this repository keeps the kinds that are content, in scan order.

Three example folders, relative to the checkout, and each is a root only if it
is there. Each holds one implementation beside the numbered scripts that run
it: ``examples/02_cloth/cloth.py`` and ``examples/03_soft/soft.py`` are one
module each, read top to bottom, and the rope in ``examples/10_rope`` is a
package. A tree that leaves a folder out simply has one root fewer.

**Only a name that is a Python identifier is a pack.** ``00_towel.py`` is a
script a reader runs, not a kind, and nothing imports it under this package:
:func:`extend_path` leaves it out of the duplicate rule and the registries' scan
skips it, so a kind's folder may hold as many numbered examples as it likes.

In the folders' reading order, which is otherwise immaterial: a name two roots
both hold is refused by :func:`extend_path` rather than resolved by position."""

ENGINE_ROOTS_ENV = "ROBOSIMSTUDIO_ENGINE_ROOTS"
"""Environment variable holding an ``os.pathsep``-separated list of engine roots.

An **engine bundle** is a package that registers a foreign simulator behind the
host-engine seam -- ``robosimstudio/engines/superdex`` and ``.../genesis`` are the
two the library ships. Before this, the ``ENGINES`` registry scanned that one
package and nothing else, so a third party's engine registered only if something
happened to import it first: a coincidence rather than a mechanism, and the same
gap the embodiment and gripper registries had before they took a populator."""

DEFAULT_ENGINE_ROOT_NAME = "engines"
"""Where this repository would keep out-of-tree engine bundles. Absent by default."""

TASK_SUFFIX = ".task"
"""What a task document is called.

The extension is the claim: these are **extended** BDDL, and the extension says
so rather than a comment inside the file. Standard BDDL keeps ``.bddl``, which is
what the exporter writes -- so the two suffixes state the relationship between
the dialect and its parent.
"""
SUITE_SUFFIX = ".suite"
"""What every variant of one suite shares, stated once beside its documents."""


class _Roots:
    """One accumulating list of content roots, for one kind of content.

    The task roots were this shape written once; domain and engine roots are the
    same shape twice more, and three copies of an accumulation rule is three
    places for it to drift. So the rule is here and the three kinds are
    instances: the checkout's own directory when it has one, then the
    environment's list, then whatever was added programmatically, de-duplicated
    on ``realpath``.

    De-duplicated on the real path rather than on the string because the same
    directory reached twice -- once relative, once through a symlink -- would
    otherwise be scanned twice and everything in it would collide with itself.

    Args:
        env: The environment variable holding an ``os.pathsep``-separated list.
        default_name: The directory this repository keeps its own in, relative
            to the checkout -- or several, in scan order, for a kind of content
            the checkout keeps in more than one place (the domain packs: see
            :data:`DEFAULT_DOMAIN_ROOT_NAMES`). Each is a root only if it is
            there. Absent -- which is the case for an installed wheel, and for a
            checkout that has not moved any content out yet -- there is simply
            no default root.
        packaged: Where a *wheel* carries this kind of content, relative to the
            package, or empty for a kind no wheel ships.

            A wheel has no repo root to climb to, and force-including the content
            at ``<site-packages>/<default_name>`` would put a bare top-level
            package beside every other installed distribution -- not a filing
            preference but a name collision waiting for somebody else's. So an
            install carries it *inside* the package, under a name no import
            resolves, and it is looked at last: a checkout has no such directory
            and an install has nothing else, so neither pays for the other.

            **One packaged directory, however many the checkout keeps.** The
            wheel maps every pack onto this one target (``pyproject.toml``'s
            ``force-include``): the implementation out of each example folder,
            pack by pack, so the numbered scripts beside it stay behind. The
            roots of a checkout arrive in an install as one:
            ``_domain_packs/cloth.py`` and ``_domain_packs/soft.py`` beside the
            rope's package. Merging is safe for the reason several roots
            are -- one name is one thing, and :func:`extend_path` has already
            refused a tree where it is not.

            The same shape ``paths._candidate_roots`` uses for the vendored
            assets, and for the same reason -- including the reason it is
            *stated*: that one was omitted when the file grew its second root, so
            the wheel shipped a copy of every mesh to a path nothing looked in,
            and every scene built from an install failed to resolve its geometry.
    """

    __slots__ = ("added", "default_names", "env", "packaged")

    def __init__(self, env: str, default_name: str | tuple[str, ...], *, packaged: str = "") -> None:
        self.env = env
        self.default_names: tuple[str, ...] = (default_name,) if isinstance(default_name, str) else tuple(default_name)
        self.packaged = packaged
        self.added: list[Path] = []

    def _checkouts(self) -> list[Path]:
        """The checkout's own roots: each ``<checkout>/<name>`` that exists and, for tasks, ``<checkout>/*/<name>``.

        A **collection** is a top-level directory that holds a ``tasks/`` of its
        own -- one per owner: the paper a suite was written for, or the in-house
        examples -- so that handing a collection to somebody is handing them a
        directory, and moving one out of the tree is a ``git mv``. Found by its
        shape, one level down and no deeper. Names beginning with ``_`` or ``.`` are skipped, as a task
        directory's would be. Only the task kind looks for collections; a domain
        or engine root is exactly the directories it names, in the order it
        names them, and a named directory that is not there is skipped.
        """
        checkout = Path(__file__).resolve().parents[1]
        out: list[Path] = [own for own in (checkout / name for name in self.default_names) if own.is_dir()]
        if self.default_names == (DEFAULT_ROOT_NAME,):
            for child in sorted(checkout.iterdir()):
                if child.name[:1] in ("_", ".") or not child.is_dir() or child.name == DEFAULT_ROOT_NAME:
                    continue
                nested = child / DEFAULT_ROOT_NAME
                if nested.is_dir():
                    out.append(nested)
        return out

    def _packaged(self) -> Path | None:
        if not self.packaged:
            return None
        candidate = Path(__file__).resolve().parent / self.packaged
        return candidate if candidate.is_dir() else None

    def _from_environment(self) -> list[Path]:
        raw = os.environ.get(self.env, "")
        return [Path(part).expanduser() for part in raw.split(os.pathsep) if part.strip()]

    def all(self) -> list[Path]:
        """Every root to scan, de-duplicated by real path, in declaration order.

        ``<env>_ONLY=1`` drops the checkout's and the wheel's own roots, so a
        copy of a shipped suite can be loaded under another root without
        colliding with the original -- which is how a converted suite is
        verified against the one it replaces.
        """
        out: list[Path] = []
        seen: set[str] = set()
        only = os.environ.get(f"{self.env}_ONLY", "") not in ("", "0")
        built_in = [] if only else [*self._checkouts(), *(e for e in (self._packaged(),) if e is not None)]
        for candidate in [*built_in, *self._from_environment(), *self.added]:
            try:
                key = str(candidate.resolve())
            except OSError:  # pragma: no cover -- an unreadable path is simply skipped
                continue
            if key not in seen:
                seen.add(key)
                out.append(candidate)
        return out

    def add(self, path: str | Path, *, what: str = "root", package: str = "") -> Path:
        """Declare a root for this process *and* for anything it spawns.

        Args:
            path: The directory. Must exist.
            what: What kind of root, for the error message.
            package: A package whose live ``__path__`` this root also belongs on,
                for the kinds that are reached by import rather than by scan.
                Extended here as well as returned, because ``__path__`` is built
                once when the package is imported and a root declared afterwards
                would otherwise be invisible to it.

                **A root added after a registry has already scanned is still not
                picked up**, because a registry loads once. That is not a gap
                today: the env var and the checkout's own directory are both read
                before anything scans, which is the whole of how content arrives.
                It becomes one the day something wants a ``--domain-root`` flag,
                and the fix then is a re-arm on the registry rather than more
                machinery here.

        Raises:
            FileNotFoundError: If ``path`` is not a directory. A root that is not
                there is almost always a typo, and finding out at the point of
                the mistake beats finding out as content nobody can name.
        """
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"{what} {path!s} is not a directory")
        if resolved not in self.added:
            self.added.append(resolved)
        live = sys.modules.get(package) if package else None
        if live is not None and hasattr(live, "__path__") and str(resolved) not in live.__path__:
            live.__path__ = extend_path(live.__path__, [resolved], kind=what)
        # Mirror into the environment so spawned workers inherit it. The pool
        # passes nothing else across the process boundary, so without this a
        # --task-root would be a parent-only fact and every worker would refuse
        # the task.
        current = [part for part in os.environ.get(self.env, "").split(os.pathsep) if part.strip()]
        if str(resolved) not in current:
            os.environ[self.env] = os.pathsep.join([*current, str(resolved)])
        return resolved

    def clear(self) -> None:
        """Forget every programmatically added root. For tests."""
        self.added.clear()


TASKS = _Roots(ROOTS_ENV, DEFAULT_ROOT_NAME)
"""Where task directories come from."""

DOMAINS = _Roots(DOMAIN_ROOTS_ENV, DEFAULT_DOMAIN_ROOT_NAMES, packaged="_domain_packs")
"""Where domain packs come from -- kinds of thing, out of tree."""

ENGINES = _Roots(ENGINE_ROOTS_ENV, DEFAULT_ENGINE_ROOT_NAME)
"""Where engine bundles come from -- foreign simulators, out of tree."""


def task_roots() -> list[Path]:
    """Every task root to scan, in declaration order."""
    return TASKS.all()


def suite_dir(name: str) -> Path:
    """The directory of one suite, whichever task root holds it.

    A suite's root is exactly the thing a collection move changes, so a tool or
    a test that spelt the root would go stale on the day a suite moved -- and
    pass on the stale path only by failing to find its fixture. Resolved against
    :func:`task_roots` in declaration order, the same order the registry scans,
    so the first root holding a directory of that name wins.
    """
    for root in task_roots():
        candidate = root / name
        if is_task_dir(candidate):
            return candidate
    raise FileNotFoundError(f"no task root holds a suite called {name!r}: roots are {task_roots()}")


def add_task_root(path: str | Path) -> Path:
    """Declare a task root for this process and anything it spawns."""
    return TASKS.add(path, what="task root")


def clear_task_roots() -> None:
    """Forget every programmatically added task root. For tests."""
    TASKS.clear()


def domain_roots() -> list[Path]:
    """Every domain root: directories holding packs -- packages or single modules -- that declare a kind of thing."""
    return DOMAINS.all()


def add_domain_root(path: str | Path) -> Path:
    """Declare a domain root for this process and anything it spawns."""
    return DOMAINS.add(path, what="domain root", package="robosimstudio.domains")


def clear_domain_roots() -> None:
    """Forget every programmatically added domain root. For tests."""
    DOMAINS.clear()


def engine_roots() -> list[Path]:
    """Every engine root: directories holding foreign-engine bundles."""
    return ENGINES.all()


def add_engine_root(path: str | Path) -> Path:
    """Declare an engine root for this process and anything it spawns."""
    return ENGINES.add(path, what="engine root", package="robosimstudio.engines")


def clear_engine_roots() -> None:
    """Forget every programmatically added engine root. For tests."""
    ENGINES.clear()


def extend_path(existing, roots, *, kind: str) -> list[str]:
    """A package's ``__path__``, plus every root that holds packs for it.

    What makes a library package extensible: ``robosimstudio.domains`` keeps the
    kinds this repository ships, and every root here contributes more, reached by
    the same import name. So ``robosimstudio.domains.cloth`` resolves whether
    cloth is in the library or on somebody's own root, and the registries'
    package scans -- which walk exactly this ``__path__`` -- find both in one
    sweep with no registry change at all.

    **A pack is a package or one module.** ``cloth/`` with an ``__init__.py`` and
    a single ``cloth.py`` are the same thing to the import system and to
    ``pkgutil.iter_modules``, so they are the same thing here: either is the name
    ``cloth``. A directory with no ``__init__.py`` is neither -- the scan would
    not yield it -- so what a moved pack leaves behind (an emptied directory
    still holding its ``__pycache__``) is not mistaken for a second copy.

    **A name that is not an identifier is not a pack.** A root may be an example
    folder -- ``examples/02_cloth`` holds ``cloth.py`` beside ``00_towel.py`` --
    and a numbered script is something a reader runs, not a kind: no ``import``
    statement can spell ``robosimstudio.domains.00_towel``, and the registries'
    scan skips the name. So it is left out of the rule below, and two folders
    that each hold a ``00_...`` script of the same name are not a clash.

    **A duplicate name across two roots raises, naming both.**
    ``pkgutil.iter_modules`` keeps a set of what it has already yielded and
    silently skips a repeat, so two roots each holding a ``cloth`` would not
    collide: the first would win and the second would never be imported, which
    is the failure the registries' duplicate rule exists to catch and which this
    would hide from it. The library's own directory is checked too -- shadowing
    a shipped kind is the same mistake.

    Args:
        existing: The package's ``__path__`` as Python built it.
        roots: Directories holding packs of this kind.
        kind: What they hold, for the error message.
    """
    out = [str(entry) for entry in existing]
    owner: dict[str, str] = {}
    for directory in [*(Path(e) for e in out), *roots]:
        if not directory.is_dir():
            continue
        for child in sorted(directory.iterdir()):
            if child.name.startswith(("_", ".")):
                continue
            if child.is_dir():
                if not (child / "__init__.py").is_file():
                    continue
                name = child.name
            elif child.suffix == ".py":
                name = child.stem
            else:
                continue
            if not name.isidentifier():
                continue  # ``00_towel.py``: an example script beside a pack, which no scan imports
            first = owner.get(name)
            if first is not None and first != str(directory):
                raise RuntimeError(
                    f"two {kind} roots both hold {name!r}: {first} and {directory}. "
                    "One name is one thing -- drop a root, or rename one of them."
                )
            owner.setdefault(name, str(directory))
        if str(directory) not in out:
            out.append(str(directory))
    return out


def is_task_dir(path: Path) -> bool:
    """Whether ``path`` is a task directory: a dir holding a document or a loader.

    A task is its documents, and the Python beside them is optional — but the
    converse also stands: a loader with no documents of its own is a *derived*
    suite, registering variants computed from another suite's (a study varying
    only an EpisodeSpec is the motivating case), and it has no document because
    the base task's document is not its to restate. A directory whose name
    begins with ``_`` or ``.`` is never one, so a root may hold private
    directories.
    """
    if not path.is_dir() or path.name.startswith(("_", ".")):
        return False
    return any(path.glob(f"*{TASK_SUFFIX}")) or (path / "task.py").is_file()
