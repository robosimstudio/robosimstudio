# SPDX-License-Identifier: Apache-2.0
"""The registry every other registry is an instance of.

A name-to-value map that refuses to be written twice, loads the packages that
populate it lazily, and guards against re-entry while it does. It lives at the
top of the package, importing nothing but the standard library, because every
layer has something to register and none of them should have to reach across a
layer boundary for the mechanism.

Three deliberate choices, unchanged since this was one file:

* **A duplicate raises.** Registering two things under one name is a mistake --
  usually a copy-pasted module -- and warning about it means the second one
  silently loses, at whichever point in an import order nobody controls. So it
  is an error, with both definition sites in the message.
* **Loading is lazy, with a re-entry guard.** Nothing is in a registry until a
  lookup needs it, and the modules that populate one register *while* being
  imported, so a lookup fired from inside one must not re-enter the load. The
  guard is what makes registering at module scope safe.
* **Registration is by value, not by class.** What is registered is an instance
  -- a spec, a suite -- rather than a class to be instantiated. Python-first
  config means a module *constructs* what it hands over.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Iterator
from typing import Any

__all__ = ["MISSING", "Registry", "unwatch", "watch"]

MISSING: Any = object()
"""What :func:`watch` is told a name held before it was first registered."""

Watcher = Callable[["Registry", str, Any, str, Any, str], None]
"""``fn(registry, name, value, origin, previous, previous_origin)``."""

_WATCHERS: list[Watcher] = []


def watch(fn: Watcher) -> Watcher:
    """Call ``fn(registry, name, value, origin, previous, previous_origin)`` after every write to any registry.

    ``value`` and ``origin`` are what the name holds now; ``previous`` is
    :data:`MISSING` for a name registered for the first time, and the value it
    replaced for a :meth:`Registry.replace`. A redundant copy
    that :meth:`Registry.register` hands back the first for writes nothing and
    is not reported. For a process that runs one file over and over -- the live
    page, rebuilding ``scene.py`` on every save -- and must take back what the
    last run registered before the next one registers it again, edited.
    """
    _WATCHERS.append(fn)
    return fn


def unwatch(fn: Watcher) -> None:
    """Stop calling ``fn``; a function not watching is ignored."""
    if fn in _WATCHERS:
        _WATCHERS.remove(fn)


def _notify(registry: Registry, name: str, previous: Any, previous_origin: str) -> None:
    value, origin = registry._items[name], registry._origins.get(name, "")
    for fn in list(_WATCHERS):
        fn(registry, name, value, origin, previous, previous_origin)


class Registry[T]:
    """A name-to-value map that refuses to be written twice.

    Args:
        kind: What it holds, used in error messages.
        packages: Modules to import on :meth:`load`, each a package whose
            submodules register on import. A package that is not installed is
            skipped: the kinds the library ships as a sibling package
            (:data:`robosimstudio.roots.DOMAIN_PACKAGES`) may be left out of a
            tree, and a scan that raised on their absence would make every
            lookup fail instead of only the names they provide.
        hint: What to add to a lookup miss, for a registry whose entries do not
            all arrive from the same place. ``CELLS`` holds four benches the
            library ships and a handful a task directory declares, and the
            second sort appears only once something has imported the benchmark
            -- so a miss there is usually a question of *when*, not of *what*,
            and the message should say which.
        same: How to tell whether a second registration under one name is the
            *same thing* rather than a clash. ``None`` refuses every duplicate.

            **This is what lets two tasks each carry their own copy.** A task is
            self-contained on purpose, so two of them staging the same cell each
            ship its declaration -- and a rule that refused the second would
            force the very shared directory the layout is trying to do without.
            What the rule is actually for survives: two *different* things
            claiming one name is still an error, named with both origins.
    """

    def __init__(
        self,
        kind: str,
        packages: tuple[str, ...] = (),
        *,
        hint: str = "",
        same: Callable[[T, T], bool] | None = None,
    ) -> None:
        self.kind = kind
        self.packages = packages
        self.hint = hint
        self.same = same
        self._discover: list[Callable[[], None]] = []
        self._items: dict[str, T] = {}
        self._origins: dict[str, str] = {}
        self._loaded = False
        self._loading = False

    def register(self, name: str, value: T, *, origin: str = "") -> T:
        if name in self._items:
            first = self._items[name]
            if self.same is not None and self.same(first, value):
                # A redundant copy, which is the point: the first one registered
                # and this one says the same thing. Returning the *registered*
                # object rather than the newcomer matters -- everything already
                # holding a reference is holding the first, and handing back a
                # twin would give two callers objects that compare equal and are
                # not identical.
                return first
            raise ValueError(
                f"{self.kind} {name!r} is already registered"
                + (f" (first from {self._origins[name]}, now from {origin})" if origin else "")
                + ". Two different things may not share a name; an identical copy is fine."
            )
        self._items[name] = value
        self._origins[name] = origin
        if _WATCHERS:
            _notify(self, name, MISSING, "")
        return value

    def replace(self, name: str, value: T, *, origin: str = "") -> T:
        """Supersede a registered value with one that includes it.

        The one write after a register. A task suite that a later root *extends*
        with more variants is re-registered as the grown suite, and everything
        that looks it up by name from then on sees the whole. Refuses a name
        nobody registered: an extension of nothing is a typo, and this is
        where it is caught.
        """
        if name not in self._items:
            raise KeyError(f"{self.kind} {name!r} is not registered, so there is nothing to replace")
        previous, previous_origin = self._items[name], self._origins.get(name, "")
        self._items[name] = value
        if origin:
            first = self._origins.get(name, "")
            self._origins[name] = f"{first} + {origin}" if first else origin
        if _WATCHERS:
            _notify(self, name, previous, previous_origin)
        return value

    def forget(self, name: str) -> None:
        """Take a name out, as if it had never been registered; a name not there is ignored.

        Not for content: a lookup that worked a moment ago failing is a worse
        bug than any duplicate. For the one process that re-runs a file which
        registers things -- the live page -- to take back what the last run
        put in before the next run puts in the edited version (see :func:`watch`).
        """
        self._items.pop(name, None)
        self._origins.pop(name, None)

    def restore(self, name: str, value: T, origin: str = "") -> None:
        """Put a value back under a name, with no duplicate check: the inverse of a :meth:`replace` taken back."""
        self._items[name] = value
        self._origins[name] = origin

    @property
    def discovers(self) -> tuple[Callable[[], None], ...]:
        """The populators installed on this registry, in the order they run.

        Public because it is a real question about a registry -- *does anything
        find content for this?* -- and one the content-seam test asks. A registry
        that scans no package and has no populator finds nothing at all, which is
        a mistake that looks exactly like an empty corpus.
        """
        return tuple(self._discover)

    def add_discover(self, fn: Callable[[], None]) -> Callable[[], None]:
        """Register another populator, run on the next :meth:`load`.

        A registry is a **mechanism**, and it belongs with the layer whose things
        it holds. *Where those things come from* is a different fact, and it
        belongs with the layer that knows about content. Four registries used to
        close that gap by reaching upward from inside a function -- ``scene``,
        ``robots`` and ``assetlib`` each held an import of the discovery
        machinery a layer above them -- and the cost was not the import. It was
        that asking for a **library** cell executed every task directory on
        disk: ``get_cell("bench_075")`` is a question about
        ``scene/builtin/benches.py`` and it loaded thirty thousand lines of
        benchmark content to answer it.

        Additive rather than a constructor argument, because more than one
        source is a real case: task roots today, and a caller pointing a process
        at a second body of content without editing the environment.

        **Adding a hook re-arms the load**, and that is not tidiness. Without it,
        a registry a lookup had already loaded would silently never see the new
        source -- importing the benchmark after a bare ``get_cell`` would install
        a hook that never ran, and the failure would be a task nobody could name
        with no error anywhere. The re-entry guard is what keeps re-arming safe:
        a lookup fired from inside a hook finds the load already in progress and
        does not restart it.
        """
        self._discover.append(fn)
        self._loaded = False
        return fn

    def load(self) -> None:
        """Import every registering module, once.

        Re-entrant calls return immediately, which is what lets a lookup fired
        from inside a suite module being imported not re-enter the load.
        """
        if self._loaded or self._loading:
            return
        self._loading = True
        try:
            for package_name in self.packages:
                try:
                    package = importlib.import_module(package_name)
                except ModuleNotFoundError as error:
                    # A scanned package a tree may leave out (``robosimstudio_materials``): its
                    # registrations are simply not there. A module *inside* a package that is
                    # there, failing to import, is a real error and is not caught here.
                    if error.name == package_name:
                        continue
                    raise
                for info in pkgutil.iter_modules(package.__path__):
                    # A domain root may be an example folder (``00_pour.py`` beside ``granular.py``): scripts, not registering modules.
                    if not info.name.isidentifier():
                        continue
                    importlib.import_module(f"{package_name}.{info.name}")
            for discover in self._discover:
                discover()
            self._loaded = True
        finally:
            self._loading = False

    def get(self, name: str) -> T:
        self.load()
        if name not in self._items:
            raise KeyError(
                f"unknown {self.kind} {name!r}; registered: {sorted(self._items)}"
                + (f". {self.hint}" if self.hint else "")
            )
        return self._items[name]

    def names(self) -> list[str]:
        self.load()
        return sorted(self._items)

    def __getitem__(self, name: str) -> T:
        return self.get(name)

    def keys(self) -> list[str]:
        return self.names()

    def values(self) -> list[T]:
        self.load()
        return [self._items[name] for name in sorted(self._items)]

    def items(self) -> list[tuple[str, T]]:
        self.load()
        return sorted(self._items.items())

    def __contains__(self, name: str) -> bool:
        self.load()
        return name in self._items

    def __iter__(self) -> Iterator[tuple[str, T]]:
        self.load()
        yield from sorted(self._items.items())

    def __len__(self) -> int:
        self.load()
        return len(self._items)


def same_source(first: Any, second: Any) -> bool:
    """Whether two callables or classes are one definition shipped twice.

    Compared by qualified name and the last component of the module name
    rather than by identity: a redundant copy -- a task directory executed
    under two synthetic package names, an entity class two tasks both ship --
    is a *different object* built from identical source, so ``is`` and ``==``
    both say no and neither is the question a registry asks.
    """
    return (getattr(first, "__module__", "").rpartition(".")[2], getattr(first, "__qualname__", None)) == (
        getattr(second, "__module__", "").rpartition(".")[2],
        getattr(second, "__qualname__", None),
    )


def register_alias(table: dict[str, str], kind: str, alias: str, cls: str | type) -> str:
    """Register ``cls`` under ``alias`` in a table of dotted paths. Identical re-registration is accepted.

    The tables that hold implementations by *path* -- render backends, control
    modes -- stay dictionaries of strings so that listing them costs no import;
    this is their one registrar.

    Args:
        table: The alias -> dotted path table.
        kind: What the table holds, for the error.
        alias: The name a caller will use.
        cls: The class, or its dotted path. A class is recorded **by path**.

    Raises:
        ValueError: If ``alias`` is taken by a different path.
    """
    path = cls if isinstance(cls, str) else f"{cls.__module__}.{cls.__qualname__}"
    first = table.get(alias)
    if first is not None and first != path:
        raise ValueError(
            f"{kind} {alias!r} is already registered as {first!r}, not {path!r}. "
            "Two different things may not share a name; an identical copy is fine."
        )
    table[alias] = path
    return path
