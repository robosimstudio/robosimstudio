# SPDX-License-Identifier: Apache-2.0
"""The one door between the page and the benchmark tier.

The page is one thing over two kinds of scene. Over a scene somebody assembled
by hand -- ``rss.Scene(...).build()`` -- it needs the library and nothing else.
Over a *task* it also needs what a task is made of: the registry that resolves
an id, the ``Env`` that compiles a document, the goal tree, the shipped
policies, a fleet of worlds. That second half is the benchmark tier, which is a
package beside this one and is not in every release.

So everything on the page that names the benchmark lives in files of its own --
:data:`ADDONS` -- and every other module reaches them through this one, lazily
and by asking. Two ways of asking, because there are two honest answers to "it
is not here":

* :func:`benchmark` **refuses**, with :class:`NotInThisRelease` -- one sentence
  saying what was asked for and what it needs. For a request that cannot be
  served without the tier: loading a task id, building a task menu.
* :func:`benchmark_or_none` **shrugs**. For a decoration that is simply absent
  without the tier and wrong to complain about: tinting a drawn feature by the
  goal clause that is about it, on a page whose scene has no goal.

The add-on modules are imported *inside* these functions, never at this
module's scope, which is what lets ``import robosimstudio_studio`` succeed on a
tree that does not hold them. Nothing else in the package may import them.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any

__all__ = ["ADDONS", "WITHHELD", "NotInThisRelease", "benchmark", "benchmark_or_none", "lazy_names"]

WITHHELD: tuple[str, ...] = ("robosimstudio_bench", "robosimstudio_policy_agent", "robosimstudio_repro")
"""The packages beside the library that a release may leave out.

Named here as strings, to recognise *their* absence in a failed import and tell
it apart from any other missing module -- a typo in an add-on, an optional
backend that is not installed -- which must keep its own traceback.
"""

ADDONS: tuple[str, ...] = ("bench_session", "bench_panels", "goal_panel", "fleet_session", "fleet_view")
"""This package's own modules that exist only where the benchmark tier does.

``bench_session`` is the door's other side: task loading, the fleet page, the
goal-aware halves of the session and the overlay. ``bench_panels`` is the task
menu and the policy runner; ``goal_panel`` the goal readout; ``fleet_session``
and ``fleet_view`` the grid of worlds. A release without the tier leaves these
five files out by path, and this list is that path list's twin.
"""


class NotInThisRelease(ImportError):
    """Something was asked of the page that needs the benchmark tier, and it is absent.

    An ``ImportError`` because that is what it is -- an optional part that is
    not installed -- and because of how Python reports one: raised from a
    module's ``__getattr__`` it survives ``from robosimstudio_studio import
    PanelTask`` with its sentence intact, where an ``AttributeError`` would be
    replaced by "cannot import name".
    """


def _is_the_tier(exc: ModuleNotFoundError) -> bool:
    """Whether a failed import failed because the tier, or an add-on file, is absent."""
    name = exc.name or ""
    if name.split(".", 1)[0] in WITHHELD:
        return True
    package, _, module = name.rpartition(".")
    return package == __package__ and module in ADDONS


def _addon(module: str, what: str) -> ModuleType:
    """Import one of :data:`ADDONS`, or say in one sentence why it cannot be."""
    try:
        return importlib.import_module(f"{__package__}.{module}")
    except ModuleNotFoundError as exc:
        if not _is_the_tier(exc):
            raise
        raise NotInThisRelease(
            f"{what} needs the benchmark tier (robosimstudio_bench), which is not in this release yet. "
            "The page itself needs no task: build a scene with rss.Scene(...).build() and open it with rss.view(sim).",
            name=exc.name,
        ) from None


def benchmark(what: str) -> ModuleType:
    """The page's benchmark half, imported on first use.

    Args:
        what: What is being asked for, as the subject of the sentence a refusal
            is written in -- ``"loading the task 'tool_sort/level2_1'"``.

    Returns:
        The ``robosimstudio_studio.bench_session`` module.

    Raises:
        NotInThisRelease: If the tier is absent. Any *other* import failure
            inside the add-on propagates as itself.
    """
    return _addon("bench_session", what)


def benchmark_or_none() -> ModuleType | None:
    """The page's benchmark half, or ``None`` where the tier is absent.

    For a caller with a correct thing to do without it -- draw in the plain
    colour, skip the readout. A caller that cannot proceed wants
    :func:`benchmark`, whose refusal says so.
    """
    try:
        return _addon("bench_session", "this")
    except NotInThisRelease:
        return None


def lazy_names(module: str, names: Mapping[str, str]) -> Callable[[str], Any]:
    """A :pep:`562` ``__getattr__`` resolving ``names`` out of the add-on modules.

    How ``robosimstudio_studio.PanelTask`` and
    ``robosimstudio_studio.controls.PanelTask`` go on meaning what they always
    did where the tier is installed, without the module that says so importing
    it: the lookup happens on attribute access, never during the import.

    Args:
        module: The dotted name of the module this ``__getattr__`` is for.
        names: Attribute name -> which of :data:`ADDONS` defines it.

    Returns:
        The function to bind as that module's ``__getattr__``. It raises
        :class:`NotInThisRelease` for a listed name the release does not carry,
        and ``AttributeError`` for any name not listed, as a module does.
    """

    def __getattr__(name: str) -> Any:
        home = names.get(name)
        if home is None:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        return getattr(_addon(home, f"{module}.{name}"), name)

    return __getattr__
