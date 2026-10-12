# SPDX-License-Identifier: Apache-2.0
"""From a backend's name to a solver: resolve, register attributes, construct.

The mechanism half of :mod:`robosimstudio.specs.physics`'s backend registry.
The registry holds dotted paths so that registering imports nothing; this
module is where a path becomes a class, and it is the *only* place a solver
class is constructed in this package. A solver Newton ships tomorrow needs no
entry here: ``backend("newton.solvers.SolverNew", ...)`` resolves through
:func:`resolve_backend_class` like everything else.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

from robosimstudio.specs.physics import ArrangementSpec, BackendSpec, get_backend

from .errors import CompositionError

__all__ = [
    "build_solver",
    "effective_backend",
    "register_custom_attributes",
    "resolve_backend_class",
    "solver_class_path",
]


def solver_class_path(spec: BackendSpec) -> str:
    """The dotted path of the solver class a backend names."""
    return get_backend(spec.solver).cls


def resolve_backend_class(spec: BackendSpec) -> type:
    """Import and return the solver class a backend names.

    Raises:
        CompositionError: When the alias is unknown or the path does not import,
            naming what was tried.
    """
    try:
        path = solver_class_path(spec)
    except KeyError as exc:
        raise CompositionError(str(exc)) from exc
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        raise CompositionError(f"backend {spec.solver!r}: {path!r} is not a dotted path to a class")
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, attr)
    except (ImportError, AttributeError) as exc:
        raise CompositionError(
            f"backend {spec.solver!r}: cannot import {path!r} ({type(exc).__name__}: {exc})"
        ) from exc
    return cls


def register_custom_attributes(
    builder: Any, arrangement: ArrangementSpec, extra: Iterable[BackendSpec] = ()
) -> list[str]:
    """Call ``register_custom_attributes(builder)`` on every backend class the arrangement names.

    **Before any entity builds.** MuJoCo's ``gravcomp`` and VBD's per-joint
    flags are custom attributes an importer writes *if they are registered*;
    registered after, the importer has already dropped them and the arm sags.
    Done here, from the arrangement, so no entity has to import a solver to
    know which attributes its builder calls will be allowed to keep.

    Returns:
        The dotted paths registered, in order, for a reader.
    """
    seen: list[str] = []
    specs = [spec for _, spec in arrangement.backends] + list(extra)
    for spec in specs:
        try:
            path = solver_class_path(spec)
        except KeyError as exc:
            raise CompositionError(str(exc)) from exc
        if path in seen:
            continue
        seen.append(path)
        cls = resolve_backend_class(spec)
        register = getattr(cls, "register_custom_attributes", None)
        if register is not None:
            register(builder)
    return seen


def effective_backend(spec: BackendSpec, *, has_particles: bool = True) -> BackendSpec:
    """``spec`` with its alias's registered defaults filled in underneath.

    The parameters the solver is actually constructed with, and what the
    manifest records: a dotted-path backend has no defaults and comes back as
    stated.
    """
    return get_backend(spec.solver).effective(spec, has_particles=has_particles)


def build_solver(spec: BackendSpec, view: Any, *, has_particles: bool = True) -> Any:
    """Construct the solver over ``view`` with the spec's effective parameters.

    Raises:
        CompositionError: Wrapping a constructor ``TypeError`` -- a misspelt
            parameter -- with the backend's name and the parameters it was
            given, which is the line somebody debugging a staging file wants.
    """
    cls = resolve_backend_class(spec)
    kwargs = effective_backend(spec, has_particles=has_particles).kwargs()
    try:
        return cls(model=view, **kwargs)
    except TypeError as exc:
        raise CompositionError(
            f"backend {spec.solver!r} ({cls.__module__}.{cls.__qualname__}) refused its parameters "
            f"{sorted(kwargs)}: {exc}"
        ) from exc
