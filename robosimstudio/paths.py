# SPDX-License-Identifier: Apache-2.0
"""Where vendored assets live, and how a spec's path resolves to a file.

Assets are repo furniture, not source: 78 MB of meshes and MJCF under the
checkout's ``assets/``, one directory per provenance, each recorded in
``PROVENANCE.md`` with where it came from and what changed on the way in. A spec
names a path relative to that root; an absolute path is passed through, which is
what lets a scratch asset be tried without vendoring it first.

**Most assets are not found here at all.** A task carries its own geometry
beside the declaration that owns it, and ``register_asset`` resolves those
against the record's home when it registers -- which is what makes a task
directory movable. What is left for this module is what *no* declaration owns:
the robot descriptions and the whole-scene exports, which belong to no task
because more than one stages them.

``ROBOSIMSTUDIO_ASSETS`` replaces the search outright, for a deployment that
keeps its large assets elsewhere entirely. Otherwise two places are searched:
the repo root's ``assets/``, and ``robosimstudio/assets/`` where a built wheel
force-includes a copy. Both are searched every lookup rather than stopping at
the first that exists, because a checkout has the first and an install has the
second and neither should pay for the other.
"""

from __future__ import annotations

import os
from pathlib import Path

from robosimstudio.specs import AssetSpec

__all__ = ["asset_root", "resolve_asset", "resolve_path"]

_ENV_VAR = "ROBOSIMSTUDIO_ASSETS"


def _candidate_roots() -> list[Path]:
    """Every place an asset could be, most specific first. There are two.

    ``assets/``
        Everything a declaration does not own: robot descriptions, gripper
        meshes, the scene-resident embodiments, and whole vendored exports whose
        internal paths cannot be rewritten. **Embodiments are not assets** --
        that is a standing rule here, not a filing accident.

    ``robosimstudio/assets/``
        Not a kind of thing, a kind of *install*: where the wheel
        force-includes a copy, because a wheel has no repo root to climb to. It
        is searched last because a checkout has no such directory and an install
        has nothing else, so neither pays for the other. It was omitted when
        this file grew its second root, which left ``pyproject.toml`` shipping a
        copy of the assets to a path nothing looked in: every scene build from
        an installed wheel failed to resolve its own meshes.

    **There is no content root, and there is no search order to know about.**
    This used to list ``tasks/<name>/assets/`` first, on the argument that a family's
    meshes sit beside the ``AssetRecord`` declaring them. They still do -- in
    ``tasks/<name>/assets/`` now -- but they are not found by searching for
    them: :func:`~robosimstudio.assetlib.registry.register_asset` rewrites a
    record's ``files`` to absolute paths at registration, from the directory the
    declaration itself was loaded from. That is what lets a task directory live
    in somebody else's repository, which a search root here could never do.

    ``ROBOSIMSTUDIO_ASSETS`` overrides both outright, which is how a deployment
    points the whole thing at a checkout somewhere else.
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        return [Path(override).expanduser().resolve()]
    return _with_collections([Path(__file__).parents[1] / "assets", Path(__file__).parents[0] / "assets"])


def _with_collections(roots: list[Path]) -> list[Path]:
    """``roots``, then the ``assets/`` beside every collection's task root.

    A collection (a top-level directory with a ``tasks/`` of its own -- see
    :mod:`robosimstudio.roots`) may ship exports its suites share, and ``(:export …)`` resolves through here:
    it is the one document clause that names a directory rather than a record,
    so a collection's exports have to be findable the way the checkout's are.
    After the checkout's own ``assets/`` and before the wheel's, in the order
    the task roots are declared; ``ROBOSIMSTUDIO_ASSETS`` still overrides the
    whole list, exactly as before.
    """
    if os.environ.get(_ENV_VAR):
        return roots
    from robosimstudio.roots import task_roots

    checkout = Path(__file__).resolve().parents[1]
    extra: list[Path] = []
    for root in task_roots():
        beside = root.parent / "assets"
        if root.parent != checkout and beside.is_dir() and beside not in roots and beside not in extra:
            extra.append(beside)
    packaged = [r for r in roots if r.parent == Path(__file__).resolve().parent]
    own = [r for r in roots if r not in packaged]
    return [*own, *extra, *packaged]


def asset_root() -> Path:
    """The directory vendored assets are resolved against.

    The first candidate that exists wins; if none does, the last is returned so
    that the error a caller gets names a concrete path rather than nothing.
    """
    roots = _candidate_roots()
    return next((root for root in roots if root.is_dir()), roots[-1])


def resolve_path(path: str | Path) -> Path:
    """Resolve one path against the asset roots, checking that it exists.

    **Every** root is searched, not just the first that happens to be a
    directory. That is the difference between two roots and one with a fallback:
    a content mesh lives beside its declaration and a robot description does not,
    so a lookup has to be able to find either without the caller knowing which
    kind of thing it asked for.
    """
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        if candidate.exists():
            return candidate.resolve()
    else:
        for root in _candidate_roots():
            found = root / candidate
            if found.exists():
                return found.resolve()
        candidate = asset_root() / candidate
    if not candidate.exists():
        searched = ", ".join(str(root) for root in _candidate_roots())
        raise FileNotFoundError(
            f"asset {path!s} not found at {candidate}. Vendored assets are looked for in {searched}; "
            f"set {_ENV_VAR} to point elsewhere."
        )
    return candidate.resolve()


def resolve_asset(spec: AssetSpec) -> Path:
    """Resolve an :class:`~robosimstudio.specs.AssetSpec` to a file."""
    return resolve_path(spec.path)
