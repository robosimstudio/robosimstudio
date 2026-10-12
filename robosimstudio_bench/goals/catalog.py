# SPDX-License-Identifier: Apache-2.0
"""The catalog: what exists, answered for the checker and for an agent.

Two audiences, one source. The **typechecker** asks narrow questions -- does this
asset exist, does it have that feature, what kind is it -- through the protocol
it declared, which this satisfies. An **authoring agent** asks the broad one:
what is there, and what can I say about it.

This lives in the asset library rather than in the compiler because it is the
asset library's own content. The compiler never imports it; it is handed in. That
inversion is what lets the layer law hold while the checker still gets real
answers -- and it means the checker's tests can hand it a five-line fake instead
of registering a world.

Nothing here computes. Every answer is a lookup into a registry something else
populated, which is why the catalog cannot drift from what the simulator would
actually build.
"""

from __future__ import annotations

from typing import Any

from robosimstudio.assetlib.record import asset_to_dict
from robosimstudio.assetlib.registry import get_asset, list_assets
from robosimstudio.assetlib.semantics import (
    HANDLERS,
    atom_to_dict,
    get_atom,
    get_kind,
    list_atoms,
    list_handlers,
    list_kinds,
)
from robosimstudio.assetlib.semantics.substrates import SUBSTRATES
from robosimstudio.scene.registry import get_cell, list_cells
from robosimstudio_bench.taskdoc.parse import COMPARISONS, CONNECTIVES, MODALS

from .combinators import combinator_names

__all__ = ["AssetCatalog", "catalog_json"]


class AssetCatalog:
    """The live registries, behind the checker's protocol.

    Satisfies :class:`~robosimstudio_bench.taskdoc.typecheck.Catalog`
    structurally -- no import in either direction, no base class, which is the
    same duck-typed arrangement the handlers use and for the same reason.
    """

    def has_asset(self, name: str) -> bool:
        return name in list_assets()

    def features_of(self, asset: str) -> dict[str, str]:
        """Feature name to kind, under both spellings. Empty for an unknown asset."""
        if not self.has_asset(asset):
            return {}
        out: dict[str, str] = {}
        for spec in get_asset(asset).features:
            out[spec.name] = spec.kind
            out[spec.name.replace("_", "-")] = spec.kind
        return out

    def states_of(self, asset: str) -> tuple[str, ...]:
        """Every word this asset defines: semantic states **and published states**.

        Both are things a task may name, and neither is an atom. A semantic state
        (``adhered``, ``open``) is the asset's own named threshold; a published
        state (``bond_fraction``, ``peel_front``) is internal state it exposes,
        which only this asset can compute and which a goal legitimately
        thresholds.

        Leaving published states out was a real hole, found by running the
        checker against the tape task: its goal reads
        ``(<= (bond-fraction tape) 0.05)``, and a catalog that knew only about
        atoms and semantic states reported the one quantity the whole task is
        scored on as "unresolved".
        """
        if not self.has_asset(asset):
            return ()
        record = get_asset(asset)
        names = [state.name for state in record.states] + [member.name for member in record.published]
        return tuple(dict.fromkeys(names + [name.replace("_", "-") for name in names]))

    def atom(self, name: str) -> Any | None:
        """One atom's spec, or ``None`` -- the checker distinguishes those.

        Tried under both spellings. BDDL is a lisp dialect and writes
        ``bond-fraction``; Python identifiers cannot, and the atom registry is
        keyed by the function's own name. Normalising here rather than at either
        end keeps the task language idiomatic and the code idiomatic, and puts
        the one place they disagree in the adapter that exists to bridge them.
        """
        for candidate in (name, name.replace("-", "_")):
            try:
                return get_atom(candidate)
            except KeyError:
                continue
        return None


def catalog_json() -> dict[str, Any]:
    """Everything an authoring agent needs, as plain data.

    The broad view: substrates, the frozen combinator set, every feature kind
    with its contracts, every atom with its signature and calibration, and every
    asset. This is what an agent reads before writing a task -- so it names
    things that exist rather than things it hoped for.
    """
    return {
        "substrates": {name: list(chain) for name, chain in SUBSTRATES},
        "combinators": combinator_names(),
        "feature_kinds": {name: _kind(name) for name in list_kinds()},
        "atoms": {name: atom_to_dict(get_atom(name)) for name in list_atoms()},
        "handlers": _handlers(),
        "assets": {name: asset_to_dict(get_asset(name)) for name in list_assets()},
        "grammar": _grammar(),
        "scenes": _scenes(),
    }


def _kind(name: str) -> dict[str, Any]:
    """One feature kind, including how its payload reaches world coordinates.

    ``points`` and ``directions`` were the omission worth fixing. They are what
    :class:`~..semantics.kinds.FeatureKind` declares them for -- a point must be
    rotated *and* translated, a direction must be rotated and must **not** be --
    and without them the catalog describes a ``cavity``'s ``axis`` and its
    ``mouth`` identically, as two lists of three numbers. Anything drawing a
    feature, or reading one outside the process that resolves it, has to know
    which is which, and the kind's own docstring says what guessing costs:
    translate a direction and the unit vector becomes a position, which
    normalises to something pointing at the body rather than along the bore.
    """
    kind = get_kind(name)
    out: dict[str, Any] = {
        "summary": kind.summary,
        "payload": kind.payload,
        "optional": kind.optional,
        "contracts": list(kind.contracts),
        "substrates": list(kind.substrates),
        "provenance": kind.provenance,
    }
    # Omitted when empty rather than sent as empty lists: most kinds carry no
    # geometry at all, and two always-present empty fields on eleven of thirteen
    # entries is noise in the one document an agent reads end to end.
    if kind.points:
        out["points"] = list(kind.points)
    if kind.directions:
        out["directions"] = list(kind.directions)
    return out


def _handlers() -> dict[str, str]:
    """Every registered implementation, and what evidence each one uses.

    A list of keys said which signatures are implemented and not what any of them
    *does*, and the difference between two implementations of one atom is exactly
    what a reader needs: ``containment`` over a rigid body is its centre against
    the box, and over a garment it is the share of particles inside. Same
    question, different evidence, and only the summary says so.
    """
    return {key: HANDLERS.get(key).summary for key in list_handlers()}


def _grammar() -> dict[str, Any]:
    """The closed surface syntax, read off the parser rather than restated.

    An agent that knows what a ``cavity`` affords still cannot write a goal
    without knowing that `and`, `or`, `any`, `not` and `worst_of` are the
    connectives and that `peak` only appears inside a comparison. That was
    written down in the spec and nowhere a program could read it, so an agent had
    to have memorised the grammar or parse English prose to find it.

    Read from :mod:`~robosimstudio_bench.taskdoc.parse`'s own tuples, so
    the catalog cannot claim a form the parser refuses -- which is the failure
    this repository has now hit three times in documentation.
    """
    return {
        "connectives": list(CONNECTIVES),
        "comparisons": list(COMPARISONS),
        "modals": list(MODALS),
        # `peak` and `integral` report a quantity rather than a verdict, so a
        # bare one is refused by the goal builder. Stated as data because the
        # distinction is not recoverable from the name.
        "quantity_modals": ["peak", "integral"],
        "reward_clauses": ["fraction", "progress", "shrinking"],
        "reward_modifiers": ["gate", "gated-by"],
        "selector": "atom:key selects one of a multi-valued measure's returns",
    }


def _scenes() -> dict[str, Any]:
    """Every cell a task may be staged in, and the surface it stands on.

    A document's ``(:scene ...)`` has to name one of these, and an agent that
    cannot enumerate them is guessing. Only the fields an author needs: where the
    work surface is and how big it is, because those are what a region's ranges
    and an object's pose are stated against.
    """
    out: dict[str, Any] = {}
    for name in list_cells():
        scene = get_cell(name)
        out[name] = {
            "surface_z": float(scene.surface_z),
            "center": [float(v) for v in scene.center],
            # `half` is optional and four scenes leave it unset -- a cell whose
            # extent comes from an imported stage rather than from a declared
            # box. `None` is the honest answer; a zero would read as a surface
            # with no area, which is a different claim.
            "half": [float(v) for v in scene.half] if scene.half is not None else None,
            "mounts": sorted(scene.mounts) if isinstance(scene.mounts, dict) else list(scene.mounts),
            "provenance": scene.provenance,
        }
    return out
