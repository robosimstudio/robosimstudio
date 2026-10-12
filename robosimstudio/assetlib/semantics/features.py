# SPDX-License-Identifier: Apache-2.0
"""Features: what an object offers, declared per object.

A feature answers one question -- *"how do I find this, given the rest
configuration?"* -- and the answer comes in three tiers, forming a spectrum from
data to code:

``literal``
    A constant, produced offline by a tool or a human and frozen in. Where most
    rigid features live: a loop's centre, normal and inner radius are three
    numbers. The degenerate case of the general form, not a different species --
    the expression simply collapsed to a number before it was written down.

``recipe``
    An expression in the closed selector language
    (:mod:`~robosimstudio.assetlib.semantics.selectors`), evaluated **once** at
    scene-ready against the rest layout. Where deformable features live, and
    robust exactly where baked indices are brittle: re-mesh the shirt and
    ``half(axis: y, side: -1)`` still means the left half.

``resolver``
    A reference to registered, versioned code, for a selection the closed
    language genuinely cannot express -- the collar curve of an arbitrary garment
    mesh, a segmentation-derived patch. Governed by the same mint gate as atoms,
    and often better run in the asset pipeline and baked down to ``literal`` with
    provenance.

The hard rule across all three: **nothing executable is ever embedded in a
sidecar.** No lambdas in YAML, no inline snippets. Sidecars carry constants,
recipes and *references*; registries and pipeline tools carry code; the compiler
joins them at load time. That is what keeps an agent-authored annotation
validatable and an agent loop auditable.

One discipline is worth stating loudly because violating it fails silently.
**Anything that references discretisation must pin the mesh hash.** A literal
list of particle indices, or a recipe containing an ``indices`` op, means nothing
the moment somebody re-meshes the asset -- and the shirt in this repo *was*
re-meshed, deliberately, with the decimation trade recorded in its suite. An
unpinned baked index is how an annotation rots without anybody noticing, so
:meth:`FeatureSpec.problems` refuses it rather than warning.

Features are frozen and JSON-able. They are deliberately **not** hash keys: their
payloads are nested JSON, and deep-freezing them to buy a hash nothing needs
would be complexity for its own sake. The compiler keys on the task, as it
always has.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from .kinds import get_kind, is_registered_kind
from .selectors import round_floats, selector_to_dict, validate_selector

__all__ = ["TIERS", "FeatureSpec", "feature_from_dict", "feature_to_dict"]

TIERS: tuple[str, ...] = ("literal", "recipe", "resolver")
"""How a feature is resolved, outermost first. A recipe carrying a measured
constant is still a ``recipe``; the tier names the mechanism the compiler must
run, not the provenance of every number inside it."""


@dataclass(frozen=True, kw_only=True)
class FeatureSpec:
    """One named, typed attribute of one object.

    Attributes:
        name: How the feature is referenced -- the ``hole`` in ``scissors.hole``.
        kind: Its feature kind (``loop``, ``rod``, ``region``, ``set``, ...). A
            plain string: the roster is open, and an unregistered kind is legal
            but uncontracted.
        tier: One of :data:`TIERS`.
        payload: The tier's content -- constants for ``literal``, a selector
            recipe under ``recipe``, a ``{"resolver": "name@v1"}`` reference for
            ``resolver``. Checked against the kind's declared payload when the
            kind is registered.
        mesh_sha: Hash of the rest layout this feature was authored against.
            Required whenever the feature references discretisation.
        provenance: Where the numbers came from -- measured off which file, baked
            by which tool, transcribed from which reference.
    """

    name: str
    kind: str
    tier: str = "literal"
    payload: dict[str, Any] = field(default_factory=dict)
    mesh_sha: str = ""
    provenance: str = ""

    @property
    def references_discretisation(self) -> bool:
        """Whether resolving this feature depends on the exact mesh layout.

        True for a literal index list and for any recipe containing an
        ``indices`` op, at any depth. Geometric recipes are false: they are
        stated in metres against the rest extent and survive a re-mesh, which is
        the entire reason to prefer them.
        """
        if self.tier == "literal":
            return any(key in self.payload for key in ("indices", "particles", "vertices"))
        if self.tier == "recipe":
            return _mentions_indices(self.payload.get("select", self.payload))
        return False

    def problems(self) -> list[str]:
        """Everything wrong with this feature, as prose, or an empty list.

        Structural only -- no sheet, no model, no GPU. What the gauntlet's schema
        and typecheck stages call, so an agent-authored sidecar is refused for
        being malformed long before anything is built.
        """
        where = f"feature {self.name!r}"
        out: list[str] = []
        if not self.name:
            out.append("feature has no name")
        if not self.kind:
            out.append(f"{where}: no kind declared")
        if self.tier not in TIERS:
            out.append(f"{where}: unknown tier {self.tier!r}; one of {sorted(TIERS)}")

        if self.tier == "recipe":
            out += validate_selector(self.payload.get("select", self.payload), where=where)
        elif self.tier == "resolver" and not str(self.payload.get("resolver", "")):
            out.append(f"{where}: tier 'resolver' needs a 'resolver' key naming registered code")

        if self.references_discretisation and not self.mesh_sha:
            out.append(
                f"{where}: references particle indices but pins no mesh_sha -- "
                "a baked index means nothing after a re-mesh, and rots silently"
            )

        if is_registered_kind(self.kind) and self.tier == "literal":
            kind = get_kind(self.kind)
            missing = kind.missing(self.payload)
            unknown = kind.unknown(self.payload)
            if missing:
                out.append(f"{where}: kind {self.kind!r} requires {missing}")
            if unknown:
                out.append(f"{where}: kind {self.kind!r} has no field(s) {unknown}")
        return out


def _mentions_indices(recipe: Any) -> bool:
    """Whether a selector recipe uses a literal index list anywhere inside it."""
    if isinstance(recipe, (list, tuple)):
        return any(_mentions_indices(item) for item in recipe)
    if not isinstance(recipe, dict):
        return False
    if recipe.get("op") == "indices":
        return True
    return _mentions_indices(recipe.get("of", []))


def feature_to_dict(spec: FeatureSpec) -> dict[str, Any]:
    """A feature as plain JSON-able data, omitting empty optional fields.

    A recipe payload goes through the selector's own canonical form, so axis
    names become indices and derived constants stop printing as binary noise --
    ``WIDTH / 6`` is exactly ``0.06999999999999999`` in float, and a catalog that
    says so is one a reviewer mistrusts and a golden diff full of nothing.
    """
    out: dict[str, Any] = {"name": spec.name, "kind": spec.kind, "tier": spec.tier}
    if spec.payload:
        payload = round_floats(dict(spec.payload))
        if spec.tier == "recipe" and "select" in payload:
            payload["select"] = selector_to_dict(spec.payload["select"])
        out["payload"] = payload
    if spec.mesh_sha:
        out["mesh_sha"] = spec.mesh_sha
    if spec.provenance:
        out["provenance"] = spec.provenance
    return out


def feature_from_dict(data: dict[str, Any], *, where: str = "feature") -> FeatureSpec:
    """Rebuild a feature from :func:`feature_to_dict` output.

    Unknown keys are refused rather than dropped, for the reason
    :func:`~robosimstudio.assetlib.materials._build` gives about material fields:
    a silently ignored typo produces something that quietly kept a default.
    """
    known = {f.name for f in fields(FeatureSpec)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {unknown}; FeatureSpec has {sorted(known)}")
    return FeatureSpec(**data)
