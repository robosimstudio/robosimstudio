# SPDX-License-Identifier: Apache-2.0
"""Declaring a rigid body: constructors named for what you already have.

A rigid body is the commonest thing in the corpus by a wide margin -- 47 of its
66 assets -- and declaring one was four names every time::

    register_asset(AssetRecord(
        name=..., summary=..., tags=...,
        sim=SimRequirements(domain="rigid",
                            contacts=contacts_with("surface", "fixtures", "robot")),
        files={"mesh": ...},
    ))

Two suites had already written that helper for themselves, four times between
them -- two sorting suites' ``_item``/``_tool`` helpers -- and all
four agreed *exactly* on the two things this supplies: ``domain="rigid"`` and
``contacts_with("surface", "fixtures", "robot")``, which 37 of the 47 declare.
What the four did not agree on was ``notes``, and two of them had drifted into
being byte-identical anyway. So the domain and the contact set are the
mechanism and travel here; the notes are the recipe and stay with the asset::

    Body.from_mesh("hammer/hammer.obj", name="hammer", texture="hammer/diffuse.png")
    Body.from_export("vab_tool_sort", name="screwdriver", tags=("screwdriver", "tool"))

Neither registers. A constructor returns a value you hold, the way pyroki's do;
what you do with it is the task's business.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.semantics.features import FeatureSpec
from robosimstudio.assetlib.semantics.published import PublishedState
from robosimstudio.assetlib.semantics.states import StateSpec
from robosimstudio.assetlib.sim import SimRequirements, contacts_with

__all__ = ["Body"]

TOUCHES = ("surface", "fixtures", "robot")
"""What a rigid object contacts unless it says otherwise.

37 of the corpus's 47 rigid assets declare exactly this. The other ten drop
``fixtures``, which is a real difference -- an object in a cell with nothing
fixed in it -- and so is stated rather than defaulted."""


def _body(
    files: dict[str, str],
    *,
    name: str,
    summary: str,
    tags: Sequence[str],
    touches: Sequence[str],
    features: Sequence[FeatureSpec],
    states: Sequence[StateSpec],
    published: Sequence[PublishedState],
    notes: str,
    provenance: str,
    mesh_sha: str,
    build: Callable[[], Any] | None = None,
) -> AssetRecord:
    """The shared body of both constructors: everything but where geometry comes from."""
    if not name:
        raise ValueError("a body needs a name: it is what a task document refers to it by")
    return AssetRecord(
        name=name,
        summary=summary,
        tags=tuple(tags),
        features=tuple(features),
        states=tuple(states),
        published=tuple(published),
        sim=SimRequirements(domain="rigid", contacts=contacts_with(*touches), notes=notes),
        files=files,
        mesh_sha=mesh_sha,
        provenance=provenance,
        build=build,
    )


class Body:
    """Constructors for a free rigid body, named for what you already have."""

    @staticmethod
    def from_mesh(
        mesh: str,
        *,
        name: str,
        summary: str = "",
        texture: str = "",
        tags: Sequence[str] = (),
        touches: Sequence[str] = TOUCHES,
        files: Mapping[str, str] = (),
        features: Sequence[FeatureSpec] = (),
        states: Sequence[StateSpec] = (),
        published: Sequence[PublishedState] = (),
        notes: str = "",
        provenance: str = "",
        mesh_sha: str = "",
        build: Callable[[], Any] | None = None,
    ) -> AssetRecord:
        """A body whose geometry is a mesh file.

        Args:
            mesh: Path to the mesh, relative to the declaring directory. Becomes
                the ``mesh`` file reference; the toolkit resolves it, so this
                holds a reference and never an absolute path.
            name: What a task document calls it.
            texture: Optional diffuse map beside the mesh. 12 of the corpus's
                rigid assets carry exactly this pair, which is why it is an
                argument rather than something to build a ``files`` dict for.
            touches: Roles this contacts. Defaults to :data:`TOUCHES`.
            files: Any further named source files -- an MJCF whose mass and
                friction a suite wants to read, a USD stage. Merged over the
                mesh and texture, so a caller can override either.
            summary, tags, features, states, published, notes, provenance,
                mesh_sha: Passed through unchanged.
        """
        named = {"mesh": mesh}
        if texture:
            named["texture"] = texture
        return _body(
            {**named, **dict(files or {})},
            name=name,
            summary=summary,
            tags=tags,
            touches=touches,
            features=features,
            states=states,
            published=published,
            notes=notes,
            provenance=provenance,
            mesh_sha=mesh_sha,
            build=build,
        )

    @staticmethod
    def from_export(
        export: str,
        *,
        name: str,
        summary: str = "",
        tags: Sequence[str] = (),
        touches: Sequence[str] = TOUCHES,
        files: Mapping[str, str] = (),
        features: Sequence[FeatureSpec] = (),
        states: Sequence[StateSpec] = (),
        published: Sequence[PublishedState] = (),
        notes: str = "",
        provenance: str = "",
        mesh_sha: str = "",
        build: Callable[[], Any] | None = None,
    ) -> AssetRecord:
        """A body that arrives inside a flattened stage rather than as its own file.

        14 of the corpus's rigid assets are this: the geometry is in an export
        the cell stages whole, and the record's job is to say the object is in
        there and what may be asked of it.

        Args:
            export: The export's name, as the staging refers to it.
            files: Further named files -- the same merge as
                :meth:`from_mesh`, which is what lets an export-backed object
                also name the per-object mesh and MJCF a suite reads for mass.
        """
        return _body(
            {"export": export, **dict(files or {})},
            name=name,
            summary=summary,
            tags=tags,
            touches=touches,
            features=features,
            states=states,
            published=published,
            notes=notes,
            provenance=provenance,
            mesh_sha=mesh_sha,
            build=build,
        )

    @staticmethod
    def from_spec(
        spec: Any,
        *,
        name: str,
        summary: str = "",
        tags: Sequence[str] = (),
        touches: Sequence[str] = TOUCHES,
        features: Sequence[FeatureSpec] = (),
        states: Sequence[StateSpec] = (),
        notes: str = "",
    ) -> AssetRecord:
        """A body that is a constructor's spec: ``rss.box(...)``, ``rss.sphere(...)``, ``rss.slab(...)``.

        The one-line way to give a task document a type to place: a document
        that says ``cube - cube_50mm`` builds whatever spec this was handed,
        named after the document's object. No geometry file is involved, so
        there are no files to name.

        Args:
            spec: What :func:`~robosimstudio.box` and the other rigid
                constructors return. Its ``at`` is where the object starts.
            name: What a task document calls the type.
        """
        record = _body(
            {},
            name=name,
            summary=summary or f"a {type(spec).__name__} built from a spec",
            tags=tags,
            touches=touches,
            features=features,
            states=states,
            published=(),
            notes=notes,
            provenance="",
            mesh_sha="",
            build=_Spec(spec),
        )
        if getattr(spec, "entity", "") != "static":
            return record
        # A fixed slab (``rss.slab``) is placed by the static kind, and touches
        # only what rests on it.
        sim = SimRequirements(domain="static", contacts=contacts_with("surface"), notes=notes)
        return replace(record, sim=sim)


@dataclass(frozen=True)
class _Spec:
    """A build factory that returns one spec. A value, so two declarations of a
    type compare by the spec they build: the same cube declared twice is one
    asset, and two different cubes under one name are a clash."""

    spec: Any

    def __call__(self) -> Any:
        return self.spec
