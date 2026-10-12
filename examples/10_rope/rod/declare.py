# SPDX-License-Identifier: Apache-2.0
"""Declaring a 1-D body: the part a rod already knows, said once.

Three tasks shipped a rod -- ``cable_route``'s cable, ``towel_hang``'s line,
``threaded_needle``'s suture -- and each wrote out the same opening: a
``point`` feature per landmark, the same sentence of provenance under each, and
``sim=SimRequirements(domain="rod", contacts=contacts_with(...))``. None of that
is a measurement of a particular rod. It is what *being* a rod means, restated
three times because there was nowhere to put it.

**A chain's landmarks are free, and that is the whole argument.** The substrate
ships its own coordinate system, so ``cable.point(0.5)`` is meaningful the moment
the thing exists -- the cable suite's ``cable.py`` calls a chain "the
easiest asset in the corpus to annotate" for exactly this reason. A landmark on
one is a *name for a number between 0 and 1*, nothing measured, so the
constructor can state its provenance truthfully on the author's behalf instead of
making them type the same sentence again.

What it does **not** do is invent measurements. Anything a rod knows only because
somebody measured it -- the wrap's anchor at s = 0.6361, the line's straightness
gate of 0.98, an axis the posts define -- is passed in and carries its own
provenance, because a number whose justification survives only as a comment is a
number nothing can check.

That rule cost this module a convenience on its first draft. A ``straight=0.95``
shorthand looked like the same kind of boilerplate as a landmark, and it is not:
**both** shipped rods justify that gate with a measurement -- the line's is "1.00
empty and 0.998 under the towel", the cable's is "the reference ships no shape
predicate" -- and a shorthand would have replaced two real provenances with one
invented sentence. Straightness is a ``states=`` entry like any other.

    ROPE = Rod.from_length(
        name="tow_rope", length=2.0, radius=0.008,
        summary="A 2 m rope.", points={"end_a": 0.0, "end_b": 1.0},
        touches=("surface", "fixtures", "hands"),
    )

...which the task then hands to the asset registry itself.

The name stays the task's: this **builds** a record and never registers one.
Which task calls a rod ``cable_rod`` and which calls it ``hang_line`` is content
vocabulary, and the library does not get to hold it. (``test_content_seam`` reads
this package for the registering call by plain text search, which is crude and
exactly right -- so even an example of one does not belong here.)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from robosimstudio.assetlib.record import AssetRecord
from robosimstudio.assetlib.semantics.features import FeatureSpec
from robosimstudio.assetlib.semantics.published import PublishedState
from robosimstudio.assetlib.semantics.states import StateSpec
from robosimstudio.assetlib.sim import SimRequirements, contacts_with

__all__ = ["Rod"]

DERIVED = "The chain's own arc-length parameterisation; nothing measured."
"""Why a landmark on a rod needs no measurement, stated once.

Honest rather than boilerplate: the number *is* the definition. Three assets
each wrote a variant of this sentence by hand and one of them had drifted."""


class Rod:
    """Constructors for a 1-D body, named for what you already have."""

    @staticmethod
    def from_length(
        *,
        name: str,
        length: float,
        radius: float,
        summary: str = "",
        points: Mapping[str, float | tuple[float, str]] = (),
        touches: Sequence[str] = ("surface", "fixtures", "hands"),
        tags: Sequence[str] = ("rod",),
        features: Sequence[FeatureSpec] = (),
        states: Sequence[StateSpec] = (),
        published: Sequence[PublishedState] = (),
        notes: str = "",
        provenance: str = "",
        self_contact: bool = False,
        build: Callable[[], Any] | None = None,
    ) -> AssetRecord:
        """A rod of ``length`` metres, as an :class:`AssetRecord`.

        Args:
            name: What the *task* calls it. Not registered here.
            length: Overall length [m]. The *geometry* is the entity's, built
                from the task's own spec -- what this is for is the default
                summary, which three assets were writing by hand off exactly
                these two numbers.
            radius: Cross-section radius [m], on the same terms.
            summary: One line for the catalog. Derived from ``length`` and
                ``radius`` when omitted.
            points: Landmark name -> normalised arc length, or
                ``(arc length, why it is named)``. Each becomes a ``point``
                feature whose provenance is :data:`DERIVED` -- true by
                construction -- plus the note, because *how* a landmark is
                derived and *why anyone named it* are different facts and the
                second one is worth keeping.
            touches: Roles this rod contacts, passed to
                :func:`~robosimstudio.assetlib.sim.contacts_with`.
            features: Anything **measured** -- an axis, an anchor, a cavity.
                Stated normally, with its own provenance.
            states, published, notes, provenance, tags, self_contact: Passed
                through unchanged.
            build: The factory a document builds this rod through; see
                :attr:`~robosimstudio.assetlib.record.AssetRecord.build`.

        Raises:
            ValueError: If a landmark is outside ``[0, 1]``, or if ``points``
                and ``features`` name the same thing -- two declarations of one
                landmark is the drift this exists to stop.
        """
        landmarks = dict(points or {})
        for landmark, entry in landmarks.items():
            s = entry[0] if isinstance(entry, tuple) else entry
            if not 0.0 <= float(s) <= 1.0:
                raise ValueError(
                    f"rod {name!r}: landmark {landmark!r} is at s={s}, outside [0, 1]. "
                    "Arc length is normalised so a reference survives a change of segment count."
                )
        clash = sorted(landmarks.keys() & {f.name for f in features})
        if clash:
            raise ValueError(f"rod {name!r}: {clash} declared both as a landmark and as a feature; state one")

        derived = tuple(
            FeatureSpec(
                name=landmark,
                kind="point",
                payload={"s": float(entry[0] if isinstance(entry, tuple) else entry)},
                provenance=f"{DERIVED} {entry[1]}" if isinstance(entry, tuple) else DERIVED,
            )
            for landmark, entry in landmarks.items()
        )
        return AssetRecord(
            name=name,
            summary=summary or f"A {length:.3g} m rod, {radius * 2000:.0f} mm thick.",
            tags=tuple(tags),
            features=(*derived, *features),
            states=tuple(states),
            published=tuple(published),
            sim=SimRequirements(
                domain="rod",
                contacts=contacts_with(*touches),
                self_contact=self_contact,
                notes=notes,
            ),
            provenance=provenance,
            build=build,
        )
