# SPDX-License-Identifier: Apache-2.0
"""Everything about the thing you clicked on.

**An interface here is per object.** `assetlib.semantics` states it: duck-typed,
freely defined, discovered rather than declared, with no capability roster to
implement. So this panel cannot render a fixed shape -- it asks
:func:`~robosimstudio.assetlib.interface.interface` to parse *this* asset's
declaration against the kinds *it* names, and draws each facet the way its tier
wants:

* a **literal** becomes its fields, each beside the meaning its kind gives it and
  marked where it is a point or a direction in the body frame;
* a **recipe** becomes the selector expression, canonicalised;
* a **resolver** becomes the versioned key it references, because that is all
  there is and pretending otherwise would be inventing content.

A receptacle's ``cavity`` and a towel's ``panel`` therefore look nothing alike on
the page, which is correct -- they are nothing alike.

**Sections beyond the declaration.** What this subject publishes right now,
tagged with how observable each quantity is. And what may be tuned on it,
generated from
:meth:`~robosimstudio.scene.entity.Entity.tunables` through the binder
the material studio already had -- which is why a rigid body gets friction and
mass sliders without anything here knowing what a rigid body is.

**Two folds, and nothing folded inside them.** What is live -- the numbers this
thing is publishing -- is simply there, because a folder that is expanded by
default is a heading wearing a triangle. Everything static is behind
``Interface`` or ``Parameters``, and behind those it is *all* there: every field,
every value, every slider, flat. One click is the whole difference between the
simple reading and the expert one; a second click for the same thing is a maze.

**Rebuilt on selection, not per frame.** viser has no way to re-point a widget at
a different object, so changing the subject means removing the folder and
building another. That happens at drain time, at most once a click; the readouts
inside it are rewritten on the ordinary refresh gate.

**It is rebuilt into a container it never gives back**, which is what keeps it in
place. viser sorts a sidebar by an order counter that increments per widget, so a
folder added after the page was built sorts *after* everything on it: clicking
anything sent the inspector to the foot of the sidebar, below the export
controls, and it stayed there. The anchor is an unlabelled folder built once, in
position, and the subject's folder is rebuilt inside it.
"""

from __future__ import annotations

from typing import Any

from .panels import Panel

__all__ = ["PanelInspector"]

TIER_MARK = {"point": " ⌖", "direction": " ➔"}
"""How a body-frame field is marked. A point is rotated *and* translated to reach
world; a direction is rotated only, and translating one turns a unit vector into a
position. The kind declares which, so the page does not guess."""


class PanelInspector(Panel):
    """The current subject, in as much detail as it has declared.

    Args:
        session: The page.
    """

    def __init__(self, session) -> None:
        super().__init__(session)
        self._host: Any = None
        """Where the subject's folder is rebuilt, so it is rebuilt in place."""
        self._folder: Any = None
        self._gui: Any = None
        self._shown: str = ""
        self._binders: list[tuple[Any, Any]] = []
        """``(binder, tunable)`` for the current subject, drained each step."""
        self._live: list[Any] = []
        """Read-only widgets rewritten on refresh."""

    # ------------------------------------------------------------------- build
    def build(self, gui) -> None:
        self._gui = gui
        # Unlabelled: viser renders such a folder with no header and no border,
        # so this claims the panel's place in the sidebar without spending a
        # heading on saying so.
        self._host = gui.add_folder(None)
        self._rebuild()

    def _rebuild(self) -> None:
        """Tear the folder down and put up the current subject's.

        Called from :meth:`drain`, so viser's own threads never touch the GUI.
        """
        if self._folder is not None:
            self._folder.remove()
        self._folder, self._binders, self._live = None, [], []
        gui, subject = self._gui, self.session.selected
        if gui is None or self._host is None:
            return
        # Nothing at all until something is selected, rather than an empty
        # ``Inspector`` heading over a line of italics telling you to click. A
        # section that is only ever a prompt is in the way on every page that
        # never needed one; clicking a thing is what makes this appear, and the
        # page's own How-to says so.
        self._shown = subject.key if subject is not None else ""
        if subject is None:
            return
        with self._host:
            self._folder = gui.add_folder(f"Inspector: {subject.name}")
        with self._folder:
            gui.add_markdown(self._identity(subject))
            self._live.append(
                (gui.add_markdown(self._published_text(subject)), lambda s=subject: self._published_text(s))
            )
            self._section_interface(gui, subject)
            self._section_parameters(gui, subject)

    # ---------------------------------------------------------------- identity
    def _identity(self, subject) -> str:
        """What this is, in one line and a sentence."""
        record = self._record(subject)
        if record is None:
            return f"**{subject.kind}** `{subject.name}` -- *not typed against the catalog*"
        # Anything that repeats what is already on the line goes: the towel is a
        # `cloth` subject of `cloth` substrate tagged `cloth`, and the line read
        # "cloth towel cloth cloth". Three facts' worth of type for one fact.
        said = {subject.kind, record.name}
        words = [record.substrate, *record.tags]
        head = f"**{subject.kind}** `{subject.name}`"
        for word in words:
            if word not in said:
                said.add(word)
                head += f" `{word}`"
        return head + (f"\n\n{record.summary}" if record.summary else "")

    def _record(self, subject):
        """This subject's declaration, or ``None`` when it has no asset."""
        from robosimstudio.assetlib.registry import find_asset

        return find_asset(subject.asset)

    # --------------------------------------------------------------- published
    def _published_text(self, subject) -> str:
        """What this entity exposes right now, with how observable each one is."""
        record = self._record(subject)
        context = getattr(self.env, "last_context", None)
        if record is None or not record.published or subject.entity is None or context is None:
            return ""
        try:
            values = subject.entity.published_states(context)
        except Exception as exc:  # a reading must not be able to take the page down
            return f"*unreadable: {exc}*"
        rows = []
        for state in record.published:
            value = values.get(state.name, values.get(state.name.replace("-", "_")))
            if value is None:
                continue
            shown = f"{value:.4g}" if isinstance(value, float) else str(value)
            unit = f" {state.unit}" if state.unit else ""
            # `internal` is the one worth marking: no camera reads it, so a
            # policy that learned from it has learned something no real robot can.
            mark = " *(internal)*" if state.tier == "internal" else ""
            rows.append(f"- {state.name} {shown}{unit}{mark}")
        return "\n".join(["**publishes**", *rows]) if rows else ""

    # --------------------------------------------------------------- interface
    def _section_interface(self, gui, subject) -> None:
        """The asset's own declaration, parsed against the kinds it names.

        One fold, every facet inside it.
        """
        record = self._record(subject)
        if record is None:
            return
        from robosimstudio.assetlib.interface import interface

        parsed = interface(record)
        if not parsed.facets:
            with gui.add_folder("Interface", expand_by_default=False):
                gui.add_markdown("*declares no features -- a rigid asset's grasp geometry is derived from its mesh*")
            return

        with gui.add_folder(f"Interface ({len(parsed.facets)})", expand_by_default=False):
            for facet in parsed.facets:
                gui.add_markdown(self._facet_text(facet))
                # Only where the overlay that draws it is on the page: a
                # checkbox nothing services is a control that lies, and its
                # requests would pile up with no one to take them.
                if facet.drawable and getattr(self.session, "overlay", None) is not None:
                    show = gui.add_checkbox(f"Draw {facet.name}", initial_value=False)
                    show.on_update(
                        lambda _, f=facet.name, w=show: self.session.overlay_request.append(
                            (subject.key, f, bool(w.value))
                        )
                    )
            if record.provenance:
                gui.add_markdown(f"> {' '.join(record.provenance.split())}")

    def _facet_text(self, facet) -> str:
        """One facet, drawn the way its tier wants. Three tiers, three renderings."""
        rows = [f"**{facet.name}** `{facet.kind}`" + ("" if facet.contracted else " *(uncontracted)*")]
        if facet.summary:
            rows.append(f"*{facet.summary}*")
        if facet.tier == "recipe":
            rows.append(f"```\n{facet.select}\n```")
        elif facet.tier == "resolver":
            rows.append(f"`resolver` → `{facet.resolver}`")
        else:
            for member in facet.members:
                mark = TIER_MARK.get(member.frame, "")
                flag = "" if member.declared else " ⚠ undeclared"
                rows.append(f"- `{member.name}`{mark}{flag} = `{member.value}`")
                if member.meaning:
                    rows.append(f"  {BLOCKQUOTE}{member.meaning}")
        if facet.missing:
            rows.append(f"\n⚠ missing required: {', '.join(facet.missing)}")
        if facet.pins_mesh:
            pinned = facet.mesh_sha[:12] if facet.mesh_sha else "**nothing**"
            rows.append(f"\npins the discretisation to {pinned}")
        if facet.provenance:
            rows.append(f"\n> {' '.join(facet.provenance.split())}")
        return "\n".join(rows)

    # -------------------------------------------------------------- parameters
    def _section_parameters(self, gui, subject) -> None:
        """What may be changed on this thing while it runs.

        Nothing here knows what a rigid body is. The entity says what it offers
        and the binder turns each described dataclass into widgets, so a field
        added to a spec grows a control without this file being edited.

        Flat, and that is the point of the section: one click and every number is
        on screen. The binder heads each group with a line of text instead of a
        folder for the same reason.
        """
        entity = subject.entity
        if entity is None:
            return
        try:
            tunables = entity.tunables(getattr(self.env, "pipeline", None))
        except Exception:
            return
        if not tunables:
            return
        from .tuning.binder import TIER_MARK as MARK
        from .tuning.binder import SpecBinder

        with gui.add_folder(f"Parameters ({len(tunables)})", expand_by_default=False):
            gui.add_markdown(
                f"`{MARK['recapture'].strip() or '~'}` re-captures  `{MARK['rebuild'].strip()}` needs a rebuild"
            )
            for tunable in tunables:
                heading = f"### {tunable.name}"
                gui.add_markdown(f"{heading}\n\n*{tunable.summary}*" if tunable.summary else heading)
                binder = SpecBinder(tunable.spec)
                binder.build(gui)
                self._binders.append((binder, tunable))

    # ------------------------------------------------------------------- drain
    def drain(self) -> None:
        """Rebuild on a selection change, then hand over any edits. Main thread."""
        subject = self.session.selected
        key = subject.key if subject is not None else ""
        if key != self._shown:
            self._rebuild()
            return
        for binder, tunable in self._binders:
            for field, value in binder.drain().items():
                # `False` means the field has nothing in this entity to write to.
                # Shown rather than raised: it is information about the entity.
                tunable.write(field, value)

    def refresh(self) -> None:
        for widget, source in self._live:
            widget.content = source()

    def on_select(self, subject) -> None:
        """Nothing to do: :meth:`drain` notices the change on the next step.

        Deliberately not rebuilding here -- selection can be set from a viser
        callback thread, and building widgets off the main thread is how a
        session acquires a heisenbug.
        """

    def close(self) -> None:
        super().close()
        if self._folder is not None:
            self._folder.remove()
            self._folder = None
        if self._host is not None:
            self._host.remove()
            self._host = None


BLOCKQUOTE = "  "
"""Indent for a field's meaning, so it reads as a caption rather than a value."""
