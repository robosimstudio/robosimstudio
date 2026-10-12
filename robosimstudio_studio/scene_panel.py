# SPDX-License-Identifier: Apache-2.0
"""What is in this scene, and which of it the page is about.

One dropdown over every subject, and a count. It exists for the things a click
cannot reach -- an object behind the bench, a region with no surface to hit, an
arm the camera is inside -- and as the answer to "what is even in here", which
the page could not previously be asked.

Deliberately one control rather than a tree. viser has folders and dropdowns and
no tree widget, and a folder per kind would put five collapsible sections in
front of a list that is usually under twenty entries. The kind is in each row
(``object · hammer_0``), so the flat list sorts into groups by reading it.
"""

from __future__ import annotations

from .panels import Panel
from .selection import Subject

__all__ = ["PanelScene"]


class PanelScene(Panel):
    """The scene's contents, as one selectable list.

    Args:
        session: The page.
    """

    def __init__(self, session) -> None:
        super().__init__(session)
        self._dropdown = None
        self._quiet = False
        """Set while writing the dropdown from a click, so following the
        selection does not fire a callback that re-selects what is already
        selected."""

    def build(self, gui) -> None:
        index = self.session.index
        options = tuple(s.label for s in index.subjects) if index else ()
        with gui.add_folder("Scene"):
            gui.add_markdown(index.summary() if index else "*empty*")
            if not options:
                return
            current = self.session.selected
            initial = current.label if current is not None and current.label in options else options[0]
            self._dropdown = gui.add_dropdown("Select", options=options, initial_value=initial)
            self._dropdown.on_update(lambda _: self._chose(str(self._dropdown.value)))

    def _chose(self, label: str) -> None:
        """Record what was picked. Viser's thread, so it may only assign."""
        if self._quiet:
            return
        index = self.session.index
        subject = next((s for s in index.subjects if s.label == label), None) if index else None
        if subject is not None:
            self.session.select_request = subject.key

    def follow(self, subject: Subject) -> None:
        """Point the dropdown at what a click selected. Main thread.

        Without this the list goes on naming whatever was last picked from it
        while the page shows something else, which reads as the page having
        ignored the click.
        """
        if self._dropdown is None:
            return
        self._quiet = True
        try:
            self._dropdown.value = subject.label
        finally:
            self._quiet = False
