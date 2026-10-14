# SPDX-License-Identifier: Apache-2.0
"""The studio's page with a ``Live`` folder on top, held still between commands.

The folder is built *first*, because viser sorts the sidebar by creation
order and the page is rebuilt from scratch on every ``adopt`` (teardown resets
the GUI) -- so it is rebuilt here, on every ``build_gui``, and the server
rewrites its readouts straight after. Markdown rather than text boxes: a text
box is one narrow line and truncates, and these are the two readouts the page
is for -- what is loaded and when, and the traceback when a rebuild did not
land. Whether a rebuild lands running or paused is the server's ``hold``; the
Pause checkbox on the page is the human's override either way, and the
studio's own ``set_paused`` keeps the two truthful. While a routine plays the
page's own stepping is held regardless (:meth:`LiveSession.steps_held`), and
both Reset buttons -- this folder's and the Simulation folder's -- go through
the one :meth:`LiveSession.reset_now`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robosimstudio_studio.session import StudioConfig, StudioSession

if TYPE_CHECKING:
    import numpy as np

    from .server import LiveServer

__all__ = ["LiveSession"]


class LiveSession(StudioSession):
    """The studio's page over a built ``Sim``, with the server's folder at the top of its sidebar."""

    def __init__(self, sim: Any, *, config: StudioConfig, host: LiveServer) -> None:
        self.host = host
        self.host_folder: Any = None
        self.host_widgets: dict[str, Any] = {}
        super().__init__(sim, config=config)

    def adopt(self, sim: Any, *, label: str = "", episode: int = 0) -> None:
        # Running, so the page can be dragged and played with the moment it
        # lands; `--hold` lands it paused instead, and `ctl run N` steps a
        # held page.
        self.paused = self.host.hold
        super().adopt(sim, label=label, episode=episode)

    def steps_held(self) -> bool:
        """While a routine plays it steps the scene itself; the studio's own step would double it."""
        return bool(self.host.script)

    def reset_now(self) -> bool:
        """A reset ends whatever routine is playing and the take filming it, then resets."""
        # Before the reset: a take's last frame is read off the scene as it stands.
        self.host.stop_scripts("reset")
        self.host.resets += 1
        return super().reset_now()

    def build_gui(self) -> None:
        gui = self.server.gui
        self.host_folder = gui.add_folder("Live")
        self.buttons: dict[str, Any] = {}
        self.board: dict[str, Any] = {}
        with self.host_folder:
            self.host_widgets = {
                "scene": gui.add_markdown(""),
                "sim": gui.add_markdown(""),
                "error": gui.add_markdown(""),
            }
            self._build_buttons()
            if self.host.ref_image is not None:
                self.host_widgets["ref"] = gui.add_image(self.host.ref_image, label=self.host.ref_label)
            for label, image in self.host.board.items():
                self.board[label] = gui.add_image(image, label=label)
        super().build_gui()
        self.host.refresh_live(force=True)

    def _build_buttons(self) -> None:
        """One button per routine the scene file binds, and Reset beside them; the main thread acts on a click."""
        gui = self.server.gui
        names = list(self.host.routine_names())
        if not names:
            return
        for name in names:
            button = gui.add_button(f"Run {name}")
            button.on_click(lambda _event, name=name: self.host.clicks.put(("play", name)))
            self.buttons[name] = button
        reset = gui.add_button("Reset")
        # The same request the Simulation folder's Reset makes.
        reset.on_click(lambda _event: setattr(self, "reset_requested", True))
        self.buttons["__reset__"] = reset

    def show_reference(self, image: np.ndarray, label: str) -> None:
        """Put a picture in the ``Live`` folder, replacing the one there."""
        self.hide_reference()
        with self.host_folder:
            self.host_widgets["ref"] = self.server.gui.add_image(image, label=label)

    def hide_reference(self) -> None:
        old = self.host_widgets.pop("ref", None)
        if old is not None:
            old.remove()

    def show(self, label: str, image: np.ndarray) -> None:
        """Put a picture on the board under ``label``, replacing the one with that label."""
        old = self.board.pop(label, None)
        if old is not None:
            old.remove()
        with self.host_folder:
            self.board[label] = self.server.gui.add_image(image, label=label)

    def clear(self, label: str = "") -> None:
        """Take one picture off the board, or all of them."""
        for key in [label] if label else list(self.board):
            old = self.board.pop(key, None)
            if old is not None:
                old.remove()
