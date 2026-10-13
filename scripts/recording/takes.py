# SPDX-License-Identifier: Apache-2.0
"""Every clip an example README shows, as the take that films it, or the reason it is kept.

A take is the example's own script, run unchanged, with a person on its page:
the gestures are :mod:`.beats`, the camera and the sidebar are set before the
first frame, and the expectations say what the physics must have done for the
clip to ship. ``scripts/record_examples.py`` films them.

A take marked ``pending`` is a draft: its example's physics is still being
changed, so it is filmed into ``outputs/record/`` for a look and never written
into ``examples/``. Clear the mark once the take has been tuned on the final
physics and its sheet looked at.

Dataclasses only, and no import of the library or a browser:
``tests/test_record_takes.py`` reads this table on a machine with neither.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .beats import (
    At,
    Beat,
    Camera,
    Drag,
    Expectation,
    Held,
    Layout,
    Orbit,
    Press,
    Rose,
    Wait,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@dataclass(frozen=True)
class Take:
    """One clip: which example, what happens on its page, and what must be true afterwards."""

    example: str
    """The script, relative to ``examples/``: ``01_rigid/00_drag_things.py``."""
    beats: tuple[Beat, ...] = ()
    name: str = ""
    """The clip's name; the script's stem by default, anything else an extra clip."""
    folder: str = ""
    """The example folder whose ``media/`` gets it; the script's own by default."""
    argv: tuple[str, ...] = ()
    """Arguments to the script: ``("--simulate",)`` for a teleop example with no arm attached."""
    needs: tuple[str, ...] = ()
    """Extras the script needs: ``uv run --extra <x>``."""
    camera: Camera | Orbit | None = None
    """``None`` keeps the page's own opening view."""
    layout: Layout = field(default_factory=Layout)
    config: tuple[tuple[str, Any], ...] = ()
    """``StudioConfig`` fields set on top of the example's, each with its reason beside it."""
    warmup: float = 0.0
    """Sim seconds stepped before the browser opens, unfilmed."""
    tail: float = 0.6
    """Sim seconds filmed after the last beat."""
    fps: int = 15
    viewport: tuple[int, int] = (1152, 648)
    """CSS pixels: the sidebar is 324 of them, the 3D view the rest."""
    dpr: float = 2.0
    size: tuple[int, int] = (960, 540)
    """The clip: the page at twice its pixels, scaled down with Lanczos."""
    budget: int = 900_000
    cursor: bool = True
    expect: tuple[Expectation, ...] = ()
    pending: str = ""
    """Why this take is a draft. Non-empty: filmed into ``outputs/record/`` only."""

    @property
    def stem(self) -> str:
        return self.name or Path(self.example).stem

    @property
    def home(self) -> str:
        return self.folder or str(Path(self.example).parent)

    @property
    def key(self) -> str:
        """``<folder>/<clip>``: how the command line and :data:`KEPT` name a clip."""
        return f"{self.home}/{self.stem}"

    @property
    def media(self) -> Path:
        """Where the clip goes when it ships."""
        return EXAMPLES / self.home / "media" / f"{self.stem}.webp"


# =============================================================================
# The takes
# =============================================================================

RIGHT = "Arm: right"
"""The right arm's fold; its title may carry a role, ``Arm: right (work)``, which a prefix still matches."""

RIGID = Take(
    "01_rigid/00_drag_things.py",
    camera=Orbit(At("cube", dz=0.06), distance=0.55, azimuth=-10.0, elevation=24.0, fov=50.0),
    warmup=1.0,  # the ball and the pin land and the jaws finish opening before the first frame
    layout=Layout(collapse=("Simulation", "Arm: left")),  # the arm that does nothing, out of the way
    beats=(
        Wait(seconds=0.6, label="the right hand, open, over the cube"),
        Drag("right", "z", to=At("cube"), seconds=1.4, label="drag the gizmo down onto the cube"),
        Wait(seconds=0.3),
        Press(text="Close", within=RIGHT, label="Close"),
        Wait(seconds=0.9, mark="closed", label="friction holds it"),
        Drag("right", "z", 0.15, seconds=1.4, label="drag it up 150 mm"),
        Wait(seconds=0.5, mark="lifted"),
        Drag("right", "y", 0.06, seconds=0.8, label="swing"),
        Drag("right", "y", -0.06, seconds=0.8, label="and back"),
        Wait(seconds=0.4, mark="swung"),
        Press(text="Open", within=RIGHT, label="Open: it drops"),
        Wait(seconds=1.2),
    ),
    expect=(
        Rose("cube", 0.12, at="lifted"),
        Held("cube", arm="right", since="closed", at="swung", tolerance=0.02),
    ),
)

TAKES: tuple[Take, ...] = (RIGID,)

KEPT: dict[str, str] = {
    "05_studio/01_render": "opens no page: it path traces a still and a video, which is what its clip shows",
    "08_policies/03_many_worlds": "opens no page: a fleet run of every world at once, filmed as a grid of tiles",
    "02_cloth/03_genesis": "needs the Genesis venv, which the recorder's environment deliberately does not carry",
    "live/live_page": "the live page rebuilding on a save of scene.py: a file being edited, not a scene being driven",
}
"""Clips no take films, each with why its existing file stands."""


def find(key: str) -> Take:
    for take in TAKES:
        if take.key == key:
            return take
    raise KeyError(key)
