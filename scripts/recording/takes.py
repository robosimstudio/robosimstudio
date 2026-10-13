# SPDX-License-Identifier: Apache-2.0
"""Every clip an example README shows, as the take that films it, or the reason it is kept.

A take is the example's own script, run unchanged, with a person on its page:
the gestures are :mod:`.beats`, the camera and the sidebar are set before the
first frame, and the expectations say what the physics must have done for the
clip to ship. ``scripts/record_examples.py`` films them.

A take marked ``pending`` is a draft: its example's physics is still being
changed, so it is filmed into ``outputs/record/`` for a look and never written
into a ``media/`` folder. Clear the mark once the take has been tuned on the
final physics and its sheet looked at.

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
    Bench,
    Camera,
    Check,
    Click,
    Corner,
    Drag,
    Expectation,
    Fold,
    Grasp,
    Held,
    Layout,
    Orbit,
    Press,
    Rose,
    Run,
    Segment,
    Select,
    Slider,
    Squashed,
    Toggle,
    Wait,
)

REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Take:
    """One clip: which script, what happens on its page, and what must be true afterwards."""

    example: str
    """The script, from the repository root: ``examples/01_rigid/00_drag_things.py``."""
    beats: tuple[Beat, ...] = ()
    name: str = ""
    """The clip's name; the script's stem by default, anything else an extra clip."""
    folder: str = ""
    """The folder whose ``media/`` gets it, from the repository root; the script's own by default."""
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
    """Seconds of scene stepped before the browser opens, unfilmed."""
    tail: float = 0.6
    """Seconds of clip filmed after the last beat."""
    fps: int = 15
    speed: float = 1.0
    """Seconds of scene per second of clip. Above 1 for a routine that runs past half a minute."""
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
        """The folder the clip belongs to, from the repository root."""
        return self.folder or str(Path(self.example).parent)

    @property
    def key(self) -> str:
        """``<folder>/<clip>``, ``examples/`` left off: how the command line and :data:`KEPT` name a clip."""
        return f"{self.home.removeprefix('examples/')}/{self.stem}"

    @property
    def media(self) -> Path:
        """Where the clip goes when it ships."""
        return REPO / self.home / "media" / f"{self.stem}.webp"


# =============================================================================
# Final: their physics is settled, and each ships when its expectations hold
# =============================================================================


def _draped(film, name: str, mark: str) -> tuple[bool, str]:
    """A sheet that went limp at ``mark`` has fallen in around its post: its footprint shrank."""
    start, limp = film.marks["start"]["spreads"][name], film.marks[mark]["spreads"][name]
    return (
        start - limp > 0.004,
        f"the {name}'s footprint went {start * 1000:.1f} -> {limp * 1000:.1f} mm across (need -4)",
    )


RIGID = "examples/01_rigid/00_drag_things.py"
RIGHT = "Arm: right"
"""The right arm's fold; its title may carry a role, ``Arm: right (work)``, which a prefix still matches."""
CUBE_VIEW = Orbit(At("cube", dz=0.06), distance=0.55, azimuth=-10.0, elevation=24.0, fov=50.0)
"""The rigid example's cube and right hand, from the bench's open side: the jaws close across the view."""

FINAL = (
    Take(
        RIGID,
        camera=CUBE_VIEW,
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
            Held("cube", arm="right", since="closed", at="swung"),
        ),
    ),
    Take(
        RIGID,
        name="drag_cube",
        folder="examples/05_studio",
        camera=Orbit(At("ball", dz=0.08), distance=0.6, azimuth=-5.0, elevation=26.0, fov=50.0),
        warmup=1.0,
        layout=Layout(collapse=("Simulation", "Arm: left", "Arm: right")),
        beats=(
            Wait(seconds=0.5, label="a ball on the bench"),
            Click(At("ball"), label="click it: a handle"),
            Wait(seconds=0.3),
            Drag("handle", "z", 0.12, seconds=1.2, label="drag the handle up"),
            Drag("handle", "y", -0.08, seconds=1.0, label="and across"),
            Wait(seconds=0.6, mark="carried", label="held by the handle"),
            Press(text="Release all", label="Release all: it falls"),
            Wait(seconds=1.4),
        ),
        expect=(Rose("ball", 0.08, at="carried"),),
    ),
    Take(
        RIGID,
        name="drive_arm",
        folder="examples/05_studio",
        camera=CUBE_VIEW,
        warmup=1.0,
        layout=Layout(collapse=("Simulation", "Arm: left")),
        beats=(
            Wait(seconds=0.5),
            Drag("right", "z", to=At("cube"), seconds=1.4, label="drag the hand down"),
            Press(text="Close", within=RIGHT, label="Close"),
            Wait(seconds=0.8, mark="closed"),
            Drag("right", "z", 0.12, seconds=1.2, label="lift"),
            Drag("right", "y", 0.10, seconds=1.0, label="carry"),
            Wait(seconds=0.4, mark="carried"),
            Drag(
                "right", "z", to=Bench(z=0.027), seconds=1.2, label="set it down"
            ),  # the cube's middle 2 mm over its rest
            Press(text="Open", within=RIGHT, label="Open"),
            Drag("right", "z", 0.08, seconds=0.8, label="and away"),
            Wait(seconds=0.8),
        ),
        expect=(Rose("cube", 0.08, at="carried"), Held("cube", since="closed", at="carried")),
    ),
    Take(
        "examples/02_cloth/01_fabrics.py",
        # The fleece in the middle, the limp tape to its left, the towelling to its right.
        camera=Orbit(At("fleece", dz=-0.04), distance=0.75, azimuth=-20.0, elevation=30.0, fov=50.0),
        warmup=3.0,  # the four sheets settle over their posts before anyone looks
        beats=(
            Wait(seconds=0.5, label="four fabrics on four posts"),
            Fold(title="Material", label="open Material"),
            Select(text="Cloth", option="fleece", within="Material", label="pick the fleece"),
            # In-plane stiffness, not bending: a square drapes over a round post
            # only by stretching in its own plane, so the jersey at 10^2 N/m
            # hangs in folds and the fleece at 10^4 stands as a tent. Measured on
            # this scene, the tape at 10^-1 N m/rad of bending still hangs.
            Slider(text="tri_ke", value=2.0, seconds=1.4, within="Material", label="tri_ke down 100x"),
            Wait(seconds=1.6, mark="limp", label="the fleece drapes like the jersey"),
            Slider(text="tri_ke", value=None, seconds=1.4, within="Material", label="and back"),
            Wait(seconds=1.6, mark="stiff", label="a tent again"),
        ),
        expect=(Check(lambda film: _draped(film, "fleece", "limp"), needs=("fleece",)),),
    ),
    Take(
        "examples/10_rope/01_between_posts.py",
        # Three-quarters on to the span: square on, the x arrow points into the lens and cannot be grabbed.
        camera=Orbit(At("cable", dz=0.03), distance=0.65, azimuth=-30.0, elevation=22.0, fov=50.0),
        warmup=1.0,
        layout=Layout(collapse=("Simulation", "Arm: left", "Arm: right")),
        beats=(
            Wait(seconds=0.5, label="a cable between two posts"),
            Click(Segment("cable", 0.5), label="click its middle: a handle"),
            Wait(seconds=0.3),
            # 50 mm: the span is 0.5 m and does not stretch, so a lift of its
            # middle much past the sag it rests in pulls the handle off the cable.
            Drag("handle", "z", 0.05, seconds=1.2, label="drag it up"),
            Drag("handle", "x", 0.05, seconds=1.0, label="and towards you"),
            Wait(seconds=0.8, mark="lifted", label="the span hangs from the handle"),
            Press(text="Release all", label="let go"),
            Wait(seconds=1.6, label="it falls back between the posts"),
        ),
        expect=(Rose(Segment("cable", 0.5), 0.025, at="lifted"),),
    ),
)


# =============================================================================
# Drafts: the gesture planned, on physics other work is still changing
# =============================================================================

PENDING = "physics being changed on another branch; tune after merge"


def grasp(example: str, thing: str, *, at: Any = None, lift: float = 0.12, pending: str = PENDING, **kw) -> Take:
    """The planned gesture as a draft: the nearest hand over the thing, down, Close, up."""
    return Take(
        example,
        camera=kw.pop("camera", Orbit(At(thing, dz=0.06), distance=0.6, azimuth=-15.0, elevation=28.0)),
        warmup=kw.pop("warmup", 1.0),
        beats=(
            Wait(seconds=0.5),
            Grasp(thing=thing, at=at, lift=lift, label=f"down onto the {thing}, Close, up"),
            Wait(seconds=1.0),
        ),
        expect=kw.pop("expect", (Rose(thing, lift * 0.6, at="lifted"),)),
        pending=pending,
        **kw,
    )


def routine(example: str, *, speed: float, success: str) -> Take:
    """A demo: the home pose, then its Run button, filmed to the routine's end."""
    return Take(
        example,
        fps=12,
        speed=speed,
        warmup=0.5,
        budget=1_500_000,
        beats=(
            Wait(seconds=0.5, label="home, hands open"),
            Run(label="Run"),
            Wait(seconds=1.5, label="done"),
        ),
        expect=(
            Check(lambda film: film.ended.endswith(": done"), "the routine ran to its end:"),
            # TODO(after merge): the demo's own success measure, once it exposes one.
            Check(lambda film: True, success),
        ),
        pending="the demo and its routine are being built on another branch",
    )


DRAFTS = (
    grasp("examples/00_getting_started/00_hello_world.py", "cube"),
    grasp("examples/00_getting_started/01_core_concepts.py", "cube"),
    grasp("examples/02_cloth/00_towel.py", "towel", at=Corner("towel", -1, -1, 0.03), lift=0.10),
    Take(
        "examples/02_cloth/00_towel.py",
        name="pinch_towel",
        camera=Orbit(At("towel"), distance=0.7, azimuth=-15.0, elevation=35.0),
        warmup=1.0,
        beats=(
            Wait(seconds=0.5),
            Click(Corner("towel", -1, -1, 0.03), label="click a corner: a handle"),
            Drag("handle", "z", 0.12, seconds=1.2, label="lift it"),
            Drag("handle", "x", 0.08, seconds=1.0, label="carry it"),
            Wait(seconds=0.6, mark="lifted"),
            Press(text="Release all", label="let go"),
            Wait(seconds=1.2),
        ),
        expect=(Rose("towel", 0.02, at="lifted"),),
        pending=PENDING,
    ),
    Take(
        "examples/02_cloth/00_towel.py",
        name="move_together",
        camera=Orbit(At("towel"), distance=0.7, azimuth=-15.0, elevation=35.0),
        warmup=1.0,
        beats=(
            Wait(seconds=0.5),
            Click(Corner("towel", -1, -1, 0.03), label="one corner"),
            Click(Corner("towel", -1, 1, 0.03), label="and the next"),
            Toggle(text="Move together", label="Move together"),
            Drag("handle", "z", 0.10, seconds=1.2, label="one drag lifts both"),
            Drag("handle", "x", 0.10, seconds=1.0, label="and carries the sheet flat"),
            Wait(seconds=0.6, mark="lifted"),
            Press(text="Release all"),
            Wait(seconds=1.2),
        ),
        expect=(Rose("towel", 0.02, at="lifted"),),
        pending=PENDING,
    ),
    grasp("examples/02_cloth/02_engines.py", "towel", at=Corner("towel", -1, -1, 0.03), lift=0.10),
    grasp("examples/02_cloth/04_your_own_cloth.py", "sheet", at=Corner("sheet", -1, -1, 0.03), lift=0.10),
    grasp("examples/03_soft/00_sponge.py", "sponge", lift=0.08),
    Take(
        "examples/03_soft/00_sponge.py",
        name="press_sponge",
        camera=Orbit(At("sponge"), distance=0.5, azimuth=-15.0, elevation=20.0),
        warmup=1.0,
        beats=(
            Wait(seconds=0.5),
            Drag("right", "z", to=At("sponge"), seconds=1.4, label="down onto the sponge"),
            Wait(seconds=0.8, mark="pressed", label="it squashes under the jaw"),
            Drag("right", "z", 0.08, seconds=1.0, label="up"),
            Wait(seconds=1.0, label="and it recovers"),
        ),
        expect=(Squashed("sponge", 0.005, at="pressed"),),
        pending=PENDING,
    ),
    grasp("examples/03_soft/01_tet_mesh.py", "gasket"),
    grasp("examples/03_soft/02_your_own_soft.py", "block"),
    grasp("examples/04_extending/00_add_a_robot.py", "cube"),
    Take(
        "examples/04_extending/01_new_kind.py",
        camera=Orbit(At("chain"), distance=0.7, azimuth=-15.0, elevation=30.0),
        warmup=1.0,
        beats=(
            Wait(seconds=0.5),
            Click(At("chain"), label="click a link: a handle"),
            Drag("handle", "z", 0.15, seconds=1.4, label="lift the chain by it"),
            Wait(seconds=0.8, mark="lifted"),
            Press(text="Release all"),
            Wait(seconds=1.2),
        ),
        expect=(Rose("chain", 0.03, at="lifted"),),
        pending=PENDING,
    ),
    grasp("examples/05_studio/00_your_own_panel.py", "cube"),
    Take(
        "examples/06_teleop/00_leader_arm.py",
        argv=("--simulate",),
        cursor=False,  # a scripted leader arm drives the hands; no one is at the mouse
        warmup=1.0,
        beats=(Wait(seconds=10.0, label="a simulated leader arm drives both hands"),),
        pending=PENDING,
    ),
    Take(
        "examples/06_teleop/01_force_feedback.py",
        argv=("--simulate",),
        cursor=False,
        warmup=1.0,
        beats=(Wait(seconds=10.0, label="a simulated leader feels the panel and the crate"),),
        pending=PENDING,
    ),
    grasp("examples/07_tasks/00_your_first_task.py", "cube"),
    grasp("examples/07_tasks/01_goals_and_rewards.py", "cube"),
    grasp("examples/07_tasks/02_episodes.py", "cube"),
    grasp("examples/08_policies/00_pick_and_place.py", "cube"),
    grasp("examples/08_policies/01_check_and_retry.py", "cube"),
    grasp("examples/08_policies/02_without_privilege.py", "cube"),
    grasp("examples/10_rope/00_cable.py", "cable", at=Segment("cable", 0.5), lift=0.10),
    grasp("examples/10_rope/02_superdex.py", "cable", at=Segment("cable", 0.5), lift=0.10, needs=("superdex",)),
    routine("demo/wipe_the_bar/wipe_the_bar.py", speed=2.0, success="the bar is wiped:"),
    routine("demo/route_the_cable/route_the_cable.py", speed=2.0, success="the cable is in its clips:"),
    routine("demo/fold_the_towel/fold_the_towel.py", speed=2.0, success="the towel is folded:"),
)

TAKES: tuple[Take, ...] = (*FINAL, *DRAFTS)

KEPT: dict[str, str] = {
    "05_studio/01_render": "opens no page: it path traces a still and a video, which is what its clip shows",
    "08_policies/03_many_worlds": "opens no page: a fleet run of every world at once, filmed as a grid of tiles",
    "02_cloth/03_genesis": "needs the Genesis venv, which the recorder's environment deliberately does not carry",
    "live/live_page": "the live page rebuilding on a save of scene.py: a file being edited, not a scene being driven",
    "06_teleop/leader_sim": "the --simulate page 06_teleop/00_leader_arm films; kept until the README shows one of the two",
    "06_teleop/force_feedback_sim": "the --simulate page 06_teleop/01_force_feedback films; kept until the README shows one",
    "09_demos/00_wipe_the_bar": "moving to demo/wipe_the_bar/, whose take films it there",
    "09_demos/01_route_the_cable": "moving to demo/route_the_cable/, whose take films it there",
    "09_demos/02_fold_the_towel": "moving to demo/fold_the_towel/, whose take films it there",
}
"""Clips no take films, each with why its existing file stands. Keyed like :attr:`Take.key`."""


def find(key: str) -> Take:
    for take in TAKES:
        if take.key == key:
            return take
    raise KeyError(key)
