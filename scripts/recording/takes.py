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
    DragTo,
    Expectation,
    Fold,
    Grasp,
    Held,
    Layout,
    Mid,
    Nearest,
    Orbit,
    Policy,
    Press,
    Rose,
    Run,
    Scroll,
    Segment,
    Select,
    Slider,
    Squashed,
    Toggle,
    Top,
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


def _spread(film, name: str, at: str, since: str = "start") -> float:
    return film.marks[at]["spreads"][name] - film.marks[since]["spreads"][name]


def _stiffened(film) -> tuple[bool, str]:
    """The tape stood out flat at ``stiff`` and fell back into its drape by ``end``."""
    out, back = _spread(film, "tape", "stiff"), _spread(film, "tape", "end")
    return out > 0.03 and abs(back) < 0.01, (
        f"the tape's footprint grew {out * 1000:.1f} mm stiffened (need 30) and ended {back * 1000:+.1f} mm from where "
        "it began (need within 10)"
    )


def _succeeded(film) -> tuple[bool, str]:
    """The task's own verdict on the scene as the take left it."""
    result = film.sim.evaluate()[-1]
    return bool(result["success"]), f"the task says success={result['success']}"


RIGID = "examples/01_rigid/00_drag_things.py"
RIGHT = "Arm: right"
"""The right arm's fold; its title may carry a role, ``Arm: right (work)``, which a prefix still matches."""
CUBE_VIEW = Orbit(At("cube", dz=0.06), distance=0.55, azimuth=-10.0, elevation=24.0, fov=50.0)
"""The rigid example's cube and right hand, from the bench's open side: the jaws close across the view."""
PANDA_VIEW = Camera(eye=(0.95, -0.25, 1.00), look_at=(0.457, -0.115, 0.85))
"""The getting-started Panda's hand, which starts 231 mm over the cube, pointing down."""
TASK_VIEW = Camera(eye=(1.15, -0.03, 1.10), look_at=(0.65, -0.03, 0.82))
"""The cube task: the cube at bench (0, 0.1), the square around (0.1, -0.15), and the Panda between."""


def lift(example: str, *, before: tuple[Beat, ...] = (), camera: Any = PANDA_VIEW, **kw) -> Take:
    """The rigid tutorials' gesture: the hand down onto the cube, Close, up 150 mm, held, Open."""
    return Take(
        example,
        camera=camera,
        warmup=1.0,  # a released cube lands and the jaws finish opening before the first frame
        beats=(
            *before,
            Wait(seconds=0.5, label="the hand, open, over the cube"),
            Drag("right", "z", to=At("cube"), seconds=1.4, label="drag the gizmo down onto the cube"),
            Wait(seconds=0.3),
            Press(text="Close", within=RIGHT, label="Close"),
            Wait(seconds=0.9, mark="closed", label="friction holds it"),
            Drag("right", "z", 0.15, seconds=1.4, label="drag it up 150 mm"),
            Wait(seconds=0.8, mark="lifted", label="held"),
            Press(text="Open", within=RIGHT, label="Open: it drops"),
            Wait(seconds=1.2),
        ),
        # The rigid tutorials' own medium test holds the same gesture to a rise over 80 mm.
        expect=(Rose("cube", 0.10, at="lifted"), Held("cube", since="closed", at="lifted", tolerance=0.01)),
        **kw,
    )


def policy(example: str, **kw) -> Take:
    """A task page's Run: the policy carries the cube to the square on its own, and the task scores it."""
    return Take(
        example,
        camera=TASK_VIEW,
        warmup=0.5,
        cursor=True,
        beats=(
            Wait(seconds=0.5, label="the cube, the square, the Panda"),
            Policy(label="Run the policy"),
            Wait(seconds=1.0, label="scored"),
        ),
        # Not the jaw width: under a policy's carry the cube turns in the pads and reads 54 to 58 mm.
        expect=(Check(_succeeded),),
        budget=1_200_000,  # a policy's run is as long as it takes, and these take 15 to 20 s
        **kw,
    )


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
            # The cube's middle 2 mm over where it rests: set down, not dropped.
            Drag("right", "z", to=Bench(z=0.027), seconds=1.2, label="set it down"),
            Press(text="Open", within=RIGHT, label="Open"),
            Drag("right", "z", 0.08, seconds=0.8, label="and away"),
            Wait(seconds=0.8),
        ),
        expect=(Rose("cube", 0.08, at="carried"), Held("cube", since="closed", at="carried")),
    ),
    Take(
        "examples/02_cloth/01_fabrics.py",
        # The tape in the middle, the jersey to its left, the fleece to its right.
        camera=Orbit(At("tape", dz=-0.04), distance=0.75, azimuth=-20.0, elevation=30.0, fov=50.0),
        warmup=3.0,  # the four sheets settle over their posts before anyone looks
        beats=(
            Wait(seconds=0.5, label="four fabrics on four posts"),
            Fold(title="Material", label="open Material"),
            Select(text="Cloth", option="tape", within="Material", label="pick the tape"),
            Scroll(pixels=360, seconds=0.6, label="to its bending stiffness"),
            # Bending, upwards. Measured on this scene, headless: the draped tape's
            # footprint goes 49 -> 98 mm across at edge_ke 100 N m/rad and back to
            # 49.3 at its own 0.1. Softening a settled sheet moves nothing -- the
            # fleece's in-plane stiffness down 100x lowered it 1.3 mm -- because
            # taking a stiffness away does not push; adding one does.
            Slider(text="edge_ke", value=2.0, seconds=1.4, within="Material", label="edge_ke up 1000x"),
            Wait(seconds=1.8, mark="stiff", label="the tape stands out flat"),
            Slider(text="edge_ke", value=None, seconds=1.4, within="Material", label="and back"),
            Wait(seconds=1.8, label="it drapes again"),
        ),
        expect=(Check(_stiffened, needs=("tape",)),),
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
    lift("examples/00_getting_started/00_hello_world.py"),
    lift("examples/00_getting_started/01_core_concepts.py"),
    lift("examples/04_extending/00_add_a_robot.py"),
    lift(
        "examples/05_studio/00_your_own_panel.py",
        camera=Camera(eye=(1.05, -0.20, 1.00), look_at=(0.55, -0.05, 0.85)),
        before=(
            Wait(seconds=0.5, label="a panel of the page's own: Cube"),
            Press(text="Toss the cube", within="Cube", label="Toss the cube"),
            Wait(seconds=1.4, label="up 11 cm at 1.5 m/s, and down"),
            DragTo("right", At("cube", dz=0.10), label="over it"),
        ),
    ),
    Take(
        "examples/06_teleop/00_leader_arm.py",
        argv=("--simulate",),
        cursor=False,  # a scripted leader arm drives the hands; no one is at the mouse, and the gizmos are hidden
        # Both hands: they work at about z 1.15, y -0.5 to +0.5, well over the cube.
        camera=Camera(eye=(1.55, -0.35, 1.30), look_at=(0.55, 0.0, 0.98)),
        warmup=0.5,
        # 9 s: both arms move through most of the frame the whole time, and 12 s was 968 KB at the bottom of the ladder.
        beats=(Wait(seconds=9.0, label="a simulated leader arm drives the follower"),),
    ),
    Take(
        "examples/07_tasks/00_your_first_task.py",
        camera=TASK_VIEW,
        warmup=1.0,
        budget=1_200_000,  # nine drags: the longest gesture of the examples
        beats=(
            Wait(seconds=0.5, label="the cube and the square"),
            DragTo("right", Bench(0.0, 0.1, 0.125), label="over the cube"),
            Drag("right", "z", to=At("cube"), seconds=1.2, label="down onto it"),
            Press(text="Close", within=RIGHT, label="Close"),
            Wait(seconds=0.8, mark="closed"),
            Drag("right", "z", 0.10, seconds=1.0, label="up"),
            DragTo("right", Bench(0.1, -0.15, 0.125), label="over the square"),
            Drag("right", "z", -0.10, seconds=1.0, label="down"),
            Press(text="Open", within=RIGHT, label="Open"),
            Drag("right", "z", 0.08, seconds=0.8, label="and away"),
            Wait(seconds=1.2, mark="placed", label="the goal turns true"),
        ),
        expect=(Check(_succeeded),),
    ),
    policy("examples/08_policies/00_pick_and_place.py"),
    policy("examples/08_policies/01_check_and_retry.py"),
    policy("examples/08_policies/02_without_privilege.py"),
)


# =============================================================================
# Drafts: the gesture planned, on physics other work is still changing
# =============================================================================

PENDING = "physics being changed on another branch; tune after merge"


def grasp(example: str, thing: str, *, at: Any = None, lift: float = 0.12, pending: str = PENDING, **kw) -> Take:
    """The planned gesture as a draft: the nearest hand over the thing, down, Close, up."""
    return Take(
        example,
        camera=kw.pop("camera", Orbit(Mid(At(thing), Nearest(thing)), distance=0.85, azimuth=-15.0, elevation=25.0)),
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


ROPE_VIEW = Orbit(Mid(Segment("cable", 0.5), Nearest("cable")), distance=1.0, azimuth=-20.0, elevation=30.0)

DRAFTS = (
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
        camera=Orbit(Mid(At("sponge"), Nearest("sponge")), distance=0.75, azimuth=-15.0, elevation=22.0),
        warmup=1.0,
        beats=(
            Wait(seconds=0.5),
            # The Panda's hand starts at bench (-0.143, -0.115), off the sponge: over it first.
            DragTo("right", Top("sponge", dz=0.08), label="over the sponge"),
            Drag("right", "z", to=Top("sponge", dz=-0.012), seconds=1.4, label="down into it"),
            Wait(seconds=0.8, mark="pressed", label="it squashes under the jaw"),
            Drag("right", "z", 0.08, seconds=1.0, label="up"),
            Wait(seconds=1.0, label="and it recovers"),
        ),
        expect=(Squashed("sponge", 0.005, at="pressed"),),
        pending=PENDING,
    ),
    grasp("examples/03_soft/01_tet_mesh.py", "gasket"),
    grasp("examples/03_soft/02_your_own_soft.py", "block"),
    Take(
        "examples/04_extending/01_new_kind.py",
        camera=Orbit(At("chain"), distance=0.7, azimuth=-15.0, elevation=30.0),
        warmup=3.0,  # the stack of loose links is still falling over at 1 s
        beats=(
            Wait(seconds=0.5),
            Click(At("chain"), label="click a link: a handle"),
            Drag("handle", "z", 0.15, seconds=1.4, label="lift the chain by it"),
            Wait(seconds=0.8, mark="lifted"),
            Press(text="Release all"),
            Wait(seconds=1.2),
        ),
        # TODO(after merge): the chain is loose links, and a handle lifts the one it
        # holds: measure that link, not the heap's middle (which rose 3.9 mm).
        expect=(Rose("chain", 0.03, at="lifted"),),
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
    grasp("examples/07_tasks/01_goals_and_rewards.py", "cube"),
    grasp("examples/07_tasks/02_episodes.py", "cube"),
    # Back to a metre: the cable runs the bench's width, and the nearer hand starts at its far end.
    grasp("examples/10_rope/00_cable.py", "cable", at=Segment("cable", 0.5), lift=0.10, camera=ROPE_VIEW),
    grasp(
        "examples/10_rope/02_superdex.py",
        "cable",
        at=Segment("cable", 0.5),
        lift=0.10,
        needs=("superdex",),
        camera=ROPE_VIEW,
    ),
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
