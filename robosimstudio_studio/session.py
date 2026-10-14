# SPDX-License-Identifier: Apache-2.0
"""One browser page for a task: load it, drive it, run its policies, keep it, film it.

The loop a person is in while developing a task is *look at it, poke it, run the
graph, watch where it went wrong, film that bit, change something, again* -- and
every lap of it used to cost a process start, which on a tape scene is 40 s. So
this is that loop on one page over one compiled scene:

* **Load** any registered task by picking it from a menu -- selecting loads,
  there is no second button -- and without restarting, because the viewer
  outlives the environment and the browser stays connected across the swap.
* **Drive** it: drag each arm's gizmo, squeeze its fingers on a slider, and
  drag the scene itself around by clicking things and pulling on them.
* **Run** any policy the task ships -- its graphs or its scripted routine --
  in process, from one control.
* **Record** what you drove, as takes, into an ordinary run directory.
* **Export** an mp4 through the offline renderer, from state captured while it
  ran.

The headless half of the stack is ``scripts/bench_run.py``: the same rollout, the same
rows, the same run directory, without a browser. This page is where a person
works; that is where a machine does.

**The interaction is a drag and a slider.** Each arm carries a 6-DOF transform
gizmo on its tool centre and tracks it every step; the fingers are a Close
slider. Nothing is proposed, flown or welded -- what a hand ends up holding is
whatever its friction and the contact settings let it hold, which is the thing a
stack coupling rigid fingers to a VBD cloth in one substep exists to show.

A click resolves the ray against the model's own geometry
(:mod:`~robosimstudio_studio.pick`) and then *takes hold* of what it
found -- a wrench on a body, a pin on the patch of cloth under the cursor -- on
a handle that is draggable the moment it is drawn
(:mod:`~robosimstudio_studio.handles`). :data:`StudioConfig.handles` of
them per kind, so a sheet can be held by two corners and carried as a pair.

**Two modes, one loop.** Teleop steps with
:meth:`~robosimstudio.core.env.EnvBase.step_teleop`, which advances from whatever
the panels wrote into ``joint_target_q`` -- an IK solve of a grasp's current leg,
or of the cursor being carried. A *policy* run -- a graph or the task's scripted routine, from the one
Policy control -- steps with
``env.step(policy.get_action(obs), mode=policy.control_mode)``, the exact shape
a headless rollout uses, because a policy that behaved
differently here than under ``scripts/bench_run.py`` would make this page a liar about
the numbers it shows. The panels that write joint targets (the arms) are not
drained while a policy holds the arm, and neither is the episode panel, whose
``drain`` ticks the stateful evaluator that ``env.step`` is already ticking.

**Export is deferred; the live view is opt-in.** Capturing is copying ``body_q``
and ``particle_q`` into a ring buffer -- microseconds -- and the render happens
when you press Export, as an offline replay of that buffer
(:class:`~robosimstudio.core.render.offline.OfflineRenderer`). No solver runs, so the video is the trajectory you watched, photographed
differently. That is the right default because a *deliverable* is better looked
at afterwards, larger, and through a better backend.

Watching through the cameras is a different question -- whether the rig a policy
is scored through is pointed at anything, whether the depth channel has the
object in it -- and the ``Cameras`` folder answers it, off until asked. It
renders on the loop's own thread, so the frames it draws are time the simulation
does not get; :class:`~robosimstudio.core.render.live.RenderBudget` bounds that
share against a *measured* frame cost and the readout says what it is actually
taking. The environment's own rig (``obs_renderer``) runs only while something
reads it -- a policy with perception nodes, a take being recorded, or that view
on its observation source. An idle scene rendering it for nobody cost 19% of its
step rate, which is why :meth:`StudioSession.wants_pixels` is a question.

**Refreshing the browser is paced on wall clock, not counted in steps.**
``render_every`` is a frame rate to hold, and a state the viewer already has is
not re-sent; see :meth:`StudioSession.refresh_view` for why a step count was the
wrong meter and why the render budget is not the right replacement.

The viser threading contract from
:mod:`~robosimstudio_studio.panels` holds throughout: callbacks assign
plain attributes and nothing else, and every Warp and file touch happens on the
main thread inside ``drain``.
"""

from __future__ import annotations

import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robosimstudio.core.profile import parse_modes
from robosimstudio.core.render.palette import GROUND
from robosimstudio.scene.registry import get_cell
from robosimstudio.specs import RecordSpec, TaskSpec

from . import seams
from .cameras import PanelCamera
from .controls import PanelRecord, PanelVideo
from .garments import GARMENT_STYLES, garment_visuals, scene_repaints_live, viewer_studio
from .inspector import PanelInspector
from .leader_arm import PanelLeaderArm
from .lighting import LightingRig, PanelLighting
from .overlay import PanelOverlay
from .panels import Panel, PanelArm, PanelCloth, PanelObject
from .pick import ScenePicker
from .regions import PanelRegions
from .render_view import PanelRenderView
from .scene_panel import PanelScene
from .scenery import draw_scenery
from .selection import SceneIndex, Subject
from .tuning.panels import PanelReadings, PanelTune

__all__ = ["CONTROL_PANELS", "SPEEDS", "Refused", "StudioConfig", "StudioSession"]

CONTROL_PANELS = (PanelArm, PanelLeaderArm)
"""Panels the studio does not drain while a policy is driving.

The arm panels write ``joint_target_q``, which is the policy's to write. The
leader-arm panel writes the same ``joint_target_q`` the arm panels do -- another
live input, not a second policy -- so it sits out for the same reason they do.

The goal panel sits out with them, and is not in this tuple only because its
class belongs to the page's benchmark half and cannot be named here: it calls
``env.evaluate()``, which ``env.step`` already calls once per step, and an
evaluator counting consecutive grasps is not indifferent to being asked twice.
:meth:`StudioSession.tick` skips it by identity -- the session's own
``goal_panel`` -- which is the same set of panels this tuple held when it could
name all three.
"""

STAGE_THICKNESS = 0.012
"""The stand-in work surface's slab thickness [m]. Thin on purpose: the real
table was hidden because its bulk swallowed the arm gizmos, and a stage that
brought the bulk back would only re-earn the hiding."""

STAGE_CLEARANCE = 0.0015
"""How far the stage top sits below the real work surface [m]. The hidden
furniture's top is exactly ``surface_z`` and so is everything resting on it, so
a stage drawn flush would be coplanar with every settled cloth and shimmer."""

STAGE_MARGIN = 1.15
"""Stage size as a fraction of the furniture it stands in for. A little larger,
so an object near the table edge still has ground under it in the picture."""

STAGE_COLOR = tuple(round(255.0 * c) for c in GROUND)
"""The stand-in's colour, 0-255. :data:`~robosimstudio.core.render.palette.GROUND`,
which is the dark slate every real table top is already painted.

Derived rather than chosen. It was a warm off-white, picked on its own merits --
seamless paper, light enough for contact shadow to be what gives the scene
depth. The merits were real and the value still disagreed with the thing it
stands in for: the slab is drawn *because* the bench is hidden, so toggling
``Show scenery`` swapped a near-white ground for a dark one and the picture
changed more than the visibility of one box. Taking the palette's own answer for
"table top, and anything else a task merely happens on top of" makes the toggle
show the furniture and nothing else, and inherits the crush measurement behind
that entry -- high enough that nothing painted against it is lost, low enough
that everything a task is about still reads.

Two things do change with the value, and both are the palette's trade already:
contact shadow has less room to darken into, so *thickness* leans more on the
environment map's shading across a curve and less on the shadow under it; and
the pale things -- a cream towel, a white film, a steel tool -- now separate
from the ground immediately instead of being found against it.
"""

SPEEDS: dict[str, float] = {"1x": 1.0, "0.5x": 0.5, "0.25x": 0.25, "0.1x": 0.1, "max": 0.0}
"""Playback rates, as a fraction of the task's control rate. ``max`` is unpaced.

Slow motion earns its place on a page about deformables: a peel front travelling
along a strip, or cloth sliding out from between two fingers, happens over a
handful of control steps at 1x and is the whole result.
"""


PANEL_SETS: dict[str, tuple[str, ...]] = {
    "all": (),
    "drive": ("inspector", "overlay", "arm", "object", "cloth"),
}
"""Named panel selections for :attr:`StudioConfig.panels`. Empty means no filter.

``drive`` is the page as a *demonstration*: the things you take hold of, and the
sliders behind whichever one you clicked. What it leaves out is the half of the
studio that is about producing something -- cameras, video, takes, material
presets, regions, the event log, a leader arm -- which is exactly the half an
example has no use for and the reason ``examples/`` was reported as showing too
much. The scene list goes with them: clicking a thing is the way in, and a
dropdown over every subject in the scene is a second one nobody asked for. The
overlay stays with the inspector: it has no folder of its own, and it is what
the inspector's ``Draw`` checkboxes draw with.

Two things are not panels and are handled either side of this. Transport --
Pause, Speed, Reset -- is built inline by :meth:`StudioSession.build_gui` and
every selection keeps it, because a page you cannot stop is not a page. The
``Appearance`` fold is built inline too but *is* gated, through
:meth:`StudioSession._wants`, because how the bench is shaded is a decision made
once a session and a page showing one rope never makes it.

A task's page keeps :data:`DOCUMENT_PANELS` as well, under any named selection.
"""

DOCUMENT_PANELS: tuple[str, ...] = ("goal", "solution", "regions")
"""What a named selection keeps on a *task's* page, on top of its own keys.

A selection is named for what a page is showing, and on a task the document is
that: what is asked (``goal``), the regions it is asked about (``regions``),
and the policies that answer it (``solution``, the Policy folder and its Run).
``drive`` was written for hand-assembled scenes and dropped all three, so
:func:`robosimstudio.view` on a task -- whose default is ``drive`` -- opened a
page with a cube, a hand and no square to put it in, no clause to turn true and
no Run to press. A bare scene has no document, and there a selection is
exactly its keys.

``task``, the menu, is not one of them: it swaps the page onto another
document, and a page opened on one task is about that task. Keys given
explicitly are taken as they are, because a caller who listed them has said
exactly what the page is for.
"""


def panel_key(panel: Any) -> str:
    """What :attr:`StudioConfig.panels` calls one panel.

    Derived from the class name rather than declared on a base class, because
    there are two panel bases and only one of them is a class: the tuning
    panels duck-type the protocol on purpose (see
    :class:`robosimstudio_studio.tuning.panels.Panel`) so the studio's module
    is not imported back into them. A ``key`` attribute wins where the derived
    spelling is wrong.
    """
    return str(getattr(panel, "key", "") or type(panel).__name__.removeprefix("Panel").lower())


@dataclass(frozen=True, kw_only=True)
class StudioConfig:
    """How a studio session is set up. Every field has a working default.

    Grouped by what it configures rather than alphabetically, because the groups
    are what a caller changes together: the environment under the page, the page
    itself, how a graph is run, and what an export produces.

    Attributes:
        scene: Which registered place to stage the task in, or ``""`` for its
            own. Must be a pairing the task declares -- the page offers only
            those, so this is for starting somewhere other than the default.
        embodiment: Which registered robot, or ``""`` for the task's own.
        gripper: Which registered hand to put on every arm, or ``""`` for the
            robot's own. Same rule.
        device: Warp device, or ``None`` for the default.
        control_mode: Override every task's declared action space. A task's
            shipped graph and routine may not speak the same one.
        max_steps: Control steps before a policy run truncates; 0 uses the
            task's own, which is sized per variant. Teleop ignores it.
        obs_renderer: Backend for the cameras a policy observes through; ``""``
            gives it no pixels. This is the environment's own rig, not a view --
            a graph with perception nodes needs it. The rig is built either way
            and only *runs* while something reads it, so leaving this on costs
            build time rather than step rate.
        obs_render_every: Control steps between observation renders.
        port: viser port; the next one up is taken when it is busy. The page
            listens on every interface (``0.0.0.0``), not only ``localhost`` --
            Newton's viewer has no host option -- so anyone who can reach the
            port can drive the scene: firewall it on a shared network, or reach
            it over an SSH tunnel (``ssh -L 8080:localhost:8080 <box>``).
        share: Request a public viser share URL as well. Off by default.
        out: Where run directories, videos and traces go.
        render_every: How often to refresh the browser, as one frame per this
            many control steps *at the task's control rate*. It is a frame rate,
            not a step count, so an unpaced session holds this picture rate
            rather than drawing as fast as it happens to step -- see
            :meth:`StudioSession.refresh_view`.
        speed: Initial playback rate, a key of :data:`SPEEDS`.
        record: What a recorded take keeps.
        settle_steps: Steps to keep stepping after a graph finishes, so the
            evaluator scores a settled scene.
        privileged: Start with the ``sim.*`` tools available to a graph.
        roll: The roll contract -- ``auto``, ``free`` or ``held``.
        motion: How ``robot.go_to_pose`` moves -- ``osc`` or ``plan``.
        video_fps: Frame rate of an exported mp4.
        capture_every: Capture one control step in this many.
        buffer: Frames of state kept for export.
        leader_arm_ports: Starting serial device per arm name, e.g.
            ``(("left", "/dev/yam_leader_left"), ("right", "/dev/yam_leader_right"))``
            -- a bimanual leader is two independent Feetech buses, one per arm,
            not one bus daisy-chaining both. Only a starting value for the
            panel's own per-arm fields; an arm named here that the loaded
            pairing does not have is simply unused.
        leader_arm_calibration: A JSON calibration file, one entry per arm
            name; see :func:`~robosimstudio.robots.teleop.lerobot_controller.load_calibration`.
            Also only a starting value.
    """

    # the environment under the page
    scene: str = ""
    embodiment: str = ""
    gripper: str = ""
    device: str | None = None
    control_mode: str = ""
    max_steps: int = 0
    num_worlds: int = 1
    """Worlds in the page. 1 is the interactive scene; above one the page
    becomes a grid of worlds each running its own episode, and the tools that
    drive *a* scene -- the arm gizmo, picking, handle dragging -- go away,
    because there is no sensible version of them across three hundred worlds."""
    fleet_solution: str = ""
    """What drives the worlds when there is more than one. Empty holds them."""
    fleet_solution_kwargs: tuple[tuple[str, Any], ...] = ()
    fleet_loop: bool = False
    """Start the next wave when one finishes, so a page left open keeps working."""
    show_worlds: int = 64
    """Draw at most this many of them, whatever the world count."""
    obs_renderer: str = "tiled"
    obs_render_every: int = 4
    # the page
    panels: str | tuple[str, ...] = "all"
    """Which panels the page builds: a key of :data:`PANEL_SETS`, or the keys.

    ``all`` is every panel the scene warrants, which is what a launcher and the
    benchmark want. :func:`robosimstudio.view` passes ``drive`` instead --
    see its docstring for why the two doors default differently. A name keeps
    :data:`DOCUMENT_PANELS` too on a task's page.
    """
    port: int = 8080
    share: bool = False
    dark_mode: bool = False
    """Dark browser chrome, and with it a near-black canvas.

    viser has no separate canvas colour: the 3D view's background follows the
    UI theme, ``#fff`` in light mode and Mantine's darkest grey in dark. The
    studio shipped hard-coded to dark, which is why every scene sat in a black
    void; light is the default now because the page is a product-viz stage --
    shadows on a white ground read, and a cream towel is not a silhouette.
    """
    out: str = "outputs/studio"
    material_dir: str = "assets/materials"
    cameras: bool = True
    """Offer the ``Cameras`` folder at all, on a task that declares any."""
    live: bool = True
    """Start with the live camera view on.

    On, because the rig is part of the task: a page that opens on a scene
    without showing what the scene is *photographed as* hides the thing half
    these tasks are scored through, and "why is the policy failing" is a
    question about the pixels more often than it is a question about the
    geometry. The budget below is what makes this affordable to leave on --
    it costs a bounded share of the loop rather than an unbounded one.
    """
    live_frustums: bool = True
    """Draw each task camera in the 3D scene as a frustum showing its own view.

    Click one to fly the browser there and look through it. This is also where a
    camera is *aimed*: with a frustum selected, ``Set from view`` adopts wherever
    you have flown to as that camera's pose.
    """
    live_fps: float = 15.0
    live_load: float = 0.3
    """Largest share of wall clock the live view may spend rendering.

    Measured, not assumed -- see :class:`~robosimstudio.core.render.live.RenderBudget`.
    A backend that costs more than the loop can spare self-limits against this
    instead of eating it -- and a path tracer's *first* frame, which is a USD
    export and a scene load rather than a render, costs seconds.
    """
    live_width: int = 0
    live_height: int = 0
    """Resolution for the ``own renderer`` source; 0 uses each camera's own."""
    live_channels: tuple[str, ...] = ("rgb",)
    backdrop: bool = True
    """Draw the scene's registered backdrop -- the floor, the walls, the lab set.

    Not to be confused with :attr:`show_scenery`, which is about the *model's*
    own statics (the bench, the table) and stays off because their bulk swallows
    the gizmos. This is the room around them, which the path tracer has always
    drawn and the page never did: ``room.py`` records the measurement that says
    why it matters more than the lights do -- two dome HDRIs of quite different
    radiance rendered to 86.77 and 86.95, because the tracer ignores the dome's
    texture entirely. **What reaches the picture is geometry.**
    """
    world_model: bool = False
    """Start with the **World model** switch on: the page's 3D view is then a
    path-traced frame of the browser's own camera, with the gizmos and handles
    drawn over it -- see :mod:`robosimstudio_studio.render_view`. Off is
    viser's own WebGL scene.

    **Off by default.** A page should open at the speed it can always sustain
    and let somebody ask for the expensive view; it also costs a browser about
    a minute to get its first path-traced frame, and a page that spends its
    first minute looking like it has not loaded reads as broken. Ignored
    where ``ovrtx`` is not installed.
    """
    render_view_scale: float = 0.5
    """Fraction of the viewport the path-traced frame is rendered at.

    Half scale renders in about a quarter the time of full and the browser
    upscales it; below a half the curve flattens, because what the tracer costs
    per frame is mostly passes rather than pixels. 1.0 is pixel-for-pixel.
    """
    render_view_passes: tuple[int, int] = (1, 6)
    """Tracer passes while moving, and once still.

    One pass is noisy and live (~40 fps at half scale); six is what an export
    converges over. The pair is the whole quality dial, because ``accumulate``
    is read per render.
    """
    show_scenery: bool = False
    """Draw the scene's own furniture -- the bench, the table, the arena.

    Off by default, and the reason is the gizmo rather than taste: an arm's
    transform control sits on the tool centre, which on a tabletop task is a few
    centimetres above a table the size of the view. The gizmo ends up *inside*
    that geometry, where the browser hit-tests the table first and the handles
    cannot be grabbed at all. Hiding the furniture leaves the robot, the objects
    and the cloth, which is what there is to look at anyway.

    **Furniture means the scene's statics, not the task's.** The panel a film
    is stuck to and the box a tape is wound onto are the task, and a peel off
    an invisible substrate is not a picture of a peel -- so only the statics
    the *scene* declares (the bench, the table) are hidden, and in their place
    a plain light stage slab is drawn at the work surface so the cloth still
    rests on something and the key light still has somewhere to cast.

    Display only. Collision is untouched -- only the ``VISIBLE`` flag is cleared,
    never ``COLLIDE_SHAPES`` -- so the bench a towel is resting on goes on
    holding it up while invisible.
    """
    render_every: int = 4
    speed: str = "1x"
    handles: int = 2
    """How many things may be held by hand at once, per kind.

    Two, because one grip can only translate a sheet and two can fold it --
    which is the same reason PhysTwin's playground offers ``--n_ctrl_parts 2``.
    Each slot costs a dragger, and a dragger is a substep hook that has to be in
    the captured graph, so this is fixed when a task loads rather than grown on
    demand; the cost of an unused slot is a kernel launch over nothing.
    """
    cloth_look: str = "fluffy"
    """How the deformables are drawn -- ``fluffy``, ``shell`` or ``flat``.

    See :mod:`~robosimstudio_studio.garments`. A starting value only:
    like the lighting it is judged by looking, so the live control in the View
    folder is the one that matters.
    """
    lighting: LightingRig = field(default_factory=LightingRig)
    record: RecordSpec = field(default_factory=RecordSpec)
    # running a graph
    settle_steps: int = 60
    privileged: bool = True
    roll: str = "auto"
    motion: str = "osc"
    # exporting video
    video_fps: int = 30
    capture_every: int = 2
    buffer: int = 900
    # teleop from a leader arm, alongside the gizmo drag
    leader_arm_ports: tuple[tuple[str, str], ...] = ()
    leader_arm_calibration: str = ""
    # measuring the loop
    profile: str = ""
    """Which profiling instruments to arm: ``wall``, ``device``, ``syncs``.

    Empty is a normal session, and costs about a hundred nanoseconds per phase
    for the disabled spans the loop carries unconditionally. See
    :class:`~robosimstudio.core.profile.Profiler` for what each mode measures
    and what it costs to measure it.
    """
    profile_steps: int = 0
    """Stop and report after this many stepped iterations. 0 serves normally."""
    profile_warmup: int = 0
    """Steps to run before the measurement starts.

    Not optional in practice on this page. A browser viewer's first frames cost
    ~300 ms and fall to ~11 ms after a hundred-odd of them, a scene's first
    steps run eager until the graph is captured, and a tiled sensor compiles its
    megakernel on first use -- so a window that includes the warm-up measures the
    warm-up. :func:`~robosimstudio.core.profile.Profiler.reset` at the boundary
    keeps the phase names and discards their samples.
    """
    profile_out: str = ""
    """Where the JSON sidecar lands; empty puts it under ``out/profile``.

    All three are read by :meth:`StudioSession.run`, as the defaults for its
    own ``steps`` and ``warmup``; the sidecar is written only when
    :attr:`profile` armed something."""


class Refused(ValueError):
    """This pairing will not be compiled, and the message *is* the reason.

    A named type only because of how it is reported. A refusal is a sentence
    somebody wrote about the pairing -- ``admits``' words, or an embodiment's
    own :attr:`~robosimstudio.specs.EmbodimentSpec.unbuildable` -- and printing
    a stack over it buries the one line that answers the click. A compile that
    broke is the other case, and there the stack is what a reader needs.

    Both leave the page untouched and put the menu back, which is the part that
    must not differ: this is a load that did not happen, not a new failure mode.
    """


class StudioSession:
    """The page: one viser server, one environment at a time, one loop.

    The viewer outlives the environment, which is what lets the task menu load
    another scene under a browser that stays connected: ``set_model`` swaps the
    model and the panels are recomposed from the new scene's handles.

    **One class over two kinds of scene.** A built
    :class:`~robosimstudio.sim.Sim` is adopted as it stands and needs nothing
    beyond the library. A *task* -- an id, a ``TaskSpec`` -- has to be resolved
    and compiled, and brings a menu, a goal and policies with it; that half
    lives in the page's benchmark add-on and is reached through :meth:`_bench`,
    so the class is the same wherever the benchmark tier is installed and says
    one sentence about it where it is not.

    Args:
        task: What to open: a built scene, or a task as ``"<suite>/<variant>"``.
        episode: Which episode to start on.
        config: Everything else.
    """

    _narrowed = ""
    """Why the page built one world when more were asked for; cleared once said."""

    def __init__(self, task, *, episode: int = 0, config: StudioConfig | None = None) -> None:
        """Open a page.

        Args:
            task: A registered task id, a built
                :class:`~robosimstudio.specs.TaskSpec`, or a
                :class:`~robosimstudio.sim.Sim` that is already built. The last
                is the whole point of the tier split: a scene you assembled by
                hand gets the same page a benchmark does, minus the panels that
                need a document.
            episode: Which episode to start at. Ignored for a ``Sim``.
            config: Ports, renderers, lighting, output directory.
        """
        self.config = config or StudioConfig()
        if not hasattr(task, "pipeline"):
            # A task, which is the benchmark half's to load. Asked for before
            # the viewer exists, so an install without that half answers with
            # its one sentence rather than with a server left on the port.
            self._bench(f"opening the studio on the task {getattr(task, 'full_name', task)!r}")
        self.viewer = viewer_studio(port=self.config.port, share=self.config.share)
        if self.config.cloth_look in GARMENT_STYLES:
            self.viewer.set_garment_style(self.config.cloth_look)
        self.server = self.viewer._server
        # Immediately after the viewer, which lights the scene with a single
        # ambient light and no environment map -- directionless, so nothing is
        # shaded by its own geometry and a folded sheet is indistinguishable
        # from a flat one. Applied to the server rather than to the environment,
        # so it survives a task being swapped underneath the page.
        self.lighting = PanelLighting(self.server, self.config.lighting, viewer=self.viewer)
        self.render_every = max(1, int(self.config.render_every))
        self.speed = SPEEDS.get(self.config.speed, 1.0)
        self.release_last_requested = False
        self.release_all_requested = False
        """A release a viser thread asked for, serviced on the main thread."""
        self.linked_request: bool | None = None
        """``Move together``, which is one setting over every kind's handles."""
        self._holding_readout: Any = None
        self._linked_toggle: Any = None
        self._release_last: Any = None
        self._release_all: Any = None
        self._each_step: list[Any] = []
        """What :meth:`each_step` was given, drained beside the panels."""
        self.root = Path(self.config.out)
        # Filled in per load: the interval is render_every at the *task's*
        # control rate, and the task changes under this page.
        self.draw_interval = 0.0
        self.view_fps = 0.0
        """Measured browser refresh rate, for the page's own readout."""
        self._drawn_step: int | None = None
        """Step the viewer was last given. ``None`` forces the next draw.

        Not a sentinel step number, because a reset puts the count back to 0 and
        "0 is fewer than ``render_every`` steps on from -1" would hold the
        viewer on the previous episode's picture until the new one had run four
        steps -- which on a scene reset while paused is forever.
        """
        self._refreshed_at = 0.0
        self._fps_window = (0.0, 0)

        self.profile = parse_modes(self.config.profile)
        """Where the loop's time went, when someone asked. Disabled by default.

        Held rather than consulted: the disabled profiler's spans are no-ops, so
        the loop carries its ``with`` blocks unconditionally and pays about a
        hundred nanoseconds each for them when nothing is being measured. A
        benchmark can also assign one directly before :meth:`run`.
        """
        # Pacing and rate-readout state, set up by begin_run. Attributes rather
        # than locals so tick() can be driven from outside run().
        self._dt = 0.0
        self._next_frame = 0.0
        self._rate_frames = 0
        self._rate_window = 0.0
        self.measured_steps = 0
        self.measured_wall = 0.0
        """The post-warm-up window :meth:`run` timed, profiler or no profiler."""

        # --- state the loop reads, written only by callbacks -------------------
        self.paused = False
        self.pause_box: Any = None
        """The page's Pause checkbox, once :meth:`build_gui` has made it; :meth:`set_paused` keeps it truthful."""
        self.reset_requested = False
        self.quit_requested = False
        self._captured_step: int | None = None
        """Step the video and the take last captured; a step taken by a script while
        the page holds is captured on the next pass. See :meth:`_capture_outside_steps`."""
        self.load_request: tuple[str | TaskSpec, int, str, str, str, int] | None = None
        self.num_worlds = max(1, int(self.config.num_worlds))
        """Worlds the page is showing. Changing it and reloading swaps the page
        between the interactive scene and a grid."""
        self.fleet: Any = None
        """The grid, when there is more than one world. It borrows this
        session's viewer, so changing the count reloads in the open tab."""
        self.show_scenery = bool(self.config.show_scenery)
        self.scenery_request = False
        """Set by the View checkbox, serviced by the loop -- it re-batches the
        viewer, which is not something to do from a viser callback thread."""
        self.cloth_look_request: str | None = None
        """Set by the View dropdown, serviced by the loop for the same reason:
        switching styles retires and rebuilds scene nodes the draw is
        iterating over."""
        self.click_ray: tuple[np.ndarray, np.ndarray] | None = None
        self.status = ""

        self.env: Any = None
        """A ``Sim``, or the benchmark's ``Env`` when the page is showing a task."""
        self.obs: dict[str, Any] = {}
        self._stage: Any = None
        self._backdrop: Any = None
        """The stand-in work-surface node, when the furniture is hidden."""
        self.episode_id = int(episode)
        self.panels: list[Panel] = []
        self.arms: dict[str, PanelArm] = {}
        self.active_arm = ""
        self.picker: ScenePicker | None = None
        self.driver: Any = None
        """The policy runner. ``None`` unless the page is showing a task."""
        self.record: PanelRecord | None = None
        self.video: PanelVideo | None = None
        self.objects: PanelObject | None = None
        self.deformables: PanelCloth | None = None
        self.tune: PanelTune | None = None
        self.readings: PanelReadings | None = None
        self.live: PanelCamera | None = None
        self.task_panel: Any = None
        """The task menu. ``None`` unless the page is showing a task."""
        self.goal_panel: Any = None
        self.inspector: Any = None
        self.leader_arm: PanelLeaderArm | None = None

        self.index: SceneIndex | None = None
        """Every subject in the compiled scene, and the click lookups. Rebuilt
        per task load, because it is a map from one model's numbering to names."""
        self.selected: Subject | None = None
        """What the page is about. One place, rather than a private string in
        each of two panels with nothing else able to ask."""
        self.select_request: str | None = None
        """A subject key a callback asked for, serviced on the main thread."""
        self.overlay_request: list[tuple[str, str, bool]] = []
        """``(subject key, feature name, show)`` from the inspector's checkboxes,
        drained by the overlay for the same reason -- a callback may only
        assign."""

        # Registered once, on the server rather than on any environment: a task
        # load rebuilds the panels but not the page, and a pointer callback that
        # had to be re-registered per load would be one more thing to forget.
        @self.server.scene.on_click()
        def _(event) -> None:
            self.click_ray = (
                np.asarray(event.ray_origin, dtype=np.float64),
                np.asarray(event.ray_direction, dtype=np.float64),
            )

        if hasattr(task, "pipeline"):
            # Already built. Nothing to compile, nothing to look up, and no
            # failure mode to fall back from.
            self.adopt(task, episode=self.episode_id)
        elif not self.load_task(task, self.episode_id, self.config.scene, self.config.embodiment, self.config.gripper):
            # Nothing to fall back to on the first load, and a page with no
            # scene in it is not worth serving.
            raise RuntimeError(f"{task} did not compile; see the traceback above")

    # ---------------------------------------------------------------- task load
    @property
    def current_task(self):
        """The task the page is showing, however many worlds it is showing it in.

        The task menu is the one panel that belongs to *both* pages -- it is how
        the world count is set in the first place -- so what it reads has to be
        the session's answer rather than an environment's.
        """
        if self.fleet is not None:
            return self.fleet.task
        return getattr(self.env, "task", None) if self.env is not None else None

    def adopt(self, sim, *, label: str = "", episode: int = 0) -> None:
        """Put an already-built scene into this page, replacing whatever was in it.

        The half of a load that is not about *tasks*: swap the model
        into the viewer, rebuild the panels around it, reset, and draw. Split out
        because a :class:`~robosimstudio.sim.Sim` somebody assembled by hand has
        nothing to compile and no registry entry to look up, and the page has no
        reason to care which it is showing -- what it needs is a model, handles
        and a step function, and both kinds have all three.

        Args:
            sim: A ``Sim``, or the benchmark's ``Env`` -- which is one.
            label: What to call it in the run directory and the status line.
                Defaults to the request's own name.
            episode: Which episode to reset to. Ignored by a bare ``Sim``, whose
                ``reset`` takes no argument because it has no episodes.
        """
        label = label or getattr(sim.request, "full_name", "") or "scene"
        self.teardown()
        self.env = sim
        self.episode_id = int(episode)
        # render_every is read at the new scene's control rate, so a 60 Hz scene
        # and a 50 Hz one are both refreshed at every-fourth-step's worth.
        self.draw_interval = self.render_every / float(sim.request.physics.control_hz)
        self.run_dir = self.root / label.replace("/", "_") / time.strftime("%Y%m%d-%H%M%S")
        self._bind(sim)
        obs, info = self.reset_env()
        self.obs = obs
        self.on_reset(obs, info)

    def _bind(self, sim) -> None:
        """Hand the page a model: the viewer, the look, and the panels built around it.

        The half of :meth:`adopt` that is about the *model*, apart so that a
        reset which recompiled one (an episode whose model key differs) can
        redo it without closing the environment that just rebuilt itself.
        """
        self.picker = ScenePicker(sim.model)
        # Before set_model: the viewer batches shapes by their flags when the
        # model is handed over, and reads visibility off that batch each frame
        # rather than off the model.
        self._set_scenery_visible(self.show_scenery)
        self.viewer.set_model(sim.model)
        # After set_model, whose clear retires the previous scene's garments:
        # which deformables exist, their colours and their fabric thicknesses are
        # all properties of the scene just compiled.
        self.viewer.set_garments(garment_visuals(sim.scene, getattr(sim, "doc", None)))
        # And whether anything in it repaints itself as it runs -- a mark being
        # polished off, a tape seam parting. Asked here because a scene is the
        # last thing that still knows; the viewer holds only a model.
        self.viewer.set_live_colors(scene_repaints_live(sim.scene))
        self._push_finishes(sim)
        # Newton's viser viewer installs its own flat ambient and clears the
        # HDRI whenever it sets up a scene, so a model swap wipes the rig and
        # the cloth comes back unlit. Re-applying is idempotent -- every node is
        # addressed by name -- and this is the only place a swap happens.
        self._adopt_look_lighting()
        self.lighting.apply()
        self._apply_backdrop()
        self._apply_stage()
        self.build_panels()
        self.build_gui()

    def _bench(self, what: str = "this"):
        """The page's benchmark half, imported on first use.

        Everything a session does that needs a *task* -- resolving an id,
        compiling an ``Env``, the task menu, the goal readout, the policy
        runner, the grid of worlds -- lives in
        ``robosimstudio_studio.bench_session`` and its neighbours, which a
        release without the benchmark tier does not carry. This is the session's
        one way to them.

        Args:
            what: What is being asked for, as the subject of the sentence a
                refusal is written in.

        Raises:
            robosimstudio_studio.seams.NotInThisRelease: If the tier is absent.
        """
        return seams.benchmark(what)

    def load_task(
        self,
        task: str | TaskSpec,
        episode: int,
        scene: str = "",
        embodiment: str = "",
        gripper: str = "",
        num_worlds: int | None = None,
    ) -> bool:
        """Compile a task into this page, replacing whatever was in it.

        ``task`` is normally a registry name. It may also be a built
        :class:`~robosimstudio.specs.TaskSpec`, which is how a rebuild carries
        edits the registry does not know about -- a tuned material, most
        obviously.

        The work is the benchmark half's, and so is the account of it: which
        pairings are synthesized, which are refused and in whose words, why the
        new scene is compiled before the old one is given up, and what a world
        count above one does. See ``robosimstudio_studio.bench_session.load_task``.

        Returns:
            Whether the task was loaded. On ``False`` the page is untouched.

        Raises:
            robosimstudio_studio.seams.NotInThisRelease: If the benchmark tier
                is absent, since there is then no such thing as a task.
        """
        label = task if isinstance(task, str) else task.full_name
        return self._bench(f"loading the task {label!r}").load_task(
            self, task, episode, scene, embodiment, gripper, num_worlds
        )

    def teardown(self) -> None:
        """Give back everything the previous task owned. Main thread.

        The page is cleared first and the backend released after, and the order
        of the *guards* matters as much as the order of the work: this used to
        return early when there was no environment, which is exactly the case a
        grid is in -- so a reload left the old widgets standing, built a second
        Task folder beside them, and left every click landing on a panel that
        was no longer being drained. Two menus, neither of which worked.
        """
        if self.env is None and self.fleet is None:
            return
        self._drop_panels()
        if self.fleet is not None:
            self.fleet.close()
            self.fleet = None
        if self.env is not None:
            self.env.close()
            self.env = None

    def _drop_panels(self) -> None:
        """Close every panel and clear the page's widgets, leaving the environment be."""
        for panel in self.panels:
            panel.close()
        self.panels, self.arms = [], {}
        self.active_arm = ""
        self.picker = None
        self.driver = self.record = self.video = self.task_panel = None
        self.objects = self.deformables = self.tune = self.readings = self.live = None
        self.leader_arm = None
        # Cheaper and more certain than removing widgets one at a time, and the
        # only GUI on this page is the panels' own.
        self.server.gui.reset()
        self.pause_box = None

    def _push_finishes(self, sim) -> None:
        """Hand the scene's materials to the page, so it shades what RTX shades.

        The same table the path tracer is given -- ``handles.finishes``, which
        the compiler fills for *every* shape rather than only the stated ones --
        and the same ``live_shapes``, which stay on the flat per-instance path
        because a GLB bakes its colour in.

        Read off the handles rather than through ``Sim._scene_facts``, which
        looks like the shared door and is not: it takes the camera specs it is
        being asked on behalf of, and its values are closures a backend's
        ``wants`` list pulls on. This is the same table one layer down.

        A scene with no handles leaves the page drawing flat, which is what it
        did before any of this existed. That is the *only* thing tolerated here:
        this used to catch everything, and what it caught was
        ``handles.live_shapes or ()`` truth-testing a numpy array -- so the page
        silently drew flat while the table it needed sat right there, fully
        populated. A guard that wide hides the bug it is standing next to.
        """
        handles = getattr(sim, "handles", None)
        if handles is None:
            self.viewer.set_finishes({}, ())
            return
        self.viewer.set_finishes(getattr(handles, "finishes", None), getattr(handles, "live_shapes", None))

    def build_panels(self) -> None:
        """Compose the page from this scene's handles, plus the studio's own.

        Three regions, in the order somebody reads them:

        **Goal** -- what is being asked, clause by clause, each with its verdict
        and its current reading, and whether the episode started where the
        document said it would.
        **Inspector** -- everything about the one thing the page is about.
        **Scene** -- everything in it, as the way to make the page about
        something else.

        Then the controls that drive the page rather than describe the scene:
        the policy runner, the cameras, recording, export, and the transport,
        which stays last and always visible because it is how you stop.
        """
        env, config = self.env, self.config
        # A page built from a *selection* is a page showing one thing, so the
        # per-kind folds go with the rest of the studio's furniture; the whole
        # page keeps them, because reaching a body a click cannot and grabbing a
        # garment's declared strip by name are things only a fold can offer.
        self.chrome = not self._wanted_panels()
        self.index = SceneIndex(env.handles, env.scene, getattr(env, "doc", None))
        self.selected = self._opening_subject()
        self.arms = (
            {name: PanelArm(self, name) for name in (env.scene.embodiment.iks if env.scene.embodiment else {})}
            if self._wants("arm")
            else {}
        )
        # The arm the task is about, which the binding declares and the handles
        # carry. Falling back to the first arm in dict order is what this used to
        # do outright, and `SceneHandles.work_arm`'s own docstring says why that
        # is wrong: the two YAM cells order their arms oppositely, so insertion
        # order picks a different physical hand on different robots. Clicking an
        # object aims whichever arm is active, so on `yam_i2rt` the page was
        # reaching for the task's object with the hand not doing the task.
        self.active_arm = env.handles.work_arm or next(iter(self.arms), "")
        # The panels that need a *document* -- the task menu, the goal readout,
        # the policy runner -- are skipped rather than degraded when the page is
        # showing a scene somebody assembled by hand. Skipped, because there is
        # nothing for them to say: a bare scene has no variants to switch
        # between, no clauses to score and no shipped policy to run. The rest of
        # the page is identical, which is the claim the tier split makes.
        #
        # They are also the three that name the benchmark, so they are built by
        # the page's benchmark half and only asked for where there is a task --
        # which is what keeps this method importing nothing a bare scene lacks.
        self.task_panel = self.goal_panel = self.driver = None
        if hasattr(env, "task"):
            self.task_panel, self.goal_panel, self.driver = self._bench("a task page's panels").document_panels(self)
        # A take is a take on whatever the page is showing. A bare scene has
        # the synthetic episode the compiler gave it, which is all the recorder
        # needs; its take is a valid run directory that cannot be rebuilt from
        # a registry id (it has none), and ``run_meta.json`` says so.
        self.record = PanelRecord(self, spec=config.record)
        self.video = PanelVideo(
            self,
            fps=config.video_fps,
            every=config.capture_every,
            max_frames=config.buffer,
        )
        self.scene_picker = PanelScene(self)
        self.inspector = PanelInspector(self)
        # Menu, then what the task asks, then the one thing you are looking at.
        # The menu is first because changing the task changes everything under
        # it; the inspector follows the goal because between them they are the
        # whole of what the page is *saying*, and putting a dropdown over every
        # subject in the scene between the two split it.
        panels: list[Panel] = [p for p in (self.task_panel, self.goal_panel, self.inspector) if p is not None]
        # Then the cameras, because what they see is a property of the task
        # rather than of the run: the rig moves when the task does, and the first
        # thing worth knowing about a task you have just selected is whether its
        # cameras are pointed at it.
        # Resolved before the camera panel is built, because it decides whether
        # that panel's inset starts on: with the viewport itself path-traced, a
        # live inset is a second render of the same scene, and a measured
        # expensive one -- frustum image updates were 32 % of a tool-sorting scene's loop
        # period and running the observation rig for nobody 19 % of step rate.
        # The frustums stay: they say where the cameras are and are how you fly
        # to one, and a camera's image is one click away when you want it.
        viewport = PanelRenderView(
            self,
            world_model=config.world_model,
            scale=config.render_view_scale,
            passes=config.render_view_passes,
        )
        if self._wants("camera") and config.cameras and env.cameras:
            self.live = PanelCamera(
                self,
                live=config.live and viewport.mode != "rtx",
                fps=config.live_fps,
                max_load=config.live_load,
                width=config.live_width,
                height=config.live_height,
                channels=config.live_channels or ("rgb",),
                frustums=config.live_frustums,
            )
            panels.append(self.live)
        # The viewport goes *first*, above even the task menu. It was under the
        # cameras on the argument that a panel about how the page draws belongs
        # beneath the ones that say what it is pointed at, and that was wrong in
        # practice: the World model switch decides which of two quite different
        # pages you are looking at, and down there it sat below the inspector's
        # prose, off the bottom of a sidebar you had to scroll. A control nobody
        # can find is a control nobody has. It is still inert until a browser
        # connects: no client, no renderer, which is what keeps a headless
        # session as cheap as it was before this existed.
        self.render_view = viewport
        panels.insert(0, self.render_view)
        # And the scene list under those. It is how you reach a subject a click
        # cannot -- one behind the bench, a region with no surface to hit -- which
        # makes it a way *in* to the inspector rather than something to read, so
        # it sits below the readouts rather than in front of them.
        panels.append(self.scene_picker)
        panels += [p for p in (self.driver, *self.arms.values()) if p is not None]
        if self.arms:
            # Only on a scene with an arm to hand over -- a scene with no
            # embodiment has nothing for a leader arm to drive.
            self.leader_arm = PanelLeaderArm(
                self, ports=dict(config.leader_arm_ports), calibration_path=config.leader_arm_calibration
            )
            panels.append(self.leader_arm)
        # Chains count: a rope scene has no plain objects at all, and gating on
        # the handles table alone is what left its one movable thing unclickable.
        # Arms count too, now that a robot's own links are grabbable: a scene
        # with a robot and nothing loose on it still has something to take hold
        # of, and gating this on objects alone left the one thing on the page
        # unclickable -- which is the bug this panel was extended to fix.
        if self._wants("object") and (env.handles.objects or self.arms):
            self.objects = PanelObject(self, handles=config.handles, chrome=self.chrome)
            panels.append(self.objects)
        if self._wants("cloth") and any(e.carries_particles for e in env.scene.by_name.values()):
            self.deformables = PanelCloth(self, handles=config.handles, chrome=self.chrome)
            panels.append(self.deformables)
            # Tuning lives beside the task rather than on a page of its own: a
            # bare strip gives a bending length, but only the task says whether
            # the fold holds or the jaws keep hold of it.
            #
            # The *material* folder stays for what the inspector's generated one
            # cannot do -- presets, save, revert, and `Apply & rebuild` for the
            # fields no live write can reach. Per-field tuning of the selected
            # cloth is the inspector's now, from the same binder.
            self.tune = PanelTune(self, material_dir=config.material_dir)
            self.readings = PanelReadings(self)
            panels += [self.tune, self.readings]
        # The overlay draws what the inspector's checkboxes ask for, and joins
        # the region boxes rather than replacing them: `PanelRegions` already
        # draws every entry in `handles.regions`, which includes the ones an
        # asset declares, and tints them by their goal clause.
        self.overlay = PanelOverlay(self)
        panels += [PanelRegions(self), self.overlay]
        if self.record is not None:
            panels.append(self.record)
        panels.append(self.video)
        self.panels = self._selected(panels)

    # ---------------------------------------------------------------- handles
    def held(self) -> list[str]:
        """What is on a handle right now, across every kind. Main thread."""
        labels: list[str] = []
        for panel in (self.objects, self.deformables):
            if panel is None or panel.handles is None:
                continue
            labels += [str(handle.label or "?") for handle in panel.handles.live]
        return labels

    def missed(self) -> str:
        """How far the last cloth grab was off, or ``""``.

        A click three centimetres wide of a sheet takes hold of nothing, and
        without a word about it the page looks broken rather than missed.
        """
        miss = getattr(self.deformables, "miss", 0.0) if self.deformables is not None else 0.0
        return f"missed by {miss * 1000:.0f} mm -- click on the cloth" if miss else ""

    def release_last(self) -> None:
        """Let go of the newest handle, whichever kind it is on. Main thread."""
        newest = None
        for panel in (self.objects, self.deformables):
            if panel is None or panel.handles is None:
                continue
            slot = panel.handles.newest()
            if slot is not None and (newest is None or slot.seq > newest[1].seq):
                newest = (panel, slot)
        if newest is not None:
            newest[0]._release(newest[1])

    def release_all(self) -> None:
        """Let go of everything the page is holding. Main thread."""
        for panel in (self.objects, self.deformables):
            if panel is None or panel.handles is None:
                continue
            for handle in list(panel.handles.live):
                panel._release(handle)

    def set_linked(self, linked: bool) -> None:
        """``Move together`` across every kind's handles. Main thread."""
        for panel in (self.objects, self.deformables):
            if panel is not None and panel.handles is not None:
                panel.handles.linked = bool(linked)

    def refresh_holding(self) -> None:
        """Say what is held, and show the controls only once there is something to let go of."""
        if self._holding_readout is None:
            return
        labels, missed = self.held(), self.missed()
        self._holding_readout.value = ", ".join(labels) if labels else (missed or "nothing")
        # The readout stays up on a miss -- that is when it has something to say
        # -- while the controls that act on a handle wait for one to exist.
        self._holding_readout.visible = bool(labels or missed)
        for widget in (self._linked_toggle, self._release_last, self._release_all):
            if widget is not None:
                widget.visible = bool(labels)

    def reset_env(self):
        """Reset whatever the page is showing, and return ``(obs, info)``.

        One method because there are two kinds of environment under this page
        and they disagree about this one call: ``Env.reset`` takes an episode
        index, ``EnvBase.reset`` does not, because a scene somebody assembled by
        hand has nothing to index. Every example is a task-less scene.
        """
        env = self.env
        return env.reset(self.episode_id) if hasattr(env, "task") else env.reset()

    def _wants(self, key: str) -> bool:
        """Whether the selection keeps this panel, asked *before* it is built.

        :meth:`_selected` filters the finished list, which is enough for a panel
        that is only a folder. It is not enough for the ones whose constructor
        acts on the scene: :class:`~robosimstudio_studio.panels.PanelObject` and
        ``PanelCloth`` each contribute a dragger per handle to the substep hooks
        and then invalidate the captured graph, so building one only to drop it
        would leave the hooks behind and pay for a recapture nothing uses.
        """
        wanted = self._wanted_panels()
        return not wanted or key in wanted

    def _wanted_panels(self) -> tuple[str, ...]:
        """The selection as keys. Empty means everything.

        Asked of the environment being built, not once per session: a page can
        load a task over a bare scene and back, and :data:`DOCUMENT_PANELS` go
        with the document.
        """
        asked = self.config.panels
        if not isinstance(asked, str):
            return tuple(asked)
        wanted = PANEL_SETS.get(asked)
        if wanted is None:
            raise ValueError(
                f"unknown panel set {asked!r}; the names are {sorted(PANEL_SETS)}, or pass the panel keys themselves"
            )
        if wanted and hasattr(getattr(self, "env", None), "task"):
            return (*wanted, *DOCUMENT_PANELS)
        return wanted

    def _selected(self, panels: list[Panel]) -> list[Panel]:
        """Keep the panels :attr:`StudioConfig.panels` asked for. Everything above
        this line is unchanged by the selection: *which panels a scene warrants*
        is a question about the scene -- a rope page has no cloth folder however
        it is configured -- and this only answers *which of those to show*.

        A panel that is dropped is also forgotten. The session holds each one by
        name as well as in the list (``self.objects``, ``self.deformables``, ...) and
        those references are what ``drain_click`` and the reset path reach for;
        left set, they would point at a panel whose ``build`` never ran, so its
        handles are ``None`` and a click on an object would raise rather than do
        nothing. The existing ``is not None`` guards then do the right thing.
        """
        wanted = self._wanted_panels()
        if not wanted:
            return panels
        keep = [p for p in panels if panel_key(p) in wanted]
        dropped = {id(p) for p in panels if panel_key(p) not in wanted}
        for name, value in list(self.__dict__.items()):
            if id(value) in dropped:
                setattr(self, name, None)
        self.arms = {name: panel for name, panel in self.arms.items() if id(panel) not in dropped}
        return keep

    def _build_appearance(self, gui) -> None:
        """The ``Appearance`` folder. See :meth:`build_gui` for why it is optional."""
        with gui.add_folder("Appearance", expand_by_default=False):
            scenery = gui.add_checkbox("Show scenery", initial_value=self.show_scenery)

            def _scenery(_event=None) -> None:
                self.show_scenery = bool(scenery.value)
                self.scenery_request = True

            scenery.on_update(_scenery)
            if any(e.carries_particles for e in self.env.scene.by_name.values()):
                look = gui.add_dropdown("Cloth look", options=GARMENT_STYLES, initial_value=self.viewer.garment_style)
                look.on_update(lambda _: setattr(self, "cloth_look_request", str(look.value)))
            # Built here rather than in ``panels``: it writes only to the viser
            # scene graph, so it neither needs a ``drain`` nor cares that the
            # environment underneath it may be swapped for another task.
            self.lighting.build(gui)

    def build_gui(self) -> None:
        gui = self.server.gui
        gui.configure_theme(control_layout="collapsible", dark_mode=self.config.dark_mode)
        # First in the sidebar, outside any fold, and **hidden until something is
        # held**. Taking hold of a thing is one click on the thing itself, so
        # letting go should not be a button two folds deep in a panel named
        # after a *kind* -- and a handle you cannot see how to remove is a
        # handle that reads as stuck. Clicking the thing again also drops it;
        # this is the version you can find without being told.
        #
        # Built once, here, rather than added when a handle appears: viser sorts
        # a sidebar by a counter that increments per widget, so anything added
        # later sorts after everything already on the page -- the same trap the
        # inspector's anchor exists for. Visibility is what changes.
        #
        # One set for every kind, which is why they are the session's.
        self._holding_readout = gui.add_text("Holding", initial_value="nothing", disabled=True)
        self._linked_toggle = gui.add_checkbox("Move together", initial_value=False)
        self._linked_toggle.on_update(lambda _: setattr(self, "linked_request", bool(self._linked_toggle.value)))
        self._release_last = gui.add_button("Release last")
        self._release_last.on_click(lambda _: setattr(self, "release_last_requested", True))
        self._release_all = gui.add_button("Release all")
        self._release_all.on_click(lambda _: setattr(self, "release_all_requested", True))
        self.refresh_holding()
        # No separate Interact folder: what you clicked belongs beside the arm
        # that is going to act on it, and the arm panels already sit under the
        # task and policy controls. The only thing left that is genuinely
        # session-wide is *which* arm, and on a one-armed scene there is no
        # choice to offer -- so it is built immediately above the arm folders
        # rather than in a section of its own.
        first_arm = next(iter(self.arms.values()), None)
        for panel in self.panels:
            if panel is first_arm and len(self.arms) > 1:
                arm = gui.add_dropdown("Arm", options=tuple(self.arms), initial_value=self.active_arm)
                # Bound as a default rather than closed over: the handle is
                # assigned inside the loop, so a late-firing callback would read
                # whatever the *last* iteration left in `arm`. Only one panel can
                # be `first_arm`, so today it is the same object either way --
                # which is exactly the kind of accident that stops being true
                # when a second dropdown is added and fails in the browser, not
                # in a test.
                arm.on_update(lambda _, arm=arm: setattr(self, "active_arm", str(arm.value)))
            panel.build(gui)
        # One folder, collapsed, and nothing folded inside it. What the scene
        # *looks* like is a handful of decisions made once a session -- is the
        # bench drawn, how is the cloth shaded, where is the key light -- and a
        # `Light` fold within this one made the sliders a second click deep for
        # no gain: opening `Appearance` at all is already saying you want them.
        #
        # Selectable like a panel even though it is built inline, because "made
        # once a session" is exactly what a page showing one rope has no use
        # for. ``drive`` leaves it out.
        if self._wants("appearance"):
            self._build_appearance(gui)
        with gui.add_folder("Simulation"):
            pause = self.pause_box = gui.add_checkbox("Pause", initial_value=self.paused)
            pause.on_update(lambda _: setattr(self, "paused", pause.value))
            # Not config.speed directly: an unrecognised key is an initial_value
            # outside the options, which viser refuses, and a page that will not
            # build is a poor answer to a typo in a config field.
            initial = self.config.speed if self.config.speed in SPEEDS else "1x"
            speed = gui.add_dropdown("Speed", options=tuple(SPEEDS), initial_value=initial)
            speed.on_update(lambda _: setattr(self, "speed", SPEEDS[str(speed.value)]))
            reset = gui.add_button("Reset")
            # The one reset path: the loop services it through reset_now().
            reset.on_click(lambda _: setattr(self, "reset_requested", True))
            quit_ = gui.add_button("Stop server")
            quit_.on_click(lambda _: setattr(self, "quit_requested", True))
            self._rate_readout = gui.add_text("Rate", initial_value="", disabled=True)
            self._status_readout = gui.add_text("Status", initial_value="", disabled=True)
        # The instruction is not repeated here. It is one sentence of English
        # saying what the goal tree says in the terms the run is scored on, so it
        # lives in that section's Detail fold -- and printing it again at the
        # foot of the sidebar, under a second copy of the task's name, was the
        # page saying one thing three times over. The eight paragraphs explaining
        # the interaction stay, folded:
        # you read them once and look past them every session after, which is
        # what a fold is for.
        with gui.add_folder("How to drive this", expand_by_default=False):
            gui.add_markdown(
                "- **Click** anything -- an object, a patch of cloth -- and it comes up on a handle "
                "you can **drag straight away**. The bead at the handle's origin says which slot it "
                f"is; there are {self.config.handles} of them per kind.\n"
                "- **Click again elsewhere** for a second handle. **Move together** then carries the "
                "pair as one rigid group from a drag on either, which is how a sheet is held by two "
                "corners with one mouse. **Click the thing again to put it down**; **Release "
                "last** and **Release all** appear at the top of this sidebar as soon as you are "
                "holding anything, beside a line saying what.\n"
                "- A handle is a wrench on a body or a pin on a patch -- **not a grasp**. The thing "
                "stays dynamic underneath, so dragging it into the jaws is a fair way to set a grasp "
                "up, and tugging on one the robot holds is how you find out whether the grip is real.\n"
                "- **Drag an arm's gizmo** to drive that arm; **Grip** opens and closes its "
                "fingers in one click, and **Grip amount** is the same travel by hand. Whether that "
                "holds is the physics' answer, not the button's.\n"
                "- **Camera gizmo** draws each camera where it stands, showing its own view. Click "
                "one to fly there; **Set from view** then points it wherever you have flown.\n"
                "- **Run** hands the arm to one of the task's policies until it finishes.\n"
                "- **Start take** keeps what you drive as an episode: one numbered folder per take "
                "(`steps.npz`, `meta.json`, `cameras/*.mp4`) beside `episode_results.jsonl` and "
                f"`run_meta.json`, under `{self.config.out}/<scene>/<time>/demos/`.\n"
                "- **Export video** renders the captured state offline, and never re-simulates.\n"
            )

    def reset_now(self) -> bool:
        """Put the scene back to its start. Main thread; True when it reset.

        Where every Reset lands -- the Simulation folder's button, the episode
        panel, and anything a subclass adds -- by way of :attr:`reset_requested`,
        which is all a viser callback may do. A subclass with its own idea of
        what a reset ends (a script playing, a take being filmed) overrides this
        rather than adding a second button that does something different.
        """
        model = getattr(self.env, "model", None)
        try:
            # An episode whose model key differs recompiles the scene in here,
            # so this is the second place a menu can ask for something that
            # will not build.
            obs, info = self.reset_env()
        except Exception as exc:
            # Unlike a task load, this cannot be made atomic: a rebuild that
            # fails half way leaves the environment in a state this page did
            # not build and cannot repair. So say so and stop stepping it,
            # rather than run a scene nobody can vouch for -- through
            # set_paused, so the checkbox says what the loop does.
            self.set_paused(True)
            self.status = f"episode {self.episode_id} did not load ({type(exc).__name__}: {exc}) -- paused"
            self._show_status()
            print(f"[studio] episode {self.episode_id} did not load; paused:")
            traceback.print_exc()
            return False
        if getattr(self.env, "model", None) is not model:
            # The recompile swapped the model, scene and pipeline under the
            # page: the viewer, the picker, the index and every dragger hooked
            # onto the old pipeline are about a scene that is gone. Rebuilt
            # here, the same way a load builds them, minus closing the env.
            self._drop_panels()
            self._bind(self.env)
        self.obs = obs
        self.on_reset(obs, info)
        self.status = "reset"
        return True

    def steps_held(self) -> bool:
        """Whether something other than this loop is stepping the scene right now.

        While it is, the loop treats the page as paused whatever the checkbox
        says -- stepping beside it would take two steps per pass -- but still
        films what it steps (:meth:`_capture_outside_steps`). Nothing does on a
        plain page; ``robosimstudio_real2sim_agent`` says so while a routine plays.
        """
        return False

    def _show_status(self) -> None:
        """Put :attr:`status` on the page now, rather than at the next rate readout."""
        readout = getattr(self, "_status_readout", None)
        if readout is not None and readout.value != self.status:
            readout.value = self.status

    def _capture_outside_steps(self) -> None:
        """Film the steps something else took while the page held. Main thread.

        A script playing on a held page steps the scene itself, so the loop's
        own capture -- which follows the loop's own step -- would leave a take
        or a video with nothing in it. One capture per pass in which the step
        count moved, of the state as it now stands.
        """
        step = int(self.env.step_count)
        if step == self._captured_step:
            return
        self._captured_step = step
        if self.video is not None:
            self.video.capture()
        if self.record is not None:
            self.record.capture(None, "")

    def on_reset(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        # A reset moves the scene without advancing the step count -- it puts it
        # back to 0 -- so the "has it stepped far enough" test would hold the
        # viewer on the previous episode's picture.
        self._drawn_step = None
        self._captured_step = int(self.env.step_count)
        for panel in self.panels:
            panel.on_reset(obs, info)

    # ------------------------------------------------------------------- clicks
    def _opening_subject(self) -> Subject | None:
        """What the page is about before anybody has clicked anything.

        **The first thing the goal is about**, not the first thing in the scene.
        Those differ and the difference is the whole answer: a tool-sorting scene's
        subjects sort with ``bin1`` at the front, so a page opening on its own
        first entry opens on a fixture -- and a fixture is precisely the thing in
        that scene nobody came to look at. Its goal's first operand is
        ``hammer_0``, which is what the task is about and what a scripted policy
        reaches for first.

        **Nothing at all when there is no goal.** A scene somebody assembled by
        hand has no "first thing the goal is about", and the fallback -- the
        first subject in scene order -- was arbitrary: it opened the page on a
        thing nobody had pointed at and gave the inspector a heading to fill
        before anyone had asked it anything. On such a page the inspector
        appears when you click something, which is when there is an answer.
        """
        index = self.index
        if index is None:
            return None
        goal = getattr(self.env, "goal", None)
        if goal is None:
            return None
        # Walking a goal tree is the benchmark half's, and only an environment
        # that half built has one.
        return self._bench("reading a goal").opening_subject(index, goal)

    def select(self, subject: Subject | None) -> None:
        """Point the page at something. Main thread.

        One place, and every panel is told.
        """
        if subject is not None and self.selected is not None and subject.key == self.selected.key:
            return
        self.selected = subject
        for panel in self.panels:
            panel.on_select(subject)

    def drain_click(self) -> None:
        """Turn the last click into a selection, and into a handle on it. Main thread.

        Whichever of the two candidates the ray met *first* wins, so clicking a
        cloth draped over a box takes the cloth and clicking the box beside it
        takes the box, with no mode to switch between them.

        **The shape index is used.**
        :class:`~robosimstudio_studio.selection.SceneIndex` maps a shape to
        whatever owns it, so everything in the scene is selectable and grabbing
        is what happens *as well*, for the things that can be grabbed.

        The panels each keep a ``Grab on click`` checkbox for a session that
        wants naming without holding, and each services the grab in its own
        ``drain`` -- on this same control step, but through the one code path
        that is allowed to call Warp.
        """
        if self.select_request is not None:
            key, self.select_request = self.select_request, None
            if self.index is not None:
                self.select(self.index.by_key(key))

        ray, self.click_ray = self.click_ray, None
        if ray is None or self.picker is None:
            return
        origin, direction = ray
        state = self.env.state
        shape = self.picker.shape_at(state, origin, direction)
        particle = self.picker.particle_at(state, origin, direction) if self.deformables is not None else None

        if particle is not None and (shape is None or particle.distance < shape.distance):
            name = self.deformables.owner_of(particle.index)
            if name:
                self._select_hit(self.index.at_particle(particle.index) if self.index else None)
                self.report_click(self.deformables.click_at(name, particle.point))
                return
        if shape is not None:
            hit = self.index.at_shape(shape.shape) or self.index.at_body(shape.body) if self.index else None
            self._select_hit(hit)
            report = self.objects.click_body(int(shape.body)) if self.objects is not None else ""
            self.report_click(report or (hit.label if hit else self._body_name(int(shape.body))))
            return
        self.report_click("nothing there")

    def _select_hit(self, subject: Subject | None) -> None:
        """Adopt what a click found, and keep the picker dropdown in step."""
        if subject is None:
            return
        self.select(subject)
        picker = getattr(self, "scene_picker", None)
        if picker is not None:
            picker.follow(subject)

    def _body_name(self, body: int) -> str:
        """A readable name for a body the handles do not have one for."""
        labels = list(getattr(self.env.model, "body_label", None) or ())
        if 0 <= body < len(labels) and labels[body]:
            return str(labels[body]).rsplit("/", 1)[-1]
        return f"body {body}"

    # -------------------------------------------------------------------- view
    def _scene_spec(self):
        """The registered place this task is staged in, or ``None``."""
        request = getattr(self.env, "request", None) if self.env is not None else None
        # ``scene`` on a TaskSpec, ``cell`` on a BuildRequest: the same fact
        # under the two names the rename left, and a page shows either.
        name = getattr(request, "scene", "") or getattr(request, "cell", "")
        return get_cell(name) if name else None

    def _scenery_shapes(self) -> np.ndarray:
        """Every shape belonging to the scene's own furniture.

        The *scene's* statics and nothing else. A task's own statics -- the
        panel a film is stuck to, the box a tape is wound onto -- are the task,
        not the room, and hiding them turns a peel into a sheet levitating over
        nothing; they stay drawn whatever the scenery toggle says. The ground
        plane is a floor rather than scenery too -- it occludes nothing and
        taking it away leaves things hanging in a void.
        """
        spec = self._scene_spec()
        furniture = {s.name for s in spec.statics} if spec is not None else set()
        ranges = [entity.range.shape_indices for name, entity in self.env.scene.statics.items() if name in furniture]
        ranges = [r for r in ranges if len(r)]
        return np.concatenate(ranges) if ranges else np.zeros(0, dtype=np.int32)

    def _set_scenery_visible(self, visible: bool) -> None:
        """Show or hide the furniture. Display only -- collision is untouched."""
        import newton

        shapes = self._scenery_shapes()
        if not len(shapes):
            return
        flags = np.asarray(self.env.model.shape_flags.numpy()).copy()
        bit = int(newton.ShapeFlags.VISIBLE)
        if visible:
            flags[shapes] |= bit
        else:
            flags[shapes] &= ~bit
        self.env.model.shape_flags.assign(flags)

    def _adopt_look_lighting(self) -> None:
        """Point the rig at the scene's own look, unless somebody set one.

        A scene states its lighting in a backend-neutral ``RenderSpec``, and
        until now the page ignored it and used its own tuned defaults. It still
        does when the caller stated a rig -- a ``--key``-style override is an
        instruction, not a suggestion -- which is what the comparison against a
        default ``LightingRig`` is asking: *did anybody touch this*.

        Only the fields the look states move; see
        :meth:`LightingRig.from_look` for why the rest stay where the page was
        tuned, and why the environment map is not one of them.
        """
        if self.config.lighting != LightingRig():
            return
        look = getattr(self.env, "look", None)
        if look is None:
            return
        self.lighting.rig = LightingRig.from_look(look)

    def _apply_backdrop(self) -> None:
        """Draw the room the scene's look names, through the page's own kit.

        The same registered builder the path tracer runs -- ``house_room_closed``,
        ``lab``, ``rgbs_lab``, ``tool_sort_shop`` -- so the page and the render
        are dressed by one description rather than two. Re-run whenever the
        scene changes, because which room a scene stands in is the scene's to
        say.
        """
        if self._backdrop is not None:
            self._backdrop.remove()
            self._backdrop = None
        if not self.config.backdrop:
            return
        look = getattr(self.env, "look", None)
        name = getattr(look, "scenery", "") or ""
        if not name:
            return
        spec = self._scene_spec()
        self._backdrop = draw_scenery(
            self.server.scene,
            name,
            cell=spec,
            surface_z=float(getattr(spec, "surface_z", 0.0) or 0.0),
        )

    def _apply_stage(self) -> None:
        """Put a plain slab where the hidden furniture's work surface was.

        The scenery default hides the bench so it cannot swallow the gizmos --
        but a towel folded on a hidden bench floats 0.7 m over the floor grid,
        which is a stranger picture than the one the hiding fixed. The stage is
        the seamless-paper answer: a thin light slab at the work surface, in
        the viser scene only, so the cloth rests on *something*, the key
        light's shadows land where contact happens, and nothing about the
        model, the physics or an export changes.

        Drawn only when there is furniture being stood in for: a scene with no
        statics (the robosuite arena, a task with no scene) keeps whatever it
        already shows, and showing the real scenery removes the stand-in.
        """
        if self._stage is not None:
            try:
                self._stage.remove()
            except Exception:
                pass
            self._stage = None
        spec = self._scene_spec()
        if self.show_scenery or spec is None or not spec.statics or not spec.surface_z > spec.ground_z:
            return
        half = spec.half if spec.half is not None else (0.65, 0.65, 0.0)
        self._stage = self.server.scene.add_box(
            "/studio/stage",
            color=STAGE_COLOR,
            dimensions=(
                2.0 * float(half[0]) * STAGE_MARGIN,
                2.0 * float(half[1]) * STAGE_MARGIN,
                STAGE_THICKNESS,
            ),
            position=(
                float(spec.center[0]),
                float(spec.center[1]),
                float(spec.surface_z) - STAGE_CLEARANCE - STAGE_THICKNESS / 2.0,
            ),
            cast_shadow=False,
            receive_shadow=True,
        )

    def apply_scenery(self) -> None:
        """Re-batch the viewer at the current setting. Main thread.

        A re-``set_model`` rather than a flag write, because the viewer decides
        visibility from the flags it captured when the model was handed over.
        """
        self._set_scenery_visible(self.show_scenery)
        # The picker caches the visible shapes on its first query; the flags
        # just changed, so what it would hit must follow what is drawn.
        if self.picker is not None:
            self.picker.invalidate()
        self.viewer.set_model(self.env.model)
        self.viewer.set_garments(garment_visuals(self.env.scene, getattr(self.env, "doc", None)))
        self.viewer.set_live_colors(scene_repaints_live(self.env.scene))
        self._push_finishes(self.env)
        self._adopt_look_lighting()
        self.lighting.apply()
        self._apply_backdrop()
        self._apply_stage()
        self._drawn_step = None

    def report_click(self, text: str) -> None:
        """Show what the last click found, on the arm that would act on it."""
        arm = self.arms.get(self.active_arm)
        if arm is not None:
            arm.report_click(text)
        else:
            self.status = text

    def set_paused(self, paused: bool) -> None:
        """Pause or resume from code, keeping the page's checkbox truthful. Main thread.

        The checkbox writes :attr:`paused` on a click; this is the other
        direction, for a driver outside the page -- ``robosimstudio_real2sim_agent``'s
        ``run N`` and ``pause`` -- so what the page shows is what the loop does.
        """
        self.paused = bool(paused)
        if self.pause_box is not None and self.pause_box.value != self.paused:
            self.pause_box.value = self.paused

    # --------------------------------------------------------------------- loop
    def begin_run(self) -> None:
        """Initialise the pacing and rate-readout state one loop iteration reads.

        Split out of :meth:`run` so that state lives on the session rather than
        in a closure over the ``while``: :meth:`tick` is the same iteration
        driven from outside, and it cannot be if the clock it paces against is a
        local variable. Public because a page driven from outside :meth:`run`
        (``robosimstudio_real2sim_agent``) must reset the clock after every :meth:`adopt`.
        """
        control_hz = (
            self.fleet.fleet.pipeline.control_hz if self.fleet is not None else self.env.request.physics.control_hz
        )
        self._dt = 1.0 / control_hz
        self._next_frame = time.perf_counter()
        self._rate_frames = 0
        self._rate_window = time.perf_counter()

    @property
    def scene(self) -> Any:
        """``server.scene`` -- add gizmos, meshes and frames here.

        Named to match :class:`robosimstudio.Viewer`, so a helper written
        against one page works on the other. It is what lets a viser component
        from anywhere -- pyroki's ``WeightTuner(server, ...)`` shape -- be
        dropped into a studio page through :meth:`each_step` and ``page=``
        without this module knowing anything about it.
        """
        return self.server.scene

    @property
    def gui(self) -> Any:
        """``server.gui`` -- add sliders, numbers and buttons here."""
        return self.server.gui

    def each_step(self, fn) -> None:
        """Call ``fn()`` once per control step, on the main thread.

        The seam a script needs to add a readout of its own: a viser callback
        may only assign, because it runs on the server's threadpool and touching
        Warp from there is undefined -- the contract every panel obeys (see
        :mod:`robosimstudio_studio.panels`). This runs where ``drain`` runs, so
        it may read state and write widgets.

        Drained in :meth:`tick` beside the panels, which means it does not run
        on a step the page did not take: paused is paused.
        """
        self._each_step.append(fn)

    def add_routines(self, routines) -> None:
        """A **Run** button per routine, a Stop, and what is playing: see :mod:`robosimstudio_studio.routine`.

        ``routines`` is one callable or a dict of name to callable. Each takes the
        scene (and, to drive the arm panels as a person does, the page as
        ``session``) and returns a generator that yields once per control step.
        """
        from .routine import PanelRoutine

        if callable(routines):
            routines = {"Run": routines}
        panel = PanelRoutine(self, routines)
        panel.build(self.server.gui)
        self.panels.append(panel)

    def run(self, *, steps: int = 0, warmup: int = 0) -> None:
        """Serve until the browser goes away or the page says stop.

        Args:
            steps: Stop after this many *stepped* iterations, instead of serving
                until asked to stop. Zero is a normal session. A non-zero value
                is what makes the page measurable without a person in front of
                it -- a benchmark drives it, counts, and gets its numbers back --
                and it counts steps rather than iterations so a paused or
                request-servicing pass does not spend the budget.
            warmup: Steps to run before the profiler's samples start counting.
                Discarded with :meth:`Profiler.reset`, which keeps the phase
                names and drops what they measured.

        Zero for either reads :attr:`StudioConfig.profile_steps` and
        :attr:`StudioConfig.profile_warmup`; a profiled run ends by printing
        the phase table and writing it to :attr:`StudioConfig.profile_out`.
        """
        steps = int(steps or self.config.profile_steps)
        warmup = int(warmup or self.config.profile_warmup)
        self.begin_run()
        stepped = 0
        self.profile.start()
        self._note_scene()
        # Timed here as well as in the profiler, so the same window can be
        # measured with the profiler off -- which is the only way to know what
        # the profiler itself costs.
        measured_from = time.perf_counter()
        try:
            while self.viewer.is_running() and not self.quit_requested:
                if not self.tick():
                    # A finished wave never steps again, so a budgeted run has
                    # to end on that rather than on a step count it can no
                    # longer reach. A page left open idles on the result, which
                    # is what somebody watching wants.
                    if steps and self.fleet is not None and self.fleet.done:
                        return
                    continue
                stepped += 1
                if warmup and stepped == warmup:
                    self.profile.reset()
                    measured_from = time.perf_counter()
                if steps and stepped >= warmup + steps:
                    return
        finally:
            self.measured_steps = stepped - min(stepped, warmup)
            self.measured_wall = time.perf_counter() - measured_from
            self._note_capture()
            if self.profile.device is not None:
                self.profile.device.drain()
            self.profile.stop()
            self._write_profile()
            self.close()

    def _write_profile(self) -> None:
        """Print the phase table and write its JSON sidecar, when anything was measured."""
        prof = self.profile
        if not prof.enabled:
            return
        path = Path(self.config.profile_out) if self.config.profile_out else None
        if path is None:
            path = self.root / "profile" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        prof.write(str(path))
        print(prof.report())
        print(f"[studio] profile written to {path}")

    def tick(self) -> bool:
        """One iteration of the loop. True when the scene actually stepped.

        Everything :meth:`run` used to do inside its ``while``. The return value
        is what lets a caller drive the page a fixed number of *steps* rather
        than a fixed number of passes, since servicing a load request or sitting
        paused returns without advancing anything.
        """
        prof = self.profile
        with prof.phase("iter"):
            with prof.phase("iter.requests"):
                if self.load_request is not None:
                    task, episode, scene, embodiment, gripper, worlds = self.load_request
                    self.load_request = None
                    self.load_task(task, episode, scene, embodiment, gripper, num_worlds=worlds)
                    control_hz = (
                        self.fleet.fleet.pipeline.control_hz
                        if self.fleet is not None
                        else self.env.request.physics.control_hz
                    )
                    self._dt = 1.0 / control_hz
                    self._next_frame = time.perf_counter()
                    return False
            if self.fleet is not None:
                # The task menu is drained first and every time: it is what sets
                # the world count, so a page that skipped it could be put into a
                # grid and never taken out of one.
                with prof.phase("iter.panels"):
                    for panel in self.panels:
                        panel.drain()
                if self.load_request is not None:
                    return False
                # A grid has its own loop -- worlds stepped together, scored on
                # a cadence, drawn on the clock. Everything below this line is
                # about driving *a* scene and has nothing to say about one.
                return self.fleet.tick()

            if self.reset_requested:
                self.reset_requested = False
                if not self.reset_now():
                    return False

            if self.scenery_request:
                self.scenery_request = False
                self.apply_scenery()

            if self.cloth_look_request is not None:
                look, self.cloth_look_request = self.cloth_look_request, None
                self.viewer.set_garment_style(look)
                self._drawn_step = None

            # ``None`` on a page showing a scene with no task: there is no
            # shipped policy to run, so nothing is driving and the loop is
            # the teleop one.
            driven = self.driver is not None and self.driver.running
            if self.linked_request is not None:
                self.set_linked(self.linked_request)
                self.linked_request = None
            if self.release_last_requested:
                self.release_last_requested = False
                self.release_last()
            if self.release_all_requested:
                self.release_all_requested = False
                self.release_all()

            if self.paused or self.steps_held():
                # The interaction panels are drained while paused: lining a
                # grasp up on a held scene is exactly when you want to, and
                # aiming costs an IK solve rather than a step. The others
                # wait -- the episode panel's drain ticks a stateful
                # evaluator, and starting a policy on a scene that is not
                # advancing would leave it hanging on its first waypoint.
                if not driven:
                    self.drain_click()
                    for panel in self.panels:
                        # PanelRegions joins them: it draws and re-poses, and
                        # touches no evaluator -- a held scene is exactly when
                        # somebody is looking at where a region actually is.
                        # PanelScene and PanelInspector for the stronger version
                        # of the same argument: pausing to look at a thing is the
                        # main reason to pause, and an inspector that stopped
                        # following the selection while held would make the pause
                        # useless. Neither scores anything. PanelRenderView and
                        # PanelCamera neither: the path-traced view has to keep
                        # following the orbit, and Look through / Set from view
                        # are camera gestures; the camera panel skips its render
                        # when the step has not changed, so held costs little.
                        if isinstance(
                            panel,
                            (
                                PanelArm,
                                PanelObject,
                                PanelCloth,
                                PanelRegions,
                                PanelScene,
                                PanelInspector,
                                PanelOverlay,
                                PanelRenderView,
                                PanelCamera,
                            ),
                        ):
                            panel.drain()
                # Set here too: a script stepping a held page renders the rig
                # for a take the same as the loop's own step would.
                self.env.obs_render_enabled = self.wants_pixels
                self._capture_outside_steps()
                self._show_status()
                self.refresh_view()
                time.sleep(0.02)
                self._next_frame = time.perf_counter()
                return False

            with prof.phase("iter.drain"):
                if not driven:
                    with prof.phase("iter.drain.click"):
                        self.drain_click()
                panels = self.panels
                for panel in panels:
                    if driven and (isinstance(panel, CONTROL_PANELS) or panel is self.goal_panel):
                        continue
                    with prof.phase(f"iter.drain.{panel.profile_name}"):
                        panel.drain()
                    if self.panels is not panels:
                        # A drain rebuilt the page (a Run whose reset
                        # recompiled the scene): the rest of this list is
                        # closed panels over a model that is gone.
                        break
                # A script's own per-step work, last: it reads what the panels
                # have just done rather than racing them, and it is drained even
                # while a policy drives, because a readout of a run is exactly
                # what somebody would add one for.
                if self._each_step:
                    with prof.phase("iter.drain.each_step"):
                        for fn in self._each_step:
                            fn()
            if self.load_request is not None:
                return False  # a load was asked for mid-drain; do not step a dying scene

            # Set after the drain, so a take opened this iteration has its
            # cameras rendered on the very first step it records rather than
            # on the one after.
            self.env.obs_render_enabled = self.wants_pixels

            self._step_and_capture(prof)
            self._rate_frames += 1
            prof.mark_step()

            self.refresh_view()

            now = time.perf_counter()
            if now - self._rate_window >= 0.5:
                # Both rates, because they are now independent: the page
                # holding 12 fps over a scene stepping at 30 is the view
                # doing its job, not the loop stalling, and a session that
                # cannot tell those apart will go looking for the wrong bug.
                rate = self._rate_frames / (now - self._rate_window)
                self._rate_readout.value = f"{rate:.0f} steps/s  |  {self.view_fps:.0f} fps view"
                self._status_readout.value = self.status
                self._rate_frames, self._rate_window = 0, now

            with prof.phase("iter.pace"):
                if self.speed > 0.0:
                    self._next_frame += self._dt / self.speed
                    # Drop the debt rather than catching up: a scene already too
                    # slow gets slower, and a stutter becomes a stall.
                    sleep = self._next_frame - time.perf_counter()
                    if sleep > 0:
                        time.sleep(sleep)
                    else:
                        self._next_frame = time.perf_counter()
        return True

    def _step_and_capture(self, prof) -> None:
        """Take the loop's one step, then film and record it -- if it was taken.

        A policy that raises hands the arm back without stepping anything, and a
        capture after it would put a step that never happened into the take: the
        state unchanged, the action read back off the arm as if teleop drove it,
        in a take whose mode is already the policy's. So the step count, not the
        call, says whether there is anything to capture.
        """
        before = int(self.env.step_count)
        with prof.phase("iter.step"):
            if self.driver is not None and self.driver.running:
                action, mode = self.driver.step()
            else:
                with self._device_span():
                    self.env.step_teleop()
                action, mode = None, ""
        if int(self.env.step_count) != before:
            with prof.phase("iter.video"):
                # ``None`` when the selection left the video panel out, the same
                # way the recorder below is ``None`` without a document.
                if self.video is not None:
                    self.video.capture()
            # After the step, and with whatever drove it: a take is evidence
            # about a trajectory, so it records the action that produced the
            # state it is looking at.
            with prof.phase("iter.record"):
                if self.record is not None:
                    self.record.capture(action, mode)
        self._captured_step = int(self.env.step_count)

    def _note_scene(self) -> None:
        """Record what is being measured, beside the measurement.

        A phase table without this is unreadable a week later: ``draw`` costing
        10 ms means one thing over eight shape batches and quite another over
        eighty, and a per-batch cost cannot be inferred from a total whose
        divisor was not written down. The capture counters are here for a
        sharper reason -- a scene that quietly fell back to eager runs several
        times slower with no other symptom, so a profile that did not check
        would blame whatever phase happened to be on screen.
        """
        prof = self.profile
        if not prof.enabled:
            return
        model, pipeline = self.env.model, self.env.pipeline
        batches = getattr(self.viewer, "_shape_instances", {})
        prof.note(
            task=getattr(self.env.request, "name", ""),
            shape_batches=len(batches),
            shapes=int(getattr(model, "shape_count", 0)),
            bodies=int(getattr(model, "body_count", 0)),
            particles=int(getattr(model, "particle_count", 0)),
            triangles=int(getattr(model, "tri_count", 0)),
            substeps=pipeline.substeps,
            control_hz=pipeline.control_hz,
            speed=self.config.speed,
            cameras=len(self.live.specs()) if self.live is not None else 0,
            cloth_look=self.config.cloth_look,
        )

    def _note_capture(self) -> None:
        """The graph-capture counters, read at the end of the measured window."""
        prof = self.profile
        if not prof.enabled:
            return
        pipeline = self.env.pipeline
        prof.note(
            steps_replayed=pipeline.steps_replayed,
            steps_eager=pipeline.steps_eager,
            recapture_failures=pipeline.recapture_failures,
            capture_error=pipeline.capture_error or "none",
        )

    @contextmanager
    def _device_span(self):
        """Wrap the solver step in the profiler's event clock, when it has one.

        Only the teleop step is bracketed. A policy step is a whole rollout's
        worth of host work around the same solver call, and recording events
        around *that* would report the device as busy for the time its own
        planner spent thinking.
        """
        clock = self.profile.device
        if clock is None:
            yield
            return
        with clock.measure():
            yield

    @property
    def wants_pixels(self) -> bool:
        """Whether anything is looking through the environment's camera rig.

        A policy that has perception nodes reads ``obs["cameras"]``, a take
        records whatever the environment rendered, and the live view reads it
        when its source is the observation rig. Nothing else does -- the 3D
        scene draws through viser, not through the rig. Left running for an idle
        scene the rig is a tiled render of two cameras every fourth step for no
        reader, measured at 19% of the step rate on a tool-sorting scene,
        which is why this is a question and not simply ``True``.

        The live view has to be asked rather than assumed, and getting that
        wrong fails quietly in the worse direction: with the rig off, its
        observation source keeps showing whatever frames the last policy run
        left in ``env.frames`` -- a stale image rather than an absent one.
        """
        return (
            (self.driver is not None and self.driver.running)
            or (self.record is not None and self.record.recording)
            or (self.live is not None and self.live.wants_obs)
        )

    def refresh_view(self) -> None:
        """Redraw the browser and the read-only text. Whichever limit is slower.

        ``render_every`` alone is a step count, and a step count asks for a
        frame rate that moves with the step rate: the same ``render_every`` of 4
        is 12.5 fps on a paced 50 Hz task and 27 fps on the same scene at
        ``max``, so the faster the loop ran the larger the share of itself it
        spent drawing. On a tool-sorting scene at ``max`` that was 23% of the
        step rate.

        A wall-clock rate alone fails the other way, and worse. On a scene that
        steps slower than the rate asked for -- which is most coupled scenes --
        holding 12.5 fps means drawing *more* often than every fourth step:
        measured on the same task, 116 frames per 200 steps against the step
        rule's 50, and the step rate fell 61%. That is the failure
        :class:`~robosimstudio.core.render.RenderBudget` was written to
        prevent, and its load cap is the principled answer -- but not here. The
        cap divides by a *measured* frame cost, and viser's frame cost is not
        measurable early: the first draws of a scene take ~300 ms and fall to
        ~11 ms only after a hundred-odd of them, so a cap set from the cold
        measurement throttles the view, and throttling the view is what stops
        the warm-up ever finishing. Measured at a 0.25 cap: 1.5 fps still 40 s
        into a session, beside a solver running at 51 steps/s.

        So both limits, and the slower one wins. Wall clock stops a fast loop
        drawing frames nobody asked for; the step count stops a slow one
        drawing the same picture twice. It can never ask for more frames than
        the step rule did, so it cannot cost throughput, and no measurement is
        involved to be wrong about.

        The readouts refresh on the wall-clock limit alone, without waiting for
        a frame: they are cheap, and they are what changes on a scene that is
        *not* advancing, since clicking something while paused re-aims a grasp
        and rewrites the verdict.
        """
        now = time.perf_counter()
        if now - self._refreshed_at < self.draw_interval:
            return
        self._refreshed_at = now
        with self.profile.phase("iter.refresh"):
            for panel in self.panels:
                with self.profile.phase(f"iter.refresh.{panel.profile_name}"):
                    panel.refresh()
            self.refresh_holding()
        step = self.env.step_count
        if self._drawn_step is not None and step - self._drawn_step < self.render_every:
            return
        self._drawn_step = step
        self.draw()

        started, frames = self._fps_window
        if now - started >= 1.0:
            self.view_fps = (frames + 1) / (now - started)
            self._fps_window = (now, 0)
        else:
            self._fps_window = (started or now, frames + 1)

    def draw(self) -> None:
        with self.profile.phase("iter.draw"):
            self.viewer.profile = self.profile
            self.viewer.begin_frame(self.env.sim_time)
            self.viewer.log_state(self.env.state)
            self.viewer.end_frame()

    def close(self) -> None:
        self.teardown()
        try:
            self.viewer.close()
        except Exception:
            pass
