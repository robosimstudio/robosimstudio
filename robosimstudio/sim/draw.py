# SPDX-License-Identifier: Apache-2.0
"""Watching a scene run, with no studio in the process.

:func:`draw` is the core tier's own viewer: Newton's ``ViewerViser``, a browser
page, and a loop. It is not the studio -- there are no panels, no tuning, no
teleoperation, no recording, and no task menu. It is what an example needs to
show that the thing you assembled does what you meant, and what a debugging
session needs when the question is "is the rope actually falling".

**The reason it exists separately from the studio** is the tier boundary. The
studio is a large, viser-heavy page over an ``Env``; importing it from a scene
that has no task is both a layer inversion and a dependency a user of the core
tier should not have to install. ``rss.view(sim)`` is the studio and says so when
viser is absent; ``rss.draw(sim)`` is this, and needs only what Newton already
brought.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

__all__ = [
    "HOME_EYE",
    "PAGE_TITLE",
    "Viewer",
    "aim_camera",
    "cell_of",
    "draw",
    "home_view",
    "title_page",
    "view",
    "viewer",
]

PAGE_TITLE = "RoboSimStudio"
"""What the browser tab says, on every page this package serves."""

HOME_EYE = (1.6, -0.8, 1.1)
"""Where a page's camera starts, from the middle of the work surface it looks at [m]: out past the surface's open
``+x`` side, which a robot standing at its own mount faces, to one side and above, 2.1 m off -- a bench and both
arms of a pair in a desktop window."""


def title_page(server: Any, title: str = PAGE_TITLE) -> None:
    """Name the browser tab ``title`` instead of viser's own ``Viser``.

    Viser's client ships ``<title>Viser</title>`` in a static ``index.html`` and
    never sets ``document.title``, so the name is set from the server instead.
    The script is queued as a persistent message, the way viser sends Plotly, so a
    browser that connects later runs it too.
    """
    from viser import _messages

    server._websock_server.queue_message(
        _messages.RunJavascriptMessage(source=f"document.title = {json.dumps(title)};")
    )


def cell_of(sim: Any) -> Any:
    """The registered cell ``sim`` was staged in, or ``None``: its compiled scene's ``handles.scene_name``, read as a
    renderer reads it (:func:`~robosimstudio.scene.registry.find_cell`), so a cell not registered here is none."""
    from robosimstudio.scene.registry import find_cell

    handles = getattr(getattr(sim, "scene", None), "handles", None)
    return find_cell(getattr(handles, "scene_name", ""))


def home_view(cell: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """``(eye, look_at)`` a page opens on: the middle of ``cell``'s work surface, from :data:`HOME_EYE` off it. With
    no cell, the world origin."""
    if cell is None:
        look_at = (0.0, 0.0, 0.0)
    else:
        look_at = (float(cell.center[0]), float(cell.center[1]), float(cell.surface_z))
    return tuple(a + b for a, b in zip(look_at, HOME_EYE, strict=True)), look_at


def aim_camera(viewer: Any, eye: Any, look_at: Any, *, move: bool = True) -> None:
    """Put a page's camera at ``eye``, looking at ``look_at``, world frame and ``+z`` up: where every browser that
    connects starts, and the point its view orbits and zooms about. ``move`` turns the browsers already open too.

    ``viewer`` is Newton's ``ViewerViser``, and this goes through its camera
    request rather than ``server.initial_camera``: Newton 1.7 opens on a camera of
    its own, 10 m out along ``+x`` and level, orbiting a point 5 m in front of it --
    4.4 m past a bench, in empty air -- and sends that to each browser as it
    connects, over whatever ``initial_camera`` says.
    """
    eye, look_at = (np.asarray(v, dtype=np.float64).reshape(3) for v in (eye, look_at))
    up = np.array((0.0, 0.0, 1.0))
    if move:
        viewer._set_camera_request(eye, look_at, up)  # initial_camera too, and every open browser
        return
    viewer._camera_request = (eye, look_at, up)
    initial = viewer._server.initial_camera
    initial.position, initial.look_at, initial.up = eye, look_at, up


class Viewer:
    """Newton's viser viewer over one built scene, with the viser server in your hands.

    This is what an example holds in its loop. It is deliberately thin: the
    Newton viewer draws the model and you talk to viser directly for everything
    else -- a gizmo for an IK target is ``viewer.scene.add_transform_controls``,
    a readout is ``viewer.gui.add_number``, exactly as they are in a pyroki
    example, because that is the API and wrapping it would only hide the
    documentation. The one thing added is :meth:`show`, which is the three calls
    Newton's viewer needs per frame in the right order, with the device sync
    that a page fed unsynced state visibly lacks.

    Args:
        sim: A :class:`~robosimstudio.sim.Sim`, or anything with ``model``,
            ``pipeline`` and ``time``.
        port: What to serve on. The URL is printed by the server.
        label: The page title.
        share: Ask viser for a public URL as well as the local one. Off by
            default: a share URL puts the page on the internet. Where the GPUs
            are on a remote box, an SSH tunnel (``ssh -L 8080:localhost:8080``)
            reaches the local page without one.

            It is viser's own share service and it is experimental: the request
            blocks until the tunnel answers, so a run that hangs before printing
            a URL is this.

    The server listens on every interface (``0.0.0.0``) whatever ``share``
    says -- Newton's ``ViewerViser`` has no host option -- so on a shared
    network anyone who can reach ``port`` can open the page and drive the
    scene. Firewall the port, or reach it through an SSH tunnel.
    """

    def __init__(self, sim: Any, *, port: int = 8080, label: str = "", share: bool = False) -> None:
        try:
            from newton.viewer import ViewerViser
        except ImportError as exc:  # pragma: no cover -- depends on the install
            raise ImportError("a viewer needs viser, which this install does not have: pip install viser") from exc

        self.sim = sim
        self._viewer = ViewerViser(port=port, label=label or getattr(sim, "name", "") or PAGE_TITLE, share=share)
        title_page(self._viewer._server)
        self._viewer.set_model(sim.model)
        aim_camera(self._viewer, *home_view(cell_of(sim)))
        # Newton greys its Reset out until something is registered, and calls it from should_step() or
        # show(), on the loop's own thread.
        reset = getattr(sim, "reset", None)
        if callable(reset):
            self._viewer.set_reset_callback(reset)

    @property
    def server(self):
        """The ``viser.ViserServer`` underneath. Newton keeps it private; the studio reads it too."""
        return self._viewer._server

    @property
    def scene(self):
        """``server.scene`` -- add gizmos, meshes and frames here."""
        return self.server.scene

    @property
    def gui(self):
        """``server.gui`` -- add sliders, numbers and buttons here."""
        return self.server.gui

    def aim_camera(self, eye: Any, look_at: Any) -> None:
        """Put the page's camera at ``eye``, looking at ``look_at``, world frame: in every browser, open or yet to
        connect (:func:`aim_camera`). The page opens on the middle of the work surface (:func:`home_view`)."""
        aim_camera(self._viewer, eye, look_at)

    def is_running(self) -> bool:
        """Whether the page is still open. The loop condition."""
        return self._viewer.is_running()

    def should_step(self) -> bool:
        """Whether the page's own **Pause** lets the next step run, or its **Step** asked for one.

        Newton 1.7's viewer puts a Simulation folder with Pause, Step and Reset on this page; it only
        records the clicks, and a loop that does not ask here steps through a pause. Its Reset calls
        ``sim.reset()``, here or in :meth:`show`; a scene with no ``reset`` leaves it greyed out.
        """
        return self._viewer.should_step()

    def show(self) -> None:
        """Draw the scene as it stands now.

        Syncs the device first: the substep block is queued, and a page fed
        unsynced state renders a frame that has not finished being computed --
        which looks like jitter and is not.
        """
        pipeline = self.sim.pipeline
        pipeline.sync()
        self._viewer.begin_frame(float(pipeline.sim_time))
        self._viewer.log_state(pipeline.state_0)
        self._viewer.end_frame()

    def close(self) -> None:
        self._viewer.close()

    def __enter__(self) -> Viewer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def viewer(sim: Any, *, port: int = 8080, label: str = "", share: bool = False) -> Viewer:
    """A :class:`Viewer` over ``sim``: the page an example drives itself.

    ::

        viewer = rss.viewer(sim)
        target = viewer.scene.add_transform_controls("/target", scale=0.15, position=(0.6, 0.0, 0.9))
        while viewer.is_running():
            ...                       # read the gizmo, command the arm
            if viewer.should_step():  # the page's Pause and Step
                sim.step(action, mode="joint_position")
            viewer.show()

    The other two doors: :func:`draw` is this with no loop of your own, and
    :func:`view` is the whole studio.
    """
    return Viewer(sim, port=port, label=label, share=share)


def draw(sim: Any, *, port: int = 8080, steps: int | None = None, label: str = "", share: bool = False) -> None:
    """Open a page on a built scene and step it until the tab closes.

    :func:`viewer` with the loop written for you: right when the question is
    "is it falling" and there is nothing to drive.

    Args:
        sim: A :class:`~robosimstudio.sim.Sim`, or anything with ``pipeline``
            and ``model`` -- duck-typed so a compiled scene works directly.
        port: What to serve on. The URL is printed.
        steps: Stop after this many control steps. ``None`` runs until you close
            the page.
        label: Shown in the page title.
        share: Ask viser for a public URL too. Off by default; see :class:`Viewer`,
            which also says what the server listens on.
    """
    # A Sim steps through its own bookkeeping -- the step count, the post-step
    # hooks -- exactly as ``sim.run`` would; a bare compiled scene has neither.
    step = getattr(sim, "_step_once", None) or sim.pipeline.step
    with viewer(sim, port=port, label=label, share=share) as page:
        taken = 0
        while page.is_running() and (steps is None or taken < steps):
            if page.should_step():
                step()
                taken += 1
            page.show()


_UNSET: Any = object()
"""Distinguishes "not passed" from a value equal to the default, so ``view`` can
amend a ``config`` the caller also gave rather than overwriting it wholesale."""

_CONFIG_FIELDS: dict[str, str] = {
    "panels": "panels",
    "render_every": "render_every",
    "port": "port",
    "share": "share",
    "handles": "handles",
    "speed": "speed",
    "dark": "dark_mode",
    "cloth_look": "cloth_look",
    "lighting": "lighting",
    "scenery": "show_scenery",
}
"""Keyword on :func:`view` -> field on ``StudioConfig``.

Nine of about sixty, chosen as the ones a script has a reason to set. The rest
are reached by passing a ``StudioConfig`` as ``config``, which is what a
launcher with a command line does; the point of this table is that a script
that wants a darker page and fewer panels does not have to import a type from
another tier to say so.
"""

_VIEW_DEFAULTS: dict[str, Any] = {"panels": "drive", "render_every": 1}
"""Where :func:`view` differs from a bare ``StudioConfig``, applied only when no
``config`` was passed."""


def view(
    sim: Any,
    *,
    panels: Any = _UNSET,
    page: Any = None,
    routines: Any = None,
    render_every: int = _UNSET,
    port: int = _UNSET,
    share: bool = _UNSET,
    handles: int = _UNSET,
    speed: str = _UNSET,
    dark: bool = _UNSET,
    cloth_look: str = _UNSET,
    lighting: Any = _UNSET,
    scenery: bool = _UNSET,
    config: Any = None,
    **kwargs: Any,
) -> None:
    """Open the **studio** on a built scene: click anything, drag it, drive it.

    The other door from :func:`draw`, and the difference is what you get rather
    than what you may pass. ``draw`` is a page and a loop; this is the whole
    interface. **Click a thing and it comes up on a handle you can drag straight
    away** -- a PD wrench on a rigid body or a robot's own link, a pin on the
    patch of cloth you hit -- and each arm carries a 6-DOF gizmo it tracks
    through IK. Nothing is welded and nothing is teleported: what a hand ends up
    holding is whatever friction and the contact settings let it hold.

    The panels that need a *document* -- the task menu, the goal readout, the
    policy runner -- are skipped when there is no task, because there is nothing
    for them to say. On a task the goal, its regions and the runner's **Run**
    are kept under ``drive`` too
    (:data:`~robosimstudio_studio.session.DOCUMENT_PANELS`): they are what a
    task's page shows. The menu is not, since it swaps the page onto another task.

    Args:
        sim: A :class:`~robosimstudio.sim.Sim`, a registered task id, or a
            ``TaskSpec``.
        panels: Which panels to build -- a key of
            :data:`~robosimstudio_studio.session.PANEL_SETS` or the keys
            themselves. **Defaults to** ``"drive"`` here and ``"all"`` on
            :class:`~robosimstudio_studio.StudioConfig`, which is one of the two
            places the two doors differ and is deliberate: this is the door a script
            opens, and a script is showing one thing, so cameras, video, takes,
            material presets and the event log are noise in front of it. A
            launcher constructs the config itself and gets the whole page.
        page: ``page(studio)``, called once with the live
            :class:`~robosimstudio_studio.StudioSession` after its panels are
            built. Where a script adds a readout or a control of its own::

                def page(studio):
                    sag = studio.gui.add_number("sag (mm)", 0.0, disabled=True)
                    studio.each_step(lambda: setattr(sag, "value", measure()))

                rss.view(sim, page=page)

            ``studio.gui`` and ``studio.scene`` are viser's own, so a component
            written against a bare ``ViserServer`` works here unchanged. The page
            opens on the middle of the work surface; ``studio.aim_camera(eye,
            look_at)`` puts the camera elsewhere.
        routines: A **Run** button for each: one callable, or a dict of name to
            callable. ``routine(sim, session)`` returns a generator that yields
            once per control step and drives the arms through their panels --
            :func:`robosimstudio_studio.routine.reach` glides a gizmo,
            :func:`~robosimstudio_studio.routine.grip` presses Grip -- so the job
            plays on the page the way a person would do it, and holds only what
            the fingers hold::

                def routine(sim, session):
                    yield from reach(session, "right", above)
                    yield from grip(session, "right", 1.0)

                rss.view(sim, routines={"Pick it up": routine})
        render_every: Draw one frame per this many control steps. **1 here**, and
            4 on :class:`~robosimstudio_studio.StudioConfig` -- the other default
            that differs between the two doors, and the one with a measurement
            behind it.

            4 suits a page that is being *measured*: drawing is a fifth of a fast
            scene's step budget and a benchmark would rather have the steps. It
            is wrong for a page being *watched*, and worst exactly where watching
            matters most -- a coupled cloth scene steps at about 30 Hz, and one
            frame per four steps is 5 fps of a towel you are trying to drag.
            Measured there: 5.7 fps at 4 against 26.6 fps at 1,
            with the step rate falling from 33.3/s to 26.6/s. A fifth of the
            throughput for four times the frame rate is the trade an interactive
            page wants and a batch job does not.
        port: What to serve on. The URL is printed.
        share: Ask viser for a public URL as well as the local one. Off by
            default, as on :class:`~robosimstudio_studio.StudioConfig`: a share
            URL puts the page on the internet. Where the GPUs are on a remote
            box, an SSH tunnel (``ssh -L 8080:localhost:8080``) reaches the
            local page without one. viser's share service is experimental and
            the request blocks until it answers, so a call that hangs before
            printing a URL is this.

            Either way the server listens on every interface (``0.0.0.0``) --
            Newton's viewer has no host option -- so on a shared network,
            firewall ``port`` or reach it through an SSH tunnel.
        handles: How many things may be held at once, per kind.
        speed: Initial playback rate, a key of
            :data:`~robosimstudio_studio.session.SPEEDS`.
        dark: Dark browser chrome and a near-black canvas.
        cloth_look: How deformables are shaded -- ``fluffy``, ``shell``, ``flat``.
        lighting: A :class:`~robosimstudio_studio.lighting.LightingRig`.
        scenery: Draw the cell's furniture. Off by default because the browser
            hit-tests it first, and a table in front of an arm swallows the
            clicks meant for it.
        config: A whole :class:`~robosimstudio_studio.StudioConfig`. A keyword
            passed above is applied on top of it, so the two compose; a keyword
            left out keeps the config's own value, ``panels`` and
            ``render_every`` included. The defaults above apply only when no
            config is given.
        **kwargs: Forwarded to
            :class:`~robosimstudio_studio.session.StudioSession` -- ``episode``.

    Raises:
        ImportError: If the studio's dependencies are absent, naming what to install.
    """
    try:
        from robosimstudio_studio.session import StudioConfig, StudioSession
    except ImportError as exc:  # pragma: no cover -- depends on the install
        raise ImportError("the studio needs viser, which this install does not have: pip install viser") from exc

    import dataclasses

    given = {
        "panels": panels,
        "render_every": render_every,
        "port": port,
        "share": share,
        "handles": handles,
        "speed": speed,
        "dark": dark,
        "cloth_look": cloth_look,
        "lighting": lighting,
        "scenery": scenery,
    }
    if config is None:
        # This door's own defaults, which differ from the config's (see above).
        # Applied only here so a caller's config keeps what it says.
        config = StudioConfig()
        given = {**_VIEW_DEFAULTS, **{name: value for name, value in given.items() if value is not _UNSET}}
    changes = {_CONFIG_FIELDS[name]: value for name, value in given.items() if value is not _UNSET}
    settings = dataclasses.replace(config, **changes)

    session = StudioSession(sim, config=settings, **kwargs)
    if routines:
        session.add_routines(routines)
    if page is not None:
        page(session)
    session.run()
