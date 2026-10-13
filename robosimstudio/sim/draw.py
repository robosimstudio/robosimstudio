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

from typing import Any

__all__ = ["Viewer", "draw", "view", "viewer"]


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
        share: Ask viser for a public URL as well as the local one. **On by
            default**, because the machine with the GPUs is usually not the
            machine with the browser, and ``localhost:8080`` on a remote box is
            a page nobody can open. Pass ``share=False`` for a local-only page.

            It is viser's own share service and it is experimental: the request
            blocks until the tunnel answers, so a run that hangs before printing
            a URL is this, and ``share=False`` is the way past it.
    """

    def __init__(self, sim: Any, *, port: int = 8080, label: str = "", share: bool = True) -> None:
        try:
            from newton.viewer import ViewerViser
        except ImportError as exc:  # pragma: no cover -- depends on the install
            raise ImportError("a viewer needs viser, which this install does not have: pip install viser") from exc

        self.sim = sim
        self._viewer = ViewerViser(port=port, label=label or getattr(sim, "name", "") or "robosimstudio", share=share)
        self._viewer.set_model(sim.model)

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

    def is_running(self) -> bool:
        """Whether the page is still open. The loop condition."""
        return self._viewer.is_running()

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


def viewer(sim: Any, *, port: int = 8080, label: str = "", share: bool = True) -> Viewer:
    """A :class:`Viewer` over ``sim``: the page an example drives itself.

    ::

        viewer = rss.viewer(sim)
        target = viewer.scene.add_transform_controls("/target", scale=0.15, position=(0.6, 0.0, 0.9))
        while viewer.is_running():
            ...                       # read the gizmo, command the arm
            sim.step(action, mode="joint_position")
            viewer.show()

    The other two doors: :func:`draw` is this with no loop of your own, and
    :func:`view` is the whole studio.
    """
    return Viewer(sim, port=port, label=label, share=share)


def draw(sim: Any, *, port: int = 8080, steps: int | None = None, label: str = "", share: bool = True) -> None:
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
        share: Ask viser for a public URL too. On by default; see :class:`Viewer`.
    """
    with viewer(sim, port=port, label=label, share=share) as page:
        taken = 0
        while page.is_running() and (steps is None or taken < steps):
            sim.pipeline.step()
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


def view(
    sim: Any,
    *,
    panels: Any = "drive",
    page: Any = None,
    routines: Any = None,
    render_every: int = 1,
    port: int = _UNSET,
    share: bool = True,
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
    for them to say.

    Args:
        sim: A :class:`~robosimstudio.sim.Sim`, a registered task id, or a
            ``TaskSpec``.
        panels: Which panels to build -- a key of
            :data:`~robosimstudio_studio.session.PANEL_SETS` or the keys
            themselves. **Defaults to** ``"drive"`` here and ``"all"`` on
            :class:`~robosimstudio_studio.StudioConfig`, which is the one place
            the two doors differ and is deliberate: this is the door a script
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
            written against a bare ``ViserServer`` works here unchanged.
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
            4 on :class:`~robosimstudio_studio.StudioConfig` -- the third of the
            three defaults that differ between the two doors, and the one with a
            measurement behind it.

            4 suits a page that is being *measured*: drawing is a fifth of a fast
            scene's step budget and a benchmark would rather have the steps. It
            is wrong for a page being *watched*, and worst exactly where watching
            matters most -- a coupled cloth scene steps at about 30 Hz, and one
            frame per four steps is 5 fps of a towel you are trying to drag.
            Measured on ``examples/scenes/cloth``: 5.7 fps at 4 against 26.6 fps at 1,
            with the step rate falling from 33.3/s to 26.6/s. A fifth of the
            throughput for four times the frame rate is the trade an interactive
            page wants and a batch job does not.
        port: What to serve on. The URL is printed.
        share: Ask viser for a public URL as well as the local one. **On by
            default here**, and off on
            :class:`~robosimstudio_studio.StudioConfig`, for the same reason
            ``panels`` differs: this is the door a person opens by hand, usually
            onto a machine that has the GPUs and not the browser, where
            ``localhost:8080`` is a page nobody can reach. A test or a batch job
            builds the config itself and gets no tunnel.

            viser's share service is experimental and the request blocks until
            it answers, so a call that hangs before printing a URL is this;
            ``share=False`` is the way past it.
        handles: How many things may be held at once, per kind.
        speed: Initial playback rate, a key of
            :data:`~robosimstudio_studio.session.SPEEDS`.
        dark: Dark browser chrome and a near-black canvas.
        cloth_look: How deformables are shaded -- ``fluffy``, ``shell``, ``flat``.
        lighting: A :class:`~robosimstudio_studio.lighting.LightingRig`.
        scenery: Draw the cell's furniture. Off by default because the browser
            hit-tests it first, and a table in front of an arm swallows the
            clicks meant for it.
        config: A whole :class:`~robosimstudio_studio.StudioConfig`. Any keyword
            above is applied on top of it, so the two compose.
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
    changes = {_CONFIG_FIELDS[name]: value for name, value in given.items() if value is not _UNSET}
    settings = dataclasses.replace(config or StudioConfig(), **changes)

    session = StudioSession(sim, config=settings, **kwargs)
    if routines:
        session.add_routines(routines)
    if page is not None:
        page(session)
    session.run()
