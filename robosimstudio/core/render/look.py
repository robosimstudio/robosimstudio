# SPDX-License-Identifier: Apache-2.0
"""The lighting a scene gets when it does not ask for one.

:class:`~robosimstudio.core.render.spec.RenderSpec` is deliberately all-``None``:
an unstated field means "keep this backend's own default", which is what let the
spec be introduced without moving a pixel. That property is worth keeping and it
is also why the spec, on its own, changed nothing -- **no shipped scene ever
stated one.** The plumbing ran from a task's spec through the compiler to five
backends and carried an empty message the whole way.

So the default has to live somewhere, and this is it: one named look, resolved at
the point a renderer is built rather than filled in per task. That placement is
the whole design decision, because it is the only one with complete coverage. A
document-level default reaches the shipped suites and nothing else; a scene built
straight from a ``BuildRequest`` -- which the bare-scene tests do, and which any
caller using the package as a library does -- would still get five different
pictures. Resolving where the renderer is made catches the observation rig, all
three display paths, the offline re-render, and every scene nobody wrote a
document for.

**Why these numbers.** The rig is chosen against the default backend's transfer
function, which is fixed: an up-facing surface receives ``ambient x 0.5`` plus
``n·l`` per light, the ambient term is compiled into the megakernel and cannot be
reached, and a shadowed surface keeps 0.3 of the key. So the key direction is
left where the tracer's own default put it -- there is no reason to disagree with
it and every recorded run in the repository was framed under it -- and what is
added is the two things that were missing.

The **fill** answers the shadows. With one light, a face the key does not reach
sits at the ambient term alone: measured, pixel 58 on a surface whose lit value
is 155. It reads as a hole rather than as a shadow, which is the single most
common complaint about the recorded videos. A second light from the opposite side
that casts no shadow of its own lifts that face to 117 while leaving the key's
own ratio at roughly 3.7:1, so the scene gains shadow detail without going flat.

The **background** answers the void. The tracer clears to black, so on an oblique
camera the horizon is most of the frame and 37.4 per cent of pixels came back
below 32. This is a cool slate that sits about 1.6:1 under the lit table: darker,
so a table edge still reads as a silhouette against it, and not black, so the
scene is standing in a room. Matching it to the ground was tried and is worse --
the horizon disappears.

**Update, RTX's own environment.** RTX draws a room (:data:`~.room.HANGAR_CLOSED`
and its siblings) that the tiled tracer has no equivalent for -- ``scenery``
is not in its ``wants``. Left at OVRTX's own baked ``"default"``
preset (a single hard key), that room's walls met the camera at a corner with one
face lit and its neighbour in the key's own shadow: not a bug in the room, a
harsh single-light rig doing on a wall what it was only ever measured against a
tabletop. ``environment="lab"`` swaps that preset for this module's own soft
top-panel-over-a-neutral-dome (see ``rtx._add_lab_lights``), which is even across
a whole wall the way the tabletop rig was only ever even across a whole bench.
``key_intensity``/``ambient`` moved with it -- 1.0 / 1.5, up from 1.05 / 0.30,
found by rendering the hangar room at a few values and reading them off, the same
way the original rig was read off a pixel value rather than assumed. The tiled
tracer reads the same two fields (its shadowed-face fraction moves too), so this
is the whole stack agreeing to be lit brighter and flatter, not an RTX-only
change wearing a shared name.
"""

from __future__ import annotations

from .spec import RenderSpec

__all__ = ["STUDIO", "look_for"]


STUDIO: RenderSpec = RenderSpec(
    light_direction=(-0.5774, 0.5774, -0.5774),
    key_intensity=1.0,
    fill_lights=(((0.90, -0.36, -0.25), 0.35),),
    ambient=(1.5, 1.5, 1.5),
    shadows=True,
    background=(0.33, 0.345, 0.39),
    environment="lab",
)
"""The house look. Every scene that states nothing renders under this.

The key is the tiled tracer's own default direction, normalised. The fill comes
from the opposite side and low, so it grazes the vertical faces the key leaves in
shadow and barely touches the floor -- and it casts no shadow itself, which is
the whole reason a second light helps rather than washing the scene out.

``environment="lab"`` is RTX's own half of this look -- read only by
the RTX renderer (``solvers/render/rtx.py``), exactly the way
``light_direction``/``fill_lights`` are read only by the tiled tracer, and no
backend errors on a field it does not use.
"""


def look_for(render: RenderSpec | None) -> RenderSpec:
    """A scene's stated look, or the house room.

    One function rather than a default argument so there is one answer to "what
    is this scene lit by", and so a caller that wants the backends' own defaults
    can still say so by passing a bare ``RenderSpec()`` -- which is not the same
    thing as saying nothing, and means something different from it.

    **What "saying nothing" resolves to is a room, not a void.** It used to be
    :data:`STUDIO` alone: a rig, a fill and a background colour, and behind the
    bench nothing at all. That was defensible while a backdrop was a thing two
    suites had drawn for themselves, and indefensible once the numbers were
    taken -- on the nine suites that had none, between 50 and 89 per cent of an
    oblique frame lands on a flat plane past two metres. So the default is
    :data:`~robosimstudio.core.render.room.HANGAR_CLOSED`, which *is* ``STUDIO``
    with a floor, four walls and a dome that carries a room's own light. Every
    field the other four backends read is unchanged by that, which is what
    makes it a change to the picture and not to the benchmark.

    Saying "no room" is still sayable, and now has a name:
    :data:`~robosimstudio.core.render.room.VOID` is exactly the old default.

    The import is local because :mod:`~robosimstudio.core.render.room` derives
    every one of its rooms from :data:`STUDIO`, so the two modules would
    otherwise import each other.
    """
    if render is not None:
        return render
    from .room import HANGAR_CLOSED

    return HANGAR_CLOSED
