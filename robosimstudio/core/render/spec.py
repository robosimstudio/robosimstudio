# SPDX-License-Identifier: Apache-2.0
"""One backend-neutral description of how a scene should be lit.

The backends behind :class:`~robosimstudio.core.render.base.RendererBase`
agree on the picture's *geometry* and on nothing about its shading: the tiled
tracer takes one directional light and OVRTX takes the name of a preset. That
spread is by design and the backends' own tests say so -- they assert depth
agreement and deliberately do not assert brightness.

What was missing is a way to *ask* for something different without knowing which
backend is about to answer. This is that: one spec, translated per backend by
:meth:`RenderSpec.kwargs_for`.

**Every field defaults to ``None``, meaning "keep the backend's own default".**
That is deliberate and it is the reason this can be introduced without moving a
single pixel: the per-backend defaults are not a common value expressed two
ways, they are different values, and a spec that tried to pick one would
change every existing render on arrival. So an unset field is not "no light" --
it is "whatever this backend already did".

A backend that cannot honour a field ignores it rather than raising. Asking a
software rasteriser for a three-light rig is not an error, it is a request it
can only partly meet, and a run that dies because a preview backend has no
ambient term would be worse than one that looks slightly different.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["RenderSpec", "merge_render_kwargs", "put"]

Vec3 = tuple[float, float, float]


@dataclass(frozen=True, kw_only=True)
class RenderSpec:
    """How to light a scene, for whichever backend is asked.

    Attributes:
        light_direction: The way the key light *travels*, world frame. Note the
            sign convention: this is the direction of propagation, not the
            direction towards the light, and the two backends that take the
            opposite convention invert it themselves.
        key_intensity: Brightness of the key light. Unitless and not comparable
            across backends -- pyrender's 2.2 and pybullet's 0.8 are the same
            picture in different scales -- so this is a knob to turn, not a
            quantity to match.
        fill_lights: Additional ``(direction, intensity)`` pairs, for backends
            that support more than one light. Backends that take exactly one
            drop these; that loss is why the key is separate rather than being
            ``fill_lights[0]``.
        ambient: Ambient RGB. Backends taking a scalar use the mean.
        shadows: Whether the key light casts shadows.
        enable_textures: Whether to sample textures at all. Off is faster and
            gives flat shading -- enough for a depth-only pass or a preview.
        environment: OVRTX lighting preset (``"default"``, ``"studio"``,
            ``"none"``, or this package's ``"lab"``). Named separately because
            it is a whole baked environment rather than a light, and no other
            backend has one.
        environment_map: An equirectangular HDR image to light the path-traced
            scene with, as a path. ``None`` keeps the preset's own dome. Only
            the RTX backend reads it.
        scenery: A registered scenery builder
            (:func:`~robosimstudio.core.render.scenery.register_scenery`) that
            dresses the path-traced picture -- a gridded mat, a wall behind the
            arms -- with geometry that exists for the camera and nothing else.
            It is a render statement rather than a static because a static
            collides, is a landmark every framing test projects, and would be
            drawn by every backend; a backdrop is none of those. RTX only.
        quality: ``"preview"`` or ``"final"``, for a backend that can trade time
            for noise. RTX renders a burst of passes per frame and averages
            them; ``final`` is the longer burst a video wants and ``preview``
            the one an interactive look wants.
        background: What fills the frame where nothing was hit, as RGB in [0, 1].

            ``None`` keeps each backend's own, and those are not one colour
            expressed five ways either: black on the tiled tracer, transparent on
            pyrender, a blue sky gradient on the GL preview, and not settable at
            all on pybullet, where the only way to fill a miss is the per-camera
            background image that already exists.

            Worth stating because a black miss is not a neutral choice. On an
            oblique or side camera the horizon is most of the frame, and measured
            on the usbc rig 37.4 per cent of pixels came back below 32 -- a scene
            floating in void rather than standing in a room.
    """

    light_direction: Vec3 | None = None
    key_intensity: float | None = None
    fill_lights: tuple[tuple[Vec3, float], ...] | None = None
    ambient: Vec3 | None = None
    shadows: bool | None = None
    enable_textures: bool | None = None
    environment: str | None = None
    environment_map: str | None = None
    scenery: str | None = None
    quality: str | None = None
    background: Vec3 | None = None

    def directional_rig(self) -> tuple[tuple[Vec3, float], ...] | None:
        """The key plus fills as one sequence, or ``None`` if neither is stated."""
        if self.light_direction is None and self.fill_lights is None:
            return None
        rig: list[tuple[Vec3, float]] = []
        if self.light_direction is not None:
            rig.append((self.light_direction, 1.0 if self.key_intensity is None else float(self.key_intensity)))
        rig.extend(self.fill_lights or ())
        return tuple(rig)


def put(out: dict[str, Any], key: str, value: Any) -> None:
    """Record a value only if it was stated.

    Shared with the renderers, which each build their own slice of a look --
    see :meth:`robosimstudio.core.render.base.RendererBase.look_kwargs`.
    """
    if value is not None:
        out[key] = value


def merge_render_kwargs(cls: Any, render: RenderSpec | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Fold a :class:`RenderSpec` into a backend's kwargs, explicit args winning.

    Explicit keyword arguments beat the spec because they are the more specific
    statement: a caller naming ``shadows=False`` on one renderer is talking
    about that renderer, while the spec is talking about the scene.

    Takes the renderer **class**, not its alias. It used to take the name and
    resolve it through a five-arm ``if backend == ...`` chain over the built-in
    aliases, which meant a backend registered through ``register_renderer`` --
    or named by a dotted path, which the registry documents as needing no
    registration at all -- was handed ``{}`` and silently rendered with no
    lighting. Asking the class is what makes the seam's own promise true.
    """
    if render is None:
        return kwargs
    merged = cls.look_kwargs(render)
    merged.update(kwargs)
    return merged
