# SPDX-License-Identifier: Apache-2.0
"""How a frame is computed: the renderer, and the fallback that stands in for it.

``core/render`` keeps what a renderer *is* -- the ``RendererBase`` contract,
the frame set, the scene flattening, the palette -- and these are the
implementations of it: ``rtx``, the OVRTX path tracer every picture is made
with, and ``tiled``, Newton's tiled ray tracer, which draws the colliders alone
and serves where ``ovrtx`` is not installed. Core reaches them by dotted path
through ``RENDERERS`` and imports neither, so a checkout without the ``[rtx]``
extra still imports core and finds out only when it asks for a frame.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import newton

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.core.render.base import RendererBase
from robosimstudio.core.render.spec import RenderSpec, merge_render_kwargs
from robosimstudio.registry import register_alias

__all__ = [
    "FALLBACK",
    "PUBLIC",
    "RENDERERS",
    "backend_class",
    "create_renderer",
    "default_renderer",
    "list_renderers",
    "look_kwargs",
    "register_renderer",
    "rtx_available",
    "supported_channels",
]

RENDERERS: dict[str, str] = {
    "rtx": "robosimstudio.solvers.render.rtx.RendererRTX",
    "tiled": "robosimstudio.solvers.render.tiled.RendererTiled",
}
"""Alias -> dotted path.

**The renderer is ``rtx``**, the OVRTX path tracer: the one that draws the room,
the materials and the lights, and the only one a page, a snapshot or a video
is made with. ``tiled`` -- Newton's tiled ray tracer, colliders only -- stays
registered as the fallback :func:`create_renderer` reaches for where ``ovrtx``
is not installed, and as the thing a test renders through; it is not offered
by :func:`list_renderers` and nothing user-facing takes its name.

Held as **strings**, which is what keeps the table free: a base install pays
for nothing until something actually asks for a renderer.

Open on the same terms every implementation slot in this stack uses: an alias
may be added with :func:`register_renderer`, and **a dotted path needs no
registration at all**. That is the extensibility contract -- a renderer nobody
here has heard of is nameable by whoever wrote it.
"""


def register_renderer(alias: str, cls: str | type) -> str:
    """Register a renderer under an alias. Identical re-registration is accepted.

    Args:
        alias: The name a caller will use.
        cls: The renderer class, or its dotted path. A class is recorded **by
            path**, so the registry itself never holds an import.

    Raises:
        ValueError: If ``alias`` is taken by a different renderer.
    """
    return register_alias(RENDERERS, "render backend", alias, cls)


PUBLIC = "rtx"
"""The renderer every picture is made with."""

FALLBACK = "tiled"
"""What stands in for it where ``ovrtx`` does not import. Never offered by name."""


def list_renderers() -> tuple[str, ...]:
    """The aliases to offer: ``rtx``, and whatever was registered after import.

    A live view rather than a frozen tuple, so a renderer registered through
    :func:`register_renderer` shows up in a menu that reads this. The fallback
    is left out on purpose -- it is a stand-in, not a choice.
    """
    return tuple(alias for alias in RENDERERS if alias != FALLBACK)


def rtx_available() -> bool:
    """Whether the path tracer would import, without importing it."""
    try:
        return importlib.util.find_spec("ovrtx") is not None
    except (ImportError, ValueError):
        return False


def default_renderer() -> str:
    """``rtx`` where it is installed, else the fallback -- the name :meth:`Sim.render` builds."""
    return PUBLIC if rtx_available() else FALLBACK


def backend_class(backend: str) -> type[RendererBase]:
    """The class implementing one backend, imported on demand.

    Separated from :func:`create_renderer` so a caller can ask what a backend
    *can produce* without building one -- negotiating channels needs the
    answer before the constructor runs, and the constructor is the expensive,
    side-effecting part (a GL context, a USD stage, an ovrtx viewer).

    Note:
        Every backend is imported here rather than at module import, so a base
        install pays for nothing until it asks. The names this accepts are the
        registered aliases, or any dotted path.
    """
    path = RENDERERS.get(backend) or (backend if "." in backend else "")
    if not path:
        raise ValueError(
            f"unknown render backend {backend!r}: neither a registered alias ({list(RENDERERS)}) nor a dotted path"
        )
    module_name, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module_name), name)


def look_kwargs(backend: str, look: RenderSpec) -> dict[str, Any]:
    """What one backend takes from a look, without constructing it.

    The other half of :func:`supported_channels`: both ask the class a question
    the constructor is too expensive to answer. Accepts a registered alias or a
    dotted path, so a renderer nobody here has heard of is asked on the same
    terms as the ones that ship.
    """
    return backend_class(backend).look_kwargs(look)


def supported_channels(backend: str) -> tuple[str, ...]:
    """What one backend can produce, without constructing it.

    Read off the class rather than restated here, so it cannot drift from what
    the constructor actually enforces.
    """
    return tuple(backend_class(backend).supported_channels)


def create_renderer(
    backend: str,
    model: newton.Model,
    cameras: Sequence[CameraSpec],
    *,
    render: RenderSpec | None = None,
    **kwargs: Any,
) -> RendererBase:
    """Build a renderer by name.

    Args:
        backend: A registered alias -- see :func:`list_renderers` -- or a
            dotted path to any class answering :class:`RendererBase`.
        model: The finalized model.
        cameras: The cameras to render.
        render: How to light the scene, backend-neutrally. Each backend takes
            the fields it can honour and ignores the rest; a field this spec
            leaves unset keeps that backend's own default, so passing ``None``
            -- or a spec that states nothing -- renders exactly as before.
        **kwargs: Forwarded to the backend, and winning over ``render`` where
            they overlap.
    """
    cls = backend_class(backend)
    return cls(model, cameras, **merge_render_kwargs(cls, render, kwargs))
