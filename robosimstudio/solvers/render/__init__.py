# SPDX-License-Identifier: Apache-2.0
"""How a frame is computed: the renderer, and the fallback that stands in for it.

``core/render`` keeps what a renderer *is* -- the ``RendererBase`` contract,
the frame set, the scene flattening, the palette -- and these are the
implementations of it: ``rtx``, the OVRTX path tracer every picture is made
with, and ``tiled``, Newton's tiled ray tracer, which draws the colliders alone
and serves where ``rtx`` cannot: no ``ovrtx`` installed, or a GPU without RT
cores (:func:`choose_renderer` has the rule). Core reaches them by dotted path
through ``RENDERERS`` and imports neither, so a checkout without the ``[rtx]``
extra still imports core and finds out only when it asks for a frame.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Sequence
from typing import Any

import newton

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.core.render.base import RendererBase
from robosimstudio.core.render.spec import RenderSpec, merge_render_kwargs
from robosimstudio.registry import register_alias

__all__ = [
    "AUTO",
    "ENV",
    "FALLBACK",
    "PUBLIC",
    "RENDERERS",
    "RT_ARCHES",
    "backend_class",
    "choose_renderer",
    "create_renderer",
    "default_renderer",
    "list_renderers",
    "look_kwargs",
    "register_renderer",
    "rt_cores",
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
registered as the fallback :func:`choose_renderer` picks where ``rtx`` cannot
run, and as the thing a test renders through; it is not offered by
:func:`list_renderers`, though :data:`ENV` and a name in code may ask for it.

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
"""What stands in for it where ``rtx`` cannot run. Never offered by name."""


def list_renderers() -> tuple[str, ...]:
    """The aliases to offer: ``rtx``, and whatever was registered after import.

    A live view rather than a frozen tuple, so a renderer registered through
    :func:`register_renderer` shows up in a menu that reads this. The fallback
    is left out on purpose -- it is a stand-in, not a choice.
    """
    return tuple(alias for alias in RENDERERS if alias != FALLBACK)


AUTO = "auto"
"""The name for "pick for me": :func:`choose_renderer` decides from the machine."""

ENV = "ROBOSIMSTUDIO_RENDERER"
"""The environment variable that names the renderer for a whole machine or shell.

A registered alias, a dotted path, or ``auto``. A name passed in code wins over it.
"""

RT_ARCHES = frozenset({75, 86, 89, 120, 121})
"""CUDA architectures whose GPUs carry RT cores, which OVRTX needs.

Turing (7.5: T4, Quadro RTX, RTX 20), Ampere's graphics parts (8.6: A10, A40,
RTX A6000, RTX 30), Ada (8.9: L4, L40S, RTX 6000 Ada, RTX 40) and Blackwell's
graphics parts (12.0 and 12.1: RTX PRO 6000, RTX 50, GB10). The compute parts
have none: V100 (7.0), A100 (8.0), Orin (8.7), H100 and H200 (9.0), B200 and
B300 (10.x). Neither ovrtx nor NVML can be asked directly, so this is a list of
known parts; an architecture it has not heard of gets ``tiled``, and
``ROBOSIMSTUDIO_RENDERER=rtx`` overrides it.
"""

_NO_RT_NAMES = ("GTX", " MX")
"""Turing chips without RT cores: GTX 16 and the MX laptop parts share 7.5."""


def rtx_available() -> bool:
    """Whether the path tracer would import, without importing it."""
    try:
        return importlib.util.find_spec("ovrtx") is not None
    except (ImportError, ValueError):
        return False


def rt_cores(device: Any = None) -> tuple[bool, str]:
    """Whether the GPU a renderer would run on has RT cores, and which GPU it is.

    Read off Warp's device table, which costs nothing: importing ovrtx to ask it
    builds CUDA state, and a failed ovrtx start on a part with no RT cores is not
    a clean exception to catch.

    Args:
        device: A Warp device or its alias; ``None`` is Warp's default.
    """
    import warp as wp

    try:
        gpu = wp.get_device(device)
    except Exception as exc:
        return False, f"no CUDA device ({exc})"
    if not gpu.is_cuda:
        return False, f"{gpu.alias} is not a CUDA device"
    label = f"{gpu.name} (sm_{gpu.arch})"
    if gpu.arch not in RT_ARCHES or any(tag in gpu.name for tag in _NO_RT_NAMES):
        return False, f"{label} has no RT cores"
    return True, label


def choose_renderer(requested: str = "", device: Any = None) -> tuple[str, str]:
    """Which renderer to build, and why, in this order:

    1. ``requested``, when it names one -- a renderer named in code;
    2. :data:`ENV`, when it names one -- a renderer named for the machine;
    3. otherwise ``rtx`` where ovrtx is installed and ``device`` has RT cores,
       else ``tiled``.

    Empty and ``auto`` name nothing, at either level. A name is used as given:
    ``rtx`` asked for on a GPU without RT cores is tried, not swapped for
    ``tiled``.

    Args:
        requested: A registered alias, a dotted path, ``auto`` or empty.
        device: The Warp device the renderer will run on; ``None`` is Warp's default.

    Returns:
        ``(backend, why)``. ``why`` is a short phrase for a log line or a readout.

    Raises:
        ValueError: If a name is neither a registered alias nor a dotted path.
        ImportError: If ``rtx`` is named but ovrtx is not installed.
    """
    for given, source in ((requested, "asked for"), (os.environ.get(ENV, ""), ENV)):
        name = given.strip()
        if not name or name == AUTO:
            continue
        if name not in RENDERERS and "." not in name:
            where = f"{ENV}={name!r}" if source == ENV else repr(name)
            raise ValueError(f"unknown renderer {where}: choose {AUTO!r}, {list(RENDERERS)} or a dotted path")
        if name == PUBLIC and not rtx_available():
            raise ImportError(
                f"renderer {PUBLIC!r} was {'set by ' + ENV if source == ENV else 'asked for'} but ovrtx is not "
                f"installed: `uv sync` installs it, as does pip install "
                f"'robosimstudio[rtx] @ git+https://github.com/robosimstudio/robosimstudio' "
                f"--extra-index-url https://pypi.nvidia.com; or leave it unset to pick by the machine"
            )
        return name, f"{ENV}={name}" if source == ENV else "asked for"
    if not rtx_available():
        return FALLBACK, "ovrtx is not installed"
    capable, gpu = rt_cores(device)
    return (PUBLIC if capable else FALLBACK), gpu


def default_renderer(device: Any = None) -> str:
    """The renderer :func:`choose_renderer` picks when nothing is asked for in code."""
    return choose_renderer("", device)[0]


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
