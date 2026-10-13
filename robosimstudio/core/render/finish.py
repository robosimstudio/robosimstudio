# SPDX-License-Identifier: Apache-2.0
"""What a surface does with light, stated beside what colour it is.

:mod:`~robosimstudio.core.render.palette` answers "what colour is this thing";
nothing answered "is it matte or glossy, metal or plastic, opaque or a film you
see the bench through".

A :class:`Finish` is that answer: roughness, metallicity, opacity, an index of
refraction, and optionally a diffuse map. It is a display statement exactly the
way ``color`` is -- the solver never reads it -- and it is resolved the same way:
a spec may state one, and a spec that states nothing gets a default keyed by the
colour it was painted (:func:`finish_for`), so every scene that predates this
module renders as sensible plastic, metal or fabric without an edit.

**Only the path-traced backend honours a finish today.** The tiled tracer,
pyrender and pybullet keep drawing colour alone, which is why a finish never
changes a recorded observation; it changes the picture a person looks at. That
asymmetry is deliberate and documented rather than papered over: a policy
evaluated on tiled pixels sees exactly what it saw before this module existed.

Textures are named rather than pathed where they are procedural. A grid mat or
a kraft-paper pattern is forty lines of numpy, and shipping the PNG instead
means a binary asset nobody can diff; :func:`register_texture` keeps the
generator beside the look that needs it and :func:`texture_path` renders it once
into a cache. A finish may still name a file on disk, which is what an authored
diffuse map is.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robosimstudio.registry import Registry

from .palette import ACCENT_PEG, DEFORMABLE, FIXTURE, FLOOR, GEAR_BRASS, GEAR_BRONZE, GROUND, MACHINE, METAL, ROBOT_DARK

__all__ = [
    "BRUSHED_METAL",
    "DEFAULT",
    "GLOSS",
    "MATTE",
    "SATIN",
    "TEXTURES",
    "Finish",
    "finish_for",
    "register_color_finish",
    "register_texture",
    "texture_cache_dir",
    "texture_path",
]

Vec3 = tuple[float, float, float]

COLOR_TOLERANCE = 2.0e-3
"""How close a painted colour has to be to a keyed one to take its finish.

Loose enough to recognise a palette entry written out by hand: a colour that
round-trips through ``float32`` on the builder is still that colour.
"""


@dataclass(frozen=True, kw_only=True)
class Finish:
    """How a surface reflects light. Display only; the solver never reads it.

    Attributes:
        roughness: Microfacet roughness in [0, 1]. 0 is a mirror, 1 is chalk.
            0.6 is the default because it is what an untextured plastic reads
            as under a path tracer, and it is the value nothing complains about.
        metallic: Metalness in [0, 1]. Dielectric at 0, conductor at 1; the
            values between are for painted or oxidised metal, not for "a bit
            shiny" -- that is :attr:`roughness`.
        opacity: Display alpha in (0, 1]. Below 1 the surface draws
            translucent, which is what a protective film or packing tape *is*.
            The cloth specs already carry an ``opacity`` for the visual-matching
            backends; a finish stating one wins where a finish is honoured.
        ior: Index of refraction for the translucent case. 1.5 is glass and
            most polymers; a value on an opaque surface is harmless.
        texture: A diffuse map -- a path on disk, or a name registered through
            :func:`register_texture`. Empty draws the shape's own colour.
        uv_scale: Texture repeats per metre along the two projected axes, for
            a primitive that arrives without UVs and is given a box projection
            in its own metres. A 60 cm mat tile stated as ``(1/0.6, 1/0.6)``
            repeats every 60 cm regardless of which box it is painted on.
        uv_offset: A shift of that projection in texture repeats, so a small
            fixture standing on a gridded mat can align its grid to the mat's.
        roughness_texture: A greyscale map modulating :attr:`roughness`, for a
            surface whose shine varies -- a crinkled film -- rather than one
            whose colour does.
    """

    roughness: float = 0.6
    metallic: float = 0.0
    opacity: float = 1.0
    ior: float = 1.5
    texture: str = ""
    uv_scale: tuple[float, float] = (1.0, 1.0)
    uv_offset: tuple[float, float] = (0.0, 0.0)
    roughness_texture: str = ""

    def __post_init__(self) -> None:
        for name in ("roughness", "metallic"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Finish.{name} must be in [0, 1], got {value}")
        if not 0.0 < float(self.opacity) <= 1.0:
            raise ValueError(f"Finish.opacity must be in (0, 1], got {self.opacity}")
        if float(self.ior) < 1.0:
            raise ValueError(f"Finish.ior must be >= 1, got {self.ior}")
        if len(self.uv_scale) != 2 or len(self.uv_offset) != 2:
            raise ValueError("Finish.uv_scale and uv_offset are (u, v) pairs")

    @property
    def translucent(self) -> bool:
        return float(self.opacity) < 1.0


DEFAULT = Finish()
"""Untextured plastic. What a spec that states nothing gets, unless its colour
says otherwise."""

MATTE = Finish(roughness=0.9)
"""Chalk, fabric, foam, paper: no visible highlight."""

SATIN = Finish(roughness=0.5)
"""Anodised or bead-blasted parts and 3D prints: a soft, broad highlight."""

GLOSS = Finish(roughness=0.3)
"""Injection-moulded shells and painted panels: a tight highlight."""

BRUSHED_METAL = Finish(roughness=0.35, metallic=0.9)
"""Stainless and aluminium as they come from the mill."""

_BY_ROLE: dict[str, Finish] = {
    "world": MATTE,
    "surface": MATTE,
    "fixtures": Finish(roughness=0.8),
    "deformables": MATTE,
    "robot": SATIN,
    "hands": SATIN,
    "self": Finish(roughness=0.8),
    "objects": DEFAULT,
}

_BY_COLOR: list[tuple[Vec3, Finish]] = [
    (METAL, BRUSHED_METAL),
    (ACCENT_PEG, BRUSHED_METAL),
    (GEAR_BRASS, BRUSHED_METAL),
    (GEAR_BRONZE, BRUSHED_METAL),
    (DEFORMABLE, MATTE),
    (ROBOT_DARK, SATIN),
    (FLOOR, MATTE),
    (GROUND, MATTE),
    (FIXTURE, Finish(roughness=0.8)),
    (MACHINE, SATIN),
]
"""Palette colours that imply a finish. Checked before the role default, because
a ``METAL`` bar in the ``objects`` role is still metal."""


def register_color_finish(color: Vec3, finish: Finish) -> None:
    """Key a finish to a colour, for a module that owns a colour outside the palette.

    The YAM's white shells are the case: ``yam_look`` states the colour, and
    stating "and it is glossy" beside it is one registration rather than a
    dependency from this module on a robot.
    """
    key = tuple(float(v) for v in color)
    for i, (existing, _) in enumerate(_BY_COLOR):
        if _close(existing, key):
            _BY_COLOR[i] = (key, finish)
            return
    _BY_COLOR.append((key, finish))


def _close(a: Vec3, b: Vec3) -> bool:
    return all(abs(float(x) - float(y)) <= COLOR_TOLERANCE for x, y in zip(a, b, strict=True))


def finish_for(color: Vec3 | None, role: str = "objects") -> Finish:
    """The finish a shape gets when its spec states none.

    By colour first -- the palette's scenery ladder and the deformable white
    were chosen to *mean* something, and the meaning carries a finish -- then
    by the entity's display role, then :data:`DEFAULT`.
    """
    if color is not None:
        key = tuple(float(v) for v in color)
        for existing, finish in _BY_COLOR:
            if _close(existing, key):
                return finish
    return _BY_ROLE.get(role, DEFAULT)


# ------------------------------------------------------------------ textures


def _same_texture(a: tuple[Callable[[], np.ndarray], int], b: tuple[Callable[[], np.ndarray], int]) -> bool:
    """One generator shipped twice -- a task module imported under two names -- is the same texture."""
    return a[1] == b[1] and getattr(a[0], "__qualname__", None) == getattr(b[0], "__qualname__", None)


TEXTURES: Registry[tuple[Callable[[], np.ndarray], int]] = Registry("texture", same=_same_texture)
"""``name -> (generator, version)``. A generator returns an ``(H, W, 3)`` or
``(H, W)`` array in sRGB, ``uint8`` or floats in [0, 1]."""


def register_texture(name: str, generator: Callable[[], np.ndarray], *, version: int = 1) -> None:
    """Register a procedural texture under ``name``.

    ``version`` is part of the cached file's name: bump it when the generator
    changes, so a cache rendered by the old one is not served for the new.
    """
    if not name or "/" in name or "\\" in name:
        raise ValueError(f"a texture name is a bare identifier, got {name!r}")
    TEXTURES.register(name, (generator, int(version)), origin=getattr(generator, "__module__", ""))


def texture_cache_dir() -> Path:
    """Where rendered procedural textures go: ``$ROBOSIMSTUDIO_CACHE/textures``
    or ``~/.cache/robosimstudio/textures``."""
    root = os.environ.get("ROBOSIMSTUDIO_CACHE")
    base = Path(root) if root else Path.home() / ".cache" / "robosimstudio"
    return base / "textures"


def texture_path(texture: str) -> str:
    """Resolve a finish's texture to a PNG on disk, rendering a procedural one once.

    A path that exists is returned absolute. A registered name is rendered into
    :func:`texture_cache_dir` the first time it is asked for. Anything else is an
    error here rather than a silently untextured shape at render time.
    """
    if not texture:
        raise ValueError("no texture stated")
    if os.path.isfile(texture):
        return os.path.abspath(texture)
    entry = TEXTURES[texture] if texture in TEXTURES else None
    if entry is None:
        raise FileNotFoundError(
            f"texture {texture!r} is neither a file nor a registered generator; have {TEXTURES.names()}"
        )
    generator, version = entry
    path = texture_cache_dir() / f"{texture}_v{version}.png"
    if not path.is_file():
        image = np.asarray(generator())
        if image.dtype != np.uint8:
            image = (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.png")
        Image.fromarray(np.ascontiguousarray(image[..., :3])).save(tmp)
        os.replace(tmp, path)
    return str(path)
