# SPDX-License-Identifier: Apache-2.0
"""Numbers beside the picture: a still of the scene measured against the photograph it should match.

A side-by-side tells an agent the case looks "a bit small and off to the
left". This tells it the case is 0.74x the photograph's size and 48 px left
of where it should be, which is a number it can move by, and a number that
either went up or did not after the next save. The loop closes on a reading
rather than an impression -- the lesson of 4DCodeBench, where the agents that
score are the ones that render, compare and edit in rounds.

Two sides, one frame. The render comes with its own segmentation (the tiled
backend's ``seg`` channel, ids named by the scene's handles), so every thing in
the scene has an exact mask for free. The photograph has none, so each thing is
*keyed*: a colour name (``case=red``), a hex colour (``socket=#c0c0c0``), a seed
pixel in the photograph (``socket=@1210,410``: the patch of that colour around
that pixel), a name restricted to a seed (``case=red@980,640``), or a mask
drawn some other way (``gasket=mask:ref/gasket.png``: a PNG the photograph's
size or not, any pixel not black being the thing). Keying is what the agent
does instead of segmenting -- it has looked at the photograph and knows the
case is the red thing. A colour key on a black, many-coloured or thin thing
finds a fragment of it; its outline on the look sheet shows that before its
number is trusted, and a mask is the way round it.

Per keyed thing: where it is in each picture, how big, the overlap. Without
keys, nothing is measured and the pictures are still made. Everything is plain
numpy, scipy and PIL, because the page's process is the one that renders and
nothing heavier is installed with it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "NAMED_COLOURS",
    "Key",
    "advice",
    "compare_masks",
    "fit_mask",
    "fit_photo",
    "load_photo",
    "mask_stats",
    "outline",
    "overlay",
    "parse_key",
    "photo_mask",
    "sim_masks",
]

NAMED_COLOURS: dict[str, dict[str, Any]] = {
    # hue windows on [0, 1); saturation and value windows on [0, 1]
    "red": {"hue": ((0.94, 1.0), (0.0, 0.045)), "sat": (0.30, 1.0), "val": (0.20, 1.0)},
    "orange": {"hue": ((0.045, 0.11),), "sat": (0.40, 1.0), "val": (0.30, 1.0)},
    "yellow": {"hue": ((0.11, 0.19),), "sat": (0.35, 1.0), "val": (0.40, 1.0)},
    "green": {"hue": ((0.19, 0.45),), "sat": (0.25, 1.0), "val": (0.15, 1.0)},
    "cyan": {"hue": ((0.45, 0.55),), "sat": (0.30, 1.0), "val": (0.20, 1.0)},
    "blue": {"hue": ((0.55, 0.72),), "sat": (0.30, 1.0), "val": (0.15, 1.0)},
    "purple": {"hue": ((0.72, 0.94),), "sat": (0.30, 1.0), "val": (0.15, 1.0)},
    "white": {"hue": (), "sat": (0.0, 0.18), "val": (0.80, 1.0)},
    "black": {"hue": (), "sat": (0.0, 1.0), "val": (0.0, 0.20)},
    "grey": {"hue": (), "sat": (0.0, 0.22), "val": (0.30, 0.85)},
    "wood": {"hue": ((0.04, 0.13),), "sat": (0.15, 0.55), "val": (0.45, 1.0)},
}
"""What a colour word means in HSV; ``grey`` is also ``chrome``, ``silver``, ``metal``, ``steel``."""
_ALIASES = {
    "chrome": "grey",
    "silver": "grey",
    "metal": "grey",
    "steel": "grey",
    "gray": "grey",
    "tan": "wood",
    "beige": "wood",
    "magenta": "purple",
    "violet": "purple",
}

_KEY = re.compile(r"^\s*([A-Za-z_][\w\-]*)\s*=\s*([^@~]*?)\s*(?:@\s*(\d+)\s*,\s*(\d+))?\s*(?:~\s*([0-9.]+))?\s*$")


@dataclass(frozen=True)
class Key:
    """One keyed thing: its name in the scene, and how to find it in the photograph."""

    name: str
    colour: str = ""
    """A colour word from :data:`NAMED_COLOURS`, or ``#rrggbb``, or empty when words or a seed stand in for it."""
    seed: tuple[int, int] | None = None
    """A pixel of the photograph the thing is at, ``(x, y)``."""
    tolerance: float = 0.0
    """For a hex colour or a seed: how far a pixel's colour may stray, 0..1 (0 takes the default)."""
    mask: str = ""
    """A PNG whose pixels that are not black are the thing, in place of a colour or a seed."""

    def describe(self) -> str:
        if self.mask:
            return f"the mask in {Path(self.mask).name}"
        parts = [self.colour or "the colour at the seed"]
        if self.seed is not None:
            parts.append(f"around ({self.seed[0]}, {self.seed[1]})")
        return " ".join(parts)


def parse_key(text: str) -> Key:
    """``name=red``, ``name=#rrggbb~0.2``, ``name=@x,y``, ``name=red@x,y``, ``name=mask:path.png``.

    A colour word or a hex colour keys by colour; a seed pixel keys by the
    colour at that pixel of the photograph, the one you looked at; a mask is
    the thing as some other tool drew it.
    """
    name, sep, how = text.partition("=")
    if sep and how.strip().startswith("mask:") and re.fullmatch(r"\s*[A-Za-z_][\w\-]*\s*", name):
        path = how.strip()[len("mask:") :].strip()
        if not path:
            raise ValueError(f"{text!r}: a mask key names a PNG, name=mask:path.png")
        return Key(name=name.strip(), mask=path)
    m = _KEY.match(text)
    if m is None:
        raise ValueError(
            f"not a key: {text!r}; a name (letters, digits, _ and -) then =red, =#rrggbb, =@x,y, =red@x,y "
            "or =mask:path.png"
        )
    name, how, sx, sy, tol = m.groups()
    how = how.strip().strip("'\"").strip()
    seed = (int(sx), int(sy)) if sx is not None else None
    lowered = _ALIASES.get(how.lower(), how.lower())
    if not how and seed is None:
        raise ValueError(f"{text!r}: a key needs a colour, a seed @x,y, or both")
    if how and not (lowered in NAMED_COLOURS or re.fullmatch(r"#[0-9a-f]{6}", lowered)):
        raise ValueError(
            f"{text!r}: no colour called {how!r}; one of {', '.join(NAMED_COLOURS)}, #rrggbb, or a pixel @x,y of the photo"
        )
    return Key(name=name, colour=lowered, seed=seed, tolerance=float(tol) if tol else 0.0)


# ------------------------------------------------------------------ masks
def _hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 ``(H, W, 3)`` to HSV floats on [0, 1]."""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    hi, lo = x.max(axis=-1), x.min(axis=-1)
    delta = hi - lo
    sat = np.where(hi > 0, delta / np.maximum(hi, 1e-6), 0.0)
    hue = np.zeros_like(hi)
    nz = delta > 1e-6
    rr = np.where(nz & (hi == r), ((g - b) / np.maximum(delta, 1e-6)) % 6.0, 0.0)
    gg = np.where(nz & (hi == g) & (hi != r), (b - r) / np.maximum(delta, 1e-6) + 2.0, 0.0)
    bb = np.where(
        nz & (hi == b) & (hi != r) & (hi != g),
        (r - g) / np.maximum(delta, 1e-6) + 4.0,
        0.0,
    )
    hue = (rr + gg + bb) / 6.0
    return np.stack([hue % 1.0, sat, hi], axis=-1)


def _rule_mask(hsv: np.ndarray, rule: dict[str, Any]) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    ok = (s >= rule["sat"][0]) & (s <= rule["sat"][1]) & (v >= rule["val"][0]) & (v <= rule["val"][1])
    windows = rule.get("hue", ())
    if windows:
        in_hue = np.zeros_like(ok)
        for lo, hi in windows:
            in_hue |= (h >= lo) & (h < hi)
        ok &= in_hue
    return ok


def _rule_for_pixel(hsv_pixel: np.ndarray, tolerance: float) -> dict[str, Any]:
    """A rule like a named colour's, centred on one pixel: a hue window for a colourful pixel, a value window for a grey one."""
    h, s, v = (float(c) for c in hsv_pixel)
    tol = tolerance or 0.06
    if s >= 0.30:
        lo, hi = (h - tol) % 1.0, (h + tol) % 1.0
        hue = ((lo, hi),) if lo < hi else ((lo, 1.0), (0.0, hi))
        return {"hue": hue, "sat": (max(0.2, 0.5 * s), 1.0), "val": (0.12, 1.0)}
    band = tolerance or 0.18
    hue: tuple = ()
    if s >= 0.06:  # pale, but tinted: chrome reads bluish-white, a pale wood table warm
        lo, hi = (h - 0.12) % 1.0, (h + 0.12) % 1.0
        hue = ((lo, hi),) if lo < hi else ((lo, 1.0), (0.0, hi))
    return {
        "hue": hue,
        "sat": (0.0, min(0.3, s + 0.08)),
        "val": (max(0.0, v - band), min(1.0, v + band)),
    }


def _hex_rule(colour: str, tolerance: float) -> dict[str, Any]:
    rgb = np.array([int(colour[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8).reshape(1, 1, 3)
    return _rule_for_pixel(_hsv(rgb)[0, 0], tolerance)


def photo_mask(photo: np.ndarray, key: Key) -> np.ndarray:
    """The pixels of ``photo`` (RGB uint8) that are the keyed thing: one connected patch, specks removed."""
    from scipy import ndimage

    if key.mask:
        return _mask_file(key.mask, photo.shape[:2])
    hsv = _hsv(photo)
    if key.colour in NAMED_COLOURS:
        rule = NAMED_COLOURS[key.colour]
    elif key.colour:
        rule = _hex_rule(key.colour, key.tolerance)
    else:
        x, y = key.seed  # type: ignore[misc]
        rule = _rule_for_pixel(hsv[min(y, hsv.shape[0] - 1), min(x, hsv.shape[1] - 1)], key.tolerance)
    raw = _rule_mask(hsv, rule)
    if not raw.any():
        return raw
    cleaned = ndimage.binary_opening(raw, iterations=1)
    if not cleaned.any():
        cleaned = raw
    labels, count = ndimage.label(cleaned)
    if count == 0:
        return cleaned
    if key.seed is not None:
        x, y = key.seed
        y, x = min(y, labels.shape[0] - 1), min(x, labels.shape[1] - 1)
        at = int(labels[y, x])
        if at == 0:
            # the seed missed the patch by a pixel or two: take the nearest labelled pixel within reach
            ys, xs = np.nonzero(labels)
            if len(ys) == 0:
                return cleaned
            d2 = (ys - y) ** 2 + (xs - x) ** 2
            i = int(np.argmin(d2))
            if d2[i] > 30**2:
                return np.zeros_like(cleaned)
            at = int(labels[ys[i], xs[i]])
        return labels == at
    sizes = ndimage.sum(cleaned, labels, index=np.arange(1, count + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def _mask_file(path: str, shape: tuple[int, int]) -> np.ndarray:
    """A mask PNG at the photograph's size: any pixel that is not black is the thing."""
    from PIL import Image

    with Image.open(path) as opened:
        image = opened.convert("RGBA")
    h, w = shape
    if image.size != (w, h):
        image = image.resize((w, h), Image.NEAREST)
    pixels = np.asarray(image)
    return (pixels[..., :3].max(axis=-1) > 0) & (pixels[..., 3] > 0)


def sim_masks(seg: np.ndarray, names_by_id: dict[int, str], wanted: list[str]) -> dict[str, np.ndarray | None]:
    """``name -> mask`` off a segmentation render; ``None`` for a name the scene has no label for."""
    ids = {name: label for label, name in names_by_id.items()}
    return {name: (seg == ids[name]) if name in ids else None for name in wanted}


def mask_stats(mask: np.ndarray | None) -> dict[str, Any] | None:
    """Where a mask is and how big: bbox, centre, area as a fraction of the frame. ``None`` for nothing."""
    if mask is None or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return {
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "centre": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
        "width": int(xs.max() - xs.min() + 1),
        "height": int(ys.max() - ys.min() + 1),
        "area": round(float(mask.sum()) / float(h * w), 5),
        "pixels": int(mask.sum()),
    }


def compare_masks(photo: np.ndarray | None, sim: np.ndarray | None) -> dict[str, Any]:
    """The readings for one thing: both sides' stats, their overlap, and the sim's offset and size against the photo."""
    p, s = mask_stats(photo), mask_stats(sim)
    out: dict[str, Any] = {
        "photo": p,
        "sim": s,
        "iou": None,
        "offset_px": None,
        "size_ratio": None,
    }
    if p is None or s is None:
        return out
    inter = float(np.logical_and(photo, sim).sum())
    union = float(np.logical_or(photo, sim).sum())
    out["iou"] = round(inter / union, 4) if union else 0.0
    out["offset_px"] = [
        round(s["centre"][0] - p["centre"][0], 1),
        round(s["centre"][1] - p["centre"][1], 1),
    ]
    out["size_ratio"] = round(float(np.sqrt(s["pixels"] / max(p["pixels"], 1))), 3)
    out["width_ratio"] = round(s["width"] / max(p["width"], 1), 3)
    out["height_ratio"] = round(s["height"] / max(p["height"], 1), 3)
    return out


# ------------------------------------------------------------------ edges
def load_photo(photo_path: str | Path) -> np.ndarray:
    from PIL import Image

    with Image.open(photo_path) as opened:
        return np.asarray(opened.convert("RGB"), dtype=np.uint8)


def fit_photo(photo: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float]:
    """The photograph at the render's size, and how far its aspect was stretched to get there (1.0 is none)."""
    from PIL import Image

    im = Image.fromarray(photo)
    stretch = (im.width / im.height) / (width / height)
    return np.asarray(im.resize((width, height), Image.BILINEAR), dtype=np.uint8), float(stretch)


def fit_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """A mask keyed on the photograph at its own size, brought to the render's."""
    from PIL import Image

    if mask.shape == (height, width):
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), Image.NEAREST)) > 127


def outline(mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    from scipy import ndimage

    return mask & ~ndimage.binary_erosion(mask, iterations=thickness)


PHOTO_COLOUR = (255, 220, 0)
SIM_COLOUR = (0, 230, 255)


def overlay(photo: np.ndarray, pairs: dict[str, tuple[np.ndarray | None, np.ndarray | None]]) -> np.ndarray:
    """The photograph with each keyed thing outlined: what the key found in it (yellow) and where the render has it (cyan)."""
    from PIL import Image, ImageDraw

    out = photo.copy()
    for p, s in pairs.values():
        if p is not None and p.any():
            out[outline(p)] = PHOTO_COLOUR
        if s is not None and s.any():
            out[outline(s)] = SIM_COLOUR
    im = Image.fromarray(out)
    draw = ImageDraw.Draw(im)
    for name, (p, s) in pairs.items():
        for mask, colour in ((p, PHOTO_COLOUR), (s, SIM_COLOUR)):
            stats = mask_stats(mask)
            if stats is None:
                continue
            x0, y0 = stats["bbox"][0], max(0, stats["bbox"][1] - 14)
            draw.text((x0 + 2, y0), name, fill=colour)
    return np.asarray(im, dtype=np.uint8)


def advice(name: str, reading: dict[str, Any], frame_width: int, key: Key | None = None) -> str:
    """One line an agent can act on."""
    p, s = reading.get("photo"), reading.get("sim")
    how = f" (keyed {key.describe()})" if key is not None else ""
    if p is None and s is None:
        return f"{name}: found in neither picture{how}"
    if p is None and key is not None and key.mask:
        return f"{name}: the mask holds nothing at this size{how}: empty, or a line too thin to survive the resize"
    if p is None:
        return f"{name}: the key found nothing in the photograph{how}; try another colour, or a seed @x,y on it"
    if s is None:
        return f"{name}: not in the render; off frame, hidden, or no thing of that name in the scene"
    spilled = key is not None and key.seed is not None and p["area"] > 0.25
    dx, dy = reading["offset_px"]
    horiz = f"{abs(dx):.0f} px {'right' if dx > 0 else 'left'}" if abs(dx) >= 1 else "level"
    vert = f"{abs(dy):.0f} px {'low' if dy > 0 else 'high'}" if abs(dy) >= 1 else "level"
    pct = 100.0 * float(np.hypot(dx, dy)) / float(frame_width)
    return (
        f"{name}: sim is {reading['size_ratio']:.2f}x the photo's size "
        f"(w {reading['width_ratio']:.2f}x, h {reading['height_ratio']:.2f}x), "
        f"centre {horiz}, {vert} ({pct:.1f}% of the frame), IoU {reading['iou']:.2f}"
        + (f"; the key spilled over {100 * p['area']:.0f}% of the photograph, tighten it with ~0.1" if spilled else "")
    )


# ------------------------------------------------------------------ what an eye sees
def region_box(
    masks: list[np.ndarray | None],
    shape: tuple[int, int],
    pad: float = 0.3,
    floor: int = 48,
) -> tuple[int, int, int, int] | None:
    """The box round every given mask, padded by ``pad`` of its size and at least ``floor`` px: where to look closely."""
    present = [m for m in masks if m is not None and m.any()]
    if not present:
        return None
    ys, xs = np.nonzero(np.logical_or.reduce(present))
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    px, py = max(floor, int(pad * (x1 - x0))), max(floor, int(pad * (y1 - y0)))
    h, w = shape
    return max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)


def crop_pair(
    photo: np.ndarray,
    render: np.ndarray,
    pm: np.ndarray | None,
    sm: np.ndarray | None,
    *,
    height: int = 240,
) -> np.ndarray | None:
    """The thing in the render beside the thing in the photograph, cut from the same box so the scale is the same."""
    from PIL import Image

    box = region_box([pm, sm], photo.shape[:2])
    if box is None:
        return None
    x0, y0, x1, y1 = box
    photo_crop = photo[y0:y1, x0:x1].copy()
    render_crop = render[y0:y1, x0:x1].copy()
    if pm is not None and pm.any():
        photo_crop[outline(pm)[y0:y1, x0:x1]] = PHOTO_COLOUR
    if sm is not None and sm.any():
        render_crop[outline(sm)[y0:y1, x0:x1]] = SIM_COLOUR
    w = max(1, int(round((x1 - x0) * height / max(1, y1 - y0))))
    tiles = [np.asarray(Image.fromarray(t).resize((w, height), Image.BILINEAR)) for t in (render_crop, photo_crop)]
    gap = np.full((height, 6, 3), 40, np.uint8)
    return np.concatenate([tiles[0], gap, tiles[1]], axis=1)


def without_things(image: np.ndarray, things: np.ndarray | None, grow: int = 10) -> np.ndarray:
    """The picture with the keyed things painted out, grown a little so their edges go too: what stands behind them."""
    from scipy import ndimage

    out = image.copy()
    if things is not None and things.any():
        out[ndimage.binary_dilation(things, iterations=grow)] = 40
    return out


def look_sheet(
    photo: np.ndarray,
    render: np.ndarray,
    rows: list[tuple[str, np.ndarray]],
    *,
    width: int = 1280,
) -> np.ndarray:
    """One picture to reason about: the pair on top, render left and photo right as in every pair, then each captioned row."""
    from PIL import Image

    h = int(round(photo.shape[0] * (width // 2) / photo.shape[1]))
    top = np.concatenate(
        [np.asarray(Image.fromarray(t).resize((width // 2, h), Image.BILINEAR)) for t in (render, photo)],
        axis=1,
    )
    blocks: list[np.ndarray] = [_caption("render  |  photo", width), top]
    for caption, given in rows:
        if given is None:
            continue
        tile = given
        if tile.shape[1] > width:
            tile = np.asarray(
                Image.fromarray(tile).resize(
                    (width, int(round(tile.shape[0] * width / tile.shape[1]))),
                    Image.BILINEAR,
                )
            )
        padded = np.full((tile.shape[0], width, 3), 24, np.uint8)
        padded[:, : tile.shape[1]] = tile
        blocks += [_caption(caption, width), padded]
    sheet = np.concatenate(blocks, axis=0)
    return sheet


def _caption(text: str, width: int, height: int = 22) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (width, height), (24, 24, 24))
    ImageDraw.Draw(im).text((6, 4), text, fill=(235, 235, 235))
    return np.asarray(im, dtype=np.uint8)
