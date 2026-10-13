# SPDX-License-Identifier: Apache-2.0
"""A filmed take as a looping WebP under a byte budget, a contact sheet, and a timeline.

The WebP is Pillow's, not ffmpeg's ``libwebp_anim``: that sets no key-frame
limit, and its lossy sub-frame updates pile up as ghost blocks where something
moved (a towel's old shadow left on the table, in the path-traced clips this
replaces). A key frame at most every :data:`KMAX` frames bounds how long a
ghost can live; the quality steps down a ladder until the file fits. libwebp
merges a frame identical to the one before into it, so a held frame costs
nothing and a decoder sees fewer frames than were filmed, at the same times.
"""

from __future__ import annotations

import json
from pathlib import Path

QUALITIES = (85, 78, 70, 62, 54, 46)
"""Tried in order until the clip fits its budget."""
HOLD_S = 1.0
"""The last frame is held this long before the loop starts again."""
KMAX = 9
"""Most frames between key frames. Compared on the rigid take (209 frames, the page and a
moving arm): at quality 70, kmax 3 is 1398 KB and kmax 9 is 865 KB, and decoded against
the frames they came from the two are the same clip -- a mean PSNR of 36.9 and 36.8 dB,
and the worst 16-pixel block off by 13.0 levels in every frame, none worse than another,
which is what a ghost block would be. Most of a page frame is the sidebar, which never
changes, and a key frame every third frame re-sent it for nothing: to fit the same 900 KB
kmax 3 had to go down to quality 46 (36.0 dB, worst block 15.4)."""


def durations(n: int, fps: float, hold: float = HOLD_S) -> list[int]:
    """Per-frame milliseconds that add up to the clip's exact length: 67, 66, 67, ... at 15 fps."""
    out = [round(1000 * (i + 1) / fps) - round(1000 * i / fps) for i in range(n)]
    if out:
        out[-1] += round(hold * 1000)
    return out


def write_webp(frames: list, path: Path, fps: float, *, budget: int, kmax: int = KMAX) -> dict:
    """Frames (``PIL.Image``) to a looping WebP within ``budget`` bytes; what was chosen."""
    path.parent.mkdir(parents=True, exist_ok=True)
    timing = durations(len(frames), fps)
    quality = QUALITIES[0]
    for quality in QUALITIES:
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=timing,
            loop=0,
            quality=quality,
            method=4,
            kmin=max(1, kmax - 1),
            kmax=kmax,
        )
        if path.stat().st_size <= budget:
            break
    size = path.stat().st_size
    return {"quality": quality, "bytes": size, "kmax": kmax, "fits": size <= budget, "budget": budget}


def sheet(
    frames: list, times: list[float], captions: list[str], path: Path, *, grid: int = 4, width: int = 480
) -> None:
    """A ``grid`` x ``grid`` contact sheet of evenly spaced frames, each labelled with sim time and beat."""
    from PIL import Image, ImageDraw, ImageFont

    count = grid * grid
    picks = sorted({round(i * (len(frames) - 1) / (count - 1)) for i in range(count)}) if len(frames) > 1 else [0]
    tile_w = width
    tile_h = round(width * frames[0].height / frames[0].width)
    out = Image.new("RGB", (tile_w * grid, tile_h * grid), "white")
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Pillow before 10.1 has one size
        font = ImageFont.load_default()
    for slot, index in enumerate(picks):
        tile = frames[index].resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(tile)
        label = f"{times[index]:.2f} s  {captions[index]}"
        box = draw.textbbox((6, 4), label, font=font)
        draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0))
        draw.text((6, 4), label, fill=(255, 255, 255), font=font)
        out.paste(tile, ((slot % grid) * tile_w, (slot // grid) * tile_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def timeline(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
