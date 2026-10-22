# SPDX-License-Identifier: Apache-2.0
"""Stills out of the reference clip, stamped with their time, and the timeline they anchor.

A clip is the reference most twins start from and the one thing the page could
not read: the agent that built the socket-case twin cut six stills by hand and
named them by what they showed, so nothing recorded *when* each was, and the
take could not be held against the clip at any moment but by eye. Here a
still is cut at a time and named by it, a contact sheet shows the whole clip
at a glance with the seconds burnt in, and ``timeline.json`` beside them is
the list of moments the twin is checked against: the first frame (before),
the last (after), and whatever the agent marks in between (the lid up, the
socket in). ``compare`` reads it to hold a recorded take against the clip
frame for frame.

ffmpeg is the one ``imageio-ffmpeg`` ships, which the library already depends
on for video, so this adds nothing to install.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

__all__ = ["TIMELINE", "contact_sheet", "cut", "extract", "parse_times", "probe", "read_timeline", "write_timeline"]

TIMELINE = "timeline.json"


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe(clip: str | Path) -> dict[str, Any]:
    """Frame rate, size and duration, read without decoding the clip."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(clip), "ffmpeg")
    try:
        meta = reader.get_meta_data()
    finally:
        reader.close()
    fps = float(meta.get("fps") or 30.0)
    width, height = (int(v) for v in meta.get("size") or (0, 0))
    duration = float(meta.get("duration") or 0.0)
    return {
        "clip": str(Path(clip).resolve()),
        "fps": fps,
        "width": width,
        "height": height,
        "duration": round(duration, 3),
        "frames": int(round(duration * fps)),
    }


def parse_times(text: str, duration: float = 0.0) -> list[tuple[float, str]]:
    """``"0,3.7:lid_up,end"`` -> ``[(0.0, ""), (3.7, "lid_up"), (duration, "")]``; ``end`` is the clip's last moment."""
    out: list[tuple[float, str]] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        when, _, label = item.partition(":")
        when = when.strip().lower()
        t = max(0.0, duration - 1e-3) if when == "end" else float(when)
        out.append((round(t, 3), label.strip()))
    return out


def _stem(t: float) -> str:
    return f"t{t:06.2f}"


def extract(clip: str | Path, times: list[float], out_dir: str | Path, *, width: int = 0) -> list[Path]:
    """One JPEG per time, ``t003.70.jpg`` for 3.7 s, at the clip's size or scaled to ``width``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = _ffmpeg()
    paths = []
    for t in times:
        path = out_dir / f"{_stem(t)}.jpg"
        # whether ffmpeg wrote a frame is read off the file, so a still left by an earlier clip must go first
        path.unlink(missing_ok=True)
        argv = [exe, "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(clip), "-frames:v", "1", "-q:v", "2"]
        if width:
            argv += ["-vf", f"scale={int(width)}:-2"]
        subprocess.run([*argv, str(path)], check=True, stdin=subprocess.DEVNULL)
        if not path.exists():
            # past the last frame: ffmpeg exits 0 and writes nothing, so step back one frame and ask again
            argv[5] = f"{max(0.0, t - 2.0 / _fps(clip)):.3f}"
            path.unlink(missing_ok=True)
            subprocess.run([*argv, str(path)], check=True, stdin=subprocess.DEVNULL)
            if not path.exists():
                raise ValueError(f"no frame at {t:.3f} s in {clip}")
        paths.append(path)
    return paths


def _fps(clip: str | Path) -> float:
    try:
        return float(probe(clip)["fps"]) or 30.0
    except Exception:
        return 30.0


def contact_sheet(
    clip: str | Path, every: float, out_path: str | Path, *, columns: int = 6, width: int = 320
) -> tuple[Path, int]:
    """The clip every ``every`` seconds, tiled, each tile stamped with its time. Returns the file and the tile count."""
    import tempfile

    from PIL import Image, ImageDraw, ImageFont

    meta = probe(clip)
    last = meta["duration"] - 1.0 / (meta["fps"] or 30.0)
    times = [round(i * every, 3) for i in range(int(meta["duration"] // every) + 1) if i * every <= last]
    with tempfile.TemporaryDirectory() as tmp:
        tiles = []
        for p in extract(clip, times, tmp, width=width):
            with Image.open(p) as opened:
                tiles.append(opened.convert("RGB"))
    if not tiles:
        raise ValueError(f"no frames in {clip}")
    tw, th = tiles[0].size
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tw, rows * th), (0, 0, 0))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=max(12, tw // 16))
    except TypeError:  # an older Pillow: one size only
        font = ImageFont.load_default()
    for i, (t, tile) in enumerate(zip(times, tiles, strict=False)):
        x, y = (i % columns) * tw, (i // columns) * th
        sheet.paste(tile, (x, y))
        text = f"{t:.1f}s"
        draw.rectangle((x, y, x + 8 + len(text) * (tw // 28), y + th // 9), fill=(0, 0, 0))
        draw.text((x + 4, y + 2), text, fill=(255, 255, 255), font=font)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=88)
    return out_path, len(tiles)


def read_timeline(ref_dir: str | Path) -> dict[str, Any]:
    path = Path(ref_dir) / TIMELINE
    return json.loads(path.read_text()) if path.exists() else {}


def write_timeline(ref_dir: str | Path, meta: dict[str, Any], anchors: list[dict[str, Any]]) -> Path:
    """Merge ``anchors`` (``t``, ``file``, ``label``) into the timeline beside the stills, sorted by time."""
    ref_dir = Path(ref_dir)
    current = read_timeline(ref_dir)
    if current.get("clip") and meta.get("clip") and str(current["clip"]) != str(meta["clip"]):
        current = {}  # another clip: its anchors are moments of a different recording
    # A fresh cut (new stills on disk) retires the anchors whose stills are gone: the clip's "end"
    # moves when it is measured better, and the old still at the old time has been deleted or never was.
    fresh = any(a.get("file") and (ref_dir / a["file"]).exists() for a in anchors)
    kept = {
        round(float(a["t"]), 3): a
        for a in current.get("anchors", [])
        if not fresh or not a.get("file") or Path(a["file"]).is_absolute() or (ref_dir / a["file"]).exists()
    }
    # a label marked again at another time moves that moment; the old one would be scored as a second
    relabelled = {str(a["label"]): round(float(a["t"]), 3) for a in anchors if a.get("label")}
    kept = {t: a for t, a in kept.items() if relabelled.get(str(a.get("label") or ""), t) == t}
    for anchor in anchors:
        t = round(float(anchor["t"]), 3)
        merged = dict(kept.get(t, {}))
        merged.update({k: v for k, v in anchor.items() if v not in ("", None)})
        merged["t"] = t
        kept[t] = merged
    timeline = {**current, **meta, "anchors": [kept[t] for t in sorted(kept)]}
    path = ref_dir / TIMELINE
    path.write_text(json.dumps(timeline, indent=1) + "\n")
    return path


def cut(
    clip: str | Path,
    *,
    at: str = "0,end",
    every: float = 0.0,
    out_dir: str | Path = "ref",
    width: int = 0,
) -> dict:
    """Stills out of the clip at the times asked, a contact sheet, and the timeline they anchor."""
    clip_path = Path(clip)
    if not clip_path.is_file():
        return {"ok": False, "error": f"no clip at {clip}"}
    out_dir = Path(out_dir)
    meta = probe(clip_path)
    if meta["duration"] <= 0:
        # ffmpeg read "Duration: N/A" (a raw .h264 stream, some webm): "end" would be 0 and no time can be sought
        return {
            "ok": False,
            "error": f"{clip} has no duration ffmpeg can read; re-mux it (ffmpeg -i {clip} -c copy out.mp4)",
        }
    times = parse_times(at, meta["duration"] - 1.0 / (meta["fps"] or 30.0))  # "end" is the last frame
    # a still is named to the hundredth, so the anchor is that hundredth: two times inside one name are one moment
    floored: dict[float, tuple[str, str]] = {}  # t -> (label, the time as asked)
    for t, label in times:
        moment = math.floor(t * 100 + 1e-6) / 100
        if moment not in floored:
            floored[moment] = (label, f"{t:g}")
            continue
        kept, asked = floored[moment]
        if label and kept and label != kept:
            return {
                "ok": False,
                "error": f"{asked}:{kept} and {t:g}:{label} fall in one still ({_stem(moment)}.jpg); "
                "mark them at least 0.01 s apart",
            }
        floored[moment] = (kept or label, asked)
    times = [(t, label) for t, (label, _) in floored.items()]
    stills_cut = extract(clip_path, [t for t, _ in times], out_dir, width=width)
    anchors = [{"t": t, "file": p.name, "label": label} for (t, label), p in zip(times, stills_cut, strict=True)]
    spacing = float(every) if every else max(0.5, round(meta["duration"] / 24.0, 1))
    sheet, tiles = contact_sheet(clip_path, spacing, out_dir / "contact.jpg")
    timeline = write_timeline(out_dir, meta, anchors)
    return {
        "ok": True,
        "clip": meta["clip"],
        "fps": meta["fps"],
        "size": f"{meta['width']}x{meta['height']}",
        "duration": meta["duration"],
        "frames": [str(p) for p in stills_cut],
        "contact": f"{sheet} ({tiles} tiles, every {spacing} s)",
        "timeline": str(timeline),
        "report": "\n".join(f"{a['t']:6.2f} s  {a['file']}  {a['label']}" for a in anchors),
    }
