# SPDX-License-Identifier: Apache-2.0
"""What a twin is held to before anyone says it matches: a still against the photograph, a take against the clip, a gate.

The page used to answer one question, "what does it look like beside the
photo", and the agent answered the rest by eye. Three more are answered here
with numbers, and every answer is written to ``run/log.jsonl`` with the hash
of the ``scene.py`` that produced it, so before and after an edit are two
lines apart rather than two impressions apart.

``compare_still``
    The scene as it stands, through a camera, against one photograph: per keyed
    thing, where it is in each and how big (:mod:`.compare`); overall, how much
    of the render's structure the photograph has.
``record`` and ``compare_take``
    A routine played on the page is recorded through a camera -- the video, a
    quarter-size segmentation of every frame, every thing's position -- and
    then held against the clip's ``timeline.json``: the first frame before
    anything happened, the last after everything did, and each moment the
    agent marked in between. What moved in the clip must have moved in the
    take, by about as much.
``gate``
    The checks that do not need a photograph: the file builds, the state is
    finite, the camera the clip was shot through is declared at the clip's
    aspect, every named thing exists, nothing is sinking into anything, and
    nothing drifts when the scene is left alone. A twin that fails these is
    not a twin yet, however it looks.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import compare as C
from . import stills

__all__ = [
    "LOG",
    "Recorder",
    "compare_still",
    "compare_take",
    "gate",
    "log_append",
    "log_read",
    "log_report",
    "scene_hash",
]

LOG = "log.jsonl"
SEG_SCALE = 2
"""A recorded take keeps its segmentation at half size: 700 frames of 640x360 ids compress to a few megabytes, and a 20 px part survives."""
MOVED_FRACTION = 0.02
"""A centre that shifts by more than this fraction of the frame width between two anchors has moved."""


# ------------------------------------------------------------------ the log
def scene_hash(path: str | Path) -> str:
    """A hash of the scene file and every ``.py`` beside it: a twin's camera often lives in a helper module."""
    path = Path(path)
    try:
        digest = hashlib.sha1(path.read_bytes())
    except OSError:
        return "missing"
    for sibling in sorted(path.parent.glob("*.py")):
        if sibling != path:
            try:
                digest.update(sibling.name.encode())
                digest.update(sibling.read_bytes())
            except OSError:
                continue
    return digest.hexdigest()[:10]


def next_number(run_dir: str | Path, prefix: str, seen: int = 0) -> int:
    """The number after the highest ``<prefix>_NNN*`` in ``run_dir`` and after ``seen``.

    The page restarts, and a counter that started again at 1 overwrote the
    agent's earlier pairs: nine compares in one session left four files. The
    files on disk are the count.
    """
    highest = int(seen)
    for path in Path(run_dir).glob(f"{prefix}_[0-9][0-9][0-9]*"):
        digits = path.name[len(prefix) + 1 : len(prefix) + 4]
        if digits.isdigit():
            highest = max(highest, int(digits))
    return highest + 1


def motion_summary(motion: dict[str, dict]) -> str:
    """One line on a take: which things did what the clip did, and which did not."""
    agreed, differed, missing = [], [], []
    for name, entry in motion.items():
        p, s = entry.get("photo"), entry.get("sim")
        if p is None or s is None:
            missing.append(name)
        elif p["moved"] == s["moved"]:
            agreed.append(name)
        else:
            differed.append(
                f"{name} {'stayed' if not s['moved'] else 'moved'} (clip {'moved' if p['moved'] else 'stayed'})"
            )
    parts = [f"{len(agreed)} of {len(motion)} did what the clip did"]
    if differed:
        parts.append("; ".join(differed))
    if missing:
        parts.append("not found at both ends: " + ", ".join(missing))
    return "; ".join(parts)


def log_append(run_dir: str | Path, entry: dict[str, Any]) -> int:
    path = Path(run_dir) / LOG
    n = 1
    if path.exists():
        with path.open() as f:
            n = sum(1 for _ in f) + 1
    row = {"n": n, "at": time.strftime("%H:%M:%S"), **entry}
    with path.open("a") as f:
        f.write(json.dumps(row, default=_plain) + "\n")
    return n


def log_read(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / LOG
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def log_report(rows: list[dict[str, Any]]) -> str:
    """One line per entry: when, what, which scene.py, the score, the note."""
    if not rows:
        return "nothing logged yet: compare, check and snapshot --note write here"
    out = ["  n  at        cmd       scene       score   note"]
    for r in rows:
        score = r.get("score")
        score = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
        passed = r.get("passed")
        if passed is not None:
            score = "PASS" if passed else "FAIL"
        out.append(
            f"{r['n']:3d}  {r['at']}  {r.get('cmd', ''):<8}  {r.get('scene', '')[:10]:<10}  {score:>6}  {r.get('note', '')}"
        )
    return "\n".join(out)


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


# ------------------------------------------------------------------ the photograph's side
def photo_masks(photo: np.ndarray, keys: list[C.Key]) -> dict[str, np.ndarray | None]:
    """Every key found in the photograph, at the photograph's size, by colour."""
    return {key.name: C.photo_mask(photo, key) for key in keys}


# ------------------------------------------------------------------ rendering with ids
def render_with_seg(sim: Any, scene: Any, *, camera: str = "", size: str = ""):
    """The scene through a camera: RGB, its segmentation, the id -> name table, and the camera used.

    ``rgb`` is the picture :meth:`~robosimstudio.Sim.render` draws, the one a
    snapshot shows. ``seg`` comes off the collider ray tracer through the same
    camera, the renderer that has it; it is internal, never a choice.
    """
    cam = stills.camera_for(sim, scene, name=camera, size=size)
    sim.sync()
    names = sim.scene.handles.segmentation_names()
    renderer = sim.make_renderer("tiled", cameras=[cam], channels=["seg"])
    try:
        seg = np.asarray(renderer.render(sim.state)[cam.name]["seg"]).astype(np.int32)
    finally:
        renderer.close()
    rgb = stills.render_still(sim, cam)
    return rgb, seg, names, cam


# ------------------------------------------------------------------ one still
def compare_still(
    rgb: np.ndarray,
    seg: np.ndarray,
    names: dict[int, str],
    photo_path: str | Path,
    keys: list[C.Key],
) -> dict:
    """Every keyed thing in the render against the photograph: its size, where it sits, how much it overlaps."""
    photo = C.load_photo(photo_path)
    h, w = rgb.shape[:2]
    fitted, stretch = C.fit_photo(photo, w, h)
    masks = C.sim_masks(seg, names, [k.name for k in keys])
    found = photo_masks(photo, keys)
    pairs: dict[str, tuple] = {}
    objects: dict[str, dict] = {}
    lines: list[str] = []
    for key in keys:
        pm = found[key.name]
        pm = C.fit_mask(pm, w, h) if pm is not None else None
        sm = masks[key.name]
        pairs[key.name] = (pm, sm)
        reading = C.compare_masks(pm, sm)
        objects[key.name] = reading
        lines.append(C.advice(key.name, reading, w, key))
    # a thing the photo shows and the render does not counts as no overlap, not as left out
    ious = [o["iou"] or 0.0 for o in objects.values() if o["photo"] is not None]
    score = round(float(np.mean(ious)), 4) if ious else 0.0
    if not keys:
        lines.append("no --key given: nothing was measured; the look sheet still shows the pair")
    if abs(stretch - 1.0) > 0.02:
        lines.append(
            f"the photo's aspect was stretched {stretch:.2f}x to fit the camera: give the camera the photo's width and height"
        )
    return {
        "objects": objects,
        "score": score,
        "stretch": round(stretch, 3),
        "lines": lines,
        "pairs": pairs,
        "fitted": fitted,
    }


def write_compare_pictures(base: Path, rgb: np.ndarray, result: dict) -> dict[str, str]:
    """The two pictures a comparison leaves, both for the agent's own eye.

    ``beside``: the render and the photograph side by side at full size, the
    first version's whole instrument and still the first thing to look at.
    ``look``: the sheet -- the pair; the photograph with every keyed thing
    outlined twice (the photo's outline yellow, the render's cyan); each thing
    cut from both pictures at the same scale; and both pictures with the
    things painted out, which is what stands behind them.
    """
    fitted = result["fitted"]
    beside = stills.save_png(np.concatenate([rgb, fitted], axis=1), base.with_name(base.stem + "_beside.png"))
    return {"beside": str(beside), "look": write_look_sheet(base, rgb, result)}


def write_look_sheet(base: Path, rgb: np.ndarray, result: dict) -> str:
    """The sheet, as :func:`write_compare_pictures` describes it."""
    fitted, pairs = result["fitted"], result["pairs"]
    rows: list[tuple[str, np.ndarray]] = [
        (
            "where each thing sits: photo outline yellow, render outline cyan",
            C.overlay(fitted, pairs),
        )
    ]
    for name, (pm, sm) in pairs.items():
        tile = C.crop_pair(fitted, rgb, pm, sm)
        if tile is not None:
            rows.append((f"{name} up close: photo | render", tile))
    things = None
    for pm, sm in pairs.values():
        for m in (pm, sm):
            if m is not None:
                things = m.copy() if things is None else (things | m)
    if things is not None:
        from PIL import Image

        h = 240
        tiles = [
            np.asarray(Image.fromarray(t).resize((int(round(t.shape[1] * h / t.shape[0])), h), Image.BILINEAR))
            for t in (C.without_things(fitted, things), C.without_things(rgb, things))
        ]
        gap = np.full((h, 6, 3), 40, np.uint8)
        rows.append(
            (
                "behind the things: photo | render",
                np.concatenate([tiles[0], gap, tiles[1]], axis=1),
            )
        )
    return str(stills.save_png(C.look_sheet(fitted, rgb, rows), base.with_name(base.stem + "_look.png")))


# ------------------------------------------------------------------ a take
class Recorder:
    """Frames, ids and positions off a script as the page plays it, written down when it ends."""

    def __init__(self, server: Any, name: str, *, camera: str = "", every: int = 4) -> None:
        from robosimstudio.core.render.video import VideoWriter

        self.server = server
        self.sim, self.scene = server.sim, server.scene
        self.name = name
        self.every = max(1, int(every))
        self.cam = stills.camera_for(self.sim, self.scene, name=camera)
        self.names = self.sim.scene.handles.segmentation_names()
        self.control_hz = float(getattr(getattr(self.sim.request, "physics", None), "control_hz", 60.0) or 60.0)
        self.fps = max(1, int(round(self.control_hz / self.every)))
        self.run_dir = Path(server.run_dir)
        self.video = self.run_dir / f"{name}.mp4"
        self.writer = VideoWriter(self.video, fps=self.fps)
        # The ids come off the collider ray tracer; the picture is what Sim.render draws.
        self.renderer = self.sim.make_renderer("tiled", cameras=[self.cam], channels=["seg"])
        self.frames: list[dict] = []
        self.segs: list[np.ndarray] = []
        self.last_step = -1
        self.started = time.perf_counter()
        self.step0 = int(self.sim.step_count)
        self.time0 = float(self.sim.time)

    def close(self) -> None:
        """Close the video writer and the id renderer; safe to call twice."""
        for thing in (self.writer, self.renderer):
            try:
                thing.close()
            except Exception:
                pass

    def maybe(self) -> None:
        step = int(self.sim.step_count)
        if self.last_step < 0 or step - self.last_step >= self.every:
            self.capture()

    def capture(self) -> None:
        sim = self.sim
        sim.sync()
        frame = self.renderer.render(sim.state)[self.cam.name]
        rgb = stills.render_still(sim, self.cam)
        seg = np.asarray(frame["seg"]).astype(np.int32)
        self.writer.append(rgb)
        self.segs.append(seg[::SEG_SCALE, ::SEG_SCALE].astype(np.uint8 if len(self.names) < 255 else np.int32))
        objects = {}
        for label, name in self.names.items():
            stats = C.mask_stats(seg == label)
            if stats is not None:
                objects[name] = {
                    "bbox": stats["bbox"],
                    "centre": stats["centre"],
                    "pixels": stats["pixels"],
                }
        positions = {}
        for name in sim.names():
            try:
                positions[name] = [round(float(v), 4) for v in sim[name].position]
            except Exception:
                pass
        self.last_step = int(sim.step_count)
        self.frames.append(
            {
                "i": len(self.frames),
                "step": self.last_step - self.step0,
                "t": round(float(sim.time) - self.time0, 4),
                "objects": objects,
                "positions": positions,
            }
        )

    def finish(self) -> dict:
        try:
            if self.frames and self.last_step != int(self.sim.step_count):
                self.capture()  # the state the script left, which is the "after"
        finally:
            # closed whatever happened above: the writer is an ffmpeg process
            self.close()
        meta = {
            "name": self.name,
            "camera": self.cam.name,
            "width": int(self.cam.width),
            "height": int(self.cam.height),
            "every": self.every,
            "fps": self.fps,
            "control_hz": self.control_hz,
            "video": str(self.video),
            "names": {str(k): v for k, v in self.names.items()},
            "seg_scale": SEG_SCALE,
            "wall_seconds": round(time.perf_counter() - self.started, 1),
            "frames": self.frames,
        }
        (self.run_dir / f"{self.name}.json").write_text(json.dumps(meta, default=_plain))
        if self.segs:
            np.savez_compressed(self.run_dir / f"{self.name}_seg.npz", seg=np.stack(self.segs))
        return {
            "take": self.name,
            "video": str(self.video),
            "frames": len(self.frames),
            "seconds": self.frames[-1]["t"] if self.frames else 0.0,
            "wall_seconds": meta["wall_seconds"],
        }


def load_take(run_dir: str | Path, name: str) -> tuple[dict, np.ndarray | None]:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / f"{name}.json").read_text())
    seg_path = run_dir / f"{name}_seg.npz"
    segs = None
    if seg_path.exists():
        with np.load(seg_path) as archive:
            segs = archive["seg"]
    return meta, segs


def _nearest_frame(frames: list[dict], t: float) -> dict:
    return min(frames, key=lambda f: abs(float(f["t"]) - t))


def compare_take(
    run_dir: str | Path,
    name: str,
    timeline: dict,
    keys: list[C.Key],
    ref_dir: str | Path,
) -> dict:
    """A recorded take against the clip's anchors: the first frame, the last, and each one marked between.

    An anchor is matched to the take by its own ``sim_t`` (seconds into the
    take) or ``sim_step`` when the agent pinned one, else by proportion: the
    clip's moment at 40 % of its length is held against the take at 40 % of
    its length. The clip and the take need not be the same length -- a robot
    is slower than a person -- but what moved in one must have moved in the
    other.
    """
    import imageio.v2 as imageio

    run_dir, ref_dir = Path(run_dir), Path(ref_dir)
    meta, segs = load_take(run_dir, name)
    frames = meta["frames"]
    if not frames:
        raise ValueError(f"take {name!r} has no frames")
    names = {int(k): v for k, v in meta["names"].items()}
    w, h = int(meta["width"]), int(meta["height"])
    anchors = list(timeline.get("anchors", []))
    if not anchors:
        raise ValueError("the timeline has no anchors; `ref frames <clip> --at 0,end` makes two")
    clip_len = float(timeline.get("duration") or anchors[-1]["t"] or 1.0)
    take_len = float(frames[-1]["t"]) or 1.0
    reader = imageio.get_reader(str(meta["video"]), "ffmpeg")
    rows: list[dict] = []
    tiles: list[tuple[np.ndarray, np.ndarray, str]] = []
    try:
        last = reader.count_frames() - 1
        for anchor in anchors:
            t_ref = float(anchor["t"])
            if anchor.get("sim_step") is not None:
                frame = min(frames, key=lambda f: abs(int(f["step"]) - int(anchor["sim_step"])))
            elif anchor.get("sim_t") is not None:
                frame = _nearest_frame(frames, float(anchor["sim_t"]))
            else:
                frame = _nearest_frame(frames, t_ref / clip_len * take_len)
            i = int(frame["i"])
            rgb = np.asarray(reader.get_data(min(i, last)))[..., :3].astype(np.uint8)
            seg = None
            if segs is not None and i < len(segs):
                seg = np.asarray(segs[i]).astype(np.int32)
            photo_path = ref_dir / anchor["file"] if not Path(anchor["file"]).is_absolute() else Path(anchor["file"])
            photo = C.load_photo(photo_path)
            fitted, _ = C.fit_photo(photo, w, h)
            objects: dict[str, dict] = {}
            pairs: dict[str, tuple] = {}
            found = photo_masks(photo, keys)
            masks = C.sim_masks(seg, names, [k.name for k in keys]) if seg is not None else {}
            for key in keys:
                pm = found[key.name]
                pm = C.fit_mask(pm, w, h) if pm is not None else None
                sm = masks.get(key.name)
                sm = C.fit_mask(sm, w, h) if sm is not None else None
                reading = C.compare_masks(pm, sm)
                objects[key.name] = reading
                pairs[key.name] = (pm, sm)
            rows.append(
                {
                    "t": t_ref,
                    "label": anchor.get("label", ""),
                    "file": str(photo_path),
                    "frame": i,
                    "sim_t": frame["t"],
                    "sim_step": frame["step"],
                    "objects": objects,
                }
            )
            label = f"{anchor.get('label') or ''} {t_ref:.1f}s / sim {float(frame['t']):.1f}s".strip()
            tiles.append(
                (
                    C.overlay(fitted, {k: (p, None) for k, (p, s) in pairs.items()}),
                    C.overlay(rgb, {k: (None, s) for k, (p, s) in pairs.items()}),
                    label,
                )
            )
    finally:
        reader.close()
    # what moved, in each
    motion: dict[str, dict] = {}
    lines: list[str] = []
    for key in keys:
        p_first, p_last = (
            rows[0]["objects"][key.name]["photo"],
            rows[-1]["objects"][key.name]["photo"],
        )
        s_first, s_last = (
            rows[0]["objects"][key.name]["sim"],
            rows[-1]["objects"][key.name]["sim"],
        )
        entry: dict[str, Any] = {}
        for side, a, b in (("photo", p_first, p_last), ("sim", s_first, s_last)):
            if a is None or b is None:
                entry[side] = None
                continue
            dx, dy = b["centre"][0] - a["centre"][0], b["centre"][1] - a["centre"][1]
            entry[side] = {
                "dx": round(dx, 1),
                "dy": round(dy, 1),
                "moved": bool(np.hypot(dx, dy) > MOVED_FRACTION * w),
            }
        motion[key.name] = entry
        p, s = entry["photo"], entry["sim"]
        if p is None or s is None:
            lines.append(f"{key.name}: not found at both ends in {'the photo' if p is None else 'the take'}")
        elif p["moved"] != s["moved"]:
            lines.append(
                f"{key.name}: {'moved' if p['moved'] else 'stayed'} in the clip but {'moved' if s['moved'] else 'stayed'} in the take"
            )
        else:
            lines.append(
                f"{key.name}: {'moved' if p['moved'] else 'stayed'} in both; clip ({p['dx']:+.0f}, {p['dy']:+.0f}) px, take ({s['dx']:+.0f}, {s['dy']:+.0f}) px"
            )
    ious = [o["iou"] or 0.0 for r in rows for o in r["objects"].values() if o["photo"] is not None]
    score = round(float(np.mean(ious)), 4) if ious else 0.0
    for r in rows:
        per = ", ".join(
            f"{n} IoU {o['iou']:.2f}" if o["iou"] is not None else f"{n} missing" for n, o in r["objects"].items()
        )
        lines.append(
            f"{r['label'] or 'anchor':<12} clip {r['t']:5.1f}s = take {r['sim_t']:5.1f}s (step {r['sim_step']}): {per}"
        )
    return {
        "take": name,
        "anchors": rows,
        "motion": motion,
        "score": score,
        "lines": lines,
        "tiles": tiles,
    }


def write_strip(base: Path, tiles: list[tuple[np.ndarray, np.ndarray, str]], tile_width: int = 426) -> str:
    """Two rows: the clip's anchors above, the take's matching frames below, each pair labelled."""
    from PIL import Image, ImageDraw

    if not tiles:
        return ""
    th = int(round(tiles[0][0].shape[0] * tile_width / tiles[0][0].shape[1]))
    sheet = Image.new("RGB", (tile_width * len(tiles), 2 * th + 18), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, (top, bottom, label) in enumerate(tiles):
        x = i * tile_width
        sheet.paste(Image.fromarray(top).resize((tile_width, th)), (x, 18))
        sheet.paste(Image.fromarray(bottom).resize((tile_width, th)), (x, 18 + th))
        draw.text((x + 4, 3), label, fill=(255, 255, 255))
    path = base.with_name(base.stem + "_strip.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return str(path)


# ------------------------------------------------------------------ the gate
def gate(
    server: Any,
    *,
    camera: str = "",
    keys: list[C.Key] | None = None,
    objects: list[str] | None = None,
    steps: int = 60,
    settle_mm: float = 2.0,
    penetration_mm: float = 1.0,
    timeline: dict | None = None,
) -> dict:
    """What a photograph cannot tell: the scene builds, stays finite, has its camera and its things, and holds still."""
    sim = server.sim
    keys = list(keys or [])
    wanted = list(objects or []) or [k.name for k in keys]
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    # only a build's own failure: a routine, a click or a take that failed leaves last_error but not this
    build_error = server.build_error
    add(
        "builds",
        not build_error,
        "the last rebuild landed" if not build_error else build_error.strip().splitlines()[-1],
    )
    finite = bool(sim.state_is_finite())
    add("finite", finite, "every number in the state is finite" if finite else "the state has gone nan")

    cameras = {c.name: c for c in sim.cameras}
    cam = cameras.get(camera) if camera else (next(iter(cameras.values())) if cameras else None)
    if cam is None:
        add(
            "camera",
            False,
            f"no camera {camera!r} declared; the scene has {sorted(cameras) or 'none'}",
        )
    else:
        detail = f"{cam.name} {cam.width}x{cam.height}, fov {cam.fov_y:.0f} deg"
        ok = True
        if timeline and timeline.get("width") and timeline.get("height"):
            clip_aspect = float(timeline["width"]) / float(timeline["height"])
            cam_aspect = float(cam.width) / float(cam.height)
            ok = abs(clip_aspect / cam_aspect - 1.0) <= 0.02
            detail += f"; clip {timeline['width']}x{timeline['height']}" + (
                "" if ok else f" -- aspect off by {100 * abs(clip_aspect / cam_aspect - 1):.0f}%"
            )
        add("camera", ok, detail)

    have = set(sim.names())
    missing = [n for n in wanted if n not in have]
    add(
        "objects",
        not missing,
        f"all of {', '.join(wanted)} are in the scene"
        if wanted and not missing
        else (f"missing: {', '.join(missing)}" if missing else "no names given to look for"),
    )

    # penetration, as the scene stands
    try:
        worst_mm, pairs = _penetration(sim)
    except Exception as e:  # a probe that cannot answer is a failed line, not a gate with no report
        add("penetration", False, f"the contact probe raised {type(e).__name__}: {e}")
    else:
        add(
            "penetration",
            worst_mm <= penetration_mm,
            f"worst overlap {worst_mm:.2f} mm" + (f" between {pairs[0]}" if pairs else ""),
        )

    # rest: leave the scene alone and see what drifts
    before = _positions(sim, wanted or list(have))
    sim.run(int(steps))
    sim.sync()
    after = _positions(sim, list(before))
    drift = {
        n: 1000.0 * float(np.linalg.norm(np.asarray(after[n]) - np.asarray(before[n]))) for n in before if n in after
    }
    worst = max(drift.items(), key=lambda kv: kv[1]) if drift else ("", 0.0)
    add(
        "rest",
        worst[1] <= settle_mm,
        f"nothing moved more than {worst[1]:.1f} mm in {steps} steps"
        if worst[1] <= settle_mm
        else f"{worst[0]} drifted {worst[1]:.1f} mm in {steps} steps left alone",
    )
    finite = bool(sim.state_is_finite())
    add("finite_after", finite, f"still finite after {steps} steps" if finite else f"went nan within {steps} steps")

    passed = all(c["ok"] for c in checks)
    report = "\n".join(f"{'PASS' if c['ok'] else 'FAIL'}  {c['name']:<12} {c['detail']}" for c in checks)
    return {
        "passed": passed,
        "checks": checks,
        "report": report,
        "drift_mm": {k: round(v, 2) for k, v in drift.items()},
    }


def _positions(sim: Any, names: list[str]) -> dict[str, list[float]]:
    out = {}
    for name in names:
        try:
            out[name] = [float(v) for v in sim[name].position]
        except Exception:
            continue
    return out


def _penetration(sim: Any) -> tuple[float, list[str]]:
    from robosimstudio.scene.snapshot import penetration_rows

    rows = penetration_rows(sim)
    if not rows:
        return 0.0, []
    worst = rows[0]
    return float(worst["depth_mm"]), [" and ".join(worst["between"])]
