# SPDX-License-Identifier: Apache-2.0
"""What a twin is held to before anyone says it matches: a still against the photograph, a take against the clip, a gate.

The eye on a picture is the instrument: every reading here makes the pictures
an agent judges, and adds a number only where one is exact. Every reading is
written to ``run/log.jsonl`` with the hash of the ``scene.py`` that produced
it and the picture it left, so before and after an edit are two lines apart.

``compare_still``
    The scene as it stands, through a camera, beside one photograph, as a pair
    and a look sheet; per keyed thing, where it is in each and how big
    (:mod:`.compare`).
``record`` and ``compare_take``
    A routine played on the page is recorded through a camera -- the video, a
    segmentation of every frame at half size each way, every thing's position -- and
    then put beside the clip's ``timeline.json``: the first frame before
    anything happened, the last after everything did, and each moment the
    agent marked in between, in a strip for the eye; and what moved in the
    take, exactly, off the recorded positions.
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
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import compare as C
from . import stills
from .scenefile import helper_files

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
TAKE_STILL_MM = 2.0
"""A thing whose centre never went farther than this from where the take started it stayed put."""


# ------------------------------------------------------------------ the log
def scene_hash(path: str | Path, files: list[Path] | None = None) -> str:
    """A hash of the scene file and its helpers, the files a save to which rebuilds
    (:func:`.scenefile.helper_files`): a twin's camera often lives in a helper module. The page takes it as each
    build reads the files, and the log ties a reading to the scene it measured by it. ``files`` are the helpers,
    when the caller has just walked for them."""
    path = Path(path)
    try:
        digest = hashlib.sha1(path.read_bytes())
    except OSError:
        return "missing"
    for helper in helper_files(path) if files is None else files:
        try:
            digest.update(str(helper.relative_to(path.resolve().parent)).encode())
            digest.update(helper.read_bytes())
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
    """One line per entry: when, what, which scene.py, what it left (PASS/FAIL, the picture or the take), the note.

    The picture is the thing to cite: a report that says it judged a look
    sheet names ``cmp_012_look.png``, and this line ties that sheet to the
    ``scene.py`` it measured.
    """
    if not rows:
        return "nothing logged yet: compare, check, a take and snapshot --note write here"
    out = ["  n  at        cmd       scene       left                      note"]
    for r in rows:
        passed = r.get("passed")
        if passed is not None:
            left = "PASS" if passed else "FAIL"
        else:
            left = next((Path(str(r[k])).name for k in ("strip", "look", "png", "video") if r.get(k)), "-")
        out.append(
            f"{r['n']:3d}  {r['at']}  {r.get('cmd', ''):<8}  {r.get('scene', '')[:10]:<10}  {left:<24}  {r.get('note', '')}"
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
    camera, the renderer that has it; it is internal, never a choice. Its
    deformables are named too (:func:`name_deformables`).
    """
    cam = stills.camera_for(sim, scene, name=camera, size=size)
    sim.sync()
    names, labels = seg_names(sim)
    # Named, not the platform's pick: ``seg`` is a channel only ``tiled`` makes, and it makes it on any device.
    renderer = sim.make_renderer("tiled", cameras=[cam], channels=["seg"])
    try:
        seg = np.asarray(renderer.render(sim.state)[cam.name]["seg"]).astype(np.int32)
    finally:
        renderer.close()
    seg = name_deformables(sim, cam, seg, labels)
    rgb = stills.render_still(sim, cam)
    return rgb, seg, names, cam


def seg_names(sim: Any) -> tuple[dict[int, str], dict[str, int]]:
    """``label -> name`` for a segmentation, the scene's deformables included, and the deformables' own labels.

    The handles name every rigid thing; a deformable has no shape, so it gets
    a label after theirs, in name order, so a take's ids mean the same thing
    from its first frame to its last.
    """
    names = dict(sim.scene.handles.segmentation_names())
    labels: dict[str, int] = {}
    nxt = max(names, default=0) + 1
    for name in sorted(getattr(sim.scene.handles, "deformables", {}) or {}):
        names[nxt] = name
        labels[name] = nxt
        nxt += 1
    return names, labels


def name_deformables(
    sim: Any, cam: Any, seg: np.ndarray, labels: dict[str, int], q: np.ndarray | None = None
) -> np.ndarray:
    """``seg`` with each particle or deformable-mesh hit given the label of the deformable it belongs to.

    The ray tracer reports such a hit by kind -- a particle, or the one
    triangle mesh every deformable shares -- not by thing. With one deformable
    of that kind in the scene, the hit is its. With several, each is projected
    into the picture by its particles and a hit goes to the deformable whose
    particles are drawn nearest that pixel. A hit no deformable can claim stays
    unnamed (0). ``q`` is the particles' positions when the caller has read them.
    """
    from robosimstudio.solvers.render.tiled import SEG_PARTICLES, SEG_SOFT_MESH

    reserved = seg < 0
    if not reserved.any():
        return seg
    out = np.where(reserved, 0, seg).astype(np.int32)
    handles = getattr(sim.scene.handles, "deformables", {}) or {}
    if not labels or not handles:
        return out
    meshed = {n for n in labels if len(getattr(handles.get(n), "triangles", ())) > 0}
    for code, kind in ((SEG_SOFT_MESH, meshed), (SEG_PARTICLES, set(labels) - meshed)):
        hit = seg == code
        if not hit.any():
            continue
        owners = sorted(kind) or sorted(labels)  # a hit of the other kind than expected still goes to a deformable
        if len(owners) == 1:
            out[hit] = labels[owners[0]]
            continue
        if q is None:
            q = _particles(sim)
        nearest = _nearest_owner(sim, cam, hit, {n: labels[n] for n in owners}, handles, q)
        if nearest is not None:
            out[hit] = nearest
    return out


def _particles(sim: Any) -> np.ndarray:
    """Every particle's position, ``(n, 3)``, read once; empty for a scene with none."""
    try:
        return np.asarray(sim.state.particle_q.numpy(), dtype=np.float64)
    except Exception:
        return np.zeros((0, 3))


def _indices(handle: Any, count: int) -> np.ndarray:
    """A deformable's particle indices that are in range."""
    idx = np.asarray(getattr(handle, "particles", ()), dtype=np.int64)
    return idx[(idx >= 0) & (idx < count)]


def _nearest_owner(
    sim: Any, cam: Any, hit: np.ndarray, labels: dict[str, int], handles: dict, q: np.ndarray
) -> np.ndarray | None:
    """For each pixel in ``hit``, the label of the deformable whose projected particle is nearest; None when none
    projects. Looked up over the hit pixels alone, so a frame costs what its deformables cover."""
    from scipy.spatial import cKDTree

    h, w = hit.shape
    pose = None
    if getattr(cam, "parent_body", -1) >= 0 and getattr(sim.state, "body_q", None) is not None:
        pose = cam.pose_on_body(sim.state.body_q.numpy())
    xs, ys, zs, ls = [], [], [], []
    for name, label in labels.items():
        idx = _indices(handles[name], len(q))
        if not len(idx):
            continue
        uv, z = cam.project(q[idx], camera_to_world=pose)
        with np.errstate(invalid="ignore"):
            x, y = np.floor(uv[:, 0]), np.floor(uv[:, 1])
        keep = (z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        xs.append(x[keep].astype(np.int64))
        ys.append(y[keep].astype(np.int64))
        zs.append(z[keep])
        ls.append(np.full(int(keep.sum()), label, np.int32))
    if not xs or not sum(len(x) for x in xs):
        return None
    x, y, z, lab = (np.concatenate(a) for a in (xs, ys, zs, ls))
    pixel = y * w + x
    order = np.lexsort((z, pixel))  # by pixel, nearest first within one
    first = np.unique(pixel[order], return_index=True)[1]  # the nearest particle at each pixel drawn on
    drawn, owner = pixel[order][first], lab[order][first]
    hy, hx = np.nonzero(hit)
    _, nearest = cKDTree(np.stack([drawn % w, drawn // w], axis=1)).query(np.stack([hx, hy], axis=1))
    return owner[nearest]


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
    if not keys:
        lines.append(
            "no --key: the pair and the look sheet are written; key a thing that keys cleanly to read its size, place and overlap"
        )
    if abs(stretch - 1.0) > 0.02:
        lines.append(
            f"the photo's aspect was stretched {stretch:.2f}x to fit the camera: give the camera the photo's width and height"
        )
    return {
        "objects": objects,
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
    rows: list[tuple[str, np.ndarray]] = []
    if pairs:
        rows.append(
            ("where each thing sits, on the photo: photo outline yellow, render outline cyan", C.overlay(fitted, pairs))
        )
    for name, (pm, sm) in pairs.items():
        tile = C.crop_pair(fitted, rgb, pm, sm)
        if tile is not None:
            rows.append((f"{name} up close: render | photo", tile))
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
            for t in (C.without_things(rgb, things), C.without_things(fitted, things))
        ]
        gap = np.full((h, 6, 3), 40, np.uint8)
        rows.append(
            (
                "behind the things: render | photo",
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
        self.names, self.deformable_labels = seg_names(self.sim)
        self.control_hz = float(getattr(getattr(self.sim.request, "physics", None), "control_hz", 60.0) or 60.0)
        self.fps = max(1, int(round(self.control_hz / self.every)))
        self.run_dir = Path(server.run_dir)
        self.video = self.run_dir / f"{name}.mp4"
        self.writer = VideoWriter(self.video, fps=self.fps)
        # The ids come off the collider ray tracer, named because only it makes ``seg``, on
        # any device; the picture is what Sim.render draws, on the platform's renderer.
        self.renderer = self.sim.make_renderer("tiled", cameras=[self.cam], channels=["seg"])
        self.frames: list[dict] = []
        self.segs: list[np.ndarray] = []
        self.started = time.perf_counter()
        self.steps = 0
        """Control steps the take has seen, counted across a reset the routine makes under it."""
        self._raw = int(self.sim.step_count)
        self._mark = reset_mark(self.sim)
        try:  # read once: a body's centre of mass does not move in it, nor does a thing change its bodies
            self.com = np.asarray(self.sim.model.body_com.numpy(), dtype=np.float64)
        except Exception:
            self.com = None
        self.bodies = take_bodies(self.sim)
        self.last_capture = -1
        self._probed: tuple | None = None
        """The probe's step count, reset mark and frame: :meth:`start`'s frame 0 when the scene has not stepped since."""

    def _look(self) -> tuple[np.ndarray, np.ndarray, Any]:
        """The scene as the take films it now: the picture, the named ids, and the particles read for both."""
        sim = self.sim
        sim.sync()
        frame = self.renderer.render(sim.state)[self.cam.name]
        rgb = stills.render_still(sim, self.cam)
        q = _particles(sim) if self.deformable_labels else None  # once a frame, for the ids and the samples both
        seg = name_deformables(sim, self.cam, np.asarray(frame["seg"]).astype(np.int32), self.deformable_labels, q)
        return rgb, seg, q

    def probe(self) -> None:
        """Film one frame, before anything the take would replace is touched: a camera, a renderer or an id lookup
        that fails, fails here, and so does a video the encoder cannot make (ffmpeg is started on frame 0, and its
        own errors arrive after that). The frame is kept for :meth:`start`."""
        from robosimstudio.core.render.video import ffmpeg_exe

        look = self._look()
        height, width = np.asarray(look[0]).shape[:2]
        if height % 2 or width % 2:
            raise ValueError(
                f"camera {self.cam.name!r} films {width}x{height}: a take's video needs an even width and height "
                "(libx264 in yuv420p)"
            )
        exe = ffmpeg_exe()  # imageio-ffmpeg's raises when it has none
        if shutil.which(exe) is None:
            raise FileNotFoundError(f"no ffmpeg at {exe!r} to encode the take")
        self._probed = (int(self.sim.step_count), reset_mark(self.sim), look)

    def start(self) -> None:
        """Take the scene as it stands as the take's beginning: its step count from here, and frame 0 -- the
        probe's, when the scene has not stepped since, rather than the same picture rendered again."""
        self._raw, self._mark, self.steps = int(self.sim.step_count), reset_mark(self.sim), 0
        self.started = time.perf_counter()
        probed, self._probed = self._probed, None
        self.capture(probed[2] if probed is not None and probed[:2] == (self._raw, self._mark) else None)

    def close(self) -> None:
        """Close the video writer and the id renderer; safe to call twice."""
        for thing in (self.writer, self.renderer):
            try:
                thing.close()
            except Exception:
                pass

    def _advance(self) -> None:
        """Count the steps taken since the last look, a reset under the take counted from 0 (:func:`steps_since`)."""
        self.steps += steps_since(self.sim, self._raw, self._mark)
        self._raw, self._mark = int(self.sim.step_count), reset_mark(self.sim)

    def maybe(self) -> None:
        self._advance()
        if self.last_capture < 0 or self.steps - self.last_capture >= self.every:
            self.capture()

    def capture(self, look: tuple | None = None) -> None:
        """Film the scene as it stands, or as ``look`` (:meth:`_look`'s) already filmed it."""
        sim = self.sim
        self._advance()
        rgb, seg, q = look if look is not None else self._look()
        self.writer.append(rgb)
        self.segs.append(seg[::SEG_SCALE, ::SEG_SCALE].astype(np.uint8 if len(self.names) < 255 else np.int32))
        try:
            body_q = np.asarray(sim.state.body_q.numpy(), dtype=np.float64)
        except Exception:
            body_q = None
        positions = {}
        for name in sim.names():
            first = (self.bodies.get(name) or [None])[0]
            if body_q is not None and first is not None and 0 <= first < len(body_q):
                positions[name] = [round(float(v), 4) for v in body_q[first, :3]]  # its first body, as Ref reads it
                continue
            try:
                positions[name] = [round(float(v), 4) for v in sim[name].position]
            except Exception:
                pass
        self.last_capture = self.steps
        self.frames.append(
            {
                "i": len(self.frames),
                "step": self.steps,
                "t": round(self.steps / self.control_hz, 4),
                "positions": positions,
                "parts": _part_positions(sim, self.com, self.bodies, q, body_q),
            }
        )

    def finish(self) -> dict:
        try:
            self._advance()
            if self.frames and self.steps != self.last_capture:
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


PARTS_KEPT = 16
"""Bodies a take keeps per jointed thing; a chain of more is sampled along its length."""


def reset_mark(sim: Any) -> int:
    """Something that changes on every ``Sim.reset``: its count of them."""
    return int(getattr(sim, "reset_count", 0))


def steps_since(sim: Any, step: int, mark: int) -> int:
    """Control steps taken since ``step`` was read with ``mark``; after a reset, the steps since the reset."""
    now = int(sim.step_count)
    if reset_mark(sim) != mark or now < step:
        return now
    return now - step


def take_bodies(sim: Any) -> dict[str, list[int]]:
    """The bodies a take follows per thing, every one or :data:`PARTS_KEPT` of a long chain: read once a take."""
    handles = getattr(getattr(sim, "scene", None), "handles", None)
    out: dict[str, list[int]] = {}
    for name in [*getattr(handles, "objects", {}), *getattr(handles, "statics", {})] if handles is not None else []:
        try:
            handle = handles.entity(name)
            bodies = tuple(getattr(handle, "bodies", ())) or (
                (handle.body,) if getattr(handle, "body", -1) >= 0 else ()
            )
        except Exception:  # one odd handle costs that thing its parts, not every thing
            continue
        if bodies:
            keep = np.unique(np.linspace(0, len(bodies) - 1, min(len(bodies), PARTS_KEPT)).round().astype(int))
            out[name] = [bodies[i] for i in keep]
    return out


def _part_positions(
    sim: Any,
    com: np.ndarray | None = None,
    bodies: dict[str, list[int]] | None = None,
    q: np.ndarray | None = None,
    body_q: np.ndarray | None = None,
) -> dict[str, list[list[float]]]:
    """Every body of every thing -- a case and its lid, a lock and its key -- where its centre of mass is now.

    A thing's position is its first body's origin, so a jointed thing whose
    base stays put reads as still while its key turns, and a lid whose origin
    is its hinge reads as still while it opens. A centre of mass moves with
    either. A chain of many links keeps :data:`PARTS_KEPT` of them, evenly
    along it; each deformable, as many of its particles. ``com``, ``bodies``
    (:func:`take_bodies`) and the particles ``q`` are what a take reads once;
    left out, they are read here. Best effort: a take is never lost for want
    of these.
    """
    from robosimstudio.engines.pose import compose_many

    handles = getattr(getattr(sim, "scene", None), "handles", None)
    if handles is None:
        return {}
    out = _deformable_samples(sim, handles, q)
    bodies = take_bodies(sim) if bodies is None else bodies
    if not bodies:
        return out
    try:
        body_q = np.asarray(sim.state.body_q.numpy() if body_q is None else body_q, dtype=np.float64)
        com = np.asarray(sim.model.body_com.numpy() if com is None else com, dtype=np.float64)
        needed = sorted({b for idx in bodies.values() for b in idx if 0 <= b < min(len(body_q), len(com))})
        # each body's centre of mass, not its origin: a lid's origin sits on its hinge and does not move as it opens
        local = np.hstack([com[needed], np.tile([0.0, 0.0, 0.0, 1.0], (len(needed), 1))])
        where = dict(zip(needed, compose_many(body_q[needed], local)[:, :3], strict=True))
    except Exception:
        return out
    out.update(
        {name: [[round(float(v), 4) for v in where[b]] for b in idx if b in where] for name, idx in bodies.items()}
    )
    return out


def _deformable_samples(sim: Any, handles: Any, q: np.ndarray | None = None) -> dict[str, list[list[float]]]:
    """:data:`PARTS_KEPT` particles of each deformable, evenly through it: a knot tied or a bag opened in place
    leaves its centre where it was."""
    deformables = getattr(handles, "deformables", None) or {}
    if not deformables:
        return {}
    q = _particles(sim) if q is None else q
    out = {}
    for name, handle in deformables.items():
        idx = _indices(handle, len(q))
        if len(idx):
            pick = idx[np.unique(np.linspace(0, len(idx) - 1, min(len(idx), PARTS_KEPT)).round().astype(int))]
            out[name] = [[round(float(v), 4) for v in q[i, :3]] for i in pick]
    return out


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


def take_motion(frames: list[dict]) -> dict[str, dict[str, float]]:
    """What moved in a take, read off the positions it recorded: per thing, where it ended and how far it went [mm].

    Both, because one alone misleads: a lid that swung open and shut again
    ends where it started, and a box nudged a few millimetres by a passing
    hand went somewhere and came back. A thing of several parts -- a case's
    bodies, a deformable's particles -- gives each the most any part did, so
    the key turning in a case whose base stays is seen.
    """
    if not frames:
        return {}
    out: dict[str, dict[str, float]] = {}
    # a deformable has no position, only the particles sampled for it
    for name in dict.fromkeys([*frames[0].get("positions", {}), *frames[0].get("parts", {})]):
        tracks = [[f["positions"][name] for f in frames if name in f.get("positions", {})]]
        parts = [f.get("parts", {}).get(name) for f in frames]
        if parts[0]:
            for b in range(len(parts[0])):
                tracks.append([p[b] for p in parts if p and b < len(p)])
        ended = farthest = 0.0
        for track in tracks:
            if not track:
                continue
            path = np.asarray(track, dtype=np.float64)
            moved = np.linalg.norm(path - path[0], axis=1)
            # each the most any of its parts did: one part may have swung out and back while another stayed moved
            ended, farthest = max(ended, float(moved[-1])), max(farthest, float(moved.max()))
        out[name] = {"ended_mm": round(1000.0 * ended, 1), "farthest_mm": round(1000.0 * farthest, 1)}
    return out


def motion_line(moved: dict[str, dict[str, float]], still_mm: float = TAKE_STILL_MM) -> str:
    """One line on a take: what moved and how, and how many things stayed put."""
    went = sorted(
        ((n, m) for n, m in moved.items() if m["farthest_mm"] > still_mm),
        key=lambda nm: -nm[1]["farthest_mm"],
    )
    parts = [
        f"{n} went {m['farthest_mm']:.0f} mm"
        + (
            f" and ended {m['ended_mm']:.0f} mm from its start"
            if abs(m["ended_mm"] - m["farthest_mm"]) > still_mm
            else ""
        )
        for n, m in went
    ]
    stayed = len(moved) - len(went)
    if stayed:
        parts.append(f"{stayed} other thing{'s' if stayed != 1 else ''} stayed within {still_mm:.0f} mm")
    return "in the take: " + ("; ".join(parts) if parts else "nothing recorded a position")


def compare_take(
    run_dir: str | Path,
    name: str,
    timeline: dict,
    keys: list[C.Key],
    ref_dir: str | Path,
    *,
    moved: dict | None = None,
) -> dict:
    """A recorded take beside the clip: each anchor's still above the take's frame at the same point of the action.

    An anchor is matched to the take by its own ``sim_t`` (seconds into the
    take) or ``sim_step`` when the agent pinned one, else by proportion: the
    clip's moment at 40 % of its length is held against the take at 40 % of
    its length. The clip and the take need not be the same length -- a robot
    is slower than a person.

    The strip is for the eye, which is what reads a clip: what moved in it,
    what touched what. A colour key cannot -- a handheld phone or a cut moves
    everything -- so nothing here says whether the take did what the clip did.
    What the take did is exact, off its recorded positions (:func:`take_motion`).
    Keys, when given, only outline the things: a colour key in both rows; a
    seed or a mask, drawn on one still, in the take's row alone, since the thing
    may have moved by the next anchor.
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
    lines: list[str] = []
    try:
        last = reader.count_frames() - 1
        for anchor in anchors:
            t_ref = float(anchor["t"])
            pinned_step, pinned_t = anchor.get("sim_step"), anchor.get("sim_t")
            if (pinned_step is not None and int(pinned_step) > int(frames[-1]["step"])) or (
                pinned_t is not None and float(pinned_t) > take_len
            ):
                lines.append(
                    f"{anchor.get('label') or 'anchor'} at {t_ref:.1f}s is pinned past the take's end "
                    f"(step {frames[-1]['step']}, {take_len:.1f}s): it is held against the last frame"
                )
            if anchor.get("sim_step") is not None:
                frame = min(frames, key=lambda f: abs(int(f["step"]) - int(anchor["sim_step"])))
            elif anchor.get("sim_t") is not None:
                frame = _nearest_frame(frames, float(anchor["sim_t"]))
            else:
                frame = _nearest_frame(frames, t_ref / clip_len * take_len)
            i = int(frame["i"])
            rgb = np.asarray(reader.get_data(min(i, last)))[..., :3].astype(np.uint8)
            photo_path = ref_dir / anchor["file"] if not Path(anchor["file"]).is_absolute() else Path(anchor["file"])
            photo = C.load_photo(photo_path)
            fitted, _ = C.fit_photo(photo, w, h)
            top, bottom = fitted, rgb
            if keys:
                seg = np.asarray(segs[i]).astype(np.int32) if segs is not None and i < len(segs) else None
                # a seed or a mask keys the one still it was drawn on: outlined in the take's row alone
                found = photo_masks(photo, [k for k in keys if k.seed is None and not k.mask])
                masks = C.sim_masks(seg, names, [k.name for k in keys]) if seg is not None else {}
                outlined_photo, outlined_take = {}, {}
                for key in keys:
                    pm = C.fit_mask(found[key.name], w, h) if found.get(key.name) is not None else None
                    sm = C.fit_mask(masks[key.name], w, h) if masks.get(key.name) is not None else None
                    outlined_photo[key.name], outlined_take[key.name] = (pm, None), (None, sm)
                top, bottom = C.overlay(fitted, outlined_photo), C.overlay(rgb, outlined_take)
            rows.append(
                {
                    "t": t_ref,
                    "label": anchor.get("label", ""),
                    "file": str(photo_path),
                    "frame": i,
                    "sim_t": frame["t"],
                    "sim_step": frame["step"],
                }
            )
            label = f"{anchor.get('label') or ''} {t_ref:.1f}s / sim {float(frame['t']):.1f}s".strip()
            tiles.append((top, bottom, label))
            lines.append(
                f"{anchor.get('label') or 'anchor':<12} clip {t_ref:5.1f}s = take {float(frame['t']):5.1f}s (step {frame['step']})"
            )
    finally:
        reader.close()
    moved = take_motion(frames) if moved is None else moved  # worked out already as the take was written
    lines.append(motion_line(moved))
    return {"take": name, "anchors": rows, "moved": moved, "lines": lines, "tiles": tiles}


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

    def add(name: str, ok: bool | None, detail: str) -> None:
        """``ok`` None is a line that looked at nothing: SKIP, and no part of the verdict."""
        checks.append({"name": name, "ok": None if ok is None else bool(ok), "detail": detail})

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
        (not missing) if wanted else None,
        f"all of {', '.join(wanted)} are in the scene"
        if wanted and not missing
        else (
            f"missing: {', '.join(missing)}"
            if missing
            else f"no names given (--object or --key), so nothing was looked for; the scene has {len(have)} things"
        ),
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
            f"worst overlap {worst_mm:.2f} mm"
            + (f" between {pairs[0]}" if pairs else "")
            + f" (bar {penetration_mm:g} mm)",
        )

    # rest: leave the scene alone and see what drifts -- a rigid thing by its centre, a deformable by its
    # particle that moved most, since a cloth can sag or a rope sink with its centre where it was
    looked_at = wanted or list(have)
    before = _positions(sim, looked_at)
    shapes_before = _deformable_points(sim, looked_at)
    bodies = {n: b for n, b in take_bodies(sim).items() if n in looked_at}
    parts_before = _part_positions(sim, bodies=bodies, q=np.zeros((0, 3))) if bodies else {}
    sim.run(int(steps))
    sim.sync()
    after = _positions(sim, list(before))
    drift = {
        n: 1000.0 * float(np.linalg.norm(np.asarray(after[n]) - np.asarray(before[n]))) for n in before if n in after
    }
    # a jointed thing by its body that moved most, at each body's centre of mass: a loose lid or a key slipping
    # in its lock leaves the base's origin where it was
    for name, now in (_part_positions(sim, bodies=bodies, q=np.zeros((0, 3))) if bodies else {}).items():
        was = parts_before.get(name)
        if was and len(was) == len(now):
            moved = float(np.linalg.norm(np.asarray(now) - np.asarray(was), axis=1).max())
            drift[name] = float(np.maximum(drift.get(name, 0.0), 1000.0 * moved))  # a NaN on either side stays
    from robosimstudio.core.render.particles import draws_points

    handles = getattr(getattr(getattr(sim, "scene", None), "handles", None), "deformables", None) or {}
    for name, points in _deformable_points(sim, list(shapes_before)).items():
        moved = np.linalg.norm(points - shapes_before[name], axis=1)
        if not len(moved):
            drift[name] = 0.0
            continue
        # a cloth or a rope by the particle that moved most; a heap or a pool, whose grains never all come to
        # rest, by the 95th percentile, so one jittering grain does not fail it
        point_body = draws_points(handles.get(name)) if handles.get(name) is not None else False
        drift[name] = 1000.0 * float(np.percentile(moved, 95) if point_body else moved.max())
    unmeasured = [n for n in looked_at if n in have and n not in drift]
    # a drift that is not a number is the worst there is: a thing that blew up did not stay put
    worst = max(drift.items(), key=lambda kv: kv[1] if np.isfinite(kv[1]) else np.inf) if drift else ("", 0.0)
    add(
        "rest",
        worst[1] <= settle_mm,
        (
            f"nothing moved more than {worst[1]:.2f} mm in {steps} steps (bar {settle_mm:g} mm)"
            if worst[1] <= settle_mm
            else f"{worst[0]} drifted {worst[1]:.2f} mm in {steps} steps left alone (bar {settle_mm:g} mm)"
        )
        + (f"; not measured (no position to read): {', '.join(unmeasured)}" if unmeasured else ""),
    )
    finite = bool(sim.state_is_finite())
    add("finite_after", finite, f"still finite after {steps} steps" if finite else f"went nan within {steps} steps")

    passed = all(c["ok"] is not False for c in checks)
    word = {True: "PASS", False: "FAIL", None: "SKIP"}
    report = "\n".join(f"{word[c['ok']]}  {c['name']:<12} {c['detail']}" for c in checks)
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


def _deformable_points(sim: Any, names: list[str]) -> dict[str, np.ndarray]:
    """Each named deformable's particles where they are now, ``(n, 3)`` [m]."""
    handles = getattr(getattr(getattr(sim, "scene", None), "handles", None), "deformables", None) or {}
    wanted = [n for n in names if n in handles]
    if not wanted:
        return {}
    q = _particles(sim)
    return {name: q[_indices(handles[name], len(q))].copy() for name in wanted}


def _penetration(sim: Any) -> tuple[float, list[str]]:
    from robosimstudio.scene.snapshot import penetration_rows

    rows = penetration_rows(sim)
    if not rows:
        return 0.0, []
    worst = rows[0]
    return float(worst["depth_mm"]), [" and ".join(worst["between"])]
