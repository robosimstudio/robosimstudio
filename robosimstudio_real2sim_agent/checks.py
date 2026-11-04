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
    things = None
    for pm, sm in pairs.values():
        for m in (pm, sm):
            if m is not None:
                things = m.copy() if things is None else (things | m)
    picture = picture_reading(rgb, fitted, things)
    lines.append(
        f"picture: colour difference {picture['delta_e']} (mean CIE76 dE) over the frame, "
        f"{picture['delta_e_behind']} behind the keyed things; PSNR {picture['psnr_db']} dB, "
        f"{picture['psnr_behind_db']} dB behind them (identical pixels read dE 0)"
    )
    if abs(stretch - 1.0) > 0.02:
        lines.append(
            f"the photo's aspect was stretched {stretch:.2f}x to fit the camera: give the camera the photo's width and height"
        )
    return {
        "objects": objects,
        "picture": picture,
        "stretch": round(stretch, 3),
        "lines": lines,
        "pairs": pairs,
        "fitted": fitted,
    }


def _lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB ``uint8`` to CIE L*a*b* under D65."""
    c = rgb[..., :3].astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])], -1)


def picture_reading(rgb: np.ndarray, photo: np.ndarray, things: np.ndarray | None = None) -> dict:
    """How far the render's pixels are from the photograph's, over the frame and behind the keyed things.

    The mean CIE76 colour difference (a dE of 2 is about the smallest a person
    sees side by side, 10 a plainly different colour) and the PSNR. "Behind" is
    every pixel neither picture's keyed things cover: the room, the table, the
    background a plate or a model has to get right.
    """
    a, b = rgb[..., :3], photo[..., :3]
    de = np.linalg.norm(_lab(a) - _lab(b), axis=-1)
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean(axis=-1)
    behind = np.ones(de.shape, dtype=bool) if things is None else ~things
    if not behind.any():
        behind = np.ones(de.shape, dtype=bool)

    def psnr(m: np.ndarray) -> float:
        return round(float(10.0 * np.log10(255.0**2 / max(float(m.mean()), 1e-6))), 1)

    return {
        "delta_e": round(float(de.mean()), 1),
        "delta_e_behind": round(float(de[behind].mean()), 1),
        "psnr_db": psnr(mse),
        "psnr_behind_db": psnr(mse[behind]),
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

    def __init__(self, server: Any, name: str, *, camera: str = "", every: int = 4, also: Any = ()) -> None:
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
        self.also: dict[str, dict[str, Any]] = {}
        """Per renderer ``--also`` names: its renderer, its video's writer and path. Each films the take's own states."""
        self.grid: dict[str, Any] | None = None
        """The side-by-side video's writer and path, while ``--also`` names another renderer."""
        self.also_skipped: list[str] = []
        """``--also`` names that are the page's own renderer, which films the take itself."""
        self.also_dir = self.run_dir / f"{name}.renderers"
        """The take's ``--also`` videos, one per renderer and ``all.mp4`` side by side: a folder of its own, made
        afresh as each take of this name begins, so nothing of an earlier one is left to read as this one's."""
        try:
            self._open_also(also, VideoWriter)
        except Exception:
            self.close()
            raise

    def _open_also(self, also: Any, writer: Any) -> None:
        """A renderer and a video per name in ``also`` other than the page's own (which films the take itself), and
        a grid of them all: one run drawn by every renderer, rather than a run per renderer that the sim, not being
        deterministic, would play differently."""
        from dataclasses import replace

        from robosimstudio.solvers.render import RENDERERS, backend_class, supported_channels

        def alias(b: str) -> str:  # a dotted path to a registered renderer goes by its alias
            return next((a for a, path in RENDERERS.items() if path == b), b)

        own = alias(self.sim.platform.renderer_for("")[0])
        names = [alias(str(b).strip()) for b in (also.split(",") if isinstance(also, str) else also) if str(b).strip()]
        for b in names:
            backend_class(b)  # an unknown name fails here, before anything is filmed
        names = list(dict.fromkeys(names))
        self.also_skipped = [b for b in names if backend_class(b) is backend_class(own)]
        if self.also_skipped:
            print(f"[rss] take {self.name!r}: {own} is the page's own renderer, which films the take itself")
        names = [b for b in names if b not in self.also_skipped]
        if not names:
            return
        bare = replace(self.cam, background="")
        for b in names:
            channels = ["rgb"] + (["depth"] if self.cam.background and "depth" in supported_channels(b) else [])
            entry = self.also[b] = {"video": self.also_dir / f"{b}.mp4"}  # an alias, or the whole path
            entry["renderer"] = self.sim.make_renderer(b, cameras=[bare], channels=channels)
            if getattr(backend_class(b), "warms_up", False):
                entry["renderer"].render(self.sim.state)  # the first frame only builds the stage
            entry["writer"] = writer(entry["video"], fps=self.fps)
        grid = self.also_dir / "all.mp4"
        self.grid = {"video": grid, "writer": writer(grid, fps=self.fps)}
        self.grid_labels = [own, *names]

    def _look_also(
        self, size: tuple[int, int], passed: dict[str, Any] | None, strict: bool = False
    ) -> dict[str, np.ndarray]:
        """The state the take films now, through each ``--also`` renderer, over the camera's plate (``passed``,
        the plate pass the take's own picture was composited on) when it has one.

        A frame a renderer fails is black in its video and tile, as :meth:`Sim.render` keeps a renderer that failed
        one frame; one that is done (``closed``: a Blender that died) or fails :data:`ALSO_MISSES` frames running is
        stopped, its video ends there and its tile stays black, said in :meth:`finish`. The take itself, and every
        other renderer, carry on. ``strict`` (the probe, before anything is filmed) raises instead: a renderer that
        cannot draw the first frame is refused."""
        out: dict[str, np.ndarray] = {}
        black = np.zeros((*size, 3), np.uint8)
        for b, entry in self.also.items():
            if not self._drawing(entry):
                out[b] = black
                continue
            try:
                frame = entry["renderer"].render(self.sim.state)[self.cam.name]
                rgb = _sized(stills.to_uint8(np.asarray(frame["rgb"])[..., :3]), size)
                entry["running"] = 0
            except Exception as exc:
                if strict:
                    raise RuntimeError(
                        f"--also {b}: the first frame would not draw: {type(exc).__name__}: {exc}"
                    ) from exc
                entry["running"] = entry.get("running", 0) + 1
                entry["missed"] = entry.get("missed", 0) + 1
                if entry["running"] < ALSO_MISSES and not getattr(entry["renderer"], "closed", False):
                    out[b] = black
                    continue
                entry["failed"] = f"{type(exc).__name__}: {exc}"[:300]
                print(
                    f"[rss] take {self.name!r}: the {b} renderer stopped ({entry['failed']}); its tile is black from here"
                )
                for k in ("renderer", "writer"):  # its video ends where it stopped
                    try:
                        entry.pop(k).close()
                    except Exception:
                        pass
                rgb = black
            else:
                if passed:
                    try:  # the renderer drew: a composite that fails leaves its picture raw, not the renderer stopped
                        depth = frame.get("depth")
                        if depth is not None and np.shape(depth)[:2] != size:
                            depth = None  # a depth of another size cannot trim this composite: the collider mask alone
                        rgb = stills.to_uint8(self.sim.over_plate(self.cam, rgb, depth=depth, plate_pass=passed))
                    except Exception as exc:
                        if "raw" not in entry:
                            entry["raw"] = f"{type(exc).__name__}: {exc}"[:300]
                            print(f"[rss] take {self.name!r}: {b}'s frames go unplated ({entry['raw']})")
            out[b] = rgb
        return out

    def _drawing(self, entry: dict) -> bool:
        """Whether an ``--also`` renderer is still drawn: it has not stopped, and its video or the grid still takes
        its frames."""
        grid = self.grid is not None and "writer" in self.grid
        return "failed" not in entry and ("writer" in entry or grid)

    def _look(self, strict: bool = False) -> tuple[np.ndarray, np.ndarray, Any, dict[str, np.ndarray]]:
        """The scene as the take films it now: the picture, the named ids, the particles read for both, and the same
        state through each ``--also`` renderer still drawing, composited on the one plate pass."""
        sim = self.sim
        sim.sync()
        frame = self.renderer.render(sim.state)[self.cam.name]
        shared = bool(self.cam.background) and any(self._drawing(e) for e in self.also.values())
        passed = sim.plate_pass(self.cam) if shared else None
        rgb = stills.to_uint8(sim.render(self.cam, plate_pass=passed)["rgb"][..., :3])
        q = _particles(sim) if self.deformable_labels else None  # once a frame, for the ids and the samples both
        seg = name_deformables(sim, self.cam, np.asarray(frame["seg"]).astype(np.int32), self.deformable_labels, q)
        return rgb, seg, q, self._look_also(rgb.shape[:2], passed, strict=strict)

    def probe(self) -> None:
        """Film one frame and encode it once, before anything the take would replace is touched: a camera, a
        renderer (an ``--also`` one too), an id lookup or a video encoder that fails, fails here. The frame is kept
        for :meth:`start`."""
        look = self._look(strict=True)
        self._write(look, encode=True)
        self._probed = (int(self.sim.step_count), reset_mark(self.sim), look)

    def _write(self, look: tuple, encode: bool = False) -> None:
        """Hand one look's pictures to their writers: the take's, each ``--also`` renderer's and the grid's; or, with
        ``encode``, only try that each encodes (the probe)."""
        rgb, extra = look[0], look[3]
        (self.writer.try_encode if encode else self.writer.append)(rgb)  # the take's own: its failure is the take's
        sides: list[tuple[dict, Any]] = [(self.also[b], f) for b, f in extra.items() if "writer" in self.also[b]]
        if self.grid is not None and "writer" in self.grid:
            labels = [self.grid_labels[0]] + [
                f"{b} (stopped)" if "failed" in self.also[b] else b for b in self.grid_labels[1:]
            ]
            sides.append((self.grid, lambda: grid_frame([rgb, *extra.values()], labels)))
        for entry, side in sides:  # an --also video or the grid that will not encode stops alone, said in finish
            try:
                frame = side() if callable(side) else side  # the grid drawn here: its failure is the grid's
                (entry["writer"].try_encode if encode else entry["writer"].append)(frame)
            except Exception as exc:
                if encode:
                    raise
                # the video, not the renderer: its tile in the grid carries on
                entry.setdefault("video_failed", f"its video stopped: {type(exc).__name__}: {exc}"[:300])
                try:
                    entry.pop("writer").close()
                except Exception:
                    pass

    def start(self, fresh: bool = False) -> None:
        """Take the scene as it stands as the take's beginning: its step count from here, and frame 0 -- the
        probe's, rather than the same picture rendered again, unless the scene has stepped since or ``fresh``
        says something else may have moved it (a dropped routine's cleanup moves things without a step)."""
        self._raw, self._mark, self.steps = int(self.sim.step_count), reset_mark(self.sim), 0
        self.started = time.perf_counter()
        self._clear_stale()
        probed, self._probed = self._probed, None
        same = not fresh and probed is not None and probed[:2] == (self._raw, self._mark)
        self.capture(probed[2] if same else None)

    def _clear_stale(self) -> None:
        """Replace the folder of an earlier take of this name's ``--also`` videos, once the probe has passed and this
        take begins (that take has finished by then): whatever it left, a crashed take's or a stopped video's too."""
        import shutil

        shutil.rmtree(self.also_dir, ignore_errors=True)
        if self.also:
            self.also_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """Close the video writers and the renderers (the id renderer, any ``--also``); safe to call twice."""
        sides = [*getattr(self, "also", {}).values(), *([self.grid] if getattr(self, "grid", None) else [])]
        extra = [e[k] for e in sides for k in ("writer", "renderer") if k in e]
        for thing in (self.writer, self.renderer, *extra):
            if thing is None:
                continue
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
        look = look if look is not None else self._look()
        _rgb, seg, q, _extra = look
        self._write(look)
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
            # the --also videos and the grid first, each on its own: one that cannot be written is said, not left
            # looking written
            for entry in [*self.also.values(), *([self.grid] if self.grid is not None else [])]:
                writer = entry.pop("writer", None)
                try:
                    if writer is not None:
                        writer.close()
                except Exception as exc:
                    entry.setdefault(
                        "video_failed", f"its video could not be written: {type(exc).__name__}: {exc}"[:300]
                    )
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
        if self.also and self.grid is not None:
            # a renderer that stopped closed its video where it stopped: offered, shorter, said in also_failed; a
            # video whose encoder failed may be unreadable and is not
            meta["also"] = {
                b: str(e["video"])
                for b, e in self.also.items()
                if "video_failed" not in e and Path(e["video"]).is_file()
            }
            if "video_failed" not in self.grid:
                meta["renderers"] = str(self.grid["video"])
            failed = {
                b: e.get("failed") or e["video_failed"]
                for b, e in [*self.also.items(), ("grid", self.grid)]
                if {"failed", "video_failed"} & set(e)
            }
            if failed:
                meta["also_failed"] = failed
            missed = {b: e["missed"] for b, e in self.also.items() if e.get("missed") and "failed" not in e}
            if missed:
                meta["also_missed"] = missed  # frames drawn black, the renderer carrying on
            unplated = {b: e["raw"] for b, e in self.also.items() if "raw" in e}
            if unplated:
                meta["also_unplated"] = unplated  # a composite that failed: those frames as the renderer drew them
        if self.also_skipped:
            meta["also_skipped"] = self.also_skipped
        (self.run_dir / f"{self.name}.json").write_text(json.dumps(meta, default=_plain))
        if self.segs:
            np.savez_compressed(self.run_dir / f"{self.name}_seg.npz", seg=np.stack(self.segs))
        return {
            "take": self.name,
            "video": str(self.video),
            "frames": len(self.frames),
            "seconds": self.frames[-1]["t"] if self.frames else 0.0,
            "wall_seconds": meta["wall_seconds"],
            **({k: meta[k] for k in ALSO_KEYS if k in meta}),
        }


def grid_frame(frames: list[np.ndarray], labels: list[str]) -> np.ndarray:
    """The frames two to a row, each labelled with its renderer; with more than two, at half size each, so the grid
    is about one frame's pixels and costs a take about one more frame's encoding."""
    from PIL import Image

    from robosimstudio.record.recorder import draw_label

    h, w = frames[0].shape[:2]
    if len(frames) > 2:
        h, w = h // 4 * 2, w // 4 * 2  # half, kept even for the encoder
    cols = 1 if len(frames) == 1 else 2
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows), (16, 20, 24))
    for i, (frame, label) in enumerate(zip(frames, labels, strict=True)):
        tile = Image.fromarray(np.asarray(frame)[..., :3])
        if tile.size != (w, h):
            tile = tile.resize((w, h), Image.BILINEAR)
        draw_label(tile, label, size=max(14, h // 22), right=False)
        sheet.paste(tile, ((i % cols) * w, (i // cols) * h))
    return np.asarray(sheet)


ALSO_MISSES = 3
"""Frames running an ``--also`` renderer may fail before it is stopped for the rest of the take."""


def _sized(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """``rgb`` at ``size`` (rows, columns): an ``--also`` renderer that draws another size is scaled to the take's,
    so its video, its black frames and the grid all agree."""
    if rgb.shape[:2] == tuple(size):
        return rgb
    from PIL import Image

    return np.asarray(Image.fromarray(rgb).resize((int(size[1]), int(size[0])), Image.BILINEAR))


ALSO_KEYS = ("also", "renderers", "also_failed", "also_missed", "also_unplated", "also_skipped")
"""What a take's ``TAKE.json`` says of its ``--also`` renderers, handed on in its summary too."""

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
