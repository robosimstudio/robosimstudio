# SPDX-License-Identifier: Apache-2.0
"""Blender's renderers -- Cycles, or EEVEE -- over a scene's visible shapes, from a Blender this process starts.

The path tracer needs an NVIDIA GPU with RT cores; ``tiled`` and ``gl`` run
anywhere but draw flat colour. This is the photoreal picture where ``rtx``
cannot run -- a Mac above all, where Cycles renders on Metal -- and a second
opinion where it can.

**Blender runs beside this process, not in it.** ``bpy`` is published as a
wheel for exactly one Python per Blender release (3.11 up to Blender 5.0, 3.13
from 5.1) and this package is on 3.12, so it cannot be imported here; and an
installed Blender is what a person on a Mac already has. So the renderer
starts one (:func:`find_blender`: ``blender=``, then ``ROBOSIMSTUDIO_BLENDER``,
then ``blender`` on the ``PATH``, then ``/Applications/Blender.app``) running
:mod:`~robosimstudio.solvers.render.blender_worker`, which connects back over a
loopback socket. The scene is sent once -- every visible shape as a mesh with
its colour, texture and finish, hung off its body; the deformables' topology;
the lights -- and each frame sends only poses, deformable vertices and the
cameras, and gets RGB and depth back. ``ROBOSIMSTUDIO_BLENDER`` may also name a
Python interpreter that imports ``bpy`` (one on 3.13 with ``pip install bpy``);
the worker is the same.

**What maps across.**

* Geometry: :func:`~robosimstudio.core.render.visual.model_visual_parts`, the
  meshes ``gl`` draws, each an object parented to an empty per body. A
  deformable is a mesh of fixed topology whose vertices move, a rope a tube, a
  particle body the surface its handle grows each frame (a liquid, a dough) or
  else a point cloud Cycles draws as spheres
  (:func:`~robosimstudio.core.render.deformables.deformable_drawables`).
* Materials: a Principled BSDF per shape, base colour decoded from the
  palette's display values to linear albedo (as ``rtx`` does), the mesh's own
  texture through its UVs, and the shape's
  :class:`~robosimstudio.core.render.finish.Finish` -- roughness, metallic,
  opacity, and a finish texture box-projected at its repeats per metre.
* Light: under ``environment="lab"`` the path tracer's own room, a 1.4 x 0.9 m
  panel 1.7 m over the work surface and a neutral dome, scaled by
  ``key_intensity`` and ``ambient``; otherwise the tiled tracer's rig, a sun
  along ``light_direction``, ``fill_lights`` as shadowless suns, ``ambient`` as
  the world and ``background`` behind everything. ``scenery`` is not drawn.
* Cameras: ``K`` exactly, principal point included (see the worker).
* Depth: distance to the image plane, 0 for a miss -- so a camera over a clean
  plate composites as it does under ``rtx`` -- from a second, one-sample render
  in which each surface emits its own view depth (Blender 5 writes the
  Combined pass alone).

**Cost.** Cycles at ``preview`` (32 samples, denoised) is about a second a
960x540 frame through CUDA or on a many-core CPU (a frame is two renders: the
picture, and a one-sample depth pass); ``final`` takes four times the
samples. EEVEE is faster where it has a GPU context and is
not offered where it has none. Starting Blender costs a second or two, and the
scene a second or two more.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import newton
import numpy as np

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.core.render.base import FrameSet, RendererBase
from robosimstudio.core.render.spec import put

__all__ = ["ENV", "QUALITY_SAMPLES", "RendererBlender", "blender_argv", "find_blender", "gpu_env"]

ENV = "ROBOSIMSTUDIO_BLENDER"
"""The Blender to run: its executable, or a Python interpreter that imports ``bpy``."""

ENGINE_ENV = "ROBOSIMSTUDIO_BLENDER_ENGINE"
"""``cycles`` (the default) or ``eevee``, for a machine; ``engine=`` in code wins."""

DEVICE_ENV = "ROBOSIMSTUDIO_BLENDER_DEVICE"
"""Cycles' device for a machine: ``auto`` (the default: CUDA, Metal, HIP, oneAPI or OptiX, else the CPU),
``cpu``, or one of those backends by name; ``device=`` in code wins."""

MAC_APP = "/Applications/Blender.app/Contents/MacOS/Blender"

QUALITY_SAMPLES: dict[str, int] = {"preview": 32, "final": 128}
"""Cycles samples per pixel, denoised, by :attr:`RenderSpec.quality`."""

LAB_PANEL_WATTS = 12.0
"""The ``lab`` panel's power per unit of ``key_intensity`` [W]: the bench top under it reads as ``rtx``'s does."""

LAB_DOME_STRENGTH = 0.55
"""The ``lab`` dome's world strength per unit of mean ``ambient``."""

SUN_PER_KEY = 2.6
"""A sun's strength per unit of key or fill intensity, for the tiled tracer's rig [W/m^2]."""

AMBIENT_STRENGTH = 0.25
"""The world's strength per unit of mean ``ambient``, for the tiled tracer's rig."""


def find_blender(blender: str = "") -> str:
    """The Blender to run: ``blender`` if given, else :data:`ENV`, else ``blender`` on the ``PATH``, else the Mac app.

    Raises:
        FileNotFoundError: If none of them is there, saying how to get one.
    """
    for raw, source in ((blender, "in code"), (os.environ.get(ENV, ""), f"by {ENV}")):
        name = raw.strip()
        if name:
            found = shutil.which(name) or (name if os.path.isfile(name) else "")
            if not found:
                raise FileNotFoundError(f"no Blender at {name!r} (given {source})")
            return os.path.abspath(found)  # Blender is started in a folder of its own
    found = shutil.which("blender") or (MAC_APP if sys.platform == "darwin" and os.path.isfile(MAC_APP) else "")
    found = os.path.abspath(found) if found else ""
    if not found:
        raise FileNotFoundError(
            "the blender renderer needs Blender (5.x; tested on 5.2 LTS): install it (blender.org; on a Mac, Blender.app in "
            f"/Applications is found by itself), put `blender` on the PATH, or name the executable with {ENV}"
        )
    return found


def blender_argv(executable: str, port: int) -> list[str]:
    """The command line that starts the worker under ``executable`` and connects it to ``port``.

    A Python interpreter (an executable whose name starts with ``python``) runs
    the worker as a script; anything else is taken for Blender itself, run in
    the background on its factory settings so no user preference or add-on
    changes the picture.
    """
    worker = str(Path(__file__).with_name("blender_worker.py"))
    if Path(executable).name.lower().startswith("python"):
        return [executable, worker, str(port)]
    return [executable, "-b", "--factory-startup", "--python-exit-code", "1", "--python", worker, "--", str(port)]


def gpu_env(device: Any) -> dict[str, str]:
    """``CUDA_VISIBLE_DEVICES`` for Blender: the scene's own GPU alone, so Cycles renders on the card the scene
    runs on and not on every card of a shared machine (CUDA's numbering, as this process's).

    A model on a CPU takes the first GPU this process can see, as ``gl`` does; a process that sees none passes
    that on, and Blender renders on the CPU.
    """
    try:
        import warp as wp

        target = wp.get_device(device)
    except Exception:
        return {}
    ordinal = int(target.ordinal) if target.is_cuda else 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return {"CUDA_VISIBLE_DEVICES": str(ordinal)}
    ids = [v.strip() for v in visible.split(",") if v.strip()]
    if not ids:
        return {"CUDA_VISIBLE_DEVICES": ""}
    return {"CUDA_VISIBLE_DEVICES": ids[ordinal] if ordinal < len(ids) else ids[0]}


def _worker_env(executable: str, extra: dict[str, str]) -> dict[str, str]:
    """The worker's environment: this process's, less what points a Python at this package's 3.12 packages --
    Blender ignores those, but an interpreter with ``bpy`` (3.13) would import a 3.12 NumPy and die."""
    env = {**os.environ, **extra}
    if Path(executable).name.lower().startswith("python"):
        for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(name, None)
    return env


def _greeting(sock: socket.socket, deadline: float, most: int = 4096) -> str:
    """The token a connection says first, or ``""``: read whole by ``deadline`` (a ``time.monotonic`` value), at most
    ``most`` bytes, so a connection that trickles or floods cannot hold up the renderer's start."""
    import json
    import struct

    def read(n: int) -> bytes:
        got = b""
        while len(got) < n:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError
            sock.settimeout(left)
            chunk = sock.recv(n - len(got))
            if not chunk:
                raise ConnectionError
            got += chunk
        return got

    try:
        (n,) = struct.unpack("<Q", read(8))
        if n > most:
            return ""
        hello = json.loads(read(n).decode("utf-8"))
    except Exception:  # anything that is not a greeting -- a timeout, a hang-up, garbage, JSON nested past recursion
        return ""
    return str(hello.get("token", "")) if isinstance(hello, dict) else ""


def _linear(rgb: Sequence[float]) -> list[float]:
    """Display values (the palette's, a texture's factor) as the linear albedo Blender shades with, as ``rtx`` decodes them."""
    from robosimstudio.core.render.palette import srgb_to_linear

    return list(srgb_to_linear(list(rgb)[:3]))


def _png(image: Any) -> bytes:
    import io

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


class RendererBlender(RendererBase):
    """Every camera through Blender, one at a time, off the model's visible shapes and its deformables.

    Args:
        model: The finalized model. One world.
        cameras: The cameras to render. They may differ in resolution and intrinsics.
        channels: Restrict every camera to these channels.
        blender: The executable to run; see :func:`find_blender`.
        engine: ``cycles`` or ``eevee``; :data:`ENGINE_ENV`, else ``cycles``.
        device: Cycles' device; :data:`DEVICE_ENV`, else ``auto``.
        quality: ``preview`` or ``final``; picks ``samples`` unless that is stated.
        samples: Samples per pixel.
        denoise: Whether Cycles denoises (OpenImageDenoise).
        threads: CPU threads for Cycles; 0 lets Blender use every core.
        timeout: Seconds to wait for Blender to start, build the scene, or render a frame.
        light_direction: The way the key light travels, for the sun rig.
        key_intensity: The key's brightness; under ``lab``, the panel's.
        fill_lights: ``((direction, intensity), ...)``, shadowless suns.
        ambient: RGB or scalar: the world's light; under ``lab``, the dome's (by its mean).
        shadows: Whether the key casts shadows.
        background: RGB in [0, 1] a camera sees where it hits nothing.
        environment: ``lab`` lights the scene as the path tracer's ``lab`` room; anything else is the sun rig.
        finishes: ``shape index -> Finish``, normally the scene handles' table.
        deformables: ``name -> DeformableHandle``, normally the scene handles' table.
        deformable_colors: ``name -> RGB`` for those.
        deformable_finishes: ``name -> Finish`` for those.
        cell: The scene's cell, whose centre the ``lab`` panel hangs over.
        surface_z: World z of the work surface [m].
    """

    name = "blender"
    supported_channels = ("rgb", "depth")
    retargets = True  # every frame sends its cameras: a new viewpoint costs no new Blender
    first_note = "the first Blender picture of a scene starts Blender and hands it the scene, which takes seconds"
    first_note_each_scene = True
    devices = ("cpu", "cuda")  # Blender is its own process and renders wherever it can, whatever device the model is on
    wants: tuple[str, ...] = (
        "finishes",
        "deformables",
        "deformable_colors",
        "deformable_finishes",
        "cell",
        "surface_z",
    )

    @classmethod
    def look_kwargs(cls, look: Any) -> dict[str, Any]:
        """What this backend takes from a look; see :meth:`RendererBase.look_kwargs`."""
        out: dict[str, Any] = {}
        put(out, "light_direction", look.light_direction)
        put(out, "key_intensity", look.key_intensity)
        put(out, "fill_lights", look.fill_lights)
        put(out, "ambient", look.ambient)
        put(out, "shadows", look.shadows)
        put(out, "background", look.background)
        put(out, "environment", look.environment)
        put(out, "quality", look.quality)
        return out

    def __init__(
        self,
        model: newton.Model,
        cameras: Sequence[CameraSpec],
        *,
        channels: Sequence[str] | None = None,
        blender: str = "",
        engine: str = "",
        device: str = "",
        quality: str = "preview",
        samples: int | None = None,
        denoise: bool = True,
        threads: int = 0,
        timeout: float = 600.0,
        light_direction: Sequence[float] = (-0.5774, 0.5774, -0.5774),
        key_intensity: float = 1.0,
        fill_lights: Sequence[tuple[Sequence[float], float]] = (),
        ambient: float | Sequence[float] = 0.3,
        shadows: bool = True,
        background: Sequence[float] | None = None,
        environment: str = "",
        finishes: Mapping[int, Any] | None = None,
        deformables: Mapping[str, Any] | None = None,
        deformable_colors: Mapping[str, Sequence[float]] | None = None,
        deformable_finishes: Mapping[str, Any] | None = None,
        cell: Any = None,
        surface_z: float = 0.0,
    ) -> None:
        super().__init__(cameras, channels=channels)
        if int(getattr(model, "world_count", 1) or 1) > 1:
            raise ValueError("the blender renderer draws one world; a fleet renders through 'tiled'")
        if quality not in QUALITY_SAMPLES:
            raise ValueError(f"unknown quality {quality!r}; choose from {sorted(QUALITY_SAMPLES)}")
        engine = (engine or os.environ.get(ENGINE_ENV, "") or "cycles").strip().lower()
        if engine not in ("cycles", "eevee"):
            raise ValueError(f"unknown Blender engine {engine!r}; choose 'cycles' or 'eevee'")
        self.model = model
        self.engine = engine
        self.timeout = float(timeout)
        self.executable = find_blender(blender)
        self._proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._tmp = ""
        try:
            self._tmp = tempfile.mkdtemp(prefix="rss_blender_")
            self._log = Path(self._tmp) / "blender.log"
            self._rest = model.particle_q.numpy() if int(getattr(model, "particle_count", 0) or 0) else None
            header, blobs, self._bodies, self._drawables = _describe(
                model,
                finishes=finishes,
                deformables=deformables,
                deformable_colors=deformable_colors,
                deformable_finishes=deformable_finishes,
                rest=self._rest,
            )
            header.update(
                op="scene",
                engine=engine,
                device=(device or os.environ.get(DEVICE_ENV, "") or "auto").strip(),
                samples=int(samples if samples is not None else QUALITY_SAMPLES[quality]),
                denoise=bool(denoise),
                threads=int(threads),
                lights=_lights(environment, light_direction, key_intensity, fill_lights, shadows, cell, surface_z),
                world=_world(environment, ambient, background),
            )
            self._start()
            reply, _ = self._ask(header, blobs)
        except BaseException:
            self.close()
            raise
        self.version = str(reply.get("version", ""))
        self.device = str(reply.get("device", ""))
        """Where Cycles renders: ``CPU``, or the GPU backend and its devices."""

    # ------------------------------------------------------------------ the worker
    def _start(self) -> None:
        import secrets

        from robosimstudio.solvers.render.blender_worker import TOKEN_ENV

        server = socket.create_server(("127.0.0.1", 0))
        # the worker says this first: a connection from anything else on the machine is dropped, not given the scene
        token = secrets.token_hex(16)
        try:
            port = server.getsockname()[1]
            with open(self._log, "wb") as log:
                self._proc = subprocess.Popen(
                    blender_argv(self.executable, port),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=self._tmp,
                    env=_worker_env(self.executable, {TOKEN_ENV: token, **gpu_env(self.model.device)}),
                )
            # in short waits, so a Blender that exits at once (a missing library, a flag it rejects) is said at once
            server.settimeout(0.5)
            deadline = time.monotonic() + self.timeout
            while self._sock is None:
                # checked every pass, not only when nothing connects: a stream of other connections cannot hold it
                if self._proc.poll() is not None:
                    raise RuntimeError(f"Blender exited {self._proc.returncode} before it connected; {self._tail()}")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Blender did not connect within {self.timeout:.0f} s; {self._tail()}")
                try:
                    candidate, _ = server.accept()
                except TimeoutError:
                    continue
                said = _greeting(candidate, min(deadline, time.monotonic() + 5.0))
                if secrets.compare_digest(said.encode("utf-8", "replace"), token.encode()):  # bytes: any text
                    self._sock = candidate
                else:
                    candidate.close()
        finally:
            server.close()
        self._sock.settimeout(self.timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def _tail(self, lines: int = 25) -> str:
        try:
            text = self._log.read_text(errors="replace").strip().splitlines()
        except OSError:
            return "no log"
        return f"{self.executable} said:\n" + "\n".join(text[-lines:])

    def _ask(self, header: dict, blobs: Sequence[Any] = ()) -> tuple[dict, list[bytes]]:
        from robosimstudio.solvers.render.blender_worker import receive, send

        if self._sock is None:
            raise RuntimeError("the blender renderer is closed")
        try:
            send(self._sock, header, list(blobs))
            reply, out = receive(self._sock)
        except (OSError, ConnectionError) as exc:
            died = self._proc is not None and self._proc.poll() is not None
            why = f"exited {self._proc.returncode}" if died else f"stopped answering ({type(exc).__name__}: {exc})"
            tail = self._tail()
            # a reply half read, or still to come, would answer the next request: this Blender is done
            self.close()
            raise RuntimeError(f"Blender {why}; {tail}") from exc
        if not reply.get("ok"):
            raise RuntimeError(
                f"Blender could not {header.get('op')}: {reply.get('error')}\n{reply.get('traceback', '')}"
            )
        return reply, out

    # ------------------------------------------------------------------ render
    def render(self, state: newton.State) -> FrameSet:
        """Every camera for one state. Each array is the caller's to keep."""
        from robosimstudio.core.math import transform_matrix

        body_q = state.body_q.numpy() if state.body_q is not None else None
        header: dict[str, Any] = {"op": "frame", "bodies": -1, "body_ids": [], "deformables": [], "points": []}
        blobs: list[Any] = []
        if self._bodies and body_q is not None:
            header["body_ids"] = list(self._bodies)
            header["bodies"] = len(blobs)
            blobs.append(np.stack([transform_matrix(body_q[b]) for b in self._bodies]).astype(np.float32))
        header["surfaces"] = []
        particle_q = state.particle_q.numpy() if self._drawables and state.particle_q is not None else None
        if particle_q is None:
            particle_q = self._rest  # a state without particles: the deformables as built
        if self._drawables:
            for part in self._drawables.meshes:
                header["deformables"].append(len(blobs))
                blobs.append(np.ascontiguousarray(part.positions(particle_q), dtype=np.float32))
            for batch in self._drawables.points:
                header["points"].append(len(blobs))
                blobs.append(np.ascontiguousarray(batch.positions(particle_q), dtype=np.float32))
            for surface in self._drawables.surfaces:
                verts, tris = surface.mesh(particle_q)
                header["surfaces"].append([len(blobs), len(blobs) + 1])
                blobs += [np.ascontiguousarray(verts), np.ascontiguousarray(tris)]
        header["cameras"] = []
        for cam in self.cameras:
            k = cam.intrinsics
            header["cameras"].append(
                {
                    "name": cam.name,
                    "width": int(cam.width),
                    "height": int(cam.height),
                    "K": [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])],
                    "near": float(cam.near),
                    "far": float(cam.far),
                    "pose": np.asarray(cam.pose_on_body(body_q), dtype=np.float64).ravel().tolist(),
                    "channels": list(self.channels[cam.name]),
                }
            )
        reply, out = self._ask(header, blobs)
        frames: FrameSet = {}
        for cam, got in zip(self.cameras, reply["cameras"], strict=True):
            h, w = int(cam.height), int(cam.width)
            frame: dict[str, np.ndarray] = {}
            if got.get("rgb", -1) >= 0:
                frame["rgb"] = np.frombuffer(out[got["rgb"]], dtype=np.uint8).reshape(h, w, 3).copy()
            if got.get("depth", -1) >= 0:
                frame["depth"] = np.frombuffer(out[got["depth"]], dtype=np.float32).reshape(h, w).copy()
            frames[cam.name] = frame
        return frames

    def close(self) -> None:
        """Stop Blender and remove its files. Idempotent."""
        sock, proc = self._sock, self._proc
        self._sock = self._proc = None
        if sock is not None:
            with contextlib.suppress(Exception):
                from robosimstudio.solvers.render.blender_worker import receive, send

                sock.settimeout(5.0)
                send(sock, {"op": "quit"})
                receive(sock)
            with contextlib.suppress(OSError):
                sock.close()
        if proc is not None:
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10.0)
        tmp = getattr(self, "_tmp", None)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
            self._tmp = ""

    @property
    def closed(self) -> bool:
        """Whether this renderer is done: closed, or its Blender lost (a broken connection closes it)."""
        return self._sock is None

    def __del__(self) -> None:
        """A renderer dropped unclosed: its Blender killed at once, not asked to quit -- a finaliser must not wait."""
        try:
            sock, proc = getattr(self, "_sock", None), getattr(self, "_proc", None)
            self._sock = self._proc = None
            if sock is not None:
                sock.close()
            if proc is not None and proc.poll() is None:
                proc.kill()
            if getattr(self, "_tmp", ""):
                shutil.rmtree(self._tmp, ignore_errors=True)
        except Exception:  # at interpreter exit the modules this uses may already be gone
            pass


# ------------------------------------------------------------------ the scene message
def _describe(
    model: newton.Model,
    *,
    finishes: Mapping[int, Any] | None,
    deformables: Mapping[str, Any] | None,
    deformable_colors: Mapping[str, Sequence[float]] | None,
    deformable_finishes: Mapping[str, Any] | None,
    rest: np.ndarray | None,
) -> tuple[dict, list[Any], list[int], Any]:
    """The ``scene`` message: every visible shape, the deformables and the particles, and the bodies to pose."""
    import hashlib

    from robosimstudio.core.render.deformables import deformable_drawables
    from robosimstudio.core.render.finish import finish_for, texture_path
    from robosimstudio.core.render.visual import model_visual_parts
    from robosimstudio.solvers.render.gl import _rgba

    finishes = dict(finishes or {})
    colors = getattr(model, "shape_color", None)
    shape_colors = np.asarray(colors.numpy() if hasattr(colors, "numpy") else (colors if colors is not None else ()))
    blobs: list[Any] = []
    meshes: list[dict] = []
    bodies: set[int] = set()
    textures: dict[str, int] = {}
    images: dict[str, int] = {}
    for part in model_visual_parts(model):
        mesh = part.mesh
        if len(getattr(mesh, "faces", ())) == 0:
            continue
        rgba, image, uv = _rgba(mesh)
        # a finish is keyed by the colour as the model states it: trimesh's copy is rounded to bytes
        stated = tuple(float(v) for v in shape_colors[part.shape][:3]) if 0 <= part.shape < len(shape_colors) else None
        finish = finishes.get(part.shape) or finish_for(stated or tuple(rgba[:3]))
        # the colour as the model states it where trimesh's is that colour rounded to bytes (an untextured part)
        base = stated if stated is not None and image is None else tuple(rgba[:3])
        entry: dict[str, Any] = {
            "name": f"shape{part.shape}",
            "body": int(part.body),
            "local": np.asarray(part.local, dtype=np.float64).ravel().tolist(),
            "color": [*_linear(base), float(rgba[3])],
            "roughness": float(finish.roughness),
            "metallic": float(finish.metallic),
            "opacity": float(finish.opacity),
            "texture": -1,
            "uv": -1,
        }
        entry["verts"] = len(blobs)
        blobs.append(np.ascontiguousarray(mesh.vertices, dtype=np.float32))
        entry["faces"] = len(blobs)
        blobs.append(np.ascontiguousarray(mesh.faces, dtype=np.int32))
        if image is not None:
            entry["uv"] = len(blobs)
            blobs.append(np.ascontiguousarray(uv, dtype=np.float32))
            # one blob per image, however many shapes share it (an atlas): each part's image is a fresh copy, so
            # it is known by its pixels
            pixels = image.convert("RGB")
            seen = hashlib.sha1(pixels.tobytes() + repr(pixels.size).encode()).hexdigest()
            if seen not in images:
                images[seen] = len(blobs)
                blobs.append(_png(pixels))
            entry["texture"] = images[seen]
        elif finish.texture:
            try:
                path = texture_path(finish.texture)
            except (ValueError, KeyError, OSError):
                path = ""
            if path:
                if path not in textures:
                    textures[path] = len(blobs)
                    blobs.append(Path(path).read_bytes())
                entry["texture"] = textures[path]
                entry["box"] = [*map(float, finish.uv_scale), *map(float, finish.uv_offset)]
                entry["color"] = [1.0, 1.0, 1.0, float(rgba[3])]
        if part.body >= 0:
            bodies.add(int(part.body))
        meshes.append(entry)

    drawables = deformable_drawables(model, deformables, deformable_colors, surfaces=True)
    deformable_list, point_list = [], []

    def look(name: str, kind: str, color) -> dict[str, Any]:
        """A deformable's material entry: its finish, else the palette's for its colour."""
        finish = (deformable_finishes or {}).get(name) or finish_for(color)
        return {
            "name": kind,
            "color": [*_linear(color), 1.0],
            "roughness": float(finish.roughness),
            "metallic": float(finish.metallic),
            "opacity": float(finish.opacity),
        }

    for i, part in enumerate(drawables.meshes):
        entry = {**look(part.name, f"deformable{i}", part.color), "faces": len(blobs)}
        blobs.append(np.ascontiguousarray(part.triangles, dtype=np.int32))
        entry["verts"] = len(blobs)
        blobs.append(np.ascontiguousarray(part.positions(rest), dtype=np.float32))
        deformable_list.append(entry)
    for i, batch in enumerate(drawables.points):
        point_list.append({**look(batch.name, f"points{i}", batch.color), "radii": len(blobs)})
        blobs.append(np.ascontiguousarray(batch.radii, dtype=np.float32))
    surface_list = [look(surface.name, f"surface{i}", surface.color) for i, surface in enumerate(drawables.surfaces)]
    header = {"meshes": meshes, "deformables": deformable_list, "points": point_list, "surfaces": surface_list}
    return header, blobs, sorted(bodies), drawables


def _lights(
    environment: str,
    light_direction: Sequence[float],
    key_intensity: float,
    fill_lights: Sequence[tuple[Sequence[float], float]],
    shadows: bool,
    cell: Any,
    surface_z: float,
) -> list[dict]:
    """The lights: the ``lab`` panel, or the sun rig (see the module docstring)."""
    if environment == "lab":
        from robosimstudio.solvers.render.rtx import LAB_PANEL_COLOR, LAB_PANEL_HEIGHT, LAB_PANEL_SIZE

        center = (
            tuple(float(v) for v in getattr(cell, "center", (0.0, 0.0, 0.0))[:2]) if cell is not None else (0.0, 0.0)
        )
        return [
            {
                "type": "area",
                "size": list(LAB_PANEL_SIZE),
                "position": [center[0], center[1], float(surface_z) + LAB_PANEL_HEIGHT],
                "direction": [0.0, 0.0, -1.0],
                "color": list(LAB_PANEL_COLOR),
                "energy": LAB_PANEL_WATTS * float(key_intensity),
                "shadow": True,
            }
        ]
    rig = [
        {
            "type": "sun",
            "direction": [float(v) for v in light_direction],
            "energy": SUN_PER_KEY * float(key_intensity),
            "shadow": bool(shadows),
        }
    ]
    for direction, intensity in fill_lights:
        rig.append(
            {
                "type": "sun",
                "direction": [float(v) for v in direction],
                "energy": SUN_PER_KEY * float(intensity),
                "shadow": False,
            }
        )
    return rig


def _world(environment: str, ambient: float | Sequence[float], background: Sequence[float] | None) -> dict:
    """The world: the ``lab`` dome, or the ambient term; and what a camera sees where it hits nothing."""
    level = float(np.mean(ambient))
    if environment == "lab":
        from robosimstudio.solvers.render.rtx import LAB_DOME_COLOR

        # the path tracer's untextured dome lights the scene and is not in a camera's ray: a miss is black there,
        # whatever background the look states (rtx takes none)
        return {"color": list(LAB_DOME_COLOR), "strength": LAB_DOME_STRENGTH * level, "background": [0.0, 0.0, 0.0]}
    tint = np.asarray(ambient, dtype=np.float64).reshape(-1)
    tint = tint / max(level, 1e-9) if tint.size == 3 else np.ones(3)
    world: dict[str, Any] = {"color": np.clip(tint, 0.0, None).tolist(), "strength": AMBIENT_STRENGTH * level}
    if background is not None:
        world["background"] = _linear(background)
    return world
