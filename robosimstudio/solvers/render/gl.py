# SPDX-License-Identifier: Apache-2.0
"""MuJoCo's OpenGL rasteriser over a scene's visible shapes: the fast picture for a scene stepped on a CPU.

The tiled tracer renders where the physics runs, which on a CPU means Warp's
CPU device: about two seconds a 1280x720 frame, a thousand seconds for a 20 s
demo's five cameras. A rasteriser on the machine's GPU draws the same frame in
about a millisecond, which is what a dataset of hundreds of demos needs. This
backend is that rasteriser: MuJoCo's own, through EGL (OSMesa where there is no
GPU at all, named by ``MUJOCO_GL=osmesa``), so it needs nothing the install
does not already have.

**What it draws is the model's visible shapes.** :func:`model_visual_parts`
turns each into a mesh in its body's frame; each becomes a render-only geom on a
mocap body, and a frame writes every body's pose from ``state.body_q`` and runs
MuJoCo's kinematics and nothing else. It does not step anything and it does not
read the solver: the model the physics steps -- on MuJoCo or not -- is never
touched. A camera rides its own mocap body, posed from
:meth:`~robosimstudio.core.cameras.CameraSpec.pose_on_body`, with its ``K``
given to MuJoCo as focal and principal pixels, so a calibrated camera keeps its
principal point.

**Deformables are drawn from the scene's handles**
(:func:`~robosimstudio.core.render.deformables.deformable_drawables`): a
sheet, a bag or a soft body's surface as a mesh whose vertices are rewritten
from ``state.particle_q`` every frame and drawn from both sides, a rope's
centerline as a tube, a heap's grains as spheres -- what the handles name and
nothing else, as the path tracer draws a scene (a renderer built by hand with
no handle table draws every triangle and every other particle, as ``tiled``). The room and the path tracer's materials are ``rtx``'s, as they are
for ``tiled``.

Each process holds one EGL display, opened on one GPU the first time a context
is made: ``egl_device`` if given, else ``MUJOCO_EGL_DEVICE_ID``, else the GPU
the model is on -- for a model on a CPU, the first GPU ``CUDA_VISIBLE_DEVICES``
leaves the process -- so a worker pinned to a card by ``CUDA_VISIBLE_DEVICES``
draws on that card. A process whose ``CUDA_VISIBLE_DEVICES`` hides every one
of the machine's NVIDIA GPUs is refused rather than drawn on the first of
them, which may be someone else's.
A machine with no GPU rasterises in software with ``MUJOCO_GL=osmesa``, set
before the process starts, or through Mesa's EGL where that is installed. A
Mac has no EGL: there the context is Apple's CGL, headless, as MuJoCo's own
renderer opens it (``MUJOCO_GL=glfw`` opens a hidden window instead).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import newton
import numpy as np

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.core.render.base import FrameSet, RendererBase
from robosimstudio.core.render.spec import put
from robosimstudio.solvers.render.tiled import SEG_PARTICLES, SEG_SOFT_MESH

__all__ = ["RendererGL", "egl_device_for"]

# CameraSpec looks down +z with +y down; a MuJoCo camera looks down -z with +y up.
_VISION_TO_GL = np.diag([1.0, -1.0, -1.0])

_DRAW_DISTANCE = 1000.0
"""How far colour is drawn [m]: the tiled tracer's ``max_distance``, so the two agree on a horizon."""

MAX_SPHERES = 50_000
"""The most grains drawn as spheres a camera: each is one MuJoCo geom set from Python, about 5 us, so a heap of
more draws every k-th grain, each k^(1/3) times larger, and says so once."""

SMALL_BATCH = 1000
"""A particle body this small is never thinned: thinning is for heaps, and a few loose particles would show it."""


def sphere_strides(sizes: Sequence[int], budget: int = MAX_SPHERES) -> list[int]:
    """Per batch of grains, draw every k-th: 1 for all when they fit the budget; else the small batches whole
    and the large ones one stride, enough to bring them under what the small ones leave -- or, when the small
    ones take more than half of it (a heap would otherwise shrink to a few huge spheres), every batch one stride. Each batch's ``ceil(n / k)`` then sums to the budget plus at
    most one a batch, which is what the scene is sized for."""
    total = sum(sizes)
    if total <= budget:
        return [1] * len(sizes)
    small = sum(n for n in sizes if n < SMALL_BATCH)
    if small > budget // 2:  # small bodies filling the budget by themselves: every batch one stride, alike
        return [-(-total // budget)] * len(sizes)
    stride = -(-(total - small) // (budget - small))
    return [1 if n < SMALL_BATCH else stride for n in sizes]


_KEY_DIFFUSE = 0.6
"""A light's diffuse level per unit of intensity: the key's at ``key_intensity`` 1."""

_EGL_CUDA_DEVICE_NV = 0x323A
_EGL_EXTENSIONS = 0x3055


def _wound_out(verts: np.ndarray, normals: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """A closed mesh's faces wound the way its vertex normals point: MuJoCo culls a face by its winding, so a tube
    wound inward (the shared tube builder's order) showed only the inside of its far wall -- a hairline on the
    bench, at the bench's depth, which a plate composite then trimmed away as the bench."""
    a, b, c = (verts[faces[:, k]] for k in range(3))
    along = np.einsum("ij,ij->i", np.cross(b - a, c - a), normals[faces].sum(axis=1))
    return faces[:, ::-1].copy() if np.count_nonzero(along < 0) > len(faces) // 2 else faces


def _area(verts: np.ndarray, faces: np.ndarray) -> float:
    """A triangle mesh's surface area [m^2]."""
    if not len(faces):
        return 0.0
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    return float(0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum())


def _quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    import mujoco

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation, dtype=np.float64).ravel())
    return quat


PLATFORMS = {"darwin": ("cgl", "glfw")}
"""The OpenGL platforms this opens a context on, by ``sys.platform``: CGL (headless) and GLFW on a Mac, which has
no EGL; EGL and OSMesa everywhere else -- where a ``glfw`` named for MuJoCo's viewer is not taken, since a GLFW
window needs a display a headless server does not have."""


def _platform() -> str:
    """The OpenGL platform, decided once for the process: one of :data:`PLATFORMS`' for this OS.

    PyOpenGL binds one platform the first time it is imported, and importing
    ``mujoco.egl`` sets ``PYOPENGL_PLATFORM=egl`` before it can fail -- after
    which ``mujoco.osmesa`` refuses to import. So there is no falling back from
    one to the other inside a process: the platform is what
    ``PYOPENGL_PLATFORM`` or ``MUJOCO_GL`` already names, else CGL on a Mac
    (which has no EGL) and EGL elsewhere -- MuJoCo's own defaults.
    """
    here = PLATFORMS.get(sys.platform, ("egl", "osmesa"))
    for variable in ("PYOPENGL_PLATFORM", "MUJOCO_GL"):
        named = os.environ.get(variable, "").strip().lower()
        if named in here:
            return named
    return here[0]


def _egl_devices() -> list[tuple[bool, int | None]]:
    """Each EGL device, in EGL's order: whether it is an NVIDIA GPU, and which CUDA device it is to this process.

    EGL lists every GPU on the machine whatever ``CUDA_VISIBLE_DEVICES`` says,
    and MuJoCo takes the first one that opens -- physical GPU 0, someone else's
    card on a shared machine. NVIDIA's ``EGL_CUDA_DEVICE_NV`` says which CUDA
    device an EGL device is, in this process's own numbering, and nothing for
    a GPU the process cannot see: with ``CUDA_VISIBLE_DEVICES=3,5``, EGL device
    5 is CUDA device 1, and EGL device 0 is ``(True, None)``. Empty when the
    driver cannot say.
    """
    import ctypes

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        from mujoco.egl import egl_ext
        from OpenGL import EGL

        attribute = egl_ext.eglGetProcAddress(b"eglQueryDeviceAttribEXT")
        string = egl_ext.eglGetProcAddress(b"eglQueryDeviceStringEXT")
        if not attribute or not string:
            return []
        query = ctypes.CFUNCTYPE(EGL.EGLBoolean, ctypes.c_void_p, EGL.EGLint, ctypes.POINTER(ctypes.c_ssize_t))(
            attribute
        )
        extensions = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, EGL.EGLint)(string)
        found: list[tuple[bool, int | None]] = []
        for device in egl_ext.eglQueryDevicesEXT():
            handle = ctypes.cast(device, ctypes.c_void_p)
            value = ctypes.c_ssize_t(-1)
            seen = bool(query(handle, _EGL_CUDA_DEVICE_NV, ctypes.byref(value)))
            nvidia = b"EGL_NV_device_cuda" in (extensions(handle, _EGL_EXTENSIONS) or b"")
            found.append((nvidia, int(value.value) if seen else None))
        return found
    except Exception:  # a driver without the extensions: MuJoCo's own choice stands
        return []


def egl_device_for(cuda_ordinal: int, devices: Sequence[tuple[bool, int | None]] | None = None) -> int | None:
    """The EGL device that is this process's CUDA device ``cuda_ordinal``, or ``None`` if none is.

    ``devices`` is :func:`_egl_devices`'s list, asked for when not given.
    """
    devices = _egl_devices() if devices is None else devices
    return next((index for index, (_, cuda) in enumerate(devices) if cuda == cuda_ordinal), None)


def _context(width: int, height: int, egl_device: int | None, cuda_ordinal: int) -> Any:
    """A current GL context on the platform :func:`_platform` names.

    Not ``mujoco.Renderer``'s, which takes its platform from ``MUJOCO_GL`` as
    ``import mujoco`` runs -- before any scene is built -- and is GLFW, which
    needs a display, when the variable is unset.

    The EGL device, which the process's one EGL display is opened on the first
    time a context is made: ``egl_device`` if given, else ``MUJOCO_EGL_DEVICE_ID``
    if set, else the GPU that is CUDA device ``cuda_ordinal`` to this process
    (:func:`egl_device_for`). A process whose ``CUDA_VISIBLE_DEVICES`` hides
    every NVIDIA GPU is refused: MuJoCo would take the first of them. Without
    the variable, a GPU CUDA cannot use -- a graphics-only container -- is
    still MuJoCo's to choose.
    """
    platform = _platform()
    if platform == "egl" and egl_device is None and "MUJOCO_EGL_DEVICE_ID" not in os.environ:
        devices = _egl_devices()
        egl_device = egl_device_for(cuda_ordinal, devices)
        hidden = "CUDA_VISIBLE_DEVICES" in os.environ and all(cuda is None for _, cuda in devices)
        if egl_device is None and hidden and any(nvidia for nvidia, _ in devices):
            raise RuntimeError(
                "the gl renderer can see no GPU here (CUDA_VISIBLE_DEVICES="
                f"{os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}), and EGL would draw on the machine's first, "
                "which may be someone else's; make one visible, name an EGL device with MUJOCO_EGL_DEVICE_ID, "
                "or set MUJOCO_GL=osmesa before the process starts"
            )
    try:
        if platform == "egl":
            if egl_device is not None:
                os.environ["MUJOCO_EGL_DEVICE_ID"] = str(int(egl_device))
            from mujoco.egl import GLContext
        elif platform == "cgl":
            from mujoco.cgl import GLContext
        elif platform == "glfw":
            from mujoco.glfw import GLContext
        else:
            from mujoco.osmesa import GLContext
        context = GLContext(width, height)
        context.make_current()
        return context
    except Exception as exc:
        other = {"egl": "osmesa", "osmesa": "egl", "cgl": "glfw", "glfw": "cgl"}[platform]
        raise RuntimeError(
            f"the gl renderer could not open an OpenGL context on {platform} ({type(exc).__name__}: {exc}); "
            f"MUJOCO_GL={other}, set before the process starts, tries {other} instead"
        ) from exc


def _rgba(mesh: Any) -> tuple[list[float], Any, Any]:
    """A part's flat colour, and its base-colour texture and UVs when it has both.

    MuJoCo multiplies a texture by its geom's colour, so a textured part's
    colour is its stated base-colour factor or white -- not trimesh's
    ``main_color``, which is a 40 % grey when no factor is stated.
    """
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    image = getattr(material, "baseColorTexture", None)
    uv = getattr(visual, "uv", None)
    if image is not None and uv is not None and len(uv) == len(mesh.vertices):
        factor = getattr(material, "baseColorFactor", None)
        rgba = [1.0, 1.0, 1.0, 1.0] if factor is None else _unit(factor)
        return rgba, image, np.asarray(uv, dtype=np.float64)
    color = getattr(material, "main_color", None)
    return ([0.6, 0.6, 0.6, 1.0] if color is None else _unit(color)), None, None


def _unit(color: Any) -> list[float]:
    """An RGBA colour in ``[0, 1]``, from trimesh's bytes."""
    rgba = [float(c) / 255.0 for c in np.asarray(color).reshape(-1)[:4]]
    return rgba + [1.0] * (4 - len(rgba))


class RendererGL(RendererBase):
    """Every camera through MuJoCo's rasteriser, one at a time, off the model's visible shapes.

    Args:
        model: The finalized model. One world.
        cameras: The cameras to render. They may differ in resolution.
        channels: Restrict every camera to these channels.
        shape_labels: ``shape index -> label id`` for ``seg``. Without it
            ``seg`` is Newton's shape indices plus one, so a miss stays 0.
        light_direction: Direction the key light travels.
        key_intensity: The key light's brightness, on the scale of the fills'.
        fill_lights: ``((direction, intensity), ...)``, drawn without shadows.
        shadows: Whether the key light casts shadows. Off by default, because a
            shadow map is a second pass per light per camera.
        background: RGB in [0, 1] behind everything; black when ``None``.
        msaa: Multisample count for the offscreen buffer, 0 for none.
        egl_device: The EGL device to render on, over ``MUJOCO_EGL_DEVICE_ID``
            and the model's GPU. The process's first context fixes the device
            for every later one.
        deformables: ``name -> DeformableHandle``, normally the scene handles'
            table; without it every triangle of the model is one mesh and every
            other particle a sphere.
        deformable_colors: ``name -> RGB`` for those.
    """

    name = "gl"
    supported_channels = ("rgb", "depth", "seg")
    devices = ("cpu", "cuda")
    wants: tuple[str, ...] = ("shape_labels", "deformables", "deformable_colors")

    @classmethod
    def look_kwargs(cls, look: Any) -> dict[str, Any]:
        """What this backend takes from a look; see :meth:`RendererBase.look_kwargs`."""
        out: dict[str, Any] = {}
        put(out, "light_direction", look.light_direction)
        put(out, "key_intensity", look.key_intensity)
        put(out, "fill_lights", look.fill_lights)
        put(out, "background", look.background)
        return out

    def __init__(
        self,
        model: newton.Model,
        cameras: Sequence[CameraSpec],
        *,
        channels: Sequence[str] | None = None,
        shape_labels: Mapping[int, int] | None = None,
        light_direction: Sequence[float] = (-1.0, 1.0, -1.0),
        key_intensity: float = 1.0,
        fill_lights: Sequence[tuple[Sequence[float], float]] = (),
        shadows: bool = False,
        background: Sequence[float] | None = None,
        msaa: int = 4,
        egl_device: int | None = None,
        deformables: Mapping[str, Any] | None = None,
        deformable_colors: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        super().__init__(cameras, channels=channels)
        import mujoco

        from robosimstudio.core.render.deformables import deformable_drawables
        from robosimstudio.core.render.visual import model_visual_parts

        if int(getattr(model, "world_count", 1) or 1) > 1:
            raise ValueError("the gl renderer draws one world; a fleet renders through 'tiled'")
        self.model = model
        self._mujoco = mujoco

        spec = mujoco.MjSpec()
        width = max(int(c.width) for c in self.cameras)
        height = max(int(c.height) for c in self.cameras)
        spec.visual.global_.offwidth, spec.visual.global_.offheight = width, height
        spec.visual.quality.offsamples = max(0, int(msaa))
        # MuJoCo's clip planes are fractions of the model's extent, which a
        # ground plane drawn a hundred metres wide would make huge, clipping
        # everything within a metre of a camera.
        spec.stat.extent = 1.0
        spec.visual.map.znear = min(float(c.near) for c in self.cameras)
        # Colour out to the tiled tracer's 1000 m, so a floor runs to the horizon in
        # both; depth past a camera's own far is still a miss (see render).
        spec.visual.map.zfar = max(_DRAW_DISTANCE, *(float(c.far) for c in self.cameras))
        spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
        spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
        spec.visual.headlight.specular = [0.0, 0.0, 0.0]
        if background is not None:
            rgb = [float(c) for c in background][:3]
            spec.add_texture(
                name="background", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT, rgb1=rgb, rgb2=rgb, width=8, height=8,
            )  # fmt: skip
        rig = [(tuple(light_direction), float(key_intensity))] + [(tuple(d), float(i)) for d, i in fill_lights]
        for index, (direction, intensity) in enumerate(rig):
            light = spec.worldbody.add_light(
                name=f"light{index}", dir=list(direction), castshadow=bool(shadows and index == 0)
            )
            light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            light.diffuse = [_KEY_DIFFUSE * intensity] * 3
            light.specular = [0.1, 0.1, 0.1]

        bodies: dict[int, Any] = {}
        for part in model_visual_parts(model):
            mesh = part.mesh
            if len(getattr(mesh, "faces", ())) == 0:
                continue
            if part.body >= 0 and part.body not in bodies:
                body = spec.worldbody.add_body(name=f"body{part.body}", mocap=True)
                # a stated inertia, so MuJoCo infers none from the meshes: a flat one (a photo on a slab) has
                # A + B = C to the rounding, and MuJoCo refuses A + B < C for a body that only carries pictures
                body.explicitinertial = True
                body.mass, body.inertia = 1.0, [1.0, 1.0, 1.0]
                bodies[part.body] = body
            parent = bodies[part.body] if part.body >= 0 else spec.worldbody
            rgba, image, uv = _rgba(mesh)
            asset = spec.add_mesh(
                name=f"mesh{part.shape}",
                uservert=np.asarray(mesh.vertices, dtype=np.float64).ravel().tolist(),
                userface=np.asarray(mesh.faces, dtype=np.int32).ravel().tolist(),
            )
            # A render-only mesh has no mass to give; a flat one has no volume.
            asset.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            material = ""
            if image is not None:
                asset.usertexcoord = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]]).ravel().tolist()
                pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
                texture = spec.add_texture(
                    name=f"texture{part.shape}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                    width=pixels.shape[1], height=pixels.shape[0], nchannel=3,
                )  # fmt: skip
                texture.data = pixels.tobytes()
                material = f"material{part.shape}"
                spec.add_material(name=material, textures=["", f"texture{part.shape}"])
            parent.add_geom(
                name=f"shape{part.shape}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"mesh{part.shape}",
                contype=0, conaffinity=0, rgba=rgba, material=material,
                pos=part.local[:3, 3].tolist(), quat=_quat_wxyz(part.local[:3, :3]).tolist(),
            )  # fmt: skip
        # An open sheet twice over, the second copy wound and lit the other way, so a cloth seen from behind
        # is drawn (a rope's tube is closed and drawn once); its vertices are rewritten every frame (_deform),
        # its topology never changes.
        self._drawables = deformable_drawables(model, deformables, deformable_colors)
        particle_q = model.particle_q.numpy() if self._drawables.meshes else None
        for index, part in enumerate(self._drawables.meshes):
            verts, normals = (np.nan_to_num(a) for a in part.vertices(particle_q))  # a collapsed tube has no frame
            faces = np.asarray(part.triangles, dtype=np.int32)
            if _area(verts, faces) < 1.0e-10:
                # collapsed as built (particles on one point): MuJoCo refuses a mesh with no area, and these
                # vertices are rewritten before every frame, so the compile is given a spread copy
                verts = verts + 1.0e-3 * np.stack(
                    [np.arange(len(verts)) % 7, np.arange(len(verts)) % 5, np.arange(len(verts)) % 3], 1
                )
            if part.tube_radius <= 0.0:
                faces = np.concatenate([faces, faces[:, ::-1] + len(verts)])
                verts, normals = np.concatenate([verts, verts]), np.concatenate([normals, -normals])
            else:
                faces = _wound_out(verts, normals, faces)
            asset = spec.add_mesh(
                name=f"deformable{index}",
                uservert=verts.ravel().tolist(),
                userface=faces.ravel().tolist(),
                usernormal=normals.ravel().tolist(),
            )
            asset.userfacenormal = faces.ravel().tolist()
            asset.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            spec.worldbody.add_geom(
                name=f"deformable{index}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"deformable{index}",
                contype=0, conaffinity=0, rgba=[*part.color, 1.0],
            )  # fmt: skip
        self._points = sum(len(p.particles) for p in self._drawables.points)
        for cam in self.cameras:
            body = spec.worldbody.add_body(name=f"camera_{cam.name}", mocap=True)
            k = cam.intrinsics
            camera = body.add_camera(name=cam.name)
            camera.resolution = [int(cam.width), int(cam.height)]
            camera.sensor_size = [float(cam.width), float(cam.height)]
            camera.focal_pixel = [float(k[0, 0]), float(k[1, 1])]
            # The centre's offset from the principal point, the half-pixel
            # included: MuJoCo moves the picture the other way from K's cx, cy.
            # Measured against tiled: cx 80 px right of centre put this picture
            # 160 px from tiled's with the sign of K, and on it with this one.
            camera.principal_pixel = [(cam.width - 1) / 2.0 - float(k[0, 2]), (cam.height - 1) / 2.0 - float(k[1, 2])]

        self.mj_model = spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)
        self._bodies = np.array(sorted(bodies), dtype=np.int64)
        self._body_mocap = np.array([self.mj_model.body(f"body{b}").mocapid[0] for b in self._bodies], dtype=np.int64)
        self._camera_mocap = [int(self.mj_model.body(f"camera_{c.name}").mocapid[0]) for c in self.cameras]
        self._camera_id = [int(self.mj_model.camera(c.name).id) for c in self.cameras]

        # A geom's segmentation id back to the label its shape carries; 0 for a miss. A deformable's mesh is the
        # tiled tracer's soft mesh (a rope's tube its particles), and the last row is every particle drawn as a sphere.
        lut = np.zeros(self.mj_model.ngeom + 2, dtype=np.int32)
        labels = {int(s): int(v) for s, v in (shape_labels or {}).items()}
        for geom in range(self.mj_model.ngeom):
            name = self.mj_model.geom(geom).name
            if name.startswith("shape"):
                shape = int(name[len("shape") :])
                lut[geom + 1] = labels.get(shape, 0) if shape_labels is not None else shape + 1
            elif name.startswith("deformable"):
                part = self._drawables.meshes[int(name[len("deformable") :])]
                lut[geom + 1] = SEG_PARTICLES if part.tube_radius > 0.0 else SEG_SOFT_MESH
        lut[-1] = SEG_PARTICLES
        self._seg_lut = lut
        self._deforming = [
            (
                int(self.mj_model.mesh(f"deformable{i}").id),
                int(self.mj_model.geom(f"deformable{i}").id),
                part,
            )
            for i, part in enumerate(self._drawables.meshes)
        ]
        self._seg_geoms: np.ndarray | None = None
        """A drawn geom's segmentation id back to its model geom plus one; see :meth:`_geoms_by_segid`."""

        import warp as wp

        device = wp.get_device(model.device)
        # A CPU model has no CUDA device of its own; it draws on the first the process can see.
        self._context = _context(width, height, egl_device, int(device.ordinal) if device.is_cuda else 0)
        self._gl = mujoco.MjrContext(self.mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._gl)
        drawn = min(self._points, MAX_SPHERES + len(self._drawables.points))  # a stride's rounding, per batch
        self._scene = mujoco.MjvScene(self.mj_model, maxgeom=self.mj_model.ngeom + drawn + 64)
        self._particle_q: np.ndarray | None = None
        self._point_frame: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        """This frame's spheres, per batch: centres, sizes and colour (see :meth:`_add_points`)."""
        self._point_range = (0, 0)
        """Where in the drawn scene :meth:`_add_points` put its spheres, for the segmentation lookup."""
        self._told_full = False
        self._strides = sphere_strides([len(b.particles) for b in self._drawables.points])
        # each batch's grains as drawn: every k-th of a heap past the budget, each 3rd-root-of-k larger, so it keeps
        # its volume; the sizes and colours fixed, the positions gathered each frame
        self._point_batches = []
        for batch, stride in zip(self._drawables.points, self._strides, strict=True):
            sizes = np.zeros((len(batch.particles[::stride]), 3))
            sizes[:, 0] = batch.radii[::stride] * stride ** (1.0 / 3.0)
            self._point_batches.append((batch.particles[::stride], sizes, np.array([*batch.color, 1.0], np.float32)))
        if any(k > 1 for k in self._strides):
            thinned = [
                f"{b.name or 'particles'} every {k}th"
                for b, k in zip(self._drawables.points, self._strides, strict=True)
                if k > 1
            ]
            print(
                f"[gl] {self._points} grains, past {MAX_SPHERES} spheres a camera: drawing {', '.join(thinned)}, larger",
                flush=True,
            )
        self._option = mujoco.MjvOption()
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

    # ------------------------------------------------------------------ render
    def _pose(self, body_q: np.ndarray | None) -> None:
        """Every drawn body and every camera where ``body_q`` puts them, then MuJoCo's kinematics."""
        mujoco, d = self._mujoco, self.mj_data
        if len(self._bodies) and body_q is not None:
            q = body_q[self._bodies]
            d.mocap_pos[self._body_mocap] = q[:, :3]
            d.mocap_quat[self._body_mocap] = q[:, [6, 3, 4, 5]]  # xyzw -> wxyz
        for cam, mocap in zip(self.cameras, self._camera_mocap, strict=True):
            pose = cam.pose_on_body(body_q)
            d.mocap_pos[mocap] = pose[:3, 3]
            d.mocap_quat[mocap] = _quat_wxyz(pose[:3, :3] @ _VISION_TO_GL)
        mujoco.mj_kinematics(self.mj_model, d)
        # Cameras and lights are placed here, not by mj_kinematics.
        mujoco.mj_camlight(self.mj_model, d)

    def _deform(self, particle_q: np.ndarray) -> None:
        """Every deformable mesh's vertices and normals where ``particle_q`` puts them, uploaded to the GL context.

        A mesh's vertices are in its geom's frame, which MuJoCo moved to the mesh's own centre when it compiled
        it; the geom hangs off the world and never moves, so the world vertices are taken into its frame.
        """
        mujoco, m, d = self._mujoco, self.mj_model, self.mj_data
        for mesh, geom, part in self._deforming:
            verts, normals = (np.nan_to_num(a) for a in part.vertices(particle_q))
            rotation, origin = d.geom_xmat[geom].reshape(3, 3), d.geom_xpos[geom]
            local = (verts - origin) @ rotation
            turned = normals @ rotation
            if part.tube_radius <= 0.0:  # a sheet's back copy
                local, turned = np.concatenate([local, local]), np.concatenate([turned, -turned])
            n = len(local)
            start, nstart = int(m.mesh_vertadr[mesh]), int(m.mesh_normaladr[mesh])
            m.mesh_vert[start : start + n] = local
            m.mesh_normal[nstart : nstart + n] = turned
            mujoco.mjr_uploadMesh(m, self._gl, mesh)

    def _add_points(self) -> None:
        """Every particle drawn as a sphere, appended to the scene after MuJoCo's own geoms.

        One ``mjv_initGeom`` a particle, from the positions :meth:`render` gathered once for the frame: about
        5 us each, so :data:`MAX_SPHERES` of them cost a quarter of a second a camera.
        """
        mujoco, scene = self._mujoco, self._scene
        self._point_range = (int(scene.ngeom), int(scene.ngeom))
        init, sphere, eye = mujoco.mjv_initGeom, mujoco.mjtGeom.mjGEOM_SPHERE, np.eye(3).ravel()
        for centres, sizes, rgba in self._point_frame:
            room = min(len(centres), int(scene.maxgeom) - int(scene.ngeom))
            if room < len(centres) and not self._told_full:
                self._told_full = True
                print(
                    f"[gl] the scene holds {scene.maxgeom} geoms: {len(centres) - room} grains are left out", flush=True
                )
            for k in range(room):
                n = scene.ngeom
                geom = scene.geoms[n]
                init(geom, sphere, sizes[k], centres[k], eye, rgba)
                geom.segid = n
                scene.ngeom = n + 1
        self._point_range = (self._point_range[0], int(scene.ngeom))

    def _draw(self, cam: CameraSpec, camera_id: int, segment: bool, update: bool = True) -> Any:
        """Draw one camera; ``update=False`` redraws the scene the last call built (rgb, then seg, of one camera)."""
        mujoco = self._mujoco
        if update:
            self._camera.fixedcamid = camera_id
            mujoco.mjv_updateScene(
                self.mj_model, self.mj_data, self._option, None, self._camera, mujoco.mjtCatBit.mjCAT_ALL, self._scene
            )
            self._add_points()
        self._scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = segment
        self._scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = segment
        viewport = mujoco.MjrRect(0, 0, int(cam.width), int(cam.height))
        mujoco.mjr_render(viewport, self._scene, self._gl)
        return viewport

    def _geoms_by_segid(self) -> np.ndarray:
        """``segid - 1 -> model geom + 1`` (0 for a drawn thing that is not a geom), for the scene just drawn.

        The scene holds the same geoms in the same order every frame for one
        model and one option, so the table is read off its geoms once rather
        than per camera per frame, and again only if their count changes.
        """
        scene = self._scene
        start, end = self._point_range
        if self._seg_geoms is None or len(self._seg_geoms) != start:
            geoms = scene.geoms[:start]
            kinds = np.array([g.objtype for g in geoms], dtype=np.int64)
            objects = np.array([g.objid for g in geoms], dtype=np.int64)
            self._seg_geoms = np.where(kinds == int(self._mujoco.mjtObj.mjOBJ_GEOM), objects + 1, 0)
        if end > start:  # the spheres _add_points appended after MuJoCo's own geoms: the lookup's last row
            return np.concatenate([self._seg_geoms, np.full(end - start, len(self._seg_lut) - 1)])
        return self._seg_geoms

    def render(self, state: newton.State) -> FrameSet:
        """Every camera for one state. Each array is the caller's to keep."""
        mujoco = self._mujoco
        self._context.make_current()
        self._pose(state.body_q.numpy() if state.body_q is not None else None)
        self._point_frame = []
        if self._drawables:
            self._particle_q = state.particle_q.numpy() if state.particle_q is not None else None
            if self._particle_q is not None and self._deforming:
                self._deform(self._particle_q)
            if self._particle_q is not None:
                for particles, sizes, rgba in self._point_batches:  # gathered once a frame, drawn once a camera
                    self._point_frame.append((np.asarray(self._particle_q[particles], dtype=np.float64), sizes, rgba))
        extent = float(self.mj_model.stat.extent)
        near = float(self.mj_model.vis.map.znear) * extent
        far = float(self.mj_model.vis.map.zfar) * extent
        frames: FrameSet = {}
        for cam, camera_id in zip(self.cameras, self._camera_id, strict=True):
            wanted = self.channels[cam.name]
            frame: dict[str, np.ndarray] = {}
            h, w = int(cam.height), int(cam.width)
            if "rgb" in wanted or "depth" in wanted:
                viewport = self._draw(cam, camera_id, segment=False)
                rgb = np.empty((h, w, 3), dtype=np.uint8) if "rgb" in wanted else None
                depth = np.empty((h, w), dtype=np.float32) if "depth" in wanted else None
                mujoco.mjr_readPixels(rgb, depth, viewport, self._gl)
                if rgb is not None:
                    frame["rgb"] = rgb[::-1].copy()  # GL's rows run bottom-up
                if depth is not None:
                    # The depth buffer, linearised to metres along the view axis;
                    # past the far plane, or the camera's own far, is a miss: 0.
                    metres = near / (1.0 - depth[::-1] * (1.0 - near / far))
                    miss = (depth[::-1] >= 1.0) | (metres > float(cam.far))
                    frame["depth"] = np.where(miss, 0.0, metres).astype(np.float32)
            if "seg" in wanted:
                drawn = "rgb" in wanted or "depth" in wanted  # the scene just built for this camera, drawn again
                viewport = self._draw(cam, camera_id, segment=True, update=not drawn)
                ids = np.empty((h, w, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(ids, None, viewport, self._gl)
                ids = ids[::-1].astype(np.int64)
                segid = ids[..., 0] + (ids[..., 1] << 8) + (ids[..., 2] << 16)
                geom = np.zeros_like(segid)
                hit = segid > 0
                table = self._geoms_by_segid() if hit.any() else None
                if table is not None and len(table):
                    geom = np.where(hit, table[np.clip(segid - 1, 0, len(table) - 1)], 0)
                frame["seg"] = self._seg_lut[geom].astype(np.int32)
            frames[cam.name] = frame
        return frames

    def close(self) -> None:
        """Free the GL context. Idempotent."""
        gl, context = getattr(self, "_gl", None), getattr(self, "_context", None)
        self._gl = self._context = None
        if context is not None:
            try:
                context.make_current()
                if gl is not None:
                    gl.free()
            finally:
                context.free()
