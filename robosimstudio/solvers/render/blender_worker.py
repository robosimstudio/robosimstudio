# SPDX-License-Identifier: Apache-2.0
"""The Blender side of the ``blender`` renderer: a scene held in Blender, rendered a camera at a time on request.

Runs inside Blender's own Python -- ``blender -b --factory-startup --python
blender_worker.py -- PORT``, or any interpreter that imports ``bpy`` -- and so
imports nothing from this repository: the standard library, ``bpy``, NumPy
(Blender bundles it) and OpenImageIO (bundled since Blender 4.0) are all it
has. :class:`~robosimstudio.solvers.render.blender.RendererBlender` starts it,
connects it back over a loopback socket and speaks the protocol below; it also
imports this module for :func:`send` and :func:`receive`, which is why ``bpy``
is imported inside :func:`main` only.

**The protocol.** A message is a JSON header and a list of binary blobs, each
length-prefixed (8 bytes, little-endian); the header names a blob by its index.

* ``scene``: the meshes (vertices, faces, UVs, a texture, a material), which
  body each hangs off and where, the deformables' fixed topology, the particles
  drawn as spheres, the particle surfaces, the lights, the world, the engine
  and its sampling. Built once.
* ``frame``: each body's world matrix, each deformable's vertices, the
  particles' positions, each particle surface's vertices and triangles (their
  counts change), and the cameras (intrinsics, size, pose, channels).
  Answered with each camera's RGB (``uint8 [H, W, 3]``, sRGB) and depth
  (``float32 [H, W]``, metres to the image plane, 0 for a miss).
* ``quit``.

**What a camera is.** ``K`` goes to Blender's camera as a focal length on a
horizontal sensor, a pixel aspect for ``fx != fy`` and a lens shift for the
principal point, pixel centres included, so a calibrated camera keeps its
principal point to the pixel. The picture is read off the EXR the render
writes (scene-linear, encoded to sRGB here, which is Blender's ``Standard``
view); depth is a second render in which every surface emits its own distance
to the image plane, since Blender 5 writes the Combined pass alone.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import sys
import tempfile
import time
from contextlib import contextmanager

_LEN = struct.Struct("<Q")
TOKEN_ENV = "ROBOSIMSTUDIO_BLENDER_TOKEN"
"""What the worker says first, from its environment: the renderer drops a connection that does not."""


# ------------------------------------------------------------------ framing (shared with the client)
def _read_exact(sock: socket.socket, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        chunk = sock.recv(min(n - got, 1 << 22))
        if not chunk:
            raise ConnectionError("the other end closed the connection")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def send(sock: socket.socket, header: dict, blobs: list | tuple = ()) -> None:
    """One message: the header, then each blob, each length-prefixed."""
    head = json.dumps({**header, "nblobs": len(blobs)}).encode("utf-8")
    parts = [_LEN.pack(len(head)), head]
    for blob in blobs:
        data = blob
        if not isinstance(blob, bytes | bytearray):
            view = memoryview(blob)
            # a strided view is copied, not refused; an empty one (a surface with nothing this frame) cannot be cast
            data = view.cast("B") if view.c_contiguous and view.nbytes else view.tobytes()
        parts += [_LEN.pack(len(data)), data]
    for part in parts:
        sock.sendall(part)


def receive(sock: socket.socket) -> tuple[dict, list[bytes]]:
    """One message: the header and its blobs."""
    (n,) = _LEN.unpack(_read_exact(sock, _LEN.size))
    header = json.loads(_read_exact(sock, n).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"not a message: a {type(header).__name__}")
    blobs = []
    for _ in range(int(header.get("nblobs", 0))):
        (size,) = _LEN.unpack(_read_exact(sock, _LEN.size))
        blobs.append(_read_exact(sock, size))
    return header, blobs


# ------------------------------------------------------------------ the scene, in Blender
VISION_TO_BLENDER = ((1.0, 0.0, 0.0, 0.0), (0.0, -1.0, 0.0, 0.0), (0.0, 0.0, -1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
"""A camera's +z-forward, +y-down basis to Blender's -z-forward, +y-up one."""

ENGINES = {"cycles": "CYCLES", "eevee": "BLENDER_EEVEE"}
DRAW_DISTANCE = 1000.0
"""How far colour is drawn [m]: gl's and the tiled tracer's horizon."""
GPU_KINDS = ("CUDA", "METAL", "HIP", "ONEAPI", "OPTIX")
"""Cycles' GPU backends in the order ``auto`` tries them. CUDA before OptiX: on one frame at a time the OptiX
path rebuilds its acceleration structure and denoiser per render, measured at 2.4 s a 960x540 frame against
CUDA's 0.8 s on the same RTX PRO 6000."""


def camera_settings(w: int, h: int, fx: float, fy: float, cx: float, cy: float, sensor_width: float = 36.0) -> dict:
    """A pinhole ``K`` as a Blender camera on a horizontally fitted sensor: lens, shift and pixel aspect.

    Square pixels unless ``fx != fy``: the vertical focal length is ``fx * aspect_x / aspect_y``, so the
    aspect takes their ratio, the larger side 1. The principal point is a lens shift, which a horizontally
    fitted sensor measures in widths -- both ways, a portrait frame included; ``K`` puts pixel centres on
    integers and Blender on half-integers, hence the half pixel, and Blender's ``shift_y`` runs up while an
    image's rows run down.
    """
    aspect = (1.0, fx / fy) if fx >= fy else (fy / fx, 1.0)
    span = float(w)
    return {
        "lens": fx * sensor_width / w,
        "shift": (((w - 1) / 2.0 - cx) / span, (cy - (h - 1) / 2.0) * (fx / fy) / span),
        "pixel_aspect": aspect,
    }


class World:
    """The scene the client described, in ``bpy.data``, and the render of one frame of it."""

    def __init__(self, header: dict, blobs: list[bytes]) -> None:
        import bpy
        import numpy as np

        self.bpy, self.np = bpy, np
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.scene = scene = bpy.context.scene
        # under the working directory the client started this in, which it removes when it closes
        self.tmp = tempfile.mkdtemp(prefix="rss_blender_", dir=os.getcwd())
        self.images: dict[int, object] = {}
        self.device = self._engine(header)
        settings = scene.render.image_settings
        if hasattr(settings, "media_type"):  # Blender 5: the multilayer formats are a media type of their own
            settings.media_type = "MULTI_LAYER_IMAGE"
        settings.file_format = "OPEN_EXR_MULTILAYER"
        scene.render.image_settings.color_depth = "32"
        scene.render.image_settings.exr_codec = "NONE"  # read straight back and deleted: no use compressing it
        scene.render.use_compositing = False
        scene.render.use_sequencer = False
        scene.render.use_persistent_data = True  # Cycles keeps the scene between renders and updates what moved
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = False
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        self._world(header.get("world", {}))
        self._lights(header.get("lights", []))
        self.bodies: dict[int, object] = {}
        self.materials: dict[str, object] = {}
        for part in header.get("meshes", []):
            self._mesh(part, blobs)
        self.deformables = [self._deformable(d, blobs) for d in header.get("deformables", [])]
        self.points = [self._points(p, blobs) for p in header.get("points", [])]
        self.surfaces = [self._surface(d, blobs) for d in header.get("surfaces", [])]
        cam_data = bpy.data.cameras.new("camera")
        cam_data.sensor_fit = "HORIZONTAL"
        cam_data.sensor_width = 36.0
        self.camera = bpy.data.objects.new("camera", cam_data)
        scene.collection.objects.link(self.camera)
        scene.camera = self.camera

    # -------------------------------------------------------------- set-up
    def _engine(self, header: dict) -> str:
        bpy, scene = self.bpy, self.scene
        engine = str(header.get("engine", "cycles")).lower()
        scene.render.engine = ENGINES.get(engine, "CYCLES")
        samples = int(header.get("samples", 32))
        if scene.render.engine != "CYCLES":
            scene.eevee.taa_render_samples = samples
            return "eevee"
        cycles = scene.cycles
        cycles.samples = samples
        cycles.use_adaptive_sampling = True
        cycles.use_denoising = bool(header.get("denoise", True))
        cycles.max_bounces = int(header.get("bounces", 6))
        cycles.use_auto_tile = False
        threads = int(header.get("threads", 0))
        if threads > 0:
            scene.render.threads_mode = "FIXED"
            scene.render.threads = threads
        wanted = str(header.get("device", "auto")).upper()
        if wanted == "CPU":
            cycles.device = "CPU"
            return "CPU"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        kinds = [wanted] if wanted in GPU_KINDS else list(GPU_KINDS)
        for kind in kinds:
            try:
                prefs.compute_device_type = kind
            except TypeError:  # not a backend this build or this machine has
                continue
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == kind]
            if not gpus:
                continue
            for d in prefs.devices:
                d.use = d.type == kind
            cycles.device = "GPU"
            return f"{kind} ({', '.join(d.name for d in gpus)})"
        if wanted != "AUTO":
            raise RuntimeError(f"Cycles has no {wanted} device here")
        cycles.device = "CPU"
        return "CPU"

    def _world(self, world: dict) -> None:
        bpy = self.bpy
        w = bpy.data.worlds.new("world")
        self.scene.world = w
        w.use_nodes = True
        nodes, links = w.node_tree.nodes, w.node_tree.links
        nodes.clear()
        out = nodes.new("ShaderNodeOutputWorld")
        light = nodes.new("ShaderNodeBackground")
        light.inputs["Color"].default_value = (*world.get("color", (0.05, 0.05, 0.05)), 1.0)
        light.inputs["Strength"].default_value = float(world.get("strength", 1.0))
        background = world.get("background")
        if background is None:
            links.new(light.outputs["Background"], out.inputs["Surface"])
            return
        # what a camera ray sees where it hits nothing, apart from what lights the scene
        seen = nodes.new("ShaderNodeBackground")
        seen.inputs["Color"].default_value = (*background, 1.0)
        seen.inputs["Strength"].default_value = 1.0
        path = nodes.new("ShaderNodeLightPath")
        mix = nodes.new("ShaderNodeMixShader")
        links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
        links.new(light.outputs["Background"], mix.inputs[1])
        links.new(seen.outputs["Background"], mix.inputs[2])
        links.new(mix.outputs["Shader"], out.inputs["Surface"])

    def _lights(self, lights: list) -> None:
        bpy = self.bpy
        from mathutils import Matrix, Vector

        for i, spec in enumerate(lights):
            kind = spec["type"]
            data = bpy.data.lights.new(f"light{i}", "SUN" if kind == "sun" else "AREA")
            data.color = tuple(spec.get("color", (1.0, 1.0, 1.0)))
            data.energy = float(spec["energy"])
            data.use_shadow = bool(spec.get("shadow", True))
            if kind == "sun":
                data.angle = float(spec.get("angle", 0.05))
            else:
                data.shape = "RECTANGLE"
                data.size, data.size_y = (float(v) for v in spec["size"])
            obj = bpy.data.objects.new(f"light{i}", data)
            obj.visible_camera = False  # a light, not a thing: the panel is not in the picture, nor in the depth
            self.scene.collection.objects.link(obj)
            # a light shines down its local -z: turn that onto the direction it travels
            direction = Vector(spec.get("direction", (0.0, 0.0, -1.0))).normalized()
            rotation = Vector((0.0, 0.0, -1.0)).rotation_difference(direction).to_matrix().to_4x4()
            obj.matrix_world = Matrix.Translation(Vector(spec.get("position", (0.0, 0.0, 0.0)))) @ rotation

    def _material(self, spec: dict, blobs: list[bytes]):
        bpy = self.bpy
        key = json.dumps({k: spec.get(k) for k in ("color", "roughness", "metallic", "opacity", "texture", "box")})
        if key in self.materials:
            return self.materials[key]
        mat = bpy.data.materials.new(f"material{len(self.materials)}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        color = list(spec.get("color", (0.6, 0.6, 0.6, 1.0)))
        bsdf.inputs["Base Color"].default_value = (*color[:3], 1.0)
        bsdf.inputs["Roughness"].default_value = float(spec.get("roughness", 0.6))
        bsdf.inputs["Metallic"].default_value = float(spec.get("metallic", 0.0))
        alpha = float(spec.get("opacity", 1.0)) * (color[3] if len(color) > 3 else 1.0)
        if alpha < 1.0:
            bsdf.inputs["Alpha"].default_value = alpha
        texture = int(spec.get("texture", -1))
        if texture >= 0:
            image = self.images.get(texture)
            if image is None:  # one image per texture sent, however many shapes share it
                path = os.path.join(self.tmp, f"texture{texture}.png")
                with open(path, "wb") as f:
                    f.write(blobs[texture])
                image = self.images[texture] = bpy.data.images.load(path)
                image.colorspace_settings.name = "sRGB"
            node = mat.node_tree.nodes.new("ShaderNodeTexImage")
            node.image = image
            box = spec.get("box")
            if box:  # a finish's map in repeats per metre, box-projected in the object's own metres
                coords = mat.node_tree.nodes.new("ShaderNodeTexCoord")
                mapping = mat.node_tree.nodes.new("ShaderNodeMapping")
                mapping.inputs["Scale"].default_value = (float(box[0]), float(box[1]), float(box[0]))
                mapping.inputs["Location"].default_value = (float(box[2]), float(box[3]), 0.0)
                node.projection = "BOX"
                node.projection_blend = 0.1
                mat.node_tree.links.new(coords.outputs["Object"], mapping.inputs["Vector"])
                mat.node_tree.links.new(mapping.outputs["Vector"], node.inputs["Vector"])
            if color[:3] != [1.0, 1.0, 1.0]:  # a stated factor tints the map, as glTF's base colour factor does
                tint = mat.node_tree.nodes.new("ShaderNodeMix")
                tint.data_type = "RGBA"
                tint.blend_type = "MULTIPLY"
                tint.inputs["Factor"].default_value = 1.0
                mat.node_tree.links.new(node.outputs["Color"], tint.inputs["A"])
                tint.inputs["B"].default_value = (*color[:3], 1.0)
                mat.node_tree.links.new(tint.outputs["Result"], bsdf.inputs["Base Color"])
            else:
                mat.node_tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
        self.materials[key] = mat
        return mat

    def _new_mesh(self, name: str, verts, faces, uv=None):
        return self._fill(self.bpy.data.meshes.new(name), verts, faces, uv)

    def _fill(self, mesh, verts, faces, uv=None):
        """``mesh`` made of these vertices and triangles (and UVs per vertex), whatever it held before."""
        np = self.np
        mesh.clear_geometry()
        verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        mesh.vertices.add(len(verts))
        mesh.vertices.foreach_set("co", verts.ravel())
        mesh.loops.add(3 * len(faces))
        mesh.loops.foreach_set("vertex_index", faces.ravel())
        mesh.polygons.add(len(faces))
        mesh.polygons.foreach_set("loop_start", np.arange(0, 3 * len(faces), 3, dtype=np.int32))
        mesh.polygons.foreach_set("loop_total", np.full(len(faces), 3, dtype=np.int32))
        if uv is not None:
            layer = mesh.uv_layers.new(name="uv")
            layer.data.foreach_set("uv", np.asarray(uv, dtype=np.float32).reshape(-1, 2)[faces.ravel()].ravel())
        mesh.update(calc_edges=True)
        mesh.validate(clean_customdata=False)
        mesh.polygons.foreach_set("use_smooth", np.full(len(mesh.polygons), len(mesh.polygons) > 64))
        return mesh

    def _mesh(self, part: dict, blobs: list[bytes]) -> None:
        bpy, np = self.bpy, self.np
        from mathutils import Matrix

        verts = np.frombuffer(blobs[part["verts"]], dtype=np.float32)
        faces = np.frombuffer(blobs[part["faces"]], dtype=np.int32)
        uv = np.frombuffer(blobs[part["uv"]], dtype=np.float32) if part.get("uv", -1) >= 0 else None
        mesh = self._new_mesh(part["name"], verts, faces, uv)
        mesh.materials.append(self._material(part, blobs))
        obj = bpy.data.objects.new(part["name"], mesh)
        self.scene.collection.objects.link(obj)
        body = int(part.get("body", -1))
        local = Matrix([part["local"][4 * r : 4 * r + 4] for r in range(4)])
        if body >= 0:
            if body not in self.bodies:
                empty = bpy.data.objects.new(f"body{body}", None)
                self.scene.collection.objects.link(empty)
                self.bodies[body] = empty
            obj.parent = self.bodies[body]
        obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.matrix_basis = local

    def _deformable(self, spec: dict, blobs: list[bytes]):
        bpy, np = self.bpy, self.np
        faces = np.frombuffer(blobs[spec["faces"]], dtype=np.int32).reshape(-1, 3)
        mesh = self._new_mesh(spec["name"], np.frombuffer(blobs[spec["verts"]], dtype=np.float32), faces)
        mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), dtype=bool))
        mesh.materials.append(self._material(spec, blobs))
        obj = bpy.data.objects.new(spec["name"], mesh)
        self.scene.collection.objects.link(obj)
        return mesh

    def _points(self, spec: dict, blobs: list[bytes]):
        """Particles as one point cloud of spheres, which Cycles draws natively at each point's radius."""
        bpy, np = self.bpy, self.np
        radii = np.frombuffer(blobs[spec["radii"]], dtype=np.float32)
        cloud = bpy.data.pointclouds.new(spec["name"] or "points")
        cloud.resize(len(radii))
        cloud.points.foreach_set("radius", radii)
        cloud.materials.append(self._material(spec, blobs))
        obj = bpy.data.objects.new(spec["name"] or "points", cloud)
        self.scene.collection.objects.link(obj)
        return cloud

    def _surface(self, spec: dict, blobs: list[bytes]):
        """A particle body's surface: a mesh rebuilt every frame, its vertex and face counts changing."""
        bpy = self.bpy
        mesh = bpy.data.meshes.new(spec["name"])
        mesh.materials.append(self._material(spec, blobs))
        obj = bpy.data.objects.new(spec["name"], mesh)
        self.scene.collection.objects.link(obj)
        return mesh

    # -------------------------------------------------------------- one frame
    def frame(self, header: dict, blobs: list[bytes]) -> tuple[dict, list]:
        np = self.np
        from mathutils import Matrix

        if header.get("bodies", -1) >= 0:
            mats = np.frombuffer(blobs[header["bodies"]], dtype=np.float32).reshape(-1, 4, 4)
            for body, m in zip(header["body_ids"], mats, strict=True):
                if body in self.bodies:
                    self.bodies[body].matrix_world = Matrix(m.tolist())
        for mesh, index in zip(self.deformables, header.get("deformables", []), strict=True):
            mesh.vertices.foreach_set("co", np.frombuffer(blobs[index], dtype=np.float32))
            mesh.update()
        for cloud, index in zip(self.points, header.get("points", []), strict=True):
            cloud.points.foreach_set("co", np.frombuffer(blobs[index], dtype=np.float32))
            cloud.update_tag()
        for mesh, (verts, faces) in zip(self.surfaces, header.get("surfaces", []), strict=True):
            self._fill(mesh, np.frombuffer(blobs[verts], dtype=np.float32), np.frombuffer(blobs[faces], dtype=np.int32))
            mesh.polygons.foreach_set("use_smooth", np.ones(len(mesh.polygons), dtype=bool))
        # every camera's picture, then every camera's depth under one switch to the depth materials: a switch
        # makes Cycles sync the scene's shaders again, so it is made once a frame, not once a camera
        cameras = header["cameras"]
        replies = [{"name": cam["name"], "rgb": -1, "depth": -1} for cam in cameras]
        out: list = []
        for cam, reply in zip(cameras, replies, strict=True):
            if "rgb" in cam.get("channels", ("rgb", "depth")):
                t = time.time()
                w, h = self._aim(cam)
                reply["rgb"], reply["rgb_seconds"] = len(out), None
                out.append(self._srgb(self._still(w, h)).tobytes())
                reply["rgb_seconds"] = round(time.time() - t, 3)
        deep = [
            (cam, reply)
            for cam, reply in zip(cameras, replies, strict=True)
            if "depth" in cam.get("channels", ("rgb", "depth"))
        ]
        if deep:
            with self._depth_mode():
                for cam, reply in deep:
                    t = time.time()
                    w, h = self._aim(cam)
                    reply["depth"] = len(out)
                    out.append(self._depth(w, h, float(cam["far"])).tobytes())
                    reply["depth_seconds"] = round(time.time() - t, 3)
        return {"ok": True, "cameras": replies}, out

    def _aim(self, cam: dict) -> tuple[int, int]:
        """The camera as ``cam`` states it -- size, intrinsics, pose -- for the next render; its ``(w, h)``."""
        from mathutils import Matrix

        scene, data = self.scene, self.camera.data
        w, h = int(cam["width"]), int(cam["height"])
        fx, fy, cx, cy = (float(v) for v in cam["K"])
        render = scene.render
        render.resolution_x, render.resolution_y = w, h
        settings = camera_settings(w, h, fx, fy, cx, cy, data.sensor_width)
        render.pixel_aspect_x, render.pixel_aspect_y = settings["pixel_aspect"]
        data.sensor_fit = "HORIZONTAL"
        data.lens = settings["lens"]
        data.shift_x, data.shift_y = settings["shift"]
        # colour out to DRAW_DISTANCE, as gl and tiled draw it, so a floor runs to the horizon; depth past the
        # camera's own far is a miss (see _depth)
        data.clip_start, data.clip_end = float(cam["near"]), max(DRAW_DISTANCE, float(cam["far"]))
        pose = Matrix([cam["pose"][4 * r : 4 * r + 4] for r in range(4)])
        self.camera.matrix_world = pose @ Matrix(VISION_TO_BLENDER)
        return w, h

    def _still(self, w: int, h: int):
        """Render the scene as it stands and read the Combined pass back, linear, ``float32 [h, w, 3]``."""
        path = os.path.join(self.tmp, "frame.exr")
        self.scene.render.filepath = path
        self.bpy.ops.render.render(write_still=True)
        return self._read_exr(path, w, h)

    def _srgb(self, linear):
        np = self.np
        linear = np.clip(linear, 0.0, 1.0)
        srgb = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1.0 / 2.4) - 0.055)
        return np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

    @contextmanager
    def _depth_mode(self):
        """The scene as the depth renders want it, and back after: every surface emitting its own view depth.

        Blender 5 writes a multilayer EXR with the Combined pass alone, whatever passes are on, so the depth
        is drawn as a colour: each surface emits Camera Data's View Z Depth (metres along the view axis), one
        sample at each pixel's centre, no denoising, under a black world, so a miss reads 0.
        """
        bpy, scene = self.bpy, self.scene
        if getattr(self, "_depth_mat", None) is None:
            mat = bpy.data.materials.new("rss_depth")
            mat.use_nodes = True
            nodes, links = mat.node_tree.nodes, mat.node_tree.links
            nodes.clear()
            camera = nodes.new("ShaderNodeCameraData")
            emit = nodes.new("ShaderNodeEmission")
            out = nodes.new("ShaderNodeOutputMaterial")
            links.new(camera.outputs["View Z Depth"], emit.inputs["Color"])
            emit.inputs["Strength"].default_value = 1.0
            links.new(emit.outputs["Emission"], out.inputs["Surface"])
            self._depth_mat = mat
            self._black = bpy.data.worlds.new("rss_black")
            self._black.use_nodes = True
            self._black.node_tree.nodes["Background"].inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        layer, world = scene.view_layers[0], scene.world
        if scene.render.engine == "CYCLES":
            one_sample = (scene.cycles, (("samples", 1), ("use_adaptive_sampling", False), ("use_denoising", False),
                          ("filter_width", 0.01), ("max_bounces", 0), ("seed", 0)))  # fmt: skip
        else:  # EEVEE: one sample, no pixel filter
            one_sample = (scene.eevee, (("taa_render_samples", 1),))
        owner, settings = one_sample
        kept = [(owner, key, getattr(owner, key)) for key, _ in settings]
        kept.append((scene.render, "filter_size", scene.render.filter_size))
        for key, value in settings:
            setattr(owner, key, value)
        scene.render.filter_size = 0.01
        layer.material_override, scene.world = self._depth_mat, self._black
        try:
            yield
        finally:
            layer.material_override, scene.world = None, world
            for target, key, value in kept:
                setattr(target, key, value)

    def _depth(self, w: int, h: int, far: float):
        """Distance to the image plane per pixel, 0 for a miss or past ``far``: a render under :meth:`_depth_mode`."""
        np = self.np
        z = self._still(w, h)[..., 0]
        return np.where((z > 0.0) & (z <= far) & np.isfinite(z), z, 0.0).astype(np.float32)

    def _read_exr(self, path: str, w: int, h: int):
        np = self.np
        import OpenImageIO as oiio

        image = oiio.ImageInput.open(path)
        if image is None:
            raise RuntimeError(f"could not read the render {path}: {oiio.geterror()}")
        try:
            spec = image.spec()
            names = list(spec.channelnames)
            pixels = np.asarray(image.read_image(0, 0, 0, spec.nchannels, "float"), dtype=np.float32)
        finally:
            image.close()
        pixels = pixels.reshape(h, w, spec.nchannels)
        picked = []
        for c in "RGB":
            hit = [i for i, n in enumerate(names) if n.endswith(f"Combined.{c}") or n == c]
            if not hit:
                raise RuntimeError(f"the render has no Combined.{c} channel: {names}")
            picked.append(pixels[..., hit[0]])
        return np.stack(picked, axis=-1)


def main(argv: list[str]) -> int:
    port = int(argv[0])
    sock = socket.create_connection(("127.0.0.1", port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    send(sock, {"token": os.environ.get(TOKEN_ENV, "")})
    world = None
    while True:
        try:
            header, blobs = receive(sock)
        except ConnectionError:
            return 0
        op = header.get("op")
        try:
            if op == "quit":
                if world is not None:
                    import shutil

                    shutil.rmtree(world.tmp, ignore_errors=True)
                send(sock, {"ok": True})
                return 0
            if op == "scene":
                import bpy

                world = World(header, blobs)
                send(sock, {"ok": True, "version": bpy.app.version_string, "device": world.device})
            elif op == "frame":
                if world is None:
                    raise RuntimeError("a frame before the scene")
                reply, out = world.frame(header, blobs)
                send(sock, reply, out)
            else:
                raise RuntimeError(f"unknown op {op!r}")
        except Exception as exc:  # said to the client, which raises it; the worker stays up for the next request
            import traceback

            send(sock, {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    sys.exit(main(args))
