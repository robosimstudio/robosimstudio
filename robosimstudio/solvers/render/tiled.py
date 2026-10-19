# SPDX-License-Identifier: Apache-2.0
"""The default backend: Newton's own tiled ray tracer.

:class:`newton.sensors.SensorTiledCamera` renders every camera of every world in
one megakernel launch, straight off the model's BVHs. Three things make it the
default here:

* It is **in-process and on device** -- no GL context, no USD stage, no second
  process, and the frame never leaves the GPU unless someone asks for it. On a
  CPU it runs the same kernels on Warp's CPU device, which is what renders a
  scene on a machine without a GPU.
* It renders **cloth natively** from ``state.particle_q``, so a deformable
  scene needs no per-frame mesh rebuild to be photographed.
* It gives **exact intrinsics**. The pinhole ray helper takes a focal length
  and film apertures with offsets, which is a strictly richer parameterisation
  than a field of view: an explicitly calibrated camera round-trips through it
  to the pixel, principal-point shift included. :func:`aperture_from_intrinsics`
  is the conversion.

Segmentation comes out as a per-pixel *shape* index, which is Newton's own
numbering and means nothing to a benchmark. The caller supplies a
``shape -> label id`` lookup (the toolkit layer builds one from its scene
handles) and this maps through it, so a task's segmentation ids are the task's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import newton
import numpy as np
import warp as wp
from newton.utils import ColorSpace

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.core.math import xform_from_matrix
from robosimstudio.core.render.base import FrameSet, RendererBase
from robosimstudio.core.render.spec import put

__all__ = ["RendererTiled", "aperture_from_intrinsics"]

# What SensorTiledCamera writes where a ray hit nothing.
_NO_SHAPE = 0xFFFFFFFF

# The ray helper's units cancel between focal length and aperture, so any
# consistent unit works; 1.0 keeps the offsets numerically small.
_FOCAL_LENGTH = 1.0

# Camera-space basis change. CameraSpec is +x right, +y down, +z forward (the
# vision convention K projects through); the ray helper builds rays as
# (film_x/f, film_y/f, -1), i.e. +y up and -z forward.
_VISION_TO_RAY = np.diag([1.0, -1.0, -1.0])


def aperture_from_intrinsics(spec: CameraSpec) -> dict[str, float]:
    """Film apertures and offsets reproducing a camera's ``K`` exactly.

    Inverting the ray helper's film equation
    ``film_x = ((px + 0.5)/W - 0.5) * h_aperture + h_offset`` against the
    pinhole ``px = fx * X/Z + cx`` gives, with the focal length free:

    ==================  ===================================
    aperture            ``W / fx``     (and ``H / fy``)
    offset              ``(W/2 - 0.5 - cx) / fx``
    vertical offset     ``(cy + 0.5 - H/2) / fy``
    ==================  ===================================

    The half-pixel terms are not decoration: the helper samples pixel centres,
    so dropping them shifts every projection by half a pixel, which is exactly
    the size of error that survives a visual check and fails a calibration one.

    Returns:
        Keyword arguments for
        :meth:`SensorTiledCamera.Utils.compute_camera_rays_pinhole`.
    """
    k = spec.intrinsics
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    w, h = float(spec.width), float(spec.height)
    return {
        "focal_length": _FOCAL_LENGTH,
        "horizontal_aperture": _FOCAL_LENGTH * w / fx,
        "vertical_aperture": _FOCAL_LENGTH * h / fy,
        "horizontal_aperture_offset": _FOCAL_LENGTH * (w * 0.5 - 0.5 - cx) / fx,
        "vertical_aperture_offset": _FOCAL_LENGTH * (cy + 0.5 - h * 0.5) / fy,
    }


def _normalized(v: Sequence[float]) -> tuple[float, float, float]:
    """A direction as a unit vector, leaving a zero vector alone."""
    x, y, z = (float(c) for c in v)
    length = (x * x + y * y + z * z) ** 0.5
    return (x, y, z) if length == 0.0 else (x / length, y / length, z / length)


def _packed_rgba(rgb: Sequence[float]) -> int:
    """An RGB triple in [0, 1] as the little-endian RGBA uint32 ClearData wants.

    R is the *low* byte -- see the tracer's own pack helper. Getting this
    backwards produces a plausible colour rather than an error, which is why it
    is one function with one test rather than three shifts at a call site.
    """
    r, g, b = (int(round(255.0 * min(max(float(c), 0.0), 1.0))) for c in rgb)
    return (0xFF << 24) | (b << 16) | (g << 8) | r


class RendererTiled(RendererBase):
    """Native camera channels in one launch; refined RGB uses a separate HDR pass.

    All cameras share one output buffer set, so they must share a resolution;
    a scene wanting two sizes builds two renderers, which is one extra launch
    and no extra state.

    Args:
        model: The finalized model.
        cameras: The cameras to render.
        channels: Restrict every camera to these channels.
        world: Which world's slice of the output :meth:`render` returns.
            :meth:`render_worlds` returns all of them and ignores this.
        shape_labels: ``shape index -> label id`` for the ``seg`` channel.
            Shapes absent from the mapping, and pixels that hit nothing, come
            back as 0. Without it ``seg`` is Newton's raw shape indices.
        refit_shapes: Refit the shape BVH before each render. Needed whenever
            rigid bodies move, which is every scene that is not a still life.
        refit_particles: Refit the particle BVH before each render. Needed
            whenever a deformable moves.
        light_direction: Direction the default directional light travels.
        shadows: Cast shadows from the default light.
        enable_textures: Whether to sample textures. Off is faster and gives
            flat shading, which is enough for a preview or a depth-only pass.
        render_config: A fully-specified
            :class:`newton.sensors.SensorTiledCamera.RenderConfig`, overriding
            the convenience arguments above.
        sample_scale: RGB subpixel grid width and height. Depth and segmentation stay native.
        exposure: Linear radiance multiplier applied before RGB encoding.
    """

    name = "tiled"
    supported_channels = ("rgb", "depth", "seg")
    # Warp kernels, so a CPU device too: 55 ms a 480x270 frame there, 210 ms at 960x540.
    devices = ("cuda", "cpu")

    wants: tuple[str, ...] = ("shape_labels",)

    @classmethod
    def look_kwargs(cls, look: Any) -> dict[str, Any]:
        """What this backend takes from a look; see :meth:`RendererBase.look_kwargs`."""
        out: dict[str, Any] = {}
        put(out, "light_direction", look.light_direction)
        put(out, "shadows", look.shadows)
        put(out, "enable_textures", look.enable_textures)
        put(out, "fill_lights", look.fill_lights)
        put(out, "background", look.background)
        put(out, "sample_scale", look.sample_scale)
        put(out, "exposure", look.exposure)
        return out

    def __init__(
        self,
        model: newton.Model,
        cameras: Sequence[CameraSpec],
        *,
        channels: Sequence[str] | None = None,
        world: int = 0,
        shape_labels: Mapping[int, int] | None = None,
        refit_shapes: bool = True,
        refit_particles: bool | None = None,
        light_direction: Sequence[float] = (-1.0, 1.0, -1.0),
        fill_lights: Sequence[tuple[Sequence[float], float]] = (),
        shadows: bool = True,
        enable_textures: bool = True,
        background: Sequence[float] | None = None,
        sample_scale: int = 1,
        exposure: float = 1.0,
        render_config=None,
    ) -> None:
        super().__init__(cameras, channels=channels)
        if not isinstance(sample_scale, int) or isinstance(sample_scale, bool) or sample_scale < 1:
            raise ValueError("sample_scale must be a positive integer")
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("exposure must be positive and finite")
        self.sample_scale = sample_scale
        self.exposure = float(exposure)
        self._refined_rgb = sample_scale != 1 or exposure != 1.0

        from newton.sensors import SensorTiledCamera

        self.model = model
        self.world = int(world)
        self.num_worlds = max(1, int(getattr(model, "world_count", 1) or 1))
        """Worlds the sensor renders in one launch.

        Read off the model rather than passed in: the sensor's buffers are
        already ``[world_count, cameras, H, W]`` and always have been, so this
        is a statement of what is being rendered rather than a request.
        """
        self.bodies_per_world = (int(model.body_count) // self.num_worlds) if model.body_count else 0
        """The stride a body-parented camera's parent index is offset by."""
        # A tile is one resolution, so a rig that mixes them is rendered as one
        # pass per resolution over the same sensor. Real rigs do mix: this suite's
        # perception camera is Isaac's 800x512 and the three DROID views it now
        # carries beside it are 16:9, and demanding one number would mean either
        # re-aiming the perception camera or cropping the views this exists to
        # match. Intrinsics still vary freely *within* a pass.
        self._groups: list[tuple[int, int, list[int]]] = []
        for index, cam in enumerate(self.cameras):
            size = (int(cam.width), int(cam.height))
            for w, h, members in self._groups:
                if (w, h) == size:
                    members.append(index)
                    break
            else:
                self._groups.append((size[0], size[1], [index]))
        self.width, self.height = self._groups[0][0], self._groups[0][1]
        self.refit_shapes = bool(refit_shapes)
        self.refit_particles = model.particle_count > 0 if refit_particles is None else bool(refit_particles)

        if render_config is None:
            render_config = SensorTiledCamera.RenderConfig(
                enable_textures=bool(enable_textures), enable_shadows=bool(shadows)
            )
        self.sensor = SensorTiledCamera(model, default_render_config=render_config)
        # Shadows are set on the config rather than left to create_default_light's
        # side effect, which Newton deprecated in 1.4.
        self.sensor.utils.create_default_light(
            enable_shadows=bool(shadows), direction=wp.vec3f(*[float(v) for v in light_direction])
        )
        if fill_lights:
            self._add_fill_lights(light_direction, fill_lights, shadows=bool(shadows))
        if background is not None:
            # Authored in *output* space, not albedo space: the kernel converts
            # clear_color to linear only when output_color_space is LINEAR, and
            # this package renders SRGB, so what is packed here is what the miss
            # pixel becomes. Set before the first render -- clear_color is a
            # wp.static in the megakernel, so changing it later respecialises it.
            self.sensor.default_clear_data.clear_color = _packed_rgba(background)

        utils = self.sensor.utils
        self._passes: list[dict] = []
        for width, height, members in self._groups:
            n = len(members)
            # One ray buffer per pass; each camera fills its own slice, which is
            # what lets a rig mix intrinsics inside a single launch.
            rays = wp.zeros((n, height, width, 2), dtype=wp.vec3f, device=model.device)
            for slot, index in enumerate(members):
                utils.compute_camera_rays_pinhole(
                    width,
                    height,
                    out_rays=rays,
                    camera_index=slot,
                    **aperture_from_intrinsics(self.cameras[index]),
                )
            wanted = set().union(*(self.channels[self.cameras[i].name] for i in members))
            self._passes.append(
                {
                    "width": width,
                    "height": height,
                    "members": members,
                    "rays": rays,
                    "transforms": wp.zeros((n, model.world_count), dtype=wp.transformf, device=model.device),
                    "host": np.zeros((n, model.world_count, 7), dtype=np.float32),
                    "color": utils.create_color_image_output(width, height, n)
                    if "rgb" in wanted and not self._refined_rgb
                    else None,
                    "depth": utils.create_forward_depth_image_output(width, height, n) if "depth" in wanted else None,
                    "shape": utils.create_shape_index_image_output(width, height, n) if "seg" in wanted else None,
                }
            )
            pass_ = self._passes[-1]
            pass_["detail"] = None
            if "rgb" in wanted and self._refined_rgb:
                render_width, render_height = width * sample_scale, height * sample_scale
                detail_rays = wp.zeros((n, render_height, render_width, 2), dtype=wp.vec3f, device=model.device)
                for slot, index in enumerate(members):
                    sample_camera = replace(
                        self.cameras[index].scaled(render_width, render_height),
                        cx=(self.cameras[index].intrinsics[0, 2] + 0.5) * sample_scale - 0.5,
                        cy=(self.cameras[index].intrinsics[1, 2] + 0.5) * sample_scale - 0.5,
                    )
                    utils.compute_camera_rays_pinhole(
                        render_width,
                        render_height,
                        out_rays=detail_rays,
                        camera_index=slot,
                        **aperture_from_intrinsics(sample_camera),
                    )
                pass_["detail"] = {
                    "rays": detail_rays,
                    "hdr": utils.create_hdr_color_image_output(render_width, render_height, n),
                    "shape": utils.create_shape_index_image_output(render_width, render_height, n),
                }

        self._label_lut: np.ndarray | None = None
        self._deformable_label: int = 0
        if shape_labels is not None:
            lut = np.zeros(model.shape_count + 1, dtype=np.int32)
            for shape, label in shape_labels.items():
                s_idx = int(shape)
                if 0 <= s_idx < model.shape_count:
                    lut[s_idx] = int(label)
                elif s_idx in (0xFFFFFFFD, 0xFFFFFFFE):
                    self._deformable_label = int(label)
            self._label_lut = lut

    def _detail_colors(self, pass_: dict, state: newton.State) -> np.ndarray | None:
        detail = pass_["detail"]
        if detail is None:
            return None
        self.sensor.update(
            state,
            camera_transforms=pass_["transforms"],
            camera_rays=detail["rays"],
            hdr_color_image=detail["hdr"],
            shape_index_image=detail["shape"],
        )
        linear = detail["hdr"].numpy()
        shape = detail["shape"].numpy()
        clear = int(self.sensor.default_clear_data.clear_color)
        background = np.array([clear & 255, (clear >> 8) & 255, (clear >> 16) & 255], dtype=np.float32) / 255
        background = np.where(background <= 0.04045, background / 12.92, ((background + 0.055) / 1.055) ** 2.4)
        linear[shape == _NO_SHAPE] = background
        if self.sample_scale > 1:
            linear = linear.reshape(
                self.num_worlds,
                len(pass_["members"]),
                pass_["height"],
                self.sample_scale,
                pass_["width"],
                self.sample_scale,
                3,
            ).mean(axis=(3, 5))
        linear *= self.exposure
        if self.sensor.default_render_config.output_color_space == ColorSpace.SRGB:
            linear = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - 0.055)
        return np.clip(np.rint(linear * 255), 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ render
    def _write_transforms(self, pass_: dict, body_q: np.ndarray | None) -> None:
        """One transform per camera per world.

        The buffer is ``[cameras, worlds]`` and used to be filled by
        broadcasting one pose across the world axis. For a camera bolted to the
        room that is right -- a wave's worlds are superimposed at the origin, so
        they all see the same place. For a camera bolted to a *wrist* it is not:
        the parent index is the template's, so every world was photographed
        through world 0's hand. Silent, and worst exactly where it matters --
        an eye-in-hand view is the one a grasp is judged from.
        """
        for slot, index in enumerate(pass_["members"]):
            cam = self.cameras[index]
            for world in range(self.num_worlds):
                # The ray helper's camera space is -z forward, +y up.
                pose = cam.pose_on_body(body_q, body_offset=world * self.bodies_per_world).copy()
                pose[:3, :3] = pose[:3, :3] @ _VISION_TO_RAY
                pass_["host"][slot, world] = xform_from_matrix(pose).astype(np.float32)
        pass_["transforms"].assign(pass_["host"])

    def render(self, state: newton.State) -> FrameSet:
        """One world's frames -- :attr:`world`'s. See :meth:`render_worlds`."""
        frames = self.render_worlds(state)
        return frames[self.world] if self.world < len(frames) else {}

    def render_worlds(self, state: newton.State, worlds: Sequence[int] | None = None) -> list[FrameSet]:
        """Every world's frames, including optional refined RGB. ``[num_worlds]`` of FrameSet.

        Args:
            state: What to photograph.
            worlds: Which worlds to read back, or ``None`` for all. The *launch*
                covers every world regardless -- one megakernel is what makes
                this cheap -- so this only narrows the host copy, and a world
                left out comes back as an empty frame set rather than as a
                missing entry.

        The sensor has always rendered every camera of every world in one
        megakernel -- its buffers are ``[worlds, cameras, H, W]`` -- and this
        class has always thrown away all but one world on the way out. That was
        free when nothing else wanted them; a wave wants every one of them, and
        wants them from the same launch rather than from ``W`` launches that
        would each redo the whole trace.

        So the launch is unchanged and only the read-back is wider. A wave's
        rendering therefore costs what a single world's already cost, plus the
        copy of the frames somebody actually asked for.
        """
        needs_body_q = any(c.parent_body >= 0 for c in self.cameras)
        body_q = state.body_q.numpy() if needs_body_q else None

        if self.refit_shapes:
            self.model.bvh_refit_shapes(state)
        if self.refit_particles:
            self.model.bvh_refit_particles(state)

        # Not ``wanted``: that name is taken, three lines into the inner loop,
        # by the *channels* a camera asked for. Sharing it made the second
        # resolution pass sort channel names and index the colour buffer with
        # them -- an ``IndexError`` a pass later than the mistake, and only on a
        # rig that mixes resolutions.
        chosen = sorted(set(range(self.num_worlds) if worlds is None else (int(w) for w in worlds)))
        out: list[FrameSet] = [{} for _ in range(self.num_worlds)]
        for pass_ in self._passes:
            self._write_transforms(pass_, body_q)
            if any(pass_[key] is not None for key in ("color", "depth", "shape")):
                self.sensor.update(
                    state,
                    camera_transforms=pass_["transforms"],
                    camera_rays=pass_["rays"],
                    color_image=pass_["color"],
                    forward_depth_image=pass_["depth"],
                    shape_index_image=pass_["shape"],
                )
            detail_colors = self._detail_colors(pass_, state)
            width, height = pass_["width"], pass_["height"]
            # One read-back per channel for the whole fleet, then sliced on the
            # host: the copy is the cost, and doing it per world would pay it
            # once per world for the same bytes.
            colors = pass_["color"].numpy() if pass_["color"] is not None else None
            depths = pass_["depth"].numpy() if pass_["depth"] is not None else None
            shapes = pass_["shape"].numpy() if pass_["shape"] is not None else None

            for world in chosen:
                rgb = (
                    detail_colors[world]
                    if detail_colors is not None
                    else (colors[world] if colors is not None else None)
                )
                depth = depths[world] if depths is not None else None
                shape = shapes[world] if shapes is not None else None
                for slot, index in enumerate(pass_["members"]):
                    cam = self.cameras[index]
                    frame: dict[str, np.ndarray] = {}
                    wanted = self.channels[cam.name]
                    if "rgb" in wanted:
                        if detail_colors is not None:
                            frame["rgb"] = np.ascontiguousarray(rgb[slot])
                        else:
                            frame["rgb"] = np.ascontiguousarray(
                                rgb[slot].view(np.uint8).reshape(height, width, 4)[..., :3]
                            )
                    if "depth" in wanted:
                        # A miss is already cleared to 0; nan_to_num only guards against
                        # a degenerate ray, which a zero-aperture camera can produce.
                        frame["depth"] = np.nan_to_num(
                            np.asarray(depth[slot], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
                        )
                    if "seg" in wanted:
                        idx = np.asarray(shape[slot], dtype=np.int64)
                        # Misses arrive as the clear value, not a shape; both land on 0.
                        miss = idx == _NO_SHAPE
                        deform_hit = (idx == 0xFFFFFFFD) | (idx == 0xFFFFFFFE)
                        if self._label_lut is None:
                            labels = np.where(miss | deform_hit, 0, idx).astype(np.int32)
                        else:
                            safe_idx = np.where((idx >= 0) & (idx < len(self._label_lut)), idx, 0)
                            labels = self._label_lut[safe_idx]
                            if self._deformable_label:
                                labels = np.where(deform_hit, self._deformable_label, labels)
                        frame["seg"] = np.where(miss, 0, labels).astype(np.int32)
                    out[world][cam.name] = frame
        return out

    def _add_fill_lights(
        self,
        key: Sequence[float],
        fills: Sequence[tuple[Sequence[float], float]],
        *,
        shadows: bool,
    ) -> None:
        """Replace the single default light with a key and one or more fills.

        ``create_default_light`` writes exactly one light and there is no public
        API for a second, but the render context holds the lights as arrays and
        reads its light count off ``lights_active.shape[0]`` -- so N of them work
        with no other change and no recompile. Reaching them means a name-mangled
        private attribute, which is acceptable only because this package pins
        newton to an exact revision; it is the same bargain :mod:`.rtx` makes for
        ``_render_products``, and it is written to fail loudly rather than
        quietly render the old rig if the attribute ever moves.

        **The stated intensities are dropped here.** The megakernel adds each
        light as ``albedo * n·l`` with no per-light scale, so on this backend a
        fill is a *direction* rather than a level and its ratio to the key comes
        from the angle it strikes at. They are honoured by pyrender and pybullet,
        which is why the spec carries them.

        What *is* honoured, and what makes the second light worth having, is
        ``lights_cast_shadow``, which is per-light: the key casts and the fill
        does not, so the fill reaches exactly the faces the key's shadow left
        sitting at the ambient term. Measured on a vertical face on the shadow
        side, 58 before and 117 after.
        """
        context = getattr(self.sensor.utils, "_Utils__render_context", None)
        if context is None:  # pragma: no cover - a newton whose internals moved
            raise AttributeError(
                "the tiled sensor no longer exposes _Utils__render_context; "
                "multi-light rigs need updating for this newton release"
            )
        directions = [_normalized(key)] + [_normalized(direction) for direction, _ in fills]
        casts = [bool(shadows)] + [False] * len(fills)
        device = context.device
        context.lights_active = wp.array([True] * len(directions), dtype=wp.bool, device=device)
        context.lights_type = wp.array([1] * len(directions), dtype=wp.int32, device=device)
        context.lights_cast_shadow = wp.array(casts, dtype=wp.bool, device=device)
        context.lights_position = wp.array([wp.vec3f(0.0)] * len(directions), dtype=wp.vec3f, device=device)
        context.lights_orientation = wp.array([wp.vec3f(*d) for d in directions], dtype=wp.vec3f, device=device)
