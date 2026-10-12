# SPDX-License-Identifier: Apache-2.0
"""The OVRTX path-traced backend: a camera rig, metric depth, finishes, scenery.

Newton's :class:`newton.viewer.ViewerRTX` is a *viewport*. It authors one USD
camera, one render product and one render var, ``/Render/Vars/LdrColor``, and
presents the previous render while the next one is in flight. Driving a
multi-camera rig through that means re-aiming the one camera and stepping the
tracer once per camera, reading back a frame that is one render behind, through
a camera whose only intrinsic is a vertical field of view and whose pose is
squeezed through a pitch and a yaw.

Two of those are survivable in an export, which is why this backend was
documented as belonging on one: a replay can throw the first frame away, and it
can afford to pay per camera. The third was never survivable anywhere. It was
only quieter in an export, because a re-render nobody is comparing against
anything looks like a picture rather than like a wrong one.

**The rig is built into the stage.** Before OVRTX opens the exported USD, this
defines one ``UsdGeom.Camera`` and one ``RenderProduct`` per
:class:`~robosimstudio.core.cameras.CameraSpec`. Each product is a *copy* of the
one Newton authored -- so the twenty-odd ``omni:rtx:`` settings it puts there are
inherited rather than restated, and stay inherited when Newton changes them --
with its own resolution, its own near plane and its own render vars, pointed at
its own camera. ``ovrtx.Renderer.step`` takes a **set** of products, so the whole rig is
one tracer step rather than one per camera, and it is stepped **synchronously**,
so the frames :meth:`RendererRTX.render` returns are of the state it was handed.

Four things fall out:

* **A rig may mix resolutions.** A tool-sorting rig is two 640x480 views and a
  320x240 wrist camera; the shared viewport could not render it at all.
* **The first frame is the first frame.** Under the async read-back the frame
  after construction was a warm-up buffer, and at one tracer pass per camera --
  what a live view uses -- *every* camera came back holding the previous
  camera's picture. Measured on a tool-sorting scene, the three cameras' mean
  brightness came back rotated by one: 106 / 75 / 80 where the right answer is
  75 / 80 / 106. That failure is silent, and a policy would have eaten it.
* **It costs less**, because the passes no longer multiply by the cameras: a rig
  is one step, not one step per camera. Three cameras at the six passes an export
  converges over, the two paths measured alternately in one session so machine
  drift cancels: 0.119 s against 0.059 s. At *one* pass they cost the same, which
  is the comparison to distrust -- one of the two answers was wrong.
* **The camera is the camera.** A per-camera USD film back carries ``fx``,
  ``fy`` *and* the principal point through its aperture offsets, and a per-camera
  transform carries roll. Both were lost before. Measured against the tiled
  tracer -- an independently written camera model, so agreement is a real check
  rather than a self-consistent one -- mean depth error over a tool-sorting
  rig's three cameras:

  =============  ==================  ==================
  camera         camera at a time    product per camera
  =============  ==================  ==================
  agentview      13.19 mm            0.08 mm
  overhead       228.29 mm           0.01 mm
  eye_in_hand    539.06 mm           0.01 mm
  =============  ==================  ==================

  The overhead camera is the loud one and the reason is legible: it is authored
  rolled 90 degrees so the two long bins lie across the frame, and a pitch and a
  yaw cannot hold a roll.

**And four things a viewport does not draw at all**, which this module adds to
the stage before OVRTX opens it:

* **Finishes.** ``ViewerUSD`` writes each shape's colour as a ``displayColor``
  primvar and binds no material, so OVRTX shades everything as the same plastic.
  :func:`decorate_stage` binds a ``UsdPreviewSurface`` per distinct
  :class:`~robosimstudio.core.render.finish.Finish` to every instance prim --
  roughness, metallicity, opacity, a texture -- with the colour still read
  *from the primvar*, so ``model.shape_color`` keeps meaning what it means
  everywhere else.
* **Live colour.** A decal field that fades its own shapes writes
  ``model.shape_color`` from a kernel; a viewer that bakes colour at load never
  sees it. The shapes the scene declares live (``handles.live_shapes``) are
  re-read every frame and written back through the attribute API, which is the
  same channel the viewer already uses for deforming mesh points.
* **One mesh per deformable.** Newton logs every cloth triangle in the model as
  one ``/model/triangles`` prim, so a film and a sponge in one scene would share
  a material. Each :class:`~robosimstudio.scene.handles.DeformableHandle` is
  logged as its own mesh, with smooth normals and planar UVs.
* **A lighting preset and a backdrop.** ``"lab"`` is a neutral dome (optionally
  an HDR image) under one large panel light, and a
  :func:`~robosimstudio.core.render.scenery.register_scenery` builder may add
  camera-only geometry through :class:`_StageKit` -- which is how
  :mod:`robosimstudio.core.render.room` puts a floor and walls behind every
  scene that never asked for one.

All of that edits the stage inside ``_init_ovrtx``, before the export OVRTX
opens, and the rig is authored in the same pass: no change to Newton, and no
Isaac Sim.

This module is where the private-attribute reach is *isolated*. It overrides
``ViewerRTX._add_camera_lights_and_render_product``, ``_render_and_display``,
``_apply_ground_material`` and ``_log_triangles``, and touches ``viewer._rtx``,
``viewer._render_product_path``, ``viewer._shape_to_batch``, ``viewer._meshes``
and ``viewer.stage``. Nothing else in this package does, which is what the
version pins are protecting: ``ovrtx`` to a release in the ``[rtx]`` extra,
Newton to a rev in ``[tool.uv.sources]``. The offline re-render path exists
precisely so a run can collect cheaply through the tiled backend and pay for RTX
once, afterwards.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import newton
import numpy as np
import warp as wp

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.core.render.base import FrameSet, RendererBase
from robosimstudio.core.render.finish import Finish, finish_for, texture_path
from robosimstudio.core.render.palette import srgb_to_linear
from robosimstudio.core.render.scenery import box_mesh, get_scenery
from robosimstudio.core.render.spec import put
from robosimstudio.core.render.uv import box_projected_uvs, planar_uvs
from robosimstudio.core.render.visual import TUBE_SIDES, tube_points, tube_triangles

__all__ = ["ENVIRONMENTS", "QUALITY_PASSES", "RendererRTX", "decorate_stage", "usd_film_back"]

_V_APERTURE_MM = 20.955
"""USD's default vertical film aperture.

Held fixed and the focal length solved for, rather than the other way round: the
units cancel between the two in every equation below, so only their ratio is
observable, and pinning the one USD tools expect keeps the focal lengths this
authors readable as the millimetres they look like.
"""

_FAR_CLIP_M = 1000.0
"""Far plane, deliberately not the camera's own.

A :class:`~robosimstudio.core.cameras.CameraSpec` states a ``far``, and this is
the one field of it this backend declines to honour. Two of the other four
backends clip there and two do not, so there is no house convention to follow --
and the two that do not are the tiled tracer, which every run in this package is
*collected* through, and the shared viewport this replaced, which every existing
path-traced re-render came out of.

Clipping would therefore have changed the picture in the one direction that is
never worth it: measured on a tool-sorting agentview, whose ``far`` is 5 m, it
puts a flat horizon across the top 28% of a frame that has always shown floor
receding into the backdrop, and it does it in a change that is otherwise about
making this backend *agree* with the tiled one. The near plane is the camera's
own -- it can only reveal geometry, never remove it, and a wrist camera authored
at 5 mm means it.
"""

_DEPTH_VAR = "DistanceToImagePlaneSD"
"""The AOV that is metres, perpendicular to the image plane.

OVRTX produces depth only under its ``*SD`` source names, and this is the one
that means what Isaac Lab's ``distance_to_image_plane`` means. Declaring
``DistanceToImagePlane`` without the suffix yields a render var that exists and
maps to an empty buffer, which is the silent version of this failure.
"""

ENVIRONMENTS = ("default", "studio", "none", "lab")
"""Lighting presets. The first three are Newton's; ``lab`` is this module's."""

QUALITY_PASSES: dict[str, int] = {"preview": 6, "final": 12}
"""Tracer passes per output frame, by quality. See :class:`RendererRTX`."""


def usd_film_back(spec: CameraSpec) -> dict[str, float]:
    """A USD camera's film back reproducing one camera's ``K`` exactly.

    An aperture offset slides the *film*, and a principal point slides the
    *image*, so the two run opposite ways -- horizontally. Vertically they do
    not, because USD's film is ``+y`` up while an image's rows run down and the
    two flips cancel. That asymmetry is the whole content of this function, it
    is invisible in any centred camera, and it is why the numbers here are the
    ones :func:`~robosimstudio.solvers.render.tiled.aperture_from_intrinsics`
    arrived at independently for Newton's ray helper:

    ==================  ==============================================
    focal length        ``v_aperture * fy / H``
    h_aperture          ``W * focal / fx``
    h_offset            ``(W/2 - 0.5 - cx) * h_aperture / W``
    v_offset            ``(cy + 0.5 - H/2) * v_aperture / H``
    ==================  ==============================================

    The half-pixel terms are not decoration either: USD's projection lands on a
    continuous image coordinate, on which pixel *i*'s centre sits at ``i + 0.5``,
    while a ``K`` is written for the index. Measured against the tiled tracer on
    the same camera, with them the two backends' depth aligns to 0.00 px and
    without them to 1.00 -- both halves, one per backend.

    Returns:
        Attribute values for a ``UsdGeom.Camera``, in millimetres of film.
    """
    k = spec.intrinsics
    fx, fy, cx, cy = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
    w, h = float(spec.width), float(spec.height)
    focal = _V_APERTURE_MM * fy / h
    h_aperture = w * focal / fx
    return {
        "focal_length": focal,
        "horizontal_aperture": h_aperture,
        "vertical_aperture": _V_APERTURE_MM,
        "horizontal_aperture_offset": (w * 0.5 - 0.5 - cx) * h_aperture / w,
        "vertical_aperture_offset": (cy + 0.5 - h * 0.5) * _V_APERTURE_MM / h,
    }


def _usd_camera_matrix(world: np.ndarray) -> np.ndarray:
    """A camera-to-world pose as the row-vector 4x4 a USD camera prim wants.

    Two conversions at once, and both are one-liners that are wrong in ways that
    look plausible. USD cameras are ``+x`` right, ``+y`` **up**, ``-z`` forward
    where this package's cameras are ``+x`` right, ``+y`` **down**, ``+z``
    forward, so two axes flip; and USD composes row vectors, so the basis lands
    in the rows and the translation in the last *row*.
    """
    mat = np.eye(4, dtype=np.float64)
    mat[0, :3] = world[:3, 0]
    mat[1, :3] = -world[:3, 1]
    mat[2, :3] = -world[:3, 2]
    mat[3, :3] = world[:3, 3]
    return mat


# ----------------------------------------------------------------- viewer
def _viewer_class():
    """``ViewerRTX`` with a camera rig, depth, per-deformable meshes and a
    stage-decoration hook.

    A factory rather than a module-level class so importing this module does not
    import ``newton.viewer``, which pulls in pyglet.
    """
    import newton.viewer as nv
    import warp as wp

    @wp.kernel(enable_backward=False)
    def _gather_vec3(src: wp.array(dtype=wp.vec3), idx: wp.array(dtype=wp.int32), dst: wp.array(dtype=wp.vec3)):
        i = wp.tid()
        dst[i] = src[idx[i]]

    class _StudioViewerRTX(nv.ViewerRTX):
        """One render product per camera, stepped as a set, on a dressed stage.

        Attributes:
            rig: The cameras, set before the first ``end_frame`` -- the stage is
                authored inside ``_init_ovrtx``, which that call triggers.
            rig_vars: Camera name -> the render var names that camera wants.
            poses: Camera name -> camera-to-world 4x4, replaced per render.
            outputs: What the last step produced, keyed by product path.
        """

        rig: tuple[CameraSpec, ...] = ()
        # Annotated rather than defaulted: these are per-instance, and a mutable
        # class-level default is a dict every viewer would share.
        rig_vars: dict[str, tuple[str, ...]]
        poses: dict[str, np.ndarray]
        outputs = None

        def camera_path(self, index: int) -> str:
            return f"/World/RigCamera_{index}"

        def product_path(self, index: int) -> str:
            # Under Newton's own HydraTextures scope, which is the location Kit
            # convention puts a render product at and OVRTX resolves against.
            return f"/Render/OmniverseKit/HydraTextures/RigProduct_{index}"

        def __init__(
            self,
            *,
            decorate: Callable[[Any], None] | None = None,
            deformables: Mapping[str, Any] | None = None,
            **kwargs,
        ):
            self._decorate = decorate
            self._deformables = dict(deformables or {})
            self._deformable_cache: dict[str, tuple] | None = None
            self.live_paths: dict[int, str] = {}
            super().__init__(**kwargs)

        # -- stage ------------------------------------------------------------
        def _add_camera_lights_and_render_product(self):
            """Author the rig beside the viewport Newton just authored."""
            super()._add_camera_lights_and_render_product()
            from pxr import Gf, Sdf, UsdGeom

            for var in {name for names in self.rig_vars.values() for name in names}:
                prim = self.stage.DefinePrim(f"/Render/Vars/{var}", "RenderVar")
                prim.CreateAttribute("sourceName", Sdf.ValueTypeNames.String, custom=False).Set(var)

            layer = self.stage.GetRootLayer()
            for index, spec in enumerate(self.rig):
                path = self.camera_path(index)
                camera = UsdGeom.Camera.Define(self.stage, path)
                film = usd_film_back(spec)
                camera.GetFocalLengthAttr().Set(film["focal_length"])
                camera.GetHorizontalApertureAttr().Set(film["horizontal_aperture"])
                camera.GetVerticalApertureAttr().Set(film["vertical_aperture"])
                camera.GetHorizontalApertureOffsetAttr().Set(film["horizontal_aperture_offset"])
                camera.GetVerticalApertureOffsetAttr().Set(film["vertical_aperture_offset"])
                camera.GetClippingRangeAttr().Set(Gf.Vec2f(float(spec.near), max(float(spec.far), _FAR_CLIP_M)))
                xform = UsdGeom.Xform(camera.GetPrim())
                xform.ClearXformOpOrder()
                xform.AddTransformOp().Set(Gf.Matrix4d(*np.eye(4).flatten().tolist()))

                # Copied rather than defined: a RenderProduct carries the RTX
                # settings Newton chose -- real-time path tracing, denoising off,
                # one sample per launch -- and a product authored from scratch
                # here would silently be a different renderer, and would stay one
                # after Newton changed its mind.
                product = self.product_path(index)
                Sdf.CopySpec(layer, Sdf.Path(self._render_product_path), layer, Sdf.Path(product))
                prim = self.stage.GetPrimAtPath(product)
                prim.GetRelationship("camera").SetTargets([Sdf.Path(path)])
                prim.GetAttribute("resolution").Set(Gf.Vec2i(int(spec.width), int(spec.height)))
                prim.GetRelationship("orderedVars").SetTargets(
                    [Sdf.Path(f"/Render/Vars/{var}") for var in self.rig_vars[spec.name]]
                )
                # The copied product carries Newton's tone-map API; only the
                # operator changes. See TONEMAP_OPERATOR for what ACES did.
                prim.CreateAttribute("omni:rtx:post:tonemap:operator", Sdf.ValueTypeNames.Token).Set(TONEMAP_OPERATOR)

        def _apply_ground_material(self):
            # Newton binds a dark, shiny material to anything named "plane";
            # the ground has a finish like every other shape now.
            if self._decorate is None:
                super()._apply_ground_material()

        def _init_ovrtx(self):
            if self._decorate is not None:
                self._decorate(self)

            # THE PATH TRACER GOES ON THE SAME GPU AS THE SIMULATION.
            #
            # ``ovrtx.RendererConfig`` carries ``active_cuda_gpus`` and Newton
            # never sets it, so OVRTX picks its own device -- in practice GPU 0.
            # On a one-GPU box that is always the sim's GPU and the omission is
            # invisible. On two, a worker whose warp device is cuda:1 renders on
            # GPU 0 and then maps the render var back into a cuda:1 array, and
            # the interop fails on the first frame:
            #
            #   RuntimeError: Operation 'unmap_attribute(4294967296)' failed:
            #     op 4: ovrtx_unmap_attribute: CUDA stream synchronization failed
            #
            # Measured: with ``--devices cuda:0,cuda:1``, every odd worker (the
            # ones round-robin put on cuda:1) died there during startup while the
            # even ones ran, so half the pool was lost and the second GPU sat at
            # 0% holding nothing but contexts.
            #
            # Set through the module attribute because the config object is
            # built inside Newton's own ``_init_ovrtx``; the wrapper only fills a
            # value the caller left as ``None``, so an explicit choice upstream
            # still wins. CUDA_VISIBLE_DEVICES is NOT an alternative here -- OVRTX
            # enumerates graphics devices separately from CUDA and skips every
            # GPU when it is set.
            import ovrtx

            ordinal = self.device.ordinal if getattr(self.device, "is_cuda", False) else None
            original = ovrtx.Renderer
            if ordinal is None:
                return super()._init_ovrtx()

            def _on_sim_gpu(*args, config=None, **kwargs):
                if config is not None and getattr(config, "active_cuda_gpus", None) is None:
                    config.active_cuda_gpus = str(int(ordinal))
                return original(*args, config=config, **kwargs)

            ovrtx.Renderer = _on_sim_gpu
            try:
                super()._init_ovrtx()
            finally:
                ovrtx.Renderer = original

        # -- deformables ------------------------------------------------------
        def _log_triangles(self, state):
            if not self._deformables:
                return super()._log_triangles(state)
            if self._deformable_cache is None:
                self._deformable_cache = self._build_deformable_cache(state)
            hidden = not self.show_triangles or self._layer_force_hidden()
            for name, (idx, tris, tris_np, out, uvs) in self._deformable_cache.items():
                tube = self._tubes.get(name)
                if tube is not None:
                    centre = state.particle_q.numpy()[np.asarray(tube["particles"])]
                    pts_np, normals_np = tube_points(centre, tube["radius"], tube["sides"])
                    out.assign(pts_np.astype(np.float32))
                    normals = wp.array(normals_np.astype(np.float32), dtype=wp.vec3, device=self.device)
                    self.log_mesh(
                        self._qualify(f"/model/deformables/{name}"),
                        out,
                        tris,
                        normals,
                        uvs,
                        hidden=hidden,
                        backface_culling=False,
                    )
                    continue
                wp.launch(_gather_vec3, dim=len(idx), inputs=[state.particle_q, idx], outputs=[out], device=self.device)
                pts = out.numpy()
                normals = wp.array(_vertex_normals(pts, tris_np), dtype=wp.vec3, device=self.device)
                self.log_mesh(
                    self._qualify(f"/model/deformables/{name}"),
                    out,
                    tris,
                    normals,
                    uvs,
                    hidden=hidden,
                    backface_culling=False,
                )

        def _build_deformable_cache(self, state) -> dict[str, tuple]:
            cache: dict[str, tuple] = {}
            self._tubes: dict[str, dict] = {}
            rest = state.particle_q.numpy()
            for name, handle in self._deformables.items():
                particles = np.asarray(handle.particles, dtype=np.int32)
                tris_np = np.asarray(handle.triangles, dtype=np.int32).reshape(-1, 3)
                radius = float(getattr(handle, "tube_radius", 0.0) or 0.0)
                if len(particles) >= 2 and len(tris_np) == 0 and radius > 0.0:
                    # A centerline: a tube swept along it, rebuilt every frame.
                    sides = TUBE_SIDES
                    tris_np = tube_triangles(len(particles), sides)
                    count = len(particles) * sides + 2
                    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(rest[particles], axis=0), axis=1))])
                    uv = np.zeros((count, 2), dtype=np.float32)
                    for k in range(sides):
                        uv[k:-2:sides, 0] = k / sides
                        uv[k:-2:sides, 1] = arc / max(2.0 * np.pi * radius, 1.0e-6)
                    self._tubes[name] = {"particles": particles, "radius": radius, "sides": sides}
                    cache[name] = (
                        None,
                        wp.array(tris_np.reshape(-1), dtype=wp.int32, device=self.device),
                        tris_np,
                        wp.empty(count, dtype=wp.vec3, device=self.device),
                        wp.array(uv, dtype=wp.vec2, device=self.device),
                    )
                    continue
                if len(particles) == 0 or len(tris_np) == 0:
                    continue
                idx = wp.array(particles, dtype=wp.int32, device=self.device)
                tris = wp.array(tris_np.reshape(-1), dtype=wp.int32, device=self.device)
                out = wp.empty(len(particles), dtype=wp.vec3, device=self.device)
                uvs = wp.array(planar_uvs(rest[particles]), dtype=wp.vec2, device=self.device)
                cache[name] = (idx, tris, tris_np, out, uvs)
            return cache

        def _write_poses(self) -> None:
            """Push the rig's camera transforms into the live renderer."""
            from ovrtx import Semantic

            self._rtx.write_attribute(
                prim_paths=[self.camera_path(i) for i in range(len(self.rig))],
                attribute_name="omni:xform",
                tensor=np.stack([_usd_camera_matrix(self.poses[spec.name]) for spec in self.rig]),
                semantic=Semantic.XFORM_MAT4x4,
            )

        def step_rig(self) -> None:
            """One tracer pass over every product, synchronously.

            Synchronous is the whole point. ``step_async`` is what makes a
            viewport smooth -- present the last frame, start the next -- and what
            makes an observation a frame stale, which for a camera that just
            moved is a picture of where it used to be.
            """
            self.outputs = self._rtx.step(
                render_products={self.product_path(i) for i in range(len(self.rig))},
                delta_time=1.0 / self.fps,
            )

        def _render_and_display(self):
            """Replaces the viewport's own render, which nothing here reads.

            ``end_frame`` calls this after pushing the frame's transforms, which
            is the first moment ``self._rtx`` exists -- the renderer is created
            inside the ``_init_ovrtx`` that the same call triggers -- and so the
            first moment the camera poses can be written.
            """
            if self._rtx is None or self._should_close:
                return
            self._write_poses()
            self.step_rig()

    return _StudioViewerRTX


def _vertex_normals(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area-weighted smooth normals, so a sheet does not render faceted."""
    p = np.asarray(points, dtype=np.float32)
    a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    fn = np.cross(b - a, c - a)
    n = np.zeros_like(p)
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(length, 1.0e-12)


# ----------------------------------------------------------------- colour
CALIBRATION = os.environ.get("ROBOSIMSTUDIO_RTX_CALIBRATION", "calibrated").strip().lower()
"""Which path-tracer calibration this process renders under: ``calibrated`` or ``legacy``.

**The default is the calibration; ``legacy`` exists so the change can be
measured, and rolled back, without a revert.** The three constants below move
every RTX frame in every task toward the tiled tracer's -- which is what every
colour-word prompt and every hue test in every vision graph was tuned on -- and
they were measured on one suite. The graphs at risk are the ones with hard-coded
pixel predicates that were tuned on RTX frames rendered BEFORE this: the cable
run's achromatic jacket floor, the receptacle's blue-face pixel count, the
routing flange's value floor. ``ROBOSIMSTUDIO_RTX_CALIBRATION=legacy`` renders exactly
what this tree rendered before the calibration landed, so a graph can be run
both ways on one checkout and the difference attributed.

Tool hanging has no exposure either way: its perceiver asks SAM3 for words and
tests no pixel value."""

_LEGACY = CALIBRATION == "legacy"
_LIGHT_SCALE = (1.0 / 0.75) if _LEGACY else 1.0

PALETTE_IS_DISPLAY = not _LEGACY
"""Whether the palette's numbers are display (sRGB-encoded) values, to be decoded before the path tracer sees them.

**They are, and the two tracers disagree about what a colour is.** The tiled
tracer writes ``model.shape_color`` to the frame as it stands -- the bench's
0.28 is the pixel value 72 -- so every colour-word prompt and every hue test in
the vision graphs was measured against the palette as display values. The
path tracer takes ``diffuseColor`` and ``displayColor`` as *linear* albedo and
encodes the result for display: fed the same numbers it renders every surface
lighter and flatter (the brass gear's (r-g)/(r-b) fell from 0.26 to 0.14, the
pinion's from 0.40 to 0.18, and the pinion failed the orange test). Decoding
the palette with the sRGB transfer function at the three places a colour
reaches the stage -- the instances' ``displayColor`` primvar, a baked material
colour, the live colour write -- gives the tracer the albedo those display
values denote."""

TONEMAP_OPERATOR = "acesApproximation" if _LEGACY else "none"
"""The RTX post tone-map operator, set on every render product of the rig.

OVRTX's default is ``acesApproximation``, a filmic curve that desaturates and
warms saturated colours: measured on the pinion, whose palette hue is 36
degrees, the ACES frame reads 42-48. ``none`` hands off untone-mapped (the
sRGB encode still applies) and the hue comes back within a degree of the tiled
frame's. ``raw`` was tried and clips the bench to white."""


def _albedo(color):
    """A palette colour as the path tracer should see it (see :data:`PALETTE_IS_DISPLAY`)."""
    color = tuple(float(v) for v in color)
    return srgb_to_linear(color) if PALETTE_IS_DISPLAY else color


def _decode_display_color(prim) -> None:
    """Rewrite a prim's ``displayColor`` primvar from display values to albedo, once."""
    from pxr import UsdGeom

    pv = UsdGeom.PrimvarsAPI(prim)
    if not pv.HasPrimvar("displayColor"):
        return
    primvar = pv.GetPrimvar("displayColor")
    marker = prim.GetAttribute("robosimstudio:displayColorDecoded")
    if marker and marker.Get():
        return
    from pxr import Gf, Sdf, Vt

    # The viewer authors the primvar as time samples at its frame index, and a
    # sample wins over a default at every time -- so each sample is decoded in
    # place rather than a default written under them.
    times = list(primvar.GetAttr().GetTimeSamples())
    for t in times or [None]:
        values = primvar.Get() if t is None else primvar.Get(t)
        if values is None:
            continue
        decoded = Vt.Vec3fArray([Gf.Vec3f(*srgb_to_linear(tuple(v))) for v in values])
        primvar.Set(decoded) if t is None else primvar.Set(decoded, t)
    prim.CreateAttribute("robosimstudio:displayColorDecoded", Sdf.ValueTypeNames.Bool, custom=True).Set(True)


# ----------------------------------------------------------------- materials
class _MaterialLibrary:
    """One ``UsdPreviewSurface`` per distinct finish, bound to many prims.

    Untextured materials take their colour from the prim's ``displayColor``
    primvar, so every shape of one finish shares a material whatever its colour
    and a live colour write needs no material of its own. A constant colour is
    only baked for prims that carry no primvar -- deformable meshes, scenery.
    """

    def __init__(self, stage, root: str = "/root/Materials") -> None:
        from pxr import Sdf, UsdGeom

        self.stage = stage
        self.root = root
        self._sdf = Sdf
        self._usdgeom = UsdGeom
        self._cache: dict[tuple, Any] = {}
        self._count = 0

    def material(self, finish: Finish, color: tuple[float, float, float] | None = None):
        """The material for ``finish``; ``color`` bakes a constant colour."""
        key = (finish, None if color is None else tuple(round(float(v), 4) for v in color))
        found = self._cache.get(key)
        if found is not None:
            return found
        made = self._build(finish, key[1])
        self._cache[key] = made
        return made

    def bind(self, prim, finish: Finish, color: tuple[float, float, float] | None = None) -> None:
        from pxr import UsdShade

        material = self.material(finish, color)
        UsdShade.MaterialBindingAPI.Apply(prim)
        UsdShade.MaterialBindingAPI(prim).Bind(material)

    def _build(self, finish: Finish, color):
        from pxr import Gf, Sdf, UsdShade

        self._count += 1
        path = f"{self.root}/finish_{self._count}"
        _ensure_scopes(self.stage, path)
        material = UsdShade.Material.Define(self.stage, path)
        surface = UsdShade.Shader.Define(self.stage, f"{path}/PreviewSurface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(finish.roughness))
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(finish.metallic))
        surface.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
        if finish.translucent:
            surface.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(finish.opacity))
            surface.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(float(finish.ior))
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

        st = None
        if finish.texture or finish.roughness_texture:
            reader = UsdShade.Shader.Define(self.stage, f"{path}/PrimvarSt")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            xf = UsdShade.Shader.Define(self.stage, f"{path}/UvTransform")
            xf.CreateIdAttr("UsdTransform2d")
            xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*[float(v) for v in finish.uv_scale]))
            xf.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(
                Gf.Vec2f(*[float(v) for v in finish.uv_offset])
            )
            st = xf.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        diffuse = surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        if finish.texture:
            tex = self._texture(f"{path}/DiffuseTexture", texture_path(finish.texture), st, "auto")
            diffuse.ConnectToSource(tex.ConnectableAPI(), "rgb")
        elif color is None:
            reader = UsdShade.Shader.Define(self.stage, f"{path}/PrimvarColor")
            reader.CreateIdAttr("UsdPrimvarReader_float3")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
            diffuse.ConnectToSource(reader.ConnectableAPI(), "result")
        else:
            diffuse.Set(Gf.Vec3f(*_albedo(color)))
        if finish.roughness_texture:
            tex = self._texture(f"{path}/RoughnessTexture", texture_path(finish.roughness_texture), st, "raw")
            surface.GetInput("roughness").ConnectToSource(tex.ConnectableAPI(), "r")
        return material

    def _texture(self, path: str, file: str, st, color_space: str):
        from pxr import Sdf, UsdShade

        tex = UsdShade.Shader.Define(self.stage, path)
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(file)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        tex.CreateOutput("r", Sdf.ValueTypeNames.Float)
        if st is not None:
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st)
        return tex


def _ensure_scopes(stage, path: str) -> None:
    from pxr import Sdf, UsdGeom

    parent = Sdf.Path(path).GetParentPath()
    parts = []
    while parent and parent != Sdf.Path.absoluteRootPath:
        parts.append(parent)
        parent = parent.GetParentPath()
    for scope in reversed(parts):
        if not stage.GetPrimAtPath(scope):
            UsdGeom.Scope.Define(stage, scope)


def _first_value(attr):
    """An attribute's value, from its first time sample when it has no default.

    ``ViewerUSD`` writes points and transforms as time samples at the frame
    index, so a plain ``Get()`` on them returns nothing -- and a prototype read
    that way has no points to project UVs from.
    """
    value = attr.Get()
    if value is None:
        times = attr.GetTimeSamples()
        if times:
            value = attr.Get(times[0])
    return value if value is not None else []


def _ensure_prototype_uvs(mesh_prim, *, metric: bool = False) -> None:
    """Give a prototype mesh an ``st`` primvar in its own metres.

    Newton's primitive prototypes arrive with UVs already -- one ``[0, 1]``
    square per face -- which is right for a label and wrong for a finish whose
    ``uv_scale`` is stated in repeats per metre: a 24 cm plate would carry the
    same number of grid cells as the 60 cm tile beside it. ``metric`` replaces
    those with a box projection of the mesh's own coordinates, so the grid on
    a plate continues the grid on the bench. An authored mesh keeps its UVs
    unless it has none.

    The mesh is unwelded first -- every triangle gets its own three vertices --
    so the projection can be stored per vertex, which is the interpolation
    every path here uses.
    """
    from pxr import Sdf, UsdGeom

    pv = UsdGeom.PrimvarsAPI(mesh_prim)
    if pv.HasPrimvar("st") and not metric:
        return
    points_attr = mesh_prim.GetPointsAttr()
    points = np.asarray(_first_value(points_attr), dtype=np.float64)
    indices = np.asarray(_first_value(mesh_prim.GetFaceVertexIndicesAttr()), dtype=np.int64)
    if points.size == 0 or indices.size == 0 or len(indices) % 3:
        return
    tris = indices.reshape(-1, 3)
    uv = box_projected_uvs(points, tris).astype(np.float32)
    unwelded = points[tris.reshape(-1)].astype(np.float32)
    for t in points_attr.GetTimeSamples() or [None]:
        points_attr.Set(unwelded) if t is None else points_attr.Set(unwelded, t)
    mesh_prim.GetFaceVertexIndicesAttr().Set(np.arange(len(unwelded), dtype=np.int32))
    normals_attr = mesh_prim.GetNormalsAttr()
    normals = np.asarray(_first_value(normals_attr), dtype=np.float32)
    if len(normals) == len(points):
        spread = normals[tris.reshape(-1)]
        for t in normals_attr.GetTimeSamples() or [None]:
            normals_attr.Set(spread) if t is None else normals_attr.Set(spread, t)
    if pv.HasPrimvar("st"):
        pv.RemovePrimvar("st")
    st = pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(uv)


# ----------------------------------------------------------------- scenery kit
class _StageKit:
    """What a scenery builder is handed: boxes and meshes with finishes."""

    def __init__(self, stage, library: _MaterialLibrary, *, cell, surface_z: float) -> None:
        self.stage = stage
        self.library = library
        self.cell = cell
        self.surface_z = float(surface_z)
        self._count = 0

    def box(self, name: str, center, half, *, color, finish: Finish, quat=None) -> None:
        points, tris, per_vertex = box_mesh(center, half, quat)
        self.mesh(name, points, tris, color=color, finish=finish, uvs=per_vertex)

    def mesh(self, name: str, points, faces, *, color, finish: Finish, uvs=None) -> None:
        from pxr import Sdf, UsdGeom

        self._count += 1
        safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name) or f"mesh_{self._count}"
        path = f"/root/scenery/{safe}"
        _ensure_scopes(self.stage, path)
        prim = UsdGeom.Mesh.Define(self.stage, path)
        pts = np.asarray(points, dtype=np.float32)
        tris = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        prim.GetPointsAttr().Set(pts)
        prim.GetFaceVertexCountsAttr().Set([3] * len(tris))
        prim.GetFaceVertexIndicesAttr().Set(tris.reshape(-1))
        prim.GetNormalsAttr().Set(_vertex_normals(pts, tris))
        prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        prim.GetDoubleSidedAttr().Set(True)
        if uvs is not None:
            st = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            )
            st.Set(np.asarray(uvs, dtype=np.float32))
        else:
            _ensure_prototype_uvs(prim)
        self.library.bind(prim.GetPrim(), finish, tuple(float(v) for v in color))


# ----------------------------------------------------------------- lighting
def _environment_map_path(environment_map: str) -> str:
    """Resolve a dome's HDR to an absolute path on disk.

    The one texture in this backend that used to bypass resolution: the raw
    string went to ``TextureFile``, and USD then resolved a relative path
    against the temporary stage OVRTX opens rather than against this
    repository -- so ``"environments/hangar.hdr"`` silently found nothing.

    Both seams are honoured, in the order that keeps each one's promise. A file
    goes through :func:`~robosimstudio.paths.resolve_path`, which searches the
    asset roots, obeys ``ROBOSIMSTUDIO_ASSETS`` and passes an absolute path
    through untouched -- which is what lets somebody try a panorama without
    vendoring it first. A registered name falls through to
    :func:`~robosimstudio.core.render.finish.texture_path`, so a *procedural*
    environment map is possible; note that path caches an 8-bit PNG, which is
    fine as a backdrop and poor as a light, since what a panorama is worth as
    illumination lives in the range an 8-bit file does not have.
    """
    from robosimstudio.paths import resolve_path

    try:
        return str(resolve_path(environment_map))
    except FileNotFoundError:
        return texture_path(environment_map)


LAB_PANEL_PER_KEY = 3750.0 * _LIGHT_SCALE
"""Panel-light intensity per unit of :attr:`RenderSpec.key_intensity`.

Three quarters of the 5000 it started at, and the dome constants below with it.
Measured on a single-gear meshing scene's overhead camera against the
tiled tracer's frame, which is what every vision prompt was tuned on: with the
palette decoded and the tone map off (see :data:`PALETTE_IS_DISPLAY`) the bench
top read 90 against tiled's 72; at three quarters it reads within a few counts.
Before either fix it read 192."""

LAB_DOME_PER_AMBIENT = 675.0 * _LIGHT_SCALE
"""Dome-light intensity per unit of mean :attr:`RenderSpec.ambient`.

Untextured. A dome carrying an image emits ``intensity x colour x texture``, and
a panorama's radiance is not normalised to one, so a textured dome takes
:data:`ROOM_DOME_PER_AMBIENT` instead."""

ROOM_DOME_PER_AMBIENT = 375.0 * _LIGHT_SCALE
"""Dome-light intensity per unit of mean ambient, when the dome carries an HDR.

Lower than :data:`LAB_DOME_PER_AMBIENT` because the image is doing part of the
work: the rooms in ``core/render/room.py`` were picked so their
upper-hemisphere mean radiance -- what an up-facing bench top integrates -- is
near 1.0, which is what makes one constant serve all of them. The two upstream
stacks this was read from bracket it at 500 and 1500 on the same panorama, in
their own renderer's units; this is a starting point to be tuned by measuring
frame mean, not a number copied from either."""

DOME_TEXTURE_POLE_FIX = (-90.0, 0.0, 0.0)
"""Rotation applied to a *textured* dome [deg, XYZ].

``UsdLuxDomeLight``'s texture is a latitude-longitude map about the stage's
**+Y** axis, and a Newton stage is Z-up -- so an unrotated panorama lays its
horizon down the vertical. This tips it back. It is applied only when there is
a texture, because it would otherwise rotate a uniform colour for nothing."""


def _add_lab_lights(
    stage,
    *,
    center,
    surface_z: float,
    environment_map: str,
    key: float,
    ambient: float,
    dome_yaw_deg: float = 0.0,
) -> None:
    """A neutral dome under one large panel: soft, near-shadowless top light.

    ``key`` and ``ambient`` are the spec's own backend-neutral knobs, in
    pyrender's scale, so the house look's 1.05 / 0.30 lands the table near the
    exposure the other backends give it.

    Given an ``environment_map`` the dome stops being a uniform colour and
    becomes a room: the image lights the scene by its own directional structure,
    and -- because ``visibleInPrimaryRay`` is set with it -- fills whatever a
    camera ray reaches without hitting geometry. That attribute is not in the
    ``UsdLux`` schema; it is a custom bool the RTX delegate reads, written the
    same way by every stack that uses one. It is set **only** alongside a
    texture: an untextured dome made visible in the primary ray would flood the
    frame with flat dome colour and quietly override
    :attr:`~robosimstudio.core.render.spec.RenderSpec.background`.

    ``dome_yaw_deg`` turns the panorama about the stage's vertical (world +Z)
    axis -- :data:`DOME_TEXTURE_POLE_FIX` only fixes *which way is up*, so an
    unrotated panorama's own front (wherever it was captured facing) lands at
    an arbitrary bearing relative to whichever way a scene's camera happens to
    look, and a doorway or light fixture can end up floating mid-wall instead
    of sitting where the rest of the room implies it should. There is no way to
    derive the right angle -- it is tuned per camera by looking -- so the
    default is 0 and a caller states what it measured.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    dome_xf = UsdGeom.Xform.Define(stage, "/root/_LabDome")
    dome_xf.ClearXformOpOrder()
    dome = UsdLux.DomeLight.Define(stage, "/root/_LabDome/light")
    dome.GetColorAttr().Set(Gf.Vec3f(0.94, 0.95, 0.97))
    if environment_map:
        dome.GetTextureFileAttr().Set(_environment_map_path(environment_map))
        dome.GetTextureFormatAttr().Set(UsdLux.Tokens.latlong)
        dome.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
        # Listed first, so it is the outermost (world-space) rotation: applied
        # after the pole fix has already put the panorama's horizon level.
        dome_xf.AddRotateZOp().Set(float(dome_yaw_deg))
        dome_xf.AddRotateXYZOp().Set(Gf.Vec3f(*DOME_TEXTURE_POLE_FIX))
    per_ambient = ROOM_DOME_PER_AMBIENT if environment_map else LAB_DOME_PER_AMBIENT
    dome.GetIntensityAttr().Set(float(per_ambient * ambient))

    cx, cy = (float(v) for v in center[:2])
    panel_xf = UsdGeom.Xform.Define(stage, "/root/_LabPanel")
    panel_xf.ClearXformOpOrder()
    panel_xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, float(surface_z) + 1.7))
    panel = UsdLux.RectLight.Define(stage, "/root/_LabPanel/light")
    panel.GetWidthAttr().Set(1.4)
    panel.GetHeightAttr().Set(0.9)
    panel.GetColorAttr().Set(Gf.Vec3f(1.0, 0.97, 0.92))
    panel.GetIntensityAttr().Set(float(LAB_PANEL_PER_KEY * key))


# ----------------------------------------------------------------- decoration
def decorate_stage(
    viewer,
    *,
    model: newton.Model,
    finishes: Mapping[int, Finish] | None,
    deformable_finishes: Mapping[str, Finish] | None,
    live_shapes: Sequence[int] | None,
    scenery: str | Callable | None = None,
    environment: str = "default",
    environment_map: str = "",
    dome_yaw_deg: float = 0.0,
    cell=None,
    surface_z: float = 0.0,
    key_intensity: float = 1.0,
    ambient: float = 0.3,
) -> None:
    """Bind finishes, dress the scene and light it, on the stage OVRTX will load.

    Pure in the sense that it only edits ``viewer.stage`` through the maps the
    viewer already holds, so it can be exercised on a plain ``ViewerUSD`` stage
    without a GPU.
    """
    stage = viewer.stage
    library = _MaterialLibrary(stage)
    finishes = dict(finishes or {})
    live = {int(s) for s in (live_shapes if live_shapes is not None else ())}
    live_paths: dict[int, str] = {}
    colors = getattr(model, "shape_color", None)
    colors = colors.numpy() if hasattr(colors, "numpy") else None

    batches = getattr(viewer, "_shape_to_batch", None) or []
    textured_prototypes: dict[str, bool] = {}
    mesh_types = {int(newton.GeoType.MESH), int(getattr(newton.GeoType, "CONVEX_MESH", newton.GeoType.MESH))}
    for s, batch in enumerate(batches):
        if batch is None:
            continue
        local = batch.model_shapes.index(s)
        path = viewer._get_path(batch.name) + f"/instance_{local}"
        prim = stage.GetPrimAtPath(path)
        if not prim:
            continue
        finish = finishes.get(s)
        if finish is None:
            color = tuple(float(v) for v in colors[s][:3]) if colors is not None and s < len(colors) else None
            finish = finish_for(color)
        if PALETTE_IS_DISPLAY:
            _decode_display_color(prim)
        library.bind(prim, finish)
        if finish.texture or finish.roughness_texture:
            # A primitive's own UVs are one square per face; a finish stated in
            # repeats per metre wants the mesh's coordinates instead.
            textured_prototypes[batch.mesh] = int(getattr(batch, "geo_type", -1)) not in mesh_types
        if s in live:
            live_paths[s] = path
    for mesh_name, metric in textured_prototypes.items():
        proto = getattr(viewer, "_meshes", {}).get(mesh_name)
        if proto is not None:
            _ensure_prototype_uvs(proto, metric=metric)
    viewer.live_paths = live_paths

    for name, finish in (deformable_finishes or {}).items():
        prim = stage.GetPrimAtPath(viewer._get_path(viewer._qualify(f"/model/deformables/{name}")))
        if not prim:
            continue
        color = None
        handle_color = getattr(finish, "_color", None)
        del handle_color
        library.bind(prim, finish, _deformable_color(viewer, name))

    center = (0.0, 0.0)
    if cell is not None:
        center = tuple(float(v) for v in getattr(cell, "center", (0.0, 0.0, 0.0))[:2])
    if environment == "lab":
        _add_lab_lights(
            stage,
            center=center,
            surface_z=surface_z,
            environment_map=environment_map,
            key=key_intensity,
            ambient=ambient,
            dome_yaw_deg=dome_yaw_deg,
        )
    if scenery:
        builder = get_scenery(scenery) if isinstance(scenery, str) else scenery
        builder(_StageKit(stage, library, cell=cell, surface_z=surface_z))


def _deformable_color(viewer, name: str) -> tuple[float, float, float]:
    colors = getattr(viewer, "_deformable_colors", None) or {}
    return tuple(float(v) for v in colors.get(name, (0.85, 0.85, 0.88)))


# ----------------------------------------------------------------- backend
class RendererRTX(RendererBase):
    """Path-traced RGB and metric depth for a whole camera rig, headless.

    Args:
        model: The finalized model.
        cameras: Cameras to render. Resolutions and intrinsics vary freely --
            each one gets its own render product -- and a principal-point offset
            is carried exactly, through the USD film back's aperture offsets.
        channels: Restrict every camera to these channels.
        environment: Lighting preset -- one of :data:`ENVIRONMENTS`. Newton's
            ``default`` is a dark key with a visible glare, ``studio`` an even
            fill several stops brighter with a sphere light in shot; ``lab`` is
            this module's soft panel over a neutral dome, which is what an
            office bench is lit by.
        environment_map: An equirectangular HDR image for the dome (``lab``).
        dome_yaw_deg: Turns ``environment_map`` about the stage's vertical axis
            [deg]; see :func:`_add_lab_lights`. 0 leaves the panorama at
            whatever bearing it was authored at.
        scenery: A registered scenery name or a builder callable, or nothing.
        quality: ``"preview"`` or ``"final"``; picks ``accumulate`` unless that
            is stated.
        fps: Frame rate stamped into the viewer, which affects motion blur.
        accumulate: Tracer passes per output frame. **This is the convergence
            control**, and it is the one knob worth understanding here. The
            product takes one sample per pixel per pass, so a single pass is a
            low-sample estimate; extra passes of the *same* instant let the
            tracer converge on it.

            It matters on a moving scene and hardly at all on a still one: the
            tracer accumulates across steps too, so a scene that has stopped
            converges whatever this says, and what a pass buys is the pixels
            motion just invalidated. Measured that way -- one recorded
            tool-sorting trajectory replayed through a *fresh* renderer
            per setting, scored against a 32-pass replay of the same states, in
            8-bit colour:

            ==============  ====================  ==================
            passes          distance from         seconds for the
                            converged             three-camera rig
            ==============  ====================  ==================
            1               1.14                  0.021
            2               0.89                  0.028
            4               0.72                  0.044
            6               0.66                  0.061
            8               0.60                  0.077
            ==============  ====================  ==================

            The knee is around four and the default sits just past it. The cost
            is *affine*, not linear -- about 13 ms of rig and read-back plus 8 ms
            a pass -- so the saving from dropping to one pass is two thirds,
            not five sixths.

            The same knob read the other way is the flicker control, which is
            what :data:`QUALITY_PASSES` is fitted to: one pass per frame is an
            independent low-sample estimate of the same image, so the Monte
            Carlo noise is different every frame and a replayed trajectory
            strobes -- the geometry rock steady and the shading crawling.
            Measured on the tape scene, mean absolute frame-to-frame difference
            over repeated renders of one unchanged state: 3.99, 1.94, 1.24,
            1.04, 0.76, 0.68, 0.72 -- a noise floor around the sixth pass, which
            is ``preview``. On a bench scene under ``lab`` lighting the ladder
            keeps falling to the twelfth, which is ``final``. Asking OVRTX for
            its offline path-tracing mode instead was tried and is ignored by
            the standalone renderer, so passes are the knob.
        finishes: ``shape index -> Finish``, normally the scene handles' table.
        deformables: ``name -> DeformableHandle``; each becomes its own mesh.
        deformable_finishes: ``name -> Finish`` for those meshes.
        deformable_colors: ``name -> RGB`` for those meshes, since a deformable
            carries no ``displayColor``.
        live_shapes: Shapes whose colour is re-read every frame.
        cell: The :class:`~robosimstudio.specs.CellSpec` the scene stands in,
            for the backdrop and the panel light's placement.
        surface_z: World z of the work surface [m].
        key_intensity: Scales the ``lab`` panel light; the spec's own key.
        ambient: Scales the ``lab`` dome light; the spec's own ambient, RGB or
            scalar (the mean is taken).

    Note:
        Whether this belongs in a control loop is now a question with an answer.
        Measured end to end on a tool-sorting scene, a scripted rollout whose
        every control step carries a three-camera path-traced observation at one
        pass runs at a median 32.4 ms a step, against 15.5 ms for the same loop
        on the default tiled rig. That is a real cost and ``render_every`` is the
        knob for it -- and it is a *step* cadence rather than a wall clock,
        deliberately, so that what a policy saw does not depend on how loaded the
        machine was.
    """

    name = "rtx"
    supported_channels = ("rgb", "depth")

    wants: tuple[str, ...] = (
        "finishes",
        "deformables",
        "deformable_finishes",
        "deformable_colors",
        "live_shapes",
        "surface_z",
        "cell",
    )

    @classmethod
    def look_kwargs(cls, look: Any) -> dict[str, Any]:
        """What this backend takes from a look; see :meth:`RendererBase.look_kwargs`."""
        out: dict[str, Any] = {}
        put(out, "environment", look.environment)
        # The ``lab`` preset scales its panel by the key and its dome by the
        # ambient, so the two knobs every other backend already takes mean
        # something here too instead of being dropped.
        put(out, "key_intensity", look.key_intensity)
        put(out, "ambient", look.ambient)
        put(out, "environment_map", look.environment_map)
        put(out, "scenery", look.scenery)
        put(out, "quality", look.quality)
        return out

    def __init__(
        self,
        model: newton.Model,
        cameras: Sequence[CameraSpec],
        *,
        channels: Sequence[str] | None = None,
        environment: str = "default",
        environment_map: str = "",
        dome_yaw_deg: float = 0.0,
        scenery: str | Callable | None = None,
        quality: str = "preview",
        fps: int = 30,
        accumulate: int | None = None,
        finishes: Mapping[int, Finish] | None = None,
        deformables: Mapping[str, Any] | None = None,
        deformable_finishes: Mapping[str, Finish] | None = None,
        deformable_colors: Mapping[str, Sequence[float]] | None = None,
        live_shapes: Sequence[int] | None = None,
        cell: Any = None,
        surface_z: float = 0.0,
        key_intensity: float = 1.0,
        ambient: float | Sequence[float] = 0.3,
    ) -> None:
        super().__init__(cameras, channels=channels)
        blank = [c.name for c in self.cameras if not self.channels[c.name]]
        if blank:
            # A render product with no ordered vars is a product OVRTX renders
            # and nobody can read, which costs a full tracer pass and returns
            # nothing. Saying so beats authoring one.
            raise ValueError(f"cameras {blank} were left with no channels to render")
        if environment not in ENVIRONMENTS:
            raise ValueError(f"unknown RTX environment {environment!r}; choose from {ENVIRONMENTS}")
        if quality not in QUALITY_PASSES:
            raise ValueError(f"unknown RTX quality {quality!r}; choose from {sorted(QUALITY_PASSES)}")
        self.model = model
        self.world = 0
        """Which world :meth:`render` returns. :meth:`render_worlds` covers all."""
        self.num_worlds = max(1, int(getattr(model, "world_count", 1) or 1))
        """Worlds in the model, each rendered in its own pass. See :meth:`render_worlds`."""
        self.bodies_per_world = (int(model.body_count) // self.num_worlds) if model.body_count else 0
        """The stride a body-parented camera's parent index is offset by."""
        # The viewport's own resolution, which nothing reads a pixel of -- the
        # rig's products carry their own. Newton allocates a framebuffer for it,
        # so it is the first camera's rather than something arbitrary and large.
        self.width, self.height = int(self.cameras[0].width), int(self.cameras[0].height)
        self.accumulate = max(1, int(QUALITY_PASSES[quality] if accumulate is None else accumulate))
        self.live_shapes = np.asarray(list(live_shapes) if live_shapes is not None else [], dtype=np.int32)
        self._live_cache: np.ndarray | None = None

        def decorate(viewer) -> None:
            decorate_stage(
                viewer,
                model=model,
                finishes=finishes,
                deformable_finishes=deformable_finishes,
                live_shapes=self.live_shapes,
                scenery=scenery,
                environment=environment,
                environment_map=environment_map,
                dome_yaw_deg=dome_yaw_deg,
                cell=cell,
                surface_z=surface_z,
                key_intensity=float(key_intensity),
                ambient=float(np.mean(ambient)) if not np.isscalar(ambient) else float(ambient),
            )

        self.viewer = _viewer_class()(
            decorate=decorate,
            deformables=deformables,
            width=self.width,
            height=self.height,
            headless=True,
            vsync=False,
            environment="none" if environment == "lab" else environment,
            fps=int(fps),
            # Every read-back below is of the step that just ran.
            async_rendering=False,
        )
        self.viewer._deformable_colors = {k: tuple(float(x) for x in v) for k, v in (deformable_colors or {}).items()}
        self.viewer.rig = tuple(self.cameras)
        self.viewer.rig_vars = {
            cam.name: tuple(
                var
                for channel, var in (("rgb", "LdrColor"), ("depth", _DEPTH_VAR))
                if channel in self.channels[cam.name]
            )
            for cam in self.cameras
        }
        self.viewer.poses = {cam.name: cam.pose_on_body(None) for cam in self.cameras}
        self.viewer.set_model(model)
        self.viewer.picking_enabled = False
        for flag in (
            "show_particles",
            "show_joints",
            "show_com",
            "show_contacts",
            "show_collision",
            "show_inertia_boxes",
        ):
            if hasattr(self.viewer, flag):
                setattr(self.viewer, flag, False)
        self._time = 0.0

    # ------------------------------------------------------------------ render
    def _read(self, cam: CameraSpec, index: int) -> dict[str, np.ndarray]:
        """One camera's channels out of the last step's product.

        Depth is metres to the image plane, with the non-finite samples -- the
        sky, and anything past the far plane -- zeroed the way every consumer of
        this stack's depth expects, so nothing downstream sees an infinity where
        it expects a range.
        """
        from ovrtx import Device

        path = self.viewer.product_path(index)
        outputs = self.viewer.outputs
        # Dict-*like*, not a dict: no ``get``, so the membership test is the
        # only way to ask without raising OVRTX's own KeyError first.
        if outputs is None or path not in outputs or not outputs[path].frames:
            raise RuntimeError(f"the RTX step produced no frame for camera {cam.name!r}")
        product = outputs[path]
        wanted = self.channels[cam.name]
        frame: dict[str, np.ndarray] = {}
        for var_name, var in product.frames[0].render_vars.items():
            with var.map(device=Device.CPU) as mapped:
                arr = np.array(np.from_dlpack(mapped), copy=True)
            if var_name == "LdrColor" and "rgb" in wanted:
                frame["rgb"] = np.ascontiguousarray(arr[:, :, :3])
            elif var_name == _DEPTH_VAR and "depth" in wanted:
                frame["depth"] = np.nan_to_num(arr.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        missing = sorted(set(wanted) - set(frame))
        if missing:
            raise RuntimeError(
                f"camera {cam.name!r} asked for {missing} but the RTX product carried none; "
                f"the render var did not resolve"
            )
        return frame

    def _push_live_colors(self) -> None:
        """Write the live shapes' current colours to their prims, if any moved."""
        viewer = self.viewer
        if viewer._rtx is None or not viewer.live_paths:
            return
        colors = getattr(self.model, "shape_color", None)
        if colors is None or not hasattr(colors, "numpy"):
            return
        current = np.asarray(colors.numpy(), dtype=np.float32)[self.live_shapes, :3]
        if self._live_cache is not None and current.shape == self._live_cache.shape:
            changed = np.flatnonzero(np.abs(current - self._live_cache).max(axis=1) > 1.0e-4)
        else:
            changed = np.arange(len(current))
        if len(changed) == 0:
            return
        paths = [
            viewer.live_paths[int(self.live_shapes[i])]
            for i in changed
            if int(self.live_shapes[i]) in viewer.live_paths
        ]
        if paths:
            written = current
            if PALETTE_IS_DISPLAY:
                written = np.asarray([srgb_to_linear(tuple(c)) for c in current], dtype=np.float32)
            tensors = [
                viewer._make_point3f_dltensor(written[i : i + 1])
                for i in changed
                if int(self.live_shapes[i]) in viewer.live_paths
            ]
            viewer._rtx.write_array_attribute(prim_paths=paths, attribute_name="primvars:displayColor", tensors=tensors)
        self._live_cache = current

    def render(self, state: newton.State) -> FrameSet:
        """The rig's frames for one world -- :attr:`world`. See :meth:`render_worlds`."""
        return self._render_one(state, self.world)

    def render_worlds(self, state: newton.State, worlds: Sequence[int] | None = None) -> list[FrameSet]:
        """Every world's frames, one world at a time. ``[num_worlds]`` of FrameSet.

        Args:
            state: What to photograph.
            worlds: Which worlds to render, or ``None`` for all. Here this is a
                real saving rather than a narrowed copy -- each world is its own
                tracer pass, so a wave where three worlds out of sixteen asked
                for a picture pays for three.

        **A pass per world, in the stage the single-world path already authors,
        and that is the point rather than a compromise.** The tiled tracer
        renders every world in one megakernel because it traces the model
        directly; this one draws a *scene*, and a replicated model's worlds are
        superimposed at the origin -- so one picture of the stage is a picture of
        every world at once.

        Two ways out of that, and this is the second. The stage could separate
        the worlds and carry a camera rig per world, which is one tracer step for
        the fleet; it also means the room, the dome and every scenery prim exist
        once and would have to be authored ``W`` times at the offsets, and every
        one of those copies is a chance for a world to be lit differently from
        the world the calibration was measured on. Hiding the other worlds
        instead leaves the stage *exactly* as a single-world run authored it --
        same lights, same dome, same exposure, same tone map -- so a batched
        frame and a serial frame differ in nothing but which bodies were drawn.

        What it costs is the per-step overhead ``W`` times over. It does not cost
        ``W`` times the *tracing*: a path tracer renders W worlds by tracing W
        worlds' pixels whichever way they are arranged, and one step over ``W x C``
        products traces the same pixels as ``W`` steps over ``C``.

        Newton's own contract is what makes it safe: ``set_visible_worlds`` is
        documented as callable between frames, and preserves the geometry cache
        across the change.
        """
        if self.num_worlds <= 1:
            return [self._render_one(state, 0)]
        wanted = sorted({int(w) for w in (range(self.num_worlds) if worlds is None else worlds)})
        out: list[FrameSet] = [{} for _ in range(self.num_worlds)]
        try:
            for world in wanted:
                self.viewer.set_visible_worlds([world])
                out[world] = self._render_one(state, world)
        finally:
            # Left as it was found, so a caller that renders one world after a
            # batch does not silently get whichever world came last.
            self.viewer.set_visible_worlds(None)
        return out

    def _world_view(self, state: newton.State, world: int) -> newton.State:
        """The state as world ``world`` alone would present it: its body poses tiled over every world.

        Newton's RTX viewer keeps one drawn instance per template shape and
        writes every world's transform into it, so the picture of a wave showed
        the LAST world's layout whichever world was asked for, and
        ``set_visible_worlds`` -- honoured by the GL viewer's kernels -- is not
        read by the RTX transform flush at all. Measured on a two-world hanging
        wave: both frames matched the serial frame of world 1's episode to
        0.07 % and differed from world 0's by 4.7 %, and moving world 0's
        wrench 80 mm in physics changed world 0's frame by 0.0 %. So every
        world of the eight-world vision waves was photographed with world 7's
        wrench. Tiling the asked world's rows over the wave makes whichever
        instance the viewer draws that world's, on either reading of the viewer.
        """
        if self.num_worlds <= 1 or state.body_q is None:
            return state
        import copy

        rows = state.body_q.numpy()
        span = self.bodies_per_world
        start = int(world) * span
        tiled = np.tile(rows[start : start + span], (self.num_worlds, 1))
        view = copy.copy(state)
        view.body_q = wp.array(tiled, dtype=wp.transform, device=state.body_q.device)
        return view

    def _render_one(self, state: newton.State, world: int) -> FrameSet:
        """One world's frames. The caller decides which worlds are visible."""
        body_q = state.body_q.numpy() if any(c.parent_body >= 0 for c in self.cameras) else None
        offset = int(world) * self.bodies_per_world
        self.viewer.poses = {cam.name: cam.pose_on_body(body_q, body_offset=offset) for cam in self.cameras}
        self._push_live_colors()
        state = self._world_view(state, world)

        # One time stamp for the whole burst. Advancing it between passes would
        # make each one a *new* frame, which is what resets the tracer's
        # accumulation and produces the flicker this loop exists to avoid -- the
        # passes have to agree that they are the same instant.
        #
        # Between *worlds* it does advance, and must: they are different pictures
        # and accumulating one onto the last would blend two worlds together.
        self._time += 1.0e-3
        self.viewer.begin_frame(self._time)
        self.viewer.log_state(state)
        self.viewer.end_frame()
        for _ in range(self.accumulate - 1):
            self.viewer.step_rig()

        return {cam.name: self._read(cam, index) for index, cam in enumerate(self.cameras)}

    def close(self) -> None:
        if self.viewer is not None:
            try:
                if getattr(self.viewer, "_rtx", None) is not None:
                    self.viewer._rtx.destroy()
                    self.viewer._rtx = None
            except Exception:
                pass
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
