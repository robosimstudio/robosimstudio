# SPDX-License-Identifier: Apache-2.0
"""The one spec that lives in core: a camera, as data every backend can consume.

Every other spec in this stack belongs to the toolkit layer. A camera is the
exception because a renderer is a core mechanism and it has to be told *what* to
render from -- and because a camera is one of the few things a policy sees
directly, so its intrinsics have to survive intact from the scene that declares
them to the frames a policy is evaluated on.

Pose convention is the vision one, matching :func:`~robosimstudio.core.math.look_at_matrix`:
the camera looks along its local **+z**, with **+x** right and **+y** down, so a
point in camera coordinates projects through ``K`` with no basis change. The
backends convert to whatever their renderer wants -- OpenGL's -z forward, USD's
-z forward with +y up -- inside themselves, which keeps the flips in one place
per backend rather than one place per scene.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .math import look_at_matrix, transform_matrix, xform_from_matrix

__all__ = ["CameraSpec"]


@dataclass(frozen=True, kw_only=True)
class CameraSpec:
    """One camera: where it is, what it sees, and at what resolution.

    Intrinsics may be given either as a vertical field of view or as an explicit
    ``(fx, fy, cx, cy)``. A benchmark that evaluates a policy on real-camera
    pixels needs the explicit form -- a Logitech C920's principal point is not
    at the image centre, and a renderer that assumes it is shifts every
    projection by a few pixels.

    Attributes:
        name: Key this camera's frames appear under in an observation.
        width: Image width [px].
        height: Image height [px].
        fov_y: Vertical field of view [deg]. Ignored when :attr:`fx` is set.
        fx: Horizontal focal length [px], or ``None`` to derive it from
            :attr:`fov_y`.
        fy: Vertical focal length [px].
        cx: Principal point x [px], defaulting to the image centre.
        cy: Principal point y [px], defaulting to the image centre.
        near: Near clip [m].
        far: Far clip [m]. A depth ray that hits nothing is reported as 0
            rather than as infinity, so a consumer never sees an infinity where
            it expects a range.
        eye: Camera position in world coordinates [m].
        look_at: Point the camera is aimed at [m]. Ignored when :attr:`xform`
            is set.
        up: World up hint for the look-at basis.
        xform: Explicit camera-to-world pose as ``(px, py, pz, qx, qy, qz, qw)``,
            overriding :attr:`eye` / :attr:`look_at`. This is how a camera
            mounted on a moving link is expressed, with the toolkit layer
            re-writing it per step.
        parent_body: Body this camera rides on, or -1 for world-fixed. When set,
            :attr:`xform` is read as a pose in that body's frame.
        parent_frame: An arm *role* -- ``"work"``, ``"support"``, or an arm's own
            name -- whose tool body this camera rides on, resolved to
            :attr:`parent_body` when the scene is compiled. A spec is written
            before a model exists, so a task cannot state a body index; and
            naming the link (``panda_link7``) would pin the camera to one
            embodiment and, worse, to one importer's collapse decisions. The
            role is what an eye-in-hand camera actually means.

            A camera bolted to something that is *not* an arm -- a mast, a
            pan-tilt head, the robot's own base -- names that body's **label**
            here instead, and the same resolution finds it. Pinning to a link
            is the wrong default and the right last resort: a real rig's
            overhead camera is screwed to one particular link and there is no
            role that means it.
        background: Path to an image composited behind everything the
            simulation did not draw, or ``""`` for none. This is what makes a
            *visual-matching* observation: the robot and the manipulated objects
            are rendered, and every other pixel comes from a real photograph of
            the real rig, so a policy trained on real data is evaluated on
            something close to what it saw.

            A path rather than an array, because a spec is frozen data that has
            to stay hashable and be readable without an image decoder. Whoever
            builds the renderer resolves it against their own asset root, loads
            it, scales it to this camera's resolution and hands the array to the
            backend's ``background`` argument -- nothing here opens a file. No
            shipped backend composites one; the field is kept for a renderer
            that does.
        channels: Which outputs to produce -- any of ``"rgb"``, ``"depth"``,
            ``"seg"``. A backend that cannot produce one raises rather than
            silently omitting it.
    """

    name: str = "camera"
    width: int = 640
    height: int = 480
    fov_y: float = 45.0
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    near: float = 0.01
    far: float = 10.0
    eye: tuple[float, float, float] = (1.0, -1.0, 1.2)
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.8)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    xform: tuple[float, float, float, float, float, float, float] | None = None
    parent_body: int = -1
    parent_frame: str = ""
    background: str = ""
    channels: tuple[str, ...] = ("rgb",)

    def __post_init__(self) -> None:
        unknown = set(self.channels) - {"rgb", "depth", "seg"}
        if unknown:
            raise ValueError(f"unknown camera channels {sorted(unknown)}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"camera {self.name!r} has a degenerate resolution {self.width}x{self.height}")

    # ------------------------------------------------------------- intrinsics
    @property
    def intrinsics(self) -> np.ndarray:
        """Pinhole ``K`` in pixels, shape ``[3, 3]``."""
        if self.fx is not None:
            fx = float(self.fx)
            fy = float(self.fy if self.fy is not None else self.fx)
        else:
            fy = 0.5 * self.height / math.tan(math.radians(self.fov_y) * 0.5)
            fx = fy
        cx = float(self.cx) if self.cx is not None else 0.5 * self.width
        cy = float(self.cy) if self.cy is not None else 0.5 * self.height
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def fov_y_from_intrinsics(self) -> float:
        """Vertical field of view [deg] implied by :attr:`intrinsics`.

        The GL viewer goes through this, so an explicitly-calibrated camera
        degrades to the closest field of view there rather than being ignored.
        It is the only backend that has to: every other one is handed the whole
        ``K``, principal point included.
        """
        k = self.intrinsics
        return float(math.degrees(2.0 * math.atan(0.5 * self.height / k[1, 1])))

    def scaled(self, width: int, height: int) -> CameraSpec:
        """This camera at another resolution, with the intrinsics scaled to match.

        The field of view is unchanged, which is what a preview render or a
        cheap collection pass wants: the same picture, fewer pixels.
        """
        k = self.intrinsics
        sx, sy = width / self.width, height / self.height
        return replace(
            self,
            width=int(width),
            height=int(height),
            fx=float(k[0, 0] * sx),
            fy=float(k[1, 1] * sy),
            cx=float(k[0, 2] * sx),
            cy=float(k[1, 2] * sy),
        )

    # ------------------------------------------------------------------- pose
    @property
    def world_matrix(self) -> np.ndarray:
        """Camera-to-world 4x4 in the +z-forward, +y-down basis.

        For a camera parented to a body this is the *local* pose; the caller
        composes it with that body's transform through :meth:`pose_on_body`.
        """
        if self.xform is not None:
            return transform_matrix(self.xform)
        return look_at_matrix(self.eye, self.look_at, self.up)

    @property
    def world_xform(self) -> np.ndarray:
        """The same pose as a ``(px, py, pz, qx, qy, qz, qw)`` row."""
        return xform_from_matrix(self.world_matrix)

    def pose_on_body(self, body_q: np.ndarray | None, *, body_offset: int = 0) -> np.ndarray:
        """This camera's world 4x4 given the step's body transforms.

        Args:
            body_q: ``State.body_q`` read back to the host, or ``None`` for a
                world-fixed camera.
            body_offset: Added to :attr:`parent_body` before the lookup, so a
                replicated model's world *w* answers with its own arm rather
                than world 0's.

                A world-fixed camera ignores it, and that is correct rather than
                convenient: every world of a wave is superimposed at the origin,
                so a camera bolted to the room sees the same place in all of
                them. A camera bolted to a *wrist* does not, and broadcasting
                world 0's wrist pose to every world is a rig that photographs
                one robot and labels the picture with sixteen episode ids.
        """
        local = self.world_matrix
        if self.parent_body < 0 or body_q is None:
            return local
        return transform_matrix(body_q[self.parent_body + int(body_offset)]) @ local

    def project(
        self, points: np.ndarray, *, camera_to_world: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """World points [m] to this camera's pixels: ``(N, 2)`` ``(u, v)`` and ``(N,)`` depth along the view [m].

        The same mapping a render applies to a point, through :attr:`intrinsics`
        and the pose, so a point read off the scene lands on the pixel the
        render drew it at. A point at or behind the camera has a depth ``<= 0``
        and a pixel that means nothing. ``camera_to_world`` is the pose to
        project from when it is not :attr:`world_matrix` -- a camera on a body,
        posed by :meth:`pose_on_body`.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pose = self.world_matrix if camera_to_world is None else np.asarray(camera_to_world, dtype=np.float64)
        local = (pts - pose[:3, 3]) @ pose[:3, :3]
        depth = local[:, 2]
        k = self.intrinsics
        with np.errstate(divide="ignore", invalid="ignore"):
            u = k[0, 0] * local[:, 0] / depth + k[0, 2]
            v = k[1, 1] * local[:, 1] / depth + k[1, 2]
        return np.stack([u, v], axis=1), depth

    def with_pose(self, eye: Sequence[float], look_at: Sequence[float]) -> CameraSpec:
        """This camera moved to another viewpoint."""
        return replace(
            self,
            eye=tuple(float(v) for v in eye),
            look_at=tuple(float(v) for v in look_at),
            xform=None,
        )
