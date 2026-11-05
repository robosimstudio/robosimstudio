# SPDX-License-Identifier: Apache-2.0
"""The renderer contract every backend implements.

One method, one return shape::

    renderer.render(state) -> {camera_name: {"rgb": u8 [H, W, 3],
                                             "depth": f32 [H, W],
                                             "seg": i32 [H, W]}}

Only the channels a :class:`~robosimstudio.core.cameras.CameraSpec` asked for
are present. Depth is metres to the image plane with **0 meaning a miss**, not
an infinity -- every consumer downstream (a VLA adapter, a point-cloud node, an
mp4 writer) has to handle misses, and zero is the one value they all handle the
same way.

Two backends live behind this: the OVRTX path tracer, which every picture is
made with, and a tiled ray tracer that renders the colliders (and cloth,
natively from ``particle_q``) where the tracer is not installed. They agree on
this contract and on nothing else, which is the point -- a scene rendered two
ways that disagrees is telling you something.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import newton
import numpy as np

from ..cameras import CameraSpec

__all__ = [
    "Frame",
    "FrameSet",
    "RendererBase",
]

Frame = dict[str, np.ndarray]
"""One camera's channels for one step."""
FrameSet = dict[str, Frame]
"""Every camera's frame for one step, keyed by camera name."""


class RendererBase:
    """Base class: holds the cameras, and reports what it can produce.

    Args:
        cameras: The cameras to render, in the order a backend that batches
            them should tile them.
        channels: Restrict every camera to these channels. ``None`` honours
            each camera's own :attr:`CameraSpec.channels`.
    """

    name: str = "base"
    """Short backend name, as it appears in run metadata."""
    supported_channels: tuple[str, ...] = ("rgb",)

    devices: tuple[str, ...] = ()
    """The Warp device kinds this backend renders on: ``"cuda"``, ``"cpu"``, or both.

    Empty means unknown and is never refused. Read off the class by
    :func:`~robosimstudio.solvers.render.choose_renderer` before anything is
    built, so a renderer named on a device it cannot use fails by name instead
    of inside a GL or CUDA context."""

    wants: tuple[str, ...] = ()
    """Scene facts this backend takes, by name, or ``()`` for none.

    A renderer cannot reach the compiled scene -- it is handed a model and a
    camera list -- so anything it needs *about* the scene (the segmentation
    label table, which shapes recolour while the scene runs, the visual parts a
    mesh rasteriser draws) has to be passed in. Declaring the names here is how
    a backend asks: :meth:`robosimstudio.sim.Sim.make_renderer` knows how to
    produce each scene fact and supplies exactly the ones a class lists.

    Stated rather than branched on, because the branch was a closed ``if
    backend == "rtx"`` chain over the five built-in names and a sixth backend --
    registered through the documented ``register_renderer`` door, or named by a
    dotted path -- received nothing from it.
    """

    @classmethod
    def look_kwargs(cls, look: Any) -> dict[str, Any]:
        """What this backend takes from a :class:`~robosimstudio.core.render.spec.RenderSpec`.

        A *look* is one statement about a scene -- where the light comes from,
        how bright the fill is, what the background is -- and each backend takes
        the part of it that it can honour, in its own units. Answering here
        rather than in a table keyed by backend name is what lets a renderer
        nobody in this tree has heard of receive a look at all.

        The default is ``{}``: a backend that says nothing is handed nothing,
        which is the honest answer.
        """
        del look
        return {}

    retargets: bool = False
    """Whether the backend reads its cameras afresh every frame, so :meth:`aim` may replace them on a built
    renderer -- what lets a caller keep one renderer across viewpoints rather than build one per camera, which
    for a backend that starts a process and hands it the scene (``blender``) is seconds each time."""

    def __init__(self, cameras: Sequence[CameraSpec], *, channels: Sequence[str] | None = None) -> None:
        self._channel_override = tuple(channels) if channels is not None else None
        self._take_cameras(cameras)

    first_note: str = ""
    """What the first picture in a process costs with this backend, when that is worth saying while it is drawn
    (the path tracer compiles its shaders, Blender starts); a page's ``status`` shows it."""

    first_note_each_scene: bool = False
    """Whether :attr:`first_note`'s cost comes again with every scene built in the process (a Blender per
    scene), rather than once a process (shaders compiled once)."""

    warms_up: bool = False
    """Whether a new renderer's first frame only builds its stage and is drawn again before it is kept."""

    def aim(self, cameras: Sequence[CameraSpec], channels: Sequence[str] | None = None) -> None:
        """Render these cameras from the next frame on.

        Args:
            cameras: The cameras.
            channels: Restrict every camera to these; ``None`` keeps the restriction the renderer was built with.

        Raises:
            NotImplementedError: If the backend builds its cameras in (:attr:`retargets` is false).
        """
        if not self.retargets:
            raise NotImplementedError(f"renderer {self.name!r} builds its cameras in; build another for these")
        kept = getattr(self, "_channel_override", None)
        self._channel_override = tuple(channels) if channels is not None else kept
        try:
            self._take_cameras(cameras)
        except ValueError:
            self._channel_override = kept
            raise

    def _take_cameras(self, cameras: Sequence[CameraSpec]) -> None:
        if not cameras:
            raise ValueError("a renderer needs at least one camera")
        override = getattr(self, "_channel_override", None)
        wanted_by: dict[str, tuple[str, ...]] = {}
        for cam in cameras:
            wanted = override if override is not None else cam.channels
            missing = set(wanted) - set(self.supported_channels)
            if missing:
                raise ValueError(
                    f"renderer {self.name!r} cannot produce {sorted(missing)} for camera {cam.name!r}; "
                    f"it supports {list(self.supported_channels)}"
                )
            wanted_by[cam.name] = tuple(wanted)
        self.cameras = list(cameras)
        self.channels: dict[str, tuple[str, ...]] = wanted_by

    def __getitem__(self, name: str) -> CameraSpec:
        for cam in self.cameras:
            if cam.name == name:
                return cam
        raise KeyError(f"no camera named {name!r}; have {[c.name for c in self.cameras]}")

    def intrinsics(self, name: str) -> np.ndarray:
        """Pinhole ``K`` of one camera, as the backend actually rendered it.

        A backend that can only approximate the requested intrinsics (a GL
        viewer taking a field of view, say) overrides this to report what it
        really used, so a recorded frame's ``K`` is never a lie.
        """
        return self[name].intrinsics

    def extrinsics(self, name: str, body_q: np.ndarray | None = None) -> np.ndarray:
        """Camera-to-world 4x4 of one camera, in the +z-forward vision basis."""
        return self[name].pose_on_body(body_q)

    def render(self, state: newton.State) -> FrameSet:
        """Render every camera for one state."""
        raise NotImplementedError

    def close(self) -> None:
        """Release GL contexts, USD stages and the like. Idempotent."""
