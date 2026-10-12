# SPDX-License-Identifier: Apache-2.0
"""Streaming mp4 writers, one per camera.

A recorder that buffers frames until close costs memory proportional to the
episode and loses everything if the run dies, so :class:`VideoWriter` opens the
file on the first frame and appends. ``imageio`` is imported inside the
constructor, which keeps a physics-only run from needing it at all.

Depth and segmentation are written as colourised previews, never as the data: an
mp4 is 8-bit and lossy, so those videos are for looking at. The metric array and
the label ids belong in the episode's ``npz``. The two colourisers live here
because a live view in a browser wants exactly the same conversion -- a channel
that is not ``rgb`` has to become ``rgb`` before anything can display it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

__all__ = ["VideoWriter", "colorize_depth", "colorize_labels", "write_video"]

# Golden-ratio conjugate: successive multiples of it are maximally spread over
# the unit interval, which is what keeps consecutive label ids far apart in hue.
_HUE_STEP = 0.6180339887498949


def colorize_depth(depth: np.ndarray, *, near: float = 0.0, far: float = 0.0) -> np.ndarray:
    """Depth as a viewable 8-bit grey image, misses left black.

    Args:
        depth: Metric depth [m], 0 for a miss.
        near: Low end of the mapped range [m]. 0 uses the frame's own minimum
            over the hits.
        far: High end [m]. 0 uses the frame's own maximum.
    """
    d = np.asarray(depth, dtype=np.float32)
    hit = d > 0
    if not hit.any():
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    lo = float(near) if near > 0 else float(d[hit].min())
    hi = float(far) if far > 0 else float(d[hit].max())
    span = max(hi - lo, 1.0e-6)
    grey = np.clip((d - lo) / span, 0.0, 1.0)
    grey = np.where(hit, 1.0 - grey, 0.0)  # near is bright, misses black
    return np.repeat((grey * 255).astype(np.uint8)[..., None], 3, axis=2)


def colorize_labels(labels: np.ndarray, *, background: int = 0) -> np.ndarray:
    """Label ids as distinguishable 8-bit colours, ``background`` left black.

    A hue walk rather than a fixed palette, because the ids are the scene's to
    assign: a task labels as many things as it has, so a colouriser cannot know
    how many colours it will be asked for, and *adjacent* ids have to land far
    apart -- consecutive integers are exactly how a scene labels two objects
    lying against each other, and a palette that walks hue linearly renders that
    pair as two shades of the same colour.

    Args:
        labels: Integer label ids, any shape.
        background: The id meaning "nothing here", rendered black.

    Returns:
        ``uint8`` RGB, shape ``(*labels.shape, 3)``.
    """
    ids = np.asarray(labels).astype(np.int64, copy=False)
    unique = np.unique(ids)
    colors = _hues_to_rgb(np.mod(unique.astype(np.float64) * _HUE_STEP, 1.0))
    colors[unique == int(background)] = 0
    # unique is sorted, so searchsorted is the lookup -- one pass over the image
    # rather than a Python loop over however many labels the scene declared.
    return colors[np.searchsorted(unique, ids)]


def _hues_to_rgb(hue: np.ndarray, *, saturation: float = 0.65, value: float = 1.0) -> np.ndarray:
    """HSV to ``uint8`` RGB for a 1-D array of hues in [0, 1)."""
    h = np.mod(np.asarray(hue, dtype=np.float64), 1.0) * 6.0
    sector = np.floor(h).astype(np.int64) % 6
    f = h - np.floor(h)
    v = np.full(h.shape, float(value))
    p = v * (1.0 - float(saturation))
    q = v * (1.0 - float(saturation) * f)
    t = v * (1.0 - float(saturation) * (1.0 - f))
    conditions = [sector == i for i in range(6)]
    r = np.select(conditions, [v, q, p, p, t, v])
    g = np.select(conditions, [t, v, v, q, p, p])
    b = np.select(conditions, [p, p, t, v, v, q])
    return (np.stack([r, g, b], axis=-1) * 255.0).round().astype(np.uint8)


class VideoWriter:
    """One mp4, appended a frame at a time.

    Args:
        path: Output file. Parent directories are created.
        fps: Frame rate stamped into the container.
        quality: imageio quality, 1..10.
    """

    def __init__(self, path: str | Path, *, fps: int = 30, quality: int = 7) -> None:
        import imageio.v2 as imageio

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.frames = 0
        # macro_block_size=1 keeps odd resolutions from being silently resized,
        # which would break the intrinsics recorded alongside the video.
        self._writer = imageio.get_writer(str(self.path), fps=self.fps, quality=quality, macro_block_size=1)

    def append(self, frame: np.ndarray) -> None:
        """Append one RGB uint8 frame, top-left origin."""
        self._writer.append_data(np.ascontiguousarray(np.asarray(frame, dtype=np.uint8)))
        self.frames += 1

    def close(self) -> None:
        """Finish the container. Idempotent."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class VideoSet:
    """A :class:`VideoWriter` per camera, opened lazily on first use.

    Args:
        root: Directory the files are written under.
        fps: Frame rate.
        suffix: Appended to each camera name, so one run can hold both
            ``wrist.mp4`` and ``wrist_depth.mp4``.
    """

    def __init__(self, root: str | Path, *, fps: int = 30, suffix: str = "") -> None:
        self.root = Path(root)
        self.fps = int(fps)
        self.suffix = suffix
        self.writers: dict[str, VideoWriter] = {}

    def append(self, name: str, frame: np.ndarray) -> None:
        writer = self.writers.get(name)
        if writer is None:
            writer = VideoWriter(self.root / f"{name}{self.suffix}.mp4", fps=self.fps)
            self.writers[name] = writer
        writer.append(frame)

    def append_frames(self, frames: Mapping[str, Mapping[str, np.ndarray]], *, depth: bool = False) -> None:
        """Append the ``rgb`` channel of every camera in a render result.

        Args:
            frames: A :data:`~robosimstudio.core.render.base.FrameSet`.
            depth: Also write a colourised depth preview per camera that has
                one, as ``<name>_depth.mp4``.
        """
        for name, frame in frames.items():
            if "rgb" in frame:
                self.append(name, frame["rgb"])
            if depth and "depth" in frame:
                self.append(f"{name}_depth", colorize_depth(frame["depth"]))

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers = {}

    def __enter__(self) -> VideoSet:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_video(path: str | Path, frames, *, fps: int = 30) -> str:
    """Encode a whole sequence at once. Returns the path written."""
    with VideoWriter(path, fps=fps) as writer:
        for frame in frames:
            writer.append(frame)
    return str(path)
