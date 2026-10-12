# SPDX-License-Identifier: Apache-2.0
"""Renderers, the frame contract they share, and the video writers.

Only the base module and the backend-free pieces are imported eagerly. The
path tracer pulls in ``ovrtx``, an optional extra, so renderers are reached
through the factory one layer up or by importing the backend module directly.
"""

from .base import Frame, FrameSet, RendererBase
from .finish import Finish, finish_for, register_texture, texture_path
from .live import RenderBudget
from .offline import OfflineRenderer, StateReplay, steps_from_arrays
from .spec import RenderSpec
from .video import VideoSet, VideoWriter, colorize_depth, colorize_labels, write_video
from .visual import ClothPart, VisualPart, cloth_triangles, flatten_scene

__all__ = [
    "ClothPart",
    "Finish",
    "Frame",
    "FrameSet",
    "OfflineRenderer",
    "RenderBudget",
    "RenderSpec",
    "RendererBase",
    "StateReplay",
    "VideoSet",
    "VideoWriter",
    "VisualPart",
    "cloth_triangles",
    "colorize_depth",
    "colorize_labels",
    "finish_for",
    "flatten_scene",
    "register_texture",
    "steps_from_arrays",
    "texture_path",
    "write_video",
]
