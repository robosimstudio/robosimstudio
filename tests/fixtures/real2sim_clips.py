# SPDX-License-Identifier: Apache-2.0
"""Clips drawn for a test: a reference video whose every frame the test knows.

The stills, the timeline and ``ctl ref frames`` all start from a clip. A clip of
one flat colour is the smallest one ffmpeg will read back with a real header,
so a test can ask what a still cut at any time looks like and know the answer.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

__all__ = ["flat_clip"]


def flat_clip(path: Path, seconds: float, colour: str, fmt: str = "mp4") -> Path:
    """A clip of one flat colour, at 10 fps; ``fmt="h264"`` writes a raw stream with no duration in its header."""
    import imageio_ffmpeg

    argv = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi"]
    argv += ["-i", f"color=c={colour}:size=64x48:rate=10:duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if fmt == "h264":
        argv += ["-f", "h264"]
    subprocess.run([*argv, str(path)], check=True, stdin=subprocess.DEVNULL)
    return path
