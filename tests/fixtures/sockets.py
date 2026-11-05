# SPDX-License-Identifier: Apache-2.0
"""Directories short enough for a Unix socket, wherever the checkout and the temp dir are.

A socket's path must fit ``sockaddr_un`` -- 108 bytes on Linux. A run directory
under a deep checkout, or ``tempfile.mkdtemp()`` under a session's long
``TMPDIR``, passes that limit, and fifteen fast tests failed on a machine where
the temp dir was deep while CI, with a short one, stayed green. :func:`short_dir`
makes its directory under ``/tmp`` itself, never under ``TMPDIR``.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import shutil
import tempfile
from collections.abc import Iterator

__all__ = ["chdir", "short_dir"]

ROOT = "/tmp" if os.path.isdir("/tmp") else None
"""Where :func:`short_dir` makes its directories: ``/tmp`` where there is one, else the temp dir."""


@contextlib.contextmanager
def short_dir(prefix: str = "rss-") -> Iterator[pathlib.Path]:
    """A fresh directory with a short path, removed on exit."""
    path = pathlib.Path(tempfile.mkdtemp(prefix=prefix, dir=ROOT))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@contextlib.contextmanager
def chdir(path: os.PathLike | str) -> Iterator[pathlib.Path]:
    """Work in ``path`` for the block and come back, even when the block raises."""
    with contextlib.chdir(path):
        yield pathlib.Path(path)
