# SPDX-License-Identifier: Apache-2.0
"""A stand-in for a Warp array, for a fast test that hands a measure host data.

Thirty test modules each grew their own: ``SimpleNamespace(numpy=lambda: q)``,
``SimpleNamespace(numpy=q.copy)``, a ``_Arr`` class, an ``_Array`` class. They
differ on the one point that matters -- whether what ``numpy()`` returns is the
array or a copy of it -- and a test that writes through a view it was handed
passes against one stand-in and fails against the other. :func:`host` says
which, by name.

Warp's own rule: a CPU array's ``numpy()`` shares its memory, a CUDA array's is
a copy. ``copy=False`` is the CPU behaviour and the default, and ``device`` says
``"cpu"``; ``copy=True`` is the device one, ``device`` says ``"cuda:0"``, for a
test of code that must write back with ``assign``.

An array handed in keeps its dtype and, with ``copy=False``, its memory: a test
that edits the array it passed sees the edit through ``numpy()``, as it did with
``SimpleNamespace(numpy=lambda: q)``. A plain list becomes float64.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["HostArray", "host"]


class HostArray:
    """``numpy()``, ``shape``, ``dtype``, ``len()``, ``assign()`` and ``fill_()`` over a numpy array."""

    def __init__(self, values: Any, dtype: Any = None, *, copy: bool = False) -> None:
        if dtype is None and not isinstance(values, np.ndarray):
            dtype = float
        self.data = np.asarray(values, dtype=dtype)
        self.copy = copy
        self.device = "cuda:0" if copy else "cpu"

    def numpy(self) -> np.ndarray:
        return self.data.copy() if self.copy else self.data

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __len__(self) -> int:
        return len(self.data)

    def assign(self, values: Any) -> None:
        self.data[...] = np.asarray(values, dtype=self.data.dtype)

    def fill_(self, value: Any) -> None:
        self.data.fill(value)


def host(values: Any, dtype: Any = None, *, copy: bool = False) -> HostArray:
    """``values`` as a :class:`HostArray`; ``copy=True`` makes ``numpy()`` hand out copies, as on a GPU.

    ``dtype=None`` keeps an ndarray's dtype (an index array stays integer, a mask
    stays bool) and makes a list float64.
    """
    return HostArray(values, dtype, copy=copy)
