# SPDX-License-Identifier: Apache-2.0
"""Soft body from a file

A rubber O-ring read out of a tetrahedral mesh file and dropped between two arms: click it and drag.

The script writes the mesh itself so you can see the format: an ``.npz`` holding
``vertices`` and ``tets``. Anything ``meshio`` reads that has tetrahedra in it
works too, so point ``AssetSpec(path=...)`` at your own file and delete the
writer. Wind every tetrahedron so its rest volume is positive: Newton drops the
others without a word. The ring runs on Newton's VBD here, on a base install.

**Key features:**

* :class:`robosimstudio.SoftSpec` with :attr:`robosimstudio.SoftSpec.asset` for a body from a file
* :class:`robosimstudio.AssetSpec` to name the file
* :func:`robosimstudio.replace` on ``rss.GENERIC`` to keep soft bodies on Newton
* :func:`robosimstudio.view` to take hold of the result
"""

import tempfile
from pathlib import Path

import numpy as np

import robosimstudio as rss

HEX_TETS = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6]], dtype=np.int64)
"""One hexahedron as five tetrahedra: ring ``k``'s four corners, then ring ``k + 1``'s."""


def write_o_ring(path: Path, *, major: float, minor: float, around: int) -> None:
    """Stand-in for your exporter: a square section swept round a closed circle."""
    angle = np.linspace(0.0, 2.0 * np.pi, around, endpoint=False)
    radial = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1)
    up = np.array([0.0, 0.0, 1.0])
    # Four corners per ring, in the order HEX_TETS expects.
    corners = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
    vertices = np.concatenate([[(major + minor * r) * rad + minor * z * up for r, z in corners] for rad in radial])
    # Local corners 0-3 are ring k and 4-7 are ring k + 1, which wraps: a loop, not a bar.
    tets = np.concatenate(
        [np.where(HEX_TETS < 4, HEX_TETS + 4 * k, (HEX_TETS - 4) + 4 * ((k + 1) % around)) for k in range(around)]
    )
    np.savez(path, vertices=vertices, tets=tets)


def main():
    cell = "void"
    embodiment = "yam_bimanual"
    major = 0.045
    minor = 0.011

    # ``GENERIC`` routes soft bodies to SuperDex, which is an extra. Without that
    # entry the mesh falls through to VBD and is an ordinary Newton deformable.
    physics = rss.replace(rss.GENERIC, backends=tuple(e for e in rss.GENERIC.backends if e[0] != "soft"))

    scene = rss.Scene(cell=cell, physics=physics, name="tet_mesh")
    scene.add(rss.robot(embodiment))
    with tempfile.TemporaryDirectory() as scratch:
        mesh = Path(scratch) / "o_ring.npz"
        write_o_ring(mesh, major=major, minor=minor, around=28)
        # The line this example is about. Yours names a file you already have.
        scene.add(rss.SoftSpec(name="gasket", asset=rss.AssetSpec(path=str(mesh)), pose=(0.0, 0.0, 0.09)))
        sim = scene.build(control_mode="joint_position")

    rss.view(sim)


if __name__ == "__main__":
    main()
