# SPDX-License-Identifier: Apache-2.0
"""Regressions for the soft examples that need no GPU: the tet ring and which class flags a body."""

from __future__ import annotations

import collections
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

TIER = "fast"

ROOT = Path(__file__).resolve().parents[1]


def _tet_mesh_module():
    spec = importlib.util.spec_from_file_location("tet_mesh_example", ROOT / "examples/03_soft/01_tet_mesh.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheORingIsConforming(unittest.TestCase):
    """The same 5-tet split in every segment left 112 interior faces looking like surface."""

    def setUp(self):
        self.module = _tet_mesh_module()

    def _ring(self, around):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "ring.npz"
            self.module.write_o_ring(path, major=0.045, minor=0.011, around=around)
            data = np.load(path)
            return data["vertices"], data["tets"]

    def test_only_the_sides_are_surface(self):
        vertices, tets = self._ring(28)
        faces = collections.Counter(
            tuple(sorted(int(i) for i in face))
            for a, b, c, d in tets
            for face in ((a, b, c), (a, b, d), (a, c, d), (b, c, d))
        )
        # Four side quads a segment, two triangles a quad.
        self.assertEqual(sum(1 for n in faces.values() if n == 1), 28 * 4 * 2)
        self.assertTrue(all(n in (1, 2) for n in faces.values()))
        a, b, c, d = (vertices[tets[:, i]] for i in range(4))
        volumes = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)
        self.assertTrue(np.all(volumes > 0.0), "every tet keeps a positive rest volume")

    def test_an_odd_loop_is_refused(self):
        with self.assertRaises(ValueError):
            self._ring(27)


class TestOnlyAHostedBodyIsInactiveToNewton(unittest.TestCase):
    """EntitySoft cleared ACTIVE whatever the backend; the class the arrangement picks decides now."""

    def _flags(self, solver):
        import newton
        from robosimstudio.domains.soft import SoftSpec

        from robosimstudio.scene.registry import resolve_entity_class
        from robosimstudio.specs.arrangement import ArrangementSpec, backend

        spec = SoftSpec(name="b", size=(0.04, 0.02, 0.01), cells=(4, 2, 2), settle_frames=0)
        cls = resolve_entity_class(
            spec, ArrangementSpec(control_hz=60.0, substeps=4, backends=(("soft", backend(solver)),))
        )
        builder = newton.ModelBuilder()
        cls(spec).build(builder, None)
        return cls, np.asarray(builder.particle_flags) & int(newton.ParticleFlags.ACTIVE)

    def test_vbd_gets_an_active_body(self):
        cls, active = self._flags("vbd")
        self.assertEqual(cls.__name__, "EntitySoft")
        self.assertTrue(np.all(active))

    def test_superdex_gets_an_inactive_one(self):
        cls, active = self._flags("superdex")
        self.assertEqual(cls.__name__, "EntitySoftSuperdex")
        self.assertFalse(np.any(active))


if __name__ == "__main__":
    unittest.main()
