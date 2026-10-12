# SPDX-License-Identifier: Apache-2.0
"""The Genesis backend, with Genesis not installed.

Everything here runs with the ``[genesis]`` extra absent, on purpose --- it is
the half of the story ``test_genesis.py`` cannot tell, and the half that matters
most for a checkout that will never build a Genesis island. The claim being
checked is the extension seam's: registering a backend imports nothing, an alias
resolves from a cold process, and the engine is asked for only when a scene
actually stages one --- not at import, and never as a silent fallback.

The arithmetic that decides whether the coupling is *correct* is also here,
because none of it needs an engine: the mass reconciliation, the quaternion
convention, and which mirrors may carry a reaction at all.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest

import numpy as np

TIER = "fast"
"""No GPU, no engine, no scene."""


def _geom(**kw):
    from robosimstudio.engines.geometry import ShapeGeometry

    base = {
        "shape": 0,
        "body": 3,
        "kind": "box",
        "scale": np.ones(3),
        "local": np.array([0.0, 0, 0, 0, 0, 0, 1.0]),
        "mu": 0.5,
        "two_way": False,
    }
    return ShapeGeometry(**{**base, **kw})


class TestRegistrationImportsNothing(unittest.TestCase):
    def test_the_alias_resolves_and_is_a_host_engine(self):
        from robosimstudio.specs.physics import get_backend

        info = get_backend("genesis")
        self.assertTrue(info.host_engine)
        self.assertEqual(info.supports, ("particles",))
        self.assertEqual(info.cls, "robosimstudio.engines.genesis.solver.SolverGenesis")

    def test_it_resolves_from_a_cold_process(self):
        """``GENERIC`` may name an alias nothing has imported; ``get_backend``
        scans the engine bundles once and retries before giving up."""
        code = "from robosimstudio.specs.physics import get_backend\nassert get_backend('genesis').host_engine\n"
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_importing_the_bundle_does_not_import_the_engine(self):
        """The registration is a dotted-path string, so it costs no engine."""
        code = (
            "import sys, robosimstudio.engines.genesis as bundle\n"
            # Was `'genesis' not in sys.modules or sys.modules['genesis'] is not
            # None`, which is true whether or not the engine was imported. The
            # generic form of this law is TestNoBundleImportsItsEngine in
            # tests/test_engine_roots.py, over every bundle and every SDK.
            "assert 'genesis' not in sys.modules, 'the bundle imported the engine at module scope'\n"
            "assert bundle.BACKEND.host_engine\n"
            "import robosimstudio.engines.genesis.bridge  # the adapter too\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    @unittest.skipIf(importlib.util.find_spec("genesis") is not None, "genesis is installed here")
    def test_a_missing_wheel_is_refused_by_name(self):
        from robosimstudio.engines.genesis.runtime import ensure_initialized

        with self.assertRaises(ImportError) as caught:
            ensure_initialized()
        message = str(caught.exception)
        self.assertIn("[genesis]", message, "the refusal names the extra to install")
        self.assertIn("genesis-world", message)


class TestTheCouplingArithmetic(unittest.TestCase):
    """The numbers that decide whether the two-way coupling is right."""

    def test_cloth_density_divides_out_the_measured_factor(self):
        """Genesis's FEM.Cloth masses twice what rho*thickness*area predicts.

        Measured across a doubled density, a doubled thickness and a halved mesh
        spacing, constant to three figures. Unreconciled, the wrench fed back to
        the arm is twice what it should be -- and it reads as "the cloth feels
        heavy" rather than as an error, which is why this is a test and not a
        comment.
        """
        from robosimstudio.engines.genesis.bridge import CLOTH_MASS_FACTOR, cloth_density

        self.assertEqual(CLOTH_MASS_FACTOR, 2.0)
        areal, thickness = 0.4, 0.002
        rho = cloth_density(areal, thickness)
        simulated_areal = rho * thickness * CLOTH_MASS_FACTOR
        self.assertAlmostEqual(simulated_areal, areal, places=9, msg="the sheet weighs what Newton says")

    def test_cloth_density_survives_a_zero_thickness(self):
        from robosimstudio.engines.genesis.bridge import cloth_density

        self.assertGreater(cloth_density(0.4, 0.0), 0.0)

    def test_the_quaternion_convention_flips_at_this_boundary(self):
        """Newton is xyzw and Genesis is wxyz. A leaked swap is a plausible
        wrong rotation rather than an error, so it is checked here."""
        from robosimstudio.engines.genesis.bridge import wxyz

        np.testing.assert_allclose(wxyz([0.0, 0.0, 0.0, 1.0]), [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(wxyz([0.1, 0.2, 0.3, 0.4]), [0.4, 0.1, 0.2, 0.3])

    def test_only_a_free_two_way_mirror_can_carry_a_reaction(self):
        """A fixed link never deviates from its aim, so its reaction is
        identically zero -- measured, with the cloth visibly resting on it. A
        two-way mirror must therefore be free-based and gravity-compensated."""
        from robosimstudio.engines.genesis.bridge import mirror_role

        coup, fixed, gravcomp = mirror_role(_geom(two_way=True, body=3))
        self.assertEqual(coup, "two_way_soft_constraint")
        self.assertFalse(fixed, "a fixed mirror reads zero force")
        self.assertEqual(gravcomp, 1.0, "or it sags between prescriptions and reads a phantom load")

    def test_everything_else_is_a_fixed_kinematic_collider(self):
        from robosimstudio.engines.genesis.bridge import mirror_role

        for geom in (_geom(two_way=False, body=3), _geom(two_way=True, body=-1)):
            coup, fixed, gravcomp = mirror_role(geom)
            self.assertEqual(coup, "ipc_only")
            self.assertTrue(fixed)
            self.assertEqual(gravcomp, 0.0, "Genesis refuses gravity_compensation with ipc_only")

    def test_the_tet_surface_is_the_faces_used_once(self):
        """A tet body reaches Genesis as a surface plus its own lattice; the
        surface has to be the boundary, not every face."""
        from robosimstudio.engines.genesis.bridge import _surface_of

        one = np.array([[0, 1, 2, 3]])
        self.assertEqual(len(_surface_of(one)), 4, "a single tet is all boundary")
        two = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        self.assertEqual(len(_surface_of(two)), 6, "two tets share one face, so 8 - 2")


class TestTheSolverIsAHostSolver(unittest.TestCase):
    def test_it_inherits_the_shared_contract(self):
        from robosimstudio.engines.genesis.solver import SolverGenesis
        from robosimstudio.engines.host import SolverHost

        self.assertTrue(issubclass(SolverGenesis, SolverHost))
        self.assertTrue(SolverGenesis.host_engine)
        for name in (
            "step",
            "notify_model_changed",
            "update_contacts",
            "reset",
            "attach_pipeline",
            "capture_rest",
            "host_tick",
        ):
            self.assertTrue(hasattr(SolverGenesis, name), f"the composer calls {name}")

    def test_open_is_the_only_thing_it_writes(self):
        """The point of SolverHost: a second engine costs one method."""
        from robosimstudio.engines.genesis.solver import SolverGenesis

        own = {n for n, v in vars(SolverGenesis).items() if callable(v)}
        self.assertEqual(own, {"open"}, f"SolverGenesis defines more than open(): {sorted(own)}")


if __name__ == "__main__":
    unittest.main()
