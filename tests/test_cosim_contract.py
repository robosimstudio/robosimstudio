# SPDX-License-Identifier: Apache-2.0
"""Every shipped host engine against the one contract, with no engine installed.

The seam is duck-typed and stays that way --- the composer finds a host solver by
``getattr`` and nothing inherits from ``CosimBridge``. What this module does is
make the contract *checkable*: a method renamed on one engine and not the other
fails here, in the fast tier, instead of on a GPU box with a wheel installed.

**Parameterised over the registry rather than over a list**, so a third engine is
covered the day it registers and cannot be added without being checked. The point
of a second engine was to find out what the seam actually requires; the point of
this file is that the answer stops drifting.
"""

from __future__ import annotations

import importlib
import inspect
import unittest

TIER = "fast"
"""Registrations and class shapes. No GPU, no engine, no scene."""

BRIDGE_SURFACE = ("tick", "attach_pipeline", "capture_rest", "restore_rest")
"""What ``SolverHost`` calls on whatever ``open`` returns."""

SOLVER_SURFACE = ("step", "notify_model_changed", "update_contacts", "reset", "host_tick")
"""What the composer calls on a host solver."""


def _host_backends():
    """Every registered host engine, by alias. Loads the bundles once."""
    from robosimstudio.specs.physics import BACKENDS, ENGINES

    ENGINES.load()
    return {alias: info for alias, info in BACKENDS.items() if info.host_engine}


def _resolve(path: str):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


class TestEveryHostEngineIsRegisteredTheSameWay(unittest.TestCase):
    def test_there_are_at_least_two(self):
        """A suite that found one engine would pass everything below vacuously,
        and one engine is exactly the state this contract was extracted from."""
        self.assertGreaterEqual(len(_host_backends()), 2, sorted(_host_backends()))

    def test_each_lives_in_its_own_bundle_under_engines(self):
        """Not under a domain. An engine that also serves a cloth belongs to no
        domain, and one hanging off whichever needed it first is a registry with
        the wrong shape -- which is what this stack had, and moved."""
        for alias, info in _host_backends().items():
            with self.subTest(engine=alias):
                self.assertTrue(
                    info.cls.startswith(f"robosimstudio.engines.{alias}."),
                    f"{alias} resolves to {info.cls}",
                )

    def test_each_declares_the_particles_it_serves(self):
        for alias, info in _host_backends().items():
            with self.subTest(engine=alias):
                self.assertIn("particles", info.supports)


class TestEverySolverSatisfiesTheHostContract(unittest.TestCase):
    def test_each_is_a_solver_host(self):
        from robosimstudio.engines.host import SolverHost

        for alias, info in _host_backends().items():
            with self.subTest(engine=alias):
                self.assertTrue(issubclass(_resolve(info.cls), SolverHost))

    def test_each_answers_everything_the_composer_calls(self):
        for alias, info in _host_backends().items():
            cls = _resolve(info.cls)
            for name in SOLVER_SURFACE:
                with self.subTest(engine=alias, method=name):
                    self.assertTrue(callable(getattr(cls, name, None)))

    def test_open_is_the_only_thing_an_engine_writes(self):
        """The whole return on ``SolverHost``: a second engine costs one method.

        If an engine defines more, either the shared layer is missing something
        that engine needs -- which belongs in ``engines/``, not in one bundle --
        or the contract has grown a second reading.
        """
        for alias, info in _host_backends().items():
            with self.subTest(engine=alias):
                own = {n for n, v in vars(_resolve(info.cls)).items() if callable(v)}
                self.assertEqual(own, {"open"}, f"{alias} defines {sorted(own)}")

    def test_the_model_is_positional_and_the_rest_is_not(self):
        """``compose.py`` builds a host solver as ``cls(model, recipe=…, substeps=…)``,
        against ``cls(model=view, **kwargs)`` for a Newton one. The difference is
        deliberate -- a host solver takes the whole model, never a view -- so it
        is asserted rather than left to a TypeError at compile time."""
        from robosimstudio.engines.host import SolverHost

        params = list(inspect.signature(SolverHost.__init__).parameters.values())
        self.assertEqual(params[1].name, "model")
        self.assertIs(params[1].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in ("recipe", "substeps"):
            with self.subTest(param=name):
                self.assertIs(
                    inspect.signature(SolverHost.__init__).parameters[name].kind,
                    inspect.Parameter.KEYWORD_ONLY,
                )


class TestEveryBridgeSatisfiesTheBridgeContract(unittest.TestCase):
    """The bridges import with no engine present, which is what makes this fast."""

    def _bridges(self):
        from robosimstudio.engines.genesis.bridge import GenesisBridge
        from robosimstudio.engines.superdex.bridge import SuperdexBridge

        return {"superdex": SuperdexBridge, "genesis": GenesisBridge}

    def test_each_answers_everything_solver_host_calls(self):
        for alias, cls in self._bridges().items():
            for name in BRIDGE_SURFACE:
                with self.subTest(engine=alias, method=name):
                    self.assertTrue(callable(getattr(cls, name, None)))

    def test_each_bridge_is_named_by_a_registered_solver(self):
        """The two dicts are the same fact twice; this checks they agree."""
        self.assertEqual(set(self._bridges()), set(_host_backends()))

    def test_registration_imports_no_engine(self):
        """The alias is a dotted-path string, so a checkout without either extra
        composes, lists and runs everything untouched."""
        import subprocess
        import sys

        code = (
            "import sys\n"
            "from robosimstudio.specs.physics import get_backend\n"
            "for alias in ('superdex', 'genesis'):\n"
            "    assert get_backend(alias).host_engine\n"
            "assert 'genesis' not in sys.modules, 'resolving the alias imported the engine'\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
