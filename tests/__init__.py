# SPDX-License-Identifier: Apache-2.0
"""Tests for every layer, run from the repo root.

The suite is tiered by what a test *needs*, and each module declares its tier as
a module-level ``TIER`` constant. ``scripts/testrun.py`` selects on that -- reading
it by AST rather than by importing, so listing the slow tier costs nothing and
does not initialise CUDA::

    uv run python scripts/testrun.py --list          # what runs where
    uv run python scripts/testrun.py --tier fast     # no GPU, seconds, every commit
    uv run python scripts/testrun.py --tier medium   # a GPU and a scene, by hand
    uv run python scripts/testrun.py --tier slow     # a GPU and minutes, by hand

Naming the modules by hand still works and is what ``unittest`` sees either way::

    python -m unittest tests.test_core tests.test_paths -v
    python -m unittest tests.test_bare_scenes -v   # needs a GPU

Listing them by hand is what this docstring used to do, and it had drifted from
the README's list in both directions -- which is the argument for a declared
tier: a module that names itself cannot be forgotten by one list and
double-counted by another. ``scripts/testrun.py`` enforces that every module
declares one: it exits non-zero, naming the module, rather than run a suite
that silently leaves one out. CI runs the fast tier; the medium and slow tiers
are run by hand on a GPU machine.

``python -m`` puts the repo root on ``sys.path``, so nothing needs installing and
``PYTHONPATH`` stays empty.
"""
