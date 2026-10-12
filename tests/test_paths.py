# SPDX-License-Identifier: Apache-2.0
"""Where a vendored file is looked for, and why there are two places.

An asset should be a thing you can hand to somebody: geometry, the
``AssetRecord`` that declares it, and its provenance, in one directory. That is
what a task's own ``assets/`` is for, and it only works if a lookup finds a
mesh sitting beside its own declaration.

The second root is not a fallback, it is a different **kind** of thing.
``assets/`` holds what no declaration owns — robot descriptions, gripper meshes,
the scene-resident embodiments. Embodiments are not assets, which is a standing
rule in this stack rather than a filing accident, so they do not live
beside the content.

The third is not a kind of thing at all, it is a kind of *install*:
``robosimstudio/assets/``, the path ``pyproject.toml`` force-includes a copy
into, because a wheel has no repo root to climb to. It is last because a
checkout does not have one and an install has nothing else.

All three are searched, in that order. A resolver that stopped at the first root
that merely *existed* would find one kind and not the others.
"""

from __future__ import annotations

import unittest

from robosimstudio import paths as asset_paths

TIER = "fast"
"""No GPU, no scene. Gates every commit."""


class TestTheRoots(unittest.TestCase):
    def test_the_wheels_own_copy_is_searched_last(self) -> None:
        """What `pyproject.toml` force-includes must be somewhere we look.

        The two content roots landed without this one, and the omission was
        silent in every checkout: the repo root's ``assets/`` is right there, so
        nothing missed the third root until an *installed* wheel — which has no
        repo root and carries its copy at ``robosimstudio/assets/`` — could not
        resolve a single one of its own meshes.
        """
        wheel_copy = asset_paths._candidate_roots()[-1]
        self.assertEqual(wheel_copy.name, "assets")
        self.assertEqual(wheel_copy.parent.name, "robosimstudio")

    def test_an_override_replaces_every_root(self) -> None:
        """`ROBOSIMSTUDIO_ASSETS` points a deployment somewhere else outright.

        Replaces rather than prepends: a machine told where the assets are is
        making a statement about all of them, and quietly still searching the
        checkout would make a missing file look present.
        """
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {asset_paths._ENV_VAR: "/nowhere/at/all"}):
            self.assertEqual([str(r) for r in asset_paths._candidate_roots()], ["/nowhere/at/all"])

    def test_a_robot_description_still_resolves(self) -> None:
        """The vendored root. Nothing declares the Panda; it is an embodiment."""
        self.assertTrue(asset_paths.resolve_path("embodiments/panda/panda_kinematics.urdf").is_file())

    def test_a_missing_file_names_every_root_it_looked_in(self) -> None:
        """Several roots make "not found" ambiguous unless the message says all."""
        with self.assertRaises(FileNotFoundError) as caught:
            asset_paths.resolve_path("no_such_family/no_such_mesh.stl")
        message = str(caught.exception)
        for root in asset_paths._candidate_roots():
            self.assertIn(str(root), message)


if __name__ == "__main__":
    unittest.main()
