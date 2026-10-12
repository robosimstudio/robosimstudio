# SPDX-License-Identifier: Apache-2.0
"""An imported mesh may carry its diffuse map, and must say when it cannot.

``EntityRigid`` loads a mesh through ``trimesh`` and hands Newton vertices and
faces. Until ``AssetSpec.texture`` there was nowhere to put the image, so every
per-object import arrived flat -- which is a real loss and not a cosmetic one on
the tasks that name these meshes. ``tool_sort`` asks for the multimeter out of
thirty textured tools of similar size, and thirty grey blocks do not contain
that question; it also asks for the compartment whose label names a tool,
which is the argument
:func:`~robosimstudio.scene.build.imports.texture_box_geoms` already makes at length for
the plates on the bin floor.

**The failure this module is really about is the silent one.** A mesh whose UVs
did not survive its converter still loads, still renders, and renders *flat* --
so a texture that did not attach looks exactly like a texture that did not exist.
``trimesh.load(force="mesh")`` drops UVs whenever it concatenates several
materials, which is precisely when a vendored object needs re-exporting rather
than shipping unnoticed. Naming a texture is therefore a claim that the geometry
can sample one, and an unmet claim raises.

No builder here and no device: the join between a mesh's coordinates and a
sibling image is arithmetic, and it is tested as arithmetic.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from robosimstudio.domains.rigid import EntityRigid
from robosimstudio.paths import resolve_path
from robosimstudio.specs import AssetSpec, ObjectSpec

TIER = "fast"
"""No GPU: a spec, a stub mesh, and the method that joins them."""

MESH = "vab_tool_sort/assets/industrial/multimeter/meshes/multimeter.obj"
TEXTURE = "vab_tool_sort/assets/industrial/multimeter/textures/diffuse.png"
"""A textured tool from the shared export: UVs on the mesh, the image beside it."""


def _vendored(path: str) -> bool:
    """Whether ``path`` resolves on this tree's asset roots."""
    try:
        resolve_path(path)
    except FileNotFoundError:
        return False
    return True


HAS_THE_TOOL = _vendored(TEXTURE)
"""Whether the textured tool is vendored here.

It ships with a task collection, and a release may carry the library without
one. Four of the six claims below never open the file -- the decision is tested
on a stub mesh and the path is only a string. The two that resolve it state both
answers: where the tool is, the path; where it is not, a refusal that names it,
which is this module's subject either way.
"""


def _mesh(uv):
    """The two attributes :meth:`EntityRigid._texturing` reads off a trimesh.

    A stub rather than a real load, and deliberately: what is under test is the
    decision, and a decision tested through a two-megabyte OBJ is a decision
    tested through ``trimesh``'s loader as well.
    """
    return SimpleNamespace(vertices=np.zeros((4, 3)), visual=SimpleNamespace(uv=uv))


def _entity(**over) -> EntityRigid:
    asset = AssetSpec(path=MESH, **{k: v for k, v in over.items() if k == "texture"})
    return EntityRigid(ObjectSpec(name="multimeter_0", category="multimeter", asset=asset))


class TestTheSpecCarriesIt(unittest.TestCase):
    def test_an_asset_states_its_map_and_defaults_to_none(self):
        self.assertEqual("", AssetSpec(path=MESH).texture)
        self.assertEqual(TEXTURE, AssetSpec(path=MESH, texture=TEXTURE).texture)

    def test_the_map_resolves_like_any_other_asset_path(self):
        """Through the same root search a mesh path goes through.

        Stated because the alternative -- a path relative to the mesh -- is the
        obvious guess and is not what this does: the two are separate fields and
        an asset may keep its map anywhere it keeps its geometry.

        Where the tool is not vendored the same search refuses the map by name
        and says where it looked -- the other half of "like any other path".
        """
        if HAS_THE_TOOL:
            self.assertTrue(resolve_path(TEXTURE).is_file())
            return
        with self.assertRaises(FileNotFoundError) as caught:
            resolve_path(TEXTURE)
        self.assertIn(TEXTURE, str(caught.exception))
        self.assertIn("looked for in", str(caught.exception))


class TestTheJoin(unittest.TestCase):
    def test_no_texture_asks_newton_for_nothing(self):
        """An untextured import must not pass ``texture=None`` either.

        ``newton.Mesh`` takes both keywords optionally, and handing it explicit
        ``None``s is a different call from omitting them -- so the untextured
        path returns an empty mapping and the call site stays the one it was.
        """
        self.assertEqual({}, _entity()._texturing(_mesh(np.zeros((4, 2)))))

    def test_a_textured_import_passes_uvs_and_a_resolved_path(self):
        """Resolved here, not by Newton: a map that is not on any root is
        refused by name at the join, before a path that leads nowhere is handed
        on to be discovered as a flat object."""
        mesh = _mesh(np.ones((4, 2), dtype=np.float64))
        if not HAS_THE_TOOL:
            with self.assertRaises(FileNotFoundError) as caught:
                _entity(texture=TEXTURE)._texturing(mesh)
            self.assertIn(TEXTURE, str(caught.exception))
            return
        got = _entity(texture=TEXTURE)._texturing(mesh)
        self.assertEqual({"uvs", "texture"}, set(got))
        self.assertEqual((4, 2), got["uvs"].shape)
        self.assertEqual(np.float32, got["uvs"].dtype)
        self.assertEqual(str(resolve_path(TEXTURE)), got["texture"])


class TestARefusalRatherThanAFlatObject(unittest.TestCase):
    """Both shapes of "the UVs are not there", which look identical when drawn."""

    def test_a_mesh_with_no_uvs_at_all_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            _entity(texture=TEXTURE)._texturing(_mesh(None))
        self.assertIn("no per-vertex UVs", str(caught.exception))

    def test_a_mesh_with_the_wrong_number_of_uvs_is_refused(self):
        """The concatenation case, and the one worth having a test for.

        ``trimesh`` keeps a ``uv`` array through some multi-material loads and
        silently leaves it the length of one submesh. Newton would then sample
        the map at whatever the other vertices index into, which is not an
        error anywhere and is visible only as a mesh wearing part of its own
        texture.
        """
        with self.assertRaises(ValueError):
            _entity(texture=TEXTURE)._texturing(_mesh(np.ones((3, 2))))


if __name__ == "__main__":
    unittest.main()
