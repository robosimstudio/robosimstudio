# SPDX-License-Identifier: Apache-2.0
"""``ROBOSIMSTUDIO_SEGMENTATION_BACKEND=palette`` is served even where the real sam3 resolves.

The in-process ``sam3.segment_text`` is the only implementation that reads
the variable; the reference bundle is real SAM3 and never does. So a run that
asks for ``palette`` on a checkout whose sibling registry ships ``sam3`` used
to get the reference instead -- silently, since the request is honoured by a
tool that was never registered. The cable delivery's whole evaluation
condition was ``--segmentation palette``, and on this tree the same thirty
USB-C Medium scenes went **2/30 to 0/30**, every one ending in
``no 'orange plug' in 'topdown'``, before anything said why.

Pure registry arithmetic against a fake registry root that "ships" sam3.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TIER = "fast"
"""No scene, no weights: a temp directory and a ToolRegistry. Gates every commit."""


def _fake_registry_shipping_sam3(root: Path) -> Path:
    bundle = root / "tools" / "sam3"
    bundle.mkdir(parents=True)
    (bundle / "SKILL.md").write_text("---\nname: sam3\ndescription: fake\n---\n")
    return root


def _registered(backend: str | None, registries: list[str]) -> set[str]:
    from gap_core.tools import ToolRegistry

    from robosimstudio_policy_agent.perception_tools import register_perception_tools

    env = dict(os.environ)
    env.pop("ROBOSIMSTUDIO_SEGMENTATION_BACKEND", None)
    if backend is not None:
        env["ROBOSIMSTUDIO_SEGMENTATION_BACKEND"] = backend
    registry = ToolRegistry()
    with mock.patch.dict(os.environ, env, clear=True):
        register_perception_tools(registry, registries=registries)
    return set(registry._tools)


HAS_GAP = importlib.util.find_spec("gap_core") is not None


@unittest.skipUnless(HAS_GAP, "the graph runtime is not installed (the [gap] extra)")
class TestThePaletteRequestWins(unittest.TestCase):
    def test_palette_registers_in_process_even_when_the_reference_resolves(self) -> None:
        from robosimstudio_policy_agent.perception_tools import reference_bundle_supplies_sam3

        with TemporaryDirectory() as tmp:
            root = _fake_registry_shipping_sam3(Path(tmp))
            self.assertTrue(
                reference_bundle_supplies_sam3([str(root)]),
                "the fixture must look like a registry that ships sam3, or the test is vacuous",
            )
            self.assertIn("sam3.segment_text", _registered("palette", [str(root)]))

    def test_auto_still_defers_to_the_reference(self) -> None:
        """The other side of the rule: unasked, the reference keeps winning."""
        with TemporaryDirectory() as tmp:
            root = _fake_registry_shipping_sam3(Path(tmp))
            self.assertNotIn("sam3.segment_text", _registered(None, [str(root)]))
            self.assertNotIn("sam3.segment_text", _registered("sam3", [str(root)]))

    def test_without_a_reference_the_in_process_tool_registers_regardless(self) -> None:
        with TemporaryDirectory() as tmp:
            self.assertIn("sam3.segment_text", _registered(None, [tmp]))


if __name__ == "__main__":
    unittest.main()
