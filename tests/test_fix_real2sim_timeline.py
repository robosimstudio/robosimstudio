"""A labelled moment marked again moves, and a contact probe that raises is a FAIL line in the gate."""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

from robosimstudio_real2sim_agent import checks, frames

TIER = "fast"


class TestRemark(unittest.TestCase):
    def test_a_label_marked_again_replaces_its_old_moment(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="tl-"))
        for name in ("t000.00.jpg", "t004.50.jpg", "t004.60.jpg", "t009.97.jpg"):
            (tmp / name).write_bytes(b"")
        frames.write_timeline(
            tmp,
            {"clip": "c.mov"},
            [
                {"t": 0.0, "file": "t000.00.jpg", "label": "start"},
                {"t": 4.5, "file": "t004.50.jpg", "label": "lid_open"},
                {"t": 9.97, "file": "t009.97.jpg", "label": "end"},
            ],
        )
        frames.write_timeline(tmp, {"clip": "c.mov"}, [{"t": 4.6, "file": "t004.60.jpg", "label": "lid_open"}])
        anchors = frames.read_timeline(tmp)["anchors"]
        self.assertEqual([(a["t"], a["label"]) for a in anchors], [(0.0, "start"), (4.6, "lid_open"), (9.97, "end")])

    def test_an_unlabelled_anchor_leaves_the_labels_alone(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="tl-"))
        frames.write_timeline(tmp, {}, [{"t": 4.5, "file": "t004.50.jpg", "label": "lid_open"}])
        frames.write_timeline(tmp, {}, [{"t": 2.0, "file": "t002.00.jpg", "label": ""}])
        self.assertEqual([a["t"] for a in frames.read_timeline(tmp)["anchors"]], [2.0, 4.5])


class _Probe:
    def update(self) -> None:
        raise RuntimeError("probe update failed")

    def penetrations(self) -> list:
        return []


class _Sim:
    cameras = ()
    contacts = _Probe()
    model = None
    handles = types.SimpleNamespace(objects={})

    def state_is_finite(self) -> bool:
        return True

    def names(self) -> list[str]:
        return []

    def run(self, steps: int) -> None:
        pass

    def sync(self) -> None:
        pass


class TestGateProbe(unittest.TestCase):
    def test_a_probe_that_raises_is_a_failed_line(self) -> None:
        server = types.SimpleNamespace(sim=_Sim(), last_error="", build_error="")
        result = checks.gate(server, steps=1)
        line = next(c for c in result["checks"] if c["name"] == "penetration")
        self.assertFalse(line["ok"])
        self.assertIn("probe update failed", line["detail"])
        self.assertFalse(result["passed"])
        self.assertIn("finite_after", [c["name"] for c in result["checks"]], "the gate ran to the end")


if __name__ == "__main__":
    unittest.main()
