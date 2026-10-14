# SPDX-License-Identifier: Apache-2.0
"""The tuning binders: set_spec records nothing, and a field set back is not pending.

A fake gui stands in for viser, firing ``on_update`` synchronously on a
``.value`` assignment that changes the value -- which is what viser does.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from unittest import mock

from robosimstudio.specs.params import REBUILD, ParamUI, ui
from robosimstudio_studio.tuning.arrangement import ArrangementBinder
from robosimstudio_studio.tuning.binder import SpecBinder

TIER = "fast"


class _Widget:
    def __init__(self, initial_value=None, **_):
        self._value = initial_value
        self._callbacks = []

    def on_update(self, fn):
        self._callbacks.append(fn)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value == self._value:
            return
        self._value = value
        for fn in self._callbacks:
            fn(None)


class _Gui:
    def __init__(self):
        self.widgets = {}

    def add_markdown(self, *_a, **_k):
        return _Widget()

    def _add(self, label, **kw):
        widget = _Widget(**kw)
        self.widgets[label] = widget
        return widget

    add_checkbox = add_dropdown = add_text = add_slider = add_number = _add


@dataclass(frozen=True)
class _Spec:
    stiff: float = field(default=13700.0, metadata=ui(tier="live", range=(1.0, 1e6), log=True))
    damp: float = field(default=0.5, metadata=ui(tier="live", range=(0.0, 1.0)))
    mode: str = field(default="a", metadata=ui(tier="live", choices=("a", "b")))


class SetSpecRecordsNothing(unittest.TestCase):
    def test_moving_widgets_to_a_new_spec_is_not_an_edit(self):
        binder = SpecBinder(_Spec())
        binder.build(_Gui())
        binder.set_spec(replace(_Spec(), stiff=13700.0 * 1.37, damp=0.9, mode="b"))
        self.assertEqual(binder.pending, {})

    def test_a_real_edit_still_records(self):
        gui = _Gui()
        binder = SpecBinder(_Spec())
        binder.build(gui)
        binder.set_spec(replace(_Spec(), damp=0.9))
        gui.widgets["damp"].value = 0.25
        self.assertEqual(binder.pending, {"damp": 0.25})


class PendingRebuildForgetsWhatWasSetBack(unittest.TestCase):
    def _binder(self):
        spec = SimpleNamespace(with_=lambda **_: spec)
        binder = ArrangementBinder(spec)
        binder._fold = mock.Mock()
        gui = _Gui()
        binder._add(gui, "rates:substeps", "substeps", 4, ParamUI(tier=REBUILD, range=(1.0, 16.0), step=1.0))
        return binder, gui.widgets["substeps *"]

    def test_set_back_to_the_built_value(self):
        binder, widget = self._binder()
        widget.value = 8
        binder.drain()
        self.assertEqual(binder.pending_rebuild(), ("rates:substeps",))
        widget.value = 4
        binder.drain()
        self.assertEqual(binder.pending_rebuild(), ())

    def _log_binder(self, built: float = 13700.0):
        spec = SimpleNamespace(with_=lambda **_: spec)
        binder = ArrangementBinder(spec)
        binder._fold = mock.Mock()
        gui = _Gui()
        binder._add(gui, "rates:edge_ke", "edge_ke", built, ParamUI(tier=REBUILD, range=(1.0, 1e6), log=True))
        return binder, gui.widgets["edge_ke *"]

    def test_a_log_slider_dragged_back_onto_its_grid_is_set_back(self):
        import math

        binder, widget = self._log_binder()
        widget.value = 5.0
        binder.drain()
        self.assertEqual(binder.pending_rebuild(), ("rates:edge_ke",))
        # The browser snaps to the 0.01 grid, which the built value is not on.
        widget.value = round(math.log10(13700.0), 2)
        self.assertNotEqual(10.0**widget.value, 13700.0)
        binder.drain()
        self.assertEqual(binder.pending_rebuild(), ())
        # And what is folded is the scene's own value, not the grid point.
        binder._fold.assert_called_with("rates:edge_ke", 13700.0)

    def test_a_log_slider_one_notch_off_is_still_an_edit(self):
        import math

        binder, widget = self._log_binder()
        widget.value = round(math.log10(13700.0), 2) + 0.01
        binder.drain()
        self.assertEqual(binder.pending_rebuild(), ("rates:edge_ke",))


if __name__ == "__main__":
    unittest.main()
