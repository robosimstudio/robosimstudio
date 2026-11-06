# SPDX-License-Identifier: Apache-2.0
"""A graph's helper modules are imported fresh each episode (robosimstudio_policy_agent.graphs.forget_graph_modules)."""

import importlib
import sys

from robosimstudio_policy_agent.graphs import forget_graph_modules

TIER = "fast"


def test_episode_state_in_a_helper_module_does_not_survive(tmp_path, monkeypatch):
    scripts = tmp_path / "graph" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "helper_cache_mod.py").write_text("_CACHE = {}\n")
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "unrelated_cache_mod.py").write_text("_CACHE = {}\n")
    monkeypatch.syspath_prepend(str(scripts))
    monkeypatch.syspath_prepend(str(other))

    helper = importlib.import_module("helper_cache_mod")
    unrelated = importlib.import_module("unrelated_cache_mod")
    helper._CACHE["geometry"] = "last episode's board"
    unrelated._CACHE["kept"] = True

    dropped = forget_graph_modules(tmp_path / "graph", None)

    assert dropped == ["helper_cache_mod"]
    assert importlib.import_module("helper_cache_mod")._CACHE == {}
    assert sys.modules["unrelated_cache_mod"] is unrelated  # outside the graph: untouched
    sys.modules.pop("helper_cache_mod", None)
    sys.modules.pop("unrelated_cache_mod", None)


def test_no_directories_is_a_no_op():
    assert forget_graph_modules(None, "") == []
