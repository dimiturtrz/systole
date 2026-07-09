"""Import-graph diagnostic ranking (devtools/graph.py, bd au0o). Tests the pure ranking + cycle
detection on a hand-built graph — no grimp / no repo scan. networkx-guarded so CI (dev extra, no
networkx) skips; runs under the devtools extra."""
import pytest

nx = pytest.importorskip("networkx")

from devtools.graph import _DEFAULTS, _STRUCTURAL, _undersized, assert_fitness, load_structure_cfg, report, _top


def test_top_ranks_descending_and_caps():
    assert _top([("a", 1), ("b", 5), ("c", 3)], 2) == [("b", 5), ("c", 3)]


def test_report_surfaces_fan_in_and_cycles():
    g = nx.DiGraph()
    g.add_edges_from([("x", "hub"), ("y", "hub"), ("z", "hub"),   # hub: fan-in 3 (load-bearing)
                      ("a", "b"), ("b", "a")])                     # a<->b: an import cycle (SCC>1)
    r = report(g, top=5)
    assert "fan-in (load-bearing)" in r and "hub" in r
    assert "import cycles (SCC>1): 1" in r


def _god_graph():
    """A module 'god' with fan-in 2 AND fan-out 2 (> degree 1) + an a<->b import cycle."""
    g = nx.DiGraph()
    g.add_edges_from([("in1", "god"), ("in2", "god"), ("god", "out1"), ("god", "out2"),
                      ("a", "b"), ("b", "a")])
    return g


def test_assert_fitness_flags_god_module_cycle_and_oversized():
    cfg = {"bottleneck_degree": 1, "file_max": 100, "file_min": 50, "betweenness_max": 1.0}
    files = [("big.py", 200), ("small.py", 10), ("ok.py", 75)]
    blocking, advisory = assert_fitness(_god_graph(), files, cfg)
    assert any("god" in b and "god-module" in b for b in blocking)   # fan-in&out both > degree
    assert any("import cycle" in b for b in blocking)                # a<->b SCC>1
    assert any("big.py" in b for b in blocking)                      # 200 > file_max 100
    assert any("small.py" in b for b in advisory)                    # 10 < file_min 50 (advisory, not blocking)
    assert not any("small.py" in b for b in blocking)


def test_assert_fitness_clean_graph_has_no_blocking():
    g = nx.DiGraph([("a", "b"), ("b", "c")])                          # a chain: no god-module, no cycle
    blocking, _ = assert_fitness(g, [("ok.py", 300)], {**_DEFAULTS, "bottleneck_degree": 8})
    assert blocking == []


def test_undersized_exempts_structural_files():
    files = [("pkg/__init__.py", 1), ("pkg/__main__.py", 3), ("pkg/logic.py", 20)]
    msgs = _undersized(files, 250)
    assert any("logic.py" in m for m in msgs)                         # real module flagged
    assert not any(f.rsplit("/", 1)[-1] in m for f in [x[0] for x in files if x[0].endswith(_STRUCTURAL)] for m in msgs)


def test_load_structure_cfg_defaults_when_absent():
    cfg = load_structure_cfg("does_not_exist.toml")
    assert cfg == _DEFAULTS and cfg["file_max"] == 750
