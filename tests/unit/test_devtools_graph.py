"""Import-graph diagnostic ranking (devtools/graph.py, bd au0o). Tests the pure ranking + cycle
detection on a hand-built graph — no grimp / no repo scan. networkx-guarded so CI (dev extra, no
networkx) skips; runs under the devtools extra."""
import pytest

nx = pytest.importorskip("networkx")

from devtools.graph import _top, report


def test_top_ranks_descending_and_caps():
    assert _top([("a", 1), ("b", 5), ("c", 3)], 2) == [("b", 5), ("c", 3)]


def test_report_surfaces_fan_in_and_cycles():
    g = nx.DiGraph()
    g.add_edges_from([("x", "hub"), ("y", "hub"), ("z", "hub"),   # hub: fan-in 3 (load-bearing)
                      ("a", "b"), ("b", "a")])                     # a<->b: an import cycle (SCC>1)
    r = report(g, top=5)
    assert "fan-in (load-bearing)" in r and "hub" in r
    assert "import cycles (SCC>1): 1" in r
