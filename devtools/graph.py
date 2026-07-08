"""Import-graph architecture diagnostic (bd au0o) — the degree>0 view vulture (degree-0 dead code) can't
give. grimp builds the honest import graph; networkx ranks structure:

  fan-in   (in_degree)   load-bearing        -> freeze its interface, test hardest
  fan-out  (out_degree)  orchestrating a lot -> decompose, hard to test isolated
  bottleneck (in*out)    classic tangle      -> prime refactor target
  betweenness            chokepoint          -> where to place a boundary/interface
  cycles (SCC>1)         tangle              -> break (import-linter gates layer cycles)

One-shot EXPLORER, not a CI gate — on a young solo repo the metrics surface 1-2 bottlenecks, not a pile;
run after big structural changes. IMPORT-level only: function-level lies on the cfg.build()/registry
dispatch. Dict-registered variants show as real edges, so a registry hub's fan-in reads expected-high —
the signal you want is fan-in/out on LOGIC modules. `python -m devtools.graph`.
"""
from __future__ import annotations

import argparse
import logging

import grimp
import networkx as nx

from core.obs import setup

log = logging.getLogger("cardioseg.devtools.graph")   # child of the "cardioseg" logger setup() configures


def build_graph(packages: list[str]) -> nx.DiGraph:
    """The honest import DiGraph (importer -> imported) over the given root packages, via grimp."""
    g = nx.DiGraph()
    for pkg in packages:
        mods = grimp.build_graph(pkg)
        for m in mods.modules:
            for dep in mods.find_modules_directly_imported_by(m):
                g.add_edge(m, dep)
    return g


def _top(pairs, n: int):
    """Top-n (name, score) by descending score — the shared ranking for every metric."""
    return sorted(pairs, key=lambda kv: -kv[1])[:n]


def report(g: nx.DiGraph, top: int) -> str:
    """Ranked fan-in / fan-out / bottleneck / chokepoint tables + the cycle list, as one text block."""
    ind, outd = dict(g.in_degree()), dict(g.out_degree())
    out = [f"import graph: {g.number_of_nodes()} modules, {g.number_of_edges()} edges", ""]
    for title, pairs in (
        ("fan-in (load-bearing)", ind.items()),
        ("fan-out (orchestrators)", outd.items()),
        ("bottleneck (fan-in x fan-out)", [(m, ind[m] * outd[m]) for m in g]),
        ("chokepoints (betweenness)", nx.betweenness_centrality(g).items()),
    ):
        out.append(f"{title}:")
        out += [f"  {score:>7.3f}  {name}" if isinstance(score, float) else f"  {score:>4}  {name}"
                for name, score in _top(pairs, top)]
        out.append("")
    cycles = [sorted(c) for c in nx.strongly_connected_components(g) if len(c) > 1]
    out.append(f"import cycles (SCC>1): {len(cycles)}")
    out += [f"  {c}" for c in cycles]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Import-graph architecture diagnostic (fan-in/out/centrality).")
    ap.add_argument("--packages", nargs="+", default=["core", "cardioseg"], help="root packages to graph")
    ap.add_argument("--top", type=int, default=10, help="rows per ranked table")
    args = ap.parse_args()
    setup()
    log.info("\n%s", report(build_graph(args.packages), args.top))


if __name__ == "__main__":
    main()
