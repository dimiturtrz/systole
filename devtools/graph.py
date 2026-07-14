"""Import-graph architecture diagnostic + fitness gate. grimp builds the honest import graph;
networkx ranks structure and `--assert` turns the measurable properties into a RATCHETED gate (the
metric arch axis import-linter's categorical layer contracts can't express):

  fan-in   (in_degree)   load-bearing        -> freeze its interface, test hardest
  fan-out  (out_degree)  orchestrating a lot -> decompose, hard to test isolated
  bottleneck (in*out)    classic tangle      -> prime refactor target
  betweenness            chokepoint          -> where to place a boundary/interface
  cycles (SCC>1)         tangle              -> break (import-linter gates layer cycles)

`report` is the one-shot EXPLORER (ranked tables). `--assert` is the GATE: it fails when a module is a
god-module (fan-in AND fan-out BOTH over a degree), a new import cycle appears, a file blows the line
ceiling, or a logic module has no strict path-mirror test (`tests/unit/<pkg>/<path>/test_foo.py`) — plus
an advisory chokepoint warning that never blocks. Thresholds live in `pyproject
[tool.structure]`, defaulted when absent. IMPORT-level only. Packages to graph are the positional argv
(default `src`). Run: `python -m devtools.graph [pkgs...]` | `python -m devtools.graph --assert [pkgs...]`.
"""

from __future__ import annotations

import argparse
import logging
import tomllib
from pathlib import Path

import grimp
import networkx as nx

from devtools.omit import coverage_omit, matches_omit

log = logging.getLogger("devtools.graph")

# Fitness thresholds — SPEC [tool.structure] defaults, overridable in pyproject. Chosen so the blocking
# rules start CLEAN on a fresh project and ratchet: fan-in&out both>8, file>750 lines, any import cycle.
_DEFAULTS = {"bottleneck_degree": 8, "file_max": 750, "file_min": 0, "betweenness_max": 0.10, "test_layout": "mirror"}
_STRUCTURAL = ("__init__.py", "__main__.py")  # package plumbing — exempt from the test-mirror rule + line floor
_ADVISORY_PREVIEW = 15  # advisory lines shown before "… +N more" (avoid log spam)


def load_structure_cfg(pyproject: str = "pyproject.toml") -> dict:
    """Fitness thresholds from pyproject [tool.structure], merged onto SPEC defaults. One config home."""
    cfg = dict(_DEFAULTS)
    p = Path(pyproject)
    if p.exists():
        cfg.update(tomllib.loads(p.read_text(encoding="utf-8")).get("tool", {}).get("structure", {}))
    return cfg


def build_graph(packages: list[str]) -> nx.DiGraph:
    """The honest import DiGraph (importer -> imported) over the given root packages, via grimp."""
    g = nx.DiGraph()
    for pkg in packages:
        mods = grimp.build_graph(pkg)
        for m in mods.modules:
            g.add_node(m)
            for dep in mods.find_modules_directly_imported_by(m):
                g.add_edge(m, dep)
    return g


def file_lines(packages: list[str]) -> list[tuple[str, int]]:
    """(path, line-count) for every .py under the root packages — the file-shape axis the graph can't see."""
    return [
        (str(f), f.read_text(encoding="utf-8").count("\n") + 1)
        for pkg in packages
        for f in sorted(Path(pkg).rglob("*.py"))
    ]


def _god_modules(g: nx.DiGraph, degree: int) -> list[str]:
    ind, outd = dict(g.in_degree()), dict(g.out_degree())
    return [
        f"{n}: fan-in {ind[n]} x fan-out {outd[n]} (both > {degree}) — god-module, split by responsibility"
        for n in g
        if ind[n] > degree and outd[n] > degree
    ]


def _cycles(g: nx.DiGraph) -> list[str]:
    return [f"import cycle (SCC>1): {sorted(c)}" for c in nx.strongly_connected_components(g) if len(c) > 1]


def _oversized(files: list[tuple[str, int]], mx: int) -> list[str]:
    return [f"{f}: {n} lines > {mx} — god-file, split" for f, n in files if n > mx]


def unmirrored(packages: list[str], layout: str = "mirror", test_root: str = "tests/unit") -> list[str]:
    """LOGIC source modules with no unit test — the universal rule is "every logic module HAS a test";
    WHERE it lives is `layout` (a `[tool.structure]` choice, so it's config, not imposed architecture):
      - "mirror" (default): STRICT path-mirror, one home per module — `<pkg>/<path>/foo.py` is covered iff
        `tests/unit/<pkg>/<path>/test_foo.py` exists (a same-purpose test under a different name/path doesn't
        count). The cardiac/mindscape discipline.
      - "flat": lenient — a `test_foo.py` exists ANYWHERE under `tests/`. Lets a flat-layout repo satisfy the
        gate without restructuring its test tree.
      - "off": no test-existence gate.
    `__init__`/`__main__` are plumbing, exempt; coverage-OMITTED shells (runners/adapters/GPU/download/viz glue,
    read from `[tool.coverage] omit`) are exempt too — a non-unit-testable shell isn't forced to carry a stub."""
    if layout == "off":
        return []
    omit = coverage_omit()
    flat_names = {p.name for p in Path("tests").rglob("test_*.py")} if layout == "flat" else set()
    out = []
    for pkg in packages:
        for f in sorted(Path(pkg).rglob("*.py")):
            if f.name in _STRUCTURAL or matches_omit(f.as_posix(), omit):
                continue
            if layout == "flat":
                if f"test_{f.name}" not in flat_names:
                    out.append(f"{f.as_posix()} — no test_{f.name} anywhere under tests/")
            else:
                mirror = Path(test_root) / f.parent / f"test_{f.name}"
                if not mirror.exists():
                    out.append(f"{f.as_posix()} — no mirrored {mirror.as_posix()}")
    return out


def _undersized(files: list[tuple[str, int]], mn: int) -> list[str]:
    """Advisory line floor. OFF at mn<=0 (the default) — no honest universal floor; small files are often
    SSOT / strategy / shared-vocab leaves, and 'too thin' is a responsibility call, not a line count."""
    if mn <= 0:
        return []
    return [
        f"{f}: {n} lines < {mn} — earn its keep? (fold, or accept a small leaf)"
        for f, n in files
        if n < mn and not f.endswith(_STRUCTURAL)
    ]


def _chokepoints(g: nx.DiGraph, mx: float) -> list[str]:
    return [
        f"{n}: betweenness {v:.3f} > {mx} — chokepoint, consider a boundary here"
        for n, v in nx.betweenness_centrality(g).items()
        if v > mx
    ]


def assert_fitness(g: nx.DiGraph, files: list[tuple[str, int]], cfg: dict) -> tuple[list[str], list[str]]:
    """(blocking, advisory) fitness violations. BLOCKING = god-module, import cycle, god-file (clean on a
    fresh project, so they ratchet); ADVISORY = line-floor (off by default) + chokepoint (never blocks)."""
    blocking = _god_modules(g, cfg["bottleneck_degree"]) + _cycles(g) + _oversized(files, cfg["file_max"])
    advisory = _undersized(files, cfg["file_min"]) + _chokepoints(g, cfg["betweenness_max"])
    return blocking, advisory


def _top(pairs, n: int):
    """Top-n (name, score) by descending score — the shared ranking for every metric."""
    return sorted(pairs, key=lambda kv: -kv[1])[:n]


def report(g: nx.DiGraph, top: int) -> str:
    """Ranked fan-in / fan-out / bottleneck / chokepoint tables + the cycle list, as one text block."""
    ind, outd = dict(g.in_degree()), dict(g.out_degree())
    out = [
        f"import graph: {g.number_of_nodes()} modules, {g.number_of_edges()} edges",
        "",
    ]
    for title, pairs in (
        ("fan-in (load-bearing)", ind.items()),
        ("fan-out (orchestrators)", outd.items()),
        ("bottleneck (fan-in x fan-out)", [(m, ind[m] * outd[m]) for m in g]),
        ("chokepoints (betweenness)", nx.betweenness_centrality(g).items()),
    ):
        out.append(f"{title}:")
        out += [
            f"  {score:>7.3f}  {name}" if isinstance(score, float) else f"  {score:>4}  {name}"
            for name, score in _top(pairs, top)
        ]
        out.append("")
    cycles = [sorted(c) for c in nx.strongly_connected_components(g) if len(c) > 1]
    out.append(f"import cycles (SCC>1): {len(cycles)}")
    out += [f"  {c}" for c in cycles]
    return "\n".join(out)


def _run_assert(packages: list[str]) -> int:
    """The gate: log advisory warnings, log blocking errors, return exit code (1 if any blocking)."""
    cfg = load_structure_cfg()
    g, files = build_graph(packages), file_lines(packages)
    blocking, advisory = assert_fitness(g, files, cfg)
    blocking += [f"test mirror: {m}" for m in unmirrored(packages, cfg["test_layout"])]  # module w/o a test blocks
    if advisory:
        shown = advisory[:_ADVISORY_PREVIEW]
        extra = f"\n  … +{len(advisory) - _ADVISORY_PREVIEW} more" if len(advisory) > _ADVISORY_PREVIEW else ""
        log.warning(
            "architecture fitness — advisory (%d, non-blocking):\n  %s",
            len(advisory),
            "\n  ".join(shown) + extra,
        )
    if blocking:
        log.error(
            "architecture fitness — BLOCKING (%d):\n  %s",
            len(blocking),
            "\n  ".join(blocking),
        )
        return 1
    log.info(
        "architecture fitness: clean (0 god-modules / cycles / god-files; %d advisory)",
        len(advisory),
    )
    return 0


def main():
    ap = argparse.ArgumentParser(description="Import-graph architecture diagnostic + fitness gate.")
    ap.add_argument(
        "packages",
        nargs="*",
        default=["src"],
        help="root packages to graph (default: src)",
    )
    ap.add_argument("--top", type=int, default=10, help="rows per ranked table")
    ap.add_argument(
        "--assert",
        action="store_true",
        dest="assert_",
        help="fitness GATE: exit 1 on a god-module / import cycle / god-file / test-mirror gap (advisory: chokepoint)",
    )
    args = ap.parse_args()
    packages = args.packages or ["src"]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.assert_:
        raise SystemExit(_run_assert(packages))
    log.info("\n%s", report(build_graph(packages), args.top))


if __name__ == "__main__":
    main()
