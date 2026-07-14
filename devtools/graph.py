"""Import-graph architecture diagnostic + fitness gate. grimp builds the honest import graph;
networkx ranks structure and `--assert` turns the measurable properties into a RATCHETED gate (the
metric arch axis import-linter's categorical layer contracts can't express):

  fan-in   (in_degree)   load-bearing        -> freeze its interface, test hardest
  fan-out  (out_degree)  orchestrating a lot -> decompose, hard to test isolated
  bottleneck (in*out)    classic tangle      -> prime refactor target
  betweenness            chokepoint          -> where to place a boundary/interface
  cycles (SCC>1)         tangle              -> break (import-linter gates layer cycles)
  instability I=Ce/(Ce+Ca)  stable vs. volatile  -> depend in the direction of stability
  main-seq. distance |A+I-1|  balance of A vs I  -> off it = zone of pain / uselessness (advisory)

`report` is the one-shot EXPLORER (ranked tables). `--assert` is the GATE: it fails when a module is a
god-module (fan-in AND fan-out BOTH over a degree), a new import cycle appears, a file blows the line
ceiling, or a logic module has no strict path-mirror test (`tests/unit/<pkg>/<path>/test_foo.py`) — plus
an advisory chokepoint warning that never blocks. Thresholds live in `pyproject
[tool.structure]`, defaulted when absent. IMPORT-level only. Packages to graph are the positional argv
(default `src`). Run: `python -m devtools.graph [pkgs...]` | `python -m devtools.graph --assert [pkgs...]`.
"""

from __future__ import annotations

import argparse
import ast
import logging
import tomllib
from pathlib import Path

import grimp
import networkx as nx

from devtools.omit import coverage_omit, matches_omit

log = logging.getLogger("devtools.graph")

# Fitness thresholds — SPEC [tool.structure] defaults, overridable in pyproject. Chosen so the blocking
# rules start CLEAN on a fresh project and ratchet: fan-in&out both>8, file>750 lines, any import cycle.
_DEFAULTS = {
    "bottleneck_degree": 8,
    "file_max": 750,
    "file_min": 0,
    "betweenness_max": 0.10,
    "main_sequence_max": 0.0,  # advisory main-sequence distance ceiling; 0 = OFF (no honest universal one)
    "test_layout": "mirror",
}
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


# Martin's stability metrics (bd x3b). Edges are importer -> imported, so in_degree = afferent coupling Ca
# (who depends on me) and out_degree = efferent Ce (who I depend on). Instability I = Ce/(Ce+Ca) and
# abstractness A = abstract-classes / classes place a module on the A-I plane; distance from the "main
# sequence" (the ideal A + I = 1 line) is D = |A + I - 1|.
_ABSTRACT_BASES = {"ABC", "ABCMeta", "Protocol"}
_ABSTRACT_DECORATORS = {"abstractmethod", "abstractproperty"}


def _dotted(node: ast.expr) -> str | None:
    """The trailing name of a Name/Attribute node (`abc.ABC` -> 'ABC'), else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_abstract(cls: ast.ClassDef) -> bool:
    """A class is abstract if it subclasses ABC/Protocol, sets metaclass=ABCMeta, or has an @abstractmethod."""
    if any(_dotted(b) in _ABSTRACT_BASES for b in cls.bases):
        return True
    if any(kw.arg == "metaclass" and _dotted(kw.value) == "ABCMeta" for kw in cls.keywords):
        return True
    return any(
        isinstance(m, ast.FunctionDef | ast.AsyncFunctionDef)
        and any(_dotted(d) in _ABSTRACT_DECORATORS for d in m.decorator_list)
        for m in cls.body
    )


def _module_file(mod: str) -> Path | None:
    """The .py file backing a dotted module name (`a.b` -> a/b.py or a/b/__init__.py), if it exists."""
    parts = mod.split(".")
    for cand in (Path(*parts).with_suffix(".py"), Path(*parts) / "__init__.py"):
        if cand.exists():
            return cand
    return None


def abstractness(mod: str) -> float | None:
    """Martin's A = abstract classes / total classes in the module (None if no backing file or no classes)."""
    f = _module_file(mod)
    if f is None:
        return None
    classes = [n for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))) if isinstance(n, ast.ClassDef)]
    if not classes:
        return None
    return sum(_is_abstract(c) for c in classes) / len(classes)


def instability(g: nx.DiGraph) -> dict[str, float]:
    """Martin's I = Ce/(Ce+Ca): 0 = maximally STABLE (only depended-on), 1 = maximally UNSTABLE (only
    depends on others). Isolated nodes (no coupling) are skipped — I is undefined there."""
    ind, outd = dict(g.in_degree()), dict(g.out_degree())
    return {n: outd[n] / (outd[n] + ind[n]) for n in g if outd[n] + ind[n] > 0}


def main_sequence_distance(g: nx.DiGraph) -> dict[str, float]:
    """D = |A + I - 1|: distance from the main sequence. High D = the zone of PAIN (stable + concrete, hard
    to extend) or the zone of USELESSNESS (abstract + unstable). Only modules with classes (A defined)."""
    out = {}
    for n, i in instability(g).items():
        a = abstractness(n)
        if a is not None:
            out[n] = abs(a + i - 1)
    return out


def _off_main_sequence(g: nx.DiGraph, mx: float) -> list[str]:
    """Advisory. OFF at mx<=0 (the default): a concrete stable leaf legitimately sits at D≈1, so there is no
    honest universal threshold — a repo opts in by setting `main_sequence_max`, then flags modules past it."""
    if mx <= 0:
        return []
    return [
        f"{n}: main-sequence distance {d:.2f} > {mx} — off the main sequence (zone of pain/uselessness)"
        for n, d in main_sequence_distance(g).items()
        if d > mx
    ]


def assert_fitness(g: nx.DiGraph, files: list[tuple[str, int]], cfg: dict) -> tuple[list[str], list[str]]:
    """(blocking, advisory) fitness violations. BLOCKING = god-module, import cycle, god-file (clean on a
    fresh project, so they ratchet); ADVISORY = line-floor (off by default) + chokepoint (never blocks)."""
    blocking = _god_modules(g, cfg["bottleneck_degree"]) + _cycles(g) + _oversized(files, cfg["file_max"])
    advisory = (
        _undersized(files, cfg["file_min"])
        + _chokepoints(g, cfg["betweenness_max"])
        + _off_main_sequence(g, cfg["main_sequence_max"])
    )
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
        ("instability I=Ce/(Ce+Ca)", instability(g).items()),
        ("main-sequence distance |A+I-1|", main_sequence_distance(g).items()),
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
