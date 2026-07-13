"""Data-clump detector (bd cardiac-seg-o07n): Fowler's 'Data Clumps' — param names that keep travelling
together across function signatures want to be ONE object (Introduce Parameter Object). Where ruff's
PLR0913 only counts a signature's ARITY, this catches WHICH params co-occur repo-wide, so a recurring
tuple like (inplane, n4, n4_params, nyul, norm) surfaces as a Recipe candidate. Bundling a long positional
list into an object also downgrades Page-Jones connascence-of-position → -of-name.

Signal: a clump = a param SET (>= `min_clump` params) that is carried WHOLE by >= `min_support`
functions (support = # signatures whose params are a superset). Only MAXIMAL sets are reported (a set is
dropped if a larger set with >= its support exists), so the recurring tuple surfaces, not its subsets.
Frequent-subset counting (not connected components) — the latter over-merges via hub params like `size`.

    python -m devtools.data_clumps core cardioseg
"""
from __future__ import annotations

import argparse
import ast
import logging
from itertools import combinations
from pathlib import Path

from core.obs import Obs

log = logging.getLogger("devtools.data_clumps")

_SELF = {"self", "cls"}
_MIN_SUPPORT = 4          # a param SET must be carried whole by >= this many functions to count
_MIN_CLUMP = 3            # a clump must bundle >= this many params (a pair is just a long-arg smell)
_MIN_PARAMS = 3           # only signatures with >= this many params can seed a clump
_MAX_CLUMP = 6            # cap enumerated subset size (bounds the candidate blow-up; real clumps are small)


def _params(fn: ast.FunctionDef) -> set[str]:
    """Positional + keyword-only param names of a function, minus self/cls."""
    a = fn.args
    return {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs) if p.arg not in _SELF}


def _functions(packages: list[str]) -> list[tuple[set[str], str]]:
    """(param-set, file) for every def with >= _MIN_PARAMS params across the packages."""
    out = []
    for pkg in packages:
        for path in sorted(Path(pkg).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    ps = _params(node)
                    if len(ps) >= _MIN_PARAMS:
                        out.append((ps, str(path)))
    return out


def _candidates(funcs: list[tuple[set[str], str]], min_clump: int) -> set[frozenset[str]]:
    """Every param subset (size min_clump.._MAX_CLUMP) drawn from some signature — the clumps to score."""
    cand: set[frozenset[str]] = set()
    for ps, _ in funcs:
        items = sorted(ps)
        for k in range(min_clump, min(len(items), _MAX_CLUMP) + 1):
            cand.update(frozenset(c) for c in combinations(items, k))
    return cand


def _support(clump: frozenset[str], funcs: list[tuple[set[str], str]]) -> list[str]:
    """Files of the functions whose params carry the WHOLE clump."""
    return [f for ps, f in funcs if clump <= ps]


def clumps(packages: list[str], min_support: int = _MIN_SUPPORT, min_clump: int = _MIN_CLUMP
           ) -> list[tuple[int, tuple[str, ...], int, list[str]]]:
    """Ranked (support, params, size, example_files) MAXIMAL data clumps, highest support first. `support`
    = functions carrying ALL the clump's params; a clump is dropped if a larger clump with >= its support
    exists (so the whole travelling tuple shows, not its subsets)."""
    funcs = _functions(packages)
    frequent = [(len(_support(cl, funcs)), cl) for cl in _candidates(funcs, min_clump)]
    frequent = sorted(((s, cl) for s, cl in frequent if s >= min_support), key=lambda r: (-r[0], -len(r[1])))
    kept: list[tuple[int, frozenset[str]]] = []
    for support, cl in frequent:
        if not any(cl < c2 and s2 >= support for s2, c2 in kept):    # keep only maximal-at-this-support
            kept.append((support, cl))
    return sorted(((s, tuple(sorted(cl)), len(cl), sorted(set(_support(cl, funcs)))[:3]) for s, cl in kept),
                  reverse=True)


def report(rows: list[tuple[int, tuple[str, ...], int, list[str]]]) -> str:
    """Ranked table: support (functions carrying the whole tuple), size, the params, example files."""
    lines = [f"{'supp':>4} {'size':>4}  {'clump (params that travel together)':45} examples"]
    for support, params, size, files in rows:
        lines.append(f"{support:>4} {size:>4}  {', '.join(params):45} {', '.join(files)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(prog="python -m devtools.data_clumps",
                                 description="find param sets that travel together (Introduce Parameter Object)")
    ap.add_argument("packages", nargs="*", default=["core", "cardioseg"])
    ap.add_argument("--min-support", type=int, default=_MIN_SUPPORT,
                    help="a param pair must co-occur in >= this many functions")
    ap.add_argument("--min-clump", type=int, default=_MIN_CLUMP, help="a clump must bundle >= this many params")
    args = ap.parse_args()
    Obs.setup()
    rows = clumps(args.packages, args.min_support, args.min_clump)
    log.info("%d data clumps\n%s", len(rows), report(rows))


if __name__ == "__main__":
    main()
