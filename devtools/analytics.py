"""Code-size / complexity analytics — a one-shot EXPLORER (like `devtools.graph`'s report mode), not a
gate. Tracks that methods stay thin and complexity stays in the engine as the codebase grows, rather than
scattering into fat leaf functions.

Per-file and per-area: code lines (non-blank, non-comment), def count, and a branch-node complexity proxy —
`if/for/while/try/except/with/and/or/ternary/comprehension`, the McCabe-style decision points, summed via
`ast` (no radon dependency). Plus the src-vs-test line ratio and the top-N largest files. Complexity-per-def
is the number to watch — a rising max means logic is leaking into leaves that should delegate.

    python -m devtools.analytics --areas core neuroscan devtools     # scan your packages
    python -m devtools.analytics --areas core --flag-over 250        # list files over a code-line budget
"""

from __future__ import annotations

import argparse
import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("devtools.analytics")

_AREAS = ["src"]  # generic default — pass --areas <your packages> (+ devtools)
_BRANCH_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.IfExp,
    ast.comprehension,
    ast.BoolOp,
)


@dataclass
class FileStat:
    path: Path
    code_lines: int
    defs: int
    branches: int  # decision points (cyclomatic proxy)


@dataclass
class AreaStat:
    name: str
    files: list[FileStat] = field(default_factory=list)

    @property
    def code_lines(self) -> int:
        return sum(f.code_lines for f in self.files)

    @property
    def defs(self) -> int:
        return sum(f.defs for f in self.files)

    @property
    def branches(self) -> int:
        return sum(f.branches for f in self.files)


class Analytics:
    """Per-area code-size + complexity explorer over a repo's declared areas."""

    def __init__(self, repo: Path, areas: list[str]) -> None:
        self.repo = repo
        self.areas = areas

    @staticmethod
    def _code_lines(source: str) -> int:
        """Non-blank lines that aren't a pure comment (a crude but stable LOC — string-literal bodies count,
        which is fine: docstrings are code you maintain)."""
        return sum(1 for ln in source.splitlines() if ln.strip() and not ln.strip().startswith("#"))

    @staticmethod
    def analyze_file(path: Path) -> FileStat:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        defs = sum(isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) for n in ast.walk(tree))
        branches = sum(isinstance(n, _BRANCH_NODES) for n in ast.walk(tree))
        return FileStat(path, Analytics._code_lines(source), defs, branches)

    def analyze(self) -> list[AreaStat]:
        stats = []
        for area in self.areas:
            root = self.repo / area
            if not root.is_dir():
                continue
            area_stat = AreaStat(area)
            for py in sorted(root.rglob("*.py")):
                if "__pycache__" in py.parts:
                    continue
                area_stat.files.append(self.analyze_file(py))
            stats.append(area_stat)
        return stats

    def _test_lines(self) -> int:
        return sum(
            self._code_lines(p.read_text(encoding="utf-8"))
            for p in (self.repo / "tests").rglob("*.py")
            if "__pycache__" not in p.parts
        )

    def report(self, *, top_n: int = 10, flag_over: int | None = None) -> None:
        stats = self.analyze()
        log.info(f"{'area':<12}{'files':>7}{'code':>8}{'defs':>7}{'branch':>8}{'br/def':>8}")
        src_lines = 0
        for a in stats:
            bpd = a.branches / a.defs if a.defs else 0.0
            log.info(f"{a.name:<12}{len(a.files):>7}{a.code_lines:>8}{a.defs:>7}{a.branches:>8}{bpd:>8.2f}")
            src_lines += a.code_lines

        tests = self._test_lines()
        ratio = tests / src_lines if src_lines else 0.0
        log.info(f"\nsrc {src_lines} : test {tests}  (ratio {ratio:.2f} test lines per src line)")

        files = sorted((f for a in stats for f in a.files), key=lambda f: f.code_lines, reverse=True)
        log.info(f"\ntop {top_n} largest (code lines):")
        for f in files[:top_n]:
            log.info(f"  {f.code_lines:>5}  {f.branches:>4}br {f.defs:>3}def  {f.path.relative_to(self.repo)}")

        if flag_over is not None:
            over = [f for f in files if f.code_lines > flag_over]
            log.info(f"\n{len(over)} file(s) over {flag_over} code lines:")
            for f in over:
                log.info(f"  {f.code_lines:>5}  {f.path.relative_to(self.repo)}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="per-area code size + complexity-per-def explorer")
    ap.add_argument("--areas", nargs="+", default=_AREAS, help="dirs to scan (default: src)")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--flag-over", type=int, default=None, help="list files above this code-line budget")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent  # devtools/analytics.py -> repo root
    Analytics(repo, args.areas).report(top_n=args.top_n, flag_over=args.flag_over)


if __name__ == "__main__":
    main()
