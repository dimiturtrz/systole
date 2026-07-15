"""Namespace-class detector: a class whose methods (folded from free funcs as `@staticmethod`) all thread
the SAME param(s) is a namespace bag with LATENT instance state — that shared param belongs in `__init__`,
not every signature. This ranks those promotion candidates so the cleanup is evidence-driven, not
eyeballed.

Signal: for each class with >=2 staticmethods and no `__init__`, count how many methods share each param
name; a param carried by >=half the methods (and >=2) is latent state. Score = sum of those shared
counts, so a class where many methods thread many common params ranks highest. Dispatcher command classes
(`add_args`+`run`) and already-stateful classes (have `__init__`) are skipped.

    python -m devtools.state_candidates src mypackage
"""

from __future__ import annotations

import argparse
import ast
import logging
from collections import Counter

from devtools._common import Trees
from devtools.omit import Omit

log = logging.getLogger("devtools.state_candidates")

_SELF = {"self", "cls"}
_MIN_METHODS = 2


class StateCandidates:
    """Rank namespace-classes by latent shared instance state across the scanned packages."""

    def __init__(self, packages: list[str]) -> None:
        self.packages = packages

    @staticmethod
    def _staticmethods(cls: ast.ClassDef) -> list[ast.FunctionDef]:
        """The @staticmethod-decorated defs of a class (the migrated free funcs)."""
        return [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in n.decorator_list)
        ]

    @staticmethod
    def _params(fn: ast.FunctionDef) -> set[str]:
        """Positional + keyword-only param names, minus self/cls."""
        args = fn.args
        names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
        return {n for n in names if n not in _SELF}

    @staticmethod
    def _is_command(names: set[str]) -> bool:
        """A CLI dispatcher command class (add_args + run) — legitimately stateless, skip."""
        return {"add_args", "run"} <= names

    @staticmethod
    def _is_pydantic_config(cls: ast.ClassDef) -> bool:
        """A pydantic config class (base `BaseModel`) — its shared params are declared FIELDS, so its
        staticmethods are apply/build over an ALREADY-stateful config, not latent instance state. Skip."""
        return any(isinstance(b, ast.Name) and b.id == "BaseModel" for b in cls.bases)

    @staticmethod
    def _is_autograd_function(cls: ast.ClassDef) -> bool:
        """A torch.autograd.Function (base `Function` / `autograd.Function`): forward/backward thread `ctx`
        by the framework API — a contract, not promotable instance state. Skip like a pydantic config."""
        return any(
            (isinstance(b, ast.Name) and b.id == "Function") or (isinstance(b, ast.Attribute) and b.attr == "Function")
            for b in cls.bases
        )

    @staticmethod
    def shared_state(cls: ast.ClassDef) -> dict[str, int]:
        """Param names carried by >=2 and >=half of a class's staticmethods = its latent instance state.
        Empty if the class has an __init__ (already stateful), is a pydantic config, an autograd.Function, a
        CLI command, or has too few methods."""
        if StateCandidates._is_pydantic_config(cls) or StateCandidates._is_autograd_function(cls):
            return {}
        if any(isinstance(n, ast.FunctionDef) and n.name == "__init__" for n in cls.body):
            return {}
        methods = StateCandidates._staticmethods(cls)
        if len(methods) < _MIN_METHODS or StateCandidates._is_command({m.name for m in methods}):
            return {}
        freq: Counter[str] = Counter()
        for m in methods:
            freq.update(StateCandidates._params(m))
        threshold = max(2, len(methods) / 2)
        return {p: c for p, c in freq.items() if c >= threshold}

    @staticmethod
    def _analyze(tree: ast.Module) -> list[tuple[int, str, int, dict[str, int]]]:
        """(score, class_name, n_methods, shared) per candidate class in one tree, score = sum shared counts."""
        out = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                shared = StateCandidates.shared_state(node)
                if shared:
                    out.append((sum(shared.values()), node.name, len(StateCandidates._staticmethods(node)), shared))
        return out

    def scan(self) -> list[tuple[int, str, str, int, dict[str, int]]]:
        """Ranked (score, class, file, n_methods, shared) across every .py under the packages, high score
        first. Coverage-omitted shells (`[tool.coverage] omit`, via omit.py — the same 'not logic' set the
        test-mirror gate exempts) are skipped: a runner/adapter's shared params are its data, not identity."""
        omit = Omit.coverage_omit()
        rows = []
        for path, tree in Trees(self.packages).walk():
            if Omit.matches_omit(str(path), omit):
                continue
            rows.extend((sc, name, str(path), n, sh) for sc, name, n, sh in self._analyze(tree))
        return sorted(rows, reverse=True)

    @staticmethod
    def report(rows: list[tuple[int, str, str, int, dict[str, int]]]) -> str:
        """Ranked table: score, class, method-count, the shared params (count), file."""
        lines = [f"{'score':>5}  {'class':22} {'meth':>4}  shared-state (methods-sharing)      file"]
        for score, name, path, n, shared in rows:
            sh = ", ".join(f"{p}×{c}" for p, c in sorted(shared.items(), key=lambda kv: -kv[1]))
            lines.append(f"{score:>5}  {name:22} {n:>4}  {sh:35} {path}")
        return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        prog="python -m devtools.state_candidates", description="rank namespace-classes by latent shared instance state"
    )
    ap.add_argument("packages", nargs="+", help="package dirs to scan (>=1 required, no 'src' fallback)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = StateCandidates(args.packages).scan()
    log.info("%d promotion candidates\n%s", len(rows), StateCandidates.report(rows))


if __name__ == "__main__":
    main()
