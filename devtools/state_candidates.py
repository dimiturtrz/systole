"""Namespace-class detector (bd cardiac-seg-h7vy.4): the y2fi in-a-class migration folded free funcs as
`@staticmethod`, so a class whose methods all thread the SAME param(s) is a namespace bag with LATENT
instance state — that shared param belongs in `__init__`, not every signature. This ranks those
promotion candidates so the cleanup is evidence-driven, not eyeballed.

Signal: for each class with >=2 staticmethods and no `__init__`, count how many methods share each param
name; a param carried by >=half the methods (and >=2) is latent state. Score = sum of those shared
counts, so a class where many methods thread many common params ranks highest. Dispatcher command classes
(`add_args`+`run`) and already-stateful classes (have `__init__`) are skipped.

    python -m devtools.state_candidates core cardioseg
"""
from __future__ import annotations

import argparse
import ast
import logging
from collections import Counter
from pathlib import Path

from core.obs import Obs

log = logging.getLogger("devtools.state_candidates")

_SELF = {"self", "cls"}
_MIN_METHODS = 2


def _staticmethods(cls: ast.ClassDef) -> list[ast.FunctionDef]:
    """The @staticmethod-decorated defs of a class (the migrated free funcs)."""
    return [n for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in n.decorator_list)]


def _params(fn: ast.FunctionDef) -> set[str]:
    """Positional + keyword-only param names, minus self/cls."""
    args = fn.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    return {n for n in names if n not in _SELF}


def _is_command(names: set[str]) -> bool:
    """A CLI dispatcher command class (add_args + run) — legitimately stateless, skip."""
    return {"add_args", "run"} <= names


def _is_pydantic_config(cls: ast.ClassDef) -> bool:
    """A pydantic config class (base `BaseModel`) — its shared params are declared FIELDS, so its
    staticmethods are apply/build over an ALREADY-stateful config, not latent instance state. Skip."""
    return any(isinstance(b, ast.Name) and b.id == "BaseModel" for b in cls.bases)


def shared_state(cls: ast.ClassDef) -> dict[str, int]:
    """Param names carried by >=2 and >=half of a class's staticmethods = its latent instance state.
    Empty if the class has an __init__ (already stateful), is a pydantic config, is a CLI command, or
    has too few methods."""
    if _is_pydantic_config(cls):
        return {}
    if any(isinstance(n, ast.FunctionDef) and n.name == "__init__" for n in cls.body):
        return {}
    methods = _staticmethods(cls)
    if len(methods) < _MIN_METHODS or _is_command({m.name for m in methods}):
        return {}
    freq: Counter[str] = Counter()
    for m in methods:
        freq.update(_params(m))
    threshold = max(2, len(methods) / 2)
    return {p: c for p, c in freq.items() if c >= threshold}


def analyze(path: Path) -> list[tuple[int, str, int, dict[str, int]]]:
    """(score, class_name, n_methods, shared) per candidate class in one file, score = sum shared counts."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            shared = shared_state(node)
            if shared:
                out.append((sum(shared.values()), node.name, len(_staticmethods(node)), shared))
    return out


def scan(packages: list[str]) -> list[tuple[int, str, str, int, dict[str, int]]]:
    """Ranked (score, class, file, n_methods, shared) across every .py under the packages, high score first."""
    rows = []
    for pkg in packages:
        for path in sorted(Path(pkg).rglob("*.py")):
            rows.extend((sc, name, str(path), n, sh) for sc, name, n, sh in analyze(path))
    return sorted(rows, reverse=True)


def report(rows: list[tuple[int, str, str, int, dict[str, int]]]) -> str:
    """Ranked table: score, class, method-count, the shared params (count), file."""
    lines = [f"{'score':>5}  {'class':22} {'meth':>4}  shared-state (methods-sharing)      file"]
    for score, name, path, n, shared in rows:
        sh = ", ".join(f"{p}×{c}" for p, c in sorted(shared.items(), key=lambda kv: -kv[1]))
        lines.append(f"{score:>5}  {name:22} {n:>4}  {sh:35} {path}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(prog="python -m devtools.state_candidates",
                                 description="rank namespace-classes by latent shared instance state")
    ap.add_argument("packages", nargs="*", default=["core", "cardioseg"],
                    help="package dirs to scan (default: core cardioseg)")
    args = ap.parse_args()
    Obs.setup()
    rows = scan(args.packages)
    log.info("%d promotion candidates\n%s", len(rows), report(rows))


if __name__ == "__main__":
    main()
