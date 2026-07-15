"""Shape-contract coverage gate (ML domain): a public boundary function whose parameter or return is an
ARRAY/TENSOR type must carry a jaxtyping annotation, so the shape is a CHECKED contract and not a silent
assumption. The mechanical ratchet behind a codebase-wide shape rollout — it SURFACES bare-array
boundaries and the author fixes each by giving it a jaxtyping type (a dtype-only `"..."` is the honest
answer for a shape-agnostic reduction).

What counts as an array annotation: `np.ndarray` / `numpy.ndarray`, `torch.Tensor` / `Tensor`, or a
repo's array aliases (`[tool.shape_contracts] array_aliases` — e.g. `Volume`/`Mask`/`Image`). What
SATISFIES the contract: a jaxtyping subscript (`Float`/`Int`/`Integer`/`Bool`/`Shaped`/… `[array, "…"]`).
A bare array annotation (or an array alias) on a public method is flagged.

Scope: only PUBLIC methods (name not underscore-prefixed) — the boundaries. Private helpers, dunders, and
CLI `add_args`/`run` handlers are exempt (interior / framework signatures). Ships ADVISORY (report-only,
exit 0). A repo opts into the blocking ratchet with `--assert` once its tree is clean — a new bare-array
boundary then fails the merge.

    python -m devtools.shape_contracts <packages>            # advisory report
    python -m devtools.shape_contracts <packages> --assert   # blocking (exit 1 on any bare boundary)
"""

from __future__ import annotations

import argparse
import ast
import logging

from devtools._common import Pyproject, Trees

log = logging.getLogger("devtools.shape_contracts")

# The universal array types every ML repo shares. Repo-specific aliases (core.types names that also denote
# an array — Volume/Mask/…) are additive via [tool.shape_contracts] array_aliases, read below.
_ARRAY_NAMES = {"ndarray", "Tensor"}
_JAXTYPING = {"Float", "Int", "Integer", "UInt", "UInt8", "Bool", "Shaped", "Num", "Inexact", "Complex"}
_EXEMPT = {"add_args", "run"}  # CLI dispatcher handlers (framework signature, args is a Namespace)


class ShapeContracts:
    """Flag public array/tensor boundaries lacking a jaxtyping shape across the scanned packages."""

    def __init__(self, packages: list[str]) -> None:
        self.packages = packages

    @staticmethod
    def array_names(pyproject: str = "pyproject.toml") -> set[str]:
        """The array-type names to flag: the builtin `ndarray`/`Tensor` plus the repo's `array_aliases` slot."""
        aliases = Pyproject.tool_section("shape_contracts", pyproject).get("array_aliases", [])
        return _ARRAY_NAMES | set(aliases)

    @staticmethod
    def _is_array_anno(node: ast.expr | None, names: set[str]) -> bool:
        """True if the annotation names a bare array type (np.ndarray / Tensor / an array alias) — the thing
        that must instead carry a jaxtyping shape. Looks through `X | None` unions."""
        if node is None:
            return False
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):  # T | None
            return ShapeContracts._is_array_anno(node.left, names) or ShapeContracts._is_array_anno(node.right, names)
        if isinstance(node, ast.Attribute):  # np.ndarray / torch.Tensor
            return node.attr in names
        if isinstance(node, ast.Name):  # Tensor / Volume / Mask …
            return node.id in names
        return False

    @staticmethod
    def _is_jaxtyping_anno(node: ast.expr | None) -> bool:
        """True if the annotation is a jaxtyping subscript (`Float[array, "…"]`) — a satisfied shape contract.
        Looks through `X | None` unions so an optional shaped tensor still counts."""
        if node is None:
            return False
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return ShapeContracts._is_jaxtyping_anno(node.left) or ShapeContracts._is_jaxtyping_anno(node.right)
        if isinstance(node, ast.Subscript):
            head = node.value
            return isinstance(head, ast.Name) and head.id in _JAXTYPING
        return False

    @staticmethod
    def _annotations(fn: ast.FunctionDef) -> list[tuple[str, ast.expr | None]]:
        """(label, annotation) for every parameter + the return of a function."""
        a = fn.args
        params = [*a.posonlyargs, *a.args, *a.kwonlyargs]
        out: list[tuple[str, ast.expr | None]] = [(p.arg, p.annotation) for p in params]
        out.append(("->return", fn.returns))
        return out

    @staticmethod
    def _bare_array_slots(fn: ast.FunctionDef, names: set[str]) -> list[str]:
        """Param/return labels whose annotation is a bare array type without a jaxtyping shape."""
        return [
            label
            for label, anno in ShapeContracts._annotations(fn)
            if ShapeContracts._is_array_anno(anno, names) and not ShapeContracts._is_jaxtyping_anno(anno)
        ]

    @staticmethod
    def _public(fn: ast.FunctionDef) -> bool:
        return not fn.name.startswith("_") and fn.name not in _EXEMPT

    @staticmethod
    def _analyze(tree: ast.Module, names: set[str]) -> list[tuple[int, str, list[str]]]:
        """(lineno, qualname, bare-slots) for every public method in a tree with a bare-array boundary."""
        out = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for m in node.body:
                    if isinstance(m, ast.FunctionDef) and ShapeContracts._public(m):
                        slots = ShapeContracts._bare_array_slots(m, names)
                        if slots:
                            out.append((m.lineno, f"{node.name}.{m.name}", slots))
        return out

    def scan(self, names: set[str] | None = None) -> list[tuple[str, int, str, list[str]]]:
        """(file, lineno, qualname, slots) for every bare-array boundary across the packages."""
        if names is None:
            names = self.array_names()
        rows = []
        for path, tree in Trees(self.packages).walk():
            rows.extend((str(path), ln, name, slots) for ln, name, slots in self._analyze(tree, names))
        return rows

    @staticmethod
    def report(rows: list[tuple[str, int, str, list[str]]]) -> str:
        lines = [f"{len(rows)} bare-array boundaries (array-typed param/return without a jaxtyping shape):"]
        for path, ln, name, slots in rows:
            lines.append(f"  {path}:{ln}  {name}  [{', '.join(slots)}]")
        return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        prog="python -m devtools.shape_contracts",
        description="flag public array/tensor boundaries lacking a jaxtyping shape",
    )
    ap.add_argument(
        "packages",
        nargs="+",
        help="package dirs to scan (>=1 required — no-arg would scan nothing and pass --assert vacuously)",
    )
    ap.add_argument(
        "--assert",
        dest="assert_clean",
        action="store_true",
        help="exit 1 if any bare-array boundary remains (the blocking CI gate)",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows = ShapeContracts(args.packages).scan()
    log.info("%s", ShapeContracts.report(rows))
    if args.assert_clean and rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
