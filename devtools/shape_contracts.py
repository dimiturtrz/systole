"""Shape-contract coverage gate (bd cardiac-seg-m8xq): a public boundary function whose parameter or
return is an ARRAY/TENSOR type must carry a jaxtyping annotation, so the shape is a checked contract and
not a silent assumption. This is the mechanical ratchet behind the codebase-wide rollout (epic wk8m):
it SURFACES bare-array boundaries, the author fixes each by giving it a jaxtyping type (a dtype-only
`"..."` is the honest answer for a shape-agnostic reduction).

What counts as an array annotation: `np.ndarray` / `numpy.ndarray`, `torch.Tensor` / `Tensor`, or the
array aliases (`Volume`/`Mask`/`Image`/`Slice2D`/`Batch`). What SATISFIES the contract: a jaxtyping
subscript (`Float`/`Int`/`Integer`/`Bool`/`Shaped`/`UInt8`/… `[array, "…"]`). A bare array annotation
(or an array alias) on a public method is flagged.

Scope: only PUBLIC methods (name not underscore-prefixed) — the boundaries. Private helpers, dunders,
and CLI `add_args`/`run` handlers are exempt (interior / framework signatures). Advisory first (like the
class-shape smells); graduates to blocking once the tree is clean.

    python -m devtools.shape_contracts core cardioseg
"""
from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path

from core.obs import Obs

log = logging.getLogger("cardioseg.devtools.shape_contracts")

_ARRAY_NAMES = {"ndarray", "Tensor", "Volume", "Mask", "Image", "Slice2D", "Batch"}
_JAXTYPING = {"Float", "Int", "Integer", "UInt", "UInt8", "Bool", "Shaped", "Num", "Inexact", "Complex"}
_EXEMPT = {"add_args", "run"}          # CLI dispatcher handlers (framework signature, args is a Namespace)


def _is_array_anno(node: ast.expr | None) -> bool:
    """True if the annotation names a bare array type (np.ndarray / Tensor / an array alias) — the thing
    that must instead carry a jaxtyping shape. Looks through `X | None` unions."""
    if node is None:
        return False
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):     # T | None
        return _is_array_anno(node.left) or _is_array_anno(node.right)
    if isinstance(node, ast.Attribute):                                    # np.ndarray / torch.Tensor
        return node.attr in _ARRAY_NAMES
    if isinstance(node, ast.Name):                                         # Tensor / Volume / Mask …
        return node.id in _ARRAY_NAMES
    return False


def _is_jaxtyping_anno(node: ast.expr | None) -> bool:
    """True if the annotation is a jaxtyping subscript (`Float[array, "…"]`) — a satisfied shape contract.
    Looks through `X | None` unions so an optional shaped tensor still counts."""
    if node is None:
        return False
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_jaxtyping_anno(node.left) or _is_jaxtyping_anno(node.right)
    if isinstance(node, ast.Subscript):
        head = node.value
        return isinstance(head, ast.Name) and head.id in _JAXTYPING
    return False


def _annotations(fn: ast.FunctionDef) -> list[tuple[str, ast.expr | None]]:
    """(label, annotation) for every parameter + the return of a function."""
    a = fn.args
    params = [*a.posonlyargs, *a.args, *a.kwonlyargs]
    out: list[tuple[str, ast.expr | None]] = [(p.arg, p.annotation) for p in params]
    out.append(("->return", fn.returns))
    return out


def _bare_array_slots(fn: ast.FunctionDef) -> list[str]:
    """Param/return labels whose annotation is a bare array type without a jaxtyping shape."""
    return [label for label, anno in _annotations(fn)
            if _is_array_anno(anno) and not _is_jaxtyping_anno(anno)]


def _public(fn: ast.FunctionDef) -> bool:
    return not fn.name.startswith("_") and fn.name not in _EXEMPT


def analyze(path: Path) -> list[tuple[int, str, list[str]]]:
    """(lineno, qualname, bare-slots) for every public function in a file with a bare-array boundary."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and _public(m):
                    slots = _bare_array_slots(m)
                    if slots:
                        out.append((m.lineno, f"{node.name}.{m.name}", slots))
    return out


def scan(packages: list[str]) -> list[tuple[str, int, str, list[str]]]:
    """(file, lineno, qualname, slots) for every bare-array boundary across the packages."""
    rows = []
    for pkg in packages:
        for path in sorted(Path(pkg).rglob("*.py")):
            rows.extend((str(path), ln, name, slots) for ln, name, slots in analyze(path))
    return rows


def report(rows: list[tuple[str, int, str, list[str]]]) -> str:
    lines = [f"{len(rows)} bare-array boundaries (array-typed param/return without a jaxtyping shape):"]
    for path, ln, name, slots in rows:
        lines.append(f"  {path}:{ln}  {name}  [{', '.join(slots)}]")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(prog="python -m devtools.shape_contracts",
                                 description="flag public array/tensor boundaries lacking a jaxtyping shape")
    ap.add_argument("packages", nargs="*", default=["core", "cardioseg"],
                    help="package dirs to scan (default: core cardioseg)")
    args = ap.parse_args()
    Obs.setup()
    rows = scan(args.packages)
    log.info("%s", report(rows))


if __name__ == "__main__":
    main()
