"""LCOM cohesion detector (bd cardiac-seg-2r6f): Chidamber-Kemerer's Lack of Cohesion of Methods, the
LCOM4 (Hitz-Montazeri) variant — a class's methods form a graph (edge = two methods share an instance
field or one calls the other); the number of CONNECTED COMPONENTS is LCOM4. LCOM4 == 1 is cohesive;
LCOM4 >= 2 means the methods split into groups touching disjoint state = the class is really N classes
fused (how `Splits` turned out — extract each component). Formalizes what `state_candidates` eyeballed:
where that ranks namespace bags by shared params, this ranks CONSTRUCTED classes by internal split.

`__init__` and static/class methods are excluded from the graph (the constructor wires every field, so
counting it would mask every split; statics hold no instance state). Only classes with >= 2 such methods
are scored.

RAW LCOM4 over-fires on a strategy-pattern codebase (which this repo mandates): a polymorphic INTERFACE
(adapters' cases/load_ed_es/meta, a Stage's is_done/run) and a null-object stub legitimately have methods
that share no field — that's the pattern, not a god-class. So two classes are excluded: (a) IMPLS — a
class subclassing a domain base (its method split just mirrors the interface, not real fusion); (b)
ABSTRACT/STUB — every method is a trivial body (`...`/pass/`raise NotImplementedError`/docstring). What
survives is a CONCRETE, STATEFUL class whose behaviour genuinely splits into disjoint-state groups (how
`Splits` looked before ModelSplit was extracted).

    python -m devtools.lcom core cardioseg
"""
from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path

from core.obs import Obs

log = logging.getLogger("cardioseg.devtools.lcom")

_SKIP = {"__init__", "__new__", "__post_init__"}
_MIN_METHODS = 2          # LCOM is undefined for < 2 methods
_MIN_SPLIT = 2            # LCOM4 >= this = disjoint state groups = split candidate
# bases that don't make a class a domain interface-impl (so it's still eligible for scoring)
_BUILTIN_BASES = {"object", "BaseModel", "Enum", "IntEnum", "StrEnum", "Protocol", "ABC", "ABCMeta",
                  "TypedDict", "NamedTuple", "Exception"}


def _is_static(fn: ast.FunctionDef) -> bool:
    return any(isinstance(d, ast.Name) and d.id in ("staticmethod", "classmethod") for d in fn.decorator_list)


def _instance_methods(cls: ast.ClassDef) -> list[ast.FunctionDef]:
    """Behaviour methods that carry instance state — excludes __init__ and static/class methods."""
    return [n for n in cls.body
            if isinstance(n, ast.FunctionDef) and not _is_static(n) and n.name not in _SKIP]


def _base_names(cls: ast.ClassDef) -> set[str]:
    return ({b.id for b in cls.bases if isinstance(b, ast.Name)}
            | {b.attr for b in cls.bases if isinstance(b, ast.Attribute)})


def _is_impl(cls: ast.ClassDef) -> bool:
    """Subclasses a DOMAIN base -> a polymorphic interface impl; its method split mirrors the contract."""
    return bool(_base_names(cls) - _BUILTIN_BASES)


def _is_abstract(cls: ast.ClassDef) -> bool:
    """An ABC / interface base (subclasses ABC, or any method is @abstractmethod) — the contract, not a
    concrete class; its methods are meant to be independent facets."""
    if "ABC" in _base_names(cls):
        return True
    return any(isinstance(m, ast.FunctionDef)
               and any(isinstance(d, ast.Name) and d.id == "abstractmethod" for d in m.decorator_list)
               for m in cls.body)


def _is_trivial(fn: ast.FunctionDef) -> bool:
    """A stub/abstract body: only a docstring, `...`, `pass`, or `raise NotImplementedError`."""
    for n in fn.body:
        if isinstance(n, ast.Pass) or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)):
            continue
        if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call) and isinstance(n.exc.func, ast.Name) \
                and n.exc.func.id == "NotImplementedError":
            continue
        if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Name) and n.exc.id == "NotImplementedError":
            continue
        return False
    return True


def _self_names(fn: ast.FunctionDef) -> set[str]:
    """Every `self.X` name referenced in a method (fields AND sibling-method refs)."""
    return {n.attr for n in ast.walk(fn)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self"}


def _linked(a: ast.FunctionDef, b: ast.FunctionDef, names_a: set[str], names_b: set[str]) -> bool:
    """Two methods are connected if they share a self.X reference, or one calls/reads the other."""
    return bool(names_a & names_b) or b.name in names_a or a.name in names_b


def lcom4(cls: ast.ClassDef) -> tuple[int, list[list[str]]]:
    """(LCOM4, components) for a class — the count of connected method groups and their member names.
    LCOM4 <= 1 (0 methods, or all connected) is cohesive; >= 2 means disjoint state groups."""
    methods = _instance_methods(cls)
    names = [_self_names(m) for m in methods]
    parent = list(range(len(methods)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            if _linked(methods[i], methods[j], names[i], names[j]):
                parent[find(i)] = find(j)

    groups: dict[int, list[str]] = {}
    for i, m in enumerate(methods):
        groups.setdefault(find(i), []).append(m.name)
    comps = sorted(groups.values(), key=len, reverse=True)
    return len(comps), comps


def _is_split_candidate(cls: ast.ClassDef) -> tuple[int, list[list[str]]] | None:
    """(lcom4, components) if `cls` is a concrete stateful class that genuinely splits, else None."""
    methods = _instance_methods(cls)
    if (len(methods) < _MIN_METHODS or _is_impl(cls) or _is_abstract(cls)
            or all(_is_trivial(m) for m in methods)):
        return None
    score, comps = lcom4(cls)
    return (score, comps) if score >= _MIN_SPLIT else None


def scan(packages: list[str]) -> list[tuple[int, str, str, list[list[str]]]]:
    """Ranked (lcom4, class, file, components) for concrete stateful classes that genuinely split
    (LCOM4 >= 2, excluding interface impls + abstract/stub classes)."""
    rows = []
    for pkg in packages:
        for path in sorted(Path(pkg).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and (hit := _is_split_candidate(node)):
                    rows.append((hit[0], node.name, str(path), hit[1]))
    return sorted(rows, reverse=True)


def report(rows: list[tuple[int, str, str, list[list[str]]]]) -> str:
    """Ranked table: LCOM4, class, file, then the disjoint method groups (each = an extractable object)."""
    lines = [f"{'lcom4':>5}  {'class':24} file"]
    for score, name, path, comps in rows:
        lines.append(f"{score:>5}  {name:24} {path}")
        for comp in comps:
            lines.append(f"{'':>7}  · {{{', '.join(sorted(comp))}}}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(prog="python -m devtools.lcom",
                                 description="rank classes by LCOM4 cohesion (>=2 = split candidate)")
    ap.add_argument("packages", nargs="*", default=["core", "cardioseg"])
    args = ap.parse_args()
    Obs.setup()
    rows = scan(args.packages)
    log.info("%d low-cohesion classes (LCOM4>=2)\n%s", len(rows), report(rows))


if __name__ == "__main__":
    main()
