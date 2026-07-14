"""Magic-literal detector (bd cardiac-seg-wir7): the untapped context PLR2004 misses.

PLR2004 flags a magic value only in a COMPARISON (`x == "acdc"`), and by default it even ALLOWS strings
(`allow-magic-value-types` defaults to `["str", "bytes"]`); un-silenced (bd cardiac-seg-1ln7) it owns the
comparison context. But the same bare literal as a dict key (`d["acdc"]`), an argument (`load("acdc")`),
an assignment, or a return value is ungated, and ruff never aggregates across files. This detector owns
that gap — the non-comparison, cross-file-frequency signal — and defers comparisons to ruff. Two smells:

  1. **recurring string literal** — a short, identifier-shaped string that appears >= THRESHOLD times is
     domain vocabulary (a vendor / phase / dataset / task tag) masquerading as a literal. It belongs in a
     `StrEnum` or a named constant: one source of truth, typo-proof, case-consistent. (This detector is
     what should have found the vendor/phase/dataset sets, by frequency, without knowing them up front.)
  2. **repeated dict key-set** — a dict literal whose keys are all constant strings, whose exact key SET
     is built in >= 2 places, is an implicit record schema. Nothing enforces every construction site uses
     the same keys, so a typo/missing key drifts silently -> it wants a dataclass / TypedDict.

Advisory only: frequency is a heuristic (some repeats are legitimately dicts/strings — polars rows, JSON,
prose is already filtered out). A regression radar, never a blocker.

    python -m devtools.magic_literals core cardioseg
"""
from __future__ import annotations

import argparse
import ast
import logging
import re
from collections import defaultdict
from pathlib import Path

from core.obs import Obs

log = logging.getLogger("cardioseg.devtools.magic_literals")

# A "vocabulary token": identifier-shaped, short, no spaces/paths/format — i.e. a domain value, not prose,
# a log message (has spaces), an f-string (a JoinedStr, not a Constant, so never counted), or a path.
_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{1,24}$")
_STRING_THRESHOLD = 4          # a token appearing this many times is vocabulary, not incidental
_KEYSET_MIN_SIZE = 2           # a record needs >= 2 keys to be a schema worth a type
_KEYSET_MIN_SITES = 2          # a key-set built in this many places is a reused (drift-prone) schema
_STOP = {"store_true", "store_false", "append"}   # argparse action literals (framework, not domain vocab)


def _is_token(value: object) -> bool:
    """A string worth counting: an identifier-shaped VALUE token (not prose/path/message/framework)."""
    return isinstance(value, str) and value not in _STOP and bool(_TOKEN.match(value))


def _string_literals(tree: ast.AST) -> list[str]:
    """Identifier-shaped string constants in a VALUE position, EXCLUDING three contexts owned elsewhere:
      - dict KEYS + subscript indices (`d["vendor"]`) — schema FIELD refs, caught by the key-set smell;
      - COMPARISON operands (`x == "GE"`) — ruff PLR2004 owns those (with allow-magic-value-types=[], bd
        cardiac-seg-1ln7), so this detector doesn't double-flag them.
    What's left is the value/arg-position recurrence ruff can't see across files. Docstrings aren't tokens."""
    excluded = {id(k) for n in ast.walk(tree) if isinstance(n, ast.Dict) for k in n.keys if k is not None}
    excluded |= {id(n.slice) for n in ast.walk(tree)               # `d["vendor"]` subscript = a field ref
                 if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)}
    excluded |= {id(operand) for n in ast.walk(tree) if isinstance(n, ast.Compare)  # x == "GE" -> ruff PLR2004
                 for operand in (n.left, *n.comparators) if isinstance(operand, ast.Constant)}
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and _is_token(n.value) and id(n) not in excluded]


def _key_sets(tree: ast.AST) -> list[tuple[frozenset[str], int]]:
    """(key-set, lineno) for each dict literal whose keys are all constant strings (>= min size)."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Dict) and n.keys and all(
                isinstance(k, ast.Constant) and isinstance(k.value, str) for k in n.keys):
            keys = frozenset(k.value for k in n.keys)  # type: ignore[union-attr]
            if len(keys) >= _KEYSET_MIN_SIZE:
                out.append((keys, n.lineno))
    return out


def scan_strings(packages: list[str]) -> list[tuple[str, int]]:
    """(token, count) for identifier-shaped strings appearing >= threshold across the packages, high first."""
    counts: dict[str, int] = defaultdict(int)
    for path in _py_files(packages):
        for value in _string_literals(ast.parse(path.read_text(encoding="utf-8"))):
            counts[value] += 1
    return sorted(((s, c) for s, c in counts.items() if c >= _STRING_THRESHOLD),
                  key=lambda sc: -sc[1])


def scan_key_sets(packages: list[str]) -> list[tuple[int, tuple[str, ...], list[str]]]:
    """(n_sites, sorted-keys, locations) for constant-string-key dict schemas built in >= min sites."""
    sites: dict[frozenset[str], list[str]] = defaultdict(list)
    for path in _py_files(packages):
        for keys, lineno in _key_sets(ast.parse(path.read_text(encoding="utf-8"))):
            sites[keys].append(f"{path}:{lineno}")
    rows = [(len(locs), tuple(sorted(keys)), locs) for keys, locs in sites.items()
            if len(locs) >= _KEYSET_MIN_SITES]
    return sorted(rows, key=lambda r: -r[0])


def _py_files(packages: list[str]) -> list[Path]:
    return [p for pkg in packages for p in sorted(Path(pkg).rglob("*.py"))]


def report(strings: list[tuple[str, int]], key_sets: list[tuple[int, tuple[str, ...], list[str]]]) -> str:
    """The two ranked tables: recurring string tokens (StrEnum candidates) + repeated dict schemas."""
    lines = [f"{len(strings)} recurring string literals (>= {_STRING_THRESHOLD}x -> StrEnum/constant candidate):"]
    lines.extend(f"  {c:>3}x  {s!r}" for s, c in strings)
    lines.append(f"{len(key_sets)} repeated dict key-sets (>= {_KEYSET_MIN_SITES} sites -> dataclass/TypedDict candidate):")
    for n, keys, locs in key_sets:
        lines.append(f"  {n:>3} sites  {{{', '.join(keys)}}}")
        lines.extend(f"           {loc}" for loc in locs)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(prog="python -m devtools.magic_literals",
                                 description="recurring string literals + repeated dict key-sets")
    ap.add_argument("packages", nargs="*", default=["core", "cardioseg"],
                    help="package dirs to scan (default: core cardioseg)")
    args = ap.parse_args()
    Obs.setup()
    log.info("%s", report(scan_strings(args.packages), scan_key_sets(args.packages)))


if __name__ == "__main__":
    main()
