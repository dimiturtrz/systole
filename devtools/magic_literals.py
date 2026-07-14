"""Magic-literal detector: the non-comparison, cross-file context ruff PLR2004 can't see.

PLR2004 flags a magic value only in a COMPARISON (`x == "foo"`), and by default it even ALLOWS strings
(`allow-magic-value-types` defaults to `["str", "bytes"]`). The same bare literal as a dict key
(`d["foo"]`), an argument (`load("foo")`), an assignment, or a return value is ungated, and ruff never
aggregates across files. This detector owns that gap — the non-comparison, cross-file-frequency signal —
and defers comparison operands to ruff. Two smells:

  1. **recurring string literal** — a short, identifier-shaped string that appears >= THRESHOLD times is
     domain vocabulary (a tag / kind / mode string) masquerading as a literal. It belongs in a `StrEnum`
     or a named constant: one source of truth, typo-proof, case-consistent.
  2. **repeated dict key-set** — a dict literal whose keys are all constant strings, whose exact key SET
     is built in >= 2 places, is an implicit record schema. Nothing enforces every construction site uses
     the same keys, so a typo/missing key drifts silently -> it wants a dataclass / TypedDict.

Frequency is a heuristic (some repeats are legitimately strings/dicts — column names, framework API vocab,
path segments; prose/messages have spaces so they're never tokens, and f-strings are JoinedStr not
Constant so they're never counted). Because a legitimate non-enum-able floor is real, the gate blocks as a
COUNT RATCHET (`--max-strings` / `--max-key-sets`), not at zero and without a per-token whitelist: the
current floor is frozen as a ceiling, and a NEW recurring literal pushes the count over and fails. Migrate
it to an enum, or raise the ceiling in the same commit with a reason. Omit the ceilings for the plain
advisory report (the scaffold ships it advisory; a repo opts into the ratchet by passing --max-* in CI).

    python -m devtools.magic_literals mypackage                        # advisory report
    python -m devtools.magic_literals mypackage --max-strings 12 --max-key-sets 3   # ratchet (CI)
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
import tomllib
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("devtools.magic_literals")

# A "vocabulary token": identifier-shaped, short, no spaces/paths/format — i.e. a domain value, not prose,
# a log message (has spaces), an f-string (a JoinedStr, not a Constant, so never counted), or a path.
_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]{1,24}$")
_STRING_THRESHOLD = 4  # a token appearing this many times is vocabulary, not incidental
_KEYSET_MIN_SIZE = 2  # a record needs >= 2 keys to be a schema worth a type
_KEYSET_MIN_SITES = 2  # a key-set built in this many places is a reused (drift-prone) schema
_STOP = {"store_true", "store_false", "append"}  # argparse action literals (framework, not domain vocab)


def _is_token(value: object) -> bool:
    """A string worth counting: an identifier-shaped VALUE token (not prose/path/message/framework)."""
    return isinstance(value, str) and value not in _STOP and bool(_TOKEN.match(value))


def _string_literals(tree: ast.AST) -> list[str]:
    """Identifier-shaped string constants in a VALUE position, EXCLUDING three contexts owned elsewhere:
      - dict KEYS + subscript indices (`d["field"]`) — schema FIELD refs, caught by the key-set smell;
      - COMPARISON operands (`x == "foo"`) — ruff PLR2004 owns those (with allow-magic-value-types=[]),
        so this detector doesn't double-flag them.
    What's left is the value/arg-position recurrence ruff can't see across files. Docstrings aren't tokens."""
    excluded = {id(k) for n in ast.walk(tree) if isinstance(n, ast.Dict) for k in n.keys if k is not None}
    excluded |= {
        id(n.slice)  # `d["field"]` subscript = a field ref
        for n in ast.walk(tree)
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
    }
    excluded |= {
        id(operand)  # x == "foo" -> ruff PLR2004
        for n in ast.walk(tree)
        if isinstance(n, ast.Compare)
        for operand in (n.left, *n.comparators)
        if isinstance(operand, ast.Constant)
    }
    return [
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and _is_token(n.value) and id(n) not in excluded
    ]


def _key_sets(tree: ast.AST) -> list[tuple[frozenset[str], int]]:
    """(key-set, lineno) for each dict literal whose keys are all constant strings (>= min size)."""
    out = []
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.Dict)
            and n.keys
            and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in n.keys)
        ):
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
    return sorted(((s, c) for s, c in counts.items() if c >= _STRING_THRESHOLD), key=lambda sc: -sc[1])


def scan_key_sets(packages: list[str]) -> list[tuple[int, tuple[str, ...], list[str]]]:
    """(n_sites, sorted-keys, locations) for constant-string-key dict schemas built in >= min sites."""
    sites: dict[frozenset[str], list[str]] = defaultdict(list)
    for path in _py_files(packages):
        for keys, lineno in _key_sets(ast.parse(path.read_text(encoding="utf-8"))):
            sites[keys].append(f"{path}:{lineno}")
    rows = [(len(locs), tuple(sorted(keys)), locs) for keys, locs in sites.items() if len(locs) >= _KEYSET_MIN_SITES]
    return sorted(rows, key=lambda r: -r[0])


def _py_files(packages: list[str]) -> list[Path]:
    return [p for pkg in packages for p in sorted(Path(pkg).rglob("*.py"))]


def report(strings: list[tuple[str, int]], key_sets: list[tuple[int, tuple[str, ...], list[str]]]) -> str:
    """The two ranked tables: recurring string tokens (StrEnum candidates) + repeated dict schemas."""
    lines = [f"{len(strings)} recurring string literals (>= {_STRING_THRESHOLD}x -> StrEnum/constant candidate):"]
    lines.extend(f"  {c:>3}x  {s!r}" for s, c in strings)
    lines.append(
        f"{len(key_sets)} repeated dict key-sets (>= {_KEYSET_MIN_SITES} sites -> dataclass/TypedDict candidate):"
    )
    for n, keys, locs in key_sets:
        lines.append(f"  {n:>3} sites  {{{', '.join(keys)}}}")
        lines.extend(f"           {loc}" for loc in locs)
    return "\n".join(lines)


def ratchet_ceilings(pyproject: str = "pyproject.toml") -> tuple[int | None, int | None]:
    """The `[tool.magic_literals] max_strings / max_key_sets` ceilings — the per-repo FACT that turns the
    advisory report into an ENFORCED ratchet (both None if the section/file is absent -> advisory)."""
    p = Path(pyproject)
    if not p.exists():
        return None, None
    cfg = tomllib.loads(p.read_text(encoding="utf-8")).get("tool", {}).get("magic_literals", {})
    return cfg.get("max_strings"), cfg.get("max_key_sets")


def check_ratchet(n_strings: int, n_key_sets: int, max_strings: int | None, max_key_sets: int | None) -> list[str]:
    """Ceiling breaches — the (count > ceiling) messages, empty when advisory (no ceilings) or under."""
    over = []
    if max_strings is not None and n_strings > max_strings:
        over.append(f"strings {n_strings} > {max_strings}")
    if max_key_sets is not None and n_key_sets > max_key_sets:
        over.append(f"key-sets {n_key_sets} > {max_key_sets}")
    return over


def main():
    ap = argparse.ArgumentParser(
        prog="python -m devtools.magic_literals",
        description="recurring string literals + repeated dict key-sets",
    )
    ap.add_argument("packages", nargs="+", help="package dirs to scan (>=1 required, no 'src' fallback)")
    ap.add_argument(
        "--max-strings",
        type=int,
        default=None,
        help="regression ratchet: exit 1 if the recurring-string count exceeds this ceiling",
    )
    ap.add_argument(
        "--max-key-sets",
        type=int,
        default=None,
        help="regression ratchet: exit 1 if the repeated-key-set count exceeds this ceiling",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Ceilings: CLI flag wins; else the [tool.magic_literals] FACT slot (the base gate — a fresh repo ships
    # 0/0, ratcheting up as its legitimate literal floor grows). Neither set -> advisory report, never bites.
    cfg_strings, cfg_key_sets = ratchet_ceilings()
    max_strings = args.max_strings if args.max_strings is not None else cfg_strings
    max_key_sets = args.max_key_sets if args.max_key_sets is not None else cfg_key_sets
    strings, key_sets = scan_strings(args.packages), scan_key_sets(args.packages)
    log.info("%s", report(strings, key_sets))
    # Ceilings freeze the legitimate non-enum-able floor: any NEW recurring literal pushes the count over
    # and fails the merge. Re-migrate it to an enum, or raise the ceiling in the SAME commit with a reason.
    if over := check_ratchet(len(strings), len(key_sets), max_strings, max_key_sets):
        log.error(
            "magic-literal ratchet exceeded (%s) — migrate the new literal or raise the ceiling with a reason",
            "; ".join(over),
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
