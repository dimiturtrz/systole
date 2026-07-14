"""The coverage-`omit` set as a shared devtools primitive — the 'not logic' shells (runners / adapters /
GPU / download / viz glue) the arch-fitness gates treat specially. The test-mirror rule (graph.py: a
shell needs no mirror test) reads it so the gate agrees with the coverage gate on what a shell is."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


def coverage_omit(pyproject: str = "pyproject.toml") -> list[str]:
    """The `[tool.coverage.run] omit` globs from pyproject (empty if the file/section is absent)."""
    p = Path(pyproject)
    if not p.exists():
        return []
    return (
        tomllib.loads(p.read_text(encoding="utf-8")).get("tool", {}).get("coverage", {}).get("run", {}).get("omit", [])
    )


def matches_omit(path: str, patterns: list[str]) -> bool:
    """True if `path` matches any coverage-omit glob (`*` = one segment, `**` = across segments)."""
    path = path.replace("\\", "/")
    for pat in patterns:
        rx = "^" + re.escape(pat.replace("\\", "/")).replace(r"\*\*", ".*").replace(r"\*", "[^/]*") + "$"
        if re.match(rx, path):
            return True
    return False
