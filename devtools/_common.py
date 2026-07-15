"""Shared devtools primitives: the package-tree AST walk and the pyproject `[tool.*]` reader, factored
out of the individual engines. Every scanning engine previously re-globbed `*.py` + re-parsed each file,
and every config-driven one re-opened pyproject — one home each for 'iterate the source' and 'read my
config section', so the walk/read logic lives in exactly one place (DRY across the eight engines)."""

from __future__ import annotations

import ast
import tomllib
from collections.abc import Iterator
from pathlib import Path


class Trees:
    """The source-tree AST walk shared by every engine that scans packages (one glob+parse home)."""

    def __init__(self, packages: list[str]) -> None:
        self.packages = packages

    def walk(self) -> Iterator[tuple[Path, ast.Module]]:
        """(path, parsed-AST) for every `*.py` under each root package, sorted within a package."""
        for pkg in self.packages:
            for path in sorted(Path(pkg).rglob("*.py")):
                yield path, ast.parse(path.read_text(encoding="utf-8"))

    def files(self) -> list[Path]:
        """Every `*.py` path under the root packages (no parse) — for line-count / path-only scans."""
        return [p for pkg in self.packages for p in sorted(Path(pkg).rglob("*.py"))]


class Pyproject:
    """Reader for a `[tool.<section>]` table from pyproject.toml — the shared config-load primitive."""

    @staticmethod
    def tool_section(section: str, pyproject: str = "pyproject.toml") -> dict:
        """The `[tool.<section>]` table (empty dict if the file or section is absent). One config home."""
        p = Path(pyproject)
        if not p.exists():
            return {}
        return tomllib.loads(p.read_text(encoding="utf-8")).get("tool", {}).get(section, {})
