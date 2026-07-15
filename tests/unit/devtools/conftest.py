"""Shared fixtures for the devtools engine tests — one home for the AST/package-writing helpers reused
across the per-engine test modules (DRY: the snippet-to-ClassDef and write-a-scan-package helpers)."""

import ast

import pytest


@pytest.fixture
def make_cls():
    """Return a helper: the first class defined in a source snippet, as an `ast.ClassDef`."""

    def _make(src: str) -> ast.ClassDef:
        return next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef))

    return _make


@pytest.fixture
def write_pkg():
    """Return a helper: write a one-module package under `root` and return its path (for scan tests)."""

    def _write(root, name: str, source: str) -> str:
        pkg = root / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "mod.py").write_text(source)
        return str(pkg)

    return _write
