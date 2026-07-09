"""core.config tests: the Config path resolver + the config-owned flagship ref.

Config resolves the ONE data root (env CARDIAC_DATA -> paths.yaml `data` -> repo-relative
fallback) and delegates the OS×path adaptation to core.paths.resolve_data_root (the full
matrix incl. the Windows-on-POSIX leak guard is tested in test_paths). Here: env is honored,
the resolver leaves an OS-native path unchanged, and per-kind roots derive from the one root.
"""
import os

import core.config as cfg
from core.config import FLAGSHIP_REF


def test_root_delegates_to_resolver(monkeypatch):
    """config._root reads CARDIAC_DATA then adapts it via core.paths.resolve_data_root — env picked
    up + an OS-native path left unchanged (no crash)."""
    monkeypatch.setenv("CARDIAC_DATA", "/data/foo/bar")
    assert "foo/bar" in cfg.Config._root()


def test_data_root_derives_kind_from_root(monkeypatch):
    """data_root('kind') = <root>/<kind> by default (no paths.yaml override for the kind)."""
    monkeypatch.setenv("CARDIAC_DATA", "/data/foo")
    raw = cfg.Config.data_root("raw")
    proc = cfg.Config.data_root("processed")
    assert os.path.basename(raw) == "raw" and os.path.basename(proc) == "processed"
    assert os.path.dirname(raw) == os.path.dirname(proc)   # same parent root


def test_flagship_is_registry_ref():
    """Flagship resolves from the mlflow registry (production alias), not a hardcoded dir. Don't
    invoke resolve here (that hits mlflow) — just guard the ref."""
    assert FLAGSHIP_REF == "production"
