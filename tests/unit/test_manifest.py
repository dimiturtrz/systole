"""Frozen test-manifest unit tests — equivalence classes over freeze / load / resolve.

A manifest is the comparability contract (immutable, identity-only). We test: it round-trips; it
resolves frozen ids to CURRENT npz paths; it flags store drift (missing ids); and the guards that
protect the contract (immutability, valid task, non-empty match) hold.
"""
import polars as pl
import pytest

from core.data.static import manifest


def _meta(rows):
    """A minimal store-like frame: one row per subject with the columns freeze/resolve touch."""
    return pl.DataFrame(rows, schema={"dataset": pl.Utf8, "subject_id": pl.Utf8, "vendor": pl.Utf8,
                                      "labelled": pl.Boolean, "path": pl.Utf8})


@pytest.fixture(autouse=True)
def _tmp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest, "MANIFEST_DIR", tmp_path)
    return tmp_path


CLOUD = _meta([
    {"dataset": "acdc", "subject_id": "p1", "vendor": "Siemens", "labelled": True,  "path": "/s/acdc/p1.npz"},
    {"dataset": "acdc", "subject_id": "p2", "vendor": "Siemens", "labelled": True,  "path": "/s/acdc/p2.npz"},
    {"dataset": "mnms1", "subject_id": "c1", "vendor": "Canon",  "labelled": True,  "path": "/s/mnms1/c1.npz"},
    {"dataset": "mnms1", "subject_id": "u1", "vendor": "Canon",  "labelled": False, "path": "/s/mnms1/u1.npz"},
])


def test_freeze_then_load_roundtrips():
    manifest.freeze("v_canon", CLOUD, test_vendors=["Canon"], task="seg4", note="hi", created="2026-07-05")
    m = manifest.load("v_canon")
    assert m["task"] == "seg4" and m["created"] == "2026-07-05" and m["note"] == "hi"
    assert m["criteria"] == {"test_datasets": [], "test_vendors": ["Canon"]}
    assert m["subjects"] == [["mnms1", "c1"]]          # labelled Canon only (u1 dropped)
    assert m["snapshot"]["n"] == 1 and m["snapshot"]["store_total"] == 4


def test_resolve_full_when_store_has_all():
    manifest.freeze("d_acdc", CLOUD, test_datasets=["acdc"], task="seg4", created="2026-07-05")
    paths, missing = manifest.resolve_paths(manifest.load("d_acdc"), CLOUD)
    assert sorted(paths) == ["/s/acdc/p1.npz", "/s/acdc/p2.npz"] and missing == []


def test_resolve_flags_drift_as_missing():
    """A frozen id absent from the current store must be reported, never silently dropped."""
    manifest.freeze("d_acdc", CLOUD, test_datasets=["acdc"], task="seg4", created="2026-07-05")
    shrunk = CLOUD.filter(pl.col("subject_id") != "p2")     # p2 vanished from the store
    paths, missing = manifest.resolve_paths(manifest.load("d_acdc"), shrunk)
    assert paths == ["/s/acdc/p1.npz"] and missing == [["acdc", "p2"]]


def test_resolve_is_preprocessing_independent():
    """Same frozen ids, a store with DIFFERENT npz paths (rebaked preprocessing) -> new paths, same set."""
    manifest.freeze("d_acdc", CLOUD, test_datasets=["acdc"], task="seg4", created="2026-07-05")
    rebaked = CLOUD.with_columns(("/n4" + pl.col("path")).alias("path"))
    paths, missing = manifest.resolve_paths(manifest.load("d_acdc"), rebaked)
    assert sorted(paths) == ["/n4/s/acdc/p1.npz", "/n4/s/acdc/p2.npz"] and missing == []


def test_immutable_no_overwrite():
    manifest.freeze("v_canon", CLOUD, test_vendors=["Canon"], task="seg4", created="2026-07-05")
    with pytest.raises(FileExistsError):
        manifest.freeze("v_canon", CLOUD, test_vendors=["Canon"], task="seg4", created="2026-07-05")


def test_bad_task_rejected():
    with pytest.raises(ValueError):
        manifest.freeze("x", CLOUD, test_datasets=["acdc"], task="seg9", created="2026-07-05")


def test_empty_match_rejected():
    with pytest.raises(ValueError):
        manifest.freeze("x", CLOUD, test_vendors=["GE"], task="seg4", created="2026-07-05")
