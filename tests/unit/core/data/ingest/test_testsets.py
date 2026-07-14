"""TestSet — coded frozen eval targets. Locks the source resolution + the content-hash drift guard."""
import argparse
import logging

import polars as pl
import pytest

from core.data.ingest import testsets as T
from core.data.ingest.testsets import MATRIX_TESTSETS, TESTSETS, Task, TestSet, TestSets

V = pl.col


def test_task_values():
    assert (Task.SEG4, Task.SEG_LV) == ("seg4", "seg_lv")   # drop-in for the task-tag literals


def _cloud():
    rows = [("acdc", "s1", "Siemens", True), ("mnms1", "c1", "Canon", True),
            ("mnms1", "c2", "Canon", False)]                   # unlabelled -> excluded
    return pl.DataFrame([{"dataset": d, "subject_id": s, "vendor": v, "labelled": lab,
                          "path": f"/s/{d}/{s}.npz"} for d, s, v, lab in rows])


def test_source_filters_labelled_and_predicate():
    ts = TestSet("canon", "seg4", V("vendor") == "Canon")      # no lock -> no guard
    s = ts.source(_cloud())
    assert s.subjects() == [("mnms1", "c1")]                    # labelled Canon only (c2 unlabelled dropped)


def test_lock_guard_raises_on_drift():
    ts = TestSet("canon", "seg4", V("vendor") == "Canon", lock="sha256:deadbeef")
    with pytest.raises(ValueError, match="drifted"):
        ts.source(_cloud())


def test_lock_guard_passes_when_hash_matches():
    ts0 = TestSet("canon", "seg4", V("vendor") == "Canon")
    good = ts0.source(_cloud()).ids_hash()
    ts = TestSet("canon", "seg4", V("vendor") == "Canon", lock=good)
    assert ts.source(_cloud()).ids_hash() == good              # no raise


def test_registry_and_battery_populated():
    assert {"canon", "ge", "scd_lv", "static_main_test", "synth_main_test"} <= set(TESTSETS)
    assert all(ts.lock.startswith("sha256:") for ts in MATRIX_TESTSETS)   # all frozen


# --- TestSets.run: --freeze writes the lockfile; --check passes on match, exits 1 on drift ---

def test_run_freeze_writes_lockfile(tmp_path, monkeypatch):
    monkeypatch.setattr(T.store, "load", lambda sources: _cloud())
    lockfile = tmp_path / "locks.json"
    monkeypatch.setattr(T, "_LOCKFILE", lockfile)
    TestSets.run(argparse.Namespace(freeze=True))
    assert lockfile.exists() and "canon" in lockfile.read_text()


def test_run_check_ok_when_locks_match(monkeypatch, caplog):
    monkeypatch.setattr(T.store, "load", lambda sources: _cloud())
    monkeypatch.setattr(T, "_LOCKS", TestSets.compute_locks(_cloud()))   # lockfile == store
    with caplog.at_level(logging.INFO):
        TestSets.run(argparse.Namespace(freeze=False))
    assert any("locks match" in r.getMessage() for r in caplog.records)


def test_run_check_raises_on_drift(monkeypatch):
    monkeypatch.setattr(T.store, "load", lambda sources: _cloud())
    monkeypatch.setattr(T, "_LOCKS", {})                                 # empty lockfile -> everything drifts
    with pytest.raises(SystemExit):
        TestSets.run(argparse.Namespace(freeze=False))
