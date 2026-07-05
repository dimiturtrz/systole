"""TestSet — coded frozen eval targets. Locks the source resolution + the content-hash drift guard."""
import polars as pl
import pytest

from core.data.testsets import TestSet, TESTSETS, MATRIX_TESTSETS

V = pl.col


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
