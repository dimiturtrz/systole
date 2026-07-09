"""static_all split family — static_main's frozen TEST + ACDC val, but the loaded cloud adds SCD so the
complement TRAIN also includes SCD (LV-only, RV masked via partial-label). Resolves against a mocked
cloud (lock neutralized) to a valid static partition; SCD lands in train, never test/val."""
import dataclasses

import polars as pl

from core.data.ingest.source import StaticSource
from core.data.ingest.split import SplitResolver
from core.data.ingest.splits import Splits, static_all

V = pl.col


def _cloud():
    rows = [("acdc", "a1", "Siemens", True), ("mnm2", "p1", "Philips", True),
            ("mnms1", "g1", "GE", True), ("mnms1", "c1", "Canon", True),
            ("cmrxmotion", "m1", "Siemens", True), ("scd", "sc1", "GE", True)]   # SCD in the loaded cloud
    return pl.DataFrame([{"dataset": d, "subject_id": s, "vendor": v, "labelled": lab,
                          "path": f"/s/{d}/{s}.npz"} for d, s, v, lab in rows])


def _unlock(monkeypatch):
    monkeypatch.setattr(static_all, "STATIC_MAIN_TEST",
                        dataclasses.replace(static_all.STATIC_MAIN_TEST, lock=""))


def test_family_declares_scd_source():
    fam = Splits.load_split("static_all")
    assert "static_all" in Splits.list_splits()
    assert "scd" in fam.sources                               # loads SCD (not in the default seg cloud)
    assert fam.versions["1.0.0"].train is None                # train = complement


def test_resolves_scd_into_train_only(monkeypatch):
    _unlock(monkeypatch)
    r = SplitResolver.resolve(Splits.load_split("static_all"), _cloud())
    assert all(isinstance(s, StaticSource) for s in (r.train, r.val, r.test))
    assert set(r.test.subjects()) == {("mnms1", "g1"), ("mnms1", "c1"), ("cmrxmotion", "m1")}  # scoped to seg, no SCD
    assert set(r.val.subjects()) == {("acdc", "a1")}
    assert ("scd", "sc1") in set(r.train.subjects())          # SCD joins TRAIN (new real centre)
    assert set(r.train.subjects()) == {("mnm2", "p1"), ("scd", "sc1")}  # Philips seg + SCD, no test/val leak
