"""synth_main split family — the magnum-opus arm: train = DynamicSource (zero real pixels), val = ACDC
real (held from test), test = SYNTH_MAIN_TEST (all seg real minus ACDC, locked). Resolves against a
mocked cloud (lock neutralized) to a valid mixed partition (dynamic train, static val/test)."""
import dataclasses

import polars as pl

from core.data.dynamic.source import DynamicSource
from core.data.ingest.source import StaticSource
from core.data.ingest.split import SplitResolver
from core.data.ingest.splits import Splits, synth_main
from core.data.ingest.splits.synth_main import POOL
from core.data.ingest.testsets import SYNTH_MAIN_TEST

V = pl.col


def _cloud():
    rows = [("acdc", "a1", "Siemens", True), ("mnm2", "p1", "Philips", True),
            ("mnms1", "g1", "GE", True), ("cmrxmotion", "m1", "Siemens", True)]
    return pl.DataFrame([{"dataset": d, "subject_id": s, "vendor": v, "labelled": lab,
                          "path": f"/s/{d}/{s}.npz"} for d, s, v, lab in rows])


def _unlock(monkeypatch):
    monkeypatch.setattr(synth_main, "SYNTH_MAIN_TEST",
                        dataclasses.replace(synth_main.SYNTH_MAIN_TEST, lock=""))


def test_registered_with_locked_testset_and_named_pool():
    assert "synth_main" in Splits.list_splits()
    d = Splits.load_split("synth_main").versions["1.0.0"]
    assert d.train is not None                                 # explicit dynamic train (not complement)
    assert POOL                                                # named constant, not a bare literal
    assert SYNTH_MAIN_TEST.lock.startswith("sha256:") and len(SYNTH_MAIN_TEST.lock) > 20


def test_resolves_to_dynamic_train_static_val_test(monkeypatch):
    _unlock(monkeypatch)
    r = SplitResolver.resolve(Splits.load_split("synth_main"), _cloud())
    assert isinstance(r.train, DynamicSource)                  # synth train, zero real pixels
    assert r.train.provenance()["kind"] == "dynamic" and r.train.bg.mode == "procedural"
    assert isinstance(r.val, StaticSource) and set(r.val.subjects()) == {("acdc", "a1")}   # real val
    assert isinstance(r.test, StaticSource)
    assert set(r.test.subjects()) == {("mnm2", "p1"), ("mnms1", "g1"), ("cmrxmotion", "m1")}  # seg minus acdc
    assert ("acdc", "a1") not in set(r.test.subjects())        # val held out of test
