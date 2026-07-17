"""synth_fov split family — the whole-FOV MRXCAT arm: train = DynamicSource over the SSMxMRXCAT
composite pool painted with FovBg (mrxcat bg), val/test identical to synth_main so the zero-real
number is comparable. Resolves against a mocked cloud (lock neutralized) to dynamic train + static
val/test with the mrxcat background strategy."""
import dataclasses

import polars as pl

from core.data.dynamic.source import DynamicSource
from core.data.ingest.source import StaticSource
from core.data.ingest.split import SplitResolver
from core.data.ingest.splits import Splits, synth_fov
from core.data.ingest.splits.synth_fov import POOL
from core.data.ingest.testsets import SYNTH_MAIN_TEST

V = pl.col


def _cloud():
    rows = [("acdc", "a1", "Siemens", True), ("mnm2", "p1", "Philips", True),
            ("mnms1", "g1", "GE", True), ("cmrxmotion", "m1", "Siemens", True)]
    return pl.DataFrame([{"dataset": d, "subject_id": s, "vendor": v, "labelled": lab,
                          "path": f"/s/{d}/{s}.npz"} for d, s, v, lab in rows])


def _unlock(monkeypatch):
    monkeypatch.setattr(synth_fov, "SYNTH_MAIN_TEST",
                        dataclasses.replace(synth_fov.SYNTH_MAIN_TEST, lock=""))


def test_registered_with_named_pool():
    assert "synth_fov" in Splits.list_splits()
    d = Splits.load_split("synth_fov").versions["1.0.0"]
    assert d.train is not None
    assert POOL == "pool_ssmfov"


def test_resolves_to_dynamic_fov_train_static_val_test(monkeypatch):
    _unlock(monkeypatch)
    r = SplitResolver.resolve(Splits.load_split("synth_fov"), _cloud())
    assert isinstance(r.train, DynamicSource)
    assert r.train.provenance()["kind"] == "dynamic" and r.train.bg.mode == "mrxcat"   # whole-FOV bg
    assert isinstance(r.val, StaticSource) and set(r.val.subjects()) == {("acdc", "a1")}
    assert isinstance(r.test, StaticSource)
    assert ("acdc", "a1") not in set(r.test.subjects())
