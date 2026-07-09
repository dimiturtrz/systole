"""synth_composite split family — train = a CompositeSource UNION of synth sources (SSM + pathology,
each its own generation node/painter), val/test same as synth_main. Resolves against a mocked cloud
(lock neutralized); train is the real consumer of the composition mechanism (not dead code)."""
import dataclasses

import polars as pl

from core.data.dynamic.source import CompositeSource, DynamicSource
from core.data.ingest.source import StaticSource
from core.data.ingest.split import SplitResolver
from core.data.ingest.splits import Splits, synth_composite

V = pl.col


def _cloud():
    rows = [("acdc", "a1", "Siemens", True), ("mnm2", "p1", "Philips", True),
            ("mnms1", "g1", "GE", True), ("cmrxmotion", "m1", "Siemens", True)]
    return pl.DataFrame([{"dataset": d, "subject_id": s, "vendor": v, "labelled": lab,
                          "path": f"/s/{d}/{s}.npz"} for d, s, v, lab in rows])


def _unlock(monkeypatch):
    monkeypatch.setattr(synth_composite, "SYNTH_MAIN_TEST",
                        dataclasses.replace(synth_composite.SYNTH_MAIN_TEST, lock=""))


def test_train_is_composite_of_two_synth_sources():
    """The registered split's train is a CompositeSource of SSM + pathology sources — the real consumer
    of the composition mechanism (so it isn't dead code)."""
    assert "synth_composite" in Splits.list_splits()
    train = Splits.load_split("synth_composite").versions["1.0.0"].train(None)   # synth train ignores the cloud
    assert isinstance(train, CompositeSource) and len(train.sources) == 2
    assert all(isinstance(s, DynamicSource) for s in train.sources)
    assert train.provenance()["kind"] == "composite"


def test_composite_sources_are_capped_distinct_pools():
    train = Splits.load_split("synth_composite").versions["1.0.0"].train(None)
    pools = {s.pool for s in train.sources}
    assert len(pools) == 2                                     # SSM + pathology, distinct pools
    assert all(s.cap is not None and s.cap > 0 for s in train.sources)   # VRAM-capped per source


def test_resolves_to_composite_train_static_val_test(monkeypatch):
    _unlock(monkeypatch)
    r = SplitResolver.resolve(Splits.load_split("synth_composite"), _cloud())
    assert isinstance(r.train, CompositeSource)               # union-of-sources train
    assert isinstance(r.val, StaticSource) and set(r.val.subjects()) == {("acdc", "a1")}
    assert isinstance(r.test, StaticSource)
    assert set(r.test.subjects()) == {("mnm2", "p1"), ("mnms1", "g1"), ("cmrxmotion", "m1")}  # seg minus acdc
