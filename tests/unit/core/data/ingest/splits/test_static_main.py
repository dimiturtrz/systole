"""static_main split family — the all-real generalization split. Resolves against a small mocked
cloud (lock neutralized) to a valid static partition: test = unseen vendors + motion, val = ACDC,
train = the labelled complement (Siemens + Philips)."""
import dataclasses

import polars as pl

from core.data.ingest.source import StaticSource
from core.data.ingest.split import SplitResolver
from core.data.ingest.splits import Splits, static_main
from core.data.ingest.testsets import STATIC_MAIN_TEST

V = pl.col


def _cloud():
    rows = [("acdc", "a1", "Siemens", True), ("mnm2", "p1", "Philips", True),
            ("mnms1", "g1", "GE", True), ("mnms1", "c1", "Canon", True),
            ("cmrxmotion", "m1", "Siemens", True)]
    return pl.DataFrame([{"dataset": d, "subject_id": s, "vendor": v, "labelled": lab,
                          "path": f"/s/{d}/{s}.npz"} for d, s, v, lab in rows])


def _unlock(monkeypatch):
    """Neutralize the frozen test lock so resolving against a mocked cloud doesn't trip the drift guard."""
    monkeypatch.setattr(static_main, "STATIC_MAIN_TEST",
                        dataclasses.replace(static_main.STATIC_MAIN_TEST, lock=""))


def test_registered_with_locked_testset():
    assert "static_main" in Splits.list_splits()
    d = Splits.load_split("static_main").versions["1.0.0"]
    assert d.train is None                                     # train = complement
    assert STATIC_MAIN_TEST.lock.startswith("sha256:") and len(STATIC_MAIN_TEST.lock) > 20


def test_resolves_to_valid_static_partition(monkeypatch):
    _unlock(monkeypatch)
    r = SplitResolver.resolve(Splits.load_split("static_main"), _cloud())
    assert all(isinstance(s, StaticSource) for s in (r.train, r.val, r.test))
    assert set(r.test.subjects()) == {("mnms1", "g1"), ("mnms1", "c1"), ("cmrxmotion", "m1")}  # unseen vendors + motion
    assert set(r.val.subjects()) == {("acdc", "a1")}          # ACDC centre-shift val
    assert set(r.train.subjects()) == {("mnm2", "p1")}        # complement: Siemens + Philips seg
    # MECE: the three source subject-sets are disjoint (no leak)
    subs = [set(r.train.subjects()), set(r.val.subjects()), set(r.test.subjects())]
    assert subs[0] & subs[1] == set() and subs[0] & subs[2] == set() and subs[1] & subs[2] == set()
