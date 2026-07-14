"""parametric split family (bd cardiac-seg-gz19) — the criteria knobs as a coded split. Resolves
against a small mocked cloud: default == the canonical xvendor partition (Canon+GE + motion held out,
ACDC val), and the constructor params express arbitrary holdouts (single-vendor test, vendor-restricted
train) that the frozen families can't — each still content-hashed per combo."""
import polars as pl

from core.data.ingest.source import StaticSource
from core.data.ingest.split import SplitResolver
from core.data.ingest.splits import Splits
from core.data.ingest.splits.parametric import Parametric

V = pl.col


def _cloud():
    rows = [("acdc", "a1", "Siemens", True), ("mnm2", "p1", "Philips", True),
            ("mnms1", "g1", "GE", True), ("mnms1", "c1", "Canon", True),
            ("cmrxmotion", "m1", "Siemens", True)]
    return pl.DataFrame([{"dataset": d, "subject_id": s, "vendor": v, "labelled": lab,
                          "path": f"/s/{d}/{s}.npz"} for d, s, v, lab in rows])


def test_registered():
    assert "parametric" in Splits.list_splits()
    assert Splits.load_split("parametric").versions["1.0.0"].train is None   # default train = complement


def test_default_matches_xvendor_partition():
    """Parametric() defaults reproduce the criteria defaults / static_main partition."""
    r = SplitResolver.resolve(Parametric(), _cloud())
    assert all(isinstance(s, StaticSource) for s in (r.train, r.val, r.test))
    assert set(r.test.subjects()) == {("mnms1", "g1"), ("mnms1", "c1"), ("cmrxmotion", "m1")}  # Canon+GE + motion
    assert set(r.val.subjects()) == {("acdc", "a1")}          # ACDC centre-shift
    assert set(r.train.subjects()) == {("mnm2", "p1")}        # complement: Siemens+Philips seg
    assert r.test_hash.startswith("sha256:")                  # content-hashed per combo


def test_single_vendor_holdout_is_coded():
    """A holdout no frozen family expresses: hold out GE alone (no dataset holdout)."""
    r = SplitResolver.resolve(Parametric(test_vendors=("GE",), test_datasets=()), _cloud())
    assert set(r.test.subjects()) == {("mnms1", "g1")}        # GE only
    assert ("mnms1", "c1") in set(r.train.subjects())         # Canon now trains (not held out)


def test_train_vendor_restriction(monkeypatch):
    """train_vendors restricts TRAIN to a vendor subset (bd 5r7n) with no test/val leak."""
    r = SplitResolver.resolve(Parametric(train_vendors=("Philips",)), _cloud())
    assert set(r.train.subjects()) == {("mnm2", "p1")}        # Philips only
    # held-out test vendors + val never leak into the restricted train
    subs = [set(r.train.subjects()), set(r.val.subjects()), set(r.test.subjects())]
    assert subs[0] & subs[1] == set() and subs[0] & subs[2] == set() and subs[1] & subs[2] == set()
