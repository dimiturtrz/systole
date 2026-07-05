"""Source + Split ingestion layer — equivalence classes over hashing, complement train, and the
test_lock drift guard. StaticMain's real-data parity vs the old xvendor split is proven separately
(scripts/_lock_static_main.py: 495/150/147, identical subject sets); here we lock the mechanism.
"""
import polars as pl
import pytest

from core.data.source import StaticSource, ids_hash
from core.data.split import SplitDef, resolve, _complement, _latest
from core.data.splits import load_split, list_splits

V = pl.col


def _cloud():
    rows = [("acdc", "s1", "Siemens", True), ("acdc", "s2", "Siemens", True),
            ("mnms1", "c1", "Canon", True), ("mnms1", "g1", "GE", True),
            ("mnms1", "u1", "Canon", False)]                       # unlabelled -> never selected
    return pl.DataFrame([{"dataset": d, "subject_id": s, "vendor": v, "labelled": lab,
                          "path": f"/s/{d}/{s}.npz"} for d, s, v, lab in rows])


class _Fam:
    name = "fam"
    versions = {"1.0.0": SplitDef(
        test=lambda c: StaticSource(c.filter(V("labelled") & (V("vendor") == "Canon")), "canon"),
        val=lambda c: StaticSource(c.filter(V("labelled") & (V("dataset") == "acdc")), "acdc"),
        test_lock="")}


def test_ids_hash_order_independent():
    assert ids_hash([("a", "1"), ("b", "2")]) == ids_hash([("b", "2"), ("a", "1")])
    assert ids_hash([("a", "1")]) != ids_hash([("a", "2")])


def test_static_source_exposes_paths_subjects_hash():
    s = StaticSource(_cloud().filter(V("vendor") == "GE"))
    assert s.paths() == ["/s/mnms1/g1.npz"]
    assert s.subjects() == [("mnms1", "g1")] and len(s) == 1
    assert s.provenance()["kind"] == "static" and s.ids_hash().startswith("sha256:")


def test_complement_is_labelled_rest():
    c = _cloud()
    test = _Fam.versions["1.0.0"].test(c)      # c1
    val = _Fam.versions["1.0.0"].val(c)        # s1, s2
    train = _complement(c, [test, val])
    assert set(train.subjects()) == {("mnms1", "g1")}          # labelled rest; u1 (unlabelled) excluded


def test_resolve_builds_triple_and_complement_train():
    r = resolve(_Fam(), _cloud())
    assert set(r.test.subjects()) == {("mnms1", "c1")}
    assert set(r.val.subjects()) == {("acdc", "s1"), ("acdc", "s2")}
    assert set(r.train.subjects()) == {("mnms1", "g1")}        # no test/val leak
    assert r.version == "1.0.0" and r.test_hash == r.test.ids_hash()


def test_lock_guard_raises_on_drift():
    fam = _Fam()
    fam.versions = {"1.0.0": SplitDef(test=fam.versions["1.0.0"].test, val=fam.versions["1.0.0"].val,
                                      test_lock="sha256:deadbeef")}
    with pytest.raises(ValueError, match="drifted"):
        resolve(fam, _cloud())


def test_lock_guard_passes_when_hash_matches():
    c = _cloud()
    good = _Fam.versions["1.0.0"].test(c).ids_hash()
    fam = _Fam()
    fam.versions = {"1.0.0": SplitDef(test=fam.versions["1.0.0"].test, val=fam.versions["1.0.0"].val,
                                      test_lock=good)}
    assert resolve(fam, c).test_hash == good                   # no raise


def test_latest_picks_highest_semver():
    assert _latest({"1.0.0": None, "1.10.0": None, "1.2.0": None}) == "1.10.0"


def test_static_main_registered_and_locked():
    assert "static_main" in list_splits()
    d = load_split("static_main").versions["1.0.0"]
    assert d.test_lock.startswith("sha256:") and len(d.test_lock) > 20   # real lock, not placeholder
    assert d.train is None                                     # train = complement
