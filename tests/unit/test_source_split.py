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


def test_static_source_materialize_is_pure_real(monkeypatch):
    import torch
    from core.data import source as src_mod
    monkeypatch.setattr("core.data.dynamic.dataset.load_to_gpu",
                        lambda paths, size, device: (torch.zeros(len(paths), 1, size, size), torch.zeros(len(paths), size, size)))
    s = StaticSource(_cloud().filter(V("labelled")))
    X, Y, fs = s.materialize(8, "cpu")
    assert X.shape == (4, 1, 8, 8) and Y.shape == (4, 8, 8) and fs is None    # pure real, never painted


def test_dynamic_source_zero_input_force_paints_all(monkeypatch):
    import numpy as np, torch
    from core.data.dynamic.source import DynamicSource
    monkeypatch.setattr("core.data.dynamic.anatomy.load_pool", lambda p: np.zeros((5, 8, 8), np.int64))
    X, Y, fs = DynamicSource(pool="p").materialize(8, "cpu")
    assert X.shape == (5, 1, 8, 8) and (X == 0).all()          # no real pixels
    assert Y.shape == (5, 8, 8)
    assert fs.dtype == torch.bool and bool(fs.all())           # every row force-painted


def test_dynamic_source_seeded_is_composite(monkeypatch):
    import numpy as np, torch
    from core.data.dynamic.source import DynamicSource
    monkeypatch.setattr("core.data.dynamic.anatomy.load_pool", lambda p: np.zeros((3, 8, 8), np.int64))

    class _Seed:                                               # 2 real rows
        def materialize(self, size, device):
            return torch.ones(2, 1, size, size), torch.ones(2, size, size, dtype=torch.long), None
        def provenance(self): return {"kind": "static", "n": 2}

    X, Y, fs = DynamicSource(pool="p", seed=_Seed()).materialize(8, "cpu")
    assert X.shape == (5, 1, 8, 8) and Y.shape == (5, 8, 8)    # 2 real ++ 3 synth
    assert fs.tolist() == [False, False, True, True, True]     # only synth rows forced
    assert (X[:2] == 1).all() and (X[2:] == 0).all()          # real pixels kept, synth zeroed


def test_static_main_registered_and_locked():
    assert "static_main" in list_splits()
    d = load_split("static_main").versions["1.0.0"]
    assert d.test_lock.startswith("sha256:") and len(d.test_lock) > 20   # real lock, not placeholder
    assert d.train is None                                     # train = complement
