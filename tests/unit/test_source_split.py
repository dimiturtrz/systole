"""Source + Split ingestion layer — equivalence classes over hashing, complement train, and the
test_lock drift guard. StaticMain's real-data parity vs the old xvendor split is proven separately
(scripts/_lock_static_main.py: 495/150/147, identical subject sets); here we lock the mechanism.
"""
import polars as pl
import pytest

from core.data.ingest.source import StaticSource, ids_hash
from core.data.ingest.split import SplitDef, resolve, _complement, _latest
from core.data.ingest.splits import load_split, list_splits

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
        val=lambda c: StaticSource(c.filter(V("labelled") & (V("dataset") == "acdc")), "acdc"))}


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
    # the test-lock drift guard now lives on the TestSet (tested in test_testsets.py)


def test_latest_picks_highest_semver():
    assert _latest({"1.0.0": None, "1.10.0": None, "1.2.0": None}) == "1.10.0"


def test_static_source_resident_is_raw_real(monkeypatch):
    import torch
    monkeypatch.setattr("core.data.dynamic.dataset.load_to_gpu",
                        lambda paths, size, device: (torch.zeros(len(paths), 1, size, size), torch.zeros(len(paths), size, size)))
    X, Y = StaticSource(_cloud().filter(V("labelled"))).resident(8, "cpu")
    assert X.shape == (4, 1, 8, 8) and Y.shape == (4, 8, 8)     # raw real, no transforms


def test_dynamic_resident_zero_input_force_paints_all(monkeypatch):
    import numpy as np, torch
    from core.data.dynamic.source import DynamicSource
    monkeypatch.setattr("core.data.dynamic.anatomy.load_pool", lambda p: np.zeros((5, 8, 8), np.int64))
    X, Y, fs = DynamicSource(pool="p")._resident(8, "cpu")
    assert X.shape == (5, 1, 8, 8) and (X == 0).all()          # no real pixels
    assert fs.dtype == torch.bool and bool(fs.all())           # every row force-painted


def test_dynamic_resident_seeded_is_composite(monkeypatch):
    import numpy as np, torch
    from core.data.dynamic.source import DynamicSource
    monkeypatch.setattr("core.data.dynamic.anatomy.load_pool", lambda p: np.zeros((3, 8, 8), np.int64))

    class _Seed:                                               # 2 real rows, exposes resident() (no materialize)
        def resident(self, size, device):
            return torch.ones(2, 1, size, size), torch.ones(2, size, size, dtype=torch.long)

    X, Y, fs = DynamicSource(pool="p", seed=_Seed())._resident(8, "cpu")
    assert X.shape == (5, 1, 8, 8) and fs.tolist() == [False, False, True, True, True]  # only synth forced
    assert (X[:2] == 1).all() and (X[2:] == 0).all()          # real pixels kept, synth zeroed


def test_dynamic_train_gen_no_global_mutation(monkeypatch):
    """The refactor's whole point: a DynamicSource configures its OWN engine via a cfg COPY — it must
    NOT mutate the passed generator cfg (the old bg_mode/synth_p poke)."""
    import numpy as np
    from core.data.dynamic.source import DynamicSource
    from core.data.dynamic.generator import GeneratorCfg
    monkeypatch.setattr("core.data.dynamic.anatomy.load_pool", lambda p: np.zeros((3, 8, 8), np.int64))
    cfg = GeneratorCfg()
    orig_bg = cfg.synth.bg_mode
    gen = DynamicSource(pool="p", bg_mode="procedural", synth_p=1.0).train_gen(8, "cpu", cfg, 4)
    assert cfg.synth.bg_mode == orig_bg                        # passed cfg UNTOUCHED (no poke)
    assert gen.cfg.synth.bg_mode == "procedural"               # engine got the override on its copy
    assert bool(gen.force_synth.all())                         # zero-input -> all rows force-painted


def test_static_main_registered_with_locked_testset():
    from core.data.ingest.testsets import STATIC_MAIN_TEST
    assert "static_main" in list_splits()
    d = load_split("static_main").versions["1.0.0"]
    assert d.train is None                                     # train = complement
    assert STATIC_MAIN_TEST.lock.startswith("sha256:") and len(STATIC_MAIN_TEST.lock) > 20


def test_synth_main_registered_with_locked_testset():
    from core.data.ingest.splits.synth_main import POOL, ZERO_REAL_BG
    from core.data.ingest.testsets import SYNTH_MAIN_TEST
    assert "synth_main" in list_splits()
    d = load_split("synth_main").versions["1.0.0"]
    assert d.train is not None                                 # explicit dynamic train (not complement)
    assert POOL and ZERO_REAL_BG                               # named constants, not bare literals
    assert SYNTH_MAIN_TEST.lock.startswith("sha256:") and len(SYNTH_MAIN_TEST.lock) > 20
