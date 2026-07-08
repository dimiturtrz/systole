"""Source + Split ingestion layer — equivalence classes over hashing, complement train, and the
test_lock drift guard. StaticMain's real-data parity vs the old xvendor split is proven separately
(scripts/_lock_static_main.py: 495/150/147, identical subject sets); here we lock the mechanism.
"""
import polars as pl

from core.data.ingest.source import StaticSource, ids_hash
from core.data.ingest.split import SplitDef, _complement, _latest, resolve
from core.data.ingest.splits import list_splits, load_split

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
    import numpy as np
    import torch

    from core.data.dynamic.source import DynamicSource
    monkeypatch.setattr("core.data.dynamic.anatomy.load_pool", lambda p: np.zeros((5, 8, 8), np.int64))
    X, Y, fs = DynamicSource(pool="p")._resident(8, "cpu")
    assert X.shape == (5, 1, 8, 8) and (X == 0).all()          # no real pixels
    assert fs.dtype == torch.bool and bool(fs.all())           # every row force-painted


def test_dynamic_resident_seeded_is_composite(monkeypatch):
    import numpy as np
    import torch

    from core.data.dynamic.source import DynamicSource
    monkeypatch.setattr("core.data.dynamic.anatomy.load_pool", lambda p: np.zeros((3, 8, 8), np.int64))

    class _Seed:                                               # 2 real rows, exposes resident() (no materialize)
        def resident(self, size, device):
            return torch.ones(2, 1, size, size), torch.ones(2, size, size, dtype=torch.long)

    X, Y, fs = DynamicSource(pool="p", seed=_Seed())._resident(8, "cpu")
    assert X.shape == (5, 1, 8, 8) and fs.tolist() == [False, False, True, True, True]  # only synth forced
    assert (X[:2] == 1).all() and (X[2:] == 0).all()          # real pixels kept, synth zeroed


def test_composite_generator_dispatches_by_index_range():
    """CompositeGenerator routes global indices to the right child gen (each its own painter) and
    concatenates — n = Σ child n, and a mixed batch paints each row via its origin source."""
    import torch

    from core.data.dynamic.generator import CompositeGenerator

    class _FakeGen:                                            # duck-types the batch() seam (n/device/batch)
        def __init__(self, n, tag):
            self.n, self.device, self.tag = n, "cpu", tag
        def batch(self, idx, *, pin=False):
            b = idx.shape[0]
            return torch.full((b, 1, 2, 2), self.tag, dtype=torch.float), \
                torch.full((b, 2, 2), self.tag, dtype=torch.long), None

    cg = CompositeGenerator([_FakeGen(3, 1.0), _FakeGen(2, 2.0)])
    assert cg.n == 5 and cg.valid is None
    x, y, v = cg.batch(torch.arange(5))                       # all rows
    assert x.shape == (5, 1, 2, 2) and v is None
    assert int((x == 1.0).sum()) == 3 * 4 and int((x == 2.0).sum()) == 2 * 4   # 3 from child0, 2 from child1
    x2, _, _ = cg.batch(torch.tensor([4, 0]))                 # one row from each child, shuffled
    assert int((x2 == 1.0).sum()) == 4 and int((x2 == 2.0).sum()) == 4


def test_synth_composite_split_resolves_to_composite_source():
    """The registered synth_composite split's train is a CompositeSource of SSM + pathology sources —
    the real consumer of the composition mechanism (so it isn't dead code)."""
    from core.data.dynamic.source import CompositeSource
    from core.data.ingest.splits import list_splits, load_split
    assert "synth_composite" in list_splits()
    train = load_split("synth_composite").versions["1.0.0"].train(None)   # synth train ignores the cloud
    assert isinstance(train, CompositeSource) and len(train.sources) == 2
    assert train.provenance()["kind"] == "composite"


def test_composite_source_unions_children(monkeypatch):
    """CompositeSource builds one CompositeGenerator over its child sources (each keeps its own bg);
    n = sum of pool sizes, provenance lists every child."""
    import numpy as np

    from core.data.dynamic.source import CompositeSource, DynamicSource
    from core.data.dynamic.synth import FlatBgCfg, ProceduralBgCfg
    from core.hparams import TrainCfg
    sizes = {"a": 4, "b": 3}
    monkeypatch.setattr("core.data.dynamic.anatomy.load_pool",
                        lambda p: np.zeros((sizes[p], 8, 8), np.int64))
    src = CompositeSource([DynamicSource(pool="a", bg=ProceduralBgCfg()),
                           DynamicSource(pool="b", bg=FlatBgCfg())])   # different painters per source
    gen = src.train_gen(8, "cpu", TrainCfg().generator, 4)
    assert gen.n == 7                                          # 4 + 3, unioned
    prov = src.provenance()
    assert prov["kind"] == "composite" and len(prov["sources"]) == 2
    assert {s["bg"] for s in prov["sources"]} == {"procedural", "flat"}   # each child kept its own bg


def test_dynamic_train_gen_no_global_mutation(monkeypatch):
    """The refactor's whole point: a DynamicSource configures its OWN engine via a cfg COPY — it must
    NOT mutate the passed generator cfg (the old bg_mode/synth_p poke)."""
    import numpy as np

    from core.data.dynamic.generator import GeneratorCfg
    from core.data.dynamic.source import DynamicSource
    from core.data.dynamic.synth import ProceduralBgCfg
    monkeypatch.setattr("core.data.dynamic.anatomy.load_pool", lambda p: np.zeros((3, 8, 8), np.int64))
    cfg = GeneratorCfg()
    orig_bg = cfg.synth.bg.mode
    gen = DynamicSource(pool="p", bg=ProceduralBgCfg(), synth_p=1.0).train_gen(8, "cpu", cfg, 4)
    assert cfg.synth.bg.mode == orig_bg                        # passed cfg UNTOUCHED (no poke)
    assert gen.cfg.synth.bg.mode == "procedural"               # engine got the override on its copy
    assert bool(gen.force_synth.all())                         # zero-input -> all rows force-painted


def test_static_source_valid_mask_partial():
    import torch
    cloud = pl.DataFrame([{"dataset": "acdc", "subject_id": "a", "labelled": True, "path": "/p/a.npz"},
                          {"dataset": "scd", "subject_id": "s", "labelled": True, "path": "/p/s.npz"}])
    # 2 slices from acdc (path 0, full), 1 from scd (path 1, LV-only)
    vm = StaticSource(cloud)._valid_mask(torch.tensor([0, 0, 1]), 4, "cpu")
    assert vm.tolist() == [[True, True, True, True], [True, True, True, True], [False, False, True, True]]


def test_static_source_valid_mask_none_when_all_full():
    import torch
    src = StaticSource(pl.DataFrame([{"dataset": "acdc", "subject_id": "a", "labelled": True, "path": "/p/a.npz"}]))
    assert src._valid_mask(torch.tensor([0, 0]), 4, "cpu") is None   # full-label -> mask-free


def test_generator_batch_slices_valid():
    import torch

    from core.data.dynamic.generator import Generator, GeneratorCfg
    from core.data.dynamic.synth import SynthCfg
    N = 4
    cfg = GeneratorCfg(synth=SynthCfg(synth_p=0.0))                  # no synth compute for the test
    valid = torch.tensor([[True] * 4, [False, False, True, True], [True] * 4, [True] * 4])
    g = Generator(cfg, torch.zeros(N, 1, 8, 8), torch.zeros(N, 8, 8, dtype=torch.long), 4, "cpu", valid=valid)
    _, _, v = g.batch(torch.tensor([1, 2]))
    assert v.tolist() == [[False, False, True, True], [True, True, True, True]]   # sliced, untouched


def test_static_main_registered_with_locked_testset():
    from core.data.ingest.testsets import STATIC_MAIN_TEST
    assert "static_main" in list_splits()
    d = load_split("static_main").versions["1.0.0"]
    assert d.train is None                                     # train = complement
    assert STATIC_MAIN_TEST.lock.startswith("sha256:") and len(STATIC_MAIN_TEST.lock) > 20


def test_synth_main_registered_with_locked_testset():
    from core.data.ingest.splits.synth_main import POOL
    from core.data.ingest.testsets import SYNTH_MAIN_TEST
    assert "synth_main" in list_splits()
    d = load_split("synth_main").versions["1.0.0"]
    assert d.train is not None                                 # explicit dynamic train (not complement)
    assert POOL                                                # named constant, not a bare literal
    assert SYNTH_MAIN_TEST.lock.startswith("sha256:") and len(SYNTH_MAIN_TEST.lock) > 20
