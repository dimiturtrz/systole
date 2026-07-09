"""DynamicSource / CompositeSource — the synth side of the Source seam: cap subsample determinism,
force_synth wiring, train_gen cfg-copy (no global mutation), and composite provenance. Mock-fed via
Anatomy.load_pool -> tiny label array (I/O-free)."""
import numpy as np
import pytest
import torch

from core.data.dynamic.generator import GeneratorCfg
from core.data.dynamic.source import CompositeSource, DynamicSource
from core.data.dynamic.synth import FlatBgCfg, ProceduralBgCfg
from core.hparams import TrainCfg


def test_dynamic_resident_zero_input_force_paints_all(monkeypatch):
    monkeypatch.setattr("core.data.dynamic.anatomy.Anatomy.load_pool", lambda p: np.zeros((5, 8, 8), np.int64))
    X, Y, fs = DynamicSource(pool="p")._resident(8, "cpu")
    assert X.shape == (5, 1, 8, 8) and (X == 0).all()          # no real pixels
    assert fs.dtype == torch.bool and bool(fs.all())           # every row force-painted


def test_dynamic_resident_seeded_is_composite(monkeypatch):
    monkeypatch.setattr("core.data.dynamic.anatomy.Anatomy.load_pool", lambda p: np.zeros((3, 8, 8), np.int64))

    class _Seed:                                               # 2 real rows, exposes resident() (no materialize)
        def resident(self, size, device):
            return torch.ones(2, 1, size, size), torch.ones(2, size, size, dtype=torch.long)

    X, Y, fs = DynamicSource(pool="p", seed=_Seed())._resident(8, "cpu")
    assert X.shape == (5, 1, 8, 8) and fs.tolist() == [False, False, True, True, True]  # only synth forced
    assert (X[:2] == 1).all() and (X[2:] == 0).all()          # real pixels kept, synth zeroed


def test_dynamic_source_cap_bounds_resident(monkeypatch):
    """cap deterministically subsamples the resident to <= cap slices (VRAM-bound the GPU preload); no
    cap or cap >= pool size = the full pool. Fixes the composite thrash: the 42k union doesn't fit VRAM."""
    monkeypatch.setattr("core.data.dynamic.anatomy.Anatomy.load_pool", lambda p: np.zeros((100, 8, 8), np.int64))
    X, Y, fs = DynamicSource(pool="p", cap=30)._resident(8, "cpu")
    assert X.shape[0] == 30 and Y.shape[0] == 30 and fs.shape[0] == 30      # capped
    assert DynamicSource(pool="p")._resident(8, "cpu")[0].shape[0] == 100   # no cap = full
    assert DynamicSource(pool="p", cap=999)._resident(8, "cpu")[0].shape[0] == 100  # cap>=size = no-op


def test_dynamic_cap_subsample_is_deterministic(monkeypatch):
    """cap uses a fixed-seed permutation — the kept slices are the SAME across two builds (reproducible
    resident, not a fresh random draw each run)."""
    monkeypatch.setattr("core.data.dynamic.anatomy.Anatomy.load_pool",
                        lambda p: np.arange(50)[:, None, None].repeat(8, 1).repeat(8, 2).astype(np.int64))
    _, Y1, _ = DynamicSource(pool="p", cap=10)._resident(8, "cpu")
    _, Y2, _ = DynamicSource(pool="p", cap=10)._resident(8, "cpu")
    assert torch.equal(Y1, Y2)                                 # same subsample both times


def test_dynamic_train_gen_no_global_mutation(monkeypatch):
    """A DynamicSource configures its OWN engine via a cfg COPY — it must NOT mutate the passed generator
    cfg (the old bg_mode/synth_p poke)."""
    monkeypatch.setattr("core.data.dynamic.anatomy.Anatomy.load_pool", lambda p: np.zeros((3, 8, 8), np.int64))
    cfg = GeneratorCfg()
    orig_bg = cfg.synth.bg.mode
    gen = DynamicSource(pool="p", bg=ProceduralBgCfg(), synth_p=1.0).train_gen(8, "cpu", cfg, 4)
    assert cfg.synth.bg.mode == orig_bg                        # passed cfg UNTOUCHED (no poke)
    assert gen.cfg.synth.bg.mode == "procedural"               # engine got the override on its copy
    assert bool(gen.force_synth.all())                         # zero-input -> all rows force-painted


def test_dynamic_provenance_carries_pool_bg_cap():
    p = DynamicSource(pool="p", bg=ProceduralBgCfg(), synth_p=0.5, cap=7, note="n").provenance()
    assert p["kind"] == "dynamic" and p["cap"] == 7 and p["bg"] == "procedural"
    assert p["synth_p"] == 0.5 and p["note"] == "n" and p["seed"] is None


def test_composite_source_unions_children(monkeypatch):
    """CompositeSource builds one CompositeGenerator over its child sources (each keeps its own bg);
    n = sum of pool sizes, provenance lists every child with its own painter."""
    sizes = {"a": 4, "b": 3}
    monkeypatch.setattr("core.data.dynamic.anatomy.Anatomy.load_pool",
                        lambda p: np.zeros((sizes[p], 8, 8), np.int64))
    src = CompositeSource([DynamicSource(pool="a", bg=ProceduralBgCfg()),
                           DynamicSource(pool="b", bg=FlatBgCfg())])   # different painters per source
    gen = src.train_gen(8, "cpu", TrainCfg().generator, 4)
    assert gen.n == 7                                          # 4 + 3, unioned
    prov = src.provenance()
    assert prov["kind"] == "composite" and len(prov["sources"]) == 2
    assert {s["bg"] for s in prov["sources"]} == {"procedural", "flat"}   # each child kept its own bg


def test_composite_source_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        CompositeSource([])
