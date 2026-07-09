"""Integration (module-PAIR / chain, mock-fed, no data/GPU): the coded-split composition seam
split.train -> Source -> Source.train_gen -> Generator.batch produces a valid model-input batch.

Distinct from the E2E train_seg smoke (tests/e2e): no train loop, no eval, no save — just that the
data-engine chain wires up and emits a finite (x, yt) of the right shape. This is the seam the
composite work (CompositeSource/CompositeGenerator) + the acq/bg union pass through; a wiring
regression (like the acq_mode / train_df class) surfaces here in CI, without real data.
"""
import numpy as np
import pytest
import torch

from core.data.ingest.splits import Splits
from core.hparams import TrainCfg

_SIZE = 16


@pytest.mark.parametrize("split", ["synth_main", "synth_composite"])
def test_coded_split_train_chain_emits_valid_batch(split, monkeypatch):
    """Each synth split's train Source builds its Generator and paints a finite batch of the right
    shape — the split->source->generator->batch chain, end to end on tiny mocked pools."""
    monkeypatch.setattr("core.data.dynamic.anatomy.Anatomy.load_pool",
                        lambda p: np.zeros((20, _SIZE, _SIZE), np.int64))
    src = Splits.load_split(split).versions["1.0.0"].train(None)     # synth train ignores the cloud
    gen = src.train_gen(_SIZE, "cpu", TrainCfg().generator, 4)
    assert gen.n > 0
    idx = torch.arange(min(8, gen.n))
    x, yt, _valid = gen.batch(idx)
    assert x.shape == (idx.shape[0], 1, _SIZE, _SIZE)         # [B,1,H,W] model input
    assert yt.shape[0] == idx.shape[0]                        # target aligned
    assert torch.isfinite(x).all()                            # paint produced real numbers


def test_composite_generator_batches_mix_both_sources(monkeypatch):
    """The composite chain unions BOTH sources into the resident set — n = Σ per-source (capped), so a
    full-range batch draws from each child's own painter (the CompositeGenerator dispatch, live)."""
    monkeypatch.setattr("core.data.dynamic.anatomy.Anatomy.load_pool",
                        lambda p: np.zeros((20, _SIZE, _SIZE), np.int64))
    src = Splits.load_split("synth_composite").versions["1.0.0"].train(None)
    gen = src.train_gen(_SIZE, "cpu", TrainCfg().generator, 4)
    assert gen.n == 40                                        # 2 sources x 20 (cap 5000 > 20 = no-op)
    x, _, _ = gen.batch(torch.arange(gen.n))
    assert x.shape == (40, 1, _SIZE, _SIZE) and torch.isfinite(x).all()
