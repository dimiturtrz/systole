"""Batch transform pipeline — the ops the Generator composes. Locks Soften (hard/soft), SynthReplace
(off / force-replace), and the build order. Augment is exercised by the training smoke + its own tests.
"""
import torch

from core.data.dynamic.generator import GeneratorCfg
from core.data.dynamic.pipeline import Batch, Pipeline, Soften, SynthReplace
from core.data.dynamic.synth import SynthCfg


def _batch(force=None):
    return Batch(x=torch.zeros(2, 1, 4, 4), y=torch.zeros(2, 4, 4, dtype=torch.long), force=force)


def test_soften_hard_when_sigma_zero():
    b = Soften(0.0, 4)(_batch())
    assert b.yt.shape == (2, 1, 4, 4)                       # hard mask + channel dim
    assert (b.yt == b.y[:, None]).all()


def test_soften_soft_when_sigma_positive():
    b = Soften(1.0, 4)(Batch(x=torch.zeros(2, 1, 8, 8), y=torch.zeros(2, 8, 8, dtype=torch.long)))
    assert b.yt.shape[:2] == (2, 4)                         # [B, C, H, W] probabilistic target


def test_synth_replace_noop_when_off():
    b0 = _batch()
    x0 = b0.x.clone()
    b = SynthReplace(SynthCfg(synth_p=0.0), 4)(b0)          # synth_p=0, no force -> off
    assert (b.x == x0).all()                                # untouched


def test_synth_replace_force_replaces_all(monkeypatch):
    monkeypatch.setattr("core.data.dynamic.pipeline.SynthPainter.synthesize_from_labels",
                        lambda mask, cfg, n, real_img: (torch.ones_like(real_img), torch.full_like(mask, 2)))
    b = SynthReplace(SynthCfg(synth_p=0.0), 4)(_batch(force=torch.ones(2, dtype=torch.bool)))
    assert (b.x == 1).all() and (b.y == 2).all()           # every forced row painted synthetic


def test_build_pipeline_order():
    ops = Pipeline.build(GeneratorCfg(), 4)
    assert [type(o).__name__ for o in ops] == ["SynthReplace", "Augment", "Soften"]
