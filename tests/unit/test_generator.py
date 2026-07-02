"""The training data engine (core.data.dynamic.generator). Contract: Generator(GeneratorCfg) yields
COLLAPSED (image, target) batches from resident real tensors — real / synth / mixed by cfg.synth, with
priors, augment and soften already applied. The train loop only calls batch()."""
import torch

from core.data.dynamic.generator import GeneratorCfg
from core.data.dynamic.synth import SynthCfg
from core.data.dynamic.augment import AugCfg
from core.data.dynamic.generator import Generator

N = 4


def _resident(n=8, h=16, w=16):
    """Tiny resident real set: n slices, each with all 4 classes in bands + plausible intensity."""
    Y = torch.zeros(n, h, w, dtype=torch.uint8)
    X = torch.randn(n, 1, h, w) * 0.3
    for c in range(N):
        Y[:, c * (h // N):(c + 1) * (h // N), :] = c
        X[:, 0, c * (h // N):(c + 1) * (h // N), :] += c          # class-correlated intensity
    return X, Y


def _gen(synth_p, **synth):
    X, Y = _resident()
    cfg = GeneratorCfg(synth=SynthCfg(synth_p=synth_p, **synth), aug=AugCfg())
    return Generator(cfg, X, Y, N, "cpu"), X, Y


def test_batch_shapes_and_soft_target():
    """Real batch (synth off): x [B,1,H,W], soft target [B,C,H,W] (soft_label_sigma default > 0)."""
    gen, X, Y = _gen(0.0)
    x, yt = gen.batch(torch.arange(4))
    assert x.shape == (4, 1, 16, 16)
    assert yt.shape == (4, N, 16, 16)
    assert torch.allclose(yt.sum(1), torch.ones(4, 16, 16), atol=1e-4)   # soft probs sum to 1


def test_synth_off_is_pure_real():
    """synth_p=0 -> generator does no synth (pure real passthrough + aug)."""
    gen, _, _ = _gen(0.0)
    assert gen.synth_on is False


def test_synth_on_runs():
    """synth_p=1 -> physical (bSSFP) synth batch, right shapes (no priors needed)."""
    gen, _, _ = _gen(1.0)
    assert gen.synth_on is True
    x, yt = gen.batch(torch.arange(4))
    assert x.shape == (4, 1, 16, 16) and yt.shape == (4, N, 16, 16)


def test_synth_changes_the_image():
    """Same indices, pure-synth vs pure-real -> different pixels (the image was invented, not loaded).
    Aug uses global RNG, so seed both identically to isolate the synth difference."""
    g_real, _, _ = _gen(0.0)
    g_syn, _, _ = _gen(1.0, bg_mode="flat")
    idx = torch.arange(4)
    torch.manual_seed(0); xr, _ = g_real.batch(idx)
    torch.manual_seed(0); xs, _ = g_syn.batch(idx)
    assert not torch.allclose(xr, xs)


def test_hard_target_when_sigma_zero():
    """soft_label_sigma=0 -> hard target [B,1,H,W] (the crisp-label path)."""
    X, Y = _resident()
    cfg = GeneratorCfg(synth=SynthCfg(synth_p=0.0), aug=AugCfg(soft_label_sigma=0.0))
    gen = Generator(cfg, X, Y, N, "cpu")
    _, yt = gen.batch(torch.arange(4))
    assert yt.shape == (4, 1, 16, 16)


def test_force_synth_paints_flagged_rows_at_synth_p0():
    """force_synth rows are painted EVERY batch (synth-anatomy mix, bd pwih) even with synth_p=0;
    non-flagged real rows pass through untouched. synth_on must be True despite synth_p=0."""
    X, Y = _resident(n=8)
    force = torch.zeros(8, dtype=torch.bool); force[4:] = True     # last 4 = synth-anatomy rows
    cfg = GeneratorCfg(synth=SynthCfg(synth_p=0.0, bg_mode="flat"),
                       aug=AugCfg(rot_deg=0.0, scale=(1.0, 1.0), translate=0.0, gamma_p=0.0,
                                  blur_p=0.0, contrast=(1.0, 1.0), noise=0.0, bias_p=0.0))
    gen = Generator(cfg, X, Y, N, "cpu", force_synth=force)
    assert gen.synth_on is True                                    # forced rows -> synth active
    assert Generator(cfg, X, Y, N, "cpu", force_synth=None).synth_on is False   # no force + synth_p=0 -> off
    torch.manual_seed(0); x, _ = gen.batch(torch.arange(8))
    # synth output is z-scored per sample (std==1); real rows keep their source std (~1.15 here).
    assert (x[4:].std(dim=(1, 2, 3)) - 1.0).abs().max() < 1e-3     # forced rows = z-scored synth
    assert (x[:4].std(dim=(1, 2, 3)) - 1.0).abs().min() > 0.05     # unforced rows are NOT z-scored synth
