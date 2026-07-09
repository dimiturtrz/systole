"""The training data engine (core.data.dynamic.generator). Contract: Generator(GeneratorCfg) yields
COLLAPSED (image, target) batches from resident real tensors — real / synth / mixed by cfg.synth, with
priors, augment and soften already applied. The train loop only calls batch()."""
import torch

from core.data.dynamic.augment import AugCfg
from core.data.dynamic.generator import CompositeGenerator, Generator, GeneratorCfg
from core.data.dynamic.synth import FlatBgCfg, SynthCfg

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
    x, yt, _ = gen.batch(torch.arange(4))
    assert x.shape == (4, 1, 16, 16)
    assert yt.shape == (4, N, 16, 16)
    assert torch.allclose(yt.sum(1), torch.ones(4, 16, 16), atol=1e-4)   # soft probs sum to 1


def test_synth_on_runs():
    """synth_p=1 -> physical (bSSFP) synth batch, right shapes (no priors needed)."""
    gen, _, _ = _gen(1.0)
    x, yt, _ = gen.batch(torch.arange(4))
    assert x.shape == (4, 1, 16, 16) and yt.shape == (4, N, 16, 16)


def test_synth_changes_the_image():
    """Same indices, pure-synth vs pure-real -> different pixels (the image was invented, not loaded).
    Aug uses global RNG, so seed both identically to isolate the synth difference."""
    g_real, _, _ = _gen(0.0)
    g_syn, _, _ = _gen(1.0, bg=FlatBgCfg())
    idx = torch.arange(4)
    torch.manual_seed(0); xr, _, _ = g_real.batch(idx)
    torch.manual_seed(0); xs, _, _ = g_syn.batch(idx)
    assert not torch.allclose(xr, xs)


def test_hard_target_when_sigma_zero():
    """soft_label_sigma=0 -> hard target [B,1,H,W] (the crisp-label path)."""
    X, Y = _resident()
    cfg = GeneratorCfg(synth=SynthCfg(synth_p=0.0), aug=AugCfg(soft_label_sigma=0.0))
    gen = Generator(cfg, X, Y, N, "cpu")
    _, yt, _ = gen.batch(torch.arange(4))
    assert yt.shape == (4, 1, 16, 16)


def test_force_synth_paints_flagged_rows_at_synth_p0():
    """force_synth rows are painted EVERY batch (synth-anatomy mix, bd pwih) even with synth_p=0;
    non-flagged real rows pass through untouched."""
    X, Y = _resident(n=8)
    force = torch.zeros(8, dtype=torch.bool); force[4:] = True     # last 4 = synth-anatomy rows
    cfg = GeneratorCfg(synth=SynthCfg(synth_p=0.0, bg=FlatBgCfg()),
                       aug=AugCfg(rot_deg=0.0, scale=(1.0, 1.0), translate=0.0, gamma_p=0.0,
                                  blur_p=0.0, contrast=(1.0, 1.0), noise=0.0, bias_p=0.0))
    gen = Generator(cfg, X, Y, N, "cpu", force_synth=force)
    torch.manual_seed(0); x, _, _ = gen.batch(torch.arange(8))
    # synth output is z-scored per sample (std==1); real rows keep their source std (~1.15 here).
    assert (x[4:].std(dim=(1, 2, 3)) - 1.0).abs().max() < 1e-3     # forced rows = z-scored synth
    assert (x[:4].std(dim=(1, 2, 3)) - 1.0).abs().min() > 0.05     # unforced rows are NOT z-scored synth


def test_batch_slices_valid():
    """A per-slice valid [N,C] mask rides through batch() sliced to the batch rows, untouched."""
    valid = torch.tensor([[True] * 4, [False, False, True, True], [True] * 4, [True] * 4])
    g = Generator(GeneratorCfg(synth=SynthCfg(synth_p=0.0)), torch.zeros(N, 1, 8, 8),
                  torch.zeros(N, 8, 8, dtype=torch.long), 4, "cpu", valid=valid)
    _, _, v = g.batch(torch.tensor([1, 2]))
    assert v.tolist() == [[False, False, True, True], [True, True, True, True]]   # sliced, untouched


def test_composite_generator_dispatches_by_index_range():
    """CompositeGenerator routes global indices to the right child gen (each its own painter) and
    concatenates — n = Σ child n, and a mixed batch paints each row via its origin source."""
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
