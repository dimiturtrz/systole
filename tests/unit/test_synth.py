"""SynthSeg-style synthetic-from-labels generation (cardioseg.training.synth, bd cardiac-seg-bgc).
The contract: a label mask in -> a z-scored image out, painted PER CLASS, fresh contrast each call,
anatomy (the mask) never consulted for intensity beyond its class id."""
import torch

from core.hparams import SynthCfg
from cardioseg.training.synth import synthesize_from_labels

N = 4  # canonical classes: 0 bg, 1 RV, 2 LV-myo, 3 LV-cav


def _mask(b=2, h=8, w=8):
    """Two samples, each with all 4 classes laid out in horizontal bands."""
    m = torch.zeros(b, h, w, dtype=torch.long)
    for c in range(N):
        m[:, c * (h // N):(c + 1) * (h // N), :] = c
    return m


def test_shape_and_zscore():
    img = synthesize_from_labels(_mask(), SynthCfg(synth_p=1.0), N)
    assert img.shape == (2, 1, 8, 8)
    assert torch.allclose(img.mean((1, 2, 3)), torch.zeros(2), atol=1e-4)   # per-sample mean ~0
    assert torch.allclose(img.std((1, 2, 3)), torch.ones(2), atol=1e-1)     # per-sample std ~1


def test_painted_per_class():
    """No texture/bias/blur/noise -> each class is a single constant intensity; distinct classes get
    distinct values (the image is generated FROM the labels)."""
    cfg = SynthCfg(synth_p=1.0, sigma=(0.0, 0.0), bias_strength=0.0, blur=(0.0, 0.0), noise=0.0)
    torch.manual_seed(0)
    img = synthesize_from_labels(_mask(), cfg, N)
    vals = torch.unique(img[0])
    assert len(vals) == N                          # one constant per class, all distinct


def test_contrast_varies_per_call():
    """Successive calls (no reseed) draw new per-class means -> different pictures from the same mask."""
    cfg, m = SynthCfg(synth_p=1.0), _mask()
    a = synthesize_from_labels(m, cfg, N)
    b = synthesize_from_labels(m, cfg, N)
    assert not torch.allclose(a, b)


def test_seed_deterministic():
    cfg, m = SynthCfg(synth_p=1.0), _mask()
    torch.manual_seed(7); a = synthesize_from_labels(m, cfg, N)
    torch.manual_seed(7); b = synthesize_from_labels(m, cfg, N)
    assert torch.equal(a, b)
