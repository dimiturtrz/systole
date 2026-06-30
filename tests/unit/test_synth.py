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
    img, msk = synthesize_from_labels(_mask(), SynthCfg(synth_p=1.0), N)
    assert img.shape == (2, 1, 8, 8) and msk.shape == (2, 8, 8)
    assert torch.allclose(img.mean((1, 2, 3)), torch.zeros(2), atol=1e-4)   # per-sample mean ~0
    assert torch.allclose(img.std((1, 2, 3)), torch.ones(2), atol=1e-1)     # per-sample std ~1


def test_painted_per_class():
    """No deform/texture/bias/blur/noise -> each class is a single constant intensity; distinct classes
    get distinct values (the image is generated FROM the labels)."""
    cfg = SynthCfg(synth_p=1.0, deform=0.0, bg_regions=0, sigma=(0.0, 0.0), bias_strength=0.0,
                   blur=(0.0, 0.0), noise=0.0)
    torch.manual_seed(0)
    img, _ = synthesize_from_labels(_mask(), cfg, N)
    vals = torch.unique(img[0])
    assert len(vals) == N                          # one constant per class, all distinct


def test_bg_regions_structure_image_not_target():
    """Structured background paints the bg blob as multiple pseudo-tissue regions (fake thorax) ->
    more distinct intensities in the image than there are real classes, BUT the returned target mask
    keeps only the real labels (bg stays 0). Teaches the net to reject surroundings."""
    cfg = SynthCfg(synth_p=1.0, deform=0.0, bg_regions=6, sigma=(0.0, 0.0), bias_strength=0.0,
                   blur=(0.0, 0.0), noise=0.0)
    torch.manual_seed(0)
    img, msk = synthesize_from_labels(_mask(), cfg, N)
    assert len(torch.unique(img[0])) > N                       # bg split into extra intensities
    assert set(msk.unique().tolist()).issubset(set(range(N)))  # target still only real labels


def test_contrast_varies_per_call():
    """Successive calls (no reseed) draw new per-class means -> different pictures from the same mask."""
    cfg, m = SynthCfg(synth_p=1.0), _mask()
    a, _ = synthesize_from_labels(m, cfg, N)
    b, _ = synthesize_from_labels(m, cfg, N)
    assert not torch.allclose(a, b)


def test_deform_invents_anatomy():
    """deform>0 warps the label map -> returned mask differs from input (new anatomy); deform=0 leaves
    it untouched (pixels-only regime)."""
    m = _mask()
    torch.manual_seed(1)
    _, warped = synthesize_from_labels(m, SynthCfg(synth_p=1.0, deform=0.3), N)
    assert not torch.equal(warped, m)              # anatomy changed
    assert set(warped.unique().tolist()).issubset(set(range(N)))   # still valid labels
    _, same = synthesize_from_labels(m, SynthCfg(synth_p=1.0, deform=0.0), N)
    assert torch.equal(same, m)                    # deform off -> mask preserved


def test_seed_deterministic():
    cfg, m = SynthCfg(synth_p=1.0), _mask()
    torch.manual_seed(7); a, ma = synthesize_from_labels(m, cfg, N)
    torch.manual_seed(7); b, mb = synthesize_from_labels(m, cfg, N)
    assert torch.equal(a, b) and torch.equal(ma, mb)
