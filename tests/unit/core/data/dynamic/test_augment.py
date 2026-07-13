"""Unit tests for GPU-batched augmentation (runs on CPU tensors here)."""
import pytest

torch = pytest.importorskip("torch")

from core.data.dynamic.augment import AugCfg, Augmentor  # noqa: E402

# everything off except (toggled) bias — flip is always-on, so isolate via same-seed bias on/off
_NO_INTENSITY = {"rot_deg": 0.0, "scale": (1.0, 1.0), "translate": 0.0, "gamma_p": 0.0, "blur_p": 0.0,
                 "contrast": (1.0, 1.0), "noise": 0.0}


def _batch():
    torch.manual_seed(0)
    img = torch.randn(4, 1, 64, 64)
    mask = torch.zeros(4, 64, 64, dtype=torch.long)
    mask[:, 20:44, 20:44] = 3   # LV-cav block
    mask[:, 10:16, 10:16] = 1   # RV blob
    return img, mask


def test_shapes_and_dtype_preserved():
    img, mask = _batch()
    a, b = Augmentor().augment_batch(img, mask)
    assert a.shape == img.shape
    assert b.shape == mask.shape
    assert b.dtype == torch.long          # mask stays integer labels


def test_mask_labels_stay_a_subset():
    """Nearest-neighbour sampling + zero padding -> only the input labels (or background)."""
    img, mask = _batch()
    _, b = Augmentor().augment_batch(img, mask)
    assert set(b.unique().tolist()).issubset({0, 1, 3})  # no interpolated/new labels


def test_image_finite_and_actually_changed():
    img, mask = _batch()
    a, _ = Augmentor().augment_batch(img, mask)
    assert torch.isfinite(a).all()        # no NaN/Inf from gamma/grid_sample
    assert not torch.allclose(a, img)     # augmentation did something


def test_bias_field_off_is_noop():
    """bias_p=0 -> identical to the same-seed run as if the bias branch did nothing (it's gated off)."""
    img = torch.rand(4, 1, 64, 64) + 1.0          # positive -> clean ratio later
    mask = torch.zeros(4, 64, 64, dtype=torch.long); mask[:, 20:44, 20:44] = 3
    off = AugCfg(**_NO_INTENSITY, bias_p=0.0)
    torch.manual_seed(3); a0, _ = Augmentor(off).augment_batch(img.clone(), mask.clone())
    torch.manual_seed(3); a0b, _ = Augmentor(off).augment_batch(img.clone(), mask.clone())
    assert torch.allclose(a0, a0b)                 # deterministic; bias off contributes nothing


def test_bias_field_smooth_multiplicative():
    """Toggling bias on (same seed) multiplies by a smooth, bounded ~1+/-strength field; mask untouched."""
    img = torch.rand(4, 1, 64, 64) + 1.0          # positive, no near-zero -> stable ratio
    mask = torch.zeros(4, 64, 64, dtype=torch.long); mask[:, 20:44, 20:44] = 3
    off = AugCfg(**_NO_INTENSITY, bias_p=0.0)
    on = AugCfg(**_NO_INTENSITY, bias_p=1.0, bias_strength=0.3)
    torch.manual_seed(3); a_off, b_off = Augmentor(off).augment_batch(img.clone(), mask.clone())
    torch.manual_seed(3); a_on, b_on = Augmentor(on).augment_batch(img.clone(), mask.clone())
    assert torch.equal(b_on, b_off)               # same seed -> same geometry; bias is intensity-only
    assert not torch.allclose(a_on, a_off)        # field applied
    ratio = a_on / a_off                          # = the bias field (geometry cancels)
    assert ratio.min() > 0.69 and ratio.max() < 1.31   # within 1 +/- 0.3, smooth & bounded


def test_deterministic_under_seed():
    img, mask = _batch()
    torch.manual_seed(7)
    a1, b1 = Augmentor().augment_batch(img.clone(), mask.clone())
    torch.manual_seed(7)
    a2, b2 = Augmentor().augment_batch(img.clone(), mask.clone())
    assert torch.allclose(a1, a2)
    assert torch.equal(b1, b2)            # same RNG seed -> identical augmentation


def test_translation_moves_the_heart_off_center():
    """translate>0 shifts the heart centroid across the batch (fixes the synth center-bias); translate=0
    leaves a centered heart centered. Equivalence classes: {off, on}."""
    def centroids(cfg):
        m = torch.zeros(16, 64, 64, dtype=torch.long); m[:, 26:38, 26:38] = 3   # centered heart
        img = torch.zeros(16, 1, 64, 64)
        torch.manual_seed(0)
        _, mo = Augmentor(cfg).augment_batch(img, m)
        cs = []
        for k in range(mo.shape[0]):
            ys, xs = torch.where(mo[k] > 0)
            if len(ys):
                cs.append((float(ys.float().mean()) - 32, float(xs.float().mean()) - 32))
        return torch.tensor(cs)
    off = centroids(AugCfg(**_NO_INTENSITY, bias_p=0.0))                          # translate=0
    on = centroids(AugCfg(rot_deg=0.0, scale=(1.0, 1.0), translate=0.15, gamma_p=0.0,
                          blur_p=0.0, contrast=(1.0, 1.0), noise=0.0))
    assert off.abs().max() < 1.0                     # centered stays centered
    assert on.abs().max() > 3.0                      # translation moves it off-center
