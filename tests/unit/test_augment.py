"""Unit tests for GPU-batched augmentation (runs on CPU tensors here)."""
import pytest

torch = pytest.importorskip("torch")

from cardioseg.training.augment import augment_batch


def _batch():
    torch.manual_seed(0)
    img = torch.randn(4, 1, 64, 64)
    mask = torch.zeros(4, 64, 64, dtype=torch.long)
    mask[:, 20:44, 20:44] = 3   # LV-cav block
    mask[:, 10:16, 10:16] = 1   # RV blob
    return img, mask


def test_shapes_and_dtype_preserved():
    img, mask = _batch()
    a, b = augment_batch(img, mask)
    assert a.shape == img.shape
    assert b.shape == mask.shape
    assert b.dtype == torch.long          # mask stays integer labels


def test_mask_labels_stay_a_subset():
    """Nearest-neighbour sampling + zero padding -> only the input labels (or background)."""
    img, mask = _batch()
    _, b = augment_batch(img, mask)
    assert set(b.unique().tolist()).issubset({0, 1, 3})  # no interpolated/new labels


def test_image_finite_and_actually_changed():
    img, mask = _batch()
    a, _ = augment_batch(img, mask)
    assert torch.isfinite(a).all()        # no NaN/Inf from gamma/grid_sample
    assert not torch.allclose(a, img)     # augmentation did something


def test_deterministic_under_seed():
    img, mask = _batch()
    torch.manual_seed(7)
    a1, b1 = augment_batch(img.clone(), mask.clone())
    torch.manual_seed(7)
    a2, b2 = augment_batch(img.clone(), mask.clone())
    assert torch.allclose(a1, a2)
    assert torch.equal(b1, b2)            # same RNG seed -> identical augmentation
