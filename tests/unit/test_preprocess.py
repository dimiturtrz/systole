"""Unit tests for preprocessing transforms (no disk / no real data)."""
import numpy as np

from cardioseg.preprocessing.preprocess import zscore, resample_inplane


def test_zscore_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    img = rng.normal(50, 10, (4, 32, 32)).astype(np.float32)
    z = zscore(img)
    assert abs(z.mean()) < 1e-4
    assert abs(z.std() - 1.0) < 1e-2


def test_resample_inplane_changes_hw_not_slices():
    img = np.zeros((10, 64, 64), dtype=np.float32)
    out, sp = resample_inplane(img, (10.0, 3.0, 3.0), target_inplane=1.5)
    assert out.shape[0] == 10                 # slices preserved
    assert out.shape[1:] == (128, 128)        # 3.0 -> 1.5 mm doubles in-plane
    assert sp == (10.0, 1.5, 1.5)


def test_resample_mask_stays_integer_labels():
    mask = np.zeros((4, 32, 32), dtype=np.uint8)
    mask[:, 8:24, 8:24] = 3                    # LV-cavity-like block
    out, _ = resample_inplane(mask, (10.0, 1.5, 1.5), target_inplane=0.75, is_mask=True)
    assert set(np.unique(out).tolist()).issubset({0, 3})   # no interpolated labels
    assert out.dtype == np.uint8
