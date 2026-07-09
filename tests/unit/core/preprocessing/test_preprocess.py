"""Unit tests for preprocessing transforms (no disk / no real data)."""
import numpy as np
import pytest

from core.preprocessing.preprocess import Preprocess


def test_zscore_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    img = rng.normal(50, 10, (4, 32, 32)).astype(np.float32)
    z = Preprocess.zscore(img)
    assert abs(z.mean()) < 1e-4
    assert abs(z.std() - 1.0) < 1e-2


def test_resample_inplane_changes_hw_not_slices():
    img = np.zeros((10, 64, 64), dtype=np.float32)
    out, sp = Preprocess.resample_inplane(img, (10.0, 3.0, 3.0), target_inplane=1.5)
    assert out.shape[0] == 10                 # slices preserved
    assert out.shape[1:] == (128, 128)        # 3.0 -> 1.5 mm doubles in-plane
    assert sp == (10.0, 1.5, 1.5)


def test_resample_mask_stays_integer_labels():
    mask = np.zeros((4, 32, 32), dtype=np.uint8)
    mask[:, 8:24, 8:24] = 3                    # LV-cavity-like block
    out, _ = Preprocess.resample_inplane(mask, (10.0, 1.5, 1.5), target_inplane=0.75, is_mask=True)
    assert set(np.unique(out).tolist()).issubset({0, 3})   # no interpolated labels
    assert out.dtype == np.uint8


# --- fit_square: equivalence classes over (H,W) vs size ---


def test_fit_square_already_square():
    """Class: exact size -> identity (no crop, no pad)."""
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.testing.assert_array_equal(Preprocess.fit_square(a, 4), a)


def test_fit_square_larger_is_centre_cropped():
    """Class: H,W > size -> centred crop, content preserved."""
    a = np.ones((8, 8), np.float32)
    a[3:5, 3:5] = 9.0                         # central 2x2 marker
    out = Preprocess.fit_square(a, 4)
    assert out.shape == (4, 4)
    assert (out == 9.0).sum() == 4            # marker survives the centred crop


def test_fit_square_smaller_is_centre_padded():
    """Class: H,W < size -> centred pad with pad_value, original in the middle."""
    a = np.full((2, 2), 5.0, np.float32)
    out = Preprocess.fit_square(a, 6, pad_value=-1.0)
    assert out.shape == (6, 6)
    assert (out == 5.0).sum() == 4            # original 2x2 preserved
    assert out[0, 0] == -1.0                  # corner is pad


def test_fit_square_anisotropic_tall_and_wide():
    """Class: mixed (crop one axis, pad the other)."""
    tall = np.ones((10, 2), np.float32)
    out = Preprocess.fit_square(tall, 6)
    assert out.shape == (6, 6)
    assert (out == 1.0).sum() == 6 * 2        # 6 rows kept, 2 cols kept, rest pad


def test_stack_slices_casts_and_stacks():
    """stack_slices: fit-squares each slice and casts the stack dtype."""
    slices = [np.ones((4, 4), np.float32), np.zeros((6, 6), np.float32)]
    out = Preprocess.stack_slices(slices, 5, dtype=np.uint8)
    assert out.shape == (2, 5, 5) and out.dtype == np.uint8


# --- blood_anchor: two-point affine vs z-score fallback ---


def test_blood_anchor_maps_air_zero_blood_one():
    """Class: enough blood+air voxels -> air->~0, blood->~1 (affine anchor)."""
    img = np.zeros((2, 20, 20), np.float32)
    gt = np.zeros((2, 20, 20), np.uint8)
    gt[:, :10, :] = 3                          # blood pool label
    img[gt == 0] = 30.0                        # air intensity
    img[gt == 3] = 130.0                       # blood intensity
    out = Preprocess.blood_anchor(img, gt)
    assert abs(float(np.median(out[gt == 0]))) < 1e-3
    assert abs(float(out[gt == 3].mean()) - 1.0) < 1e-3


def test_blood_anchor_falls_back_to_zscore():
    """Boundary: too few anchor voxels -> identical to plain z-score."""
    img = np.random.default_rng(0).normal(50, 5, (2, 8, 8)).astype(np.float32)
    gt = np.zeros((2, 8, 8), np.uint8)         # no blood -> fallback
    np.testing.assert_allclose(Preprocess.blood_anchor(img, gt), Preprocess.zscore(img), rtol=1e-5)


# --- preprocess_case: pure pipeline with an injected loader (no disk) ---


def _fake_loader(_patient_dir):
    rng = np.random.default_rng(2)
    img = rng.normal(50, 10, (4, 40, 40)).astype(np.float32)
    gt = np.zeros((4, 40, 40), np.uint8)
    gt[:, 10:30, 10:30] = 3
    return {"spacing": (10.0, 3.0, 3.0), "group": "DCM",
            "ED": {"img": img, "gt": gt}, "ES": {"img": img * 1.1, "gt": gt}}


def test_preprocess_case_zscore_default():
    """Class: default norm -> both frames resampled + z-scored, spacing updated."""
    out = Preprocess.preprocess_case("/tmp/patientX", _fake_loader, target_inplane=1.5)
    assert out["patient"] == "patientX" and out["group"] == "DCM"
    assert out["ed_img"].shape[1:] == (80, 80)          # 3.0 -> 1.5 doubles in-plane
    assert out["ed_gt"].dtype == np.uint8
    assert abs(float(out["ed_img"].mean())) < 1e-3       # z-scored
    np.testing.assert_allclose(out["spacing"], [10.0, 1.5, 1.5], rtol=1e-5)


def test_preprocess_case_blood_norm_and_missing_frame():
    """Class: norm='blood' path + a loader missing ES -> ES keys absent."""
    def _ed_only(_pd):
        d = _fake_loader(_pd); del d["ES"]; return d
    out = Preprocess.preprocess_case("/tmp/pY", _ed_only, norm="blood")
    assert "ed_img" in out and "es_img" not in out


def test_preprocess_case_with_n4():
    """Class: n4=True routes the image through N4 bias correction before z-score."""
    pytest.importorskip("SimpleITK")
    out = Preprocess.preprocess_case("/tmp/pN", _fake_loader, n4=True)
    assert np.isfinite(out["ed_img"]).all()
    assert abs(float(out["ed_img"].mean())) < 1e-3      # still z-scored after N4


def test_preprocess_case_with_nyul(monkeypatch):
    """Class: nyul_standard given -> nyul_transform applied before z-score (injected, no fit needed)."""
    import core.preprocessing.preprocess as pp
    seen = {}

    def _fake_nyul(img, standard, mask=None):
        seen["called"] = True
        return img
    monkeypatch.setattr(pp.Nyul, "transform", staticmethod(_fake_nyul))
    out = pp.Preprocess.preprocess_case("/tmp/pNy", _fake_loader, nyul_standard=[0.0, 0.5, 1.0])
    assert seen.get("called") and "ed_img" in out
