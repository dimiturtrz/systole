"""Unit tests for N4 bias-field correction."""
import numpy as np
import pytest
import torch

pytest.importorskip("SimpleITK")
from core.preprocessing.n4 import N4Cfg, _n4_sitk, _smooth3d, n4_bias, n4_gpu


def _biased():
    """Uniform tissue box + a smooth L->R multiplicative bias (the thing N4 should remove)."""
    D, H, W = 8, 96, 96
    base = np.zeros((D, H, W), np.float32)
    base[:, 24:72, 24:72] = 100.0
    xx = np.broadcast_to(np.linspace(0.5, 2.0, W), (D, H, W))   # 0.5..2.0 across x
    return base, (base * xx).astype(np.float32)


def _cov(a, fg):
    return float(np.std(a[fg]) / (np.mean(a[fg]) + 1e-6))


def test_removes_smooth_bias():
    base, biased = _biased()
    fg = base > 0
    out = n4_bias(biased, (8.0, 1.5, 1.5))
    assert _cov(out, fg) < _cov(biased, fg) / 3   # markedly more uniform


def test_shape_dtype_finite():
    _, biased = _biased()
    out = n4_bias(biased, (8.0, 1.5, 1.5))
    assert out.shape == biased.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_flat_input_no_crash():
    flat = np.ones((4, 32, 32), np.float32)
    out = n4_bias(flat)                  # no spacing, nothing to correct
    assert out.shape == flat.shape and np.isfinite(out).all()


def test_n4cfg_validates_ranges():
    """N4Cfg boundary class: shrink>=1, fwhm>0 enforced by pydantic."""
    c = N4Cfg(shrink=2, iters=(10, 10), fwhm=0.2)
    assert c.shrink == 2 and c.iters == (10, 10)
    with pytest.raises(ValueError, match="shrink"):
        N4Cfg(shrink=0)
    with pytest.raises(ValueError, match="fwhm"):
        N4Cfg(fwhm=0.0)


# --- pure-torch path (n4_gpu / _smooth3d), CPU device so it runs on CI ---


def test_smooth3d_preserves_mean_and_smooths():
    """_smooth3d class: normalized Gaussian conserves the DC level; variance drops."""
    rng = np.random.default_rng(1)
    v = torch.as_tensor(rng.normal(5.0, 1.0, (6, 24, 24)).astype(np.float32))
    s = _smooth3d(v, sigma=2.0)
    assert s.shape == v.shape
    assert abs(float(s.mean()) - float(v.mean())) < 0.1   # DC preserved
    assert float(s.std()) < float(v.std())                 # smoothed


def test_n4_gpu_removes_bias_cpu():
    """n4_gpu class: biased volume on CPU device -> more uniform foreground, finite."""
    base, biased = _biased()
    fg = base > 0
    out = n4_gpu(biased, device="cpu", iters=6)
    assert out.shape == biased.shape and out.dtype == np.float32
    assert np.isfinite(out).all()
    assert _cov(out, fg) < _cov(biased, fg)


def test_n4_gpu_too_few_fg_passthrough():
    """n4_gpu boundary: < _MIN_FG_VOXELS positive voxels -> unchanged float32 passthrough."""
    tiny = np.zeros((2, 4, 4), np.float32)
    tiny[0, 0, 0] = 1.0                    # 1 positive voxel < 16
    out = n4_gpu(tiny, device="cpu")
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, tiny)


def test_n4_gpu_degenerate_range_breaks_early():
    """n4_gpu boundary: flat foreground (zero log-range) breaks the loop, returns finite."""
    flat = np.full((3, 16, 16), 7.0, np.float32)
    out = n4_gpu(flat, device="cpu", iters=8)
    assert out.shape == flat.shape and np.isfinite(out).all()


def test_n4_sitk_error_fallback(monkeypatch):
    """_n4_sitk class: an ITK RuntimeError inside Execute -> pass the input through unchanged."""
    import core.preprocessing.n4 as n4mod

    class _Boom:
        def SetMaximumNumberOfIterations(self, *_): pass
        def SetBiasFieldFullWidthAtHalfMaximum(self, *_): pass
        def Execute(self, *_): raise RuntimeError("itk hiccup")

    monkeypatch.setattr(n4mod.sitk, "N4BiasFieldCorrectionImageFilter", _Boom)
    _, biased = _biased()
    out = _n4_sitk(biased, (8.0, 1.5, 1.5))
    np.testing.assert_array_equal(out, biased.astype(np.float32))
