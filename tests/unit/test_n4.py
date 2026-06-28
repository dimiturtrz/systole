"""Unit tests for N4 bias-field correction."""
import numpy as np
import pytest

pytest.importorskip("SimpleITK")
from core.preprocessing.n4 import n4_bias


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
