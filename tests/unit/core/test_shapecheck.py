"""core.shapecheck — the @shapecheck decorator (bd cardiac-seg-zwno). In the test env beartype is
installed, so the decorator enforces jaxtyping shapes at call time (a wrong shape raises); the
boundary functions that wear it are exercised in test_shape_guards."""
import numpy as np
import pytest
from jaxtyping import Float, TypeCheckError

from core.shapecheck import shapecheck, shapecheck_off


@shapecheck
def _sum(x: Float[np.ndarray, "d h w"]) -> float:
    return float(x.sum())


@shapecheck_off
def _sum_hot(x: Float[np.ndarray, "d h w"]) -> float:
    return float(x.sum())


def test_correct_shape_passes():
    assert _sum(np.ones((2, 4, 4), np.float32)) == 32.0


def test_shapecheck_off_is_inert_even_in_test_env():
    """The hot-path escape pins checker=None, so a wrong-ndim array does NOT raise — the annotation is
    documentation only, no per-call check, even where beartype is installed."""
    assert _sum_hot(np.ones(4, np.float32)) == 4.0     # 1D where "d h w" expected — not checked


def test_wrong_shape_raises_in_test_env():
    """beartype is present (dev extra) -> the decorator is live, so a wrong-ndim array raises."""
    with pytest.raises(TypeCheckError):
        _sum(np.ones(4, np.float32))


def test_shapecheck_is_a_decorator():
    """It wraps a function and returns a callable (identity-inert in prod, enforcing here)."""
    assert callable(shapecheck)
    assert callable(_sum)
