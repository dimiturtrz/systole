"""core.shapecheck — the @shapecheck decorator (bd cardiac-seg-zwno). In the test env beartype is
installed, so the decorator enforces jaxtyping shapes at call time (a wrong shape raises); the
boundary functions that wear it are exercised in test_shape_guards."""
import numpy as np
import pytest
from jaxtyping import Float, TypeCheckError

from core.shapecheck import shapecheck, shapecheck_off
from core.types import Real


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


@shapecheck
def _scalars(a: Real, b: float, c: int) -> float:
    return float(a) + b + c


def test_scalar_convention_tower_and_numpy_aliases():
    """bd dnx6 convention: the numeric tower lets an int satisfy `float` (and np.float64, a real float
    subclass); the Real/Integral aliases admit np.float32/np.int* that are NOT python-scalar subclasses."""
    assert _scalars(np.float32(1.0), 2, 3) == 6.0        # Real admits np.float32; int->float via tower
    assert _scalars(1.0, np.float64(2.0), 3) == 6.0      # np.float64 is a float subclass (tower not needed)


def test_bare_float_still_rejects_np_float32():
    """A param left as bare `float` (not Real) still rejects np.float32 — the alias is required where a
    numpy scalar can arrive (documents WHY Spacing/Real exist)."""
    with pytest.raises(TypeCheckError):
        _scalars(1.0, np.float32(2.0), 3)                # b: float (not Real) <- np.float32 rejected


def test_wrong_shape_raises_in_test_env():
    """beartype is present (dev extra) -> the decorator is live, so a wrong-ndim array raises."""
    with pytest.raises(TypeCheckError):
        _sum(np.ones(4, np.float32))


def test_shapecheck_is_a_decorator():
    """It wraps a function and returns a callable (identity-inert in prod, enforcing here)."""
    assert callable(shapecheck)
    assert callable(_sum)
