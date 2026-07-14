"""core.types tests: the shape/units vocabulary (aliases + Spacing) AND the @shapecheck decorators that
live here (bd cardiac-seg-zwno; folded from test_shapecheck when shapecheck.py was collapsed into
core.types). beartype is a base dependency, so @shapecheck is LIVE — a wrong shape/dtype raises at the
call; @shapecheck_off is the never-checked hot-path escape."""
from numbers import Real

import numpy as np
import pytest
from jaxtyping import Float, TypeCheckError

from core import types
from core.types import shapecheck, shapecheck_off


def test_array_aliases_are_ndarray():
    """Volume / Slice2D / Image / Mask / Batch all alias np.ndarray (shape documented, not enforced)."""
    assert types.Volume is np.ndarray
    assert types.Slice2D is np.ndarray
    assert types.Image is np.ndarray
    assert types.Mask is np.ndarray
    assert types.Batch is np.ndarray


def test_spacing_is_real_triple():
    """Spacing = (z, y, x) mm -> a 3-tuple of the stdlib `numbers.Real` (store headers hand back
    np.float32, which registers as numbers.Real, so the runtime-checked boundary admits it)."""
    assert types.Spacing == tuple[Real, Real, Real]


# --- @shapecheck / @shapecheck_off ---

@shapecheck
def _sum(x: Float[np.ndarray, "d h w"]) -> float:
    return float(x.sum())


@shapecheck_off
def _sum_hot(x: Float[np.ndarray, "d h w"]) -> float:
    return float(x.sum())


@shapecheck
def _scalars(a: Real, b: float, c: int) -> float:
    return float(a) + b + c


def test_correct_shape_passes():
    assert _sum(np.ones((2, 4, 4), np.float32)) == 32.0


def test_wrong_shape_raises():
    """beartype is live (base dep) -> the decorator enforces, so a wrong-ndim array raises."""
    with pytest.raises(TypeCheckError):
        _sum(np.ones(4, np.float32))


def test_shapecheck_off_is_inert():
    """The hot-path escape pins checker=None, so a wrong-ndim array does NOT raise — the annotation is
    documentation only, no per-call check."""
    assert _sum_hot(np.ones(4, np.float32)) == 4.0     # 1D where "d h w" expected — not checked


def test_scalar_convention_tower_and_numbers_tower():
    """The numeric tower lets an int satisfy `float` (and np.float64, a real float subclass); the stdlib
    `numbers.Real` admits np.float32 which is NOT a python-float subclass."""
    assert _scalars(np.float32(1.0), 2, 3) == 6.0        # numbers.Real admits np.float32; int->float via tower
    assert _scalars(1.0, np.float64(2.0), 3) == 6.0      # np.float64 is a float subclass (tower not needed)


def test_bare_float_still_rejects_np_float32():
    """A param left as bare `float` (not numbers.Real) still rejects np.float32 — documents why the
    boundary uses Real where a numpy scalar can arrive."""
    with pytest.raises(TypeCheckError):
        _scalars(1.0, np.float32(2.0), 3)                # b: float (not Real) <- np.float32 rejected
