"""core.types tests (thin): the module is pure type-alias vocabulary — no runtime logic.

Just assert the aliases are importable and are the expected underlying types, so the mirror
exists and a rename/removal breaks a test.
"""
import numpy as np

from core import types


def test_array_aliases_are_ndarray():
    """Volume / Slice2D / Image / Mask / Batch all alias np.ndarray (shape documented, not enforced)."""
    assert types.Volume is np.ndarray
    assert types.Slice2D is np.ndarray
    assert types.Image is np.ndarray
    assert types.Mask is np.ndarray
    assert types.Batch is np.ndarray


def test_spacing_is_float_triple():
    """Spacing = (z, y, x) mm -> a 3-tuple of float."""
    assert types.Spacing == tuple[float, float, float]
