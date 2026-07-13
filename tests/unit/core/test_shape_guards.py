"""jaxtyping + beartype tensor-boundary enforcement (bd cardiac-seg-zwno, loosened bd cardiac-seg-ai0q).

Proves the `@shapecheck` decorator makes the boundary annotations LIVE when beartype is installed (the
test env). The specs are tail-dim / variadic on purpose (broadcasting-friendly): the reduction fns are
DTYPE-only (a voxel count doesn't care about ndim — 2D/3D/batched all valid), and ejection_fraction TIES
its ed/es masks to one grid. Measure is the cheap boundary to exercise (no model/GPU) — the same
decorator guards Inference.predict_volume and the synth painter.
"""
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from core.measure import Measure

_SPACING = (10.0, 1.5, 1.5)


def test_integer_mask_any_ndim_accepted():
    """Shape-agnostic: label_volume_ml is a reduction, so 2D and 3D integer masks both pass — the spec
    only pins the dtype, not the ndim (broadcasting freedom, bd ai0q)."""
    assert Measure.label_volume_ml(np.full((4, 32, 32), 3, np.uint8), 3, _SPACING) > 0   # 3D
    assert Measure.label_volume_ml(np.full((32, 32), 3, np.uint8), 3, _SPACING) > 0      # 2D — also fine


def test_wrong_dtype_mask_raises():
    """A float array where an integer label map is required raises — the dtype is the contract."""
    with pytest.raises(TypeCheckError):
        Measure.label_volume_ml(np.zeros((4, 32, 32), np.float32), 3, _SPACING)


def test_expected_volume_wrong_dtype_raises():
    """expected_volume_ml wants a float prob map (any shape); an integer array is rejected on dtype."""
    with pytest.raises(TypeCheckError):
        Measure.expected_volume_ml(np.zeros((4, 32, 32), np.int64), _SPACING)


def test_mismatched_ed_es_shapes_raise():
    """ejection_fraction ties ed_mask and es_mask to the SAME '*grid' — different grids can't be paired
    (a mismatched-phase bug), even though each alone is shape-agnostic."""
    ed = np.zeros((4, 32, 32), np.uint8)
    es = np.zeros((5, 32, 32), np.uint8)   # different D — a real bug (mismatched phases)
    with pytest.raises(TypeCheckError):
        Measure.ejection_fraction(ed, es, _SPACING)


def test_matched_ed_es_shapes_pass():
    """Same-grid ed/es resolve to a finite EF triple (the tie is satisfied)."""
    ed = np.zeros((4, 32, 32), np.uint8); ed[1:3, 8:24, 8:24] = 3
    es = np.zeros((4, 32, 32), np.uint8); es[1:3, 10:22, 10:22] = 3
    ef, edv, esv = Measure.ejection_fraction(ed, es, _SPACING)
    assert edv > esv > 0 and 0 < ef < 100
