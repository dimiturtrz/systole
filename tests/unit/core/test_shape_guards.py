"""jaxtyping + beartype tensor-shape enforcement (bd cardiac-seg-zwno).

Proves the `@shapecheck` decorator makes the boundary annotations LIVE when beartype is installed (the
test env): a wrong-ndim or mismatched-shape array raises at the call instead of silently broadcasting.
Measure is the cheap boundary to exercise (no model/GPU) — the same decorator guards
Inference.predict_volume and the synth painter.
"""
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from core.measure import Measure

_SPACING = (10.0, 1.5, 1.5)


def test_correct_shape_passes():
    """A [D,H,W] integer mask is accepted (the annotation is satisfied, not merely ignored)."""
    mask = np.zeros((4, 32, 32), np.uint8)
    mask[1:3, 8:16, 8:16] = 3
    assert Measure.label_volume_ml(mask, 3, _SPACING) > 0


def test_wrong_ndim_raises():
    """A 2D [H,W] mask where [D,H,W] is required raises — the shape check fires under the hook."""
    with pytest.raises(TypeCheckError):
        Measure.label_volume_ml(np.zeros((32, 32), np.uint8), 3, _SPACING)


def test_mismatched_ed_es_shapes_raise():
    """ejection_fraction ties ed_mask and es_mask to the SAME 'd h w' — different grids can't be paired."""
    ed = np.zeros((4, 32, 32), np.uint8)
    es = np.zeros((5, 32, 32), np.uint8)   # different D — a real bug (mismatched phases)
    with pytest.raises(TypeCheckError):
        Measure.ejection_fraction(ed, es, _SPACING)


def test_float_prob_wrong_ndim_raises():
    """expected_volume_ml wants a [D,H,W] float prob map; a flat vector is rejected."""
    with pytest.raises(TypeCheckError):
        Measure.expected_volume_ml(np.zeros(32, np.float32), _SPACING)
