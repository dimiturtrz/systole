"""Unit tests for cardioview's pure mask geometry (no torch/vtk needed)."""
import numpy as np

from cardioview.geometry import keep_largest, bbox_slices


def test_keep_largest_drops_islands():
    a = np.zeros((10, 10, 10), bool)
    a[2:7, 2:7, 2:7] = True  # big blob (125 voxels)
    a[0, 0, 0] = True  # stray island
    out = keep_largest(a)
    assert out.sum() == 125
    assert not out[0, 0, 0]


def test_keep_largest_noop_when_single_component():
    a = np.zeros((5, 5, 5), bool)
    a[1:4, 1:4, 1:4] = True
    assert np.array_equal(keep_largest(a), a)


def test_bbox_slices_tight_with_zero_margin():
    a = np.zeros((10, 10, 10), bool)
    a[3:6, 4:8, 2:5] = True
    sl = bbox_slices(a, spacing=(1, 1, 1), margin_mm=0)
    assert sl == (slice(3, 6), slice(4, 8), slice(2, 5))


def test_bbox_slices_adds_margin_in_voxels_and_clamps():
    a = np.zeros((10, 10, 10), bool)
    a[5, 5, 5] = True
    # 4 mm margin at 2 mm spacing = 2 voxels each side; near edges clamp to bounds.
    sl = bbox_slices(a, spacing=(2, 2, 2), margin_mm=4)
    assert sl == (slice(3, 8), slice(3, 8), slice(3, 8))


def test_bbox_slices_anisotropic_spacing():
    a = np.zeros((10, 10, 10), bool)
    a[5, 5, 5] = True
    # z spacing 10 mm -> 12 mm margin = 1 voxel; in-plane 1.5 mm -> 8 voxels (clamped).
    sl = bbox_slices(a, spacing=(10.0, 1.5, 1.5), margin_mm=12.0)
    assert sl[0] == slice(4, 7)  # z: ±1
    assert sl[1] == slice(0, 10)  # y: ±8 clamped
    assert sl[2] == slice(0, 10)  # x: ±8 clamped


def test_bbox_slices_empty_mask_returns_full():
    a = np.zeros((4, 4, 4), bool)
    assert bbox_slices(a, spacing=(1, 1, 1)) == (slice(0, 4), slice(0, 4), slice(0, 4))
