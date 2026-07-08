"""Module-pair chain tests for cardioview geometry — the crop pipeline the exporters use:
bbox_slices -> crop, and the ED/ES -> sampled-frame mapping (nearest_index).
Pure numpy, no torch/vtk."""
import numpy as np

from cardioview.geometry import bbox_slices, nearest_index

SPACING = (10.0, 1.5, 1.5)  # (z, y, x) mm


def _heart_mask():
    """A single heart blob on an anisotropic short-axis grid."""
    m = np.zeros((6, 40, 40), bool)
    m[2:5, 10:20, 10:20] = True
    return m


# --- bbox_slices -> crop ----------------------------------------------------

def test_bbox_slices_output_is_a_valid_crop_window():
    """bbox_slices output crops the mask without losing a labeled voxel."""
    m = _heart_mask()
    box = bbox_slices(m, SPACING, margin_mm=0)
    cropped = m[box]
    assert cropped.sum() == m.sum()          # every True voxel is inside the box
    assert cropped.shape == (3, 10, 10)      # tight to the blob


def test_margin_widens_the_crop_but_stays_in_bounds():
    """A positive margin grows the box per axis (voxels = margin_mm / spacing), clamped to shape."""
    m = _heart_mask()
    tight = bbox_slices(m, SPACING, margin_mm=0)
    padded = bbox_slices(m, SPACING, margin_mm=12.0)
    for ax in range(3):
        assert (padded[ax].stop - padded[ax].start) >= (tight[ax].stop - tight[ax].start)
        assert padded[ax].start >= 0 and padded[ax].stop <= m.shape[ax]


# --- ED/ES frame -> sampled-frame index ------------------------------------

def test_nearest_index_maps_phase_to_sampled_frame():
    """The cine is sampled at a stride; ED/ES frame numbers map to the closest sampled frame."""
    sampled = [0, 4, 8, 12]                      # e.g. every-4th-frame export
    assert nearest_index(sampled, 5) == 1        # ED=5  -> frame 4 (index 1)
    assert nearest_index(sampled, 11) == 3       # ES=11 -> frame 12 (index 3)
    # chains into selecting that frame's mask without an out-of-range index
    masks = [np.full((2, 2), i, np.uint8) for i in range(len(sampled))]
    assert masks[nearest_index(sampled, 5)][0, 0] == 1
