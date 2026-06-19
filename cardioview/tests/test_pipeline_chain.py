"""Module-pair chain tests for cardioview geometry — the crop pipeline the exporters use:
keep_largest -> bbox_slices -> crop, and the ED/ES -> sampled-frame mapping (nearest_index).
Pure numpy, no torch/vtk."""
import numpy as np

from cardioview.geometry import keep_largest, bbox_slices, nearest_index

SPACING = (10.0, 1.5, 1.5)  # (z, y, x) mm


def _mask_with_island():
    """A main blob + one far stray voxel (model false positive)."""
    m = np.zeros((6, 40, 40), bool)
    m[2:5, 10:20, 10:20] = True   # main component
    m[0, 38, 38] = True           # island, far corner
    return m


# --- keep_largest -> bbox_slices -> crop -----------------------------------

def test_keep_largest_tightens_the_bbox_chain():
    """Cropping after keep_largest excludes the island that would otherwise widen the box."""
    raw = _mask_with_island()
    raw_box = bbox_slices(raw, SPACING, margin_mm=0)
    chained_box = bbox_slices(keep_largest(raw), SPACING, margin_mm=0)
    # the island drags the raw box out to the corner on every axis; the chain stays tight
    for ax in range(3):
        assert (chained_box[ax].stop - chained_box[ax].start) <= (raw_box[ax].stop - raw_box[ax].start)
    # the island voxel is outside the chained crop, inside the raw one
    assert not (chained_box[0].start <= 0 < chained_box[0].stop and chained_box[1].start <= 38 < chained_box[1].stop)


def test_crop_preserves_the_kept_component():
    """bbox_slices output is a valid crop window for keep_largest output — no kept voxel lost."""
    kept = keep_largest(_mask_with_island())
    box = bbox_slices(kept, SPACING, margin_mm=0)
    cropped = kept[box]
    assert cropped.sum() == kept.sum()          # every kept voxel is inside the box
    assert cropped.shape == (3, 10, 10)         # tight to the main blob


def test_chain_on_clean_mask_is_identity_box():
    """No island -> keep_largest is a no-op -> same box (chain doesn't distort a clean input)."""
    clean = np.zeros((6, 40, 40), bool)
    clean[2:5, 10:20, 10:20] = True
    assert bbox_slices(keep_largest(clean), SPACING, 0) == bbox_slices(clean, SPACING, 0)


# --- ED/ES frame -> sampled-frame index ------------------------------------

def test_nearest_index_maps_phase_to_sampled_frame():
    """The cine is sampled at a stride; ED/ES frame numbers map to the closest sampled frame."""
    sampled = [0, 4, 8, 12]                      # e.g. every-4th-frame export
    assert nearest_index(sampled, 5) == 1        # ED=5  -> frame 4 (index 1)
    assert nearest_index(sampled, 11) == 3       # ES=11 -> frame 12 (index 3)
    # chains into selecting that frame's mask without an out-of-range index
    masks = [np.full((2, 2), i, np.uint8) for i in range(len(sampled))]
    assert masks[nearest_index(sampled, 5)][0, 0] == 1
