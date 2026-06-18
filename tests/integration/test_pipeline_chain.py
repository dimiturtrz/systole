"""Module-pair / pipeline-chain integration (data-free, synthetic).

The unit tests check each stage in isolation. These check that stages chain: A's
output is a valid input to B, and the A->B chain produces what each unit promises.
No ACDC data, no GPU — synthetic masks/volumes only, so these always run.
"""
import numpy as np

from cardioseg.preprocessing.preprocess import resample_inplane, zscore
from cardioseg.training.dataset import fit_square
from cardioseg.evaluation.measure import ejection_fraction, label_volume_ml
from cardioseg.evaluation.evaluate import dice, assd, surface_distances


def _lv_cube(d, hw, half, label=3):
    """[d, hw, hw] mask with a centred (2*half)^3 cube of `label` (a toy LV pool)."""
    m = np.zeros((d, hw, hw), dtype=np.uint8)
    c = hw // 2
    z0, z1 = d // 2 - half, d // 2 + half
    m[z0:z1, c - half:c + half, c - half:c + half] = label
    return m


# --- preprocess -> dataset ------------------------------------------------

def test_resample_then_fit_square_yields_model_input():
    """resample_inplane output is a valid fit_square input -> exact [size,size] slices."""
    vol = np.random.rand(3, 40, 50).astype(np.float32)
    res, sp = resample_inplane(vol, (10.0, 2.0, 2.0), target_inplane=1.5)  # 2/1.5 -> upsamples
    assert sp == (10.0, 1.5, 1.5)
    size = 64
    stack = np.stack([fit_square(s, size) for s in res])
    assert stack.shape == (3, size, size)          # chain lands on the model's input grid
    assert stack.dtype == res.dtype


def test_mask_chain_preserves_labels():
    """A label volume through resample(is_mask) -> fit_square stays integer-labelled."""
    mask = _lv_cube(3, 40, 8)
    res, _ = resample_inplane(mask, (10.0, 2.0, 2.0), target_inplane=1.5, is_mask=True)
    stack = np.stack([fit_square(s, 96) for s in res])  # 96 > resampled H,W -> pad path
    assert set(np.unique(stack).tolist()).issubset({0, 3})
    assert stack.shape == (3, 96, 96)


# --- measure <-> evaluate (one mask pair feeds both) ----------------------

def test_one_mask_pair_feeds_ef_and_overlap():
    """The same ED/ES masks are valid inputs to BOTH measure (EF) and evaluate (Dice/surface)."""
    ed = _lv_cube(8, 64, 10)   # larger pool (diastole)
    es = _lv_cube(8, 64, 6)    # smaller pool (systole)
    spacing = (10.0, 1.5, 1.5)

    ef, edv, esv = ejection_fraction(ed, es, spacing)
    assert edv > esv > 0                           # diastole larger
    assert 0 < ef < 100
    assert np.isclose(edv, label_volume_ml(ed, 3, spacing))

    # same masks, evaluate side
    assert dice(ed, ed, 3) == 1.0                  # self-overlap perfect
    assert dice(ed, es, 3) < 1.0                   # different pools -> imperfect
    assert assd(ed, es, 3) > 0                     # boundaries differ -> nonzero distance


def test_degenerate_ed_propagates_nan_not_crash():
    """Empty ED (equivalence class: EDV==0) -> EF NaN; evaluate stays defined on the same input."""
    empty = np.zeros((8, 64, 64), dtype=np.uint8)
    es = _lv_cube(8, 64, 6)
    ef, edv, _ = ejection_fraction(empty, es, (10.0, 1.5, 1.5))
    assert edv == 0 and np.isnan(ef)
    assert dice(empty, empty, 3) == 1.0            # vacuous overlap defined (no crash)
    sd = surface_distances(empty, es, 3)
    assert sd.size == 0 or np.all(np.isnan(sd))    # absent label -> empty/nan, not error
