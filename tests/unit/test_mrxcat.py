"""MRXCAT2.0 source adapter (core.data.dynamic.mrxcat). The pure, testable core is the label remap
to_canonical — verified geometrically on the bundled phantom (myo ring encloses LV-cav); here we pin
the equivalence classes of the code→canonical mapping so a scheme change can't silently corrupt pools."""
import numpy as np
from core.data.dynamic.mrxcat import to_canonical


def test_cardiac_codes_map_to_canonical():
    # MRXCAT myLabels: LV_wall=1, RV_wall=2, LV_blood=5, RV_blood=6 -> canonical 0 bg/1 RV/2 myo/3 LV-cav
    raw = np.array([[1, 5, 6], [2, 0, 36]])          # myo, LV-cav, RV-cav | RV-wall, air, aorta
    got = to_canonical(raw)
    assert got.tolist() == [[2, 3, 1], [0, 0, 0]]    # RV-wall/air/aorta -> bg (no canonical class)
    assert got.dtype == np.uint8


def test_unmapped_codes_are_background():
    # one representative per non-cardiac XCAT group (muscle/blood7/liver/fat/bone) -> all background
    raw = np.array([3, 4, 7, 8, 13, 50, 31, 99])
    assert (to_canonical(raw) == 0).all()


def test_empty_and_allbg_are_zero():
    assert to_canonical(np.zeros((4, 4), int)).sum() == 0


def test_tissue_map_keeps_heart_and_surrounding_organs():
    """Whole-FOV paint map (q4ww): heart codes → canonical heart classes; surrounding organs kept as
    paintable tissue classes (lung/liver/fat), body soft tissue → muscle, outside/bone → bg. NB code 2
    is a BROAD raw-XCAT label (not just RV wall) → muscle, not myo (render caught the stray-myo bug)."""
    from core.data.dynamic.mrxcat import to_tissue_map
    raw = np.array([[1, 5, 6], [15, 13, 50], [2, 9, 0], [31, 0, 0]])
    #                myo LVcav RVcav | lung liver fat | broad→musc soft→musc outside | bone outside outside
    got = to_tissue_map(raw)
    assert got.tolist() == [[2, 3, 1], [4, 5, 7], [6, 6, 0], [0, 0, 0]]
    assert got.dtype == np.uint8


def test_heart_crop_scale_recovers_offcentre_heart():
    """Regression: MRXCAT hearts are small + OFF-CENTRE in a big whole-torso frame; a plain centre
    fit_square crops them away (empty pool). _heart_crop_scale must recover + centre them."""
    from core.data.dynamic.mrxcat import _heart_crop_scale
    big = np.zeros((920, 920), np.uint8)                 # whole-torso-sized frame
    big[60:90, 100:130] = 2                              # myo ring-ish, off-centre (top-left)
    big[68:82, 108:122] = 3                              # LV-cav inside
    big[60:90, 130:150] = 1                              # RV-cav beside
    sq = _heart_crop_scale(big, size=128, target_px=80)
    assert sq.shape == (128, 128)
    assert set(np.unique(sq)) >= {0, 1, 2, 3}            # all classes survived the crop+scale
    # heart is now roughly centred, not lost to a corner
    ys, xs = np.where(sq > 0)
    assert 30 < ys.mean() < 98 and 30 < xs.mean() < 98
