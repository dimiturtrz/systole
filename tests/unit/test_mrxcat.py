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
