"""Static geometry/biomarker panel (core.data.analysis.static_compare, bd uy4d): mask-derived metric
distributions compared synth-vs-real by W1. Contract: geom_metrics returns interpretable geometry;
compare returns per-metric W1 that is ~0 for identical sets and grows when a metric shifts."""
import numpy as np

from core.data.analysis.static_compare import geom_metrics, compare


def _mask(lvc_r=4, myo_r=7, h=48, w=48):
    m = np.zeros((h, w), np.uint8)
    yy, xx = np.ogrid[:h, :w]
    d = (yy - h // 2) ** 2 + (xx - w // 2) ** 2
    m[d <= myo_r ** 2] = 2               # myo disc
    m[d <= lvc_r ** 2] = 3               # LV cavity inside
    m[(yy - h // 2) ** 2 + (xx - (w // 2 + 12)) ** 2 <= 5 ** 2] = 1   # RV cavity beside
    return m


def test_geom_metrics_are_sane():
    g = geom_metrics(_mask())
    assert g is not None
    assert g["lvc_area"] > 0 and g["myo_area"] > g["lvc_area"]     # myo ring bigger than cavity
    assert 0 < g["lvc_sphericity"] <= 2.0                          # a disc is ~round (discrete P inflates small)
    assert g["myo_thickness"] > 0
    assert geom_metrics(np.zeros((16, 16), np.uint8)) is None       # empty -> None


def test_compare_zero_for_identical_grows_on_shift():
    same = [_mask() for _ in range(20)]
    w1_same = compare(same, same)
    assert all(v["w1"] < 1e-6 for v in w1_same.values())           # identical sets -> W1 0
    bigger_lv = [_mask(lvc_r=6) for _ in range(20)]                # larger LV cavity
    w1_shift = compare(same, bigger_lv)
    assert w1_shift["lvc_area"]["w1"] > 0                          # metric moved -> W1 > 0
