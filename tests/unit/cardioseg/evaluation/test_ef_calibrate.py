"""EF bias calibration: axis_report scores agreement before/after applying a fitted linear correction
to one axis' EF pairs. Pure — no model/data (the CLI run() is the network+GPU shell around this)."""
import numpy as np

from cardioseg.evaluation.ef_calibrate import EfCalibrate
from core.measure import EfCalibration, Measure


def _rows(gt, pred):
    return [{"ef_gt": g, "ef_pred": p} for g, p in zip(gt, pred, strict=True)]


def test_axis_report_calibration_removes_the_fitted_bias():
    """Fit on an axis, report it back: the calibrated bias collapses to ~0 (in-sample the fit is exact-ish)."""
    gt = np.array([30.0, 45.0, 55.0, 62.0, 70.0])
    pred = gt - 6.0                                   # a constant -6 EF bias (masks good, ratio low)
    cal = Measure.fit_ef_calibration(gt, pred)
    ax = EfCalibrate.axis_report(cal, _rows(gt, pred))
    assert ax["n"] == 5
    assert ax["bias"][0] == -6.0                      # uncalibrated bias
    assert abs(ax["bias"][1]) < 0.1                   # calibrated ~0
    assert ax["mae"][1] <= ax["mae"][0]               # never worse in-sample


def test_axis_report_identity_calibration_is_a_noop():
    """Identity calibration (a degenerate/absent fit) leaves every stat unchanged before==after."""
    gt = np.array([40.0, 55.0, 65.0])
    pred = np.array([44.0, 50.0, 68.0])
    ax = EfCalibrate.axis_report(EfCalibration(1.0, 0.0), _rows(gt, pred))
    assert ax["mae"][0] == ax["mae"][1]
    assert ax["bias"][0] == ax["bias"][1]
    assert ax["loa"][0] == ax["loa"][1]


def test_axis_report_transfer_bias_can_survive():
    """Transfer class: a fit from one axis applied to a DIFFERENT-biased axis need not zero its bias —
    post-hoc calibration is domain-shift-limited (the point of reporting test separately from val)."""
    cal = EfCalibration(1.0, 6.0)                     # a +6 correction fit elsewhere
    gt = np.array([30.0, 50.0, 70.0])
    pred = gt - 12.0                                  # this axis is biased -12, not -6
    ax = EfCalibrate.axis_report(cal, _rows(gt, pred))
    assert ax["bias"][0] == -12.0 and ax["bias"][1] == -6.0   # correction under-shoots the larger shift
