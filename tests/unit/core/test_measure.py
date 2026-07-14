"""Pure spacing-aware measurements — equivalence classes over the Bland-Altman EF agreement and the
soft expected-volume readout. The hard volume/EF path is exercised via test_volumes + the adapters;
here we pin the statistics helper (sample-count classes + NaN handling) and the probabilistic volume.
"""
import numpy as np

from core.measure import LOA_Z, EfCalibration, Measure


def test_ef_statistics_empty_is_all_nan():
    """n=0 class: no valid pairs -> every stat NaN, n=0 (never a divide-by-zero)."""
    ef_stats = Measure.ef_statistics([], [])
    assert ef_stats.n == 0
    assert all(np.isnan(getattr(ef_stats, field)) for field in ("bias", "sd", "mae", "mean_gt"))
    assert all(np.isnan(v) for v in ef_stats.loa)


def test_ef_statistics_single_pair_sd_zero():
    """n=1 boundary: sample SD is undefined (ddof=1) -> defined as 0.0, LoA collapses to the bias."""
    ef_stats = Measure.ef_statistics([50.0], [56.0])
    assert ef_stats.n == 1
    assert ef_stats.bias == 6.0 and ef_stats.sd == 0.0
    assert ef_stats.mae == 6.0 and ef_stats.mean_gt == 50.0
    assert ef_stats.loa == [6.0, 6.0]


def test_ef_statistics_many_pairs():
    """n>1 class: bias = mean(pred-gt), sd = sample SD (ddof=1), LoA = bias +- 1.96 sd, MAE = mean|d|."""
    gt = np.array([40.0, 50.0, 60.0])
    pred = np.array([44.0, 48.0, 66.0])          # diffs +4,-2,+6
    ef_stats = Measure.ef_statistics(gt, pred)
    assert ef_stats.n == 3
    assert abs(ef_stats.bias - (8 / 3)) < 1e-9
    assert abs(ef_stats.sd - np.std([4, -2, 6], ddof=1)) < 1e-9
    assert abs(ef_stats.mae - 4.0) < 1e-9         # mean(|4|,|2|,|6|)
    assert abs(ef_stats.mean_gt - 50.0) < 1e-9
    assert abs(ef_stats.loa[0] - (ef_stats.bias - LOA_Z * ef_stats.sd)) < 1e-9


def test_ef_statistics_drops_nan_pairs():
    """NaN class: an undefined EF (EDV<=0 upstream) is dropped before stats — n counts valid pairs only."""
    gt = np.array([40.0, np.nan, 60.0])
    pred = np.array([44.0, 30.0, np.nan])        # pair 0 valid; pairs 1,2 have a NaN
    ef_stats = Measure.ef_statistics(gt, pred)
    assert ef_stats.n == 1                         # only pair 0 survives
    assert ef_stats.bias == 4.0


def test_fit_ef_calibration_recovers_linear_bias():
    """Exact-fit class: pred = (gt - b)/a  -> fitting gt ~ a*pred + b recovers (a,b), apply undoes it."""
    gt = np.array([30.0, 45.0, 55.0, 70.0])
    pred = 0.8 * gt - 4.0                              # a known affine distortion of the true EF
    cal = Measure.fit_ef_calibration(gt, pred)
    assert abs(cal.slope - 1.25) < 1e-6 and abs(cal.intercept - 5.0) < 1e-6   # inverse of (0.8, -4)
    assert np.allclose(cal.apply(pred), gt)


def test_fit_ef_calibration_degenerate_is_identity():
    """Boundary classes: <2 pairs, or no spread in pred (all equal) -> identity (no distortion)."""
    assert Measure.fit_ef_calibration([50.0], [40.0]) == EfCalibration(1.0, 0.0)
    assert Measure.fit_ef_calibration([40.0, 60.0], [50.0, 50.0]) == EfCalibration(1.0, 0.0)


def test_fit_ef_calibration_drops_nan_pairs():
    """NaN class: undefined-EF pairs dropped before the fit (a NaN in either member removes the pair)."""
    gt = np.array([30.0, np.nan, 55.0, 70.0])
    pred = np.array([20.0, 40.0, np.nan, 60.0])       # only pairs 0 and 3 survive -> 2 points, a clean line
    cal = Measure.fit_ef_calibration(gt, pred)
    assert abs(cal.slope - 1.0) < 1e-6 and abs(cal.intercept - 10.0) < 1e-6   # (30,20),(70,60): gt=pred+10


def test_ef_calibration_apply_passes_nan_through():
    """apply is a plain affine map; a NaN EF (undefined) stays NaN, never a fabricated corrected value."""
    out = EfCalibration(1.25, 5.0).apply(np.array([40.0, np.nan]))
    assert out[0] == 55.0 and np.isnan(out[1])


def test_expected_volume_ml_is_prob_mass_times_voxel():
    """Soft readout: E[vol] = (Σ prob) x voxel volume — a p=0.5 boundary voxel contributes half a voxel."""
    vv = Measure.voxel_volume_ml((10.0, 1.5, 1.5))
    prob = np.full((2, 4, 4), 0.5, dtype=np.float32)   # 32 voxels x 0.5 = 16 voxels of mass
    assert abs(Measure.expected_volume_ml(prob, (10.0, 1.5, 1.5)) - 16.0 * vv) < 1e-4
