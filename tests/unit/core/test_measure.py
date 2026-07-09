"""Pure spacing-aware measurements — equivalence classes over the Bland-Altman EF agreement and the
soft expected-volume readout. The hard volume/EF path is exercised via test_volumes + the adapters;
here we pin the statistics helper (sample-count classes + NaN handling) and the probabilistic volume.
"""
import numpy as np

from core.measure import LOA_Z, Measure


def test_ef_statistics_empty_is_all_nan():
    """n=0 class: no valid pairs -> every stat NaN, n=0 (never a divide-by-zero)."""
    s = Measure.ef_statistics([], [])
    assert s["n"] == 0
    assert all(np.isnan(s[k]) for k in ("bias", "sd", "mae", "mean_gt"))
    assert all(np.isnan(v) for v in s["loa"])


def test_ef_statistics_single_pair_sd_zero():
    """n=1 boundary: sample SD is undefined (ddof=1) -> defined as 0.0, LoA collapses to the bias."""
    s = Measure.ef_statistics([50.0], [56.0])
    assert s["n"] == 1
    assert s["bias"] == 6.0 and s["sd"] == 0.0
    assert s["mae"] == 6.0 and s["mean_gt"] == 50.0
    assert s["loa"] == [6.0, 6.0]


def test_ef_statistics_many_pairs():
    """n>1 class: bias = mean(pred-gt), sd = sample SD (ddof=1), LoA = bias +- 1.96 sd, MAE = mean|d|."""
    g = np.array([40.0, 50.0, 60.0])
    p = np.array([44.0, 48.0, 66.0])          # diffs +4,-2,+6
    s = Measure.ef_statistics(g, p)
    assert s["n"] == 3
    assert abs(s["bias"] - (8 / 3)) < 1e-9
    assert abs(s["sd"] - np.std([4, -2, 6], ddof=1)) < 1e-9
    assert abs(s["mae"] - 4.0) < 1e-9         # mean(|4|,|2|,|6|)
    assert abs(s["mean_gt"] - 50.0) < 1e-9
    assert abs(s["loa"][0] - (s["bias"] - LOA_Z * s["sd"])) < 1e-9


def test_ef_statistics_drops_nan_pairs():
    """NaN class: an undefined EF (EDV<=0 upstream) is dropped before stats — n counts valid pairs only."""
    g = np.array([40.0, np.nan, 60.0])
    p = np.array([44.0, 30.0, np.nan])        # pair 0 valid; pairs 1,2 have a NaN
    s = Measure.ef_statistics(g, p)
    assert s["n"] == 1                         # only pair 0 survives
    assert s["bias"] == 4.0


def test_expected_volume_ml_is_prob_mass_times_voxel():
    """Soft readout: E[vol] = (Σ prob) x voxel volume — a p=0.5 boundary voxel contributes half a voxel."""
    vv = Measure.voxel_volume_ml((10.0, 1.5, 1.5))
    prob = np.full((2, 4, 4), 0.5, dtype=np.float32)   # 32 voxels x 0.5 = 16 voxels of mass
    assert abs(Measure.expected_volume_ml(prob, (10.0, 1.5, 1.5)) - 16.0 * vv) < 1e-4
