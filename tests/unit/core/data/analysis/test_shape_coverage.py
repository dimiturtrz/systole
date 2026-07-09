"""Shape-coverage analysis (core.data.analysis.shape_coverage). The testable cores are shape_features
(interpretable scale/position-invariant descriptors for one 2D label map) and coverage (does the synth
shape cloud cover — and extrapolate beyond — the real cloud). Fed synthetic label maps; the CLI (main,
plotting) is coverage-omitted."""
import numpy as np

from core.data.analysis.shape_coverage import _MIN_FG_PX, ShapeCoverage


def _ring_mask(scale=1.0, cx=20, cy=20):
    """A canonical LV-ish label map: myo ring (2) around a cavity (3), plus an RV blob (1). All radii and
    the RV blob scale together with `scale` so composition fractions are genuinely scale-invariant."""
    size = int(40 * scale)
    r_out, r_in = 15 * scale, 9 * scale
    cx, cy = cx * scale, cy * scale
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    m = np.zeros((size, size), np.uint8)
    m[d <= r_out] = 2                                  # myo
    m[d <= r_in] = 3                                   # cavity overrides
    m[int(5 * scale):int(12 * scale), int(3 * scale):int(10 * scale)] = 1   # RV blob (scaled)
    return m


def test_shape_features_empty_below_threshold():
    """A near-empty mask (< _MIN_FG_PX foreground) yields None (shape stats unstable)."""
    m = np.zeros((40, 40), np.uint8)
    m[0, :3] = 2                                       # 3 fg px < 40
    assert m[m > 0].size < _MIN_FG_PX
    assert ShapeCoverage.shape_features(m) is None


def test_shape_features_length_and_fractions():
    """Seven descriptors; the three class-area fractions sum to 1 (they partition the foreground)."""
    f = ShapeCoverage.shape_features(_ring_mask())
    assert f is not None and f.shape == (7,)
    assert abs(f[0] + f[1] + f[2] - 1.0) < 1e-9        # rv/n + myo/n + lvc/n == 1


def test_shape_features_scale_invariant_composition():
    """Class-area FRACTIONS are scale-invariant: a 2x-larger heart has the same composition descriptors."""
    small = ShapeCoverage.shape_features(_ring_mask(scale=1.0))
    big = ShapeCoverage.shape_features(_ring_mask(scale=2.0))
    assert np.allclose(small[:3], big[:3], atol=0.02)  # composition fractions stable across scale


def test_feats_from_masks_skips_empty():
    """_feats_from_masks stacks only the non-empty masks, giving an [N,7] matrix."""
    masks = [_ring_mask(), np.zeros((40, 40), np.uint8), _ring_mask(cx=22)]
    feats = ShapeCoverage._feats_from_masks(masks)
    assert feats.shape == (2, 7)


def test_feats_from_masks_all_empty_is_empty():
    """No usable masks -> a well-formed (0,7) array (not a crash)."""
    feats = ShapeCoverage._feats_from_masks([np.zeros((40, 40), np.uint8)])
    assert feats.shape == (0, 7)


def test_coverage_identical_clouds_full_coverage():
    """When synth == real, every real point has a zero-distance synth neighbour -> coverage 1.0."""
    real = ShapeCoverage._feats_from_masks([_ring_mask(cx=c) for c in range(18, 26)])
    cov = ShapeCoverage.coverage(real, real.copy())
    assert cov["coverage_at_synth_radius"] == 1.0
    assert cov["real_to_synth_nn_median"] == 0.0
    assert cov["n_real"] == cov["n_synth"] == len(real)


def test_coverage_disjoint_synth_extrapolates():
    """A synth cloud far from real: real is poorly covered and synth extrapolates beyond real."""
    real = ShapeCoverage._feats_from_masks([_ring_mask(cx=c) for c in range(18, 24)])
    synth = real.copy()
    synth[:, 0] += np.arange(1.0, len(synth) + 1.0)     # spread synth out well beyond the real p95
    cov = ShapeCoverage.coverage(real, synth)
    assert cov["coverage_at_synth_radius"] < 1.0
    assert cov["synth_beyond_real_frac"] > 0.0
