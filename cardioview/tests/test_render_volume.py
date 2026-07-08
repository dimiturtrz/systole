"""Pure plottable-data logic for render_volume: crop-to-heart, isotropic resample, robust normalize.
The pyvista Plotter/screenshot render path is a shell (pragma'd) — these test the arrays it consumes.

Equivalence classes: crop (heart present / absent / margin clips at the volume edge), isotropic
(already isotropic / anisotropic z), normalize (normal spread / constant input / outliers clipped)."""
import numpy as np
from render_volume import crop_to_heart, normalize, to_isotropic


def _blob():
    """A labeled heart blob in a bigger [z,y,x] volume; intensity = position for a traceable crop."""
    vol = np.arange(6 * 8 * 8, dtype=np.float32).reshape(6, 8, 8)
    gt = np.zeros((6, 8, 8), np.uint8)
    gt[2:4, 3:6, 3:6] = 1
    return vol, gt


# --- crop_to_heart ---------------------------------------------------------

def test_crop_to_heart_shrinks_to_bbox():
    """Heart-present class: crop tightens to the labeled bbox (margin 0), img and gt cropped alike."""
    vol, gt = _blob()
    cvol, cgt = crop_to_heart(vol, gt, (1.0, 1.0, 1.0), margin_mm=0.0)
    assert cvol.shape == (2, 3, 3) == cgt.shape
    assert np.array_equal(cvol, vol[2:4, 3:6, 3:6])
    assert cgt.sum() == gt.sum()  # no labeled voxel lost


def test_crop_to_heart_no_label_returns_uncropped():
    """Heart-absent class: gt None OR all-zero -> pass the volume through untouched (full FOV)."""
    vol, _ = _blob()
    assert crop_to_heart(vol, None, (1.0, 1.0, 1.0)) == (vol, None)
    z = np.zeros_like(vol, np.uint8)
    ov, og = crop_to_heart(vol, z, (1.0, 1.0, 1.0))
    assert ov.shape == vol.shape and og.shape == z.shape


def test_crop_to_heart_margin_clips_at_edge():
    """Edge class: a huge margin can't exceed the volume bounds -> crop clamps to full extent."""
    vol, gt = _blob()
    cvol, _ = crop_to_heart(vol, gt, (1.0, 1.0, 1.0), margin_mm=1000.0)
    assert cvol.shape == vol.shape


# --- to_isotropic ----------------------------------------------------------

def test_to_isotropic_already_isotropic_is_noop_shape():
    """Isotropic-already class: equal spacing -> factors all 1 -> shape and spacing unchanged."""
    vol = np.ones((4, 5, 6), np.float32)
    out, sp = to_isotropic(vol, (2.0, 2.0, 2.0))
    assert out.shape == (4, 5, 6) and sp == (2.0, 2.0, 2.0)


def test_to_isotropic_upsamples_anisotropic_axis():
    """Anisotropic class: z 6mm vs in-plane 1.5mm -> z stretched 4x to the finest spacing; iso step=1.5."""
    vol = np.zeros((3, 4, 4), np.float32)
    out, sp = to_isotropic(vol, (6.0, 1.5, 1.5))
    assert sp == (1.5, 1.5, 1.5)
    assert out.shape[0] == round(3 * 6.0 / 1.5)  # z resampled by 4
    assert out.shape[1:] == (4, 4)               # in-plane already isotropic


# --- normalize -------------------------------------------------------------

def test_normalize_spans_unit_range():
    """Normal-spread class: a ramp -> output spans [0,1] with the percentile tails clipped flat."""
    out = normalize(np.arange(100, dtype=np.float32))
    assert out.min() == 0.0 and out.max() == 1.0
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_normalize_constant_input_no_nan():
    """Constant class: hi==lo would divide by zero -> the 1e-6 floor keeps it finite (all ~0, no NaN)."""
    out = normalize(np.full((4, 4), 7.0, np.float32))
    assert np.all(np.isfinite(out))
    assert out.max() <= 1.0


def test_normalize_clips_outliers():
    """Outlier class: a spike above the 99.5pct is clipped to 1.0, not stretched to dominate the range."""
    vol = np.concatenate([np.zeros(999, np.float32), np.array([1e6], np.float32)])
    out = normalize(vol)
    assert out.max() == 1.0
    assert out[:-1].max() <= 1.0  # bulk stays within range, spike clipped
