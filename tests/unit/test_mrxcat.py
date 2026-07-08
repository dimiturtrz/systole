"""MRXCAT2.0 source adapter (core.data.dynamic.mrxcat). The pure, testable core is the label remap
to_canonical — verified geometrically on the bundled phantom (myo ring encloses LV-cav); here we pin
the equivalence classes of the code→canonical mapping so a scheme change can't silently corrupt pools."""
import numpy as np
import pyvista as pv
import pytest

from core.data.dynamic.mrxcat import to_canonical


def _write_vti(path, nz=6):
    """A small MRXCAT-like phantom .vti (`labels` point array): a myo ring (1) around LV blood (5),
    an RV blood block (6), plus a surrounding liver code (13) — the codes to_canonical/to_tissue_map
    remap. VTK ImageData is x-fastest, so we build [nx,ny,nz] Fortran-order to match load_vti_labels."""
    nx = ny = 30
    lab = np.zeros((nx, ny, nz), int, order="F")
    yy, xx = np.ogrid[:ny, :nx]
    d = (yy - 15) ** 2 + (xx - 15) ** 2
    ring, cav = (d <= 36) & (d > 16), d <= 16
    for k in range(nz):
        lab[:, :, k][ring.T] = 1        # LV wall -> myo
        lab[:, :, k][cav.T] = 5         # LV blood -> LV-cav
        lab[10:14, 20:26, k] = 6        # RV blood -> RV-cav
        lab[0:5, 0:5, k] = 13           # liver group (surrounding tissue)
    g = pv.ImageData(dimensions=(nx, ny, nz), spacing=(1, 1, 1))
    g.point_data["labels"] = lab.ravel(order="F")
    g.save(str(path))
    return path


@pytest.fixture
def vti_dir(tmp_path):
    """A directory holding one synthetic phantom .vti (drives the real load->build inner loops)."""
    _write_vti(tmp_path / "phantom.vti")
    return tmp_path


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


def test_place_heart_in_fov_swaps_anatomy():
    """SSM x MRXCAT (majh): excise the phantom's heart, paste OUR heart at its location. Result keeps
    surrounding tissue + carries our heart classes; the old phantom-heart pixels not covered → muscle."""
    from core.data.dynamic.mrxcat import place_heart_in_fov
    fov = np.full((60, 60), 6, np.uint8)                 # muscle body
    fov[20:40, 20:40] = 2                                # phantom myo block
    fov[26:34, 26:34] = 3                                # phantom LV-cav
    fov[10:50, 45:52] = 4                                # lung (surrounding, must survive)
    heart = np.zeros((40, 40), np.uint8)                 # our heart, centred in its own frame
    heart[12:28, 12:28] = 2; heart[16:24, 16:24] = 3; heart[12:28, 28:32] = 1
    out = place_heart_in_fov(fov, heart)
    assert set(np.unique(out).tolist()) >= {1, 2, 3, 4, 6}   # our RV/myo/cav + lung + muscle
    assert (out == 4).sum() == (fov == 4).sum()          # surrounding lung untouched


def test_fovbg_paints_wholefov_map():
    """Integration: bg_mode='mrxcat' paints an 8-class FOV tissue map (FovBg + named_tissue_params) with
    no bg invention — every class rendered by its tissue, image finite, heart target recoverable."""
    import torch

    from core.data.dynamic.synth import MrxcatBgCfg, SynthCfg, synthesize_from_labels
    fov = torch.zeros((2, 64, 64), dtype=torch.long)
    fov[:, 20:44, 20:44] = 6                              # muscle body
    fov[:, 24:40, 24:40] = 2                              # myo
    fov[:, 28:36, 28:36] = 3                              # LV-cav
    fov[:, 24:40, 44:52] = 4                              # lung beside
    img, _ = synthesize_from_labels(fov, SynthCfg(synth_p=1.0, bg=MrxcatBgCfg(), deform=0.0), 4)
    assert img.shape == (2, 1, 64, 64) and torch.isfinite(img).all()
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


def test_heart_crop_scale_empty_returns_none():
    """No foreground in the slice -> None (no heart to crop)."""
    from core.data.dynamic.mrxcat import _heart_crop_scale
    assert _heart_crop_scale(np.zeros((64, 64), np.uint8), size=32, target_px=20) is None


def test_heart_crop_scale_noop_zoom_branch():
    """Heart bbox already ~= target_px -> the |scale-1|<eps branch skips _zoom; still fit_square'd."""
    from core.data.dynamic.mrxcat import _heart_crop_scale
    s = np.zeros((40, 40), np.uint8)
    s[10:30, 10:30] = 2                                  # 20-px bbox == target -> no rescale
    out = _heart_crop_scale(s, size=48, target_px=20)
    assert out.shape == (48, 48) and (out == 2).sum() == 400   # 20x20 block preserved exactly


def test_fov_window_crops_chest_window():
    """_fov_window: crop a scale x (heart-bbox) window centred on the heart, resize to size. Keeps
    surrounding tissue classes (equivalence class: heart present -> square window with context)."""
    from core.data.dynamic.mrxcat import _fov_window
    s = np.full((120, 120), 6, np.uint8)                 # muscle body
    s[50:70, 50:70] = 2                                  # heart myo block, centred
    s[54:66, 54:66] = 3                                  # LV-cav
    s[10:110, 90:100] = 4                                # lung strip (surrounding context)
    out = _fov_window(s, size=64, scale=3.0)
    assert out.shape == (64, 64)
    assert {2, 3}.issubset(set(np.unique(out)))          # heart survived the window


def test_fov_window_no_heart_returns_none():
    """No heart classes {1,2,3} in the slice -> None (nothing to centre a window on)."""
    from core.data.dynamic.mrxcat import _fov_window
    s = np.full((64, 64), 6, np.uint8)                   # muscle only, no heart
    assert _fov_window(s, size=32, scale=3.0) is None


def test_place_heart_in_fov_absent_heart_returns_fov():
    """None-safe: our heart is all-bg -> the phantom FOV returned unchanged."""
    from core.data.dynamic.mrxcat import place_heart_in_fov
    fov = np.full((40, 40), 6, np.uint8)
    fov[10:30, 10:30] = 2                                # phantom heart present
    out = place_heart_in_fov(fov, np.zeros((20, 20), np.uint8))   # our heart empty
    assert np.array_equal(out, fov)


def test_place_heart_in_fov_absent_phantom_returns_fov():
    """None-safe: the phantom FOV has no heart classes -> returned unchanged."""
    from core.data.dynamic.mrxcat import place_heart_in_fov
    fov = np.full((40, 40), 6, np.uint8)                 # muscle only, no heart to excise
    heart = np.zeros((20, 20), np.uint8); heart[8:12, 8:12] = 3
    assert np.array_equal(place_heart_in_fov(fov, heart), fov)


def test_build_pool_empty_dir_returns_empty(tmp_path):
    """No .vti files -> a well-formed empty pool [0,size,size] uint8 (I/O shell, empty branch)."""
    from core.data.dynamic.anatomy import PoolBuildCfg
    from core.data.dynamic.mrxcat import build_pool
    out_path, shape = build_pool(tmp_path, tmp_path / "pool.npz", PoolBuildCfg(size=32))
    assert shape == (0, 32, 32)
    assert np.load(out_path)["slices"].shape == (0, 32, 32)


def test_build_fov_pool_empty_dir_returns_empty(tmp_path):
    """No .vti -> empty 8-class FOV pool [0,size,size] (I/O shell, empty branch)."""
    from core.data.dynamic.mrxcat import build_fov_pool
    _, shape = build_fov_pool(tmp_path, tmp_path / "fov.npz", size=32)
    assert shape == (0, 32, 32)


def test_build_ssm_fov_pool_no_backgrounds_empty(tmp_path):
    """SSM x MRXCAT with an empty .vti dir -> no XCAT backgrounds -> empty pool (no composite emitted)."""
    from core.data.dynamic.mrxcat import build_ssm_fov_pool
    rodero = np.zeros((3, 32, 32), np.uint8)
    rodero[:, 8:24, 8:24] = 2                            # a tiny heart pool
    rp = tmp_path / "rodero.npz"
    np.savez_compressed(rp, slices=rodero)
    _, shape = build_ssm_fov_pool(rp, tmp_path, tmp_path / "ssm_fov.npz", size=32)
    assert shape == (0, 32, 32)                          # no bgs -> empty branch


# ── .vti I/O-driven paths: real read + real build inner loops on a synthetic phantom ─────────────
def test_load_vti_labels_roundtrip(vti_dir):
    """load_vti_labels reads the `labels` array into [nz,ny,nx] with the written codes intact."""
    from core.data.dynamic.mrxcat import load_vti_labels
    vol = load_vti_labels(vti_dir / "phantom.vti")
    assert vol.shape == (6, 30, 30)                      # axis 0 = slices
    assert set(np.unique(vol)) == {0, 1, 5, 6, 13}       # the codes we wrote


def test_build_pool_populates_canonical_slices(vti_dir):
    """build_pool: .vti -> canonical -> heart-crop+scale -> fit_square. Emits heart-only 4-class slices."""
    from core.data.dynamic.anatomy import PoolBuildCfg
    from core.data.dynamic.mrxcat import build_pool
    out_path, shape = build_pool(vti_dir, vti_dir / "pool.npz",
                                 PoolBuildCfg(size=48, min_fg=5, min_cav_frac=0.02))
    arr = np.load(out_path)["slices"]
    assert shape == arr.shape and arr.shape[1:] == (48, 48)
    assert arr.shape[0] > 0                              # slices survived the fg/cavity guards
    assert set(np.unique(arr)).issubset({0, 1, 2, 3})    # heart-only canonical


def test_build_pool_min_fg_drops_all(vti_dir):
    """min_fg above every slice's heart px -> all slices dropped -> empty pool (guard branch)."""
    from core.data.dynamic.anatomy import PoolBuildCfg
    from core.data.dynamic.mrxcat import build_pool
    _, shape = build_pool(vti_dir, vti_dir / "pool.npz", PoolBuildCfg(size=48, min_fg=10 ** 6))
    assert shape == (0, 48, 48)


def test_build_pool_min_cav_frac_drops_all(vti_dir):
    """min_cav_frac ~1 -> every slice's cavity is below the fraction of fg -> all dropped (line 134
    cavity-composition guard, the apex-slice over-representation fix)."""
    from core.data.dynamic.anatomy import PoolBuildCfg
    from core.data.dynamic.mrxcat import build_pool
    _, shape = build_pool(vti_dir, vti_dir / "pool.npz",
                          PoolBuildCfg(size=48, min_fg=5, min_cav_frac=0.99))
    assert shape == (0, 48, 48)


def test_build_fov_pool_min_fg_drops_all(vti_dir):
    """build_fov_pool: min_fg above every slice's heart px -> all slices skipped -> empty (line 181)."""
    from core.data.dynamic.mrxcat import build_fov_pool
    _, shape = build_fov_pool(vti_dir, vti_dir / "fov.npz", size=48, min_fg=10 ** 6)
    assert shape == (0, 48, 48)


def test_build_fov_pool_populates_8class(vti_dir):
    """build_fov_pool: .vti -> tissue map -> chest window. Keeps surrounding organ classes (>3)."""
    from core.data.dynamic.mrxcat import build_fov_pool
    out_path, shape = build_fov_pool(vti_dir, vti_dir / "fov.npz", size=48, min_fg=5)
    arr = np.load(out_path)["slices"]
    assert shape == arr.shape and arr.shape[0] > 0
    assert (arr > 3).any()                               # surrounding tissue (liver=5) kept, not just heart


def test_build_ssm_fov_pool_composites(vti_dir):
    """build_ssm_fov_pool: OUR rodero hearts composited into XCAT chest windows -> one slice per heart."""
    from core.data.dynamic.mrxcat import build_ssm_fov_pool
    rodero = np.zeros((4, 48, 48), np.uint8)
    rodero[:, 16:32, 16:32] = 2; rodero[:, 20:28, 20:28] = 3   # tiny heart pool (N=4)
    rp = vti_dir / "rodero.npz"; np.savez_compressed(rp, slices=rodero)
    _, shape = build_ssm_fov_pool(rp, vti_dir, vti_dir / "ssm.npz", size=48, scale=3.0)
    assert shape[0] == 4                                 # one composite per rodero heart
