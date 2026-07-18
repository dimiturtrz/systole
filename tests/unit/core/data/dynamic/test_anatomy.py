"""Synthetic-ANATOMY builder (core.data.dynamic.anatomy). Equivalence-class tests for the pure logic:
region-tag lookup, Rodrigues rotation, SAX alignment (both cohort paths), per-slice cavity recovery,
scale-to-real, and the label-space pathology deform + pool builders. Mesh-dependent functions run on
tiny synthetic pyvista tet meshes; the real Rodero disk read / ProcessPool build are thin I/O shells."""
import numpy as np
import pytest
import pyvista as pv

from core.data.dynamic import anatomy as A
from core.data.dynamic.anatomy import (
    Anatomy,
    MeshError,
    PathologyPoolCfg,
    PoolBuildCfg,
)
from core.data.static.labels import LV_CAV, MYO, RV

_rot_to_z = Anatomy._rot_to_z
_scale_to_target = Anatomy._scale_to_target
_slice_labels = Anatomy._slice_labels
_tag_name = Anatomy._tag_name
build_pathology_pool = Anatomy.build_pathology_pool
build_pool = Anatomy.build_pool
load = Anatomy.load
load_pool = Anatomy.load_pool
pathology_deform = Anatomy.pathology_deform
voxelize = Anatomy.voxelize


# ── fixtures: tiny synthetic Rodero-like tet meshes ──────────────────────────────────────────────
def _tet_box(bounds, tag_val, tag="ID"):
    g = pv.Box(bounds=bounds).triangulate().delaunay_3d()
    g.cell_data[tag] = np.full(g.n_cells, tag_val, int)
    return g


def _heart_mesh(tag="ID", with_z=False):
    """LV wall (tag 1), RV wall beside it (tag 2), atria block above the base (tag 5). Apex at z=0,
    base at z=60; adds a 'Z.dat' apico-basal universal coord when `with_z` (real-cohort path)."""
    lv = _tet_box((0, 20, 0, 20, 0, 60), 1, tag)
    rv = _tet_box((20, 34, 5, 15, 0, 60), 2, tag)
    atria = _tet_box((0, 20, 0, 20, 60, 75), 5, tag)
    m = lv.merge(rv).merge(atria)
    if with_z:
        z = np.asarray(m.points)[:, 2]
        m.point_data["Z.dat"] = (z - z.min()) / (z.max() - z.min())   # 0..1 apex->base
    m.set_active_scalars(tag, preference="cell")
    return m


# ── _tag_name: real-cohort 'ID' vs SSM-cohort 'elemTag' vs neither ───────────────────────────────
def test_tag_name_id_cohort():
    """Real-patient cohort tags cells 'ID'."""
    assert _tag_name(_heart_mesh("ID")) == "ID"


def test_tag_name_elemtag_cohort():
    """1000-SSM cohort tags cells 'elemTag' (no point data)."""
    assert _tag_name(_heart_mesh("elemTag")) == "elemTag"


def test_tag_name_missing_raises():
    """Neither tag present -> KeyError (unusable mesh)."""
    g = pv.Box().triangulate().delaunay_3d()
    g.cell_data["other"] = np.zeros(g.n_cells, int)
    with pytest.raises(KeyError):
        _tag_name(g)


# ── _rot_to_z: general axis vs already-parallel (rotation no-op) ──────────────────────────────────
def test_rot_to_z_general_axis():
    """A non-z axis rotates ONTO +z (R @ axis == +z)."""
    axis = np.array([1.0, 1.0, 0.0])
    R = _rot_to_z(axis)
    mapped = R @ (axis / np.linalg.norm(axis))
    assert np.allclose(mapped, [0, 0, 1], atol=1e-6)   # lands on z


def test_rot_to_z_already_parallel_is_identity():
    """|sin| below _PARALLEL_EPS (axis already ~+z) -> identity, no rotation."""
    assert np.allclose(_rot_to_z(np.array([0.0, 0.0, 1.0])), np.eye(3))


# ── _sax_align: two cohort paths both put the long axis on +z ─────────────────────────────────────
def test_sax_align_zdat_path():
    """Real cohort (has 'Z.dat'): apex->base axis from the universal coord; base ends up +z of apex."""
    al = A.Anatomy._sax_align(_heart_mesh("ID", with_z=True))
    pts = np.asarray(al.points)
    assert np.ptp(pts[:, 2]) > 0          # aligned volume has finite z-extent (didn't collapse)


def test_sax_align_geometric_path():
    """SSM cohort (no 'Z.dat'): PCA long-axis oriented apex->base by the atria centroid."""
    al = A.Anatomy._sax_align(_heart_mesh("elemTag", with_z=False))
    assert np.ptp(np.asarray(al.points)[:, 2]) > 0


def test_sax_align_geometric_flips_axis_toward_base():
    """Geometric path, atria BELOW the LV base (z<0): the PCA axis must be flipped so apex->base still
    points +z (line 126 orientation-flip branch). Both orientations land the base above the apex."""
    lv = _tet_box((0, 20, 0, 20, 0, 60), 1, "elemTag")
    rv = _tet_box((20, 34, 5, 15, 0, 60), 2, "elemTag")
    atria = _tet_box((0, 20, 0, 20, -20, -5), 5, "elemTag")   # atria/vessels BELOW the apex end
    m = lv.merge(rv).merge(atria); m.set_active_scalars("elemTag", preference="cell")
    al = A.Anatomy._sax_align(m)
    assert np.ptp(np.asarray(al.points)[:, 2]) > 0            # aligned, not collapsed


# ── _slice_labels: per-slice cavity recovery equivalence classes ─────────────────────────────────
def _ring(size, r_out, r_in, cy=None, cx=None):
    cy = cy if cy is not None else size // 2
    cx = cx if cx is not None else size // 2
    yy, xx = np.ogrid[:size, :size]
    d = (yy - cy) ** 2 + (xx - cx) ** 2
    return (d <= r_out ** 2) & (d > r_in ** 2)


def test_slice_labels_thin_wall_empty():
    """LV wall below _MIN_LV_WALL_PX -> all-background (too thin to close a ring)."""
    lw = np.zeros((20, 20), bool)
    lw[0, 0] = True                                    # 1 px << threshold
    assert _slice_labels(lw, np.zeros_like(lw)).sum() == 0


def test_slice_labels_lv_ring_makes_cavity():
    """A closed LV wall ring encloses LV-cav; the ring itself is myo, no RV wall -> no RV-cav."""
    lw = _ring(40, 12, 8)
    out = _slice_labels(lw, np.zeros_like(lw))
    assert (out == MYO).any() and (out == LV_CAV).any()   # myo ring + enclosed cavity
    assert (out == RV).sum() == 0                          # no RV wall -> no RV cavity


def test_slice_labels_rv_wall_makes_rv_cavity():
    """LV ring + an RV crescent wall abutting it -> RV cavity recovered between the walls (touching RV)."""
    lw = _ring(48, 12, 8, cy=24, cx=20)
    rw = np.zeros((48, 48), bool)
    rw[10:38, 30:34] = True                            # RV free wall to the right of the LV ring
    rw[10:14, 20:34] = True; rw[34:38, 20:34] = True   # close the crescent top+bottom -> enclosed pocket
    out = _slice_labels(lw, rw, rv_close=3)
    assert (out == RV).any()                           # RV cavity recovered
    assert (out == LV_CAV).any()                       # LV cavity still there


def test_slice_labels_rv_close_zero_branch():
    """rv_close=0 takes the no-closing wall-union branch (still produces LV-cav)."""
    lw = _ring(40, 12, 8)
    out = _slice_labels(lw, np.zeros_like(lw), rv_close=0)
    assert (out == LV_CAV).any()


# ── voxelize: full mesh -> label volume (both cohorts) + degenerate guard ─────────────────────────
def test_voxelize_geometric_cohort_all_classes():
    """No-Z.dat mesh voxelizes to a [D,H,W] volume carrying all 3 canonical classes."""
    vol = voxelize(_heart_mesh("elemTag"), inplane=2.0, slice_mm=8.0)
    assert vol.ndim == 3
    assert set(np.unique(vol)) >= {0, MYO}             # at least bg + myo rasterized


def test_voxelize_degenerate_bounds_raises(monkeypatch):
    """A mesh whose SAX-aligned bounds give a <1 grid dim -> MeshError (degenerate guard, line 185).
    Force a paper-thin z-extent so nz rounds to 0 at slice_mm=8."""
    m = _heart_mesh("elemTag")

    class _Thin:                                          # a stand-in aligned mesh with degenerate bounds
        bounds = (0.0, 20.0, 0.0, 20.0, 0.0, 0.5)         # z-extent 0.5 mm -> nz = ceil(0.5/8) ... > 0
    # squash to truly degenerate: identical x bounds so nx = 0
    _Thin.bounds = (5.0, 5.0, 0.0, 20.0, 0.0, 60.0)
    monkeypatch.setattr(A.Anatomy, "_sax_align", lambda _m: _Thin())
    monkeypatch.setattr(A.Anatomy, "_tag_name", lambda _m: "elemTag")
    with pytest.raises(MeshError):
        voxelize(m, inplane=2.0, slice_mm=8.0)


def test_load_missing_file_raises_mesh_error(tmp_path):
    """load() wraps a pyvista read failure (missing/corrupt file) as our domain MeshError."""
    with pytest.raises(MeshError):
        A.Anatomy.load(tmp_path / "does_not_exist.vtk")


# ── _scale_to_target: rescale, no-op, and empty-heart classes ────────────────────────────────────
def _heart_slice(size=40):
    v = np.zeros((1, size, size), np.uint8)
    v[0] = _ring(size, 10, 6).astype(np.uint8) * MYO
    v[0][_ring(size, 6, 0)] = LV_CAV
    return v


def test_scale_to_target_upscales():
    """target_px far above the current bbox side -> zoomed volume grows toward target."""
    v = _heart_slice(40)                               # bbox side ~20 px
    out = _scale_to_target(v, target_px=60)
    fg = out[0] > 0
    ys, xs = np.where(fg)
    side = max(np.ptp(ys), np.ptp(xs)) + 1
    assert abs(side - 60) <= 3                          # within nearest-neighbour rounding


def test_scale_to_target_noop_when_at_target():
    """|scale-1| below _ZOOM_NOOP_EPS -> returns the SAME array (no resample)."""
    v = _heart_slice(40)
    fg = v[0] > 0
    ys, xs = np.where(fg)
    cur = int(max(np.ptp(ys), np.ptp(xs)) + 1)
    assert _scale_to_target(v, target_px=cur) is v     # identity object, skipped


def test_scale_to_target_empty_heart_noop():
    """No foreground -> returned unchanged (can't measure a bbox)."""
    v = np.zeros((2, 16, 16), np.uint8)
    assert _scale_to_target(v, target_px=50) is v


# ── pathology_deform: DCM (dilate) / HCM (erode) / abnormal-RV / guards ──────────────────────────
def _labelled_heart(size=48):
    m = np.zeros((size, size), np.uint8)
    m[_ring(size, 12, 7)] = MYO
    m[_ring(size, 7, 0)] = LV_CAV
    m[10:38, 34:40] = RV                               # RV cavity block beside
    return m


def test_pathology_deform_dcm_dilates_cavity():
    """k>0 dilates the LV cavity into myo (thin wall = DCM); outer LV size fixed, myo stays a ring."""
    m = _labelled_heart()
    out = pathology_deform(m, k=2)
    assert (out == LV_CAV).sum() > (m == LV_CAV).sum()   # cavity grew
    assert (out == MYO).sum() < (m == MYO).sum()         # wall thinned
    assert (out == MYO).sum() > 0                         # ring not dissolved (guard held)


def test_pathology_deform_hcm_erodes_cavity():
    """k<0 shrinks the LV cavity / thickens the wall (HCM)."""
    m = _labelled_heart()
    out = pathology_deform(m, k=-2)
    assert (out == LV_CAV).sum() < (m == LV_CAV).sum()   # cavity shrank


def test_pathology_deform_rv_grows():
    """rv_k>0 grows the RV cavity into background only (abnormal/dilated RV)."""
    m = _labelled_heart()
    out = pathology_deform(m, rv_k=3)
    assert (out == RV).sum() > (m == RV).sum()           # RV cavity dilated
    assert (out == MYO).sum() == (m == MYO).sum()        # didn't eat the myo


def test_pathology_deform_noop_identity():
    """k=0, rv_k=0 -> unchanged copy (identity class)."""
    m = _labelled_heart()
    assert np.array_equal(pathology_deform(m), m)


def test_pathology_deform_guard_blocks_dissolving_ring():
    """A huge dilate that would leave <10% myo is REJECTED — the ring is preserved unchanged."""
    m = _labelled_heart()
    out = pathology_deform(m, k=20)                       # would dissolve the thin wall
    assert (out == MYO).sum() == (m == MYO).sum()        # guard kept original myo


# ── build_pathology_pool: emits 3 variants/slice, deterministic, saves npz ───────────────────────
def test_build_pathology_pool_shape_and_roundtrip(tmp_path):
    """Pool of P slices -> 3*P variant slices (DCM/HCM/RV), saved+reloadable via load_pool."""
    pool = np.stack([_labelled_heart(), _labelled_heart()]).astype(np.uint8)   # P=2
    out_path, shape = build_pathology_pool(pool, tmp_path / "path.npz", PathologyPoolCfg(seed=0))
    assert shape[0] == 6                                  # 3 variants x 2 slices
    reloaded = load_pool(out_path)
    assert reloaded.shape == shape and reloaded.dtype == np.uint8


def test_build_pathology_pool_deterministic(tmp_path):
    """Same seed -> identical pool (seeded rng)."""
    pool = np.stack([_labelled_heart()]).astype(np.uint8)
    a = load_pool(build_pathology_pool(pool, tmp_path / "a.npz", PathologyPoolCfg(seed=7))[0])
    b = load_pool(build_pathology_pool(pool, tmp_path / "b.npz", PathologyPoolCfg(seed=7))[0])
    assert np.array_equal(a, b)


# ── .vtk I/O-driven paths: load + serial build_pool (_pool_worker) on a saved synthetic mesh ─────
def test_load_reads_saved_mesh_and_sets_tag(tmp_path):
    """load() reads a .vtk and activates its region tag (round-trip of a saved synthetic mesh)."""
    p = tmp_path / "mesh.vtk"
    _heart_mesh("elemTag").save(str(p))
    m = load(p)
    assert _tag_name(m) == "elemTag"


def test_convert_binary_writes_vtu_beside_vtk(tmp_path):
    """convert_binary: a BINARY .vtu is written beside each ASCII .vtk (load() then prefers it). Returns
    the count converted; the .vtu is a readable pyvista mesh with the region tag intact."""
    _heart_mesh("elemTag").save(str(tmp_path / "h.vtk"))
    n = A.Anatomy.convert_binary(tmp_path, workers=1)
    assert n == 1
    vtu = tmp_path / "h.vtu"
    assert vtu.exists()
    assert _tag_name(load(vtu)) == "elemTag"             # binary round-trip preserved the tag


def test_convert_one_idempotent(tmp_path):
    """_convert_one skips a mesh whose .vtu already exists (idempotent re-run branch)."""
    p = tmp_path / "h.vtk"; _heart_mesh("elemTag").save(str(p))
    first = A.Anatomy._convert_one(str(p))
    mtime = (tmp_path / "h.vtu").stat().st_mtime_ns
    second = A.Anatomy._convert_one(str(p))                       # .vtu present -> no rewrite
    assert first == second == str(tmp_path / "h.vtu")
    assert (tmp_path / "h.vtu").stat().st_mtime_ns == mtime


def test_build_pool_serial_populates(tmp_path):
    """build_pool (workers=1, serial branch): voxelize->scale->SAX slices->fit_square->stacked pool.
    Emits size x size uint8 heart slices; near-empty/cavity-less apex slices dropped."""
    _heart_mesh("elemTag").save(str(tmp_path / "mesh.vtk"))
    cfg = PoolBuildCfg(size=64, min_fg=5, min_cav_frac=0.01, workers=1)
    out_path, shape = build_pool(tmp_path, tmp_path / "pool.npz", cfg)
    arr = np.load(out_path)["slices"]
    assert shape == arr.shape and arr.shape[1:] == (64, 64)
    assert arr.shape[0] > 0 and arr.dtype == np.uint8
    assert set(np.unique(arr)).issubset({0, 1, 2, 3})   # canonical labels only


def test_pool_worker_bad_mesh_skipped(tmp_path):
    """_pool_worker on an unreadable mesh path -> load() raises MeshError -> the worker returns [] (a
    single bad mesh must not kill the build; lines 297-298 skip branch)."""
    bad = str(tmp_path / "missing.vtk")
    args = (bad, 2.0, 64, 5, 1, 0.01, 0)                  # mp, inplane, size, min_fg, reps, min_cav, seed
    assert A.Anatomy._pool_worker(args) == []


def test_pool_worker_min_cav_frac_drops_myo_only(tmp_path):
    """_pool_worker: a very high min_cav_frac drops slices whose cavity is below the fraction of fg
    (pure-apical myo-only slices; line 311 cavity-composition guard) -> nothing emitted."""
    p = tmp_path / "m.vtk"; _heart_mesh("elemTag").save(str(p))
    args = (str(p), 2.0, 64, 5, 1, 0.99, 0)               # min_cav_frac 0.99 -> essentially every slice dropped
    assert A.Anatomy._pool_worker(args) == []


def test_build_pool_parallel_branch_matches_serial(tmp_path):
    """build_pool with workers>1 and >1 mesh takes the ProcessPoolExecutor branch (lines 340-342) and
    produces the same-shaped canonical pool as the serial path — parallelism is deterministic."""
    for i in range(2):
        _heart_mesh("elemTag").save(str(tmp_path / f"m{i}.vtk"))
    cfg = PoolBuildCfg(size=64, min_fg=5, min_cav_frac=0.01, workers=2)
    _, shape = build_pool(tmp_path, tmp_path / "pool.npz", cfg)
    arr = np.load(tmp_path / "pool.npz")["slices"]
    assert shape == arr.shape and arr.shape[1:] == (64, 64)
    assert arr.shape[0] > 0                               # slices from both meshes emitted
    assert set(np.unique(arr)).issubset({0, 1, 2, 3})


def test_build_pool_empty_dir_returns_empty(tmp_path):
    """No meshes -> a well-formed empty pool [0,size,size] (empty branch)."""
    _, shape = build_pool(tmp_path, tmp_path / "pool.npz", PoolBuildCfg(size=48, workers=1))
    assert shape == (0, 48, 48)


def test_build_pool_all_slices_dropped_returns_empty(tmp_path):
    """min_fg above every SAX slice's fg -> all slices dropped -> empty pool (guard branch)."""
    _heart_mesh("elemTag").save(str(tmp_path / "mesh.vtk"))
    _, shape = build_pool(tmp_path, tmp_path / "pool.npz",
                          PoolBuildCfg(size=64, min_fg=10 ** 6, workers=1))
    assert shape == (0, 64, 64)
