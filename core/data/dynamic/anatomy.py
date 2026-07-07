"""Synthetic ANATOMY from the Rodero four-chamber SSM (bd cardiac-seg-1vl / bwp).

Volumetric VTK heart meshes (Rodero/King's, Zenodo 4593739/4590294, CC-BY): tetrahedral myocardial
walls tagged by cell_data['ID'] (ID 1 = LV myo, ID 2 = RV myo; IDs 3-24 = atria/vessels/valves) with
per-point universal ventricular coords (RHO transmural endo->epi, Z apico-basal). We turn a mesh into
a SAX-aligned 3-class label volume — canonical RV cavity=1, LV myo=2, LV cavity=3 — then 2D short-axis
slices for the physics painter (synth.py). This is COVERAGE-BY-CONSTRUCTION anatomy (SSM mode-
extrapolation gives shapes beyond our real cohort) — the diversity deformation of real masks can't reach.

Cavities aren't meshed (they're the empty interior of endocardium); we recover them by 2D hole-filling
of the walls per SAX slice: the LV ring encloses LV-cav; the LV+RV walls together enclose RV-cav.
pyvista + vtk are the `viz` extra (imported lazily). ED-only, CT-derived (see bd 1vl caveats).
"""
from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
)
from scipy.ndimage import label as cc_label
from scipy.ndimage import zoom as _zoom

from core.config import DEFAULT_INPLANE, DEFAULT_SIZE
from core.data.static.labels import LV_CAV  # 3
from core.obs import setup
from core.preprocessing.preprocess import fit_square

log = logging.getLogger("cardioseg.anatomy")

LV_ID, RV_ID = 1, 2                    # cell tag: LV / RV myocardium (rest = atria/vessels, dropped)
_SENTINEL = -5.0                       # universal-coord sentinel guard (real values in [-1..1]; atria=-10)


def _tag_name(mesh) -> str:
    """The per-cell region-tag array. Two Rodero packagings: real-patient cohort (4590294) tags it
    'ID' + ships 'Z.dat' universal coords; the 1000 SSM-sampled cohort (4506930) tags it 'elemTag'
    with NO point data. Same 1..24 scheme (1=LV, 2=RV myo)."""
    for k in ("ID", "elemTag"):
        if k in mesh.cell_data:
            return k
    raise KeyError(f"no region-tag cell array (looked for ID/elemTag); have {list(mesh.cell_data)}")


def _rot_to_z(axis: np.ndarray) -> np.ndarray:
    """Rodrigues rotation matrix taking unit `axis` -> +z."""
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(axis, z); s = np.linalg.norm(v); c = float(np.dot(axis, z))
    if s < 1e-6:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _sax_align(mesh):
    """Rotate the mesh so the LV long axis (apex->base) points +z, so plain axial slices of the
    voxelized volume are short-axis. Uses the 'Z.dat' apico-basal coord when present (real cohort);
    else derives the axis geometrically (PCA long-axis of the LV myocardium, oriented apex->base by
    the atria/vessel centroid) — the 1000 SSM cohort ships no universal coords."""
    pts = np.asarray(mesh.points)
    tn = _tag_name(mesh)
    if "Z.dat" in mesh.point_data:
        Z = np.asarray(mesh.point_data["Z.dat"])
        valid = (Z >= 0.0) & (Z <= 1.0)                   # ventricular points only (atria = -10 sentinel)
        apex = pts[valid & (Z < 0.15)].mean(0)
        base = pts[valid & (Z > 0.85)].mean(0)
        axis = base - apex
        R = _rot_to_z(axis)
        origin = apex
    else:
        tags = np.asarray(mesh.cell_data[tn])
        lvpts = np.asarray(mesh.threshold([LV_ID, LV_ID], scalars=tn).points)
        c = lvpts.mean(0)
        _, _, vt = np.linalg.svd(lvpts - c, full_matrices=False)
        axis = vt[0]                                       # LV long axis = 1st principal component
        hi = int(tags.max())
        if hi > RV_ID:                                     # orient toward base (atria/vessels = tag>2)
            basec = np.asarray(mesh.threshold([RV_ID + 1, hi], scalars=tn).points).mean(0)
            if np.dot(basec - c, axis) < 0:
                axis = -axis
        proj = (lvpts - c) @ axis
        apex = c + axis * proj.min()                       # apex = far LV tip opposite the base
        R = _rot_to_z(axis)
        origin = apex
    out = mesh.copy()
    out.points = (pts - origin) @ R.T
    return out


def _wall_mask(mesh, region_id: int, grid, tag: str) -> np.ndarray:
    """Boolean grid-point mask of one region's wall solid (its closed tet-shell surface encloses the
    grid points that lie in the myocardium)."""
    sub = mesh.threshold([region_id, region_id], scalars=tag)
    surf = sub.extract_surface().triangulate()
    sel = grid.select_enclosed_points(surf, tolerance=0.0, check_surface=False)
    return np.asarray(sel["SelectedPoints"]).astype(bool)


def voxelize(mesh, inplane: float = DEFAULT_INPLANE, slice_mm: float = 8.0,
             rv_close: int = 3) -> np.ndarray:
    """SAX-aligned 3-class label volume [D, H, W] (RV-cav 1 / LV-myo 2 / LV-cav 3) from a Rodero mesh.
    Walls rasterized via enclosed-points; cavities recovered by 2D hole-fill per SAX slice (LV ring ->
    LV-cav; LV+RV walls together -> RV-cav). D slices at `slice_mm`, in-plane at `inplane` (mm)."""
    m = _sax_align(mesh)
    tn = _tag_name(m)
    x0, x1, y0, y1, z0, z1 = m.bounds
    nx = int(np.ceil((x1 - x0) / inplane)); ny = int(np.ceil((y1 - y0) / inplane))
    nz = int(np.ceil((z1 - z0) / slice_mm))
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(inplane, inplane, slice_mm),
                        origin=(x0, y0, z0))
    lv = _wall_mask(m, LV_ID, grid, tn).reshape(nz, ny, nx)   # ImageData points iterate x-fastest -> (z,y,x)
    rv = _wall_mask(m, RV_ID, grid, tn).reshape(nz, ny, nx)
    out = np.zeros((nz, ny, nx), dtype=np.uint8)
    for k in range(nz):
        lw, rw = lv[k], rv[k]
        if lw.sum() < 8:
            continue
        lv_cav = binary_fill_holes(lw) & ~lw                          # inside the LV wall ring
        # RV cavity: RV free wall + LV(septal) wall enclose it, but the RV crescent rarely closes a
        # ring in-plane (hinge gaps to the LV wall) -> raw fill leaks/empties. Close the wall union to
        # bridge the small gaps, fill, subtract walls+LV-cav, then keep only components touching the RV
        # wall (drops any fill that leaked into background). This is what lifted RV-cav off ~0.21.
        walls = binary_closing(lw | rw, iterations=rv_close) if rv_close > 0 else (lw | rw)
        both = binary_fill_holes(walls) & ~(lw | rw)
        rv_cav = both & ~lv_cav
        if rw.any() and rv_cav.any():
            lbl, n = cc_label(rv_cav)
            near = binary_dilation(rw, iterations=2)
            keep = {int(v) for v in np.unique(lbl[near & rv_cav]) if v}
            rv_cav = np.isin(lbl, list(keep)) if keep else np.zeros_like(rv_cav)
        out[k][rv_cav] = 1                                            # RV cavity
        out[k][lw] = 2                                               # LV myocardium (RV wall -> bg)
        out[k][lv_cav] = LV_CAV                                      # LV cavity (3)
    return out


# Real ACDC heart size at 1.5 mm in-plane: per-patient MAX fg-bbox longest side (px), measured over
# 150 patients x ED+ES (core/data/static/acdc): p5=62 p50=74 p95=89. Rodero voxelizes ~25-35% smaller
# in-frame (~47 px), so we globally rescale each synth heart to a target sampled from this real range —
# a per-heart factor that CENTERS scale on real AND spreads it across the real spread (domain-random),
# while preserving each heart's own apico-basal size profile (uniform in-plane zoom, not per-slice).
REAL_SIZE_PX = (58, 92)               # sample target max-bbox side here (measured real p5..p95, widened)


def _scale_to_target(vol: np.ndarray, target_px: int) -> np.ndarray:
    """Uniformly in-plane rescale a [D,H,W] label volume so its GLOBAL max fg-bbox longest side hits
    `target_px` (nearest-neighbour, label-preserving). No-op if the heart has no foreground."""
    fg = vol > 0
    if not fg.any():
        return vol
    zs, ys, xs = np.where(fg)
    cur = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1
    f = target_px / max(cur, 1)
    if abs(f - 1.0) < 1e-3:
        return vol
    return _zoom(vol, (1.0, f, f), order=0)           # in-plane only; slices (z) untouched


def load(path: str | Path):
    """Read a Rodero mesh (pyvista); sets the region tag active for thresholding. Prefers a sibling
    BINARY .vtu (4.5x faster load than ASCII .vtk: ~0.9s vs ~4.1s) if present — see convert_binary."""
    p = Path(path)
    vtu = p.with_suffix(".vtu")
    m = pv.read(str(vtu if (p.suffix == ".vtk" and vtu.exists()) else p))
    m.set_active_scalars(_tag_name(m), preference="cell")
    return m


def _convert_one(mp: str) -> str:
    out = Path(mp).with_suffix(".vtu")
    if not out.exists():
        pv.read(mp).save(str(out), binary=True)
    return str(out)


def convert_binary(mesh_dir: str | Path, workers: int = 0) -> int:
    """One-time: write a BINARY .vtu beside every ASCII .vtk (parallel). load() then uses it (4.5x
    faster). Returns count converted. ASCII parse is ~half the per-mesh voxelize cost."""
    vtks = [str(p) for p in sorted(Path(mesh_dir).rglob("*.vtk"))]
    nw = workers if workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    with ProcessPoolExecutor(max_workers=nw) as ex:
        list(ex.map(_convert_one, vtks, chunksize=1))
    return len(vtks)


def pathology_deform(mask: np.ndarray, k: int = 0, rv_k: int = 0) -> np.ndarray:
    """Label-space pathology SOURCE (bd vpn5): remodel LV and/or RV to synthesize the DCM/HCM/abnormal-RV
    tail the healthy SSM misses, keeping the OUTER LV size fixed + myo a RING (topology-safe — the naive-
    deform dead-end, bd bwp). k>0 = DILATE LV cavity into myo (thin wall -> DCM); k<0 = shrink/thicken
    (-> HCM). rv_k>0 = grow the RV cavity into background (-> abnormal/dilated RV; real RV-group RV/LV
    ~2.2 vs synth ~1.3). Returns a new label map (uint8)."""
    lvc, myo = mask == LV_CAV, mask == 2
    wall = lvc | myo                                          # LV cavity + myocardium (outer size fixed)
    out = mask.copy()
    if lvc.any() and myo.any() and k != 0:
        new_lvc = binary_dilation(lvc, iterations=k) & wall if k > 0 else binary_erosion(lvc, iterations=-k)
        new_myo = wall & ~new_lvc
        if new_myo.sum() >= 0.10 * wall.sum():               # guard: don't dissolve the myo ring
            out[wall] = 2
            out[new_lvc] = LV_CAV
    if rv_k > 0 and (mask == 1).any():                       # abnormal RV: grow RV cavity into bg only
        grown = binary_dilation(mask == 1, iterations=rv_k) & ~(out == 2) & ~(out == LV_CAV)
        out[grown] = 1
    return out


def build_pathology_pool(pool: np.ndarray, out_path: str | Path, k_dcm=(1, 6), k_hcm=(1, 4),
                         rv=(2, 6), seed: int = 0) -> tuple[Path, tuple]:
    """Turn a healthy label pool into a PATHOLOGY pool: per slice emit DCM (LV cavity dilated ~U[k_dcm]),
    HCM (eroded ~U[k_hcm]), and abnormal-RV (RV grown ~U[rv]) variants (topology-safe pathology_deform).
    The composite source covering the DCM/HCM/RV tail the SSM misses (bd vpn5/uch6)."""
    rng = np.random.default_rng(seed)
    out = []
    for m in pool:
        out.append(pathology_deform(m, k=int(rng.integers(k_dcm[0], k_dcm[1] + 1))))
        out.append(pathology_deform(m, k=-int(rng.integers(k_hcm[0], k_hcm[1] + 1))))
        out.append(pathology_deform(m, rv_k=int(rng.integers(rv[0], rv[1] + 1))))
    arr = np.stack(out).astype(np.uint8)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, slices=arr)
    return out_path, arr.shape


def _pool_worker(args) -> list[np.ndarray]:
    """One mesh -> its fit_square'd SAX label slices (the per-mesh unit of build_pool, run in a worker
    process). Seeded per-mesh (seed+index) so the pool is deterministic regardless of finish order."""
    mp, inplane, size, min_fg, scale_reps, min_cav_frac, mseed = args
    rng = np.random.default_rng(mseed)
    out = []
    try:
        vol = voxelize(load(mp), inplane=inplane)
    except Exception:                                    # a malformed mesh shouldn't kill the whole build
        return out
    for _ in range(max(1, scale_reps)):
        tgt = int(rng.integers(REAL_SIZE_PX[0], REAL_SIZE_PX[1] + 1))
        sv = _scale_to_target(vol, tgt)
        for k in range(sv.shape[0]):
            s = sv[k]
            fg = int((s > 0).sum())
            if fg < min_fg:
                continue
            # drop pure-apical myo-only slices (no cavity): the mesh contributes its whole apico-basal
            # stack, so apex slices (~all myo, no cavity) over-represent 11x vs real (19% vs 1.8%). Require
            # some cavity to match the real slice composition — cleans the synth over-spread (bd uy4d).
            if min_cav_frac > 0 and int(((s == 1) | (s == LV_CAV)).sum()) < min_cav_frac * fg:
                continue
            out.append(fit_square(s, size, 0).astype(np.uint8))
    return out


def build_pool(mesh_dir: str | Path, out_path: str | Path, size: int = DEFAULT_SIZE,
               inplane: float = DEFAULT_INPLANE, min_fg: int = 40, scale_reps: int = 1,
               seed: int = 0, workers: int = 0, min_cav_frac: float = 0.05) -> tuple[Path, tuple]:
    """Voxelize every *.vtk in `mesh_dir` -> scale-match to real -> SAX slices -> fit_square to `size`
    -> stacked label pool, saved to `out_path` (npz 'slices' [N,size,size] uint8). The synthetic-
    ANATOMY training pool: label maps only (the physics painter adds contrast per batch). Near-empty
    apex/base slices dropped. Each mesh is globally rescaled to a target max-bbox side sampled from the
    real ACDC size distribution (REAL_SIZE_PX); `scale_reps` emits that many independent scale draws per
    mesh. Meshes are voxelized in PARALLEL (`workers` processes; 0 -> cpu-2, the bottleneck is the
    per-mesh select_enclosed_points on ~2M-cell tets) — embarrassingly parallel, deterministic."""
    # discover meshes by STEM: prefer the binary .vtu (4.5x faster load; the ASCII .vtk is a redundant
    # duplicate and is pruned from storage), fall back to .vtk for a fresh, unconverted download.
    stems = {p.with_suffix("") for p in Path(mesh_dir).rglob("*.vtu")}
    stems |= {p.with_suffix("") for p in Path(mesh_dir).rglob("*.vtk")}
    meshes = sorted((s.with_suffix(".vtu") if s.with_suffix(".vtu").exists() else s.with_suffix(".vtk"))
                    for s in stems)
    nw = workers if workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    jobs = [(str(mp), inplane, size, min_fg, scale_reps, min_cav_frac, seed + i)
            for i, mp in enumerate(meshes)]
    slices: list[np.ndarray] = []
    if nw == 1 or len(jobs) <= 1:
        for j in jobs:
            slices.extend(_pool_worker(j))
    else:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            for part in ex.map(_pool_worker, jobs, chunksize=1):
                slices.extend(part)
    arr = np.stack(slices) if slices else np.zeros((0, size, size), np.uint8)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, slices=arr)
    return out_path, arr.shape


def load_pool(path: str | Path) -> np.ndarray:
    """Load the anatomy pool [N, size, size] (label maps) built by build_pool."""
    return np.load(str(path))["slices"]


def _main():
    """Voxelize one Rodero mesh -> SAX label volume; save a mid-slice montage to eyeball the chambers."""
    setup()
    ap = argparse.ArgumentParser(description="Rodero mesh -> SAX 3-class label volume (bd 1vl).")
    ap.add_argument("--mesh", required=True, help="path to a Rodero .vtk mesh")
    ap.add_argument("--inplane", type=float, default=DEFAULT_INPLANE)
    ap.add_argument("--out", default=None, help="montage PNG (default: <mesh>_sax.png)")
    a = ap.parse_args()
    vol = voxelize(load(a.mesh), inplane=a.inplane)
    counts = {int(c): int((vol == c).sum()) for c in np.unique(vol)}
    log.info(f"volume {vol.shape}  label counts {counts}  (1=RVcav 2=LVmyo 3=LVcav)")
    D = vol.shape[0]
    ks = [int(D * f) for f in (0.3, 0.45, 0.6, 0.75)]            # mid-ventricular slices
    cmap = np.array([[0, 0, 0], [91, 141, 239], [255, 202, 91], [239, 83, 80]], np.uint8)
    tiles = [cmap[vol[k]] for k in ks]
    montage = np.concatenate(tiles, axis=1)
    out = a.out or (str(Path(a.mesh).with_suffix("")) + "_sax.png")
    Image.fromarray(montage).save(out)
    log.info(f"wrote {out}  (slices {ks})")


if __name__ == "__main__":
    _main()
