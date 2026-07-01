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

from pathlib import Path

import numpy as np

from core.data.static.labels import LV_CAV  # 3
from core.config import DEFAULT_SIZE, DEFAULT_INPLANE

LV_ID, RV_ID = 1, 2                    # Rodero cell_data['ID']: LV / RV myocardium (rest dropped)
_SENTINEL = -5.0                       # universal-coord sentinel guard (real values in [-1..1]; atria=-10)


def _sax_align(mesh):
    """Rotate the mesh so the LV long axis (apex->base, from the Z apico-basal coord) points +z, so
    plain axial slices of the voxelized volume are short-axis. Returns the rotated pyvista mesh."""
    pts = np.asarray(mesh.points)
    Z = np.asarray(mesh.point_data["Z.dat"])
    valid = (Z >= 0.0) & (Z <= 1.0)                       # ventricular points only (atria = -10 sentinel)
    apex = pts[valid & (Z < 0.15)].mean(0)
    base = pts[valid & (Z > 0.85)].mean(0)
    axis = base - apex
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(axis, z); s = np.linalg.norm(v); c = float(np.dot(axis, z))
    if s < 1e-6:                                          # already aligned
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))   # Rodrigues
    out = mesh.copy()
    out.points = (pts - apex) @ R.T
    return out


def _wall_mask(mesh, region_id: int, grid) -> np.ndarray:
    """Boolean grid-point mask of one region's wall solid (its closed tet-shell surface encloses the
    grid points that lie in the myocardium)."""
    import pyvista as pv
    sub = mesh.threshold([region_id, region_id], scalars="ID")
    surf = sub.extract_surface().triangulate()
    sel = grid.select_enclosed_points(surf, tolerance=0.0, check_surface=False)
    return np.asarray(sel["SelectedPoints"]).astype(bool)


def voxelize(mesh, inplane: float = DEFAULT_INPLANE, slice_mm: float = 8.0) -> np.ndarray:
    """SAX-aligned 3-class label volume [D, H, W] (RV-cav 1 / LV-myo 2 / LV-cav 3) from a Rodero mesh.
    Walls rasterized via enclosed-points; cavities recovered by 2D hole-fill per SAX slice (LV ring ->
    LV-cav; LV+RV walls together -> RV-cav). D slices at `slice_mm`, in-plane at `inplane` (mm)."""
    import pyvista as pv
    from scipy.ndimage import binary_fill_holes, label as cc_label
    m = _sax_align(mesh)
    x0, x1, y0, y1, z0, z1 = m.bounds
    nx = int(np.ceil((x1 - x0) / inplane)); ny = int(np.ceil((y1 - y0) / inplane))
    nz = int(np.ceil((z1 - z0) / slice_mm))
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(inplane, inplane, slice_mm),
                        origin=(x0, y0, z0))
    lv = _wall_mask(m, LV_ID, grid).reshape(nz, ny, nx)   # ImageData points iterate x-fastest -> (z,y,x)
    rv = _wall_mask(m, RV_ID, grid).reshape(nz, ny, nx)
    out = np.zeros((nz, ny, nx), dtype=np.uint8)
    for k in range(nz):
        lw, rw = lv[k], rv[k]
        if lw.sum() < 8:
            continue
        lv_cav = binary_fill_holes(lw) & ~lw                          # inside the LV wall ring
        both = binary_fill_holes(lw | rw) & ~(lw | rw)               # both cavities enclosed by LV+RV
        rv_cav = both & ~lv_cav
        out[k][rv_cav] = 1                                            # RV cavity
        out[k][lw] = 2                                               # LV myocardium (RV wall -> bg)
        out[k][lv_cav] = LV_CAV                                      # LV cavity (3)
    return out


def load(path: str | Path):
    """Read a Rodero VTK mesh (pyvista); sets 'ID' active for thresholding."""
    import pyvista as pv
    m = pv.read(str(path))
    m.set_active_scalars("ID", preference="cell")
    return m


def _main():
    """Voxelize one Rodero mesh -> SAX label volume; save a mid-slice montage to eyeball the chambers."""
    import argparse
    ap = argparse.ArgumentParser(description="Rodero mesh -> SAX 3-class label volume (bd 1vl).")
    ap.add_argument("--mesh", required=True, help="path to a Rodero .vtk mesh")
    ap.add_argument("--inplane", type=float, default=DEFAULT_INPLANE)
    ap.add_argument("--out", default=None, help="montage PNG (default: <mesh>_sax.png)")
    a = ap.parse_args()
    vol = voxelize(load(a.mesh), inplane=a.inplane)
    counts = {int(c): int((vol == c).sum()) for c in np.unique(vol)}
    print(f"volume {vol.shape}  label counts {counts}  (1=RVcav 2=LVmyo 3=LVcav)")
    from PIL import Image
    D = vol.shape[0]
    ks = [int(D * f) for f in (0.3, 0.45, 0.6, 0.75)]            # mid-ventricular slices
    cmap = np.array([[0, 0, 0], [91, 141, 239], [255, 202, 91], [239, 83, 80]], np.uint8)
    tiles = [cmap[vol[k]] for k in ks]
    montage = np.concatenate(tiles, axis=1)
    out = a.out or (str(Path(a.mesh).with_suffix("")) + "_sax.png")
    Image.fromarray(montage).save(out)
    print(f"wrote {out}  (slices {ks})")


if __name__ == "__main__":
    _main()
