"""Chamber surface meshes from a label mask (comp-geometry, bd cardiac-seg-7c9.1).

marching cubes per chamber -> Taubin-smoothed, decimated surface -> exported as:
  - GLB (glTF 2.0): one colored multi-chamber scene — web-standard, loads in cardioview / any 3D viewer;
  - STL: one file per chamber — universal (CAD / 3D print), single-mesh, uncolored.

Output goes to `<data>/meshes/<subject>/` (external, gitignored) — never the repo. Spacing-aware
(surfaces in world mm). Reusable: cardioview's web export and any eval/inspection call this. pyvista +
skimage are optional (the `viz` extra); import lazily so core stays importable without them.
Ref: Lorensen & Cline (marching cubes) 1987.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.config import data_root
from core.data.static.labels import CLASSES        # {label: (name, hexcolor)}

MESH_MM = 2.5      # surface resample step (coarser than voxels -> fewer triangles, still smooth)
DECIMATE = 0.7     # fraction of triangles to drop (smaller files)


def _largest_cc(binary: np.ndarray) -> np.ndarray:
    """Largest connected component of a boolean volume (drop stray islands before meshing)."""
    from scipy.ndimage import label as cc_label
    lab, n = cc_label(binary)
    if n <= 1:
        return binary
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == int(sizes.argmax())


def chamber_surface(mask: np.ndarray, label: int, spacing, iso: float = MESH_MM,
                    decimate: float = DECIMATE):
    """Smooth chamber surface (pyvista PolyData, world mm) or None if the chamber is absent/tiny.
    largest-CC -> isotropic linear resample (no z-staircase) -> zero-pad (cap open base/apex) ->
    marching cubes -> Taubin smooth -> decimate -> oriented normals."""
    import pyvista as pv
    from scipy.ndimage import zoom
    from skimage.measure import marching_cubes
    binary = _largest_cc(mask == label)
    if binary.sum() < 8:
        return None
    soft = zoom(binary.astype(np.float32), tuple(s / iso for s in spacing), order=1)
    if soft.max() < 0.5:
        return None
    soft = np.pad(soft, 1, mode="constant")
    verts, faces, _, _ = marching_cubes(soft, level=0.5, spacing=(iso, iso, iso))
    verts = verts[:, [2, 1, 0]]                              # (z,y,x) -> (x,y,z)
    fp = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()
    mesh = pv.PolyData(verts, fp).smooth_taubin(n_iter=24, pass_band=0.05)
    if decimate:
        mesh = mesh.decimate(decimate)
    return mesh.compute_normals(auto_orient_normals=True, split_vertices=False)


def export_glb(mask: np.ndarray, spacing, path: str | Path, iso: float = MESH_MM) -> Path:
    """Colored multi-chamber GLB (glTF 2.0). Myocardium semi-transparent so the cavity shows."""
    import pyvista as pv
    pl = pv.Plotter(off_screen=True)
    for label, (_name, color) in CLASSES.items():
        mesh = chamber_surface(mask, label, spacing, iso)
        if mesh is not None:
            pl.add_mesh(mesh, color=color, opacity=0.55 if label == 2 else 1.0, smooth_shading=True)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pl.export_gltf(str(path)); pl.close()
    return path


def export_stl(mask: np.ndarray, spacing, out_dir: str | Path, stem: str, iso: float = MESH_MM) -> list[Path]:
    """One STL per chamber -> out_dir/<stem>_<chamber>.stl. Returns the written paths."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for label, (name, _color) in CLASSES.items():
        mesh = chamber_surface(mask, label, spacing, iso)
        if mesh is not None:
            p = out_dir / f"{stem}_{name.replace('-', '')}.stl"
            mesh.save(str(p)); written.append(p)
    return written


def export_meshes(mask: np.ndarray, spacing, subject: str, formats=("glb", "stl"),
                  iso: float = MESH_MM, root: str | Path | None = None) -> Path:
    """Write chamber meshes for one subject to <data>/meshes/<subject>/ (or `root`). GLB (colored
    scene) + STL (per chamber) by default. Returns the subject dir."""
    out = Path(root or data_root("meshes")) / subject
    out.mkdir(parents=True, exist_ok=True)
    if "glb" in formats:
        export_glb(mask, spacing, out / f"{subject}.glb", iso)
    if "stl" in formats:
        export_stl(mask, spacing, out, subject, iso)
    return out


def _main():
    """Export chamber meshes for one consolidated subject npz. Location layering: default is
    <data>/meshes/ (config — the paths.yaml data root); --out overrides per invocation (argv)."""
    import argparse
    ap = argparse.ArgumentParser(description="Export chamber meshes (GLB + STL) from a subject npz.")
    ap.add_argument("--npz", required=True, help="consolidated subject npz (has ed_gt/es_gt + spacing)")
    ap.add_argument("--frame", default="ed", choices=["ed", "es"])
    ap.add_argument("--subject", default=None, help="output stem (default: npz filename)")
    ap.add_argument("--out", default=None, help="output root (default: <data>/meshes/ from paths.yaml)")
    ap.add_argument("--formats", nargs="+", default=["glb", "stl"], choices=["glb", "stl"])
    a = ap.parse_args()
    z = np.load(a.npz, allow_pickle=True)
    mask, spacing = z[f"{a.frame}_gt"], tuple(float(s) for s in z["spacing"])
    subject = a.subject or Path(a.npz).stem
    out = export_meshes(mask, spacing, subject, tuple(a.formats), root=a.out)
    print(f"wrote {a.formats} for {subject} -> {out}")


if __name__ == "__main__":
    _main()
