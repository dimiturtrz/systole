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

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from jaxtyping import Integer
from scipy.ndimage import zoom
from skimage.measure import marching_cubes

from core.config import Config
from core.data.static.labels import CLASSES  # {label: (name, hexcolor)}
from core.data.static.mri.base import Phase
from core.postprocess import Postprocess
from core.types import Spacing, shapecheck

log = logging.getLogger("cardioseg.mesh")

MESH_MM = 2.5      # surface resample step (coarser than voxels -> fewer triangles, still smooth)
DECIMATE = 0.7     # fraction of triangles to drop (smaller files)

_MIN_CHAMBER_VOXELS = 8   # below this a chamber is absent/tiny -> skip meshing
_ISO_LEVEL = 0.5          # marching-cubes iso-surface level on the resampled soft mask


class Mesh:
    """Chamber-mesh generation + export (the free helpers folded in as staticmethods): per-chamber
    marching-cubes surface, colored GLB scene, per-chamber STL, and the subject-dir export driver."""

    @staticmethod
    @shapecheck
    def chamber_surface(mask: Integer[np.ndarray, "d h w"], label: int, spacing: Spacing,
                        iso: float = MESH_MM, decimate: float = DECIMATE):
        """Smooth chamber surface (pyvista PolyData, world mm) or None if the chamber is absent/tiny.
        largest-CC -> isotropic linear resample (no z-staircase) -> zero-pad (cap open base/apex) ->
        marching cubes -> Taubin smooth -> decimate -> oriented normals."""
        binary = Postprocess.largest_cc_binary(mask == label)
        if binary.sum() < _MIN_CHAMBER_VOXELS:
            return None
        soft = zoom(binary.astype(np.float32), tuple(s / iso for s in spacing), order=1)
        if soft.max() < _ISO_LEVEL:
            return None
        soft = np.pad(soft, 1, mode="constant")
        verts, faces, _, _ = marching_cubes(soft, level=_ISO_LEVEL, spacing=(iso, iso, iso))
        verts = verts[:, [2, 1, 0]]                              # (z,y,x) -> (x,y,z)
        fp = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()
        mesh = pv.PolyData(verts, fp).smooth_taubin(n_iter=24, pass_band=0.05)
        if decimate:
            mesh = mesh.decimate(decimate)
        return mesh.compute_normals(auto_orient_normals=True, split_vertices=False)

    @staticmethod
    @shapecheck
    def export_glb(mask: Integer[np.ndarray, "d h w"], spacing: Spacing, path: str | Path,
                   iso: float = MESH_MM) -> Path:
        """Colored multi-chamber GLB (glTF 2.0). Myocardium semi-transparent so the cavity shows."""
        pl = pv.Plotter(off_screen=True)
        for label, (_name, color) in CLASSES.items():
            mesh = Mesh.chamber_surface(mask, label, spacing, iso)
            if mesh is not None:
                pl.add_mesh(mesh, color=color, opacity=0.55 if label == 2 else 1.0, smooth_shading=True)  # noqa: PLR2004 (2 = LV-myo label id)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        pl.export_gltf(str(path)); pl.close()
        return path

    @staticmethod
    @shapecheck
    def export_stl(mask: Integer[np.ndarray, "d h w"], spacing: Spacing, out_dir: str | Path,
                   stem: str, iso: float = MESH_MM) -> list[Path]:
        """One STL per chamber -> out_dir/<stem>_<chamber>.stl. Returns the written paths."""
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for label, (name, _color) in CLASSES.items():
            mesh = Mesh.chamber_surface(mask, label, spacing, iso)
            if mesh is not None:
                p = out_dir / f"{stem}_{name.replace('-', '')}.stl"
                mesh.save(str(p)); written.append(p)
        return written

    @staticmethod
    @shapecheck
    def export_meshes(mask: Integer[np.ndarray, "d h w"], spacing: Spacing, subject: str, formats: Any = ("glb", "stl"),  # noqa: PLR0913  independent mesh-export inputs
                      iso: float = MESH_MM, root: str | Path | None = None) -> Path:
        """Write chamber meshes for one subject to <data>/meshes/<subject>/ (or `root`). GLB (colored
        scene) + STL (per chamber) by default. Returns the subject dir."""
        out = Path(root or Config.data_root("meshes")) / subject
        out.mkdir(parents=True, exist_ok=True)
        if "glb" in formats:
            Mesh.export_glb(mask, spacing, out / f"{subject}.glb", iso)
        if "stl" in formats:
            Mesh.export_stl(mask, spacing, out, subject, iso)
        return out

    @staticmethod
    def add_args(ap: Any) -> None:
        ap.add_argument("--npz", required=True, help="consolidated subject npz (has ed_gt/es_gt + spacing)")
        ap.add_argument("--frame", default=Phase.ED, type=Phase, choices=list(Phase))
        ap.add_argument("--subject", default=None, help="output stem (default: npz filename)")
        ap.add_argument("--out", default=None, help="output root (default: <data>/meshes/ from paths.yaml)")
        ap.add_argument("--formats", nargs="+", default=["glb", "stl"], choices=["glb", "stl"])

    @staticmethod
    def run(args: Any) -> None:
        z = np.load(args.npz, allow_pickle=True)
        mask, spacing = z[f"{args.frame}_gt"], tuple(float(s) for s in z["spacing"])
        subject = args.subject or Path(args.npz).stem
        out = Mesh.export_meshes(mask, spacing, subject, tuple(args.formats), root=args.out)
        log.info(f"wrote {args.formats} for {subject} -> {out}")
