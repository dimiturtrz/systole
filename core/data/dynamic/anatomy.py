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

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float, Integer
from PIL import Image
from pydantic import BaseModel
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
)
from scipy.ndimage import label as cc_label
from scipy.ndimage import zoom as _zoom

from core.config import _VALIDATE, DEFAULT_INPLANE, DEFAULT_SIZE
from core.data.static.labels import LV_CAV, MYO, RV  # 3 / 2 / 1
from core.preprocessing.preprocess import Preprocess
from core.shapecheck import shapecheck
from core.types import Integral


class MeshError(Exception):
    """A mesh could not be read or voxelized into a usable label volume (bad file, degenerate
    geometry, no foreground). The pool builder skips these — one bad mesh must not kill the build."""


class PoolBuildCfg(BaseModel):
    """Voxelization + slice-selection knobs for `build_pool` (offline anatomy-pool build). Bundles what
    were 7 loose args; each default is a measured/physical choice (see field comments), not a free knob."""
    model_config = _VALIDATE
    size: int = DEFAULT_SIZE
    inplane: float = DEFAULT_INPLANE
    min_fg: int = 40             # drop a SAX slice with fewer foreground px than this (near-empty apex/base)
    scale_reps: int = 1          # independent real-size scale draws emitted per mesh
    seed: int = 0
    workers: int = 0             # 0 -> cpu_count-2
    min_cav_frac: float = 0.05   # require cavity >= this fraction of fg (drops pure-apical myo-only slices)


class PathologyPoolCfg(BaseModel):
    """DCM / HCM / abnormal-RV remodel ranges for `build_pathology_pool`. Each is an inclusive U[lo,hi]
    voxel radius for the morphological deform: dcm = LV-cav dilate, hcm = LV-cav erode, rv = RV grow."""
    model_config = _VALIDATE
    k_dcm: tuple[int, int] = (1, 6)
    k_hcm: tuple[int, int] = (1, 4)
    rv: tuple[int, int] = (2, 6)
    seed: int = 0


log = logging.getLogger("cardioseg.anatomy")

LV_ID, RV_ID = 1, 2                    # cell tag: LV / RV myocardium (rest = atria/vessels, dropped)
_SENTINEL = -5.0                       # universal-coord sentinel guard (real values in [-1..1]; atria=-10)

_PARALLEL_EPS = 1e-6                    # |sin| below this -> axis already ~parallel to +z (no rotation)
_APEX_Z_FRAC = 0.15                    # apico-basal coord Z below this = apical band (apex centroid)
_BASE_Z_FRAC = 0.85                    # apico-basal coord Z above this = basal band (base centroid)
_ZOOM_NOOP_EPS = 1e-3                  # |scale factor - 1| below this -> skip the rescale (no-op)

_MIN_LV_WALL_PX = 8   # fewer LV-wall px than this in a SAX slice -> too little to close a cavity ring, skip

# Real ACDC heart size at 1.5 mm in-plane: per-patient MAX fg-bbox longest side (px), measured over
# 150 patients x ED+ES (core/data/static/acdc): p5=62 p50=74 p95=89. Rodero voxelizes ~25-35% smaller
# in-frame (~47 px), so we globally rescale each synth heart to a target sampled from this real range —
# a per-heart factor that CENTERS scale on real AND spreads it across the real spread (domain-random),
# while preserving each heart's own apico-basal size profile (uniform in-plane zoom, not per-slice).
REAL_SIZE_PX = (58, 92)               # sample target max-bbox side here (measured real p5..p95, widened)


class Anatomy:
    """Rodero SSM anatomy engine (the free functions folded in as staticmethods): mesh region-tag lookup,
    SAX alignment, wall/cavity voxelization, real-size scaling, label-space pathology deform, and the
    offline pool builders (`build_pool`, `build_pathology_pool`, `load_pool`)."""

    @staticmethod
    def _tag_name(mesh) -> str:
        """The per-cell region-tag array. Two Rodero packagings: real-patient cohort (4590294) tags it
        'ID' + ships 'Z.dat' universal coords; the 1000 SSM-sampled cohort (4506930) tags it 'elemTag'
        with NO point data. Same 1..24 scheme (1=LV, 2=RV myo)."""
        for key in ("ID", "elemTag"):
            if key in mesh.cell_data:
                return key
        raise KeyError(f"no region-tag cell array (looked for ID/elemTag); have {list(mesh.cell_data)}")

    @staticmethod
    @shapecheck
    def _rot_to_z(axis: Float[np.ndarray, "3"]) -> Float[np.ndarray, "3 3"]:
        """Rodrigues rotation matrix taking unit `axis` -> +z."""
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        z_axis = np.array([0.0, 0.0, 1.0])
        cross = np.cross(axis, z_axis); sine = np.linalg.norm(cross); cosine = float(np.dot(axis, z_axis))
        if sine < _PARALLEL_EPS:
            return np.eye(3)
        skew_matrix = np.array([[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]])
        return np.eye(3) + skew_matrix + skew_matrix @ skew_matrix * ((1 - cosine) / (sine * sine))

    @staticmethod
    def _sax_align(mesh):
        """Rotate the mesh so the LV long axis (apex->base) points +z, so plain axial slices of the
        voxelized volume are short-axis. Uses the 'Z.dat' apico-basal coord when present (real cohort);
        else derives the axis geometrically (PCA long-axis of the LV myocardium, oriented apex->base by
        the atria/vessel centroid) — the 1000 SSM cohort ships no universal coords."""
        points = np.asarray(mesh.points)
        tag_name = Anatomy._tag_name(mesh)
        if "Z.dat" in mesh.point_data:
            long_axis_position = np.asarray(mesh.point_data["Z.dat"])   # 0=apex .. 1=base, atria = -10 sentinel
            ventricular = (long_axis_position >= 0.0) & (long_axis_position <= 1.0)
            apex = points[ventricular & (long_axis_position < _APEX_Z_FRAC)].mean(0)
            base = points[ventricular & (long_axis_position > _BASE_Z_FRAC)].mean(0)
            axis = base - apex
            rotation_matrix = Anatomy._rot_to_z(axis)
            origin = apex
        else:
            cell_tags = np.asarray(mesh.cell_data[tag_name])
            lv_points = np.asarray(mesh.threshold([LV_ID, LV_ID], scalars=tag_name).points)
            lv_centroid = lv_points.mean(0)
            _, _, principal_axes = np.linalg.svd(lv_points - lv_centroid, full_matrices=False)
            axis = principal_axes[0]                           # LV long axis = 1st principal component
            max_tag = int(cell_tags.max())
            if max_tag > RV_ID:                                # orient toward base (atria/vessels = tag>2)
                base_centroid = np.asarray(mesh.threshold([RV_ID + 1, max_tag], scalars=tag_name).points).mean(0)
                if np.dot(base_centroid - lv_centroid, axis) < 0:
                    axis = -axis
            long_axis_projection = (lv_points - lv_centroid) @ axis
            apex = lv_centroid + axis * long_axis_projection.min()   # apex = far LV tip opposite the base
            rotation_matrix = Anatomy._rot_to_z(axis)
            origin = apex
        out = mesh.copy()
        out.points = (points - origin) @ rotation_matrix.T
        return out

    @staticmethod
    @shapecheck
    def _wall_mask(mesh, region_id: int, grid, tag: str) -> Bool[np.ndarray, "*n"]:
        """Boolean grid-point mask of one region's wall solid (its closed tet-shell surface encloses the
        grid points that lie in the myocardium)."""
        region_mesh = mesh.threshold([region_id, region_id], scalars=tag)
        surface = region_mesh.extract_surface().triangulate()
        selection = grid.select_enclosed_points(surface, tolerance=0.0, check_surface=False)
        return np.asarray(selection["SelectedPoints"]).astype(bool)

    @staticmethod
    @shapecheck
    def _slice_labels(lw: Bool[np.ndarray, "h w"], rw: Bool[np.ndarray, "h w"],
                      rv_close: int = 3) -> Integer[np.ndarray, "h w"]:
        """Pure per-SAX-slice cavity recovery: given the boolean LV-wall (`lw`) and RV-wall (`rw`) rasters,
        return a 3-class label slice (RV-cav 1 / LV-myo 2 / LV-cav 3). Empty (all-bg) if the LV wall is too
        thin to close a cavity ring. LV ring -> LV-cav; LV+RV walls together -> RV-cav (see voxelize doc)."""
        out = np.zeros(lw.shape, dtype=np.uint8)
        if lw.sum() < _MIN_LV_WALL_PX:
            return out
        lv_cav = binary_fill_holes(lw) & ~lw                          # inside the LV wall ring
        # RV cavity: RV free wall + LV(septal) wall enclose it, but the RV crescent rarely closes a
        # ring in-plane (hinge gaps to the LV wall) -> raw fill leaks/empties. Close the wall union to
        # bridge the small gaps, fill, subtract walls+LV-cav, then keep only components touching the RV
        # wall (drops any fill that leaked into background). This is what lifted RV-cav off ~0.21.
        walls = binary_closing(lw | rw, iterations=rv_close) if rv_close > 0 else (lw | rw)
        wall_enclosed = binary_fill_holes(walls) & ~(lw | rw)
        rv_cav = wall_enclosed & ~lv_cav
        if rw.any() and rv_cav.any():
            components, _ = cc_label(rv_cav)
            near_rv_wall = binary_dilation(rw, iterations=2)
            kept_labels = {int(label_value) for label_value in np.unique(components[near_rv_wall & rv_cav]) if label_value}
            rv_cav = np.isin(components, list(kept_labels)) if kept_labels else np.zeros_like(rv_cav)
        out[rv_cav] = RV                                              # RV cavity
        out[lw] = MYO                                                 # LV myocardium (RV wall -> bg)
        out[lv_cav] = LV_CAV                                          # LV cavity (3)
        return out

    @staticmethod
    @shapecheck
    def voxelize(mesh, inplane: float = DEFAULT_INPLANE, slice_mm: float = 8.0,
                 rv_close: int = 3) -> Integer[np.ndarray, "d h w"]:
        """SAX-aligned 3-class label volume [D, H, W] (RV-cav 1 / LV-myo 2 / LV-cav 3) from a Rodero mesh.
        Walls rasterized via enclosed-points; cavities recovered by 2D hole-fill per SAX slice (LV ring ->
        LV-cav; LV+RV walls together -> RV-cav). D slices at `slice_mm`, in-plane at `inplane` (mm)."""
        aligned_mesh = Anatomy._sax_align(mesh)
        tag_name = Anatomy._tag_name(aligned_mesh)
        x0, x1, y0, y1, z0, z1 = aligned_mesh.bounds
        nx = int(np.ceil((x1 - x0) / inplane)); ny = int(np.ceil((y1 - y0) / inplane))
        nz = int(np.ceil((z1 - z0) / slice_mm))
        if min(nx, ny, nz) < 1:                              # degenerate/empty mesh -> no grid to rasterize
            raise MeshError(f"degenerate mesh bounds {aligned_mesh.bounds} -> grid {(nx, ny, nz)}")
        grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(inplane, inplane, slice_mm),
                            origin=(x0, y0, z0))
        lv = Anatomy._wall_mask(aligned_mesh, LV_ID, grid, tag_name).reshape(nz, ny, nx)   # ImageData points iterate x-fastest -> (z,y,x)
        rv = Anatomy._wall_mask(aligned_mesh, RV_ID, grid, tag_name).reshape(nz, ny, nx)
        out = np.zeros((nz, ny, nx), dtype=np.uint8)
        for k in range(nz):
            out[k] = Anatomy._slice_labels(lv[k], rv[k], rv_close=rv_close)
        return out

    @staticmethod
    @shapecheck
    def _scale_to_target(vol: Integer[np.ndarray, "d h w"], target_px: Integral) -> Integer[np.ndarray, "d h2 w2"]:
        """Uniformly in-plane rescale a [D,H,W] label volume so its GLOBAL max fg-bbox longest side hits
        `target_px` (nearest-neighbour, label-preserving). No-op if the heart has no foreground."""
        fg = vol > 0
        if not fg.any():
            return vol
        _, row_indices, col_indices = np.where(fg)   # z discarded — bbox is in-plane (rows, cols) only
        current_side = max(row_indices.max() - row_indices.min(), col_indices.max() - col_indices.min()) + 1
        scale_factor = target_px / max(current_side, 1)
        if abs(scale_factor - 1.0) < _ZOOM_NOOP_EPS:
            return vol
        return _zoom(vol, (1.0, scale_factor, scale_factor), order=0)   # in-plane only; slices (z) untouched

    @staticmethod
    def load(path: str | Path):
        """Read a Rodero mesh (pyvista); sets the region tag active for thresholding. Prefers a sibling
        BINARY .vtu (4.5x faster load than ASCII .vtk: ~0.9s vs ~4.1s) if present — see convert_binary."""
        p = Path(path)
        vtu = p.with_suffix(".vtu")
        try:
            m = pv.read(str(vtu if (p.suffix == ".vtk" and vtu.exists()) else p))
            m.set_active_scalars(Anatomy._tag_name(m), preference="cell")
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            # pyvista read/parse failures (missing/corrupt file, no region tag) surface as these — wrap as
            # our domain error so callers catch MeshError, not a blanket Exception.
            raise MeshError(f"cannot read mesh {p}: {e}") from e
        return m

    @staticmethod
    def _convert_one(mesh_path: str) -> str:
        out = Path(mesh_path).with_suffix(".vtu")
        if not out.exists():
            pv.read(mesh_path).save(str(out), binary=True)
        return str(out)

    @staticmethod
    def convert_binary(mesh_dir: str | Path, workers: int = 0) -> int:
        """One-time: write a BINARY .vtu beside every ASCII .vtk (parallel). load() then uses it (4.5x
        faster). Returns count converted. ASCII parse is ~half the per-mesh voxelize cost."""
        vtk_paths = [str(p) for p in sorted(Path(mesh_dir).rglob("*.vtk"))]
        n_workers = workers if workers > 0 else max(1, (os.cpu_count() or 4) - 2)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(Anatomy._convert_one, vtk_paths, chunksize=1))
        return len(vtk_paths)

    @staticmethod
    @shapecheck
    def pathology_deform(mask: Integer[np.ndarray, "*grid"], k: int = 0,
                         rv_k: int = 0) -> Integer[np.ndarray, "*grid"]:
        """Label-space pathology SOURCE (bd vpn5): remodel LV and/or RV to synthesize the DCM/HCM/abnormal-RV
        tail the healthy SSM misses, keeping the OUTER LV size fixed + myo a RING (topology-safe — the naive-
        deform dead-end, bd bwp). k>0 = DILATE LV cavity into myo (thin wall -> DCM); k<0 = shrink/thicken
        (-> HCM). rv_k>0 = grow the RV cavity into background (-> abnormal/dilated RV; real RV-group RV/LV
        ~2.2 vs synth ~1.3). Returns a new label map (uint8)."""
        lv_cavity, myo = mask == LV_CAV, mask == MYO
        wall = lv_cavity | myo                                    # LV cavity + myocardium (outer size fixed)
        out = mask.copy()
        if lv_cavity.any() and myo.any() and k != 0:
            new_lv_cavity = binary_dilation(lv_cavity, iterations=k) & wall if k > 0 else binary_erosion(lv_cavity, iterations=-k)
            new_myo = wall & ~new_lv_cavity
            if new_myo.sum() >= 0.10 * wall.sum():               # guard: don't dissolve the myo ring
                out[wall] = 2
                out[new_lv_cavity] = LV_CAV
        if rv_k > 0 and (mask == 1).any():                       # abnormal RV: grow RV cavity into bg only
            grown = binary_dilation(mask == RV, iterations=rv_k) & ~(out == MYO) & ~(out == LV_CAV)
            out[grown] = 1
        return out

    @staticmethod
    @shapecheck
    def build_pathology_pool(pool: Integer[np.ndarray, "n h w"], out_path: str | Path,
                             cfg: PathologyPoolCfg | None = None) -> tuple[Path, tuple]:
        """Turn a healthy label pool into a PATHOLOGY pool: per slice emit DCM (LV cavity dilated ~U[k_dcm]),
        HCM (eroded ~U[k_hcm]), and abnormal-RV (RV grown ~U[rv]) variants (topology-safe pathology_deform).
        The composite source covering the DCM/HCM/RV tail the SSM misses (bd vpn5/uch6)."""
        cfg = cfg or PathologyPoolCfg()
        rng = np.random.default_rng(cfg.seed)
        out = []
        for mask in pool:
            out.append(Anatomy.pathology_deform(mask, k=int(rng.integers(cfg.k_dcm[0], cfg.k_dcm[1] + 1))))
            out.append(Anatomy.pathology_deform(mask, k=-int(rng.integers(cfg.k_hcm[0], cfg.k_hcm[1] + 1))))
            out.append(Anatomy.pathology_deform(mask, rv_k=int(rng.integers(cfg.rv[0], cfg.rv[1] + 1))))
        pool_array = np.stack(out).astype(np.uint8)
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, slices=pool_array)
        return out_path, pool_array.shape

    @staticmethod
    def _pool_worker(args) -> list[np.ndarray]:
        """One mesh -> its fit_square'd SAX label slices (the per-mesh unit of build_pool, run in a worker
        process). Seeded per-mesh (seed+index) so the pool is deterministic regardless of finish order."""
        mesh_path, inplane, size, min_fg, scale_reps, min_cav_frac, mesh_seed = args
        rng = np.random.default_rng(mesh_seed)
        out = []
        try:
            vol = Anatomy.voxelize(Anatomy.load(mesh_path), inplane=inplane)
        except MeshError:                                   # bad file / degenerate geometry -> skip this mesh
            return out
        for _ in range(max(1, scale_reps)):
            target_px = int(rng.integers(REAL_SIZE_PX[0], REAL_SIZE_PX[1] + 1))
            scaled_volume = Anatomy._scale_to_target(vol, target_px)
            for k in range(scaled_volume.shape[0]):
                slice_labels = scaled_volume[k]
                fg = int((slice_labels > 0).sum())
                if fg < min_fg:
                    continue
                # drop pure-apical myo-only slices (no cavity): the mesh contributes its whole apico-basal
                # stack, so apex slices (~all myo, no cavity) over-represent 11x vs real (19% vs 1.8%). Require
                # some cavity to match the real slice composition — cleans the synth over-spread (bd uy4d).
                if min_cav_frac > 0 and int(((slice_labels == RV) | (slice_labels == LV_CAV)).sum()) < min_cav_frac * fg:
                    continue
                out.append(Preprocess.fit_square(slice_labels, size, 0).astype(np.uint8))
        return out

    @staticmethod
    def build_pool(mesh_dir: str | Path, out_path: str | Path,
                   cfg: PoolBuildCfg | None = None) -> tuple[Path, tuple]:
        """Voxelize every *.vtk in `mesh_dir` -> scale-match to real -> SAX slices -> fit_square to `cfg.size`
        -> stacked label pool, saved to `out_path` (npz 'slices' [N,size,size] uint8). The synthetic-
        ANATOMY training pool: label maps only (the physics painter adds contrast per batch). Near-empty
        apex/base slices dropped. Each mesh is globally rescaled to a target max-bbox side sampled from the
        real ACDC size distribution (REAL_SIZE_PX); `cfg.scale_reps` emits that many independent scale draws
        per mesh. Meshes are voxelized in PARALLEL (`cfg.workers` processes; 0 -> cpu-2, the bottleneck is the
        per-mesh select_enclosed_points on ~2M-cell tets) — embarrassingly parallel, deterministic."""
        cfg = cfg or PoolBuildCfg()
        # discover meshes by STEM: prefer the binary .vtu (4.5x faster load; the ASCII .vtk is a redundant
        # duplicate and is pruned from storage), fall back to .vtk for a fresh, unconverted download.
        stems = {p.with_suffix("") for p in Path(mesh_dir).rglob("*.vtu")}
        stems |= {p.with_suffix("") for p in Path(mesh_dir).rglob("*.vtk")}
        meshes = sorted((s.with_suffix(".vtu") if s.with_suffix(".vtu").exists() else s.with_suffix(".vtk"))
                        for s in stems)
        n_workers = cfg.workers if cfg.workers > 0 else max(1, (os.cpu_count() or 4) - 2)
        jobs = [(str(mesh_path), cfg.inplane, cfg.size, cfg.min_fg, cfg.scale_reps, cfg.min_cav_frac, cfg.seed + i)
                for i, mesh_path in enumerate(meshes)]
        slices: list[np.ndarray] = []
        if n_workers == 1 or len(jobs) <= 1:
            for job in jobs:
                slices.extend(Anatomy._pool_worker(job))
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                for part in ex.map(Anatomy._pool_worker, jobs, chunksize=1):
                    slices.extend(part)
        pool_array = np.stack(slices) if slices else np.zeros((0, cfg.size, cfg.size), np.uint8)
        out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, slices=pool_array)
        return out_path, pool_array.shape

    @staticmethod
    @shapecheck
    def load_pool(path: str | Path) -> Integer[np.ndarray, "n s s"]:
        """Load the anatomy pool [N, size, size] (label maps) built by build_pool."""
        return np.load(str(path))["slices"]

    @staticmethod
    def _cmd_view(args) -> None:  # pragma: no cover
        """Voxelize one Rodero mesh -> SAX label volume; save a mid-slice montage to eyeball the chambers."""
        vol = Anatomy.voxelize(Anatomy.load(args.mesh), inplane=args.inplane)
        counts = {int(label): int((vol == label).sum()) for label in np.unique(vol)}
        log.info(f"volume {vol.shape}  label counts {counts}  (1=RVcav 2=LVmyo 3=LVcav)")
        depth = vol.shape[0]
        slice_indices = [int(depth * f) for f in (0.3, 0.45, 0.6, 0.75)]   # mid-ventricular slices
        cmap = np.array([[0, 0, 0], [91, 141, 239], [255, 202, 91], [239, 83, 80]], np.uint8)
        slice_images = [cmap[vol[k]] for k in slice_indices]
        montage = np.concatenate(slice_images, axis=1)
        out = args.out or (str(Path(args.mesh).with_suffix("")) + "_sax.png")
        Image.fromarray(montage).save(out)
        log.info(f"wrote {out}  (slices {slice_indices})")

    @staticmethod
    def _cmd_build_pool(args) -> None:  # pragma: no cover
        """Build the healthy SSM anatomy pool from a mesh dir (the ad-hoc REPL build, now committed)."""
        cfg = PoolBuildCfg(size=args.size, scale_reps=args.scale_reps, workers=args.workers)
        out_path, shape = Anatomy.build_pool(args.mesh_dir, args.out, cfg)
        log.info(f"wrote pool {out_path}  shape {shape}")

    @staticmethod
    def _cmd_build_pathology_pool(args) -> None:  # pragma: no cover
        """Build the DCM/HCM/abnormal-RV pathology pool from a healthy pool npz."""
        out_path, shape = Anatomy.build_pathology_pool(Anatomy.load_pool(args.pool), args.out)
        log.info(f"wrote pathology pool {out_path}  shape {shape}")

    @staticmethod
    def _cmd_convert_binary(args) -> None:  # pragma: no cover
        """One-time: write a binary .vtu beside every ASCII .vtk (faster load)."""
        n = Anatomy.convert_binary(args.mesh_dir, args.workers)
        log.info(f"converted {n} meshes under {args.mesh_dir}")

    @staticmethod
    def add_args(ap):
        sub = ap.add_subparsers(dest="cmd", required=True)

        v = sub.add_parser("view", help="voxelize one mesh -> SAX label montage PNG")
        v.add_argument("--mesh", required=True, help="path to a Rodero .vtk mesh")
        v.add_argument("--inplane", type=float, default=DEFAULT_INPLANE)
        v.add_argument("--out", default=None, help="montage PNG (default: <mesh>_sax.png)")

        bp = sub.add_parser("build-pool", help="voxelize a mesh dir -> healthy anatomy pool npz")
        bp.add_argument("--mesh-dir", required=True, help="dir of Rodero .vtk/.vtu meshes")
        bp.add_argument("--out", required=True, help="output pool npz")
        bp.add_argument("--size", type=int, default=DEFAULT_SIZE)
        bp.add_argument("--workers", type=int, default=0)
        bp.add_argument("--scale-reps", type=int, default=1)

        pp = sub.add_parser("build-pathology-pool", help="healthy pool -> DCM/HCM/RV pathology pool npz")
        pp.add_argument("--pool", required=True, help="input healthy pool npz")
        pp.add_argument("--out", required=True, help="output pathology pool npz")

        cb = sub.add_parser("convert-binary", help="write binary .vtu beside each ASCII .vtk (faster load)")
        cb.add_argument("--mesh-dir", required=True, help="dir of Rodero .vtk meshes")
        cb.add_argument("--workers", type=int, default=0)

    @classmethod
    def run(cls, args):  # pragma: no cover
        {"view": cls._cmd_view, "build-pool": cls._cmd_build_pool,
         "build-pathology-pool": cls._cmd_build_pathology_pool,
         "convert-binary": cls._cmd_convert_binary}[args.cmd](args)
