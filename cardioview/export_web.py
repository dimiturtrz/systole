"""Precompute the web viewer's assets: per-patient chamber meshes (.glb) + an EF manifest.

The browser app (cardioview/web) is pure rendering — all inference/geometry happens here,
offline, and ships as glb + JSON. ED and ES share one crop/grid so the chambers align when
you toggle phases. Volumes (EDV/ESV/EF) come from the full-res predicted masks × the
patient's real spacing — the viz crop/resample never touches the numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
import torch
from scipy.ndimage import zoom
from skimage.measure import marching_cubes

from cardioseg.preprocessing.preprocess import preprocess_case, resample_inplane, zscore
from cardioseg.training.dataset import split_patients
from cardioseg.data.mri.data import acdc_cases, parse_info_cfg
from cardioseg.evaluation.measure import ejection_fraction
from common import CHAMBERS, SIZE, MODELS, load_model, patient_dir, masks as build_masks
from geometry import keep_largest, bbox_slices, nearest_index

OUT = Path("cardioview/web/public/data")
DECIMATE = 0.7  # fraction of triangles to drop — smaller files, faster web
MESH_MM = 2.5   # surface resample step; coarser than voxels -> far fewer triangles, still smooth


def chamber_surface(mask_anis: np.ndarray, label: int, spacing, iso: float):
    """Smooth chamber surface: largest-CC binary, linear-resampled to isotropic (no
    z-staircase), marching cubes, Taubin-smoothed, decimated. In (x,y,z) world mm."""
    binary = keep_largest(mask_anis == label)
    if binary.sum() < 8:
        return None
    soft = zoom(binary.astype(np.float32), tuple(s / iso for s in spacing), order=1)
    if soft.max() < 0.5:
        return None
    # Pad a zero border so chambers touching the (truncated) stack edges get capped — else
    # marching cubes leaves the base/apex open and the cavity looks hollow.
    soft = np.pad(soft, 1, mode="constant")
    verts, faces, _, _ = marching_cubes(soft, level=0.5, spacing=(iso, iso, iso))
    verts = verts[:, [2, 1, 0]]  # (z,y,x) -> (x,y,z)
    fp = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()
    mesh = pv.PolyData(verts, fp).smooth_taubin(n_iter=24, pass_band=0.05)
    if DECIMATE:
        mesh = mesh.decimate(DECIMATE)
    return mesh.compute_normals(auto_orient_normals=True, split_vertices=False)


def heldout_set() -> set[str]:
    _, val = split_patients(list(acdc_cases()), 0.2, 0)
    return {c.name for c in val}


def shared_crop(masks: dict, spacing, margin_mm: float = 12.0):
    """Crop every phase to the union heart bbox + margin (kept anisotropic — the mesher
    resamples per-chamber with linear interp for smooth surfaces). Returns crops + iso step."""
    union = np.zeros_like(next(iter(masks.values())), dtype=bool)
    for m in masks.values():
        union |= m > 0
    crop = bbox_slices(union, spacing, margin_mm)
    return {t: m[crop] for t, m in masks.items()}, float(min(spacing))


def export_glb(mask_anis: np.ndarray, spacing, path: Path) -> None:
    pl = pv.Plotter(off_screen=True)
    for label, (_name, color) in CHAMBERS.items():
        mesh = chamber_surface(mask_anis, label, spacing, MESH_MM)
        if mesh is not None:
            pl.add_mesh(mesh, color=color, opacity=1.0 if label != 2 else 0.55, smooth_shading=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.export_gltf(str(path))
    pl.close()


def volumes(masks: dict, spacing) -> dict:
    """EDV (ml full), ESV (ml empty), EF (%) from the LV cavity — full-res, real spacing."""
    if "ED" in masks and "ES" in masks:
        ef, edv, esv = ejection_fraction(masks["ED"], masks["ES"], spacing, lv_label=3)
        return {"ef": round(ef, 1), "edv": round(edv, 1), "esv": round(esv, 1)}
    return {}


def upsert_manifest(entry: dict) -> None:
    """Insert/replace one patient in manifest.json (lets static + animated coexist)."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"
    data = json.loads(path.read_text()) if path.exists() else []
    data = [e for e in data if e["patient"] != entry["patient"]]
    data.append(entry)
    data.sort(key=lambda e: e["patient"])
    path.write_text(json.dumps(data, indent=2))


def load_4d(patient: str):
    """Load the cine as [t, z, y, x] + spacing (z, y, x) mm."""
    img = nib.load(str(patient_dir(patient) / f"{patient}_4d.nii.gz"))
    arr = np.transpose(np.asanyarray(img.dataobj), (3, 2, 1, 0))  # x,y,z,t -> t,z,y,x
    xs, ys, zs = img.header.get_zooms()[:3]
    return arr, (zs, ys, xs)


def frame_indices(patient: str):
    """0-based ED, ES frame indices (Info.cfg parsing reused from cardioseg)."""
    cfg = parse_info_cfg(patient_dir(patient))
    return int(cfg["ED"]) - 1, int(cfg["ES"]) - 1


def run(patients, source, model, device):
    held = heldout_set()
    for p in patients:
        case = preprocess_case(patient_dir(p))
        spacing = tuple(float(s) for s in case["spacing"])
        masks = build_masks(case, source, model, device)
        crop_masks, iso = shared_crop(masks, spacing)
        glb = {}
        for tag, m in crop_masks.items():
            fn = f"{p}_{tag}_{source}.gltf"
            export_glb(m, spacing, OUT / fn)
            glb[tag] = fn
        entry = dict(patient=p, group=case.get("group"), held_out=(p in held), source=source,
                     pred=volumes(masks, spacing), gt=volumes(build_masks(case, "gt"), spacing), glb=glb)
        upsert_manifest(entry)
        print(f"  {p:11} {str(case.get('group')):5} static  EF {entry['pred'].get('ef')}% "
              f"(GT {entry['gt'].get('ef')}%)")


def run_animate(patients, model, device, stride=1):
    """Segment every cine frame -> per-frame chamber glb -> a beating-cycle entry."""
    held = heldout_set()
    for p in patients:
        vol, spacing = load_4d(p)
        rspacing = (spacing[0], 1.5, 1.5)
        frames_t = list(range(0, vol.shape[0], stride))
        masks = {}
        for k, t in enumerate(frames_t):
            img = zscore(resample_inplane(vol[t].astype(np.float32), spacing, 1.5)[0])
            masks[k] = predict_volume_local(model, img, device)
        crop_masks, iso = shared_crop(masks, rspacing)
        files = []
        for k in range(len(frames_t)):
            fn = f"{p}_f{k:02d}_pred.gltf"
            export_glb(crop_masks[k], rspacing, OUT / fn)
            files.append(fn)
        edi, esi = frame_indices(p)
        ed_k = nearest_index(frames_t, edi)
        es_k = nearest_index(frames_t, esi)
        ef, edv, esv = ejection_fraction(masks[ed_k], masks[es_k], rspacing, lv_label=3)
        case = preprocess_case(patient_dir(p))
        gt = volumes(build_masks(case, "gt"), tuple(float(s) for s in case["spacing"]))
        entry = dict(patient=p, group=case.get("group"), held_out=(p in held), source="pred",
                     pred={"ef": round(ef, 1), "edv": round(edv, 1), "esv": round(esv, 1)}, gt=gt,
                     frames=files, ed_idx=ed_k, es_idx=es_k,
                     glb={"ED": files[ed_k], "ES": files[es_k]})
        upsert_manifest(entry)
        print(f"  {p:11} {str(case.get('group')):5} BEATING {len(files)} frames  "
              f"EF {entry['pred']['ef']}% (GT {gt.get('ef')}%)")


def predict_volume_local(model, vol_img, device):
    from cardioseg.evaluation.validate import predict_volume

    return predict_volume(model, vol_img, SIZE, device)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["static", "animate"], default="static")
    # One per condition: NOR (normal) · DCM (dilated) · HCM (thick wall) · MINF (infarct).
    ap.add_argument("--patients", nargs="*", default=["patient073", "patient010", "patient021", "patient053"])
    ap.add_argument("--source", default="pred", choices=["pred", "gt"])
    ap.add_argument("--model", default="acdc_aug", choices=list(MODELS))
    ap.add_argument("--stride", type=int, default=1, help="cine frame stride (animate)")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(MODELS[a.model], device) if a.source == "pred" or a.mode == "animate" else None
    if a.mode == "animate":
        run_animate(a.patients, model, device, a.stride)
    else:
        run(a.patients, a.source, model, device)
    print(f"manifest: {OUT/'manifest.json'}")


if __name__ == "__main__":
    main()
