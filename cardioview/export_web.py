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
import torch
from scipy.ndimage import zoom

from core.preprocessing.preprocess import preprocess_case, resample_inplane, zscore
from core.data.static.splits import split_patients
from core.data.static.mri.acdc import acdc_cases, parse_info_cfg, load_ed_es
from core.measure import ejection_fraction
from core.inference import predict_volume
from core.postprocess import largest_cc_per_class
from core.config import data_root
from core.mesh import export_glb                    # reusable chamber-mesh tool (bd 7c9.1)
from common import CHAMBERS, SIZE, MODELS, DEFAULT_MODEL, load_model, model_dir, patient_dir, masks as build_masks, square_stack
from geometry import keep_largest, bbox_slices, nearest_index

# Web assets live OUT of the repo, under the data root (<data>/meshes/cardioview/) — never committed.
# The viewer is served from here (vite base / copy step, bd follow-up), not from cardioview/web/public.
OUT = Path(data_root("meshes")) / "cardioview"


def heldout_set(model_name: str) -> set[str]:
    """Subject IDs the model did NOT train on — derived from the run's saved config
    (the flagship holds out ALL ACDC, not an 80/20 val slice). Falls back to the legacy
    80/20 ACDC val split for older runs without a config.json."""
    run = model_dir(MODELS[model_name])
    cfg_path = run / "config.json"
    if cfg_path.exists():
        from core.hparams import from_json
        from core.data.static import store, splits
        dc = from_json(cfg_path).data
        meta = store.load(list(dc.sources))
        train, val, test = splits.make_split(meta, dc.test_datasets, dc.test_vendors, dc.val_frac,
                                             val_datasets=dc.val_datasets, val_vendors=dc.val_vendors)
        # "held out" = anything the model did NOT train on (val OR test) — ACDC is now val, still unseen.
        return set(val.get_column("subject_id").to_list()) | set(test.get_column("subject_id").to_list())
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




def volumes(masks: dict, spacing) -> dict:
    """EDV (ml full), ESV (ml empty), EF (%) from the LV cavity — full-res, real spacing."""
    if "ED" in masks and "ES" in masks:
        ef, edv, esv = ejection_fraction(masks["ED"], masks["ES"], spacing, lv_label=3)
        return {"ef": round(ef, 1), "edv": round(edv, 1), "esv": round(esv, 1)}
    return {}


def upsert_manifest(entry: dict, model_name: str) -> None:
    """Insert/replace one patient in manifest.json. Manifest = {model, hearts}; the web
    reads `model` to know which bundled .onnx to load (so it follows what was exported)."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    if isinstance(data, list):  # back-compat with the old array form
        data = {"hearts": data}
    hearts = [e for e in data.get("hearts", []) if e["patient"] != entry["patient"]]
    hearts.append(entry)
    hearts.sort(key=lambda e: e["patient"])
    path.write_text(json.dumps({"model": model_name, "hearts": hearts}, indent=2))


def load_4d(pdir, name: str):
    """Load the cine as [t, z, y, x] + spacing (z, y, x) mm."""
    img = nib.load(str(pdir / f"{name}_4d.nii.gz"))
    arr = np.transpose(np.asanyarray(img.dataobj), (3, 2, 1, 0))  # x,y,z,t -> t,z,y,x
    xs, ys, zs = img.header.get_zooms()[:3]
    return arr, (zs, ys, xs)


def frame_indices(pdir):
    """0-based ED, ES frame indices (Info.cfg parsing reused from core)."""
    cfg = parse_info_cfg(pdir)
    return int(cfg["ED"]) - 1, int(cfg["ES"]) - 1


def run(patients, source, model, device, model_name):
    held = heldout_set(model_name)
    for p in patients:
        pdir = patient_dir(p)  # p may be an ID or a full path
        name = pdir.name
        case = preprocess_case(pdir, loader=load_ed_es)
        spacing = tuple(float(s) for s in case["spacing"])
        masks = build_masks(case, source, model, device)
        crop_masks, iso = shared_crop(masks, spacing)
        glb = {}
        for tag, m in crop_masks.items():
            fn = f"{name}_{tag}_{source}.gltf"
            export_glb(m, spacing, OUT / fn)
            glb[tag] = fn
        entry = dict(patient=name, group=case.get("group"), held_out=(name in held), source=source,
                     pred=volumes(masks, spacing), gt=volumes(build_masks(case, "gt"), spacing), glb=glb)
        upsert_manifest(entry, model_name)
        print(f"  {name:11} {str(case.get('group')):5} static  EF {entry['pred'].get('ef')}% "
              f"(GT {entry['gt'].get('ef')}%)")


def run_animate(patients, model, device, model_name, stride=1):
    """Segment every cine frame -> per-frame chamber glb -> a beating-cycle entry."""
    held = heldout_set(model_name)
    for p in patients:
        pdir = patient_dir(p)  # p may be an ID or a full path
        name = pdir.name
        vol, spacing = load_4d(pdir, name)
        rspacing = (spacing[0], 1.5, 1.5)
        frames_t = list(range(0, vol.shape[0], stride))
        masks = {}
        grays = {}
        for k, t in enumerate(frames_t):
            img = zscore(resample_inplane(vol[t].astype(np.float32), spacing, 1.5)[0])
            masks[k] = largest_cc_per_class(predict_volume(model, img, SIZE, device, tta=True))
            grays[k] = square_stack(img)  # [D,SIZE,SIZE] grayscale aligned with the mask
        # slice view: crop to the heart bbox over the whole cine (+margin) -> a compact loaf, not
        # giant mostly-empty FOV planes. Same crop every frame so the stack is stable.
        union = np.zeros((SIZE, SIZE), bool)
        for k in masks:
            union |= (masks[k] > 0).any(axis=0)
        ys, xs = np.where(union)
        M = 14
        r0, r1, c0, c1 = ((max(0, int(ys.min()) - M), min(SIZE, int(ys.max()) + M + 1),
                           max(0, int(xs.min()) - M), min(SIZE, int(xs.max()) + M + 1))
                          if ys.size else (0, SIZE, 0, SIZE))
        slice_files = []
        for k in range(len(frames_t)):
            sfn = f"{name}_f{k:02d}_slices.png"
            save_strip(grays[k][:, r0:r1, c0:c1], masks[k][:, r0:r1, c0:c1], OUT / sfn)
            slice_files.append(sfn)
        crop_masks, iso = shared_crop(masks, rspacing)
        files = []
        for k in range(len(frames_t)):
            fn = f"{name}_f{k:02d}_pred.gltf"
            export_glb(crop_masks[k], rspacing, OUT / fn)
            files.append(fn)
        edi, esi = frame_indices(pdir)
        ed_k = nearest_index(frames_t, edi)
        es_k = nearest_index(frames_t, esi)
        ef, edv, esv = ejection_fraction(masks[ed_k], masks[es_k], rspacing, lv_label=3)
        case = preprocess_case(pdir, loader=load_ed_es)
        gt = volumes(build_masks(case, "gt"), tuple(float(s) for s in case["spacing"]))
        entry = dict(patient=name, group=case.get("group"), held_out=(name in held), source="pred",
                     pred={"ef": round(ef, 1), "edv": round(edv, 1), "esv": round(esv, 1)}, gt=gt,
                     frames=files, ed_idx=ed_k, es_idx=es_k,
                     glb={"ED": files[ed_k], "ES": files[es_k]},
                     slices=slice_files, sliceD=int(masks[0].shape[0]))
        upsert_manifest(entry, model_name)
        print(f"  {name:11} {str(case.get('group')):5} BEATING {len(files)} frames  "
              f"EF {entry['pred']['ef']}% (GT {gt.get('ef')}%)")




def save_strip(gray: np.ndarray, mask: np.ndarray, path: Path) -> None:
    """One cine frame's (already heart-cropped) slices -> a vertical RGBA PNG strip [D*H, W]:
    R = grayscale (percentile-windowed for real MRI contrast), G = label (0..3). The web decodes it
    directly (W from image width, H from height/D)."""
    from PIL import Image

    nz, h, w = gray.shape
    lo, hi = np.percentile(gray, [1, 99])  # window: z-scored MRI has outliers; min/max washes out
    g8 = np.clip((gray - lo) / ((hi - lo) or 1) * 255, 0, 255).astype(np.uint8)
    rgba = np.zeros((nz * h, w, 4), np.uint8)
    for z in range(nz):
        rgba[z * h:(z + 1) * h, :, 0] = g8[z]
        rgba[z * h:(z + 1) * h, :, 1] = mask[z]
        rgba[z * h:(z + 1) * h, :, 3] = 255
    Image.fromarray(rgba, "RGBA").save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["static", "animate"], default="static")
    # Default hearts come from paths.yaml (cardioview.hearts); fallback = one per condition.
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--source", default="pred", choices=["pred", "gt"])
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--stride", type=int, default=1, help="cine frame stride (animate)")
    a = ap.parse_args()
    # canned demo hearts (one per pathology); override with --patients (IDs or full paths).
    patients = a.patients or ["patient073", "patient006", "patient021", "patient053"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(MODELS[a.model], device) if a.source == "pred" or a.mode == "animate" else None
    if a.mode == "animate":
        run_animate(patients, model, device, a.model, a.stride)
    else:
        run(patients, a.source, model, device, a.model)
    print(f"manifest: {OUT/'manifest.json'}")


if __name__ == "__main__":
    main()
