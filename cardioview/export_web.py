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

import numpy as np
import pyvista as pv
import torch
from scipy.ndimage import zoom

from cardioseg.preprocessing.preprocess import preprocess_case
from cardioseg.training.dataset import fit_square, split_patients
from cardioseg.data.mri.data import acdc_cases
from cardioseg.evaluation.measure import ejection_fraction
from render_overlay import CHAMBERS, SIZE, MODELS, load_model, chamber_mesh, patient_dir

OUT = Path("cardioview/web/public/data")


def heldout_set() -> set[str]:
    _, val = split_patients(list(acdc_cases()), 0.2, 0)
    return {c.name for c in val}


def gt_masks(case: dict) -> dict:
    out = {}
    for tag in ("ED", "ES"):
        k = tag.lower()
        if f"{k}_gt" in case:
            out[tag] = np.stack([fit_square(s, SIZE, 0) for s in case[f"{k}_gt"]]).astype(np.uint8)
    return out


def pred_masks(case: dict, model, device) -> dict:
    from cardioseg.evaluation.validate import predict_volume

    out = {}
    for tag in ("ED", "ES"):
        k = tag.lower()
        if f"{k}_img" in case:
            out[tag] = predict_volume(model, case[f"{k}_img"], SIZE, device)
    return out


def shared_crop_iso(masks: dict, spacing, margin_mm: float = 12.0):
    """Crop every phase to the union heart bbox + margin, resample to isotropic (nearest)."""
    union = np.zeros_like(next(iter(masks.values())), dtype=bool)
    for m in masks.values():
        union |= m > 0
    sl = []
    for ax, n in enumerate(union.shape):
        idx = np.any(union, axis=tuple(a for a in range(3) if a != ax)).nonzero()[0]
        pad = int(round(margin_mm / spacing[ax]))
        sl.append(slice(max(0, idx[0] - pad), min(n, idx[-1] + 1 + pad)))
    crop = (sl[0], sl[1], sl[2])
    iso = float(min(spacing))
    factors = tuple(s / iso for s in spacing)
    return {t: zoom(m[crop], factors, order=0) for t, m in masks.items()}, iso


def export_glb(mask_i: np.ndarray, iso: float, path: Path) -> None:
    pl = pv.Plotter(off_screen=True)
    for label, (_name, color) in CHAMBERS.items():
        mesh = chamber_mesh(mask_i, label, iso)
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


def run(patients, source, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(MODELS[model_name], device) if source == "pred" else None
    held = heldout_set()
    manifest = []
    for p in patients:
        case = preprocess_case(patient_dir(p))
        spacing = tuple(float(s) for s in case["spacing"])
        masks = pred_masks(case, model, device) if source == "pred" else gt_masks(case)
        gtm = gt_masks(case)
        crop_masks, iso = shared_crop_iso(masks, spacing)
        glb = {}
        for tag, m in crop_masks.items():
            fn = f"{p}_{tag}_{source}.glb"
            export_glb(m, iso, OUT / fn)
            glb[tag] = fn
        entry = dict(patient=p, group=case.get("group"), held_out=(p in held), source=source,
                     pred=volumes(masks, spacing), gt=volumes(gtm, spacing), glb=glb)
        manifest.append(entry)
        print(f"  {p:11} {str(case.get('group')):5} EF {entry['pred'].get('ef')}% "
              f"(GT {entry['gt'].get('ef')}%)  held_out={p in held}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT/'manifest.json'} ({len(manifest)} patients)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patients", nargs="*", default=["patient006", "patient009", "patient010"])
    ap.add_argument("--source", default="pred", choices=["pred", "gt"])
    ap.add_argument("--model", default="acdc_aug", choices=list(MODELS))
    a = ap.parse_args()
    run(a.patients, a.source, a.model)


if __name__ == "__main__":
    main()
