"""Segmentation overlay: chamber surfaces over the MRI volume — GT or model prediction.

Everything renders in the model's preprocessed grid (in-plane resampled to 1.5 mm, square
256), so the volume, the ground-truth mask, and the predicted mask align with no
back-mapping. Chambers are marching-cubes surfaces (LV cavity / myocardium / RV) over a
dim translucent intensity raycast. Ejection fraction (pred and GT) is computed from the
LV-cavity volumes at ED vs ES and shown in the title.

Usage:
    PYTHONPATH=. conda run -n pytorch_training_env python cardioview/render_overlay.py \
        --patient patient001 --phase ED --source pred
    ... --source gt    # ground-truth masks instead of the model
    ... --interactive  # rotatable window
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom
from skimage.measure import marching_cubes

from cardioseg.preprocessing.preprocess import preprocess_case
from cardioseg.training.dataset import fit_square
from cardioseg.evaluation.measure import ejection_fraction
from render_volume import normalize, to_imagedata, patient_dir

# label convention (verified): 1=RV, 2=LV-myo, 3=LV-cavity
CHAMBERS = {
    3: ("LV cavity", "#ef5350"),
    2: ("LV myocardium", "#ffca5b"),
    1: ("RV cavity", "#5b8def"),
}
SIZE = 256
MODELS = {"acdc": "runs/acdc/model.pth", "acdc_aug": "runs/acdc_aug/model.pth"}


def square_stack(vol_zyx: np.ndarray) -> np.ndarray:
    return np.stack([fit_square(s.astype(np.float32), SIZE, 0.0) for s in vol_zyx])


def load_model(weights: str, device):
    import torch
    from cardioseg.training.model import build_unet

    model = build_unet(spatial_dims=2, out_channels=4).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    return model


def masks_for(case: dict, source: str, model, device) -> dict:
    """Return {ED: mask[D,256,256], ES: ...} as GT or predicted, on the square grid."""
    from cardioseg.evaluation.validate import predict_volume

    out = {}
    for tag in ("ED", "ES"):
        key = tag.lower()
        if f"{key}_img" not in case:
            continue
        if source == "gt":
            out[tag] = np.stack([fit_square(s, SIZE, 0) for s in case[f"{key}_gt"]]).astype(np.uint8)
        else:
            out[tag] = predict_volume(model, case[f"{key}_img"], SIZE, device)
    return out


def crop_and_iso(img_zyx, mask_zyx, spacing_zyx, margin_mm=12.0):
    """Crop both to the heart bbox + margin, then resample both to isotropic voxels."""
    sl = []
    for ax, n in enumerate(img_zyx.shape):
        idx = np.any(mask_zyx > 0, axis=tuple(a for a in range(3) if a != ax)).nonzero()[0]
        pad = int(round(margin_mm / spacing_zyx[ax]))
        sl.append(slice(max(0, idx[0] - pad), min(n, idx[-1] + 1 + pad)))
    crop = (sl[0], sl[1], sl[2])
    img, mask = img_zyx[crop], mask_zyx[crop]
    iso = float(min(spacing_zyx))
    factors = tuple(s / iso for s in spacing_zyx)
    img_i = zoom(img.astype(np.float32), factors, order=1)
    mask_i = zoom(mask, factors, order=0)  # nearest — preserve labels
    return img_i, mask_i, (iso, iso, iso)


def chamber_mesh(mask_zyx, label, iso):
    """Marching-cubes surface for one label, in (x,y,z) world mm to match the volume."""
    import pyvista as pv

    binary = (mask_zyx == label).astype(np.float32)
    if binary.sum() < 8:
        return None
    verts, faces, _, _ = marching_cubes(binary, level=0.5, spacing=(iso, iso, iso))
    verts = verts[:, [2, 1, 0]]  # (z,y,x) -> (x,y,z), the volume's world order
    faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()
    return pv.PolyData(verts, faces_pv).smooth_taubin(n_iter=20, pass_band=0.05)


def render(patient, phase, source, out, interactive, model_name, margin_mm):
    import pyvista as pv
    import torch

    case = preprocess_case(patient_dir(patient))
    spacing = tuple(float(s) for s in case["spacing"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None if source == "gt" else load_model(MODELS[model_name], device)
    split_tag = ""
    if source == "pred":  # honesty: was this patient in the model's training set?
        from cardioseg.data.mri.data import acdc_cases
        from cardioseg.training.dataset import split_patients
        _, val = split_patients(list(acdc_cases()), 0.2, 0)
        held = patient in {c.name for c in val}
        split_tag = "  held-out" if held else "  TRAIN-seen"
        if not held:
            print(f"WARNING: {patient} was in training — pred overstates the model. Use a held-out patient.")
    masks = masks_for(case, source, model, device)
    if phase not in masks:
        raise SystemExit(f"phase {phase} unavailable for {patient}")

    # EF (both phases) for the title — the measurement, pred or GT
    ef_txt = ""
    if "ED" in masks and "ES" in masks:
        ef, _, _ = ejection_fraction(masks["ED"], masks["ES"], spacing, lv_label=3)
        ef_g, _, _ = ejection_fraction(
            *(np.stack([fit_square(s, SIZE, 0) for s in case[f"{t}_gt"]]) for t in ("ed", "es")),
            spacing, lv_label=3)
        ef_txt = f"   EF {source} {ef:.0f}%  (GT {ef_g:.0f}%)"

    img = square_stack(case[f"{phase.lower()}_img"])
    img_i, mask_i, iso = crop_and_iso(img, masks[phase], spacing, margin_mm)

    pl = pv.Plotter(off_screen=not interactive, window_size=(1000, 1000))
    pl.set_background("#0e1116")
    grid = to_imagedata(normalize(img_i) * 255.0, iso)
    pl.add_volume(grid, scalars="intensity", cmap="bone",
                  opacity=[0.0, 0.0, 0.02, 0.04, 0.08, 0.14, 0.25],  # dim backdrop
                  shade=False, show_scalar_bar=False, blending="composite")
    for label, (name, color) in CHAMBERS.items():
        mesh = chamber_mesh(mask_i, label, iso[0])
        if mesh is not None:
            pl.add_mesh(mesh, color=color, opacity=1.0 if label != 2 else 0.55,
                        smooth_shading=True, specular=0.3, label=name)
    pl.add_legend(bcolor="#161a20", border=False, size=(0.26, 0.16))
    pl.view_isometric()
    pl.camera.azimuth = 35
    pl.camera.elevation = 20
    pl.add_text(f"{patient}  {phase}  [{source}]{split_tag}{ef_txt}", font_size=11, color="#cdd6e0")

    if interactive:
        pl.show()
        return
    out = out or f"cardioview/out/{patient}_{phase}_{source}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(out)
    print(f"saved {out}  (iso {mask_i.shape} @ {round(iso[0],2)} mm){ef_txt}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", default="patient001")
    ap.add_argument("--phase", default="ED", choices=["ED", "ES"])
    ap.add_argument("--source", default="pred", choices=["pred", "gt"])
    ap.add_argument("--model", default="acdc_aug", choices=list(MODELS))
    ap.add_argument("--margin", type=float, default=12.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()
    render(args.patient, args.phase, args.source, args.out, args.interactive, args.model, args.margin)


if __name__ == "__main__":
    main()
