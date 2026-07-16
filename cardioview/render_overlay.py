"""Segmentation overlay: chamber surfaces over the MRI volume — GT or model prediction.

Everything renders in the model's preprocessed grid (in-plane resampled to 1.5 mm, square
256), so the volume, the ground-truth mask, and the predicted mask align with no
back-mapping. Chambers are marching-cubes surfaces (LV cavity / myocardium / RV) over a
dim translucent intensity raycast. Ejection fraction (pred and GT) is computed from the
LV-cavity volumes at ED vs ES and shown in the title.

Usage:
    uv run python cardioview/render_overlay.py \
        --patient patient001 --phase ED --source pred
    ... --source gt    # ground-truth masks instead of the model
    ... --interactive  # rotatable window
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from common import (
    CHAMBERS,
    DEFAULT_MODEL,
    MODELS,
    load_model,
    log_setup,
    patient_dir,
    square_stack,
)
from common import (
    masks as build_masks,
)
from geometry import bbox_slices
from render_volume import normalize, to_imagedata
from scipy.ndimage import zoom
from skimage.measure import marching_cubes

from core.data.static.mri.acdc import AcdcAdapter
from core.data.static.splits import Splits
from core.measure import Measure
from core.preprocessing.preprocess import Preprocess

log = logging.getLogger("cardioview.render_overlay")

MIN_MESH_VOXELS = 8        # skip a chamber whose binary mask is smaller than this (marching-cubes noise)
MYO_LABEL = 2              # LV myocardium — rendered semi-transparent so cavities stay visible


def crop_and_iso(img_zyx, mask_zyx, spacing_zyx, margin_mm=12.0):
    """Crop both to the heart bbox + margin, then resample both to isotropic voxels."""
    crop = bbox_slices(mask_zyx > 0, spacing_zyx, margin_mm)
    img, mask = img_zyx[crop], mask_zyx[crop]
    iso = float(min(spacing_zyx))
    factors = tuple(s / iso for s in spacing_zyx)
    img_i = zoom(img.astype(np.float32), factors, order=1)
    mask_i = zoom(mask, factors, order=0)  # nearest — preserve labels
    return img_i, mask_i, (iso, iso, iso)


def chamber_mesh(mask_zyx, label, iso):  # pragma: no cover  (marching_cubes + pyvista PolyData — mesh/render shell)
    """Marching-cubes surface for one label, in (x,y,z) world mm to match the volume."""
    binary = (mask_zyx == label).astype(np.float32)
    if binary.sum() < MIN_MESH_VOXELS:
        return None
    verts, faces, _, _ = marching_cubes(binary, level=0.5, spacing=(iso, iso, iso))
    verts = verts[:, [2, 1, 0]]  # (z,y,x) -> (x,y,z), the volume's world order
    faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).ravel()
    return pv.PolyData(verts, faces_pv).smooth_taubin(n_iter=20, pass_band=0.05)


@dataclass
class OverlayCfg:
    """One overlay-render request: what to load, how to crop, and where to write."""
    patient: str = "patient001"
    phase: str = "ED"
    source: str = "pred"
    model_name: str = DEFAULT_MODEL
    margin_mm: float = 12.0
    out: str | None = None
    interactive: bool = False
    html: str | None = None
    gltf: str | None = None


def _split_tag(patient: str) -> str:
    """Honesty tag: was this patient in the model's training set? (warns on TRAIN-seen preds)."""
    _, val = Splits.split_patients(list(AcdcAdapter().cases()), 0.2, 0)
    held = patient in {c.name for c in val}
    if not held:
        log.warning("%s was in training — pred overstates the model. Use a held-out patient.", patient)
    return "  held-out" if held else "  TRAIN-seen"


def _ef_title(masks: dict, case: dict, spacing, source: str) -> str:
    """EF (both phases) for the scene title — pred vs GT."""
    if "ED" not in masks or "ES" not in masks:
        return ""
    ef, _, _ = Measure.ejection_fraction(masks["ED"], masks["ES"], spacing, lv_label=3)
    ef_g, _, _ = Measure.ejection_fraction(
        *(square_stack(case[f"{t}_gt"], np.uint8) for t in ("ed", "es")), spacing, lv_label=3)
    return f"   EF {source} {ef:.0f}%  (GT {ef_g:.0f}%)"


def _write_scene(pl, cfg: OverlayCfg, mask_i, iso, ef_txt: str) -> None:  # pragma: no cover  (file-write shell)
    """Dispatch the built plotter to gltf / html / interactive / screenshot per cfg."""
    if cfg.gltf:
        Path(cfg.gltf).parent.mkdir(parents=True, exist_ok=True)
        pl.export_gltf(cfg.gltf)
        log.info("saved %s  (glb for the web viewer)%s", cfg.gltf, ef_txt)
        return
    if cfg.html:
        Path(cfg.html).parent.mkdir(parents=True, exist_ok=True)
        pl.export_html(cfg.html)
        log.info("saved %s  (rotatable web scene)%s", cfg.html, ef_txt)
        return
    if cfg.interactive:
        pl.show()
        return
    out = cfg.out or f"cardioview/out/{cfg.patient}_{cfg.phase}_{cfg.source}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(out)
    log.info("saved %s  (iso %s @ %s mm)%s", out, mask_i.shape, round(iso[0], 2), ef_txt)


def _build_plotter(cfg: OverlayCfg, img_i, mask_i, iso, title: str):  # pragma: no cover  (render shell)
    """Assemble the pyvista scene: dim intensity backdrop (screenshot only) + chamber surfaces."""
    pl = pv.Plotter(off_screen=not cfg.interactive, window_size=(1000, 1000))
    pl.set_background("#0e1116")
    # Volume backdrop doesn't export to vtk.js/glTF cleanly — skip it for web export.
    if not (cfg.html or cfg.gltf):
        grid = to_imagedata(normalize(img_i) * 255.0, iso)
        pl.add_volume(grid, scalars="intensity", cmap="bone",
                      opacity=[0.0, 0.0, 0.02, 0.04, 0.08, 0.14, 0.25],  # dim backdrop
                      shade=False, show_scalar_bar=False, blending="composite")
    for label, (name, color) in CHAMBERS.items():
        mesh = chamber_mesh(mask_i, label, iso[0])
        if mesh is not None:
            pl.add_mesh(mesh, color=color, opacity=0.55 if label == MYO_LABEL else 1.0,
                        smooth_shading=True, specular=0.3, label=name)
    pl.add_legend(bcolor="#161a20", border=False, size=(0.26, 0.16))
    pl.view_isometric()
    pl.camera.azimuth = 35
    pl.camera.elevation = 20
    pl.add_text(title, font_size=11, color="#cdd6e0")
    return pl


def render(cfg: OverlayCfg) -> None:  # pragma: no cover  (render shell)
    case = Preprocess.preprocess_case(patient_dir(cfg.patient), loader=AcdcAdapter().load_ed_es)
    spacing = tuple(float(s) for s in case["spacing"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None if cfg.source == "gt" else load_model(MODELS[cfg.model_name], device)
    split_tag = _split_tag(cfg.patient) if cfg.source == "pred" else ""
    masks = build_masks(case, cfg.source, model, device)
    if cfg.phase not in masks:
        raise SystemExit(f"phase {cfg.phase} unavailable for {cfg.patient}")
    ef_txt = _ef_title(masks, case, spacing, cfg.source)
    img = square_stack(case[f"{cfg.phase.lower()}_img"])
    img_i, mask_i, iso = crop_and_iso(img, masks[cfg.phase], spacing, cfg.margin_mm)
    title = f"{cfg.patient}  {cfg.phase}  [{cfg.source}]{split_tag}{ef_txt}"
    pl = _build_plotter(cfg, img_i, mask_i, iso, title)
    _write_scene(pl, cfg, mask_i, iso, ef_txt)


def main():  # pragma: no cover  (argparse CLI entry — render shell)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", default="patient001")
    ap.add_argument("--phase", default="ED", choices=["ED", "ES"])
    ap.add_argument("--source", default="pred", choices=["pred", "gt"])
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--margin", type=float, default=12.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--html", default=None, help="export a rotatable standalone web page (vtk.js)")
    ap.add_argument("--gltf", default=None, help="export chamber meshes as .glb for the web viewer")
    args = ap.parse_args()
    log_setup()
    render(OverlayCfg(patient=args.patient, phase=args.phase, source=args.source,
                      model_name=args.model, margin_mm=args.margin, out=args.out,
                      interactive=args.interactive, html=args.html, gltf=args.gltf))


if __name__ == "__main__":
    main()
