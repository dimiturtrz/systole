"""Intensity volume raycast of an ACDC cardiac MRI frame.

True GPU volume rendering of the greyscale MRI — the actual 3D heart from voxels.
Spacing-aware: ACDC short-axis stacks are ~6–7x anisotropic (z ≈ 10 mm vs ~1.5 mm
in-plane), so we resample to isotropic with SciPy before raycasting, else the heart
looks crushed and the slices read as steps.

Usage:
    uv run python cardioview/render_volume.py \
        --patient patient001 --phase ED --out cardioview/out/patient001_ED.png
    ... --interactive        # open a window instead of a screenshot
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from core.data.mri.acdc import load_ed_es
from common import patient_dir  # noqa: F401  (re-exported; used here and by render_overlay)
from geometry import bbox_slices


def crop_to_heart(vol_zyx, gt_zyx, spacing_zyx, margin_mm: float = 15.0):
    """Crop to the labeled-heart bounding box + a mm margin, so we show the heart not the chest."""
    if gt_zyx is None or not np.any(gt_zyx > 0):
        return vol_zyx, gt_zyx
    crop = bbox_slices(gt_zyx > 0, spacing_zyx, margin_mm)
    return vol_zyx[crop], gt_zyx[crop]


def to_isotropic(vol_zyx: np.ndarray, spacing_zyx: tuple[float, float, float]):
    """Resample a [z,y,x] volume to isotropic voxels at the finest in-plane spacing."""
    iso = float(min(spacing_zyx))
    factors = tuple(s / iso for s in spacing_zyx)
    out = zoom(vol_zyx.astype(np.float32), factors, order=1)
    return out, (iso, iso, iso)


def normalize(vol: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    """Robust min-max to [0,1] (ACDC intensity is uncalibrated)."""
    lo, hi = np.percentile(vol, [lo_pct, hi_pct])
    return np.clip((vol - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def to_imagedata(vol_zyx: np.ndarray, spacing_zyx: tuple[float, float, float]):
    """Wrap a [z,y,x] array as a pyvista ImageData (VTK wants x-fastest)."""
    import pyvista as pv

    nz, ny, nx = vol_zyx.shape
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=spacing_zyx[::-1])  # spacing as (x,y,z)
    grid.point_data["intensity"] = vol_zyx.transpose(2, 1, 0).flatten(order="F")
    return grid


def render(patient: str, phase: str, out: str | None, interactive: bool,
           crop: bool = True, margin_mm: float = 15.0) -> None:
    import pyvista as pv

    d = load_ed_es(patient_dir(patient))
    if phase not in d:
        raise SystemExit(f"phase {phase} not available for {patient} (have {list(d)})")
    vol = d[phase]["img"]
    gt = d[phase].get("gt")
    spacing = d["spacing"]
    crop_vol, _ = crop_to_heart(vol, gt, spacing, margin_mm) if crop else (vol, gt)
    iso_vol, iso_spacing = to_isotropic(crop_vol, spacing)
    grid = to_imagedata(normalize(iso_vol) * 255.0, iso_spacing)

    pl = pv.Plotter(off_screen=not interactive, window_size=(1000, 1000))
    pl.set_background("#0e1116")
    # Translucent ramp — fade the dark background, stay see-through at the bright end so
    # the chambers/blood pool read instead of a solid opaque shell.
    opacity = [0.0, 0.0, 0.03, 0.07, 0.14, 0.26, 0.5]
    pl.add_volume(
        grid,
        scalars="intensity",
        cmap="bone",
        opacity=opacity,
        shade=True,
        ambient=0.3,
        diffuse=0.7,
        specular=0.3,
        blending="composite",
        show_scalar_bar=False,
    )
    pl.view_isometric()
    pl.camera.azimuth = 35
    pl.camera.elevation = 20
    pl.add_text(f"{patient}  {phase}", font_size=12, color="#cdd6e0")

    if interactive:
        pl.show()
        return
    out = out or f"cardioview/out/{patient}_{phase}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(out)
    print(f"saved {out}  (vol {vol.shape} @ {tuple(round(s,2) for s in spacing)} mm "
          f"-> iso {iso_vol.shape} @ {round(iso_spacing[0],2)} mm)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient", default="patient001")
    ap.add_argument("--phase", default="ED", choices=["ED", "ES"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--no-crop", dest="crop", action="store_false", help="render the full FOV, not just the heart ROI")
    ap.add_argument("--margin", type=float, default=15.0, help="crop margin around the heart, mm")
    args = ap.parse_args()
    render(args.patient, args.phase, args.out, args.interactive, args.crop, args.margin)


if __name__ == "__main__":
    main()
