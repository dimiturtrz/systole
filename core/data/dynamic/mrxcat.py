"""MRXCAT2.0 phantom SOURCE (bd cardiac-seg-hpy) — the whole-FOV / motion / physics-PV generation source.

MRXCAT2.0 (Buoso, Joyce, Schulthess, Kozerke, *J. Cardiovasc. Magn. Reson.* 25:25, 2023; MIT) couples an
XCAT torso phantom with a biophysical LV model + a texturizer → whole-FOV cardiac cine with realistic
SURROUNDING ANATOMY, cardiac/respiratory MOTION, and physics-based partial volume. It enters our composite
generation DAG (see `GENERATION.md`) as the *whole-thing* source: breadth the SSM-heart-only pool
(`anatomy.py`) can't reach (lung/liver/chest wall, motion, realistic PV).

We consume MRXCAT's LABEL VOLUMES (VTK `.vti`, `labels` array) only — the same shape currency as the SSM
pool — and paint contrast with OUR bSSFP painter (`synth.py`); MRXCAT's own (MATLAB) grayscale image is not
needed. So the heavy, gated parts of MRXCAT never run here: the MATLAB MR-sim and XCAT (Duke-licensed) torso
generation. The runnable, dependency-light Python stage (`MakePhantom.py`, `runXCAT=False`,
`use_texturizer=False`) produces `.vti` label + T1/T2/PD maps on the bundled example with no MATLAB/XCAT.
The tool itself is vendored under `external/mrxcat2` (gitignored, MIT); THIS module only ADAPTS its output
into our canonical pool (`anatomy.load_pool` format — one shared home for the pool schema).

NB MRXCAT2.0 paints the MYOCARDIUM UNIFORM (`fixLVTexture(..., 'meanLV')`, default `textureLV=False`) *"to
match real image uniformity"* — independent confirmation of our myo over-spread finding (real myo σ is
vendor-stable ~.43–.52; our synth ~4×). We keep our canonical target 4-class; MRXCAT's value is SHAPE +
whole-FOV coverage, painted by the shared engine.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.ndimage import zoom as _zoom

from core.config import DEFAULT_SIZE
from core.data.static.labels import LV_CAV  # 3
from core.obs import setup
from core.preprocessing.preprocess import fit_square

from .anatomy import REAL_SIZE_PX, load_pool

log = logging.getLogger("cardioseg.mrxcat")

# XCAT/MRXCAT label codes → our canonical {0 bg, 1 RV-cav, 2 LV-myo, 3 LV-cav}. Cardiac codes from MRXCAT2.0
# `myLabels(LV_wall=1, RV_wall=2, LV_blood=5, RV_blood=6, Peri=50, Aorta=36)` (MakePhantom.py). Everything
# else (RV wall = no canonical RV-myo class, skeletal muscle, blood 7/8, air/liver/fat/bone, aorta, peri…)
# → background. This is the SEGMENTATION target; whole-FOV surrounding anatomy still shows in the painted
# image (its tissue is background for the mask, contrast for the eye) — the coverage MRXCAT adds.
_XCAT_TO_CANON = {1: 2, 5: LV_CAV, 6: 1}          # LV_wall→myo, LV_blood→LV-cav, RV_blood→RV-cav


def to_canonical(vol: np.ndarray) -> np.ndarray:
    """Remap an MRXCAT/XCAT integer label volume to our canonical 4-class map (uint8). Unmapped codes → bg."""
    out = np.zeros_like(vol, dtype=np.uint8)
    for src, dst in _XCAT_TO_CANON.items():
        out[vol == src] = dst
    return out


# Whole-FOV TISSUE map (bd q4ww): unlike to_canonical (which collapses everything but the heart to bg
# for the SEG TARGET), this keeps the surrounding organs as PAINTABLE classes so the synth image carries
# realistic whole-FOV context — MRXCAT's edge over the heart-only SSM pool. Class → tissue NAME is fixed
# (aligned to mri_physics.TISSUE) so the painter renders each by physical bSSFP contrast. XCAT code groups
# per MRXCAT's own `defineTissuePropertiesMRXCAT` (authoritative). Seg target still = to_canonical (heart
# only); this drives the painted background only — two consistent views of the same phantom volume.
FOV_TISSUE = {0: "lung", 1: "blood", 2: "myocardium", 3: "blood",  # 0 bg/air (lung=darkest TISSUE) | 1 RV-cav
              4: "lung", 5: "liver", 6: "muscle", 7: "fat"}          # 2 myo | 3 LV-cav | + surrounding organs
_XCAT_TO_FOV = {1: 2, 5: 3, 6: 1,                       # heart: LV-wall→myo, LV-blood→LV-cav, RV-blood→RV-cav
                #  (code 2 is a BROAD raw-XCAT label, not just RV wall — render showed stray myo; → muscle)
                15: 4, 16: 4,                            # lung (air-filled)
                13: 5, 40: 5, 41: 5, 42: 5, 43: 5, 52: 5,  # liver group
                50: 7, 99: 7,                            # fat
                7: 6, 8: 6}                               # other blood → soft tissue (not a bright cavity)
_BONE = {31, 32, 33, 34, 35, 51}                          # cortical bone → dark → bg tier


def to_tissue_map(vol: np.ndarray) -> np.ndarray:
    """Whole-FOV paint map (uint8 0..7, classes = `FOV_TISSUE`): heart + surrounding organs, each a
    paintable tissue. Body soft tissue (unlisted, non-air, non-bone codes) → muscle; outside air / bone → bg."""
    out = np.zeros_like(vol, dtype=np.uint8)
    body = (vol != 0) & ~np.isin(vol, list(_BONE))        # inside body, not bone
    out[body] = 6                                          # default soft tissue = muscle
    for src, dst in _XCAT_TO_FOV.items():
        out[vol == src] = dst
    return out


def load_vti_labels(path: str | Path) -> np.ndarray:
    """Read an MRXCAT `.vti` phantom → integer label volume [nz, ny, nx] (the `labels` point array).
    pyvista (the `viz` extra, lazy) — same reader family as `anatomy.load`. VTK ImageData is x-fastest;
    reshape Fortran-order to (nx,ny,nz) then move to (nz,ny,nx) so axis 0 indexes short-axis-ish slices."""
    g = pv.read(str(path))
    nx, ny, nz = g.dimensions
    lab = np.asarray(g.point_data["labels"]).reshape((nx, ny, nz), order="F")
    return np.moveaxis(lab, 2, 0)                  # → [nz, ny, nx]


def _heart_crop_scale(s: np.ndarray, size: int, target_px: int) -> np.ndarray | None:
    """MRXCAT is WHOLE-TORSO (920²): the heart is a small, OFF-CENTRE region, so a plain centre
    `fit_square` crops it away. Crop to the heart bbox, uniformly scale (nearest, label-preserving) so
    its longest side hits `target_px`, then centre `fit_square` to `size` — heart-filling like the SSM
    pool (`anatomy._scale_to_target` intent), drop-in for the shared painter. None if ~empty."""
    ys, xs = np.where(s > 0)
    if ys.size == 0:
        return None
    crop = s[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    f = target_px / max(max(crop.shape), 1)
    if abs(f - 1.0) > 1e-3:
        crop = _zoom(crop, f, order=0)
    return fit_square(crop, size, 0).astype(np.uint8)


def build_pool(vti_dir: str | Path, out_path: str | Path, size: int = DEFAULT_SIZE,
               min_fg: int = 40, min_cav_frac: float = 0.05, seed: int = 0) -> tuple[Path, tuple]:
    """Every `*.vti` in `vti_dir` → canonical label volume → SAX slices → heart-crop+scale-to-real →
    `fit_square` to `size` → stacked label pool, saved to `out_path` (npz `slices` [N,size,size] uint8 —
    the shared `anatomy.load_pool` schema, so the generator consumes it identically). Heart bbox is
    scaled to a target side sampled from the real ACDC size band (`anatomy.REAL_SIZE_PX`) so MRXCAT
    hearts frame like the SSM pool. Near-empty and cavity-less slices dropped (same composition guard as
    `anatomy.build_pool`: apex slices with no cavity over-represent vs real)."""
    rng = np.random.default_rng(seed)
    slices: list[np.ndarray] = []
    for vp in sorted(Path(vti_dir).rglob("*.vti")):
        canon = to_canonical(load_vti_labels(vp))
        for k in range(canon.shape[0]):
            s = canon[k]
            fg = int((s > 0).sum())
            if fg < min_fg:
                continue
            if min_cav_frac > 0 and int(((s == 1) | (s == LV_CAV)).sum()) < min_cav_frac * fg:
                continue
            sq = _heart_crop_scale(s, size, int(rng.integers(REAL_SIZE_PX[0], REAL_SIZE_PX[1] + 1)))
            if sq is not None:
                slices.append(sq)
    arr = np.stack(slices) if slices else np.zeros((0, size, size), np.uint8)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, slices=arr)
    return out_path, arr.shape


def canonical_from_fov(fov: np.ndarray) -> np.ndarray:
    """Seg target from a whole-FOV tissue map: keep heart classes {1,2,3}, everything else → bg. The FOV
    heart codes coincide with `to_canonical` (both from XCAT 6/1/5), so this is the aligned 4-class label."""
    out = np.zeros_like(fov, dtype=np.uint8)
    m = np.isin(fov, (1, 2, 3))
    out[m] = fov[m]
    return out


def _fov_window(s: np.ndarray, size: int, scale: float) -> np.ndarray | None:
    """Crop a `scale`×(heart-bbox) chest WINDOW centred on the heart (realistic cardiac FOV — surrounding
    lung/liver/chest wall, not the whole torso), resize to `size` (nearest). Keeps the whole-FOV context
    the SSM pool lacks, unlike `_heart_crop_scale` (heart-only). None if no heart."""
    heart = np.isin(s, (1, 2, 3))
    if not heart.any():
        return None
    ys, xs = np.where(heart)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * scale / 2) + 1
    y0, y1 = max(0, cy - half), min(s.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(s.shape[1], cx + half)
    win = _zoom(s[y0:y1, x0:x1], size / max(y1 - y0, x1 - x0), order=0)
    return fit_square(win, size, 0).astype(np.uint8)


def build_fov_pool(vti_dir: str | Path, out_path: str | Path, size: int = DEFAULT_SIZE,
                   min_fg: int = 40, scale: float = 3.0) -> tuple[Path, tuple]:
    """WHOLE-FOV pool (bd q4ww): every `*.vti` → `to_tissue_map` → per slice, crop a chest window around
    the heart (`scale`× heart bbox) → resize to `size` → stacked 8-class FOV maps (npz `slices`). Unlike
    `build_pool` (heart-only 4-class), these keep surrounding organs so the painter (bg_mode='mrxcat')
    renders realistic whole-FOV context; the seg target is `canonical_from_fov`. Slices with no heart
    dropped."""
    slices: list[np.ndarray] = []
    for vp in sorted(Path(vti_dir).rglob("*.vti")):
        fov = to_tissue_map(load_vti_labels(vp))
        for k in range(fov.shape[0]):
            if int(np.isin(fov[k], (1, 2, 3)).sum()) < min_fg:
                continue
            w = _fov_window(fov[k], size, scale)
            if w is not None:
                slices.append(w)
    arr = np.stack(slices) if slices else np.zeros((0, size, size), np.uint8)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, slices=arr)
    return out_path, arr.shape


def place_heart_in_fov(fov: np.ndarray, heart: np.ndarray) -> np.ndarray:
    """SSM × MRXCAT (bd majh): drop OUR heart (canonical 1/2/3, heart-centred) into an XCAT whole-FOV
    tissue map — excise the phantom's own heart (→ muscle) and paste ours scaled to that heart's size at
    its location. Gives OUR anatomy diversity inside MRXCAT's realistic surrounding context, without XCAT
    generation or MATLAB. Returns an 8-class FOV map (paint via bg_mode='mrxcat'); None-safe (returns fov
    unchanged if either heart is absent)."""
    hm = np.isin(fov, (1, 2, 3))
    hys, hxs = np.where(heart > 0)
    if not hm.any() or hys.size == 0:
        return fov
    ys, xs = np.where(hm)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    target = max(ys.max() - ys.min(), xs.max() - xs.min()) + 1     # XCAT heart size to match
    out = fov.copy()
    out[hm] = 6                                                     # excise phantom heart → muscle
    crop = heart[hys.min():hys.max() + 1, hxs.min():hxs.max() + 1]
    crop = _zoom(crop, target / max(crop.shape), order=0)
    h, w = crop.shape
    y0, x0 = cy - h // 2, cx - w // 2
    for yy in range(max(0, -y0), min(h, out.shape[0] - y0)):        # paste non-bg heart pixels, clipped
        for xx in range(max(0, -x0), min(w, out.shape[1] - x0)):
            v = crop[yy, xx]
            if v:
                out[y0 + yy, x0 + xx] = v
    return out


def build_ssm_fov_pool(rodero_pool: str | Path, vti_dir: str | Path, out_path: str | Path,
                       size: int = DEFAULT_SIZE, scale: float = 3.0, seed: int = 0) -> tuple[Path, tuple]:
    """SSM × MRXCAT pool (bd majh): each OUR Rodero heart (from `rodero_pool`, canonical) is composited
    into a random XCAT whole-FOV chest window → 8-class FOV map = our anatomy diversity + MRXCAT context.
    Paint with bg_mode='mrxcat'. Reuses `_fov_window` for the XCAT backgrounds and `place_heart_in_fov`."""
    hearts = load_pool(rodero_pool)                                # [N,size,size] canonical (heart-centred)
    bgs = []                                                       # XCAT whole-FOV chest windows (with heart)
    for vp in sorted(Path(vti_dir).rglob("*.vti")):
        fov = to_tissue_map(load_vti_labels(vp))
        for k in range(fov.shape[0]):
            if int(np.isin(fov[k], (1, 2, 3)).sum()) >= 40:
                w = _fov_window(fov[k], size, scale)
                if w is not None:
                    bgs.append(w)
    rng = np.random.default_rng(seed)
    slices = [place_heart_in_fov(bgs[int(rng.integers(len(bgs)))], h) for h in hearts] if bgs else []
    arr = np.stack(slices).astype(np.uint8) if slices else np.zeros((0, size, size), np.uint8)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, slices=arr)
    return out_path, arr.shape


def _main():
    """Probe one MRXCAT `.vti`: raw + canonical label counts (sanity-check the remap before building)."""
    setup()
    ap = argparse.ArgumentParser(description="MRXCAT .vti → canonical label sanity probe (bd hpy).")
    ap.add_argument("--vti", required=True, help="path to an MRXCAT/XCAT phantom .vti")
    a = ap.parse_args()
    vol = load_vti_labels(a.vti)
    raw = {int(c): int((vol == c).sum()) for c in np.unique(vol)}
    canon = to_canonical(vol)
    cc = {int(c): int((canon == c).sum()) for c in np.unique(canon)}
    log.info(f"vti {vol.shape}\nraw label counts: {raw}\ncanonical (0bg 1RV 2myo 3LVcav): {cc}")


if __name__ == "__main__":
    _main()
