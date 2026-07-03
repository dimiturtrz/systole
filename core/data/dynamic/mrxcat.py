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

from pathlib import Path

import numpy as np

from core.data.static.labels import LV_CAV  # 3
from core.config import DEFAULT_SIZE

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


def load_vti_labels(path: str | Path) -> np.ndarray:
    """Read an MRXCAT `.vti` phantom → integer label volume [nz, ny, nx] (the `labels` point array).
    pyvista (the `viz` extra, lazy) — same reader family as `anatomy.load`. VTK ImageData is x-fastest;
    reshape Fortran-order to (nx,ny,nz) then move to (nz,ny,nx) so axis 0 indexes short-axis-ish slices."""
    import pyvista as pv
    g = pv.read(str(path))
    nx, ny, nz = g.dimensions
    lab = np.asarray(g.point_data["labels"]).reshape((nx, ny, nz), order="F")
    return np.moveaxis(lab, 2, 0)                  # → [nz, ny, nx]


def build_pool(vti_dir: str | Path, out_path: str | Path, size: int = DEFAULT_SIZE,
               min_fg: int = 40, min_cav_frac: float = 0.05) -> tuple[Path, tuple]:
    """Every `*.vti` in `vti_dir` → canonical label volume → SAX slices → `fit_square` to `size` → stacked
    label pool, saved to `out_path` (npz `slices` [N,size,size] uint8 — the shared `anatomy.load_pool`
    schema, so the generator consumes it identically). Near-empty and cavity-less slices dropped (same
    composition guard as `anatomy.build_pool`: apex slices with no cavity over-represent vs real)."""
    from core.preprocessing.preprocess import fit_square
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
            slices.append(fit_square(s, size, 0).astype(np.uint8))
    arr = np.stack(slices) if slices else np.zeros((0, size, size), np.uint8)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, slices=arr)
    return out_path, arr.shape


def _main():
    """Probe one MRXCAT `.vti`: raw + canonical label counts (sanity-check the remap before building)."""
    import argparse
    ap = argparse.ArgumentParser(description="MRXCAT .vti → canonical label sanity probe (bd hpy).")
    ap.add_argument("--vti", required=True, help="path to an MRXCAT/XCAT phantom .vti")
    a = ap.parse_args()
    vol = load_vti_labels(a.vti)
    raw = {int(c): int((vol == c).sum()) for c in np.unique(vol)}
    canon = to_canonical(vol)
    cc = {int(c): int((canon == c).sum()) for c in np.unique(canon)}
    print(f"vti {vol.shape}\nraw label counts: {raw}\ncanonical (0bg 1RV 2myo 3LVcav): {cc}")


if __name__ == "__main__":
    _main()
