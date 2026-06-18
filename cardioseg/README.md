# cardioseg — the pipeline

The science layer: ACDC cardiac MRI → segmentation → ejection fraction → honest evaluation.
Python (PyTorch + MONAI). The browser demo ([cardioview](../cardioview/)) consumes what this
produces.

## Pipeline
1. **Data** — modality loader + normalization (ACDC short-axis cine MRI; NIfTI, spacing-aware
   in mm; geometric LV/RV disambiguation).
2. **Segment** — 2D U-Net (MONAI) → per-voxel labels (bg / RV / LV-myo / LV-cav).
3. **Measure** — chamber volumes (voxel count × voxel volume, mm³ → mL); **EF = (EDV − ESV) / EDV**.
4. **Evaluate** — Dice + HD95 / ASSD per structure + EF vs GT; **failure ranking** (the worst
   cases decide clinical trust, not the mean).
5. **Geometry/viz** — marching-cubes chamber meshes; error-distribution plots.

## Setup
```bash
pip install -e .                  # from repo root (installs cardioseg)
# torch CUDA build (CPU wheel won't train); Blackwell/RTX 5090 needs torch>=2.7:
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
Real data: register for **ACDC** (Creatis / humanheart-project). Data lives **outside the repo**
(licensing + size) under a `raw/` ↔ `processed/` split. Point at it via `paths.yaml` (copy the
template — gitignored, machine-specific):
```bash
cp paths.example.yaml paths.yaml      # then edit:
#   data:
#     raw: /path/to/data/raw/mri/acdc          # ACDC inputs (dir holding training/)
#     processed: /path/to/data/processed/mri/acdc   # preprocess cache (npz)
```
Loaded by `cardioseg/config.py` (OmegaConf); env vars `CARDIAC_DATA_ROOT` /
`CARDIAC_PROCESSED_ROOT` override (handy for CI).

## Train + evaluate
```bash
python -m cardioseg.training.train --acdc --epochs 40   # -> runs/acdc/{model.pth,metrics.json}
python -m cardioseg.evaluation.distribution --run runs/acdc   # KDE + Bland-Altman -> runs/acdc/plots/
python -m cardioseg.training.export_onnx --run runs/acdc      # model.onnx (+INT8) for the web viewer
```

## Results (held-out, seed 0)
| structure | Dice | HD95 (mm) | ASSD (mm) | published ACDC |
|---|---|---|---|---|
| LV cavity | **0.93** | 2.1 | 0.5 | ~0.93–0.96 |
| LV myocardium | 0.82 | 3.0 | 0.8 | ~0.88–0.92 |
| RV cavity | 0.86 | 10.0 | 1.6 | ~0.88–0.92 |
| **mean** | **0.87** | | | |

- **EF vs GT:** MAE ~3%, bias −1.5%, 95% LoA [−8, +5] (clinical equivalence ≈ ±5%).
- **Metrics rank classes differently** — by Dice RV > myo, but by boundary (HD95) RV is *worst*
  (10 mm): Dice punishes the thin myo ring; RV's boundary is messy (basal slices + stray voxels).
  That's why both an overlap *and* a boundary metric are reported.
- Full HD is the fragile max (one stray voxel → ~200 mm); **HD95** is the robust report.
- `runs/acdc/plots/`: per-class boundary-distance **KDE** + EF **Bland–Altman** — the
  distribution behind the single numbers.

Published column = context, not a trophy: ACDC is single-centre/homogeneous → "competent on a
clean benchmark," not SOTA or clinical-grade. Cross-vendor generalization (e.g. M&Ms) is the
untested hard part.

## Layout
```
cardioseg/
  data/mri/data.py        # ACDC loader (NIfTI, spacing-aware) + geometric LV/RV id, Info.cfg
  preprocessing/preprocess.py   # resample in-plane + z-score; param-keyed disk cache
  training/
    model.py              # MONAI U-Net factory (2D/3D)
    dataset.py            # ACDC 2D-slice dataset, patient-level split
    train.py              # training loop (synthetic + real ACDC)
    export_onnx.py        # trained U-Net -> ONNX (+INT8 quant), torch-parity gated
  evaluation/
    measure.py            # chamber volumes + ejection fraction (spacing-aware)
    evaluate.py           # Dice / surface distances (HD/HD95/ASSD) / failure ranking
    validate.py           # per-class Dice + EF vs GT on held-out patients
    distribution.py       # boundary-distance KDE + EF Bland-Altman
    losses.py             # compound Dice + cross-entropy
  analysis/{eda,viz}.py   # ACDC reality-check + marching-cubes surface mesh
config.py                 # paths.yaml loader (OmegaConf)
```
Tests: `tests/unit` (geometry, metrics, preprocessing) + `tests/integration` (real ACDC, skips
if data absent).
