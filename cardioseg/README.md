# cardioseg — the pipeline

The science layer: cardiac MRI → segmentation → ejection fraction → evaluation, set up for
**domain generalization** (train on multi-vendor **M&M-2**, test on held-out single-centre
**ACDC**). Python (PyTorch + MONAI). The browser demo ([cardioview](../cardioview/)) consumes
what this produces.

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
Real data: register for **ACDC** ([Creatis](https://www.creatis.insa-lyon.fr/Challenge/acdc/)).
Data lives **outside the repo**
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

**M&M-2** (multi-vendor, 360 subjects; register at the [M&Ms-2 challenge](https://www.ub.edu/mnms-2/)) sits **beside**
ACDC — e.g. `data/raw/mri/mnm2/` while ACDC is `…/mri/acdc/`. Auto-discovered as a sibling of the
raw root, or point at it with `CARDIAC_MNM2_ROOT`. Its ground-truth labels are the *opposite* of
ACDC (LV=1 vs LV=3); the loader remaps to the ACDC convention on load (verified geometrically),
so one model spans both datasets.

## Train + evaluate
```bash
# flagship: train multi-vendor M&M-2, hold ACDC out entirely as the test set
python -m cardioseg.training.train --dataset mnm2 --test acdc --epochs 80   # -> runs/mnm2_to_acdc/ (heavy aug needs the epochs)
# single-centre baseline + its OOD drop on M&M-2 (the reverse direction)
python -m cardioseg.training.train --dataset acdc --test mnm2 --epochs 40   # -> runs/acdc_to_mnm2/
python -m cardioseg.evaluation.distribution --run runs/mnm2_to_acdc   # KDE + Bland-Altman -> plots/
python -m cardioseg.training.export_onnx --run runs/mnm2_to_acdc      # model.onnx (+INT8) for the web viewer
```
Training auto-tunes for the GPU: DataLoader workers (`--workers`), mixed precision, cudnn.benchmark.

## Results (seed 0, patient-level splits)
Flagship = **M&M-2 → ACDC** (train multi-vendor, test 100 held-out single-centre patients),
with heavy augmentation (80 ep) + largest-CC postprocessing + test-time augmentation:

| structure | Dice | published ACDC |
|---|---|---|
| LV cavity | **0.94** | ~0.93–0.96 |
| LV myocardium | 0.86 | ~0.88–0.92 |
| RV cavity | 0.88 | ~0.88–0.92 |
| **mean** | **0.89** | |

**EF vs GT: MAE 6.7%** (cross-dataset, bias −6.0%, 95% LoA [−25, +13]). **Diversity buys
robustness — the asymmetry proves it:**

| train → test | mean Dice | RV | EF MAE |
|---|---|---|---|
| ACDC → ACDC (in-domain) | 0.87 | 0.85 | 4.7% |
| ACDC → M&M-2 (out-of-distribution) | 0.70 | 0.59 | 9.1% |
| M&M-2 → ACDC (generalization, flagship) | 0.87 | 0.84 | 9.4% |

*Asymmetry table is the base model (identical config across directions, for a fair A/B); heavy
aug + largest-CC + TTA lift the flagship to 0.89 Dice / 6.7% EF (top table).*

- Single-centre training loses ~17 Dice points off its home dataset (RV collapses 0.85 → 0.59);
  multi-vendor training carries to a new centre with **no segmentation drop**.
- **EF transfers worse than Dice** — volume calibration shifts across centres (in-domain EF MAE
  4.7% → cross-dataset ~9%); the chambers are right, the absolute mL drift.
- **Surface metrics** (flagship eval): by Dice RV > myo, but by boundary (HD95) RV is
  *worst* (~5.4 mm) — Dice punishes the thin myo ring, RV's boundary is messy (basal slices + stray
  voxels). Full HD is the fragile max (one stray voxel → ~200 mm); **HD95** is the robust report.
- `runs/<run>/plots/`: per-class boundary-distance **KDE** + EF **Bland–Altman** (flagship below).

![EF Bland–Altman — M&M-2 model on held-out ACDC: difference distribution + bias / 95% LoA](docs/media/ef_bland_altman.png)
![Per-class boundary-distance KDE — M&M-2 model on held-out ACDC](docs/media/boundary_kde.png)

Published column = context, not a trophy: even multi-vendor, this is "competent on public
benchmarks," not clinical-grade. M&M-2 is 3 vendors / 1.5–3T — broader than ACDC, still not the
full deployment distribution.

## Layout
```
cardioseg/
  data/mri/data.py        # ACDC loader (NIfTI, spacing-aware) + geometric LV/RV id, Info.cfg
  data/mri/mnm2.py        # M&M-2 loader (multi-vendor) + label remap to ACDC convention, vendor/disease meta
  preprocessing/preprocess.py   # resample in-plane + z-score; param-keyed disk cache
  training/
    model.py              # MONAI U-Net factory (2D/3D)
    dataset.py            # 2D-slice dataset, patient-level split (dataset-agnostic loader)
    train.py              # training loop (--dataset acdc|mnm2, --test for cross-dataset; workers+AMP)
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
