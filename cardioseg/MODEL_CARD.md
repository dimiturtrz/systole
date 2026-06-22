# Model Card — cardioseg 2D U-Net (generalization split)

A 2D short-axis cardiac MRI segmenter (LV cavity, LV myocardium, RV cavity) trained for **domain
generalization** across scanner vendors and centres, with ejection fraction derived from the masks.
This card follows Mitchell et al. 2019; the failure-mode and limitation sections are the point.

## Model details
- **Architecture:** MONAI 2D U-Net, 4-class (bg / RV / LV-myo / LV-cav), residual units. Shape from
  `ModelCfg` (`cardioseg/hparams.py`): channels (16,32,64,128,256), strides (2,2,2,2), 2 res-units.
- **Task:** per-slice short-axis segmentation → chamber volumes → EF = (EDV − ESV) / EDV.
- **Training recipe:** Dice+CE loss, Adam (lr 1e-3), GPU-batched augmentation (flip/rotate/scale +
  intensity gamma/contrast/blur/noise), AMP, early stopping (patience 20, 128-epoch ceiling,
  best-val checkpoint), largest-connected-component postprocessing, test-time augmentation (4 flips).
- **Provenance:** every run serializes its full config to `runs/<run>/config.json` and embeds it in
  `metrics.json`. Seed 0. Note `cudnn.benchmark=True` → runs are not bit-identical across machines.
- **Reproduce:** `python -m cardioseg.training.train --out runs/gen` (default split holds out ACDC + Canon; ~6 min, RTX-class GPU).

## Intended use
- **In scope:** research / methods demonstration of cross-vendor cardiac MRI segmentation and EF
  estimation on public benchmark data (ACDC, M&M-1, M&M-2).
- **Out of scope:** **not a medical device, not clinical-grade, not for diagnosis or patient care.**
  Trained/evaluated only on public challenge data — a narrower distribution than real deployment
  (limited vendors/centres/pathologies, curated acquisition). Do not use on non-short-axis views,
  non-cine sequences, or other anatomy.

## Training data
Pooled multi-source "data cloud" (`data/store.py`), **451 labelled subjects**:
- **M&M-2** (Siemens / Philips / GE, 1.5T+3T) + **M&Ms-1 excluding Canon** (Siemens / Philips / GE).
- ACDC and the Canon vendor are **held out entirely** (never in train/val).
- Splits are our own (a polars query, `data/splits.py`) — challenge splits are not inherited.
- Label conventions remapped to canonical per dataset (verified geometrically); M&Ms-1 withheld-GT
  cases (zero-filled masks) are flagged and excluded.

## Evaluation — two-axis generalization split
One model, held out along two independent shift axes (DataCfg criteria: `test_datasets=('acdc',)`,
`test_vendors=('Canon',)`):

| held-out axis | n | what it isolates |
|---|---|---|
| **ACDC** | 150 | centre / protocol shift (single-vendor Siemens, fully independent) |
| **Canon** (M&Ms-1) | 9 | **unseen vendor** (a scanner brand never in training) |

## Performance

**ACDC-150 (centre/protocol shift):**

| structure | Dice | HD95 (mm) | ASSD (mm) |
|---|---|---|---|
| LV cavity | 0.94 | 1.5 | 0.27 |
| LV myocardium | 0.86 | 2.1 | 0.46 |
| RV cavity | 0.91 | 3.0 | 0.49 |
| **mean** | **0.90** | | |

EF vs GT: **MAE 7.1%**, bias −6.6% (n=150).

**Canon-9 (unseen vendor):** mean Dice **0.85**. EF MAE is **not reported as a stable number** — at
n=9 it swings (7.2% under a M&M-2-only model, 15.4% here) while Dice holds ~0.85. Canon is the honest
*Dice* signal for unseen-vendor robustness; its EF is noise (the M&Ms-1 challenge withholds GT for
most of its Testing split — 320 on disk, 213 labelled, Canon 50 → 9 usable).

### Benchmark (nnU-Net, same split)
nnU-Net trained on the **identical generalization split** (Dataset029_BATTERY, 50 epochs / 1 fold —
a floor), scored by the same eval layer:

| ACDC-150 | nnU-Net (50ep/fold0) | this model |
|---|---|---|
| mean Dice | **0.912** | 0.901 |
| EF MAE / bias | **5.6% / −4.2%** | 7.1% / −6.6% |
| Canon-9 mean Dice | 0.876 | 0.853 |

nnU-Net is ~**1 Dice point** ahead with **better EF** (smaller ES-over-seg bias) — *even at a floor
setting*; the full ceiling (1000ep × 5-fold + TTA, `cardiac-seg-yp3`) would be higher. Honest position:
this 2D U-Net is **competent, ~1 point under nnU-Net's floor** on the same data. (One inversion: our
boundary is tighter — LV-cav HD95 1.5 vs 3.5mm — from largest-CC + TTA, which this nnU-Net run lacked.)

## Where it fails (stratified)
- **HCM / small cavities — the main EF failure mode.** EF bias is **entirely end-systolic cavity
  over-segmentation** (~fixed absolute mL of boundary + papillary voxels), so its *fractional* impact
  scales inversely with cavity size: corr(ES cavity, ESV ratio) = −0.50. Dilated (huge cavity) is EF-
  unbiased (≈ −0.5%); hypertrophic (tiny cavity) is worst (EF MAE ~13.8%). Dice stays flat (~0.90)
  across pathologies — masks aren't worse, the EF *ratio* is range-dependent. (See
  `research/deep_dives/2026-06-21_ef-bias-mechanism-esv-overseg.md`.)
- **RV boundary** is the loosest (HD95 3.0 mm vs myo 2.1, LV-cav 1.5) — basal slices + stray voxels.
- **Vendor:** the GE minority-vendor Dice deficit (0.879 under M&M-2-only) **closed** under pooled
  training (0.903) — vendor diversity, not harmonization, fixed it.

## Limitations & caveats
- Single modality (cine MRI short-axis), 2D slice model — no long-axis, no 3D context.
- Systematic EF under-prediction (bias ≈ −6%) from ES over-segmentation; **not** corrected by a
  constant offset (size-dependent → would over-correct dilated hearts) and **not** a measurement bug
  (volumes are computed correctly). The honest fix is segmentation-side at ES (`cardiac-seg-7oe`).
- Unseen-vendor evidence is thin (Canon n=9); Dice is the only readable metric there.
- Public-benchmark performance ≠ deployment performance. "Competent on public benchmarks," not clinical.

## References
- Datasets: ACDC (Bernard 2018), M&Ms-1 (Campello 2021 TMI), M&M-2.
- Method/results detail: `cardioseg/README.md`. Config/provenance: `runs/<run>/config.json`.
