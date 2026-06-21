# ROADMAP — systole

A deliberate ramp into cardiac imaging. The throughline is one bounded,
honestly-evaluated capability — **segment the heart → measure cardiac function
(ejection fraction) → show where it fails** — pursued across three modalities, in a
rhythm of *understand the data → look at the data → solve a bounded problem*.

Status tags: ✅ done · 🔄 doing · ⬜ planned.

## Where it stands (2026-06)
**MRI lane is delivered, and went past the original Gate 1.** Beyond "train on ACDC,
report Dice + EF," the model is now set up for **domain generalization**: trained on the
multi-vendor **M&M-2** set (360 subjects, 3 vendors) and tested on **held-out ACDC**
(single-centre, 100 patients it never saw).

- **Segmentation generalizes** — M&M-2 → ACDC mean Dice **0.87**, equal to the in-domain
  ACDC ceiling. The reverse (ACDC → multi-vendor) collapses to **0.70** (RV 0.85 → 0.59):
  diversity in training is what holds up. That asymmetry is the headline result.
- **EF does not (yet)** — cross-dataset EF MAE **9.4%**, bias −8.9%, LoA ±27 — *not*
  clinically usable (clinical bar ≈ ±5%). EF is a ratio of two volumes, so it amplifies
  per-frame mask error. Good masks, fragile derived number. See **EF paths** below.
- **Shipped alongside:** [cardioview](cardioview/) browser viewer (beating 3D hearts +
  in-browser ONNX segmentation), [mri-sim](mri-sim/) acquisition visualizer, surface
  metrics (HD95/ASSD) + error-distribution plots (boundary KDE, EF Bland–Altman).

Datasets on disk (`D:/data/volumetric/mri/`, out of repo): **ACDC** + **M&M-2** (in use),
**M&Ms-1** and **task02_heart** (MSD left-atrium) available but not wired.

## EF paths — from a weak number to a usable one
The cross-dataset EF is the honest weak spot; the roadmap out of it, in effort order:
1. ✅ **Postprocess masks** (largest-CC) — dropped false-positive specks: EF MAE 9.4 → 8.2%,
   bias −8.9 → −7.2%, HD RV 191 → 59 mm. Eval-only.
2. ✅ **Test-time augmentation** — in-plane flip averaging at inference, no retrain.
3. ✅ **Heavy augmentation + early stopping** (GPU-batched) — wider geometry + vendor-style
   intensity jitter; trained to a val-Dice plateau (~95 ep, best checkpoint kept). RV Dice
   0.84 → **0.89**, mean 0.87 → **0.90**, EF MAE 8.2 → **6.3%**, LoA ±27 → ±15. Now ~1 Dice pt /
   0.8 EF pts behind nnU-Net SOTA, on a deployable ONNX model.
4. ⬜ **Cross-scanner intensity harmonization** — today it's per-volume z-score only;
   vendor-aware histogram standardization may tighten the spread. (`bd cardiac-seg-qfz`)
5. ⬜ **Bias calibration** — held-out linear EF correction, reported as such.
6. ⬜ **Stronger segmentation** — nnU-Net baseline (done, see baselines/), 3D context.
7. ⬜ **Eval rigor** — 5-fold CV instead of one split (`bd cardiac-seg-4ev`); uncertainty /
   calibration flags (`bd cardiac-seg-iq7`).

## Open structural decision — the machine axis + dataset roles
The held-out test (ACDC) is **single-vendor**, so vendor/machine generalization isn't tested on
held-out data — only the cross-*centre* drop is. To test the machine axis properly we need a
**multi-vendor held-out test**. We have the data: **M&Ms-1** (375 subj, 4 vendors incl. Canon,
6 centres) is bigger and broader than M&M-2 — so it's arguably the better *training* set, with the
roles reshuffled. Undecided for now: M&Ms-1 vs M&M-2 as train; whether ACDC becomes test, or just
the cardioview demo set (single-vendor but clean). Tracked: `bd cardiac-seg-bsz`. Deferred — settle
once the EF levers land.

## How this is driven — the circuit
Field understanding drives the roadmap. Each topic runs the loop:
1. **Research** — teacher grounds it (internal + web) → `research/`.
2. **Theory** — study writeup → `learning/<date>_<topic>.md`.
3. **Quiz (on demand)** — open-form questions on that theory; answers + score logged.
4. **Sharpen** — update the relevant grid cell; set the next concrete step.

Build log = git history. Theory artifacts = `learning/`.

## The grid
Three modalities × three steps, all converging on **cardiac function (EF)** — one story, not three.

| Modality | Theory | Data viz | Problem solved |
|---|---|---|---|
| **MRI** (ACDC + M&M-2) | ✅ acquisition physics ([mri-sim](mri-sim/)), short-axis geometry | ✅ [cardioview](cardioview/) 3D viewer + held-out EF | ✅ seg LV/myo/RV → EF, **cross-dataset (DG)**; EF quality ⬜ |
| **CT** (MM-WHS)  | ⬜ HU calibration, CTA | ⬜ EDA | ⬜ whole-heart / chamber seg (`bd cardiac-seg-fzm`) |
| **echo** (CAMUS) | ⬜ ultrasound, 2D+t | ⬜ EDA | ⬜ LV seg → EF, Simpson's biplane (`bd cardiac-seg-q38`) |

Scaffold (✅): spacing-aware EF/volume math, Dice/HD95/ASSD + failure ranking, MONAI U-Net,
ONNX export (INT8, parity-gated), dataset-agnostic loader (ACDC + M&M-2, label remap),
cross-dataset train/test harness, `cardioseg/` + `cardioview/` + `mri-sim/` structure.

## The geometry thread (cross-cutting)
Computational geometry turns per-voxel labels into a clinical number, and recurs in every cell:
- voxel count → physical volume (mm³ → mL) → EF
- marching-cubes surface mesh per chamber (used live in cardioview)
- myocardial wall thickness
- **Simpson's biplane** (echo): stack-of-disks volume
- spacing / resampling to a common grid; anisotropy handling
- ED↔ES (and later cross-modality) correspondence

## Gates
- **Gate 1 — MRI presentable** ✅ — ACDC seg, Dice per structure, EF vs GT, failure case,
  repo public. *Done and exceeded (domain generalization, multi-vendor).*
- **Gate 2 — EF you could defend** 🔄 — close the cross-dataset EF gap (postproc →
  harmonization → calibration), add eval rigor (CV, uncertainty), honest clinical-grade-gap
  writeup (`bd cardiac-seg-upd`).
- **Gate 3 — CT lane** ⬜ — MM-WHS whole-heart/chamber seg, reuse the shared pipeline.
- **Gate 4 — echo lane** ⬜ — CAMUS LV seg → EF via Simpson's biplane.

## Cross-cutting threads
- **Structure** — `cardioseg/` (pipeline) + `cardioview/` (viewer) + `mri-sim/` (acquisition).
  A modality is added only when it's real — no empty speculative folders. Data mirrors
  `D:/data/volumetric/{mri,ct,echo}/`, out of the repo (licensing + size).
- **Clinical-grade gap** — the honest "hard 80%": multi-vendor robustness (now measured, not
  assumed), validation rigour, measurement precision, licensing / DICOM PII.

---
*Build log lives in the git history. Theory writeups under `learning/`; research under `research/`.*
