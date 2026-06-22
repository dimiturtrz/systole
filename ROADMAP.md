# ROADMAP — systole

A deliberate ramp into cardiac imaging. The throughline is one bounded,
honestly-evaluated capability — **segment the heart → measure cardiac function
(ejection fraction) → show where it fails** — pursued across three modalities, in a
rhythm of *understand the data → look at the data → solve a bounded problem*.

Status tags: ✅ done · 🔄 doing · ⬜ planned.

## Where it stands (2026-06)
**MRI lane is delivered, and went past the original Gate 1.** Beyond "train on ACDC,
report Dice + EF," the model is now set up for **domain generalization**: trained on a pooled
multi-vendor cloud (**M&M-2 + M&Ms-1**, 564 labelled subjects, 4 vendors) and held out along
**two axes** — **ACDC** (centre/protocol shift, 150 it never saw) and **Canon** (unseen vendor).

- **Segmentation generalizes** — pooled → held-out ACDC mean Dice **0.88** (ED+ES), near the in-domain
  ACDC ceiling; unseen-vendor Canon **0.84**. The reverse (single-centre ACDC → multi-vendor)
  collapses to **0.70** (RV 0.85 → 0.59): diversity in training is what holds up. That asymmetry
  is the headline result.
- **EF improved, not yet clinical** — held-out ACDC EF MAE **6.2%**, bias −5.7%, LoA ±13 — down
  from the ±27 pre-postproc start, still past the ≈±5% clinical bar. EF is a ratio of two volumes,
  so it amplifies per-frame mask error. Good masks, fragile derived number. See **EF paths** below.
- **Shipped alongside:** [cardioview](cardioview/) browser viewer (beating 3D hearts +
  in-browser ONNX segmentation), [mri-sim](mri-sim/) acquisition visualizer, surface
  metrics (HD95/ASSD) + error-distribution plots (boundary KDE, EF Bland–Altman).

Datasets on disk (out of repo, under `<data>/raw/`): **ACDC + M&M-2 + M&Ms-1** all wired into the
data cloud; **task02_heart** (MSD left-atrium) available but not wired.

## EF paths — from a weak number to a usable one
The cross-dataset EF is the honest weak spot; the roadmap out of it, in effort order:
1. ✅ **Postprocess masks** (largest-CC) — dropped false-positive specks: EF MAE 9.4 → 8.2%,
   bias −8.9 → −7.2%, HD RV 191 → 59 mm. Eval-only.
2. ✅ **Test-time augmentation** — in-plane flip averaging at inference, no retrain.
3. ✅ **Heavy augmentation + early stopping + multi-source pooling** (GPU-batched) — wider geometry +
   vendor-style intensity jitter; trained to a val-Dice plateau (~95 ep, best checkpoint kept); pooled
   M&M-2 + M&Ms-1. RV Dice 0.84 → **0.88**, mean 0.87 → **0.88** (ED+ES), EF MAE 8.2 → **6.2%**, LoA ±27 → ±13.
   **~2–3 Dice points under the nnU-Net floor** (0.88 vs 0.912); EF roughly level (6.2 vs 5.6%); on a
   deployable ONNX model at ~57× fewer params.
4. ⬜ **Cross-scanner intensity harmonization** — today it's per-volume z-score only; vendor-aware
   histogram standardization may tighten the spread. **Deprioritized by evidence:** in-domain M&M-2
   vendors are level (val split, ED+ES: Siemens/Philips 0.867, GE 0.880) — **no minority-vendor
   deficit** for harmonization to close. So the cheap robustness lever is more multi-vendor data
   (already in); harmonization is a smaller, unproven win here. (`bd cardiac-seg-qfz`, low priority)
5. ⬜ **Bias calibration** — held-out linear EF correction, reported as such.
6. ⬜ **Stronger segmentation + SOTA benchmark** — nnU-Net baseline (done, see baselines/), 3D
   context; beyond the nnU-Net floor, benchmark **CardioSAM** (current OSS SOTA) on our cross-vendor
   split — cite + run cross-vendor only, since its published numbers are in-distribution (`bd cardiac-seg-0h7`).
7. ⬜ **Eval rigor** — 5-fold CV instead of one split (`bd cardiac-seg-4ev`); uncertainty /
   calibration flags (`bd cardiac-seg-iq7`).

## Resolved (thinly): the machine axis is now tested
Earlier the held-out test was single-vendor ACDC — only the cross-*centre* drop was measured. **Now
the split holds out two axes** (criteria over the data cloud): ACDC (centre/protocol shift) **and
Canon** (a scanner vendor never in training). **M&Ms-1** (320 on disk / 213 labelled, 4 vendors incl.
Canon) is pooled into training, so the flagship trains on 4 vendors. The machine axis *is* tested —
but Canon is **n=9 labelled** (M&Ms-1 withholds most Testing GT): enough for a Dice signal (~0.84),
too thin for EF. **Still open:** leave-one-vendor-out (n up to ~190 for GE/Philips) for proper
unseen-vendor stats. Tracked: `bd cardiac-seg-bsz`.

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
| **MRI** (ACDC + M&M-2 + M&Ms-1) | ✅ acquisition physics ([mri-sim](mri-sim/)), short-axis geometry | ✅ [cardioview](cardioview/) 3D viewer + held-out EF | ✅ seg LV/myo/RV → EF, **cross-dataset (DG)**; EF quality ⬜ |
| **CT** (MM-WHS)  | ⬜ HU calibration, CTA | ⬜ EDA | ⬜ whole-heart / chamber seg (`bd cardiac-seg-fzm`) |
| **echo** (CAMUS) | ⬜ ultrasound, 2D+t | ⬜ EDA | ⬜ LV seg → EF, Simpson's biplane (`bd cardiac-seg-q38`) |

Scaffold (✅): spacing-aware EF/volume math, Dice/HD95/ASSD + failure ranking, MONAI U-Net,
ONNX export (INT8, parity-gated), dataset-agnostic loader (ACDC + M&M-2 + M&Ms-1, label remap),
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
  A modality is added only when it's real — no empty speculative folders. Data lives out of the
  repo under `<data>/raw/` (licensing + size).
- **Clinical-grade gap** — the honest "hard 80%": multi-vendor robustness (now measured, not
  assumed), validation rigour, measurement precision, licensing / DICOM PII.

---
*Build log lives in the git history. Theory writeups under `learning/`; research under `research/`.*
