# ROADMAP — systole

A deliberate ramp into cardiac imaging. The throughline is one bounded,
honestly-evaluated capability — **segment the heart → measure cardiac function
(ejection fraction) → show where it fails** — pursued across three modalities, in a
rhythm of *understand the data → look at the data → solve a bounded problem*.

Status tags: ✅ done · 🔄 doing · ⬜ planned.

## Where it stands (2026-06)
**MRI lane is delivered, and went past the original Gate 1.** Beyond "train on ACDC,
report Dice + EF," the model is now set up for **domain generalization**: trained on
Siemens+Philips from M&M-2+M&Ms-1 (**495 labelled subjects**, 2 vendors); val = ACDC-150
(centre/protocol shift); test = Canon+GE (unseen vendors, held out entirely).

- **Segmentation generalizes** — val ACDC mean Dice **0.88** (ED+ES), near the in-domain ACDC
  ceiling; unseen-vendor Canon+GE both **0.839** (n=78 combined — two independent scanners agree).
  The reverse (single-centre ACDC → multi-vendor) collapses to **0.70** (RV 0.85 → 0.59): diversity
  in training is what holds up. That asymmetry is the headline result.
- **EF improved, not yet clinical** — val ACDC EF MAE **6.5%**, bias −5.6%, LoA [−20.1, +8.9] — down
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
   M&M-2 + M&Ms-1 (Siemens+Philips, 495 subjects). RV Dice 0.84 → **0.88**, mean 0.87 → **0.88**
   (ED+ES), EF MAE 8.2 → **6.5%**, LoA ±27 → [−20.1, +8.9].
   **~2–3 Dice points under the nnU-Net floor** (0.88 vs 0.912); EF roughly level (6.5 vs 5.6%); on a
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

## Augmentation + synthetic data — cross-vendor robustness, three tiers (planned)
<!-- Tier 1 = augmentation (perturb real images); Tiers 2-3 = synthetic generation (invent from labels) — distinct families. -->
A distinctive angle for the cross-vendor lane: don't just train on real images — perturb or regenerate
them to widen the contrast space the model sees, then measure the cross-vendor robustness gain vs the
no-aug baseline. The differentiation is the **honest cross-vendor eval**, not the generator itself
(`bd cardiac-seg-chm`). The "synthetic" axis is a spectrum, not one thing — three tiers, increasing in
how much of the *picture* is generated (anatomy is always real ACDC; none invents new hearts):

```
real image  ──perturb──>           Tier 1: augmentation   (TorchIO)
real labels ──paint random──>      Tier 2: synth-appearance (SynthSeg pattern)
real labels ──simulate physics──>  Tier 3: synth-physics  (CMRsim / Bloch)
```

- **Tier 1 — augmentation of real images.** Real image stays; perturb it (bias field, Rician noise,
  k-space ghosting/spike, gamma/contrast shift) so it looks like another scanner. Anatomy *and* pixels
  real, roughened. Library: **TorchIO** (`RandomBiasField/Ghosting/Spike/Motion/Noise` + histogram
  matching) — extends `augment.py`, no sim build. **1–2 days.** Highest evidence: the M&Ms challenge
  credits intensity-driven aug + histogram matching for the vendor gap (Zeng et al. 2021 hit 0.905 LV
  Dice on an unseen vendor). Lowest effort, proven payoff.
- **Tier 2 — synthetic appearance from labels (SynthSeg pattern).** Discard the real image's
  intensities; keep only the **label map**; paint each region a randomly-sampled brightness each epoch →
  a fully invented picture over real anatomy. Trains a **contrast-agnostic** model (generalize to *any*
  scanner, not mimic one). No real pixels, no T1/T2 maps, no Bloch. Library: **SynthSeg** (Billot 2023,
  cardiac+CT-extended, FreeSurfer OSS); DRIFTS (2024) shows +5 DSC from per-label intensity clustering.
  **3–5 days.** The distinctive full-synth angle.
- **Tier 3 — synthetic physics from labels (Bloch sim).** Same label→image route as Tier 2, but assign
  per-tissue T1/T2/PD (literature values: myo T1≈1000 ms@1.5 T, blood≈1600, fat≈250) and *simulate*
  vendor-like signal under varied TR/TE/flip. Physically grounded. Library: **CMRsim** (Weine 2024, ETH,
  *MRM*) — the only cardiac-purpose-built Python Bloch sim; MRzero/FaBiAN do this for brain. **2–4 weeks**,
  no published cardiac-seg-aug precedent, and may still miss vendor recon-pipeline effects (k-space
  filtering, SENSE/GRAPPA). The maximalist showcase, not the cheap robustness win.

**Recommended order:** Tier 1 first (measure Canon Dice delta) → Tier 2 if more contrast diversity is
needed → Tier 3 only with a specific hypothesis that k-space/physics, not intensity distribution, is the
failure mode. **All three are wanted** as a synthetic-data story; effort/payoff just differs.

**Sim libs evaluated** (cite, don't blind-reinvent): TorchIO/MONAI (Tier 1), SynthSeg (Tier 2),
CMRsim / MRzero / FaBiAN (Tier 3 Python Bloch); KomaMRI (Julia — wrong ecosystem for a Python loop),
JEMRIS (C++ — very high integration cost), MRiLab (MATLAB — dead since 2017). Full evaluation:
`research/deep_dives/2026-06-24_mri-sim-libs-eval.md`. Earlier landscape survey:
`research/deep_dives/2026-06-22_cardiac-segmentation-oss-landscape-unified.md`.

## Resolved: the machine axis is now tested (n=78, two vendors agree)
Earlier the held-out test was single-vendor ACDC — only the cross-*centre* drop was measured. **Now
the split holds out unseen vendors entirely:** Canon (n=9 labelled, M&Ms-1 withholds most Testing GT)
and GE (n=69) are both excluded from training and scored independently. Both return Dice **0.839** and
EF MAE **~11–12%** — independent agreement at n=78 makes this a robust unseen-vendor signal. Training
narrows to Siemens+Philips (495 subjects); GE moved from train to test. **Still open:**
leave-one-vendor-out CV for proper confidence intervals. Tracked: `bd cardiac-seg-bsz`.

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
