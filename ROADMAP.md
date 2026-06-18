# ROADMAP — systole

A deliberate ramp into cardiac imaging. The throughline is one bounded,
honestly-evaluated capability — **segment the heart → measure cardiac function
(ejection fraction) → show where it fails** — pursued across three modalities, in a
rhythm of *understand the data → look at the data → solve a bounded problem*.

Status tags: ✅ done · 🔄 doing · ⬜ planned. Each cell is meant to leave the repo
in a presentable, honest snapshot (see README *Status*).

## How this is driven — the circuit
Field understanding drives the roadmap (I can't fully spec a deliverable before I
understand the field, so understanding sharpens the plan bottom-up). Each topic runs
the loop:

1. **Research** — teacher grounds it (internal + web) → `research/` (tracked; cited
   raw material behind the writeups).
2. **Theory** — study writeup → `learning/<date>_<topic>.md`.
3. **Quiz (on demand)** — when I ask, open-form questions on that theory; my answers
   + honest assessment + score logged into the topic file.
4. **Sharpen** — update the relevant grid cell; set the next concrete step.

Build log = git history. Theory artifacts = `learning/`. The grid below is the
current best understanding of the plan, refined each cycle.

## The grid
Three modalities × three steps. All three converge on **cardiac function (EF)**, so
they tell one story, not three. Public datasets exist for every cell — each "problem
solved" is real, not synthetic.

| Modality | Theory | Data viz | Problem solved |
|---|---|---|---|
| **MRI** (ACDC)   | 🔄 acquisition physics, short-axis geometry | ⬜ EDA on ACDC | ⬜ seg LV/myo/RV → EF |
| **CT** (MM-WHS)  | ⬜ HU calibration, CTA | ⬜ EDA | ⬜ whole-heart / chamber seg |
| **echo** (CAMUS) | ⬜ ultrasound, 2D+t | ⬜ EDA | ⬜ LV seg → EF (Simpson) |

Scaffold (✅): synthetic pipeline, spacing-aware EF/volume math,
Dice/Hausdorff/failure ranking, MONAI U-Net, `core/` + `modalities/` structure.

## The geometry thread (cross-cutting theory, not a 10th task)
Computational geometry is what turns per-voxel labels into a clinical number, and it
recurs in every "problem solved" cell:
- voxel count → physical volume (mm³ → mL) → EF
- marching-cubes surface mesh per chamber; surface area
- myocardial wall thickness
- **Simpson's biplane** (echo): stack-of-disks volume — pure geometry
- spacing / resampling to a common grid; anisotropy handling
- ED↔ES registration (and, later, cross-modality)

Each modality's "problem solved" is where a slice of this thread gets exercised and
written up.

## Working rhythm
Per modality the work alternates deliberately:
**theory (incl. recording physics + geometry) → data visualization → solve the
bounded problem → evaluate honestly.** Theory earns the right to model; the
visualization keeps the theory honest against the actual data; the evaluation
(Dice/Hausdorff + failure analysis + measured-vs-GT function) is the point.

## Cross-cutting threads
- **Modalities** — code is `core/` (shared) + `modalities/<m>/` (per-modality data +
  preprocess). A modality folder is added **only when that modality is real** — no
  empty speculative folders. Mirrors `D:/data/volumetric/{mri,ct,echo}/`.
- **Compliance / clinical-grade gap** — data licensing (ACDC/MM-WHS/CAMUS
  redistribution terms, DICOM PII → data stays out of the repo), plus an honest
  writeup of what separates these demos from clinical use (multi-scanner/vendor
  robustness, validation, measurement precision). The "hard 80%".

## Order of attack
1. **MRI lane end-to-end first** — theory → ACDC EDA → 2D U-Net → EF vs GT →
   failure analysis. This is Gate 1 (the application-link milestone).
2. **CT lane** (MM-WHS) — reuse `core/`, add `modalities/ct/`.
3. **echo lane** (CAMUS) — add `modalities/echo/`; Simpson's-biplane geometry.
4. Deepen: 3D where it beats 2D, calibration, proper VTK render, clinical-grade gap writeup.

## Gate 1 — presentable / first public push (MRI only)
- ACDC training runs, reasonable Dice per structure on a patient-split val set.
- **EF computed + compared to GT** on the val set.
- Dice per structure + ≥1 failure case shown in README.
- Pre-public review (secrets / data / honesty). Flip repo public.

---
*Build log lives in the git history. Diary / research / per-modality theory
writeups will live under `notes/` once the first one is written.*
