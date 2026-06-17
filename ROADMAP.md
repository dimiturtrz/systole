# ROADMAP — cardiac-seg

A deliberate ramp into volumetric medical imaging. The throughline is one bounded,
honestly-evaluated task — **segment cardiac chambers → measure ejection fraction →
show where it fails** — built up in a rhythm of *understand the data → look at the
data → model with a goal*, not a moonshot.

Status tags: ✅ done · 🔄 doing · ⬜ planned. Each phase is meant to leave the repo in
a presentable, honest snapshot (see README *Status*).

## How to read this
The work alternates deliberately:
**recording theory → data visualization → modelling theory → data visualization →
models with a goal**, looping as scope grows. Theory earns the right to model; the
visualization keeps the theory honest against the actual data.

## Cross-cutting threads
These run through every phase rather than being one-off steps:
- **Recording theory** — how the data is physically acquired (MRI cine, k-space
  basics, short-axis stack geometry, why MR intensity is uncalibrated). Drives the
  right preprocessing + inductive bias.
- **Data exploration / viz** — shapes, spacing, intensity distributions, class
  balance, ED/ES framing, label overlays, montages. The reality check.
- **Example problems** — small bounded tasks with a result (segment one structure,
  compute one EF, rank the worst N cases) rather than open-ended poking.
- **Modalities** — MRI first (ACDC). CT / echo are later siblings; code splits into
  a shared `core/` + per-modality adapters **only when the 2nd modality lands**
  (no empty speculative folders before then). Mirrors `D:/data/volumetric/{mri,ct,echo}/`.
- **Compliance / clinical-grade gap** — data licensing (ACDC redistribution terms,
  DICOM PII), and an honest writeup of what separates this demo from clinical use
  (multi-scanner/vendor robustness, validation, measurement precision). The "hard 80%".

## Phases

### Phase 0 — Scaffold ✅
Synthetic pipeline runs end-to-end; EF/volume math + Dice/Hausdorff/failure ranking
implemented and unit-tested; MONAI U-Net factory; synthetic training loop. Verified
by hand (EF units, self-Dice). → initial commit.

### Phase 1 — Recording theory + data acquisition 🔄
- Write up MRI cine acquisition basics + ACDC structure (ED/ES via `Info.cfg`,
  4 labels, short-axis anisotropy) in notes.
- Acquire ACDC (Creatis registration). Stage under `D:/data/volumetric/mri/acdc/`.
- Compliance note: licensing, why data stays out of the repo.

### Phase 2 — Data exploration / viz ⬜
- EDA on real ACDC: volume shapes, voxel spacing spread, per-volume intensity
  distributions, class balance per structure, ED vs ES LV size.
- Visuals: slice montages, GT label overlays, a spacing/intensity summary figure.
- Decide normalization (per-volume z-score) + target resampling spacing **from the
  data**, not by assumption.

### Phase 3 — Modelling theory ⬜
- 2D (slice-wise) vs 3D given anisotropic short-axis stacks — why 2D is the strong
  cheap baseline first.
- Loss (DiceCE), patient-level train/val split (no leakage), augmentation choices.

### Phase 4 — Models with a goal ⬜
- Train 2D U-Net on ACDC. **Goal: reasonable Dice per structure on a val split.**
- Track per-structure Dice; save the worst cases.

### Phase 5 — Measure + evaluate (the point) ⬜
- Predicted masks → LV volumes at ED/ES → **EF; compare predicted EF vs GT EF**
  (the clinical-relevance check).
- Hausdorff per structure; `rank_failures` worst-first → look at **why** they fail.
- Optional: confidence calibration.

### Phase 6 — Polish + ship (Gate 1) ⬜
- Results table (Dice per structure, EF MAE vs GT) + ≥1 failure figure in README.
- Fill `PROCESS.md` log. Pre-public review (secrets/data/honesty). Push public.

## Later (Gate 2+)
- 3D model if it beats 2D. Deeper failure analysis + calibration.
- Proper VTK render / screenshot for the README.
- 2nd modality (CT or echo) → introduce `core/` + modality adapters.
- Clinical-grade gap writeup (compliance thread).

## Log
- **2026-06-17** — Scaffold + initial commit. Agent-built skeleton: synthetic
  fixture, spacing-aware EF/volume math, Dice/Hausdorff/failure ranking, MONAI
  U-Net factory, synthetic training loop, ACDC NIfTI loader stub. Verified by
  hand: EF units (EDV > ESV > 0, EF in 0–100%) and self-Dice = 1.0 via smoke
  tests (3 passed). Data root made env-configurable (`CARDIAC_DATA_ROOT`) so
  patient data stays outside the repo. Docs consolidated into README + ROADMAP.
