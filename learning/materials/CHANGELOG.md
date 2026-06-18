# Learning materials — changelog

## 2026-06-17 — Theory build-out (A2 + B + C + artifacts + cross-modality)
Goal: prepare the **full application-relevant theory** so the learner is ready for all of
it, before Phase D (hands-on). Research-grounded (see
`../../research/deep_dives/2026-06-17_application-curriculum-and-gaps.md`).

### Added
- **`common/`** — new cross-modality folder (CT/echo will reuse, no duplication):
  - `README.md`
  - `cardiac-anatomy-and-cycle.md` (B1)
  - `ejection-fraction.md` (B2)
  - `G_geometry-and-volumetry.md` (B3 + geometry thread)
  - `M_segmentation-theory.md` (C2 — U-Net/nnU-Net, 2D vs 3D, loss, patient-level splits)
  - `E_evaluation-theory.md` (Dice/Jaccard, Hausdorff/HD95, ASSD, Bland-Altman EF agreement,
    calibration, failure analysis, clinical-grade gap)
- **`mri/06_cardiac-mri.md`** (A2) — bSSFP cine, ECG gating (prospective/retrospective),
  segmented k-space, short-axis stack, ED/ES, anisotropy → 2D.
- **`mri/07_artifacts.md`** — artifacts as segmentation/EF failure modes, priority-ranked.
- **`mri/08_acdc-dataset.md`** (C1) — ACDC structure, files, labels, geometry, loading notes.

### Changed
- `mri/curriculum.md` — A2 ✅ + new A3 (artifacts); Phase B/C marked ✅ with links to
  `common/`; Phase D flagged as the only remaining work.
- `field-map.md` — coverage checklist ticked: MRI theory complete; cross-cutting theory
  done; CT/echo theory pending.
- `mri/README.md` — index now includes 06–08 and points to `common/`.

### Honesty flags carried into the materials
- **ACDC label encoding** (`0=bg,1=RV,2=LV-myo,3=LV-cavity`) = community convention,
  **NOT** confirmed from an official numeric source → **verify with `np.unique` on a real
  mask at EDA** (first Phase-D step). Repo `synth.py` currently uses a different mapping →
  fix at the code phase.
- Several numbers (nnU-Net per-class Dice, EF normal-range bounds, Gibbs EF impact) marked
  "reported / verify" inline.

### Status after this build
**MRI-lane theory: complete.** Remaining = **Phase D (code)**: EDA → 2D U-Net → EF vs GT →
failure analysis. Quiz coverage so far: A1 (`../quizzes/mri/A1_2026-06-17.md`, ~68%).
