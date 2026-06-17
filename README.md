# cardiac-seg — cardiac MRI segmentation + ejection-fraction measurement

A focused project to ramp into volumetric medical imaging: segment the cardiac
chambers from short-axis cine MRI (ACDC), then derive clinical measurements
(chamber volumes, **ejection fraction**) from the segmentation — with the
evaluation rigor that decides whether a measurement can be trusted.

**Honest scope.** I come from audio / acoustic-signal ML (end-to-end modeling,
evaluation, edge). Volumetric medical imaging is a deliberate ramp; this repo is
me building competence in the gap — the modeling, the geometry/measurement, and
the domain understanding — on public data, not a claim of prior medical-imaging
experience.

## Pipeline
1. **Data** — ACDC short-axis cine MRI (LV blood pool, RV, myocardium; ED + ES
   frames). NIfTI volumes with per-voxel spacing (mm).
2. **Segment** — 2D/3D U-Net (MONAI) → per-voxel labels.
3. **Measure** — chamber volumes (voxel count × voxel volume, mm³ → mL);
   **EF = (EDV − ESV) / EDV** from end-diastolic vs end-systolic volume.
4. **Evaluate** — Dice + Hausdorff per structure; **where it fails** (worst
   cases, calibration of confidence) — the part that decides clinical trust.
5. **Geometry/viz** — marching-cubes surface mesh (VTK) per chamber.

## Layout
```
src/
  synth.py      # synthetic cardiac-like volumes + masks (runs before real data)
  data.py       # ACDC loader (NIfTI), spacing-aware
  model.py      # MONAI U-Net factory
  train.py      # training loop
  measure.py    # volumes + ejection fraction (spacing-aware)
  evaluate.py   # Dice / Hausdorff / failure ranking
  viz.py        # marching-cubes mesh (VTK)
tests/
  test_smoke.py # end-to-end on the synthetic fixture (no real data needed)
```

## Quickstart
```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # smoke test on synthetic fixture
python -m src.train --synthetic     # train a few steps on synthetic data
```
Real data: register for ACDC (Creatis / humanheart-project). Data lives **outside
the repo** (licensing + size) — keep it under a modality-organised root, e.g.
`D:/data/volumetric/mri/acdc/`, and point the loader at it:
```bash
export CARDIAC_DATA_ROOT=/path/to/volumetric/mri/acdc   # or set in your env
```
(falls back to `./data/acdc`, which is gitignored).

## Status (2026-06-17)
Early — building in the open.
- **Working:** synthetic pipeline runs end-to-end; EF/volume math verified +
  unit-tested; Dice/Hausdorff/failure-ranking implemented; smoke tests green.
- **In progress:** ACDC real-data loader → Dataset/DataLoader → first 2D baseline.
- **Planned:** real Dice/EF numbers vs GT, failure analysis, viz; later CT/echo.

See `ROADMAP.md` for goals, sequence, and the dated build log.

## How it's built
Agent-driven build, human-owned judgment — the workflow I use day to day. Coding
agents scaffold the boilerplate, loaders, and API plumbing; I own the modelling
decisions, the measurement correctness, and the evaluation. The EF/volume math is
spacing-aware and unit-checked by hand; the metrics and failure ranking are the
point — a measurement is only as good as the evidence it holds up under. A
synthetic fixture (`src/synth.py`) makes the whole pipeline runnable and testable
before real data, so the skeleton is verified, not hypothetical.

What transfers from my audio / acoustic-signal ML background is data-structure
reasoning (how data correlates along each axis → the right inductive bias) and
evaluation discipline. The clinical specifics — anatomy, acquisition physics,
what's clinically salient, scan artefacts — I'm learning; this repo is part of that.
