# cardiac-imaging — segmentation + cardiac function across MRI, CT, echo

Segmentation and **cardiac-function measurement (ejection fraction)** across imaging
modalities — **MRI first, CT and echo to follow** — with the evaluation rigor that
decides whether a measurement can be trusted. The connective tissue is the
computational geometry: how you go from per-voxel labels to a clinical number,
whatever the modality.

**Honest scope.** I come from audio / acoustic-signal ML (end-to-end modeling,
evaluation, edge). Volumetric / cardiac imaging is a deliberate ramp; this repo is
me building competence in the gap — the modeling, the geometry/measurement, and the
domain understanding — on public data, not a claim of prior medical-imaging
experience. Today only the MRI lane is underway; CT and echo are planned (see
`ROADMAP.md`), not done.

## The shape of the work
A 3×3 grid — three modalities, each taken through the same three steps — tied
together by one cross-cutting geometry thread:

| Modality | Theory | Data viz | Problem solved |
|---|---|---|---|
| **MRI** (ACDC) | acquisition physics, short-axis geometry | EDA on ACDC | seg LV/myo/RV → **EF** |
| **CT** (MM-WHS) | HU calibration, CTA | EDA | whole-heart / chamber seg |
| **echo** (CAMUS) | ultrasound, 2D+t | EDA | LV seg → **EF** (Simpson) |

All three converge on **cardiac function (EF)**, so the modalities tell one story,
not three. **Computational geometry** is the thread through every "problem solved":
voxel count → volume → EF, marching-cubes surface meshes, wall thickness, Simpson's
biplane (stack-of-disks), spacing/resampling, ED↔ES registration.

## Pipeline (per modality)
1. **Data** — modality-specific loader + normalization (e.g. ACDC short-axis cine
   MRI; NIfTI volumes with per-voxel spacing in mm).
2. **Segment** — 2D/3D U-Net (MONAI) → per-voxel labels.
3. **Measure** — chamber volumes (voxel count × voxel volume, mm³ → mL);
   **EF = (EDV − ESV) / EDV**.
4. **Evaluate** — Dice + Hausdorff per structure; **where it fails** (worst cases,
   calibration) — the part that decides clinical trust.
5. **Geometry/viz** — marching-cubes surface mesh per chamber.

## Layout
```
src/
  core/                 # shared, modality-agnostic
    model.py            #   MONAI U-Net factory
    train.py            #   training loop
    measure.py          #   volumes + ejection fraction (spacing-aware)
    evaluate.py         #   Dice / Hausdorff / failure ranking
    viz.py              #   marching-cubes mesh
  modalities/
    mri/                # CT/, echo/ added when each is real (not before)
      data.py           #   ACDC loader (NIfTI), spacing-aware
      synth.py          #   synthetic cardiac-like fixture (runs before real data)
tests/
  test_smoke.py         # end-to-end on the synthetic fixture (no real data needed)
```

## Quickstart
```bash
pip install -r requirements.txt
python -m pytest tests/ -q                 # smoke test on synthetic fixture
python -m src.core.train --synthetic       # train a few steps on synthetic data
```
Real data: register for ACDC (Creatis / humanheart-project). Data lives **outside
the repo** (licensing + size) — keep it under a modality-organised root, e.g.
`D:/data/volumetric/mri/acdc/`, and point the loader at it:
```bash
export CARDIAC_DATA_ROOT=/path/to/volumetric/mri/acdc
```
(falls back to `./data/acdc`, which is gitignored).

## Status (2026-06-17)
Early — building in the open.
- **Working:** synthetic pipeline runs end-to-end; EF/volume math verified +
  unit-tested; Dice/Hausdorff/failure-ranking implemented; smoke tests green;
  `core/` + `modalities/` structure in place.
- **In progress (MRI):** ACDC loader → Dataset/DataLoader → first 2D baseline.
- **Planned:** real Dice/EF vs GT + failure analysis (MRI), then CT (MM-WHS) and
  echo (CAMUS). 8 of the 9 grid cells are empty today — roadmap, not claim.

See `ROADMAP.md` for the full grid, goals, and dated build log.

## How it's built
Agent-driven build, human-owned judgment — the workflow I use day to day. Coding
agents scaffold the boilerplate, loaders, and API plumbing; I own the modelling
decisions, the measurement correctness, and the evaluation. The EF/volume math is
spacing-aware and unit-checked by hand; the metrics and failure ranking are the
point — a measurement is only as good as the evidence it holds up under. A synthetic
fixture (`src/modalities/mri/synth.py`) makes the whole pipeline runnable and
testable before real data, so the skeleton is verified, not hypothetical.

What transfers from my audio / acoustic-signal ML background is data-structure
reasoning (how data correlates along each axis → the right inductive bias) and
evaluation discipline. The clinical specifics — anatomy, acquisition physics, what's
clinically salient, scan artefacts — I'm learning; this repo is part of that.
