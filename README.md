# cardiac-imaging — segmentation + cardiac function across MRI, CT, echo

Segmentation and **cardiac-function measurement (ejection fraction)** across imaging
modalities — **MRI first, CT and echo to follow** — with the evaluation rigor that
decides whether a measurement can be trusted. The connective tissue is the
computational geometry: how you go from per-voxel labels to a clinical number,
whatever the modality.

The three modalities converge on one capability (cardiac function), so they tell a
single story rather than three disjoint demos. For the full plan — the 3×3 grid
(modality × theory / data-viz / solved-problem), the geometry thread, and the
milestones — see **[ROADMAP.md](ROADMAP.md)**. The dated build log is the git
history.

**Honest scope.** I come from audio / acoustic-signal ML (end-to-end modeling,
evaluation, edge). Cardiac imaging is a deliberate ramp; this repo is me building
competence in the gap — the modeling, the geometry/measurement, and the domain
understanding — on public data, not a claim of prior medical-imaging experience.
Today only the MRI lane is underway; CT and echo are planned, not done.

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

## How it's built
Agent-driven build, human-owned judgment — the workflow I use day to day. Coding
agents scaffold the boilerplate, loaders, and API plumbing; I own the modelling
decisions, the measurement correctness, and the evaluation. The EF/volume math is
spacing-aware and unit-checked by hand; the metrics and failure ranking are the
point. What transfers from my audio / acoustic-signal ML background is data-structure
reasoning (how data correlates along each axis → the right inductive bias) and
evaluation discipline; the clinical specifics I'm learning as I go.
