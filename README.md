# systole — cardiac segmentation + function (MRI, CT, echo)

**What this is.** systole picks up a new domain in the open — one bounded problem: cardiac
**segmentation → ejection fraction**, with the evaluation that decides whether a measurement can
be trusted. The connecting thread is geometry: per-voxel labels → a clinical number. Built on
public data, learning the field by shipping in it — an **LLM-driven learning track**
([`learning/`](learning/): theory write-ups + self-quizzes) runs alongside the code. Across
modalities eventually (**MRI now; CT, echo planned**); today only the MRI lane is underway.

Three pieces, general → specific (each links into its folder for depth); full plan + milestones
in **[ROADMAP.md](ROADMAP.md)**.

## Understand the acquisition — [mri-sim](mri-sim/)
Interactive 3D visualizer of the MRI **signal pipeline** — spins → slice select →
phase/frequency encode → k-space → reconstructed image, on one clock. Built to *understand the
acquisition* the segmentation model consumes. TypeScript + vtk.js; models the actual signal path.

![mri-sim demo](mri-sim/docs/media/demo.gif)

## See the model work — [cardioview](cardioview/)
Browser viewer (TS + vtk.js) of the flagship model's output on ACDC patients it never trained on
(it learned on M&M-2): predicted chambers (LV cavity / myocardium / RV) as a **beating 3D heart**
with **EDV / ESV / LVEF vs ground truth**. Or drop in your own `.nii.gz` → segmented **in-browser** (ONNX).

![cardioview — held-out heart: predicted chambers, beating](cardioview/docs/media/demo.gif)

*A heart from a different dataset than the model trained on — the result, shown; the numbers below back it.*

## The pipeline + results — [cardioseg](cardioseg/)
The science layer: data → preprocess → 2D U-Net → measure (EF) → evaluate. The flagship model
is set up for **domain generalization**: trained on the multi-vendor **M&M-2** challenge set
(360 subjects, 3 scanner vendors, 8 pathologies, 1.5T + 3T) and tested on **held-out ACDC**
(single-centre, 100 patients it never saw). Seed 0, patient-level splits.

**M&M-2 → ACDC** (train multi-vendor, test held-out single-centre):

| structure | Dice | published ACDC |
|---|---|---|
| LV cavity | **0.93** | ~0.93–0.96 |
| LV myocardium | 0.84 | ~0.88–0.92 |
| RV cavity | 0.84 | ~0.88–0.92 |
| **mean** | **0.87** | |

**EF vs ground truth: MAE 9.4%** (cross-dataset; volume calibration shifts across centres).

**Why train on M&M-2, not ACDC?** Diversity in training buys robustness, and it's asymmetric: a
single-centre (ACDC) model drops ~17 Dice points tested across vendors (mean 0.87 → 0.70, RV
0.85 → 0.59), while the multi-vendor model carries to a new centre with no segmentation drop.
**Where it still fails:** RV is the weak structure everywhere; EF transfers worse than Dice
(calibration). Full per-direction table + surface metrics (HD95 / ASSD) + error-distribution
plots (boundary KDE + EF Bland–Altman) → **[cardioseg/](cardioseg/)**.

## Data
Datasets live **outside the repo** (licensing + size) and **none is committed**.
- **M&M-2** ([challenge](https://www.ub.edu/mnms-2/)) — multi-vendor training set: 360 subjects,
  3 vendors, 8 pathologies, 1.5T + 3T.
- **ACDC** ([challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/)) — single-centre
  held-out test set: 100 patients, 5 pathologies.

Both are register-gated; label conventions differ (M&M-2 LV=1 vs ACDC LV=3) and are remapped on load.

**Config — one global file.** Every project (cardioseg pipeline + cardioview viewer) reads paths
from a single gitignored `paths.yaml` at the repo root. Copy the template and edit:
```bash
cp paths.example.yaml paths.yaml
```
```yaml
data:
  raw: D:/data/.../mri/acdc        # ACDC root (the dir holding training/)
  processed: D:/data/.../mri/acdc  # preprocess cache (npz; auto-created)
cardioview:
  hearts:                          # canned demo hearts — full paths or bare IDs
    - .../acdc/training/patient073
```
M&M-2 is auto-discovered as a sibling of the ACDC root (or set `CARDIAC_MNM2_ROOT`). Env vars
`CARDIAC_DATA_ROOT` / `CARDIAC_PROCESSED_ROOT` override the file (handy for CI). Loaded by
`cardioseg/config.py` (OmegaConf).

## Tests
```bash
pip install -e .
python -m pytest tests/ -q     # unit + ACDC integration (integration skips without data)
```
Two layers, wide base to thin top (full rationale in [tests/README.md](tests/README.md)):
- **Unit — equivalence-class.** Partition each input space by behaviour; test one representative
  per class plus its boundaries, not exhaustive cases (e.g. `fit_square`: larger-than-target →
  crops, smaller → pads, equal → identity).
- **Integration — module pairs.** If A and B share a pipeline, test the A→B chain on the same
  inputs/outputs the units promise — A's output is a valid B input, the chain keeps each unit's
  guarantee. Catches interface drift (shape/dtype/label/spacing) that per-unit tests miss.

**Running** the projects is separate (each differs) — see the folder READMEs: pipeline train/eval →
[cardioseg/](cardioseg/); viewers → [cardioview/](cardioview/) and [mri-sim/](mri-sim/) (`npm run dev`).

## How it's built
Agent-driven build, human-owned judgment — coding agents scaffold the plumbing; I own the
modeling decisions, the measurement correctness, and the evaluation. Data-structure reasoning and
evaluation discipline carry over from prior ML work; the clinical specifics I learn as I go
([learning/](learning/)).
