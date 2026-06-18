# systole — cardiac segmentation + function (MRI, CT, echo)

**What this is.** systole is where I pick up a new domain in the open — one bounded problem:
cardiac **segmentation → ejection fraction**.

Segmentation → **ejection fraction** across imaging modalities (**MRI now; CT, echo planned**),
with the evaluation that decides whether a measurement can be trusted. The connecting
thread is the geometry: per-voxel labels → a clinical number. Three pieces, general → specific
(each links into its folder for depth); full plan + milestones in **[ROADMAP.md](ROADMAP.md)**.

## Understand the acquisition — [mri-sim](mri-sim/)
Interactive 3D visualizer of the MRI **signal pipeline** — spins → slice select →
phase/frequency encode → k-space → reconstructed image, on one clock. Built to *understand the
acquisition* the segmentation model consumes. TypeScript + vtk.js; models the actual signal path.

![mri-sim demo](mri-sim/docs/media/demo.gif)

## See the model work — [cardioview](cardioview/)
Browser viewer (TS + vtk.js) of the model's output on held-out patients: predicted chambers (LV
cavity / myocardium / RV) as a **beating 3D heart** with **EDV / ESV / LVEF vs ground truth**
and a `held-out` tag. Or drop in your own `.nii.gz` → segmented **in-browser** (ONNX).

![cardioview — held-out heart: predicted chambers, beating](cardioview/docs/media/demo.gif)

*Held-out patient — the result, shown; the numbers below back it.*

## The pipeline + results — [cardioseg](cardioseg/)
The science layer: data → preprocess → 2D U-Net → measure (EF) → evaluate. Held-out, seed 0
(2D U-Net, patient-level 80/20 split, 20 patients across all five pathology groups):

| structure | Dice | HD95 (mm) | published ACDC |
|---|---|---|---|
| LV cavity | **0.93** | 2.1 | ~0.93–0.96 |
| LV myocardium | 0.82 | 3.0 | ~0.88–0.92 |
| RV cavity | 0.86 | 10.0 | ~0.88–0.92 |
| **mean** | **0.87** | | |

**EF vs ground truth: MAE ~3%** (bias −1.5%, 95% LoA [−8, +5]; clinical equivalence ≈ ±5%). The
published column is *context, not a trophy*: ACDC is single-centre and homogeneous → "competent
on a clean benchmark," **not** SOTA or clinical-grade; multi-vendor robustness is untested (the
hard part). **Where it fails:** RV boundary is the weak spot (HD95 10 mm), HCM EF errors largest.
Full method, training, error-distribution plots (boundary KDE + EF Bland–Altman) → **[cardioseg/](cardioseg/)**.

## Scope
I come from audio / acoustic-signal ML (modeling, evaluation, edge); cardiac imaging
is a deliberate ramp. The approach: build the problem, evaluate it, and visualize it,
with an **LLM-driven learning track** ([`learning/`](learning/): theory write-ups +
self-quizzes) running alongside — learning a field by shipping in it. Competence built in the
gap on public data, not a claim of prior medical-imaging experience; the ramp is the point, not
a disclaimer. Today only the MRI lane is underway; CT and echo are planned, not done.

## Install & test
```bash
pip install -e .
python -m pytest tests/ -q     # unit + ACDC integration (integration skips without data)
```
**Running** is per project (each differs) — see the folder READMEs: pipeline train/eval →
[cardioseg/](cardioseg/); viewers → [cardioview/](cardioview/) and [mri-sim/](mri-sim/) (`npm run dev`).

## How it's built
Agent-driven build, human-owned judgment — coding agents scaffold the plumbing; I own the
modeling decisions, the measurement correctness, and the evaluation. What transfers from audio
ML is data-structure reasoning and evaluation discipline; the clinical specifics I learn as I go
([learning/](learning/)).
