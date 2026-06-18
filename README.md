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

**Why train on M&M-2 instead of ACDC?** Because diversity in training buys robustness — and the
flip proves it. Train single-centre and test across vendors → it *collapses*; train multi-vendor
and test single-centre → it *holds*:

| train → test | mean Dice | RV | EF MAE |
|---|---|---|---|
| ACDC → ACDC (in-domain) | 0.87 | 0.85 | 4.7% |
| ACDC → M&M-2 (out-of-distribution) | **0.70** | 0.59 | 9.1% |
| **M&M-2 → ACDC (generalization, flagship)** | **0.87** | 0.84 | 9.4% |

The single-centre model drops ~17 Dice points off its home dataset (RV worst, 0.85 → 0.59); the
multi-vendor model carries to a new centre with no segmentation drop. **Where it still fails:**
RV is the weak structure in every setting; EF transfers worse than Dice (calibration). Surface
metrics (HD95 / ASSD) + error-distribution plots (boundary KDE + EF Bland–Altman) → **[cardioseg/](cardioseg/)**.

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
