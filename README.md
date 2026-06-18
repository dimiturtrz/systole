# systole — cardiac segmentation + function (MRI, CT, echo)

**What this is.** systole is where I pick up a new domain in the open — one bounded problem, end
to end: cardiac **segmentation → ejection fraction**.

![cardioview — held-out heart: predicted chambers, beating](cardioview/docs/media/demo.gif)

*The model's output on a **held-out** patient — predicted chambers (LV cavity / myocardium /
RV) as a beating 3D heart with **EDV / ESV / LVEF vs ground truth**. Drop in your own
`.nii.gz` and it segments in-browser. Interactive viewer: **[cardioview](cardioview/)**.*

Segmentation → **ejection fraction** across imaging modalities (**MRI now; CT, echo planned**),
with the evaluation rigor that decides whether a measurement can be trusted. The connecting
thread is the geometry: how you go from per-voxel labels to a clinical number. Full plan +
milestones in **[ROADMAP.md](ROADMAP.md)**.

## Results — MRI / ACDC (held-out)
2D U-Net, patient-level 80/20 split, 20 held-out patients across all five pathology groups.
Baseline, `--seed 0`. Numbers from `runs/acdc/metrics.json`.

| structure | Dice | HD95 (mm) | published ACDC |
|---|---|---|---|
| LV cavity | **0.93** | 2.1 | ~0.93–0.96 |
| LV myocardium | 0.82 | 3.0 | ~0.88–0.92 |
| RV cavity | 0.86 | 10.0 | ~0.88–0.92 |
| **mean** | **0.87** | | |

**EF vs ground truth: MAE ~3%** (bias −1.5%, 95% limits of agreement [−8, +5]; clinical
equivalence ≈ ±5%). The published column is *context, not a trophy*: ACDC is single-centre and
homogeneous, so matching it means "competent on a clean benchmark" — **not** SOTA or
clinical-grade, and multi-vendor robustness is untested (the hard part). **Where it fails:** RV
boundary is the weak spot (HD95 10 mm) and thick-walled HCM EF errors are largest — closing the
myo/RV gap (augmentation) is the next lever. Error-distribution plots (boundary KDE + EF
Bland–Altman) and the full method live in **[cardioseg/](cardioseg/)**.

## The pieces
- **[cardioseg/](cardioseg/)** — the pipeline (the science): data → preprocess → 2D U-Net →
  measure (EF) → evaluate (Dice / HD95 / ASSD, failure ranking). Train + eval + results detail.
- **[cardioview/](cardioview/)** — browser viewer: predicted chambers as a beating 3D heart +
  EF vs GT; or drop in your own `.nii.gz` → segmented in-browser (ONNX). The result, shown.
- **[mri-sim/](mri-sim/)** — interactive MRI-physics visualizer (spins → k-space → image) —
  understanding the acquisition the model consumes.
- **[learning/](learning/)** — LLM-driven theory write-ups + self-quizzes, alongside the build.

## Honest scope
I come from audio / acoustic-signal ML (end-to-end modeling, evaluation, edge); cardiac imaging
is a deliberate ramp. The approach: build the problem end-to-end, evaluate it, and visualize it,
with the [`learning/`](learning/) track running alongside — learning a field by shipping in it.
Competence built in the gap on public data, not a claim of prior medical-imaging experience;
the ramp is the point, not a disclaimer. Today only the MRI lane is underway; CT and echo are
planned, not done.

## Run
```bash
pip install -e .
python -m pytest tests/ -q     # unit + real-ACDC integration (integration skips without data)
```
Per-project setup — data, training, the viewer — is in each folder's README; start with
**[cardioseg/](cardioseg/)** for the pipeline.

## How it's built
Agent-driven build, human-owned judgment — coding agents scaffold the plumbing; I own the
modeling decisions, the measurement correctness, and the evaluation (the EF/volume math is
spacing-aware and unit-checked; the failure ranking is the point). What transfers from audio ML
is data-structure reasoning and evaluation discipline; the clinical specifics I learn as I go
([learning/](learning/)).
