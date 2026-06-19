# Systole — cardiac segmentation + function (MRI, CT, echo)

**What this is.** systole segments the heart from cardiac MRI and computes its **ejection
fraction** — then asks the question most demos skip: *does that number survive a change of
scanner?* The model trains on a multi-vendor dataset and is tested on a held-out single-centre one
it never saw; segmentation generalizes (Dice **0.87**, matching the in-domain ceiling), while EF —
a ratio of volumes — is the honest hard part. The thread tying it together is geometry: per-voxel
labels → a clinical number.

It's also how **I'm** ramping into cardiac imaging: built on public data, with an LLM-driven
learning track ([`learning/`](learning/): theory write-ups + self-quizzes) alongside the code.
Across modalities (**MRI now; CT, echo planned**); three pieces, general → specific, each linking
into its folder. Full plan → **[ROADMAP.md](ROADMAP.md)**.

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
The science layer: data → preprocess → 2D U-Net → measure (EF) → evaluate. The flagship model is
set up for **domain generalization**: trained on multi-vendor **M&M-2** (360 subjects, 3 vendors,
8 pathologies, 1.5T + 3T), tested on **held-out ACDC** (single-centre, 100 patients it never saw).

### Ejection fraction — the clinical output
EF is the number a clinician acts on, so it's the result that matters. Cross-dataset
(M&M-2 → ACDC, after largest-CC postprocessing): **MAE 8.2%**, bias **−7.2%** (systematic
underprediction), 95% LoA [−34, +19], plus one EF-collapse failure. The chambers are right;
absolute volumes drift as calibration shifts across centres.

![EF Bland–Altman — M&M-2 model on held-out ACDC: error distribution + bias / 95% LoA](cardioseg/docs/media/ef_bland_altman.png)

**Not clinically usable yet** — MAE 8.2% and LoA ±26 are still well past the ±5% clinical bar. The
plot splits the error in two: a systematic **−7.2% bias** (the curve sits left of zero —
correctable) and a **wide spread** with one hard failure (harder). EF is a *ratio* of two volumes,
so it magnifies per-frame segmentation error — the masks are good (Dice 0.87), the derived number isn't.

Paths from here, roughly in effort order:
- ✅ **Largest-CC postprocessing** (applied) — dropping stray false-positive islands cut EF MAE
  9.4 → 8.2%, bias −8.9 → −7.2%, and collapsed the boundary HD (RV 191 → 59 mm). Free, no retrain.
- **Cross-scanner intensity harmonization** — today it's per-volume z-score only; vendor-aware
  histogram standardization may tighten the spread. (Dice already transfers, so this is a smaller
  lever for EF than for segmentation — but it's the obvious gap.)
- **Bias calibration** — a held-out linear EF correction, honest if reported as such.
- **Stronger segmentation** — nnU-Net baseline, 3D context, or vendor-targeted augmentation.

### Segmentation
Per-structure Dice, M&M-2 → ACDC:

| structure | Dice |
|---|---|
| LV cavity | **0.93** |
| LV myocardium | 0.84 |
| RV cavity | 0.84 |
| **mean** | **0.87** |

Train it the other way — single-centre ACDC, tested across vendors — and it drops to **0.70**
(RV 0.85 → 0.59). Diversity in training is what holds up; RV is the weak structure throughout.
Per-direction table, surface metrics (HD95 / ASSD), boundary KDE → **[cardioseg/](cardioseg/)**.

### Baseline — nnU-Net (SOTA reference)
nnU-Net as a *baseline*, not a dependency — quarantined in
[baselines/nnunet/](baselines/nnunet/), scored by the **same** eval. M&M-2 → ACDC:

| segmenter | mean Dice | RV | EF MAE |
|---|---|---|---|
| ours (deployable / ONNX) | 0.87 | 0.84 | 8.2% |
| nnU-Net (50 ep, 1 fold) | **0.91** | **0.91** | **5.5%** |

nnU-Net wins at its *floor* (full 1000-ep × 5-fold recipe goes higher) — biggest on
**RV** (+0.06) and **EF** (8.2 → 5.5%). We deploy the simpler ONNX-exportable model on
purpose; the gap names the levers to close it (instance norm, finer spacing, heavier aug,
TTA). The segmenter is a commodity — the value is the shared measurement + evaluation that
scored both.

## Honest limits — the clinical-grade gap
Competent on public benchmarks, **not** clinical-grade. The specific gaps, measured rather than assumed:
- **EF precision.** 95% LoA are ±20–26% — far past the ±5% clinical bar. And part of the
  underprediction is *intrinsic*: even nnU-Net (SOTA) keeps a **−4% bias** cross-domain, so a better
  segmenter tightens the spread but doesn't erase the lean — it's calibration, not just model quality.
- **Robustness is partial.** Multi-vendor generalization is *tested* (3 vendors, M&M-2 → ACDC), not
  *solved* — still 3 vendors / mostly 1.5T, not a deployment distribution; a 4-vendor / 6-centre set
  (M&Ms-1) sits unused.
- **Validation is thin.** One 80/20 split — no cross-validation, confidence intervals, or
  test–retest; no per-case uncertainty / out-of-distribution flag.
- **Not a device.** Public research data only; no DICOM/PII handling, no prospective or regulatory
  validation.

The route out of each gap is concrete — EF paths above + **[ROADMAP.md](ROADMAP.md)** Gate 2
(largest-CC ✅, then harmonization, calibration, augmentation, eval rigor).

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
