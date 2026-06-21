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
(M&M-2 → ACDC, with heavy augmentation + early stopping + largest-CC + TTA): **MAE 6.3%**,
bias **−5.6%** (systematic underprediction), 95% LoA [−21, +9]. The chambers are right;
absolute volumes drift as calibration shifts across centres.

![EF Bland–Altman — M&M-2 model on held-out ACDC: error distribution + bias / 95% LoA](cardioseg/docs/media/ef_bland_altman.png)

**Not clinically usable yet** — MAE 6.3% and LoA ±15 are still past the ±5% clinical bar. The
plot splits the error in two: a systematic **−5.6% bias** (the curve sits left of zero —
correctable) and a **spread** (tightened by the work below). EF is a *ratio* of two volumes,
so it magnifies per-frame segmentation error — the masks are good (Dice 0.90), the derived number isn't.

**Where it fails — stratified (read carefully).** Dice is uniform across pathologies (~0.90), so
**the masks aren't worse anywhere** — the EF spread is a *ratio* effect, not a segmentation one.
EF MAE largely tracks **EF magnitude** (DCM, low-EF dilated ventricles, MAE 2%; high-EF hearts
harder) — a denominator artifact, not worse masks. **HCM is the genuine outlier**: same EF range as
NOR (~60–68%) but **double the error** (11% vs 5%) — small thick-walled cavities amplify a fixed
volume error. (The vendor breakdown is murkier — GE has the lowest Dice but its higher EF MAE is
partly a higher-EF cohort + n=11; see [cardioseg/](cardioseg/).)

![EF error by pathology — Dice flat, HCM EF MAE spikes (small-cavity sensitivity)](cardioseg/docs/media/strata_pathology_acdc.png)

Paths from here, roughly in effort order:
- ✅ **Largest-CC postprocessing** (applied) — dropping stray false-positive islands cut EF MAE
  9.4 → 8.2%, bias −8.9 → −7.2%, and collapsed the boundary HD (RV 191 → 59 mm). Free, no retrain.
- ✅ **Test-time augmentation** (applied) — averaging over in-plane flips. Inference-time, no retrain.
- ✅ **Heavy augmentation + early stopping** (applied) — wider geometry + vendor-style intensity
  jitter (gamma / contrast / blur), GPU-batched; trained to a val-Dice plateau (~95 epochs, kept
  the best checkpoint). RV Dice 0.84 → **0.89**, EF MAE 8.2 → **6.3%**, LoA ±27 → ±15.
- **Cross-scanner intensity harmonization** — today it's per-volume z-score only; vendor-aware
  histogram standardization may tighten the spread. (Dice already transfers, so this is a smaller
  lever for EF than for segmentation — but it's the obvious gap.)
- **Bias calibration** — a held-out linear EF correction, honest if reported as such.
- **Stronger segmentation** — nnU-Net baseline, 3D context, or vendor-targeted augmentation.

### Segmentation — ours vs SOTA
Per-structure, M&M-2 → ACDC, our deployable model vs the nnU-Net SOTA baseline (**same eval**):

| | params | FLOPs | LV-cav | myo | RV | **mean Dice** | EF MAE |
|---|---|---|---|---|---|---|---|
| **ours** (ONNX-deployable) | **1.6 M** | **0.8 G** | 0.94 | 0.86 | 0.89 | **0.90** | 6.3% |
| nnU-Net (SOTA baseline) | 92 M | 19 G | 0.95 | 0.87 | 0.91 | 0.91 | 5.5% |

<sub>params + FLOPs measured (fvcore, single forward; nnU-Net at its 256×320 patch — inference adds tiling + TTA on top).</sub>

**How we train:** a 2D U-Net on multi-vendor M&M-2 — heavy GPU augmentation + early stopping +
largest-CC + TTA — ONNX-exported for cardioview's in-browser inference. **The alternative,** nnU-Net
(self-configuring SOTA), runs as a *quarantined baseline* ([baselines/nnunet/](baselines/nnunet/)),
scored through the same eval but **not deployed** (its sliding-window + TTA pipeline doesn't
clean-export). It leads by ~1 Dice pt / 0.8 EF pt — at **~57× the parameters and ~23× the FLOPs**.
That gap is the price of a tiny, fully-owned, in-browser-exportable model.

*Caveat — this nnU-Net is under-powered on purpose:* 50 epochs / 1 fold / 2D, not its full recipe
(1000 ep × 5-fold ensemble + TTA + config search). So 0.91 is its **floor** — true nnU-Net would lead
by more. The baseline proves *"I can run + score SOTA through my own eval,"* not *"I matched it."*

**Diversity buys robustness:** train it the *other* way — single-centre ACDC, tested across vendors —
and it collapses to **0.70** mean (RV 0.85 → 0.59); the multi-vendor model holds. Per-direction table +
surface metrics (HD95 / ASSD) → **[cardioseg/](cardioseg/)**.

## Honest limits — the clinical-grade gap
Competent on public benchmarks, **not** clinical-grade. The specific gaps, measured rather than assumed:
- **EF precision.** 95% LoA are ±15% — still past the ±5% clinical bar. And part of the
  underprediction is *intrinsic*: even nnU-Net (SOTA) keeps a **−4% bias** cross-domain, so a better
  segmenter tightens the spread but doesn't erase the lean — it's calibration, not just model quality.
- **The held-out test is single-vendor.** We *train* multi-vendor (M&M-2, 3 vendors) but *test* on
  ACDC — one centre, **one scanner vendor (Siemens)**. So we test cross-*centre* generalization, but
  the **machine axis is not directly tested on held-out data** (you can't see vendor-robustness with
  one vendor in the test set). The fix is a multi-vendor held-out test — **M&Ms-1** (4 vendors incl.
  Canon, 6 centres) is on disk and adapter-ready for exactly this; dataset roles (which set trains vs
  tests) are still to be decided. Until then, vendor breakdowns are in-domain only (caveated).
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

**Config — one path, everything derived.** Copy the template, set a single `data` root:
```bash
cp paths.example.yaml paths.yaml      # then: data: /abs/path/to/cardiac-data
```
Under that root you create **`raw/`** and drop the (register-gated) downloads in —
`raw/acdc/`, `raw/mnm2/`, `raw/MnM/` (M&Ms-1). That's the only manual step. Everything else is
automatic: **`<data>/processed/`** (the preprocess cache) is created on first run, datasets are
discovered by name, and the per-dataset label conventions are remapped to canonical on load. Env
`CARDIAC_DATA` overrides the file (CI). Loaded by `cardioseg/config.py`.

## Data normalization
MRI intensity is uncalibrated (no Hounsfield-like anchor), so inter-scanner variance is the core
problem. We organize it by **5 source buckets × 2 axes**: *knowable* variance is removed at the
right layer (parse it / correct it physically); *unknowable* variance is normalized statistically.

| bucket | knowable → correct/parse | unknowable → normalize |
|---|---|---|
| **machine** | spacing, vendor, field, scanner, centre | recon scale → z-score / Nyúl |
| **scan** | bias field → N4 | receive gain → z-score |
| **patient** | age/sex/BSA, heart locate + orient | body shape → augmentation |
| **temporal** | ED/ES frames | residual motion |
| **annotation** | label convention, papillary rule | inter-observer → irreducible floor |

Most knowable fields are *parsed from what the datasets ship* (Info.cfg / CSV / folders) — reproducible,
deterministic. A few come from the dataset papers (cited, verified-flagged). Full schema, per-dataset
coverage, and the build pipeline → **[cardioseg/normalization/](cardioseg/normalization/)**.

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

## License
Code is **MIT** ([LICENSE](LICENSE)). The datasets (ACDC, M&M-2) are **not** included and carry
their own licenses — register with each provider; see [Data](#data).
