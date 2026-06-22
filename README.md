# Systole — cardiac segmentation + function (MRI, CT, echo)

**What this is.** systole segments the heart from cardiac MRI and computes its **ejection
fraction** — then asks the question most demos skip: *does that number survive a change of
scanner?* The model trains on a multi-vendor dataset and is tested on a held-out single-centre one
it never saw; segmentation generalizes (Dice **0.91**, matching the in-domain ceiling), while EF —
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
set up for **domain generalization**: trained on a pooled multi-vendor cloud (**M&M-2 + M&Ms-1**,
564 labelled subjects, 4 vendors, 1.5T + 3T), held out along **two axes** — **ACDC** (centre/protocol
shift, 150 it never saw) and **Canon** (a scanner vendor never in training).

### Ejection fraction — the clinical output
EF is the number a clinician acts on, so it's the result that matters. On held-out **ACDC-150**
(centre/protocol shift), the pooled multi-vendor model with heavy augmentation + early stopping +
largest-CC + TTA: **MAE 5.9%**, bias **−5.2%** (systematic underprediction), 95% LoA [−18, +8]. The
chambers are right; absolute volumes drift as calibration shifts across centres.

![EF Bland–Altman — flagship on held-out ACDC: error distribution + bias / 95% LoA](cardioseg/docs/media/ef_bland_altman.png)

**Not clinically usable yet** — MAE 5.9% and LoA ±13 are still past the ±5% clinical bar. The
plot splits the error in two: a systematic **−5.2% bias** (the curve sits left of zero —
correctable) and a **spread** (tightened by the work below). EF is a *ratio* of two volumes,
so it magnifies per-frame segmentation error — the masks are good (Dice 0.91), the derived number isn't.

**Where it fails — stratified (read carefully).** Dice is uniform across pathologies (~0.90), so
**the masks aren't worse anywhere** — the EF spread is a *ratio* effect, not a segmentation one.
EF MAE largely tracks **EF magnitude** (DCM, low-EF dilated ventricles, MAE 2%; high-EF hearts
harder) — a denominator artifact, not worse masks. **HCM is the genuine outlier**: same EF range as
NOR (~60–70%) but **more than double the error** (12% vs 5%) — small thick-walled cavities amplify a
fixed volume error. (The vendor breakdown is murkier — GE has the lowest Dice but its higher EF MAE is
partly a higher-EF cohort + n=11; see [cardioseg/](cardioseg/).)

![EF error by pathology — Dice flat, HCM EF MAE spikes (small-cavity sensitivity)](cardioseg/docs/media/strata_pathology_acdc.png)

Paths from here, roughly in effort order:
- ✅ **Largest-CC postprocessing** (applied) — dropping stray false-positive islands cut EF MAE
  9.4 → 8.2%, bias −8.9 → −7.2%, and collapsed the boundary HD (RV 191 → 59 mm). Free, no retrain.
- ✅ **Test-time augmentation** (applied) — averaging over in-plane flips. Inference-time, no retrain.
- ✅ **Heavy augmentation + early stopping** (applied) — wider geometry + vendor-style intensity
  jitter (gamma / contrast / blur), GPU-batched; trained to a val-Dice plateau (~95 epochs, kept
  the best checkpoint), plus multi-source pooling (M&M-2 + M&Ms-1). RV Dice 0.84 → **0.92**,
  EF MAE 8.2 → **5.9%**, LoA ±27 → ±13.
- **Cross-scanner intensity harmonization** — today it's per-volume z-score only; vendor-aware
  histogram standardization may tighten the spread. (Dice already transfers, so this is a smaller
  lever for EF than for segmentation — but it's the obvious gap.)
- **Bias calibration** — a held-out linear EF correction, honest if reported as such.
- **Stronger segmentation** — nnU-Net baseline, 3D context, or vendor-targeted augmentation.

### Segmentation — ours vs SOTA
Per-structure, M&M-2 → ACDC, our deployable model vs the nnU-Net SOTA baseline (**same eval**):

| | params | FLOPs | LV-cav | myo | RV | **mean Dice** | EF MAE |
|---|---|---|---|---|---|---|---|
| **ours** (ONNX-deployable) | **1.6 M** | **0.8 G** | 0.95 | 0.85 | 0.92 | **0.91** | 5.9% |
| nnU-Net (SOTA baseline) | 92 M | 19 G | 0.95 | 0.88 | 0.91 | 0.91 | 5.6% |

<sub>params + FLOPs measured (fvcore, single forward; nnU-Net at its 256×320 patch — inference adds tiling + TTA on top).</sub>

**How we train:** a 2D U-Net on multi-vendor M&M-2 — heavy GPU augmentation + early stopping +
largest-CC + TTA — ONNX-exported for cardioview's in-browser inference. **The alternative,** nnU-Net
(self-configuring SOTA), runs as a *quarantined baseline* ([baselines/nnunet/](baselines/nnunet/)),
scored through the same eval but **not deployed** (its sliding-window + TTA pipeline doesn't
clean-export). On the ACDC axis the two are **roughly level** (Dice 0.91 vs 0.912, EF 5.9 vs 5.6%) —
at **~57× the parameters and ~23× the FLOPs**. nnU-Net keeps a slight edge on the unseen-vendor Canon
axis. Holding level with SOTA's floor at that size is the point of a tiny, in-browser-exportable model.

*Caveat — this nnU-Net is under-powered on purpose:* 50 epochs / 1 fold / 2D, not its full recipe
(1000 ep × 5-fold ensemble + TTA + config search). So 0.912 is its **floor** — the full recipe would
pull ahead. The baseline proves *"I can run + score SOTA through my own eval, and match its floor,"*
not *"I matched its ceiling."*

**Diversity buys robustness:** train it the *other* way — single-centre ACDC, tested across vendors —
and it collapses to **0.70** mean (RV 0.85 → 0.59); the multi-vendor model holds. Per-direction table +
surface metrics (HD95 / ASSD) → **[cardioseg/](cardioseg/)**.

## Honest limits — the clinical-grade gap
Competent on public benchmarks, **not** clinical-grade. The specific gaps, measured rather than assumed:
- **EF precision.** 95% LoA are ±13% — still past the ±5% clinical bar. And part of the
  underprediction is *intrinsic*: even nnU-Net (SOTA) keeps a **−4% bias** cross-domain, so a better
  segmenter tightens the spread but doesn't erase the lean — it's calibration, not just model quality.
- **The unseen-vendor test is thin.** We hold out two axes — ACDC (centre shift) **and Canon** (a
  scanner vendor never in training). So the machine axis *is* tested now — but Canon has only **n=9
  labelled** (M&Ms-1 withholds most Testing GT), enough for a Dice signal (~0.87) but too noisy for EF.
  Leave-one-vendor-out (n up to ~190 for GE/Philips) would give proper unseen-vendor stats; not yet run.
- **Validation is thin.** One 80/20 split — no cross-validation, confidence intervals, or
  test–retest; no per-case uncertainty / out-of-distribution flag.
- **Not a device.** Public research data only; no DICOM/PII handling, no prospective or regulatory
  validation.

The route out of each gap is concrete — EF paths above + **[ROADMAP.md](ROADMAP.md)** Gate 2
(largest-CC ✅, then harmonization, calibration, augmentation, eval rigor).

## Data
Three register-gated public sets, **outside the repo** (licensing + size), unified by per-dataset
adapters into one **data cloud** (830 subjects, 4 vendors, harmonized pathology + demographics):
**M&M-2** ([link](https://www.ub.edu/mnms-2/), 360, 3 vendors), **ACDC**
([link](https://www.creatis.insa-lyon.fr/Challenge/acdc/), 150, Siemens), **M&Ms-1** (320 on disk /
213 labelled, 4 vendors incl. Canon). Each adapter remaps labels + parses metadata to a common schema;
we pull *all* data and make our own splits. Flagship pooled multi-source model holds across a 2-axis
generalization held-out split (ACDC centre-shift 0.91 Dice, Canon unseen-vendor 0.87). Full schema + results →
**[cardioseg/](cardioseg/)**.

**Config — one path, everything derived.** Copy the template, set a single `data` root:
```bash
cp paths.example.yaml paths.yaml      # then: data: /abs/path/to/cardiac-data
```
Under that root you create **`raw/`** and drop the (register-gated) downloads in —
`raw/acdc/`, `raw/mnm2/`, `raw/MnM/` (M&Ms-1). That's the only manual step. Everything else is
automatic: **`<data>/processed/`** (the preprocess cache) is created on first run, datasets are
discovered by name, and the per-dataset label conventions are remapped to canonical on load. Env
`CARDIAC_DATA` overrides the file (CI). Loaded by `cardioseg/config.py`.

## Quickstart (order of operations)
```bash
# 1. install (torch CUDA build FIRST — plain PyPI is CPU-only):
pip install torch --index-url https://download.pytorch.org/whl/cu128   # >=2.7 (Blackwell/RTX 5090)
pip install -e .                       # core; extras as needed: .[n4,export,viz,nnunet,dev] or .[all]
# 2. point at the data (one path) + drop register-gated downloads under <data>/raw/<dataset>/:
cp paths.example.yaml paths.yaml       # set: data: /abs/path/to/cardiac-data
# 3. consolidate raw -> the homogeneous store (processed/<ds>/<paramkey>/{data,meta.csv}); first run only:
python -m cardioseg.data.store         # auto-runs on first train too; this just prints the cloud summary
# 4. train (split = DataCfg criteria; default holds out ACDC + Canon). Full config -> runs/<run>/config.json:
python -m cardioseg.training.train --out runs/gen
# 5. evaluate / export:
python -m cardioseg.evaluation.distribution --run runs/gen --eval acdc   # + --eval canon; plots + strata
python -m cardioseg.training.export_onnx --run runs/gen                  # needs .[export]
```
A run is reproducible from its `config.json` (the split criteria + all hyperparams are serialized).
nnU-Net baseline is separate: `pip install -e .[nnunet]`, then `baselines/nnunet/` (convert → `run_battery.sh`).

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
