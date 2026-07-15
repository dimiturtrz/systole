# Systole — cardiac segmentation + function (MRI, CT, echo)

**What this is.** systole segments the heart from cardiac MRI and computes its **ejection
fraction** — then asks the question most demos skip: *does that number survive a change of
scanner?* The model trains on two vendors (Siemens + Philips, 495 subjects) and is evaluated on
a centre-shift held-out set (ACDC, val) and two fully unseen vendors (Canon + GE, test);
segmentation generalizes (Dice **0.87** on ACDC, **0.84** on unseen Canon+GE), while EF —
a ratio of volumes — is the honest hard part. The thread tying it together is geometry: per-voxel
labels → a clinical number.

It's also how **I'm** ramping into cardiac imaging: built on public data, with an LLM-driven
learning track ([`learning/`](learning/): theory write-ups + self-quizzes) alongside the code.
Across modalities (**MRI now; CT, echo planned**); three pieces, general → specific, each linking
into its folder. Full plan → **[docs/PLAN.md](docs/PLAN.md)**; roadmap → **[docs/ROADMAP.md](docs/ROADMAP.md)**.

## Understand the acquisition — [mri-sim](mri-sim/)
Interactive 3D visualizer of the MRI **signal pipeline** — spins → slice select →
phase/frequency encode → k-space → reconstructed image, on one clock. Built to *understand the
acquisition* the segmentation model consumes. TypeScript + vtk.js; models the actual signal path.

![mri-sim demo](mri-sim/docs/media/demo.gif)

## See the model work — [cardioview](cardioview/)
Browser viewer (TS + vtk.js) of the flagship model's output on ACDC patients it never trained on
(trained on Siemens+Philips only, 495 subjects): predicted chambers (LV cavity / myocardium / RV) as a **beating 3D heart**
with **EDV / ESV / LVEF vs ground truth**. Or drop in your own `.nii.gz` → segmented **in-browser** (ONNX).

![cardioview — held-out heart: predicted chambers, beating](cardioview/docs/media/demo.gif)

*A heart from a different dataset than the model trained on — the result, shown; the numbers below back it.*

## The pipeline + results — [cardioseg](cardioseg/)
The science layer: data → preprocess → 2D U-Net → measure (EF) → evaluate. The flagship model is
set up for **domain generalization**: trained on Siemens + Philips only (**M&M-2 + M&Ms-1**, 495
labelled subjects, 2 vendors, 1.5T + 3T), validated on **ACDC** (centre/protocol shift, 150
subjects, used for early-stopping + calibration), and tested on **Canon + GE** (two scanner vendors
never seen in training, n=78 total).

### Ejection fraction — the clinical output
EF is the number a clinician acts on, so it's the result that matters. On **ACDC-150**
(val, centre/protocol shift), the model with heavy augmentation + 50% synthetic augmentation +
early stopping + largest-CC + TTA: **MAE 7.1%**, bias **−6.4%** (systematic underprediction), 95%
LoA [−23.2, +10.4]. The chambers are right; absolute volumes drift as calibration shifts across
centres. On the two **unseen-vendor test sets** (Canon n=9, GE n=69), EF MAE is ~10–11%: Canon
9.9%, GE 11.0% — a larger cross-vendor gap that reflects the harder OOD shift.

![EF Bland–Altman — flagship on held-out ACDC: error distribution + bias / 95% LoA](cardioseg/docs/media/ef_bland_altman.png)

**Not clinically usable yet** — MAE 7.1% on val and ~10–11% on unseen vendors are still past the ±5%
clinical bar. The plot splits the error in two: a systematic **−6.4% bias** (the curve sits left
of zero — correctable) and a **spread** (tightened by the work below). EF is a *ratio* of two
volumes, so it magnifies per-frame segmentation error — the masks are good (Dice 0.87), the
derived number isn't.

**Where it fails — stratified (read carefully).** Dice is uniform across pathologies (~0.90), so
**the masks aren't worse anywhere** — the EF spread is a *ratio* effect, not a segmentation one.
EF MAE largely tracks **EF magnitude** (DCM, low-EF dilated ventricles, MAE 2%; high-EF hearts
harder) — a denominator artifact, not worse masks. **HCM is the genuine outlier**: same EF range as
NOR (~60–70%) but **more than double the error** (12% vs 6%) — small thick-walled cavities amplify a
fixed volume error. (The vendor breakdown is murkier — GE has the lowest Dice but its higher EF MAE is
partly a higher-EF cohort + n=11; see [cardioseg/](cardioseg/).)

![EF error by pathology — Dice flat, HCM EF MAE spikes (small-cavity sensitivity)](cardioseg/docs/media/strata_pathology_acdc.png)

Paths from here, roughly in effort order:
- ✅ **Largest-CC postprocessing** (applied) — dropping stray false-positive islands cut EF MAE
  9.4 → 8.2%, bias −8.9 → −7.2%, and collapsed the boundary HD (RV 191 → 59 mm). Free, no retrain.
- ✅ **Test-time augmentation** (applied) — averaging over in-plane flips. Inference-time, no retrain.
- ✅ **Heavy augmentation + early stopping** (applied) — wider geometry + vendor-style intensity
  jitter (gamma / contrast / blur), GPU-batched; trained to a val-Dice plateau (~95 epochs, kept
  the best checkpoint), plus multi-source pooling (M&M-2 + M&Ms-1). Lifts the ACDC-val axis
  (RV Dice 0.84 → ~0.88, EF MAE 8.2 → ~6.5 on the real-only base) before the synth-aug trade below.
- ✅ **Synthetic augmentation** (applied) — the flagship now mixes in **50% physics-based synthetic
  bSSFP images generated from the labels** (`core/data/dynamic/synth.py`) alongside real-image aug.
  Measured effect: it **improved unseen-vendor EF** (Canon 12.1 → **9.9%**, GE 11.5 → **11.0%**) at
  a small ACDC-val cost (Dice 0.88 → **0.87**, EF MAE 6.5 → **7.1%**) — a modest cross-vendor EF gain,
  not a Dice gain. This is the diversify lever that actually moved cross-vendor EF.
- **Cross-scanner intensity harmonization** — today it's per-volume z-score only; vendor-aware
  histogram standardization (Nyúl) was **tried and measured NULL** (0.857 vs 0.864 mean Dice — no
  gain). Dice already transfers, so harmonization had nothing to close; synthetic augmentation, not
  harmonization, was the lever that helped EF.
- **Bias calibration** — a held-out linear EF correction, honest if reported as such.
- **Stronger segmentation** — nnU-Net baseline, 3D context, or vendor-targeted augmentation.

### Segmentation — ours vs SOTA
**A 1.6 M-param model: ~3–4 Dice points under nnU-Net's floor on unseen vendors (Canon 0.836 vs 0.866, GE 0.834 vs 0.878), at ~57× fewer parameters — and it exports to run in the browser. EF gap is model-class epistemic: nnU-Net Canon 2.6% / GE 4.3% vs ours 9.9% / 11.0% — a stronger model class substantially closes it; ours trades that for ONNX portability.**

Per-structure, unseen-vendor Canon+GE (ED+ES), our deployable model vs the nnU-Net SOTA baseline (**same eval**):

<!-- results:compare -->
| model | params | FLOPs | Canon Dice | Canon EF MAE | GE Dice | GE EF MAE |
|---|---|---|---|---|---|---|
| **ours** (ONNX-deployable) | **1.6 M** | **0.8 G** | 0.84 | 9.9% | 0.83 | 11.0% |
| nnU-Net (SOTA baseline) | 92 M | 19 G | 0.87 | **2.6%** | 0.88 | **4.3%** |
<!-- /results:compare -->
<sub>numbers auto-filled from `cardioseg/RESULTS.json` (`cardioseg/evaluation/sync_numbers.py`) — do not hand-edit the table above.</sub>

<sub>params + FLOPs measured (fvcore, single forward; nnU-Net at its 256×320 patch — inference adds tiling + TTA on top).</sub>

**How we train:** a 2D U-Net on Siemens+Philips (495 subjects) — heavy GPU augmentation + **50% synthetic
augmentation** (physics-based bSSFP images generated from the labels, `core/data/dynamic/synth.py`) +
early stopping + largest-CC + TTA — ONNX-exported for cardioview's in-browser inference. The synthetic
mix is the diversify lever that **improved unseen-vendor EF** (Canon 12.1 → 9.9%, GE 11.5 → 11.0%) at a
small ACDC-val cost (Dice 0.88 → 0.87, EF 6.5 → 7.1%). The flagship also trains with **soft
(diffuse) boundary labels** — boundary voxels are partial-volume mixes, so the target is a distribution,
not a hard 0/1 — which leaves Dice/EF unchanged but improves calibration (ECE 0.093 → 0.081 on ACDC val;
`research/deep_dives/2026-06-29_soft-labels-calibration-vs-ef.md`). **The alternative,** nnU-Net
(self-configuring SOTA), runs as a *quarantined baseline* ([baselines/nnunet/](baselines/nnunet/)),
scored through the same eval but **not deployed** (its sliding-window + TTA pipeline doesn't
clean-export). On unseen-vendor Canon+GE (same split, same eval), nnU-Net leads by **~3–4 Dice points**
(Canon 0.866 vs 0.836, GE 0.878 vs 0.834). **EF gap is large and model-class epistemic**: nnU-Net Canon
2.6% vs ours 9.9%; GE 4.3% vs 11.0% — demonstrating the cross-vendor EF gap was reducible by a
stronger model class. cardioseg trades that for ONNX portability at **~57× fewer parameters and ~23×
fewer FLOPs**; nnU-Net's sliding-window pipeline doesn't clean-export.

*This nnU-Net is under-powered on purpose:* 50 epochs / 1 fold / 2D, not its full recipe
(1000 ep × 5-fold ensemble + TTA + config search). So 0.866/0.878 is its **floor** — the full recipe
would pull further ahead. The baseline proves *"I can run + score SOTA through my own eval,"*
not *"I matched its ceiling."*

**Diversity buys robustness:** train it the *other* way — single-centre ACDC, tested across vendors —
and it collapses to **0.70** mean (RV 0.85 → 0.59); the multi-vendor model holds. Per-direction table +
surface metrics (HD95 / ASSD) → **[cardioseg/](cardioseg/)**. (A harder version of this test — training
on *purely synthetic* images, zero real pixels — is in
[Domain shift & normalization](#domain-shift--normalization) below.)

## Honest limits — the clinical-grade gap
Competent on public benchmarks, **not** clinical-grade. The specific gaps, measured rather than assumed:
- **EF precision.** On the ACDC val (centre-shift) 95% LoA are [−23, +10]; on unseen vendors wider
  still — both far past the ±5% clinical bar. And part of the
  the cross-vendor EF gap is **model-class epistemic**: nnU-Net on unseen Canon achieves bias −1.4%,
  LoA [−8.2, +5.4]; on GE bias +0.9%, LoA [−11.7, +13.5] — near the ±5% clinical bar, demonstrating
  the gap was reducible by a stronger model class. Our model's larger gap (bias ~−11%) is a model-class
  limitation, not an irreducible floor.
- **Unseen-vendor EF degrades.** Two vendors never in training (Canon n=9, GE n=69, total n=78)
  both score Dice **0.84** and EF MAE **~10–11%** — Canon and GE agree independently, so the
  cross-vendor signal is robust, but the clinical gap is real (~10–11% vs 7.1% on ACDC val). Canon
  has only 9 labelled subjects (M&Ms-1 withholds most Testing GT), so GE (n=69) carries the EF
  signal; Canon confirms Dice. The LoA on unseen vendors is not yet characterised in full.
- **Validation is thin.** One 80/20 split — no cross-validation, confidence intervals, or
  test–retest; no per-case uncertainty / out-of-distribution flag.
- **Not a device.** Public research data only; no DICOM/PII handling, no prospective or regulatory
  validation.

The route out of each gap is concrete — EF paths above + **[docs/ROADMAP.md](docs/ROADMAP.md)** Gate 2
(largest-CC ✅, then harmonization, calibration, augmentation, eval rigor).

## Data
Three register-gated public sets, **outside the repo** (licensing + size), unified by per-dataset
adapters into one **data cloud** (830 subjects, 4 vendors, harmonized pathology + demographics):
**M&M-2** ([link](https://www.ub.edu/mnms-2/), 360, 3 vendors), **ACDC**
([link](https://www.creatis.insa-lyon.fr/Challenge/acdc/), 150, Siemens), **M&Ms-1** (320 on disk /
213 labelled, 4 vendors incl. Canon). Each adapter remaps labels + parses metadata to a common schema;
we pull *all* data and make our own splits. Flagship model trained on Siemens+Philips only; validated on ACDC (centre-shift, 0.87 Dice) and
tested on Canon+GE (unseen vendors, 0.84 Dice, n=78). Full schema + results →
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
# 1. install — uv resolves the torch CUDA wheel automatically (pinned in [tool.uv.sources]):
uv sync --all-extras                   # creates .venv from pyproject + uv.lock; prefix commands with `uv run`
# 2. point at the data (one path) + drop register-gated downloads under <data>/raw/<dataset>/:
cp paths.example.yaml paths.yaml       # set: data: /abs/path/to/cardiac-data
# 3. consolidate raw -> the homogeneous store (processed/<ds>/<paramkey>/{data,meta.csv}); first run only:
uv run python -m core.data.store  # auto-runs on first train too; this just prints the cloud summary
# 4. train (split = DataCfg criteria; default holds out ACDC + Canon). Artifacts (weights + config +
#    metrics + onnx + card) register into the mlflow model registry — the model store:
uv run python -m cardioseg.training.train --alias production   # --alias production = make it the flagship
# 5. evaluate / export — --run takes a registry ref (alias | version | run-id), default = production:
uv run python -m cardioseg.evaluation.distribution --run production --eval acdc   # + --eval canon; plots + strata
uv run python -m core.export_onnx --run production
```
A run is reproducible from its `config.json` (the split criteria + all hyperparams are serialized) and
the env from `uv.lock`. Cross-platform: same `uv sync` on Windows + Linux; the linux-only GPU extra
(cupy/cucim) installs on Linux, skips on Windows (code falls back to CPU).
nnU-Net baseline is separate (`baselines/nnunet/`, convert → `run_battery.sh`); `nnunetv2` is in `--all-extras`.

## Domain shift & normalization
The reason a model trained on one scanner fails on another is **domain shift** — specifically
**covariate shift** (a.k.a. **acquisition shift** / scanner *batch effects*): the images change, the
anatomy→label relationship doesn't. MRI intensity is uncalibrated (no Hounsfield-like anchor), so this
is *the* core obstacle to cross-vendor generalization. Beating it is **domain generalization** (DG) —
generalizing to vendors held out of training (the M&Ms challenge setting, and ours).

Two opposing forces handle it, and they're **duals** (same physics model, run forward or inverse):
- **harmonize / normalize** (strip the nuisance variance) — z-score, N4 bias correction, Nyúl;
- **augment / domain-randomize** (add variance so the model learns invariance).

We pick by **knowability**: *knowable* variance (a parsable number or an estimable field) is removed at
its source; *unknowable* variance is normalized statistically or learned around via augmentation. The
variance is mapped as **5 source buckets × 2 forces**:

| bucket | shift | knowable → correct / parse | unknowable → normalize / diversify |
|---|---|---|---|
| **machine** | covariate | spacing, vendor, field, scanner, centre | recon scale → z-score / Nyúl; contrast → augment |
| **scan** | covariate | bias field → N4 | receive gain → z-score |
| **patient** | covariate+label | age/sex/BSA, heart locate + orient | body habitus → augmentation |
| **temporal** | — | ED/ES frames | residual motion → augment |
| **annotation** | concept | label convention, papillary rule | inter-observer → **irreducible LoA floor** |

Knowable fields are *parsed from what the datasets ship* (Info.cfg / CSV / folders) — deterministic,
reproducible; a few are paper-cited (verified-flagged). The diversify force splits into **augmentation**
(perturb real images — physics transforms in `training/augment.py`) and **synthetic generation** (invent
images from labels — SynthSeg / Bloch sim, a separate concern). **Full taxonomy, the factor-by-factor
strip-vs-diversify registry, per-dataset coverage, and the build pipeline →
[cardioseg/preprocessing/normalization/](cardioseg/preprocessing/normalization/README.md).**

### Synthetic training data — can a model learn from images it invents?
A SynthSeg-style experiment: train the segmenter on images **generated from the labels** — discard
every real pixel, paint each anatomical class from sampled intensities — and ask whether it still
segments *real* unseen-vendor MRI. The point is a **contrast-agnostic** model: it never sees a scanner,
so it can't overfit one.

Naive randomized synth collapses. The engineering is *why*, and fixing it by **diagnosis, not guessing**
(train-loss curves + per-class intensity stats + rendered images at each step). Cross-vendor mean Dice
(held-out Canon/GE + cmrxmotion, n=147; real-trained baseline **0.864**):

| training data | real px? | mean Dice | what it isolated |
|---|---|---|---|
| 100% synth, random per-class contrast | none | 0.32 | random intensities destroy the blood-bright/myo-dark cue |
| + realistic priors (measured per-class) | none | 0.39 | heart contrast fixed → background is the wall |
| synth heart on **real** background *(diagnostic)* | bg only | 0.77 | proves the background blocks transfer, not the heart |
| + background from real-intensity partition | **none** | **0.66** | bg gets real tissue *shapes* → pure-synth viable |

A U-Net trained on **zero real pixels** segments unseen-vendor cardiac MRI at **0.66 Dice** — short of
the 0.86 real-trained model, but a working demonstration that label-only contrast-agnostic training
transfers. As *augmentation* (synth mixed with real) it's Dice-neutral but **helps cross-vendor EF** —
the flagship now trains with **50% synthetic augmentation**, which cut unseen-vendor EF MAE (Canon 12.1
→ 9.9%, GE 11.5 → 11.0%) at a small ACDC-val cost (Dice 0.88 → 0.87). Honest read: the method
works; the remaining gap is the synthetic background's realism, and the next lever is a learned (GAN)
generator. The data engine lives behind one `Generator`/`GeneratorCfg` seam (`training/synth.py`,
`training/generator.py`); the per-class intensity priors are provenance-tracked reference data.

## Tests
```bash
uv sync --all-extras
uv run pytest tests/ -q        # unit + ACDC integration (integration skips without data)
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

## Development
Tooling and quality gates are provisioned by an in-house copier template (sdlc-scaffold); refresh with
`uvx copier update`, which pins the template version in `.copier-answers.yml`. The gate theory — the
test pyramid, the static-analysis axes — has its one home in that template's docs, not duplicated here.

The structural-guardrail analyzers are an installed package (`sdlc-devtools`, pinned in the `devtools`
extra), not vendored source — so an engine update is a one-line pin bump, no source diff in PRs. They
import as `devtools`, so gates run as `uv run --extra devtools python -m devtools.<graph|magic_literals|
shape_contracts|lcom> <packages>`; the ast-grep shape rules and jscpd config ship inside the package and
are located with `python -m devtools.config sgconfig|jscpd`. The `[tool.structure]` / `[tool.magic_literals]`
/ `[tool.shape_contracts]` blocks in `pyproject.toml` tune them.

## References
- **ACDC** — Bernard et al., *Deep Learning Techniques for Automatic MRI Cardiac Multi-structures
  Segmentation and Diagnosis: Is the Problem Solved?*, IEEE TMI 2018.
- **M&Ms** (M&Ms-1) — Campello et al., *Multi-Centre, Multi-Vendor and Multi-Disease Cardiac
  Segmentation: The M&Ms Challenge*, IEEE TMI 2021.
- **M&Ms-2** — Martín-Isla et al., *M&Ms-2: Multi-Disease, Multi-View & Multi-Center Right
  Ventricular Segmentation in Cardiac MRI*, 2023.
- **nnU-Net** — Isensee et al., *nnU-Net: a self-configuring method for deep learning-based
  biomedical image segmentation*, Nature Methods 2021.
- **SynthSeg** — Billot et al., *SynthSeg: domain randomisation for segmentation of brain scans of any
  contrast and resolution*, Medical Image Analysis 2023. (Basis for the synthetic-training experiment.)
- **MRI simulators** (synthetic generation is built — see *Synthetic training data* above; the Bloch-sim
  physics path remains planned, [ROADMAP](docs/ROADMAP.md)):
  **KomaMRI** (Julia, GPU Bloch), **JEMRIS** (C++, full Bloch), **MRiLab** (MATLAB/GPU).

## License
Code is **MIT** ([LICENSE](LICENSE)). The datasets (ACDC, M&M-2) are **not** included and carry
their own licenses — register with each provider; see [Data](#data).
