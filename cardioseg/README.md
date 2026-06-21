# cardioseg — the pipeline

The science layer: cardiac MRI → segmentation → ejection fraction → evaluation, set up for
**domain generalization** (train on multi-vendor **M&M-2**, test on held-out single-centre
**ACDC**). Python (PyTorch + MONAI). The browser demo ([cardioview](../cardioview/)) consumes
what this produces.

## Pipeline
1. **Data** — modality loader + normalization (ACDC short-axis cine MRI; NIfTI, spacing-aware
   in mm; geometric LV/RV disambiguation).
2. **Segment** — 2D U-Net (MONAI) → per-voxel labels (bg / RV / LV-myo / LV-cav).
3. **Measure** — chamber volumes (voxel count × voxel volume, mm³ → mL); **EF = (EDV − ESV) / EDV**.
4. **Evaluate** — Dice + HD95 / ASSD per structure + EF vs GT; **failure ranking** (the worst
   cases decide clinical trust, not the mean).
5. **Geometry/viz** — marching-cubes chamber meshes; error-distribution plots.

## Setup
```bash
pip install -e .                  # from repo root (installs cardioseg)
# torch CUDA build (CPU wheel won't train); Blackwell/RTX 5090 needs torch>=2.7:
pip install torch --index-url https://download.pytorch.org/whl/cu128
```
Data lives **outside the repo** (licensing + size). Set **one** path in `paths.yaml` and lay the
register-gated downloads under `<data>/raw/<dataset>/`:
```bash
cp paths.example.yaml paths.yaml      # then: data: /abs/path/to/cardiac-data
#   <data>/raw/acdc/      register: https://www.creatis.insa-lyon.fr/Challenge/acdc/
#   <data>/raw/mnm2/      register: https://www.ub.edu/mnms-2/
#   <data>/raw/MnM/       register: M&Ms-1  (optional, broadest multi-site set)
#   <data>/processed/     preprocess cache — auto-created, leave it alone
```
That's the only manual step. Datasets are discovered by name under `raw/`; the preprocess cache is
created on first run; per-dataset label conventions (M&M-2/M&Ms-1 have LV=1, ACDC LV=3) are
remapped to canonical on load (verified geometrically), so one model spans them. Env `CARDIAC_DATA`
overrides the file (CI). Loaded by `cardioseg/config.py`; adapters live in `cardioseg/data/mri/`
behind a `DatasetAdapter` interface (add a dataset = one file + one registry line).

## Data cloud
Data = scans + metadata in one homogeneous **store**. Per-dataset **adapters**
(`data/mri/{acdc,mnm2,mnms1}.py` behind a `DatasetAdapter` interface) read raw→canonical; `data/store.py`
consolidates each into a self-contained processed set mirroring `raw/`:

```
processed/<dataset>/<paramkey>/        # paramkey = inplane1p5 | inplane1p5_n4
    data/<subject>.npz                 # resampled + z-scored ed/es img+gt + spacing
    meta.csv                           # common schema (read with polars)
```

`store.load(names)` ensures each is processed (builds if the folder is missing) then returns **one
polars frame** over them — concatenating the per-dataset `meta.csv` *is* the data cloud (no separate
inventory). **Splits are queries over it** (`data/splits.py`): roles aren't baked into datasets. We
pull **all** data and make our **own** splits — challenge splits aren't inherited.

```bash
python -m cardioseg.data.store                 # consolidate all + print the cloud summary
python -m cardioseg.training.train --battery   # train; hold out ACDC + Canon (one split rule)
```

| dataset | n | vendors | demographics | role |
|---|---|---|---|---|
| **ACDC** | 150 (train 100 + test 50, both labelled) | Siemens only | height/weight → BSA | clean cross-centre test |
| **M&M-2** | 360 | Siemens 219 / Philips 88 / GE 53 | — | multi-vendor train |
| **M&Ms-1** | 320 on disk / **213 labelled** | Philips 125 / Siemens 95 / GE 50 / **Canon 50** (9 labelled) | age / sex / h+w | broadest; **Canon = clean unseen-vendor test** (overlap-free) |

**Unification** each adapter handles: label remap to canonical (0 bg / 1 RV / 2 myo / 3 LV-cav —
M&M-2 & M&Ms-1 are LV=1, verified geometrically); ED/ES selection; `meta()` parsing of acquisition +
demographics from the dataset's own sidecars (Info.cfg / CSV). `null` is a valid value (unknown).

**Stratification axes** (the cloud's columns): **vendor** (4, everywhere), **harmonized pathology**
(`data/mri/pathology.py` → normal / dilated / hypertrophic / ischemic / rv_congenital / other —
the 3 vocabularies collapsed), and **demographics** for fairness (sex + age-band on M&Ms-1; **BSA**
on 283 = ACDC + M&Ms-1, salvaged from height+weight). Pooled cloud: 830 subjects, balanced pathology.

**Withheld-GT caveat:** M&Ms-1 ships 320 cases but only **213 have usable masks** — the challenge
zero-fills the GT for much of Testing (the gt *file* exists but is all-background), so train/eval
must filter on mask *content*, not file existence (`mnms1_cases(labelled_only=True)`). This is why
**Canon = 50 on disk but n=9 labelled** — thin, but the only unseen-vendor data with public masks.

**Overlap caveat:** M&Ms-1 ⊃ ~195 of M&M-2 (shared NOR/HCM/LV per the M&M-2 docs; mapping
unavailable) — so the two can't be each other's clean held-out test; ACDC is the only fully
independent set. **Adopted split (2-axis battery):** train M&M-2 → test ACDC (centre shift) +
Canon (unseen vendor); the overlapping M&Ms-1 270 parks pending dedup (`bd cardiac-seg-3ah`).
Dataset-role decisions track in `bd cardiac-seg-bsz`.

## Train + evaluate
```bash
# flagship: pooled-train, hold out ACDC (centre) + Canon (vendor) — one split rule
python -m cardioseg.training.train --battery        # -> runs/battery/ (128-ep ceiling, early-stops ~106, ~6 min)
# legacy single-source direction (e.g. the asymmetry A/B)
python -m cardioseg.training.train --dataset acdc --test mnm2 --epochs 40   # -> runs/acdc/
python -m cardioseg.evaluation.distribution --run runs/battery --eval acdc    # KDE + Bland-Altman -> plots/
python -m cardioseg.evaluation.distribution --run runs/battery --eval canon   # the unseen-vendor axis
python -m cardioseg.training.export_onnx --run runs/battery   # model.onnx (+INT8) for the web viewer
```
Training reads the consolidated store (builds `processed/<ds>/` on first run), logs phase timings +
per-epoch batch-rate to `runs/<run>/train.log`. In-RAM dataset + GPU-batched augment → `--workers`
parallelizes store consolidation, not the loader (DataLoader runs workers=0; AMP + cudnn.benchmark).

> Intended-use envelope, stratified metrics, failure modes + provenance → **[MODEL_CARD.md](MODEL_CARD.md)**.

## Results (seed 0, patient-level splits)
Flagship = the **generalization battery**: train on the pooled multi-source cloud (M&M-2 + M&Ms-1
ex-Canon, 451 subjects), hold out **two axes** — ACDC (centre/protocol shift) and Canon (unseen
vendor) — as one declarative split rule (`data/splits.py`). Heavy aug + early stopping + largest-CC
+ TTA. On the **ACDC-150** axis:

| structure | Dice | HD95 (mm) | ASSD (mm) | published ACDC |
|---|---|---|---|---|
| LV cavity | **0.94** | 1.5 | 0.27 | ~0.93–0.96 |
| LV myocardium | 0.86 | 2.1 | 0.46 | ~0.88–0.92 |
| RV cavity | 0.91 | 3.0 | 0.49 | ~0.88–0.92 |
| **mean** | **0.90** | | | |

**EF vs GT: MAE 7.1%** (bias −6.6%, n=150). **Two-axis battery** (one model, our own splits — the
challenge splits aren't inherited):

| held-out axis | n | mean Dice | EF MAE |
|---|---|---|---|
| **ACDC** (centre / protocol shift, single-vendor) | 150 | 0.90 | 7.1% |
| **Canon** (unseen vendor, M&Ms-1) | 9 | 0.85 | 15.4% \* |

\* Canon **n=9** — too thin to read EF on: the same axis gave EF MAE 7.2% under a M&M-2-only model,
15.4% here (one collapsed case dominates 9), while Dice held ~0.85 both ways. The M&Ms-1 challenge
withholds GT for most of its Testing split (320 on disk, 213 labelled; Canon 50 → 9 with masks), so
Canon is the honest *Dice* signal for unseen-vendor robustness; its EF is noise. Pooling M&Ms-1 into
training (451 vs 290 subjects) lifted the solid ACDC axis (mean 0.89 → 0.90, **RV 0.88 → 0.91, HD95
5.0 → 3.0 mm** — the extra RV diversity paid off).

**Diversity buys robustness — the asymmetry proves it:**

| train → test | mean Dice | RV | EF MAE |
|---|---|---|---|
| ACDC → ACDC (in-domain) | 0.87 | 0.85 | 4.7% |
| ACDC → M&M-2 (out-of-distribution) | 0.70 | 0.59 | 9.1% |
| M&M-2 → ACDC (generalization, flagship) | 0.87 | 0.84 | 9.4% |

*Asymmetry table is the base model (identical config across directions, for a fair A/B); the pooled
battery + heavy aug + largest-CC + TTA lift the flagship to 0.90 Dice / 7.1% EF on ACDC-150 (top table).*

- Single-centre training loses ~17 Dice points off its home dataset (RV collapses 0.85 → 0.59);
  multi-vendor training carries to a new centre — and a new **vendor** — with **no segmentation drop**.
- **EF transfers worse than Dice** — volume calibration shifts across centres (in-domain EF MAE
  4.7% → cross-dataset ~6–9%); the chambers are right, the absolute mL drift.
- **Surface metrics:** RV still has the loosest boundary (HD95 3.0 mm vs myo 2.1, LV-cav 1.5) — basal
  slices + stray voxels — but the pooled-train RV diversity tightened it sharply (was 5.0 mm). ASSD
  stays sub-mm everywhere; full HD is the fragile max (one stray voxel → ~200 mm), **HD95** is robust.
- `runs/<run>/plots/`: per-class boundary-distance **KDE** + EF **Bland–Altman** (flagship below).

![EF Bland–Altman — battery model on held-out ACDC: difference distribution + bias / 95% LoA](docs/media/ef_bland_altman.png)
![Per-class boundary-distance KDE — battery model on held-out ACDC](docs/media/boundary_kde.png)

### Stratified — where it actually fails
Pooled numbers average over the failures. Broken down (same model, same eval; `distribution.py`
emits these + `stratified.json`):

**By pathology** (held-out ACDC). Dice is flat (~0.90 everywhere) → **masks aren't worse anywhere**;
the EF spread is a *ratio* effect. `gtEF` is given because EF MAE isn't comparable across groups
with different cavity sizes — a fixed volume error moves EF more when the cavity is small:

| pathology | gtEF | mean Dice | EF MAE | EF bias |
|---|---|---|---|---|
| dilated (DCM) | 20% | 0.91 | **2.1%** | −0.5% |
| ischemic (MINF) | 31% | 0.91 | 4.2% | −3.9% |
| rv_congenital | 57% | 0.90 | 8.0% | −8.0% |
| normal (NOR) | 62% | 0.92 | 7.4% | −7.1% |
| **hypertrophic (HCM)** | 70% | 0.91 | **13.8%** | −13.3% |

**Mechanism (decomposed, not hand-waved — `4yf`):** split EF into its two volumes and the bias
localizes cleanly. **EDV is accurate** (ACDC pred/gt 1.01 → ED cavity convention matches across
datasets, *not* an annotation bug); **ESV is over-predicted ~19%**, and that alone produces the whole
−5.8% EF bias. The over-segmentation is a roughly **fixed absolute mL** at the cavity boundary
(partial-volume + papillaries bulging into the small contracted ES cavity) — so its *fractional*
impact scales inversely with cavity size: corr(ES cavity, ESV ratio) = **−0.50**. DCM (huge cavity)
is **unbiased** (ESV ratio 0.99, EF bias −0.6); HCM (tiny cavity) is worst (ESV ratio 1.51, −10.9).
That is the HCM "outlier" — not EF-range sensitivity, but a fixed ES boundary over-seg seen through a
small denominator. **Consequence:** the fix is segmentation-side at ES (boundary-aware loss), *not* a
constant EF subtraction (size-dependent → would over-correct DCM) and *not* `measure.py` (volumes are
right). See `research/deep_dives/2026-06-21_ef-bias-mechanism-esv-overseg.md`.

![EF error by pathology — Dice flat, HCM EF MAE spikes (small-cavity sensitivity)](docs/media/strata_pathology_acdc.png)

**By vendor** (in-domain M&M-2 val, battery model) — read with care:

| vendor | n | gtEF | mean Dice | EF MAE |
|---|---|---|---|---|
| Siemens | 43 | 48% | 0.889 | 12.1% |
| Philips | 20 | 59% | 0.910 | 11.7% |
| **GE** | 9 | 55% | 0.903 | 11.0% |

The **GE minority-vendor deficit closed**: under the M&M-2-only model GE had the lowest Dice (0.879);
pooling M&Ms-1 into training lifted it to **0.903**, now level with Philips and *above* Siemens —
more vendor diversity in train, not harmonization, fixed the gap. That weakens the standalone case
for intensity harmonization (`qfz`): the cheapest robustness lever is still more multi-vendor data.
(EF MAE ~11–12% across vendors here is the in-domain pathology mix — the ESV small-cavity effect, not
a vendor signal; the clean per-vendor read is Dice.)

![Dice + EF MAE by vendor — GE deficit closed under pooled training](docs/media/strata_vendor_mnm2.png)

Published column = context, not a trophy: even multi-vendor, this is "competent on public
benchmarks," not clinical-grade. M&M-2 is 3 vendors / 1.5–3T — broader than ACDC, still not the
full deployment distribution.

## Layout
```
cardioseg/
  data/mri/base.py        # DatasetAdapter interface + shared primitives (load_nifti, labels, LV/RV id)
  data/mri/acdc.py        # ACDC adapter (canonical labels, Info.cfg meta)
  data/mri/mnm2.py        # M&M-2 adapter (multi-vendor; label_map remaps to canonical)
  data/mri/mnms1.py       # M&Ms-1 adapter (6-centre/4-vendor; 4D ED/ES + CSV)
  data/mri/registry.py    # name -> adapter (add a dataset = one file + one line)
  data/store.py           # consolidate adapters -> processed/<ds>/<paramkey>/{data,meta.csv}; load() = polars cloud
  data/splits.py          # splits as polars queries (battery: hold out acdc + Canon)
  preprocessing/preprocess.py   # the per-subject transform: resample in-plane + (N4) + z-score
  training/
    model.py              # MONAI U-Net factory (2D/3D)
    dataset.py            # 2D-slice dataset over consolidated npz paths
    train.py              # training loop (--dataset/--test, or --battery; workers+AMP)
    export_onnx.py        # trained U-Net -> ONNX (+INT8 quant), torch-parity gated
  evaluation/
    measure.py            # chamber volumes + ejection fraction (spacing-aware)
    evaluate.py           # Dice / surface distances (HD/HD95/ASSD) / failure ranking
    validate.py           # per-class Dice + EF vs GT on held-out patients
    distribution.py       # boundary-distance KDE + EF Bland-Altman
    losses.py             # compound Dice + cross-entropy
  analysis/{eda,viz}.py   # ACDC reality-check + marching-cubes surface mesh
config.py                 # paths.yaml loader (OmegaConf)
```
Tests: `tests/unit` (geometry, metrics, preprocessing) + `tests/integration` (real ACDC, skips
if data absent).
