# Model Card — cardioseg 2D U-Net (generalization split)

A 2D short-axis cardiac MRI segmenter (LV cavity, LV myocardium, RV cavity) trained for **domain
generalization** across scanner vendors and centres, with ejection fraction derived from the masks.
This card follows Mitchell et al. 2019; the failure-mode and limitation sections are the point.

## Model details
- **Architecture:** MONAI 2D U-Net, 4-class (bg / RV / LV-myo / LV-cav), residual units. Shape from
  `ModelCfg` (`cardioseg/hparams.py`): channels (16,32,64,128,256), strides (2,2,2,2), 2 res-units.
- **Task:** per-slice short-axis segmentation → chamber volumes → EF = (EDV − ESV) / EDV.
- **Training recipe:** Dice+CE loss, Adam (lr 1e-3), GPU-batched augmentation (flip/rotate/scale +
  intensity gamma/contrast/blur/noise), AMP, early stopping (patience 20, 128-epoch ceiling,
  best-val checkpoint), largest-connected-component postprocessing, test-time augmentation (4 flips).
- **Provenance:** every run serializes its full config to `runs/<run>/config.json` and embeds it in
  `metrics.json`. Seed 0. Note `cudnn.benchmark=True` → runs are not bit-identical across machines.
- **Reproduce:** `python -m cardioseg.training.train --out runs/gen` (default split holds out ACDC + Canon; ~6 min, RTX-class GPU).

## Intended use
- **In scope:** research / methods demonstration of cross-vendor cardiac MRI segmentation and EF
  estimation on public benchmark data (ACDC, M&M-1, M&M-2).
- **Out of scope:** **not a medical device, not clinical-grade, not for diagnosis or patient care.**
  Trained/evaluated only on public challenge data — a narrower distribution than real deployment
  (limited vendors/centres/pathologies, curated acquisition). Do not use on non-short-axis views,
  non-cine sequences, or other anatomy.

## Training data
Pooled multi-source "data cloud" (`data/store.py`), **495 labelled subjects** (Siemens+Philips only):
- **M&M-2** (Siemens / Philips, 1.5T+3T) + **M&Ms-1** (Siemens / Philips) — GE excluded from train.
- ACDC = val (centre/protocol shift, early stopping + calibration); Canon+GE = **test (held out entirely)**.
- Splits are our own (a polars query, `data/splits.py`) — challenge splits are not inherited.
- Label conventions remapped to canonical per dataset (verified geometrically); M&Ms-1 withheld-GT
  cases (zero-filled masks) are flagged and excluded.

## Evaluation — generalization split
One model, one declarative split rule (DataCfg criteria: `test_datasets=('acdc',)`,
`test_vendors=('Canon','GE')`):

| split role | n | what it isolates |
|---|---|---|
| **ACDC** — val | 150 | centre / protocol shift (single-vendor Siemens; early stopping + calibration) |
| **Canon** — test | 9 | **unseen vendor** (never in training; M&Ms-1 withholds most GT) |
| **GE** — test | 69 | **unseen vendor** (never in training; independent of Canon) |

## Performance

**ACDC-150 (centre/protocol shift):**

<!-- results:acdc -->
| structure | Dice | HD95 (mm) | ASSD (mm) |
|---|---|---|---|
| LV cavity | 0.92 | 2.1 | 0.58 |
| LV myocardium | 0.86 | 2.1 | 0.56 |
| RV cavity | 0.88 | 7.0 | 0.90 |
| **mean** | **0.88** | | |
<!-- /results:acdc -->
<sub>auto-filled from `RESULTS.json` (`cardioseg/evaluation/sync_numbers.py`) — do not hand-edit.</sub>

Dice + HD95/ASSD pool **both phases (ED+ES)** — ES is the harder phase, so this is the honest read
(ED-only would be ~2 Dice points higher). EF vs GT: **MAE 6.5%**, bias −5.6%, 95% LoA [−20.1, +8.9] (n=150).

**Unseen-vendor test (Canon n=9 + GE n=69 = 78 total):** both return mean Dice **0.839** and EF MAE
**~11–12%** (Canon 11.9%, bias −11.9%; GE 11.3%, bias −10.9%) — independent agreement at n=78 makes
this a robust unseen-vendor signal. Canon n=9 is thin because M&Ms-1 withholds GT for most of its
Testing split (320 on disk, 213 labelled, Canon 50 → 9 usable); GE n=69 is the larger leg.

### Benchmark (nnU-Net, same split)
> **Provisional:** nnU-Net numbers below are on the **old split** (ACDC as test, 564-subject train).
> Re-run on the new split (ACDC=val, Canon+GE=test, 495-subject train) is pending.

nnU-Net trained on the **identical generalization split** (Dataset029_BATTERY, 50 epochs / 1 fold —
a floor), scored by the same eval layer:

<!-- results:cardcompare -->
| ACDC-150 | nnU-Net (50ep/fold0) | this model |
|---|---|---|
| mean Dice | 0.912 | 0.884 |
| EF MAE / bias | 5.6% / -4.2% | 6.5% / -5.6% |
| Canon-9 Dice | 0.876 | 0.84 |
<!-- /results:cardcompare -->

Both numbers pool ED+ES. nnU-Net leads by **~2.8 Dice points** on ACDC (0.912 vs 0.884) and on
unseen-vendor Canon (0.876 vs 0.84) — *even at its floor setting* (50ep/1fold; the full 1000ep ×
5-fold + TTA ceiling, `cardiac-seg-yp3`, would pull further ahead). **EF is roughly level** (6.5% vs
5.6%). cardioseg's boundary is tighter on LV-cav (HD95 2.1 vs 3.3 mm) and myo (2.1 vs 2.9) — from
largest-CC + TTA, which this nnU-Net run lacked. Net: a competent deployable model ~2–3 Dice points
under nnU-Net's floor, EF-comparable, at ~57× fewer parameters.
Full baseline details + its own card → [`baselines/nnunet/MODEL_CARD.md`](../baselines/nnunet/MODEL_CARD.md).

## Where it fails (stratified)
- **HCM / small cavities — the main EF failure mode.** EF bias is **entirely end-systolic cavity
  over-segmentation** (~fixed absolute mL of boundary + papillary voxels), so its *fractional* impact
  scales inversely with cavity size: corr(ES cavity, ESV ratio) = −0.50. Dilated (huge cavity) is EF-
  unbiased (≈ −0.2%); hypertrophic (tiny cavity) is worst (EF MAE ~11.9%). Dice stays fairly flat (~0.88)
  across pathologies — masks aren't worse, the EF *ratio* is range-dependent. (See
  `research/deep_dives/2026-06-21_ef-bias-mechanism-esv-overseg.md`.)
- **RV boundary** is the loosest (HD95 5.8 mm vs myo 2.1, LV-cav 2.1) — basal slices + the small ES cavity.
- **Vendor:** in-domain M&M-2 vendors are level (val split, ED+ES: Siemens/Philips 0.87, GE 0.88) — **no
  minority-vendor deficit**; pooled multi-vendor training (not harmonization) is the robustness lever.

## Limitations & caveats
- Single modality (cine MRI short-axis), 2D slice model — no long-axis, no 3D context.
- Systematic EF under-prediction (bias ≈ −5%) from ES over-segmentation. **Structure (diagnostic):**
  the bias is **proportional** (diff vs EF slope −0.18/EF%, r=−0.56, p<1e-13) and **pathology-structured**
  (DCM ≈ 0% → HCM −11.5%), so a single global offset would over-correct DCM and under-correct HCM —
  an honest correction would have to be per-subgroup, which needs target labels (domain adaptation, not
  zero-shot). So it's reported, not patched. Also **not** corrected by a
  constant offset (size-dependent → would over-correct dilated hearts) and **not** a measurement bug
  (volumes are computed correctly). The honest fix is segmentation-side at ES (`cardiac-seg-7oe`).
- Unseen-vendor test is Canon n=9 + GE n=69 (n=78 total). Canon GT is thin (M&Ms-1 withholds most Testing labels); GE n=69 is the larger leg. Both vendors agree at Dice 0.839 / EF MAE ~11–12%.
- Public-benchmark performance ≠ deployment performance. "Competent on public benchmarks," not clinical.

## References
- Datasets: ACDC (Bernard 2018), M&Ms-1 (Campello 2021 TMI), M&M-2.
- Method/results detail: `cardioseg/README.md`. Config/provenance: `runs/<run>/config.json`.
