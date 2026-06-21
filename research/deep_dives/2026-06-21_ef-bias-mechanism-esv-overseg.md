# EF bias mechanism: ES cavity over-segmentation, not annotation convention

**Task:** cardiac-seg-4yf (papillary/basal convention ACDC vs M&M-2 as EF-bias source).
**Verdict:** convention hypothesis **rejected**. The −5.8% EF bias is systematic **end-systolic
cavity over-segmentation** by the model, magnitude ~fixed in absolute mL → EF bias scales inversely
with cavity size. Model-wide (both domains), not a cross-dataset convention mismatch.

## Evidence (runs/mnm2_150, no re-inference — decomposed from saved EF rows)

Decompose EF=(EDV−ESV)/EDV into its two volumes:

| set | EDV pred/gt | ESV pred/gt | EF bias |
|---|---|---|---|
| M&M-2 val (in-domain) | 1.06 | **1.24** | −7.0 |
| ACDC-150 (cross-domain) | **1.01** | **1.19** | −5.8 |

- **EDV is accurate** (ACDC ratio 1.01). ED cavity convention matches ACDC↔M&M-2 — consistent with
  the 2026-06-20 finding that ACDC/M&Ms-1/M&Ms-2 share the LV-cav convention (papillaries +
  trabeculae **included**). So this is NOT a convention bug; `measure.py` computes volumes correctly.
- **ESV is over-predicted ~19–24%** in both domains. Over-filling ESV alone drags EF down — it is the
  entire bias.

## Mechanism: fixed-absolute over-seg → inverse size scaling (ACDC-150)

| ES cavity tercile | esv_gt | ESV ratio | abs mL bias |
|---|---|---|---|
| smallest ⅓ | 33 mL | **1.41** | +12.3 |
| middle ⅓ | 69 mL | 1.16 | +10.6 |
| largest ⅓ | 195 mL | **1.01** | −1.0 |

corr(esv_gt, ESV ratio) = **−0.50**. Per-pathology confirms:

| pathology | ESV ratio | EF bias |
|---|---|---|
| DCM (huge dilated cavity) | 0.99 | **−0.6** (unbiased) |
| MINF | 1.07 | −3.9 |
| RV | 1.14 | −6.2 |
| NOR | 1.24 | −7.3 |
| HCM (tiny thick-walled cavity) | **1.51** | **−10.9** |

The over-segmentation is roughly a **fixed number of mL** at the cavity boundary (partial-volume
voxels + papillary muscles bulging into the small contracted ES cavity, mislabeled blood pool).
That's negligible vs a large cavity (DCM → unbiased) and dominant vs a small one (HCM → −10.9). This
is the mechanism behind the long-noted "HCM EF outlier."

## Consequences for the fixes

- **`measure.py`: no fix.** Volumes are correct; the segmentation over-fills ES.
- **EF bias subtraction (`lnd`): wrong by construction.** The error is cavity-size-dependent, not a
  constant offset — a constant subtraction would over-correct DCM (already unbiased) and under-correct
  HCM. Plus the DG-leakage argument (subtracting test-measured bias = leakage; a held-out calibrator
  is domain-shifted). Report bias+LoA honestly (model card); any calibration is an explicitly-fenced
  domain-adaptation demo, never folded into the zero-shot number.
- **The real lever is segmentation-side at the ES small cavity** (new task): boundary-aware loss (the
  over-seg is a boundary/partial-volume phenomenon), harder small-structure weighting, or explicit
  papillary handling at ES. Separate model-improvement work, not a calibration patch.

## Repro
`runs/mnm2_150/metrics.json` → decompose ef_rows: esv = edv·(1−ef/100); ESV ratio = esv_pred/esv_gt.
