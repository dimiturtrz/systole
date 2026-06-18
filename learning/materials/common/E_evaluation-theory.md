# Evaluation theory (medical segmentation)

**This is the project's point.** A measurement is only as good as the evidence it holds
up under. Mean Dice is *not* enough for a clinical number.

## Overlap metrics
**Dice (DSC)** `= 2|P∩G| / (|P|+|G|)`, range [0,1]. ACDC SOTA ≈ LV 0.93–0.96, myo
0.88–0.92, RV 0.88–0.92. *(reported; verify against the leaderboard.)*
**Jaccard / IoU** `= |P∩G|/|P∪G| = DSC/(2−DSC)` — monotone with Dice; rarely reported separately.

## Surface / boundary metrics
**Hausdorff distance (HD)** = worst-case surface error (max of mins, both directions).
Very **outlier-sensitive** — one stray voxel dominates.
**HD95** = 95th percentile (drops the worst 5%). The medical standard; report in **mm**
(good methods ~2–5 mm).
**ASSD / MASD** = mean symmetric surface distance (mm) — sensitive to **systematic**
boundary offsets that Dice hides.

## Why Dice alone misleads for clinical EF
1. **Scale-normalized:** Dice 0.95 on a big heart can still hide 5–10 mL volume error =
   3–5% EF error on a *small* heart.
2. **Doesn't decompose by phase:** high mean Dice but systematic **ES** under-segmentation
   → biased EF even if ED is perfect.
3. **Blind to smooth offsets:** a contour shifted 1 mm inward everywhere keeps Dice >0.9
   but biases volume ~5–10 mL.

→ So evaluate the **clinical quantity directly**: predicted EDV/ESV/EF vs ground truth.

## Volume & EF agreement — use Bland–Altman
For EDV, ESV, SV, EF (per patient):
- Plot **(prediction − reference)** vs their mean.
- Report **bias** (mean difference) and **limits of agreement** (bias ± 1.96·SD).
- Rough clinical-equivalence targets: EF bias < 2–3%, LoA within ±5–8%.
- **MAE** (mL for volumes, % for EF) = the simple scalar to headline.
- **Pearson r** alone is **misleading** (high r coexists with large bias) — prefer Bland–Altman.

## Calibration / uncertainty (good practice, not required for ACDC)
MC-dropout or ensemble spread → per-pixel uncertainty → flag unreliable cases (clinical
QC). Worth a mention/figure if time.

## Failure analysis — the senior signal
- Rank cases **worst-first** (lowest Dice / largest EF error). (`evaluate.rank_failures`.)
- Break down by **ACDC pathology group** (NOR/MINF/DCM/HCM/ARV ×30): HCM = thick wall,
  hard RV/LV boundary; ARV = thin RV, heavy partial-volume; MINF = scar alters texture.
- Report: failure histogram, worst-5% with **why** (which slices/artifacts), not just a
  mean. *This* is what separates a demo from a serious eval.

## The clinical-grade gap (state it honestly)
- **Domain shift:** ACDC = 1 centre, 2 field strengths. Cross-vendor/scanner → Dice
  drops ~5–15% absolute. The M&M (Multi-centre Multi-vendor) challenge tests this.
- A demo ≠ clinical tool: needs prospective validation, multi-rater ground truth, edge
  cases (devices, congenital, post-surgical), regulatory clearance, audit trail. Say
  what the numbers do **not** guarantee.
