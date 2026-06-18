# B2 · Ejection fraction — what & why

EF is the deliverable. It's the headline marker of systolic heart function, and cardiac
MRI is its volumetric **gold standard** (measures real 3D volume from the slice stack,
no geometric shape assumption — unlike echo).

## The formula
```
SV (stroke volume) = EDV − ESV
EF (%) = SV / EDV × 100 = (EDV − ESV) / EDV × 100
```
- **EDV** = end-diastolic volume (fullest), mL
- **ESV** = end-systolic volume (emptiest), mL
- Computed from the **LV cavity** (label 3) volume at the **ED** and **ES** frames.

The pipeline: segment LV cavity at ED and ES → count voxels → convert to mL (see
[G_geometry-and-volumetry.md](G_geometry-and-volumetry.md)) → EF.

## Normal ranges & clinical thresholds (LV)
| | LVEF |
|---|---|
| Normal | ~**52–74%** (≈52–72% M, 54–74% F; population-dependent) |
| Mildly reduced | 41–49% |
| Reduced | **< 40%** |

Heart-failure categories (2022 AHA/ACC/HFSA):
- **HFrEF** (reduced) — EF < 40%
- **HFmrEF** (mildly reduced) — EF 41–49%
- **HFpEF** (preserved) — EF ≥ 50% (with diastolic dysfunction)

*(Numbers ≈ consistent with the earlier foundations deep-dive; exact lower-bound of
"normal" varies by population/method — treat as reference, not gospel.)*

## Why EF accuracy is subtle (the key point for ML)
EF is a **ratio of two volumes**, both estimated from segmentations:
- A small **systematic boundary offset** biases EDV and ESV → biases EF, even at high
  Dice. (See [E_evaluation-theory.md](E_evaluation-theory.md): why Dice alone misleads.)
- Errors at **ES** matter disproportionately (small cavity → small absolute errors are
  large relative errors).
- A few mL of volume error = several % EF error in **small hearts** (HCM, low-EDV HFrEF).
- An EF MAE of ~5% can flip a patient's **HFrEF↔HFmrEF** category — clinically real.

So the project's measure of success isn't mean Dice — it's **predicted EF vs ground-
truth EF** (bias + limits of agreement), plus where it fails.

## Related metrics (know they exist)
- **Cardiac output** = SV × heart rate (needs HR from ECG); normal 4–8 L/min.
- **LV mass** = LV-myocardium volume × 1.05 g/mL (from label 2's region).
- **Strain** (regional deformation, ~−20% normal) — needs dedicated sequences/feature
  tracking, **not** computed from standard cine segmentation. Out of scope; worth
  knowing it exists.
