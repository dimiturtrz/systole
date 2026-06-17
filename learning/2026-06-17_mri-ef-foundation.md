# Cardiac MRI & ejection fraction — the foundation

*Topic 1 · MRI lane · theory. Status: foundations (internal knowledge); specifics
flagged below to be research-grounded.*

## Why EF (the clinical question)
The heart is a pump. Each beat: the ventricle fills (**diastole**), then contracts
and ejects (**systole**). **Ejection fraction = the fraction of the filled volume
pumped out per beat.** Left-ventricular EF is *the* headline number for systolic
heart function — low EF defines heart failure with reduced EF (HFrEF). So
"segment heart → measure EF" is a clinically real, valued task, not a toy.

$$EF = \frac{EDV - ESV}{EDV} \times 100\%$$

- **EDV** = end-diastolic volume — fullest, just before contraction.
- **ESV** = end-systolic volume — emptiest, end of contraction.
- Normal LV EF ≈ 50–70%; below ~40% = reduced. *(⚠ rough range — exact guideline
  cutoffs to verify before quoting clinically.)*

## The anatomy you segment
Four chambers: two atria (top, fill) and two ventricles (bottom, pump). EF concerns
the ventricles. ACDC labels exactly the ventricular-function structures:

| label | structure | role |
|---|---|---|
| 0 | background | — |
| 1 | **LV blood pool** (cavity) | its volume → EF |
| 2 | **myocardium** (LV muscle wall) | thickness / mass |
| 3 | **RV blood pool** (cavity) | right-ventricle function |

Atria aren't labeled — EF is ventricular, so they aren't needed.

## The chain to EF
EF = how much blood the LV ejects → need the LV **cavity** (blood) volume at two
time points. Segment label `1` on the **ED frame** and the **ES frame**, count
voxels, × physical voxel volume (mm³ → mL) → EDV, ESV → EF. That whole chain is the
pipeline; it's why label `1` is the star.

## To research-ground next
- Exact clinical EF thresholds (normal / mildly–severely reduced) per current guideline.
- Cine MRI acquisition specifics (short-axis stack, slice thickness/gap, ED/ES via `Info.cfg`).
- Why MR intensity is uncalibrated (→ per-volume normalization).

---

## Quiz log
*(pending — run when ready: say "quiz me")*
