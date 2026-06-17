# 08 · The ACDC dataset (C1)

What you'll actually load. **ACDC** = Automated Cardiac Diagnosis Challenge (MICCAI
2017; Bernard et al., IEEE TMI 2018).

## Size & split
- **150 patients**: **100 training**, **50 test** (held out, server-scored).
- **5 pathology groups, 30 each** (20 train / 10 test per group):
  - **NOR** normal · **MINF** prior MI (scar) · **DCM** dilated CM (big LV) ·
    **HCM** hypertrophic CM (thick myo) · **ARV** abnormal RV.
- The pathology split is gold for **failure analysis** (per-group Dice/EF breakdown).

## Per-patient files (NIfTI)
```
patientXXX/
  patientXXX_4d.nii.gz          # full 4D cine (H × W × slices × time)
  patientXXX_frameED.nii.gz     # 3D volume at end-diastole
  patientXXX_frameED_gt.nii.gz  # ED ground-truth mask
  patientXXX_frameES.nii.gz     # 3D volume at end-systole
  patientXXX_frameES_gt.nii.gz  # ES ground-truth mask
  Info.cfg                      # ED/ES frame indices, Group, Height, Weight, NbFrame
```
Only **ED and ES** frames are annotated (the two you need for EF).

## Labels — VERIFY at EDA
Community convention:
```
0 = background   1 = RV cavity   2 = LV myocardium   3 = LV cavity
```
> ⚠ **Not confirmed from an official numeric source** in our research (inferred from
> published papers/code). **First EDA step: `np.unique()` on a `*_gt.nii.gz`** and
> cross-check the challenge eval script. Also: the repo's `synth.py` currently uses a
> **different** mapping (`1=LV…`) — **fix at the code phase** once verified.
> RV myocardium is **not** labeled (folded into background).

## Geometry
- In-plane spacing ~**1.4–1.7 mm** (median ~1.5); slice **5–10 mm** (median ~10) →
  strong **anisotropy** → 2D models (see [../common/segmentation-theory.md](../common/segmentation-theory.md)).
- **Read spacing from the NIfTI header**, per volume — don't assume.

## Loading notes (for Phase D)
- NIfTI axis order is x,y,z; our `data.load_nifti` transposes to **(z,y,x) = (D,H,W)**
  and returns spacing as (z,y,x) mm. Keep that convention end-to-end.
- **Normalize per-volume** (MR intensity uncalibrated) — z-score or percentile-clip.
- Optionally **resample** to a consistent in-plane spacing (nnU-Net targets ~1.56 mm).
- **Patient-level splits only** (no slice leakage).

## Getting it
Register at **Creatis / humanheart-project**. Data stays **outside the repo** (licensing
+ size); point the loader via `CARDIAC_DATA_ROOT` (e.g. `D:/data/volumetric/mri/acdc`).
Fallback if blocked: MSD `Task02_Heart` (LA-only, **no EF** — breaks the EF story).

## SOTA to benchmark against
nnU-Net (2D default on ACDC): mean Dice ~**0.92** (LV ~0.95, myo ~0.89, RV ~0.90).
*(Reported in prior research; confirm on the official leaderboard.)* The aim isn't to
beat this — it's a **reasonable Dice + honest EF agreement + failure analysis**.
