# B1 · Cardiac anatomy & the cardiac cycle

What you must recognize in the images and why it matters for measuring function.

## The four chambers
- **Atria** (left, right) — top; thin-walled; *fill* the ventricles. **Not segmented**
  in short-axis cardiac function work — EF is a *ventricular* measure.
- **Ventricles** (left, right) — bottom; the pumps:
  - **Left ventricle (LV)** — thick wall, ~circular in short-axis; pumps oxygenated
    blood to the **body** (high pressure). The LV cavity volume drives **EF**.
  - **Right ventricle (RV)** — thin wall (~3–5 mm), crescent-shaped in short-axis;
    pumps to the **lungs** (low pressure).
- **Myocardium** — the heart muscle wall. In this work it means the **LV muscle wall**
  (≈8–12 mm at end-diastole, thicker at end-systole). RV wall is too thin to label
  reliably at standard resolution.

## The cardiac cycle
The heart alternates:
- **Diastole** — relaxation + filling → ventricle reaches **maximum** volume.
- **Systole** — contraction + ejection (starts at the ECG R-wave) → ventricle reaches
  **minimum** volume, ~300–500 ms after the R-wave.

Two frames matter for function:
- **End-diastole (ED)** — fullest cavity, largest area; ≈ first frame after the R-wave.
- **End-systole (ES)** — emptiest cavity, thickest myocardium.

EF compares the LV cavity volume at **ED** vs **ES** (see
[ejection-fraction.md](ejection-fraction.md)).

## The labels you segment (ACDC convention)
| Label | Structure | Note |
|---|---|---|
| 0 | background | includes RV wall, pericardium, lungs, etc. |
| 1 | **RV cavity** | |
| 2 | **LV myocardium** | the muscle wall |
| 3 | **LV cavity** | its volume → EF |

> ⚠ **Verify before trusting.** `0=bg, 1=RV, 2=LV-myo, 3=LV-cavity` is the **community
> convention** (published ACDC papers + code), **not** confirmed from an official
> numeric source in our research. **Confirm with `np.unique()` on a real ground-truth
> NIfTI at EDA.** Also note: the repo's `synth.py` currently uses a *different* mapping
> (`1=LV…`) — flagged to fix at the code phase. Don't hard-code label meaning from
> memory; read it from the data.

## Papillary muscles (a measurement gotcha)
Small muscles protruding from the LV wall into the cavity. **ACDC/CMR convention
includes them in the LV cavity (label 3)**, with the trabeculations. They're ~12% of
EDV — excluding them inflates LV volume ~12% and raises EF. **Be consistent across ED
and ES, and match the dataset's convention.** [ECR Journal]
