# 06 · Cardiac MRI (A2) — imaging a moving organ

The MRI-specific bridge from generic physics (01–05) to the heart. Core problem:
**the heart beats and the chest breathes** — motion. The fixes define cardiac MRI.

## The sequence: bSSFP cine
- **bSSFP** = balanced Steady-State Free Precession (vendor names: TrueFISP / FIESTA /
  balanced-FFE). A **gradient-echo** variant where gradient moments are balanced (zeroed)
  each TR → signal ∝ **T2/T1**, giving **bright blood / dark myocardium** = crisp
  endocardial borders.
- **Fast:** TR/TE ≈ **4/2 ms** → temporal resolution ~**30–50 ms** per cardiac phase.
- **Cine** = many frames across one heartbeat → a movie; you pick **ED** and **ES** from it.

## Acquisition geometry
- **Short-axis stack:** prescribed perpendicular to the LV long axis (using long-axis
  4-/2-chamber views to set the mitral-valve plane and apex), base → apex,
  **10–15 slices** for whole-heart coverage.
- **Resolution:** in-plane ~**1.25–1.7 mm**, slice **5–8 mm** (ACDC: mostly 5 mm,
  sometimes 8 mm; ~5 mm gaps) → **anisotropy ~1:5 to 1:7**.
- **Consequence:** that anisotropy is *the* reason **2D models dominate** (see
  [../common/M_segmentation-theory.md](../common/M_segmentation-theory.md)).

## ECG gating — exploit the periodic beat
You can't freeze the heart in one acquisition, so you **synchronize to the ECG** and
stitch across beats.
- **Segmented k-space:** acquire ~**10–16 phase-encode lines per heartbeat** for a given
  cardiac phase; fill one slice's k-space over ~**10–16 beats** ≈ a **10–16 s breath-hold**
  per slice.
- **Prospective gating:** trigger on the R-wave, acquire a fixed window after it. Can
  **miss late end-diastole** (gap before the next R-wave). Safer for arrhythmia.
- **Retrospective gating:** acquire continuously + record ECG; afterward **sort lines
  into phases by timing within each R-R interval**. Covers the full cycle (incl. ED) →
  **preferred for EF**, and copes with **variable rhythm** (bins by fraction of the beat).
- Whole cine stack exam ≈ **15–20 min** (slice-by-slice breath-holds).

## ED / ES in the data
- **ED** = largest LV cavity (≈ first frame after R-wave). **ES** = smallest cavity
  (~300–400 ms post-R-wave), thickest myocardium.
- **ACDC annotates both ED and ES** frames with ground-truth masks, and `Info.cfg` gives
  the frame indices (see [08_acdc-dataset.md](08_acdc-dataset.md)).

## When it breaks (→ artifacts)
- **Arrhythmia** breaks the "every beat is the same" assumption → ghosting; use
  arrhythmia rejection or **real-time MRI** (faster + undersampled, lower res).
- bSSFP brings its own artifact — **off-resonance dark banding** — and motion/flow
  issues. Covered in [07_artifacts.md](07_artifacts.md).

*Application takeaway:* you need bSSFP-cine + gating only to the depth that explains the
**data you'll load** (ED/ES frames, short-axis stack, anisotropy, gating-related motion)
— not to design sequences. The rest of the sequence zoo is out of scope (see
[../field-map.md](../field-map.md)).
