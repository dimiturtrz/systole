# Directed synth — inverse-rendering domain adaptation vs the randomization tax

**Date:** 2026-07-17 · **Scope:** cross-task (zero-real generation lane, inverse direction) · **Beads:**
cardiac-seg-ncph / 6i8g / xvx0 / n25b · **Status:** operator done + validated; A/B in progress (results §4)

Tests whether the ~0.17 **randomization tax** (the color gap of zero-real synth vs real; bd xmcf,
`2026-07-16_what_is_wrong_with_our_synth.md`) is **recoverable** by pointing the generator at a specific
deployment vendor instead of randomizing blindly. The inverse direction's live branch — see below for why
this is *not* the identifiability-blocked branch.

## Why this is the live inverse branch (and param-recovery is dead)

`ncph` began as "fit the generator's physics params to a real scan" (a qMRI digital twin). `ixea` + `5ev5`
(`2026-07-16_twin_tissue_identifiability.md`) proved that **dead on standard cine**: the heart has two tissue
levels (blood, myo), uncalibrated MRI leaves a free affine gain, and two levels map onto two levels under any
acquisition — so the params are degenerate, and the fit returns the literature prior.

But the degeneracy is on the **params, not the appearance** — many param sets render the same image. If the
goal is a *directed generator* (not a measurement), we don't need the params; we fit the **observable
envelope** and let the physics prior set the (leak-free) heart contrast. That sidesteps the wall entirely.

## Method — fit the spectral envelope, hold contrast at the prior

`core/data/analysis/directed.py` (`python -m core.data directed --vendor GE`):

1. Load a target vendor's **unlabeled** real images; take a seeded fit partition (rest held out for the A/B).
2. Compute the whole-FOV **radial power spectrum** (rotationally-averaged |FFT|², normalized to a shape).
3. Grid-search the leak-free envelope knobs — blur σ, k-space keep, Rician noise std, and noise-band-limiting
   (color the noise before the low-pass vs white) — to minimize log-PSD distance to the target.
4. Heart **contrast stays at the physics prior** — no our-data statistics enter the tissue model. The only
   thing fit is the acquisition envelope, from the target's *image* spectrum (no masks) — the defining input
   of unsupervised domain adaptation, not a label leak.

The z-score in the painter normalizes absolute level, so the envelope that actually reaches the model is
exactly the spectral shape (bg histogram + resolution + noise) — the xmcf suspects.

## §3 — envelope fit result (bd 6i8g, done)

**GE** (n_fit 585 slices / n_holdout 586, seed 0): generic synth PSD-distance **2.88 → directed 0.19** (Δ2.69).
Best envelope: **blur σ 1.0** (up from the generic (0,1) sweep), **noise 0.035** (down from 0.05),
**band-limited = True** (colored, added before the low-pass). This is exactly the xmcf prediction: generic
synth carries a flat **white** high-frequency plateau (noise added in the image domain after blur/k-space);
the fit removes it by recoloring + blurring toward the target's roll-off. The fit found the physically-sensible
correction, not a pathological one.

Caveat: PSD match is **necessary, not sufficient**. `nk70.1` showed the noise knob is load-bearing
augmentation (dropping it to physical SNR collapsed Dice). The directed envelope only mildly lowers noise
(0.05→0.035) and recolors it — not the nk70 collapse — but the training A/B is the real verdict.

## §4 — the 3-point A/B (bd xvx0, in progress)

Zero-real arms on `synth_main` (Rodero pool, procedural bg), quick (40ep), seed 0. Evaluated per-vendor with
the pipeline's per-volume + largest-CC + TTA path (`scratchpad/directed_ab_eval.py`); the fitted target (GE)
is scored **only on its seed-0 subject-level holdout half**, disjoint from the fit slices.

Reference anchors (real-trained `production`, same holdout eval): GE **0.835**, Canon **0.836**.

| Arm | envelope | GE-holdout | Canon | Siemens |
|-----|----------|-----------|-------|---------|
| generic-composition | default (noise 0.05, blur 0–1, white) | _pending_ | _pending_ | _pending_ |
| directed | blur 1.0, noise 0.035, band-limited | _pending_ | _pending_ | _pending_ |

**Interpretation grid (decided up front):**
- **directed > generic on GE, ≤ on Canon/Siemens** → the randomization tax is real and **partly recoverable**;
  directed synth trades diversity for target-fidelity. Closes the xmcf question from the inverse side.
- **directed ≈ generic (flat)** → the tax is **not global-spectral** — it's structural (spatial/boundary),
  not envelope. Matching the PSD isn't enough; corroborates the xmcf "not fixable by global stats" read.
- **directed < generic** → target-fitting the envelope **overfits** and loses the aug robustness (nk70
  noise-is-load-bearing) → directed generation is the wrong lever here.

_Verdict to be written once §4 lands._
