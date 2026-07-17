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

| Arm | envelope | GE-holdout (n=35) | Canon (n=9) | Siemens (n=522) |
|-----|----------|-----------|-------|---------|
| generic-composition | default (noise 0.05, blur 0–1, white) | **0.542** | 0.434 | **0.531** |
| directed | blur 1.0, noise 0.035, band-limited | **0.516** (−0.026) | 0.462 (+0.028) | **0.479** (−0.052) |

(zero-real, quick 40ep, seed 0; per-class GE generic RV/myo/cav 0.48/0.49/0.66 → directed 0.40/0.53/0.61.)

**Verdict — directed lost. The tax is not global-spectral; the envelope knobs cost aug-robustness.**

The pre-registered **directed < generic** branch fired. Directed **regressed the fitted target GE** (−0.026)
and — the reliable signal — **regressed the largest, least-noisy test Siemens by −0.052** (n=522, well above
the ±0.02–0.03 single-seed cross-vendor floor). The only uptick, Canon +0.028, is an **unfitted** vendor at
**n=9** — inside the noise, not the mechanism (if envelope-matching worked it would show on GE, the vendor it
was fit to; it didn't).

Mechanism: the fitted envelope (blur σ 1.0 up, noise 0.035 + band-limited down/colored) makes synth
**cleaner and less broadly augmented**. That is exactly the `nk70.1` failure mode (dropping the white-noise
texture collapsed Dice because it was load-bearing *augmentation*, not fidelity) — here in milder form. The
model needs the broadband white-noise + blur-sweep **diversity** more than it needs per-image spectral
realism. Matching the PSD improved a cosmetic input statistic (2.88 → 0.19) and moved Dice the **wrong way**.

This **converges with `nk70` and `xmcf`**: per-image input-distribution realism does not predict Dice
(nk70.1's core finding), and the ~0.17 randomization tax is **not recoverable by matching the target's global
spectral envelope** — it is structural (the domain-randomization → invariant-cue-only trade is by design;
`what_is_wrong_with_our_synth.md`), and forcing target-fidelity trades away the diversity that makes zero-real
synth generalize at all. Envelope-directed DA is the **wrong lever**.

Scope/caveats: single-seed, quick — but the direction is consistent across the two informative vendors (GE
flat-negative on the target, Siemens clearly negative on the biggest n), and the mechanism is the
independently-established nk70 one, so no multi-seed is warranted to *confirm a negative* (owner's stage-gated
rigor: don't harden a lever that doesn't work). Slice-vs-subject partition granularity (§3 fit was slice-level,
eval subject-level) is immaterial — the envelope is fit to GE's *vendor* spectral signature, robust to which
half.

## What is kept vs killed

- **KEPT** — `directed.py` operator + `radial_psd`/`psd_distance` (correct, reusable, unit-tested); it *did*
  match the spectrum (6i8g). The tool is sound; the **hypothesis** it tested is what failed.
- **KILLED** — envelope-matching as a DA lever for zero-real Dice. Do not fit blur/noise/k-space to a target
  spectrum expecting a Dice gain; it costs aug-robustness.
- **Still open under `ncph`** (different value props, not refuted here): the twin *demo* (fit to one labeled
  scan → controlled-fidelity artifact), and harmonization. Both are fidelity/product goals, not Dice levers —
  and this result says a Dice lever is not where the inverse direction pays off.
