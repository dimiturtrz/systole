# The −0.17 repaint color gap: a spectrum gap, most likely the randomization tax (xmcf)

> **Read-after note.** First drafted claiming the gap "irreducible." That is over-claimed — see the Verdict and
> "What is actually owed" sections. The honest state: physics-randomized *appearance* moves are exhausted, but
> the *why* points to the domain-randomization tax (synthesis doc), and the model's own errors were never
> examined. Kept the spectrum measurements; retracted the verdict.

**Date:** 2026-07-16 · **Lane:** color-fault attribution (xmcf) · **Follows:** nk70 (appearance-correctness refuted)

## Question

Repaint (real masks + synth color) tops out at **0.68** cross-vendor; real-trained is **0.854**. With shape
held perfect, the residual **−0.17 is pure COLOR**. nk70 attacked color via *levels + within-class spread*
(d′ / per-class σ) and proved it Dice-dead. This epic used the right compass — **Dice arms + input-distribution
measurements**, never fidelity metrics — to attribute that −0.17 to a specific mechanism, or prove it
irreducible. Three suspects: (1) preprocessing asymmetry, (2) heart texture/frequency spectrum, (3) per-image
z-score coupling. Stage-gated: free inspection → one no-training probe (the gate) → training arms only for
suspects that pass.

## Phase 0 — the preprocessing suspect is mostly a phantom

Traced both recipes (`core/preprocessing/preprocess.py`, `core/data/dynamic/synth.py`):

| step | real-test | synth-train | asymmetry |
|---|---|---|---|
| resample 1.5 mm in-plane | yes (interp; ACDC ≈1.5 → near-identity) | born on grid | minor (synth blur/k-space covers) |
| N4 bias | **off by default** | — | none |
| Nyúl histogram | **off by default** | — | none |
| z-score granularity | **per-VOLUME** (all slices share μ/σ) | **per-SLICE** | **real, see #3** |
| fit_square 256 | yes | yes | none |

The "real = resample + [N4] + [Nyúl] + z-score" framing over-stated it: the brackets are **off** in the
default zero-real lane. The only genuine preprocessing asymmetry is **z-score granularity** — which is
suspect #3, not #1. Suspect #1 folds away; Arm 1 has nothing to symmetrize.

## Phase 1 (gate) — the input gap is the SPECTRUM

One no-training probe (`scratchpad/xmcf_probe.py`): synth painted on real masks (repaint-style, procedural bg,
per-slice z-score) vs real-test slices (per-volume z-score). Three reads:

**(a) Histogram.** heart-region pixel W1 0.49 > whole-FOV 0.24. Synth heavier-tailed (more dark + bright).
Per-class levels: blood too bright (RV 2.04 vs 1.55, cav 2.05 vs 1.65) — the known fi33 level gap, already
Dice-dead.

**(b) Radial PSD — the smoking gun.** 2D-FFT azimuthal average, log₁₀ power:

| k / Nyq | real | synth (white) | Δ |
|---|---|---|---|
| 0.10 | +5.51 | +5.04 | −0.47 |
| 0.25 | +4.26 | +4.76 | **+0.50** |
| 0.50 | +3.15 | +4.74 | **+1.59** |
| 0.75 | +2.01 | +4.73 | **+2.72** |
| 1.00 | +1.39 | +4.73 | **+3.33** |

Real rolls off **six decades** (natural-image colored spectrum). Synth drops, then hits a **dead-flat white
plateau (log₁₀P ≈ 4.73)** and sits **10¹·⁶–10³·³× above real** across the whole mid/high band. That plateau is
`noise=0.05`: **white** Rician added in the **image domain, after** the resolution-limiting blur/k-space — so
it carries power *above the acquisition band*, which real noise cannot. Real texture is spectrally colored;
synth's is white.

**(c) z-coupling.** per-slice mean: real +0.016 ± **0.068**, synth 0 ± 0; per-slice std real 0.864 ± 0.131 vs
synth 1.0. The per-volume-vs-per-slice asymmetry is real but **small** — fails the gate, no arm.

**Gate verdict:** #1 dead, **#2 spectrum = big**, #3 too small. One arm survives.

## Arm 2 — physically band-limiting the noise does not close the gap

The fault is physical and leak-free to fix: real MRI noise is white *in k-space* and band-limited by the same
acquisition window as the signal — it should not live above the signal's band. Added an opt-in
`noise_bandlimited` (inject Rician **before** blur/k-space, so the resolution op co-limits it like the signal),
run zero-real with `noise_bandlimited=true kspace=0.85` (physical: ACDC acq matrix ≈227 over the 256 grid ≈0.89
Nyq). Constant energy, color derived from **acquisition physics** — not fitted to real's PSD (that would be the
blood_scale/d′-target soft leak).

Probe first (no training): the band-limited variant zeroes the **>0.85 Nyq tail** (Δ at Nyquist −9.67) but the
**mid-band plateau 0.25–0.75 Nyq is unchanged** (Δ +0.65/+1.68/+2.22 vs white +0.50/+1.59/+2.72) — z-score
renormalizes the removed tail energy straight back into the mid-band.

Training arm (`runs/xmcf_arm2`, seed0, --quick):

| zero-real | baseline (white) | Arm 2 (band-limited + kspace) | Δ |
|---|---|---|---|
| VAL mean | 0.676 | 0.650 | −0.026 |
| TEST mean | 0.613 | **0.602** | −0.011 (at noise floor) |

**Dice-flat.** Trimming the unphysical super-band tail neither collapsed (nk70's energy-drop did → 0.385) nor
helped. The load-bearing part is the **mid-band plateau**, and it stays.

## Verdict — irreducible, and why

The spectrum gap is **co-bound to the load-bearing augmentation**. The white-noise plateau is simultaneously:
- the **single biggest input-distribution gap** (10²× real across the mid-band), and
- **load-bearing for generalization** (nk70.1: lowering its energy toward real's roll-off collapses zero-real
  Dice 0.613 → 0.385).

Matching real's colored roll-off *means* removing mid-band energy — which is exactly the collapse. Band-limiting
at held energy (the only leak-free move that keeps the energy) leaves the mid-band untouched → Dice-flat. There
is no configuration of the physical noise model that closes the spectral gap without killing the augmentation.

All three suspects are ruled out as Dice levers **within the moves tried**. Stronger caveat added after review
(owner pushback): "irreducible" is over-claimed. It rests on ONE arm (band-limit, confounded with a k-space PSF
on the signal) plus the nk70 noise-lowering collapse — that is *"the levers I tried failed,"* not a proof of
impossibility. And the whole epic stayed in the **input-distribution-statistics** frame (PSD / histograms /
z-means) — the exact fidelity-metric frame nk70.1 showed does **not** predict Dice. We kept matching the
generator's output distribution and never once looked at the **model's actual errors**. See the cross-task
synthesis `interpretations/converging/2026-07-16_what_is_wrong_with_our_synth.md` for the reframe: the 0.17 is
most likely the **domain-randomization tax** (a stable intensity→tissue mapping the test shares with real train
and randomization forfeits by design), of which the spectrum gap is a *symptom*, not the cause.

## What is actually owed (not done here)

- **Failure analysis of the 0.68 repaint model** — per-slice / per-structure / per-vendor Dice + the 20 worst
  overlays. Never done. Decides whether 0.17 is a uniform low-contrast-boundary loss (the randomization tax) or
  a targetable defect (a vendor / slice-band / structure).
- **A truly-OOD test** (a domain absent from the real train set) — the discriminating experiment for the
  stable-mapping hypothesis. If the real↔synth gap narrows OOD, the 0.17 is the shared mapping, not a synth
  fault. Both experiments are in the synthesis doc.

The pivot to **learned / inverse color** (adversarial appearance matching, or the digital-twin fit-to-scan) is a
*candidate*, not a conclusion — and it was reaching past what we understand. Do the failure analysis + OOD test
first; they are cheaper and they test the mechanism before another generator change.

## Kept infrastructure

`noise_bandlimited` (default **off**) stays: it is the *physically-correct* noise model (acquisition-band-limited,
unlike the default image-domain white), valuable for the tight-fidelity twin direction even though it is a
Dice-neutral diversity knob — same disposition as trabecular-PV (fi33). Gate-clean (ruff / ast-grep).

## Honesty

Single-seed arm, but the decisive deltas across this and nk70 (−0.011 here, −0.23 nk70) bracket the conclusion
well past the ±0.01 noise floor and agree with the no-training PSD probe. This is a negative result reported
faithfully: the color axis is now *attributed* (spectrum), not just exhausted, and the reason it is irreducible
(aug/fidelity tension) is mechanistic, not a shrug. Arm 2 confounds band-limit with a k-space PSF on the signal;
the probe shows isolating band-limit alone (kspace=0) would barely color the noise, so the keep/kill is safe on
one arm per stage-gating.
