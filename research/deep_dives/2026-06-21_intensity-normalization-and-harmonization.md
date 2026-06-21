# Intensity normalization & cross-scanner harmonization — what's worth doing

**Date**: 2026-06-21
**Status**: grounded (web-cited; reasoning is ours)
**Why**: decide what to build for `qfz` (N4 / Nyúl) and how hard to lean on it. Companion to the
design in [`cardioseg/normalization/README.md`](../../cardioseg/normalization/README.md) — that's
the *architecture*; this is the *evidence + the honest "is it worth it" call*.

## TL;DR
- **MRI intensity is uncalibrated** → some standardization is forced (no absolute scale).
- **The strongest cross-scanner lever is multi-centre TRAINING — not preprocessing.** Lit: data
  augmentation and image harmonization give "similar *limited* generalization"; multi-centre
  training + transfer learning improve significantly more. **We already train multi-vendor
  (M&M-2)** → we're already on the strongest lever. Nyúl/N4 are incremental on top.
- **Nyúl is a classic robust baseline (1999), not SOTA.** Results are mixed: some studies show it
  beats z-score (Dice 84% vs 78%); others recommend z-score for first-order-feature robustness.
  So it *can* help — but it's dataset-dependent, not a guaranteed win.
- **Augmentation often matches explicit harmonization** and is more generalization-honest (learns
  invariance, bakes no train-derived reference into preprocessing). nnU-Net wins with z-score +
  heavy aug and *no* explicit harmonization.
- **Honest limitation:** harmonization to a reference is "bound by the least informative scanner"
  in the fit pool, and forces unseen scanners onto *our* reference (adaptation flavor; can distort
  a genuinely-different histogram). → **Validate, don't assume.**

## The methods (intensity family)

| method | knobs | idea | nature |
|---|---|---|---|
| **z-score** | 2 (mean, std) | per-scan, center+scale | assumption-free, per-scan |
| **WhiteStripe** | 2, on a reference tissue | z-score using normal-appearing white matter as anchor | per-scan, tissue-anchored |
| **Nyúl–Udupa** | ~10 landmarks | match histogram percentiles to a train-fit standard ("interlingua" hub) | per-scan apply, **train-fit reference** |
| **N4 (bias field)** | — | estimate + divide out the smooth coil-inhomogeneity field | per-scan, *spatial* (different axis) |
| **ComBat / ComBat-GAM** | per-site location/scale | harmonize *features* across known sites | retrospective, needs site labels |
| **Learned (CycleGAN / style-blind AE / diffusion)** | network | translate one scanner's appearance to another | frontier, heavier, can hallucinate |

N4 is orthogonal (spatial bias, not global scale) → applies *before* any of the global ones.
Nyúl/z-score/WhiteStripe occupy the same slot (pick one). ComBat needs the site key we usually
lack at inference. Learned methods are the research frontier (and risk inventing anatomy).

## The generalization ↔ adaptation spectrum
```
assumption-free  ───────────────────────────────►  adaptation
per-scan z-score → intensity augmentation → Nyúl(train-ref) → ComBat/test-time-BN
```
- z-score & augmentation make no cross-population assumption → generalization-pure.
- Nyúl maps each scan to a *train-derived* standard → per-scan apply but train-biased reference.
- For a project whose thesis IS generalization, the lever order should respect this: multi-vendor
  training + augmentation first (done), explicit harmonization only if it *measurably* adds.

## What the evidence says for OUR setup
1. **Multi-centre training is the big win** — and we do it (M&M-2 → ACDC, mean Dice 0.90 cross-domain).
2. **Augmentation helped** (our heavy GPU aug: RV 0.84 → 0.89, EF MAE 8.2 → 6.3%). It's the
   generalization-honest lever and it's already paying.
3. **Explicit harmonization (Nyúl) is therefore likely a small/marginal lever for us** — Dice
   already transfers well; the residual EF error is partly *intrinsic calibration* (even SOTA
   nnU-Net keeps a −4% bias). Harmonization tightens spread, doesn't fix bias.
4. **N4 (bias correction) is the safer, assumption-free half of `qfz`** — physical, per-scan, no
   train-reference. Worth doing regardless.

## Decision for `qfz`
- **N4: build it** — clean physical step, no generalization compromise.
- **Nyúl: an A/B experiment, not a default** — fit on the multi-vendor train pool (so the hub isn't
  vendor-biased), apply per-scan, test on held-out ACDC vs z-score and vs aug-only. **Expect
  possibly-null/negative.** Report honestly either way — "explicit harmonization didn't beat learned
  invariance on multi-vendor-trained cardiac seg" is a genuine finding, consistent with nnU-Net.
- **Keep z-score as the safe default** (optimizer conditioning + assumption-free).

## Implementation reference
`jcreinhold/intensity-normalization` (Python) implements Nyúl / WhiteStripe / z-score / N4 wrappers
— usable as a reference or dependency rather than reimplementing landmark fitting.

## Open / caveats
- Nyúl needs a foreground mask (fit on tissue, not air) — a threshold/Otsu step.
- "Bound by the least informative scanner": the hub quality depends on the fit pool's diversity.
- Cardiac cine is all bSSFP (same sequence) → histogram-comparability assumption mostly holds; would
  break on cross-sequence (T1 vs T2) data.

## Sources
- [Contrast augmentation for multi-scanner MRI generalization (Frontiers 2021)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.708196/full)
- [Intensity augmentation for domain transfer, breast MRI (arXiv 1909.02642)](https://arxiv.org/pdf/1909.02642)
- [Effect of intensity standardization on DL WML segmentation, multi-centre FLAIR (arXiv 2307.03827)](https://arxiv.org/pdf/2307.03827)
- [Systematic review of image harmonization, multi-centre/device (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S0169260724001962)
- [Standardization of brain MR across machines/protocols (Sci Reports 2020)](https://www.nature.com/articles/s41598-020-69298-z)
- [jcreinhold/intensity-normalization (impl reference)](https://github.com/jcreinhold/intensity-normalization)
- nnU-Net normalization (per-patient z-score + conditional intensity norm + gamma aug) — Isensee et al.
