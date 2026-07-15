# Over-separation is two physical legs: sharp boundaries + over-wide mean-gap

**Date:** 2026-07-15 · **Task:** synth generation fidelity · **Beads:** l45x (step 1 of the over-separation epic 2gk5)

## TL;DR

Synth over-separates myo from blood — per-slice myo|cav d′ **4.40 vs real 2.65** (1.66×). This is a
train/test *difficulty* mismatch (net learns cartoon-clean boundaries, fails real's messy ones), and it
decomposes into **two independent, physically-fixable legs** — neither tuned to a target:

1. **Boundary sharpness.** `pv_sigma` and `kspace` were **both 0 by default** → synth keeps *all* spatial
   frequencies = infinitely sharp tissue edges. Real MRI has finite in-plane resolution. Turning on the
   **physically-derived** k-space PSF (`kspace=0.7`, from geometry — see below) drops d′ 4.40 → 3.91,
   closing **28%** of the gap. Adding partial-volume blur reaches **45%** (d′ 3.62, ratio 1.36×).
2. **Over-wide mean-gap.** Even with resolution physics on, the per-class *location* (mean-brightness)
   gaps persist — myo 0.32z too dark, RV 0.54z, cav 0.42z off. A wider myo↔cav *mean* separation is d′
   the resolution kernel cannot touch (blur spreads boundaries, it doesn't move class means). This is the
   whole-FOV composition / z-score leg (hpy/FovBg), independent of leg 1.

d′ is read out where the physics lands — **never tuned to 2.65** (that would be a soft leak).

## The physical derivation (leak-free)

**k-space PSF fraction.** Real cine bSSFP true in-plane resolution ≈ 1.8–2.2mm (partial-Fourier + GRAPPA
under-sampling), reconstructed then resampled to the model's **1.5mm** working grid (`TARGET_INPLANE`).
The kept fraction of the working-grid k-space = working_pixel / true_res ≈ 1.5/2.0 = **0.7** (range
0.68–0.83). `kspace=0` (all frequencies kept) is the unphysical default; 0.7 is the geometry, not a fit
to d′ or to test Dice. Sinc PSF + slight Gibbs ringing → boundary voxels mix myo+blood → boundary d′
falls. Corroborated: `pv_sigma` (a Gaussian mean-map blur) stacks *partially* with kspace (3.62 combined
< 3.91 either alone) — different kernels (sinc low-pass vs local Gaussian), so PV is not a pure
double-count of in-plane resolution.

## Seeded readout (deterministic paint, `--seed 0`)

Per-slice LV-myo|LV-cav d′, real target **2.65**, baseline gap 1.75 (4.40 − 2.65):

| config | d′ | ratio synth/real | gap closed |
|---|---|---|---|
| baseline (kspace 0, pv 0) | 4.40 | 1.66× | 0% |
| kspace 0.7 (derived) | 3.91 | 1.47× | 28% |
| pv_sigma 0.6 | 3.75 | 1.42× | 37% |
| kspace 0.7 + pv 0.6 | 3.62 | 1.36× | 45% |

(The paint is stochastic — unseeded run-to-run scatter is ±~0.4 on this d′. `SynthFidelity.run` now seeds
`torch.manual_seed(args.seed)` so config deltas are the signal, not noise. Seeding ≠ tuning; the diagnostic
is just reproducible.)

Distance mode with resolution on confirms leg 2 is untouched: **location** (mean gap, z) myo 0.32 / RV 0.54
/ cav 0.42; **shape** (mean-centered W1) all small (myo 0.13, cav 0.07) — resolution physics lowers d′ via
boundary spread without distorting within-class shape.

## Why physics only closes ~45% (mechanism, not number-watching)

d′ = |mean_myo − mean_cav| / pooled-SD. The k-space/PV kernels raise the **denominator** (boundary mixing
widens each class's spread) but barely move the **numerator** (class means). Synth's numerator is *itself*
too big: synth myo −0.52z, cav +1.90z → mean gap ~2.4z, vs real myo +0.24, cav +2.23 → ~2.0z. The extra
0.4z of synth mean-gap is leg 2 (myo painted too dark under partition-bg z-score composition — see the
myo-separability-artifact writeup). **No resolution knob can close a mean-gap.** So the two legs are
genuinely additive: resolution physics for the denominator, whole-FOV composition for the numerator.
Together they should close most of the over-separation; each is physical and parameter-free-to-target.

## Consequences

- **l45x delivered:** over-separation *attributed* — sharp boundaries (missing PSF/PV) + over-wide mean-gap
  (composition). The default painter under-models finite acquisition resolution (kspace=pv_sigma=0); the
  physically-correct fix is derived from geometry, self-balancing, and lands d′ at 1.36× real (a **finding**:
  in-plane resolution alone is not the whole story — residual is through-plane PV, concentrated at base/apex
  rather than a global blur, plus real intra-myocardial heterogeneity clean bSSFP signal doesn't reproduce).
- **Next (5xf6):** does driving synth d′ toward real 2.65 — via these physical acquisition regimes (sweep
  kspace across its physical range 0.5–0.9 = legit resolution domain-rand axis) — actually raise zero-real
  Dice? Train one quick single-seed arm per regime, add the 2 points we have (real d′2.65→0.85, baseline
  synth d′4.4→0.56), read the (measured-d′, Dice) scatter. If it holds → over-separation is a real lever;
  if flat → d′ is a description of an irreducible appearance gap → pivot to augmentation/inverse-twin.
- d′ stays a **readout**, never a training target or a knob fit to 2.65 (soft-leak rule).
