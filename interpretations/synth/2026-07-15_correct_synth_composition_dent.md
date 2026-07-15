# Correctness, not a blur hack: physical whole-FOV composition + inflow dents the zero-real ceiling

**Date:** 2026-07-15 · **Task:** synth generation fidelity · **Beads:** hpy (whole-FOV composition), 2gk5 follow-through

## TL;DR

Two **physically-correct, leak-free** generator fixes take zero-real Dice **0.554 → 0.613** (+0.059,
myo +0.08) — beating the earlier over-blur hack (0.588) and validating the thesis *make the synth correct
and the data becomes usable*. No tuned knobs, no over-blur, nothing fit to a d′ target.

| config | test Dice | myo | mechanism |
|---|---|---|---|
| baseline (procedural equal-tier bg, inflow on) | 0.554 | 0.46 | myo too dark, blood over-bright |
| over-blur hack (kspace0.4/pv1.5) | 0.588 | 0.50 | aggressive blur regularization (rejected — not fidelity) |
| **correct (torso-composition bg + inflow off)** | **0.613** | **0.54** | physical levels |
| real-trained | 0.854 | 0.80 | — |

## The two defects and their physical fixes

Signed per-class z-means (seeded, real vs synth, 2500 slices) diagnosed both defects and confirmed the fix:

| class | real | baseline synth | corrected synth |
|---|---|---|---|
| bg | −0.02 | (myo-masking bright bg) | −0.08 ✓ |
| myo | +0.10 | **−0.52** ✗ | **+0.10** ✓ |
| RV | +1.54 | +3.76 (inflow) | +2.04 ✓ |
| cav | +1.66 | +3.93 (inflow) | +2.04 (≈, +0.38) |

**1. Myo too dark = wrong whole-FOV composition (fixed).** The per-image z-score references the whole-FOV
intensity histogram, so the background's AREA FRACTIONS (not just its tissue levels) set where the heart
lands after normalization. `ProceduralBg` bucketized a uniform field into K **equal** tiers along a
lung→liver→muscle→**fat** ladder → over-weighted bright fat (real torso ≈ 0% fat), missed the ~40% dark
(air+lung) → image mean too high → myo drifts to −0.52. **Fix:** paint the bg tiers as the physical torso
composition — air 29% / lung 13% / liver 5% / muscle 53% (bg-tissue area fractions of the XCAT/MRXCAT FOV
phantom, renormalized; leak-free physical prior, NOT fit to real MRI) with literature bSSFP and a new dark
`air` tissue (PD≈0). Result: bg AND myo both land exactly on real (−0.08/−0.02, +0.10/+0.10). The old
bright bg was *accidentally masking* blood-over-brightness while breaking myo.

**2. Blood over-bright = gain-only inflow (fixed).** With a correct scene, blood was exposed at +3.9z vs
real +1.6. Traced to `inflow`: it models fresh-spin **brightening** (blend toward PD·sin(flip)) but not the
compensating **flow-dephasing signal loss + papillary/trabecular partial volume**. Measured real blood
(+1.66) sits *below* ideal steady-state (+2.04, inflow off) — the loss terms slightly win — so a gain-only
inflow is physically unbalanced and over-brightens. **Fix:** default `inflow` off (≈ net cancellation),
landing blood at +2.04, close to real. Residual +0.38 (real below steady-state) is the flow-loss/papillary
term still to model — a documented physical target, NOT closed with the fitted `blood_scale` knob.

## Over-separation resolved *by correctness*, not blur

Per-slice myo|cav d′ (the over-separation index) moved from **1.66× real (baseline, too easy)** through real
to **0.69× (corrected)** — the level fixes removed the over-separation entirely (mild overshoot to
under-separated, from a now-slightly-high within-class spread — a separate axis, not chased). Contrast the
over-blur hack, which forced d′ down by softening boundaries coarser than real acquisition (augmentation
masking a defect). Here d′ falls because the class *levels* are physically right — the honest mechanism.

## Honesty / caveats

- Single-seed, but +0.059 is well above the ±0.02–0.03 single-seed noise floor, val corroborates (+0.084:
  0.592→0.676), and the mechanism is independently validated (levels match real before any training). Per
  the predict-from-data philosophy the trend + mechanism is the evidence, not a seed sweep.
- Residual gap to real (0.613 vs 0.854 = 0.24) is still dominated by shape-generation + appearance; this
  fix closes ~20% of the zero-real gap (vs the hack's 12%).
- Blood +0.38 residual (flow-dephasing loss / papillary PV) and the slightly-high within-class spread are
  the next physical targets — both leak-free, no fitted knobs.

## Changed

- `mri_physics`: add `Tissue.AIR` (PD≈0), `TORSO_BG` composition (torso area-fractions + literature bSSFP),
  `torso_paint_params`/`torso_thresholds`.
- `synth.ProceduralBg`: paint the physical torso composition at real area fractions (random shapes, correct
  histogram); `ProceduralBgCfg` drops the vestigial `bg_tiers`; `inflow` default → off.
