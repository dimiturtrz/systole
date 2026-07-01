# Synthetic-data fidelity investigation — why pure-synth caps, and where it breaks

**Date:** 2026-06-30 · **Context:** bd cardiac-seg-276 / bgc / 7co / nhz / jeh · physics-based
SynthSeg-style generator (`cardioseg/data/synth.py`, `mri_physics.py`).

## Goal
Train a segmenter on 100% synthetic images (label→image, zero real pixels) that transfers to real
cross-vendor MRI — and understand *why it breaks* so the fix is evidence-led, not guessed.

## Result ladder (cross-vendor mean Dice; held-out Canon/GE + cmrxmotion, n=147; real baseline 0.864)
| training data | mean Dice | note |
|---|---|---|
| random per-class contrast (pure-synth) | 0.32 | random intensities destroy the blood-bright/myo-dark cue |
| realistic measured priors, flat bg | 0.39 | heart fixed; background is the wall |
| **partition bg (real shapes)** | **0.66** | the working pure-synth recipe |
| hybrid (synth heart + REAL bg) | 0.77 | proves bg realism is the ceiling, not the heart |
| bSSFP physics paint (single-field) | 0.62 | physics chosen on principle; ~stats, rides no flow artifact |
| stats (measured priors) | 0.66 | wins by ~0.04 — partly via an RV/cav flow artifact |

## The diagnostic loop (the actual deliverable)
Three reusable tools, each committed + tested:
1. **`evaluation/attribution.py`** — per-GT-class confusion + Captum saliency. *What the model learns.*
2. **`evaluation/synth_fidelity.py`** — per-class Wasserstein-1 synth-vs-real, split into **location**
   (mean shift) vs **shape** (centered). *Where the data breaks.*
3. The loop: generate → train → attribution → fidelity → localize → fix → re-measure.

## What it found
- **Attribution:** the pure-synth model shortcuts on the **bright central LV-cavity blob** → systematically
  **under-segments RV** (recall 0.45 → predicted background). Directional, not random.
- **Fidelity:** **blood is the worst-matched class** (cav W1 ~0.9, RV ~0.55; myo/bg < 0.35).
- **Decomposition (the key result):** the cav gap is almost pure **LOCATION** (0.93 location, 0.09 shape).
  Blood's distribution *shape* is right — it's just **too bright by ~0.9 z**. **Not an architecture
  ceiling.** myo/bg are the inverse (location fine, mild shape mismatch).

## Physics levers tried to close the cav-location gap — all failed
| lever | result | why |
|---|---|---|
| flow (blood-pool texture spread) | dud (within ±0.04 noise) | location isn't spread |
| off-resonance bSSFP banding (correct physics, tested) | **worse** (cav loc 0.89→1.48) | lowering blood signal gets re-referenced by per-image z-score |
| vessel-bright bg ladder | dud (0.89→0.88) | partition area small; z-renorm again |

## Conclusion
**The cav-too-bright gap is a COMPOSITION / normalization effect, not a blood-signal effect.** It lives
in per-image z-score space: synth's *background* intensities (painted by approximate tissue physics)
don't match real bg magnitudes — only the *shapes* do (partition). So the whole-image mean/std differ,
and the cavity lands at the wrong relative z. **No blood-signal physics knob can close it** (flow,
banding, inflow/through-plane PV are all signal levers → ruled out by the same logic).

This also explains why **measured-stats edges physics** (0.66 vs 0.62): stats fits the real per-class
intensities by construction, so its composition matches; physics is principled but approximate on the
full FOV. Physics was kept **on principle** (interpretable, no learned model, physical cross-vendor
diversity via TR/flip/field sweeps), with this fidelity gap characterized, not hidden.

## Where the remaining fidelity lever is
- **Composition-matched bg** — paint the background to match real bg intensity (data-driven bg, physics
  heart) — would close the z-norm gap but reintroduces measured stats for bg.
- **Learned generator (GAN/diffusion, bd ap6)** — the only route to full-FOV real-matching appearance.
- Pure-synth caps ~0.62–0.66 with physics; the **diagnostic loop + the localized root cause** is the
  durable result.

## Update — augmentation + the calibration path (2026-07-01)
- **Synth-as-augmentation (physics, synth_p=0.5): 0.852 vs 0.864 baseline** — neutral/slightly below.
  Uncalibrated synth doesn't help as augmentation either. **Both pure-synth and augmentation are gated
  on the same root cause** (uncalibrated physics contrast → composition off).
- **Hybrid physics-heart + data-bg: FAILED** (cav loc 0.9→1.76) — **scale incompatibility**: bSSFP is
  arbitrary signal units, real-bg stats are z-units; reconciling = data-anchoring the heart = stats.
  So it's all-physics or all-stats, not a naive hybrid.
- **The pure-synth fix (insight): the units are valid but UNCALIBRATED.** Generic acquisition params →
  contrast matches no real vendor. Calibrate to per-vendor/field acquisition (TR/flip/field) as a
  **normalization axis in reference data** (`reference/acquisition.yaml`; we already parse vendor+field).
  Stays all-physics (one scale). Synth should also **emit metadata** (the sampled vendor/acquisition) so
  it flows the same normalization path as real + enables per-vendor fidelity. First brick landed:
  `mri_physics.acquisition_for(vendor, ref)` (typical per-vendor cine bSSFP, reference-overridable, tested).
  Remaining (`bd ulw`): synth samples the real vendor/field distribution → acquisition_for → paint +
  emit meta → measure per-vendor fidelity + pure-synth transfer.
