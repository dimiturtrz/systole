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

## Correction — the cav gap sign was WRONG; blood is too DIM, not too bright (2026-07-01)
The conclusion above ("cav too bright by ~0.9z", "no blood-signal knob can close it", line ~32/43/46)
is **refuted by measurement**. It rested on one direction only: banding *lowers* blood → cav loc got
*worse* (0.89→1.48), read as "z-renorm noise." Nobody tried *raising* blood.

New probe `synth.blood_scale` (multiplies blood-pool MEAN, off=1.0) swept on `synth_fidelity`
(val set, synth_p=1):

| blood_scale | mean W1 | LV-cav W1 | cav location | RV location | LV-myo W1 |
|---|---|---|---|---|---|
| 1.0 (off) | 0.515 | 0.929 | 0.912 | 0.555 | 0.318 |
| **1.6** | **0.385** | 0.380 | 0.356 | **0.004** | 0.625 |
| 2.0 | 0.488 | 0.394 | 0.036 | 0.301 | 0.824 |

So the gap closes by making blood **brighter**: scale≈1.6 cuts mean synth-real W1 **25%** and nearly
perfectly matches RV location (0.555→0.004) — RV being the exact class the model shortcuts away. The
banding-worsened-it result was **evidence blood was too dim**, mis-signed. Tradeoff: myo worsens
(0.318→0.625) via the per-image z-norm coupling (brighter blood shifts global mean/std), so mean W1
bottoms near 1.6 not 2.0. It's **semi-empirical** (a level-match scalar compensating z-norm
composition, not a first-principles flow term) — physics gives the *shape* (0.05, ~perfect), this
matches the *level*. Also re-confirmed: **vessel-bright bg ladder → worse** (mean W1 0.515→0.57; bg
composition already matches, location ~0.005), so the gap is NOT background — it's the blood pools.

**Open (the decisive test):** does the 25% fidelity gain + RV-location fix convert to pure-synth
Dice / RV recall (vs 0.62 physics / 0.66 stats)? A pure-synth training run at blood_scale=1.6 vs 1.0
(same protocol; the delta is protocol-independent) settles it. If yes, the "caps ~0.66, needs a
learned generator" ceiling falls to one physics-ish scalar.

### RESULT — it converts. Pure-synth ceiling beaten by one scalar (2026-07-01)
Pure-synth (synth_p=1), same protocol, held-out cross-vendor test, single seed:

| blood_scale | RV Dice | myo Dice | cav Dice | **mean** | RV recall |
|---|---|---|---|---|---|
| 1.0 (control) | 0.577 | 0.632 | 0.739 | **0.649** | 0.491 |
| **1.6** | 0.681 | 0.665 | 0.757 | **0.701** | **0.598** |

**+0.052 mean Dice, RV +0.104, RV recall 0.49→0.60** — the shortcut class recovers, every class up.
Control 0.649 matches the documented ~0.62–0.66; treatment **0.701 clears the 0.66 ceiling**. So the
earlier conclusion "no blood-signal knob can close it → needs a learned generator" is **wrong**: a
single blood-pool level scalar (found via the fidelity metric) does. Note myo *Dice* rose despite its
*fidelity W1* worsening at 1.6 → W1 is an imperfect Dice proxy (the z-norm coupling was benign).
Caveats: single seed (Δ0.05 > the ~0.04 noise band but not yet replicated); 1.6 is semi-empirical
(level-match, physics gives the shape); 0.701 still < 0.77 hybrid / 0.86 real — narrowed, not closed.
Next: seed-replicate; test blood_scale as *augmentation* (synth_p=0.5) — the flagship-relevant path
(uncalibrated synth-aug was neutral at 0.852 vs 0.864; calibrated may flip it positive).

### The derivation chain replaces the magic scalar (2026-07-01)
Owner's grounding chain for any physics number: **public (papers/SAR) → stats (our data) → simulate +
voxel-match (lever 3, "flying blinder but we have 10k+ voxels")**. Applied to the blood-level gap:
1. **public** — flip is SAR-capped, field-driven; TR floor ~3ms (deep-dive 2026-07-01_mri-vendor-acq).
2. **stats** — `real_levels` showed synth blood too DIM per vendor; but flip can't be analytically
   inverted from z-scored 2-tissue+bg-mix data (underdetermined).
3. **simulate + voxel-match** — sweep the physical `inflow` param, minimize real-vs-synth blood-voxel
   W1 (3.1M blood voxels): **inflow ≈ 0.15** → mean W1 **0.368** (vs 0.385 for the empirical
   blood_scale=1.6; 0.506 at inflow=0), RV location 0.522→**0.022**. And 0.15 sits inside the
   physiological single-TR range (f≈0.02–0.39) the physics predicts — voxel-fit AGREES with the model.

So `blood_scale=1.6` (a fit scalar) is replaced by `inflow` (entry-slice enhancement: blend blood
toward fresh PD·sin(flip)), a physical mechanism whose value is derived by voxel-matching and
cross-checked against the physiological range — slightly better fidelity, with an argument. Acquisition
(TR/flip) is now derived per field (`derive_acq`, contrast-optimal flip=53@1.5T capped by SAR) and
wired into the paint. **Pending:** does derived inflow=0.15 + derived flip convert to pure-synth Dice
(vs 0.701 at blood_scale=1.6)? Training run next.

### Replication + augmentation (2026-07-01)
- **Pure-synth REPLICATES.** Seed 1: 0.604 → 0.679 (Δ+0.075, RV +0.145); seed 0 was 0.649 → 0.701
  (Δ+0.052). Two-seed mean **0.627 → 0.690 (+0.063)**, RV the biggest gainer both times. The
  single-seed caveat is retired — blood_scale=1.6 is a real pure-synth transfer gain.
- **Augmentation is marginal.** synth_p=0.5: bs1.0 0.845 → bs1.6 0.850 (Δ+0.005, RV +0.022). Does
  NOT beat the 0.864 real-only baseline. Why: with 50% real in the batch, the real images already
  supply the correct blood level, so calibrating the synth half adds little. **Calibration matters
  only when synth is the sole signal.** → blood_scale stays OFF by default (no flagship gain); it is
  the recommended setting for *pure*-synth training, not augmentation.
- **Net:** the pure-synth ceiling is genuinely raised (~0.66 → ~0.69, RV shortcut substantially
  fixed) by one measured scalar — but the gap to real (0.86) / hybrid (0.77) is narrowed, not closed.
  The differentiated result (physics-based synthetic *training*, cross-vendor, ceiling moved by a
  fidelity-found calibration) stands; the learned-generator route is no longer the only lever, though
  it remains the path to fully close the residual.

### Why blood_scale=1.6 is a hack, and the principled replacement (2026-07-01)
The scalar was fit to the *average* fidelity gap. Two measurements show that's wrong:

**(1) The blood-level gap is strongly VENDOR-dependent** (`synth_fidelity.by_vendor`, blood_scale=1.0,
painted from real masks, ≤1200 slices/vendor):

| vendor | LV-cav location | RV location | mean W1 | cav shape |
|---|---|---|---|---|
| Siemens | 0.77 | 0.41 | 0.49 | 0.10 |
| GE | 1.25 | 0.91 | 0.72 | 0.09 |
| Philips | 1.79 | 1.25 | 0.91 | 0.21 |
| Canon | (155 labelled slices — skipped) | | | |

A **2.3× spread** (0.77→1.79), pure LOCATION; **shape is small + vendor-invariant** everywhere — so the
physics *contrast* is right, only the blood *level* is wrong, by a machine-dependent amount. RV and
LV-cav (both blood) move together per vendor → one per-vendor blood factor fixes both pools. A global
1.6 under-corrects Philips, over-corrects Siemens.

**(2) The generator isn't machine-conditioned at all.** Audit: `acquisition_for(vendor)` (the per-vendor
cine TR/flip table) is **never called in the paint**. TR/flip are sampled from GLOBAL ranges
(`cfg.tr_ms`, `cfg.flip_deg`); the sampled vendor is a **cosmetic tag** (metadata only). So contrast
is not generated relative to the machine — the 2.3× vendor spread cannot be reproduced.

**Principled program (replaces the scalar):**
1. **Wire `acquisition_for` into the paint** — vendor → its TR/flip (± jitter for intra-vendor spread),
   not a global range. Makes contrast genuinely machine-respective ("generator correct wrt its knobs").
2. **Add inflow enhancement** — bSSFP eq is steady-state; cine blood is partly fresh (unsaturated)
   inflow → brighter than steady-state. Model it sequence-conditioned (TR-dependent fresh fraction),
   swept from a physiological range — a physics parameter, per-vendor by construction, **not fit to our
   test**. This is what the vendor-dependent location gap is: un-modeled inflow, TR-driven → vendor-driven.
3. **Validate external / untuned** — confirm on a vendor/dataset not used to find the fix (Canon, or an
   M&Ms split we didn't tune fidelity on). Guards against test-fitting the calibration.
New reusable tool: `analysis/synth_fidelity.by_vendor` + `--by-vendor` CLI.

### Measured per-vendor real blood level — the calibration target (2026-07-01)
`reference.build_real_levels` (`--real-levels`) derives per-vendor per-class real intensity (z-units)
from the store images → `<data>/reference/real_levels.yaml` (computed, verified=true). Real LV-cav /
RV level:

| vendor | real LV-cav | real RV |
|---|---|---|
| Siemens | +1.39 | +1.26 |
| GE | +1.83 | +1.76 |
| Canon | +1.87 | +1.87 |
| Philips | +2.17 | +1.99 |

Spread 1.39→2.17 (0.78z). Synth at blood_scale=1.0 sits ~+1.33 → matches Siemens, far below the
others — the vendor-dependent gap, quantified as an absolute target (incl. Canon, the unseen test
vendor the fidelity tool couldn't reach). The generator's machine-conditioned blood level should aim
at these per-vendor. **Caveat:** per-image z-score conflates acquisition contrast with dataset
FOV/composition (Philips=M&Ms, Siemens=ACDC+M&Ms), so this is a valid match-target but a confounded
pure-physics measurement — the literature TR/flip (web research, in flight) disentangles it.
