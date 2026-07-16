# What is wrong with our synth — a hypothesis from everything we've measured

**Date:** 2026-07-16 · **Scope:** cross-task synthesis (zero-real generation lane) · **Status:** hypothesis + discriminating experiment, NOT yet proven

## The thing to explain

| training color | shape source | cross-vendor Dice |
|---|---|---|
| real | real masks | **0.854** |
| synth (physical, randomized) | real masks (repaint) | **0.68** |
| synth (physical, randomized) | synth masks (generate) | **0.61** |

Shape generation costs ~0.07. **Color costs 0.17** (0.854 → 0.68, shape held perfect). This doc is about the 0.17.

## What every batch actually showed

- **Composition (levels) — the ONLY lever that moved Dice** (+0.059, 0.554→0.613). It fixed a *gross* error:
  myo was painted at the wrong z entirely (−0.52 vs real +0.10). Fixing the global level of the intensity→tissue
  map helped.
- **Everything finer than levels — dead.** within-class spread (−0.23 when "corrected"), blood-mean/trabecular-PV
  (−0.05), texture (−0.30), spectrum band-limit (−0.01). Per-image realism does not move Dice.
- **Coverage — dead.** composite+pathology lifted shape coverage 0.78→0.94, Dice neutral. More shapes, no Dice.
- **noise=0.05 (broadband white) — load-bearing.** Lowering it toward physical SNR collapses zero-real
  0.613→0.385. The model *needs* heavy broadband augmentation to transfer at all.

One pattern fits all of it: **once the global level of the intensity→tissue map is roughly right, nothing else
about per-image appearance matters, and the model survives only because broadband augmentation carries it.**

## The hypothesis: domain randomization forfeits the stable mapping, and that IS the 0.17

Repaint gives the model **real shapes** and **randomized physical color**. Each synth image has a *different*
random contrast (swept TR / flip / vendor / field — by design, for generalization). So across the training set
there is **no stable intensity→tissue mapping**: myo is not reliably any particular brightness. The model can
only learn the **domain-invariant** cue — relative ordering and boundary geometry — because that is the only
thing consistent across randomized images. It is *forced* to throw away absolute-intensity discrimination.

The real-trained model does the opposite: real MRI has a **consistent** intensity→tissue mapping (myo darker
than blood, specific boundary gradients, coil/flow patterns that co-vary with anatomy the same way every scan).
It learns to exploit that stable mapping. On this test set — cmrxmotion + Canon/GE, which is **not very OOD**
because the real train set is already multi-vendor (ACDC + M&M, 4 vendors) — the test **shares that mapping**
with the real training data. So the real model's extra 0.17 is **the value of a stable intensity→tissue mapping
that the test happens to share and that domain randomization deliberately destroyed.**

This is not a bug in the generator. **It is the price of domain randomization**, collected on a benchmark that
rewards exactly the thing randomization forfeits.

### Why this hypothesis explains every result at once

- **Fidelity is dead** because per-image realism does not create a *stable* mapping — every synth image is still
  independently random. Making one image look real doesn't give the model a consistent myo-intensity to latch.
- **Composition (levels) helped** because a *gross* level error corrupts even the invariant cue (if myo and bg
  overlap in z, ordering itself breaks). Fixing levels restores the invariant cue; it doesn't add a stable map.
- **noise is load-bearing** because an invariant-only model is fragile; broadband augmentation is what lets it
  transfer to real texture at all. It is doing generalization work, not fidelity work.
- **Coverage is dead** because more shapes don't add a mapping.
- **The spectrum gap is real but not the mechanism** — it is a *symptom* of the same broadband augmentation that
  is load-bearing. (This corrects the xmcf "irreducible" framing: see below.)

## What this says about whether synth "serves us"

For **matching real-trained Dice on an in-distribution-ish test: it structurally cannot**, and no amount of
fidelity/diversity will close it — that gap is the randomization tax. But that was never synth's job. Its value
is where the real-trained model *also* has no stable mapping:

1. **Genuinely unseen domains / no annotations** — a vendor or sequence real never saw. There the real model's
   0.17 advantage should *evaporate* (its mapping doesn't transfer either), and synth's invariance should pay.
2. **Augmentation on top of real** (real + synth vs real).
3. **The controlled / digital-twin direction** — reduce contrast *variance* toward a known deployment domain.
   This is the only lever that recovers the 0.17, and it trades away generalization to do it (so it is a twin
   move, not a diversity move). This is the project's second direction, and the hypothesis says it is where the
   color-Dice actually lives.

## The discriminating experiment (cheap, decides the hypothesis)

If the 0.17 is a *shared-mapping* advantage and not a synth defect, then **on a test that is OOD for the REAL
model, the real↔synth gap must narrow.** We already have the ingredient: cross-dataset showed a single-centre
real model drops to 0.695 OOD (ACDC→M&M-2). Test:

- Take a domain **absent from the real train set** (a held-out vendor/sequence the real model never saw).
- Measure real-trained vs synth-trained there.
- **Predict:** real drops toward synth; the gap shrinks well below 0.17. If it does → the 0.17 is the shared
  mapping, hypothesis holds, synth's value is exactly unseen-domain. If the gap *holds* at 0.17 even OOD → the
  hypothesis is wrong and synth has a genuine defect independent of the shared mapping.

**Second, still-undone check (owner flagged):** failure analysis of the 0.68 repaint model — per-slice /
per-structure / per-vendor Dice + the 20 worst overlays. The hypothesis predicts failures cluster at
**low-contrast boundaries** (where the invariant cue is weakest) and are **roughly uniform**, not a few
pathological slices. If instead the loss is a specific structure/vendor/slice band → it's a targetable defect,
not the randomization tax.

## Honesty

This is a hypothesis that *fits* every result retrospectively — that is its weakness. It has not been tested
forward. The two experiments above are designed to *break* it, and either could. What is solid: (1) fidelity /
coverage do not move zero-real Dice (many arms, multi-seed on the neutral ones); (2) only gross-level correction
moved it; (3) broadband noise is load-bearing. The interpretation of *why* — the stable-mapping tax — is the
part still owed a forward test. Until then it is the best available account, not a verdict.
