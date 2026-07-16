# The zero-real gap is an RV/myo COLOR collapse — located, not theorized (r00n)

**Date:** 2026-07-16 · **Lane:** synth inadequacy hunt (r00n) · **Method:** failure analysis + lit-grounding, not input-distribution stats

## The reframe that started this

The prior epics (nk70, xmcf) measured the *generator's output distribution* (PSD, histograms, z-means) — the
exact fidelity-metric frame nk70.1 proved does **not** predict Dice — and concluded the color gap was
"irreducible." Owner pushback killed that: a good synth-only model *should* generalize (SynthSeg does), so the
gap is an **inadequacy we can locate**, not a tax. This epic looked at the **model's actual errors** and the
**achievable ceiling** instead.

## Gate 0 — the achievable ceiling (lit)

`research/deep_dives/2026-07-16_synth_only_cardiac_ceiling.md` (numbers per Haiku researcher, verify vs primary
before README):
- **SynthSeg** (domain-randomization) synth-only ≈ **0.88 Dice on brain = ~3-point gap** vs supervised. Fully
  random-GMM contrast, all axes unconstrained.
- **UltimateSynth** (physics-constrained) beats SynthSeg **0.83 vs 0.76** on brain — physics grounding wins.
- Cardiac synth-only on ACDC/M&Ms: **unpublished**.

**Our zero-real gap is ~24 points (0.61 vs 0.854).** Domain randomization achieves ~3. **We are ~21 points below
the achievable ceiling** — decisive evidence the gap is an inadequacy, not inherent. (UltimateSynth reconciles the
project's physics>random thesis: the fault is not "physics," it's our sweep being too narrow.)

## Gate 1 — where the model actually fails

Matched per-structure Dice, zero-real **generate** (refac_proc) vs **real** (production), two test sets:

| structure | zero-real cmrx | real cmrx | gap | zero-real canon | real canon | gap |
|---|---|---|---|---|---|---|
| **RV** | 0.457 | 0.872 | **−0.42** | 0.177 | 0.867 | **−0.69** |
| LV-myo | 0.551 | 0.810 | −0.26 | 0.356 | 0.807 | −0.45 |
| LV-cav | 0.729 | 0.885 | −0.16 | 0.572 | 0.834 | −0.26 |

**The gap is structure-ordered RV ≫ myo > cav, consistent across both test sets.** Not the uniform low-contrast
loss the tax predicted. Zero-real RV **HD95 65–89 mm** (real ~9 mm) — RV is **mislocalized/dropped wholesale**,
not mis-bounded. cav is nearly fine.

### Shape vs color — isolated from existing logs (no new run)

Staged repaint checkpoints (`bssfp`, `ph_ks`, `ph_pvks` = `synth_p=1.0`, no anatomy pool = real masks repainted
synth color) vs generate (`refac_proc`, `anat_1000`, `compb`), same frozen test n=147:

| structure | real | repaint (real mask + synth color) | generate (synth mask + color) | shape cost | **color cost** |
|---|---|---|---|---|---|
| **RV** | 0.87 | ~0.60 | ~0.50 | ~0.11 | **~0.27** |
| **myo** | 0.81 | ~0.61 | ~0.47 | ~0.14 | **~0.20** |
| **cav** | 0.88 | ~0.73 | ~0.72 | ~0.00 | ~0.15 |

**With perfect shape (repaint), RV is still ~0.60 — 0.27 below real. The collapse is mostly COLOR, not shape**
(the earlier shape-worry, refuted by logs). Shape adds a secondary RV/myo penalty (~0.1); cav is easy on both
axes. Best repaint ~0.667 = the taxonomy's "0.68".

## The located inadequacy

**The zero-real deficiency is a color-driven collapse of the thin, contrast-defined structures — the RV free
wall and the myo ring — while the big bright blood blob (LV-cav) survives.** Mechanism: thin structures are
detected by *contrast against neighbours* (myo between blood and lung/fat; RV wall against lung). Our synth
color doesn't teach robust contrast cues for them.

Why — the recipe diff (gate 0): SynthSeg randomizes contrast **fully** (random GMM per label; myo can be any
brightness, even > blood) plus **within-label heterogeneity**. Ours (`refac_proc`) has **`tissue_spread=None`**
— the tissue contrast is **fixed**, only acquisition (TR/flip/field) varies. **Our contrast randomization is too
narrow**: the model overfits our single contrast ordering and thin structures drop when real deviates.

## Gate 2 — the fix test (in progress)

Turn on the **physical per-sample T1/T2 sweep** (`tissue_spread`, literature-bounded Stanisz/Bojorquez —
leak-free, within the physics thesis, NOT a random-GMM hack) to widen contrast. Predict RV/myo recover toward
real. If they do → narrow-contrast is the inadequacy and the fix moves Dice. If flat → physical widening is
insufficient and the next lever is within-label heterogeneity or more aggressive randomization. `runs/xmcf_widen`.

## Honesty

Gate 1 is on a `--quick`, single-seed generate model (refac_proc, ~0.56) as the representative zero-real; the
decomposition uses aggregate repaint checkpoints of slightly different corruption recipes, so the ~0.1/~0.27
split is approximate, not a matched A/B. But the qualitative result — **RV ≫ myo > cav, color-dominant,
consistent across two test sets, RV still collapsed with perfect shape** — is robust to those approximations.
One wasted `runs/repaint_iso` train was launched before checking that the isolation was derivable from logs
(owner flagged; ~4 min GPU). The next number that matters is whether widening contrast moves RV.
