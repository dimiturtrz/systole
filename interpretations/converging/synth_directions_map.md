# Synth → real: directions map (living tracker)

**Purpose:** single source for "where are we" on the synth-generalization effort. Updated as directions move.
Each row = maturity tag + best number + verdict + next lever. Keep it honest; keep it current.

**North star (owner):** domain generalization via **all-synthetic training**. **Zero-real is always the big goal.**
Everything else supports it or is a diagnostic for it.

---

## THE GOAL — A. Zero-real GENERATE (synth mask + synth color)

**Best: ~0.61 test** (composition-fixed) · real-trained 0.854 · **blocker (CORRECTED 2026-07-16 by looking at
predictions): RV OMISSION in a patient-clustered tail, NOT a uniform color collapse.**

Pooled RV 0.457 was a **bimodal artifact**: median per-case RV (mid-slice) = **0.853** (real-trained level),
but a tail of ~7–10% of cases (patient-clustered: P004/P011/P014) score RV **0.0** — the model segments the LV
**perfectly** (myo ring + cav) and **entirely omits the RV** (0 RV pixels predicted) though the RV blood pool is
large and clearly visible. Not mis-colored, not confused with LV, not mis-bounded — **absent.** This is a
**detection / RV-shape-coverage / recall** failure, NOT color (the LV color is fine). Confirms why the contrast
levers were dead — the problem was never contrast. Viz: `scratchpad/rv_fail.png` (mechanism), rebuild via
`scratchpad/rv_fail_viz.py`. Sub-lever tree:

| lever | status | evidence |
|---|---|---|
| composition / whole-FOV levels | ✅ **WIN +0.059** (on) | fixed myo z −0.52→correct (2gk5) |
| color fidelity (spread / blood-mean / spectrum) | ❌ dead | nk70, xmcf — fidelity≠Dice, ×5 |
| contrast diversity (tissue_spread width, contrast_random) | ❌ **refuted** | nttu — widen flat, randomize *hurts* RV; RV needs contrast *structure* not diversity |
| shape coverage (composite / pathology pool) | ❌ dice-neutral | coverage 0.78→0.94, Dice flat |
| noise / high-freq aug | 🔒 load-bearing (constraint) | nk70.1 — can't lower (→0.385) |
| **RV recall** (Tversky FN-penalty, β>α) | 🟡 **directional, marginal** | vs MATCHED baseline (0.594): RV +0.044, myo −0.010, mean +0.013 (near noise). RV-specific gain w/ myo tradeoff; single-seed, not confirmed. Fixes under-seg, not omission |
| **RV omission tail** (0-px total misses) | 🩺 **root-caused (nttu.5)**: RV-vs-**bg** recall, NOT coverage | all 9 omitted slices have RV softmax present (max 0.21–0.57, med 0.41, **never zero**); winner = **bg** 71–98%; apical/small-RV (z 7–8). Threshold/recall-fixable → nttu.7 on-target; coverage (nttu.4/.8) is NOT the cure |
| within-label heterogeneity (multi-Gaussian paint) | ⬜ deprioritized | it's a *color* lever; the blocker isn't color |
| **learned shape prior** (pathology tail) | ⬜ filed (vpn5) | reaches the 22% DCM/HCM tail SSM misses |

**Frontier redirected by the failure viz:** the blocker is RV **omission** on a patient tail, so the levers are
**RV detection/recall** and **RV-shape coverage**, NOT color/heterogeneity. Open disambiguation: is the tail a
recall problem (model under-fires RV), a coverage problem (these RV shapes absent from synth), or an
appearance-quirk of specific scans? Characterize the failing patients (all-slices? common vendor/motion/RV-size?)
to pick the lever.

---

## Diagnostic for A

**B. Zero-real REPAINT** (real mask + synth color) — **~0.68**, measured. Not deployable; it's the *color-axis
isolator*: color costs 0.17, shape 0.07. Told us the A blocker is mostly color, not shape.

---

## Support (secondary to A)

| # | direction | best | maturity | note |
|---|---|---|---|---|
| C | Real + synth **AUGMENTATION** (mix) | ~0.84 | under-measured | does synth help *on top of* real? (pwih) — no clean matched A/B yet |
| D | Controlled / **INVERSE / twin** | — | not started | fit color to a scan; project's 2nd named direction (ncph) |
| E | **EF lane** | cal −4pp | mature ✅ | calibration (tb58), Kaggle weak-sup (2b7m); own objective |
| F | nnU-Net baseline | — | reference | quarantined SOTA ref |

---

## Where we are right now (2026-07-16)

A is the goal, best ~0.61 (composition). This session: **looked at the failures** and reframed the blocker from
"RV color collapse" (a pooling artifact) to **RV omission/under-seg** (detection, not color) — which retroactively
explains why every color/contrast lever was dead. Then tested the **recall lever** (Tversky FN-penalty): RV
+0.044 vs matched baseline but a myo −0.01 tradeoff, mean +0.013 (near noise, single-seed) — **directional, not a
confirmed keep**; it fixes *under-seg*, not the *omission* tail.

**Root-caused (nttu.5, 2026-07-16):** the omission tail is **NOT** zero-activation — every omitted slice carries
real RV softmax (max 0.21–0.57, never <0.05), lost to **background** at argmax (bg wins 71–98% of the GT-RV region).
So a loss reweight CAN recover it (activation exists to amplify) — it's RV-vs-bg **recall/calibration**, apical/small-RV
slices, **not** shape coverage. Kills the "detector doesn't fire" hypothesis.

**Open next:** (1) **nttu.7 recall — now the primary lever** (Tversky FN-penalty raises RV over bg exactly here):
milder β + multi-seed to see if the RV gain survives without the myo cost; a class-prior/logit-bias at inference is
a cheaper same-family test. (2) nttu.4/.8 shape-coverage **deprioritized** for omission (root cause isn't coverage).
Do **not** pivot off A. Method note: always run the MATCHED default baseline before claiming a mover.
