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
| **RV recall — TARGETED** (inference RV logit-bias) | ✅ **KEEP (nttu.7)** — free, no-retrain | val-tuned b≈1.5–2.0 on existing 0.61 model: **val RV +0.043** (0.473→0.516, monotone/6-step), val mean +0.018; transfers to cmrx test **RV +0.014, mean +0.012**. Post-hoc, val-fit, leak-clean (tb58 family) |
| **RV recall — GLOBAL** (Tversky FN-penalty, β>α) | ❌ **KILL (nttu.7)** | matched 40ep A/B: β=0.6 vs dice_ce → val mean **−0.030**, test **−0.048**, myo/cav both bleed (test −0.061/−0.073), **RV flat**. Global FN-penalty over-segs every class, doesn't localize the RV-vs-bg margin. Mechanism sound, instrument wrong (global≠targeted) |
| **RV omission tail** (0-px total misses) | 🩺 **root-caused (nttu.5)**: RV-vs-**bg** recall, NOT coverage | all 9 omitted slices have RV softmax present (max 0.21–0.57, med 0.41, **never zero**); winner = **bg** 71–98%; apical/small-RV (z 7–8). TARGETED recall recovers it (logit-bias ✅); coverage (nttu.4/.8) is NOT the cure |
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

A is the goal, best ~0.61 (composition). Reframed the blocker from "RV color collapse" (a pooling artifact) to
**RV omission/under-seg** (detection, not color), root-caused (nttu.5) to **RV-vs-bg recall** on apical/small-RV
slices: every omitted slice carries real RV softmax (max 0.21–0.57, never <0.05) lost to **background** at argmax.

**nttu.7 (2026-07-16) — recall IS a real lever; the question was "used right?", answered by 4 matched 40ep arms:**

| lever | test RV | test mean | read |
|---|---|---|---|
| Tversky **bg-INCLUDED** (β=0.6) | flat | **−0.048** | **BUG**, not a fair test — `include_background=True` penalizes bg-FN = rewards predicting MORE bg = **suppresses foreground recall**. Doubly broken. |
| Tversky **bg-EXCLUDED, global** | −0.037 | −0.022 | fixing the bug recovers the bleed (myo +0.073, cav vs bg-incl), but RV still flat — **global≠targeted** |
| **RV-targeted** (class-weight ×3) | **+0.022** | −0.019 | targeted source recall **lifts RV** (val +0.028), but ×3 over-corrects → **cav −0.069** (steals softmax mass from the other blood pool) |
| **logit-bias** (targeted, inference) | +0.014 | **+0.012** | cleanest: RV up (+0.043 val, monotone/6-step), **cav held**, free, no-retrain (tb58 family, leak-clean) |

**Takeaway (corrected — the earlier flat "Tversky KILL" was wrong):** recall is a **real RV lever both post-hoc and
at source** — the moment the pressure is RV-*targeted*, RV lifts. What failed was (1) a genuine `include_background`
bug (fixed in `DiceCETversky`, now bg-excluded like HD/HER — recovered ~+0.028 mean) and (2) using a *global*-
foreground loss for an RV-specific problem. The **logit-bias dominates** (same/bigger RV gain, no cav cost, free);
a source-level RV-weight is viable-but-needs-tuning (×3 costs cav; a milder weight likely lands RV+ cleaner).

**nttu epic CLOSED (8/8, 2026-07-16).** Diagnostics committed (nttu.6): `python -m cardioseg.evaluation
rv_omission --run <zero-real> --mode {probe,bias}` reproduces the nttu.5 recall-vs-coverage split + the nttu.7
logit-bias sweep. Coverage levers (nttu.4/.8) resolved by root cause — the model fires RV softmax on the failing
slices, so those shapes ARE covered; the gap is argmax recall, not coverage.

**Open next:** (1) optionally productionize the RV logit-bias as an opt-in inference class-prior (small, val-fit;
label post-hoc like tb58) — filed, not urgent (gain modest, +0.012 test mean). (2) nttu.4/.8 shape-coverage
**deprioritized** for omission (root cause isn't coverage). Do **not** pivot off A. Method note: always run the
MATCHED default baseline before claiming a mover; a monotone dose-response beats a single-seed delta.
