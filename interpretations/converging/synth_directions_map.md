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
| **the 0.17 color gap: tax vs inadequacy** | ⚖️ **verdict (49b7)**: BOTH, in named proportions | repaint failure analysis + truly-OOD test — see below |

**Frontier redirected by the failure viz:** the blocker is RV **omission** on a patient tail, so the levers are
**RV detection/recall** and **RV-shape coverage**, NOT color/heterogeneity.

**Tax vs inadequacy — RESOLVED (49b7, 2026-07-16, `2026-07-16_repaint_failure_tax_verdict.md`).** Looked at the
repaint model's actual errors (not input stats). The color gap **decomposes**, it is not one uniform tax:
- **RV recall collapse** (largest chunk): RV 0.37–0.44 on unseen GE/Canon vs 0.59 on cmrx, HD95 50–103 mm —
  vendor-gated recall failure (the nttu defect amplified under OOD color), **targetable**, lever in hand (nttu.7).
- **shared-mapping tax** (real, partial): on truly-OOD SCD the real model drops 0.83→0.71 while repaint stays
  domain-flat ~0.57 → the real↔repaint gap narrows **0.25→0.14**. Half the LV color advantage evaporates OOD, as
  the tax predicts — so that half is the price of randomization and **synth's value is the unseen-domain regime**.
- **residual LV inadequacy** (~0.14, survives truly-OOD): real still beats repaint where the tax says it
  shouldn't → a **findable** synth defect, not a law. Do NOT accept 0.61 as a ceiling; do NOT pivot wholesale to
  learned color on an "irreducible tax" story (only half-true).

Lever ranking out of the verdict: (1) ~~productionize RV logit-bias (we55)~~ **DONE → declined as a global
default**: the val-fit global RV bias helps where RV collapses (Canon +0.044/Siemens +0.014) but **over-segments
GE where RV is already healthy** (RV −0.106), so pooled cross-vendor is **−0.004** — the RV deficit is
vendor-heterogeneous, a global constant can't fix it. `Inference.logit_bias` primitive kept (opt-in, off); a
**conditional/confidence-gated** RV prior (ru27) **DONE — correct instrument, marginal + un-tunable gain**: the
gate (`gated_biased_pred`, `--mode gated`) recovers Canon's collapse (RV +0.018 @τ=0.85) while sparing healthy
GE (−0.002) where the global bias wiped it (−0.106) — vendor-heterogeneity *is* separable by the model's own
per-slice confidence. But the omission is a small apical tail (fires on 1–8% of slices), safe ceiling ~Canon
+0.006 mean / pooled ≈0, and the gain lives on unseen collapsed vendors the leak-free val (healthy acdc) can't
fit against → not a shippable lever, real RV fix is at **source** (recall/coverage). (2) name the
residual-inadequacy axis
(boundary contrast / finite-res PV) + test one arm; (3) `hpy` (MRXCAT2 MRI-native contrast) justified by the
residual, `ncph` (twin) is the home for the *tax* portion only.

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

**Source RV class-weight — REFUTED (2026-07-16, matched 4-point sweep).** The nttu.7 "×3 RV +0.022" did **not
reproduce**. Ran base vs `ce_weight` ×1.5 / ×2 / ×3 (MONAI `DiceCELoss(weight=)` weights **both** Dice+CE — the
full nttu.7 instrument), matched 40ep single-seed off `--from-config refac_proc`. Every up-weight is **≤ base on
RV and mean**, both splits (TEST Canon+GE n=147): base RV 0.427/mean 0.542 · ×1.5 0.363/0.506 · ×2 0.426/0.530 ·
×3 0.312/0.492. No arm lifts RV; cav bleeds (0.738→0.680). nttu.7's single point was **noise** (single-seed
cross-vendor RV scatters ±0.02–0.03). **KILL.** Combined with we55 (global logit-bias) + ru27 (gated logit-bias),
the RV-recall lever is now **exhausted on both axes** — post-hoc *and* source. Global reweighting (CE, Dice, or
logit) cannot fix a **localized apical-slice detection** failure: it only trades cav mass, never creates RV where
the detector is absent. The RV-collapse chunk is not cheaply recoverable — the real fix is coverage/architecture
or accept it. Reusable infra kept: `DiceCECfg.ce_weight` (no-op default) + `train --from-config`.

**RV deficit SHAPE — it's not omission, it's a partial-quality continuum (egeh, 2026-07-16, `--mode deficit`).**
Chased the "coverage/architecture" RV lever to its root. Per-slice RV Dice over 1165 GT-RV-present cross-vendor
slices: **8% <0.05 · 11% [0.05,0.3) · 20% [0.3,0.6) · 26% [0.6,0.8) · 34% ≥0.8**, mean **0.600**. The gap is a
**broad continuum**, not a cliff. **True 0-px omissions = 3 slices (0.26%)** — negligible; all apical, all with a
confident-RV neighbour at z±1 (2.5D signal exists but recovering 3 slices is worthless). Of the 93 near-miss
(<0.05) slices, only 3 are absent — the other ~90 predict RV in the **wrong place** (mislocation, not omission).
So coverage is dead (shape present, nttu.5) AND omission is a non-lever (3 slices). The RV deficit is broad
partial-quality + mislocation on OOD-color vendors = the verdict's vendor-gated RV-under-color, **capped**. A 2.5D
build would be a speculative general-quality bet against a per-slice-color root — not pursued. **The RV chapter is
closed: no cheap RV win exists.** Frontier shifts off RV → residual-LV-inadequacy (uw5p) / new generation sources.

**nttu epic CLOSED (8/8, 2026-07-16).** Diagnostics committed (nttu.6): `python -m cardioseg.evaluation
rv_omission --run <zero-real> --mode {probe,bias}` reproduces the nttu.5 recall-vs-coverage split + the nttu.7
logit-bias sweep. Coverage levers (nttu.4/.8) resolved by root cause — the model fires RV softmax on the failing
slices, so those shapes ARE covered; the gap is argmax recall, not coverage.

**Open next:** (1) optionally productionize the RV logit-bias as an opt-in inference class-prior (small, val-fit;
label post-hoc like tb58) — filed, not urgent (gain modest, +0.012 test mean). (2) nttu.4/.8 shape-coverage
**deprioritized** for omission (root cause isn't coverage). Do **not** pivot off A. Method note: always run the
MATCHED default baseline before claiming a mover; a monotone dose-response beats a single-seed delta.
