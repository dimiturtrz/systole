# Zero-real tax diagnostic — verdict from the model's actual errors

**Date:** 2026-07-16 · **Scope:** cross-task (zero-real generation lane) · **Bead:** cardiac-seg-49b7 ·
**Status:** measured verdict (single-flagship, eval-only, no retrain)

Gate before any generation-direction pivot (ncph / hpy). Tests the **domain-randomization-tax** hypothesis
for the zero-real color gap by looking at *what the model gets wrong* — the check the whole xmcf epic skipped
(it stayed in input-distribution-stats, the frame nk70.1 proved doesn't predict Dice). Two eval-only arms on
the **repaint** model `bssfp` (real masks + `synth_p=1.0` randomized synth color = color axis isolated).

## The prior question

`interpretations/converging/2026-07-16_what_is_wrong_with_our_synth.md` framed it: is the ~0.17 color gap a
**tax** (domain randomization forfeits a stable intensity→tissue map the not-very-OOD test shares with real
training) or a **synth inadequacy** (our manifold doesn't cover real; physical synth-only SOTA reaches higher,
so 0.61 isn't a law)? Owner's decisive counter: SynthSeg trains synth-only and nears real, so the tax can't be
the whole story — leaning on it is a cope. The doc prescribed: **failure analysis of the repaint model** +
**a truly-OOD test**. This is that.

## Arm 1 — repaint failure signature (distribution.py, ED+ES pooled, TTA, largest-CC)

| eval set | n | RV | myo | cav | mean | RV HD95 | cav HD95 |
|---|---|---|---|---|---|---|---|
| GE (mnm2, leak-guarded) | 53 | 0.444 | 0.614 | 0.586 | 0.548 | 50 mm | 88 mm |
| Canon | 9 | 0.374 | 0.593 | 0.611 | 0.526 | 103 mm | 20 mm |
| cmrxmotion | 69 | 0.589 | 0.583 | 0.573 | 0.582 | 21 mm | 84 mm |

Stratified (the discriminator — *targetable subset* vs *uniform*):
- **GE by scanner:** SIGNA EXCITE 0.535 / Signa HDxt 0.555 — uniform.
- **GE by pathology:** rv_congenital 0.597 / normal 0.523 / other 0.451 / HCM 0.552 / dilated 0.621 — **no
  pathological subset carries it; "normal" is not better than diseased.**
- **GE by field:** 1.5T 0.537 / 3.0T 0.586 — uniform.
- **cmrx by motion grade:** 0.569 / 0.596 — uniform.

**Two things are true at once:**
1. **LV (myo+cav) is uniform ~0.58–0.61 across every stratum** — vendor, scanner, pathology, field, motion. No
   targetable subset. This is the *uniform* component the tax predicts.
2. **But it is NOT gentle low-contrast boundary erosion** (the tax's specific prediction of moderate HD95). HD95
   is **20–103 mm** — gross whole-slice / whole-structure misses, not a few mm of boundary slip.
3. **RV collapses, gated by unseen vendor:** GE 0.44 / Canon 0.37 (HD95 50–103 mm) but cmrx 0.59. The RV detector
   fails *exactly where color is most OOD* — the nttu RV-omission (recall-vs-background) defect, amplified under
   color shift. This is **targetable** (nttu.7 logit-bias already recovers it), **not** a color tax.

So the repaint gap is **not one uniform boundary tax**. It is: a uniform LV color component + a vendor-gated RV
recall collapse (the single biggest chunk of the cross-vendor number).

## Arm 2 — truly-OOD test (matrix.py, frozen manifests, per-model OOD guard)

| model | canon (real's turf) | ge (real's turf) | scd_lv (truly-OOD, LV-only) |
|---|---|---|---|
| **real** (v60, synth_p=0.5) | 0.846 `[LEAK]` | 0.832 `[LEAK]` | **0.707** `[OOD]` |
| **repaint** (bssfp) | 0.574 `[OOD]` | 0.604 `[OOD]` | **0.569** `[OOD]` |
| gap | 0.27 | 0.23 | **0.14** |

The real model **saw** Canon+GE (in-domain, marked LEAK) — its 0.84 there is with its stable mapping intact. On
**SCD** (a GE-vendor set neither model trained on, genuinely OOD) the real model **drops ~0.13 to 0.707**, while
repaint is **domain-flat** (~0.57–0.60 everywhere). **The gap closes from ~0.25 to ~0.14 exactly as the tax
predicts:** the real model forfeits its shared-mapping advantage when the domain is truly unseen, and synth's
invariance partially pays. (Caveat: SCD is LV-only, so real's drop also sheds its strong RV; repaint is flat
regardless, so the direction is unambiguous even if the magnitude isn't perfectly matched.)

EF corroborates: real EF MAE stays low even OOD (SCD 3.45%) while repaint is 14–21% everywhere — EF needs the
volume mapping repaint never had.

## Verdict — it is BOTH, in named proportions

| component | size (cross-vendor) | nature | lever |
|---|---|---|---|
| **RV recall collapse** | the largest chunk (RV 0.37–0.44 vs LV 0.60) | targetable defect, vendor-gated | nttu.7 logit-bias / RV-targeted recall — **already have it** |
| **shared-mapping tax** | ~half the LV color gap (0.27→0.14 OOD) | real, by-design | evaporates OOD → **synth's value IS unseen-domain** (north-star validated) |
| **residual LV inadequacy** | ~0.14, survives truly-OOD | genuine synth defect | boundary contrast / under-modeled physics — findable |

- **The tax is real but partial** — ~half the LV color advantage is the shared mapping, and it *does* evaporate
  on truly-OOD real. That half is not a synth defect; it is the price of randomization, and it confirms synth's
  actual value is the unseen-domain / no-annotation regime.
- **Owner's counter stands:** a residual ~0.14 gap survives even truly-OOD, where the tax says there should be
  none. That residual is a **findable inadequacy**, not a law — do not accept 0.61 as a ceiling.
- **The biggest single lever is not color at all** — it is RV recall under OOD color (a targetable recall
  defect), which we already have a leak-clean inference lever for.

## What this decides (directions map)

- **Do NOT pivot wholesale to learned/inverse color** on a "tax is irreducible" story — the story is only
  half-true and the residual is targetable.
- **Rank the levers:** (1) RV recall under OOD color (productionize nttu.7 logit-bias into the zero-real lane;
  bd we55) — biggest chunk; (2) the residual LV inadequacy — name one axis (boundary contrast / finite-res PV)
  and test one arm before hardening; (3) `hpy` (MRXCAT2 MRI-native contrast) is justified by the residual
  inadequacy, not refuted by the tax.
- **`ncph` (inverse twin)** remains the right home for the *tax* portion (reduce contrast variance toward a known
  deployment domain), but it is a twin move that trades generalization — not the fix for the residual inadequacy.

## Honesty / caveats

- Single-flagship, eval-only (no retrain, no seeds) — per finding-stage rigor; the effects are large (0.25 vs
  0.14 gap; RV 0.37 vs LV 0.60) and multi-set, well above single-seed scatter.
- The repaint cross-vendor mean measured here (~0.53–0.55) is **below the 0.68** cited in prior docs. The 0.68
  was likely ED-only or a different repaint variant (`ph_ks`/`ph_pvks`) or pre-regression; not resolved here.
  The **signature and the OOD narrowing** — the actual question — do not depend on the absolute.
- `v60` is `synth_p=0.5` (real+50% synth), not pure real, and it **saw** Canon+GE (LEAK there) — it is the
  real-informed baseline, not a clean cross-vendor holdout. The clean matched-OOD point is SCD (both OOD).
- SCD is LV-only, so the RV component is excluded from the truly-OOD gap; the LV narrowing is the honest signal.
