# Zero-real ceiling, lit-grounded: is 0.61 far below achievable, or near it?

**Date:** 2026-07-16 · **Scope:** cross-task (zero-real generation lane) · **Beads:** completes the converging
"lit-ground the ceiling" item · **Status:** strategic verdict (literature + this-session exhaustion)

Gate on whether to keep drilling the ~0.14 zero-real residual (49b7 verdict) or declare near-ceiling. Pairs the
literature (`research/deep_dives/2026-07-16_synth_only_cardiac_ceiling.md`, citations [S1]–[S14]) with what every
parametric micro-lever did this cycle. Haiku gathered the numbers; the judgment here is ours.

## The question

Our zero-real synth-only cross-vendor SAX Dice is **~0.61** (mean RV/myo/LV-cav); real-trained is **~0.85**. The
49b7 verdict split the 0.24 gap into tax (evaporates OOD) + RV collapse (now closed) + a **~0.14 residual LV
inadequacy** it called "findable — don't accept 0.61 as a law." Is that residual actually findable?

## What the literature says (and the one comparator that matters)

| result | zero-real? | test | mean-ish Dice | why it does / doesn't bound us |
|---|---|---|---|---|
| **[S12] label-space synth** | **yes** (image-to-image from labels, no real imgs) | **MSCMR** (real, multi-seq) | Myo .57 / LV .78 / RV .63 ≈ **0.66** | **the apples-to-apples comparator — a hard real multi-domain cardiac set, zero real images in training** |
| SynthSeg (GMM random) [S2] | yes | **MMWHS** (whole-heart, n=20, +CT) | ~0.87 | **easier target** — single-centre whole-heart, not multi-vendor SAX cine; not our regime |
| XCAT-GAN [S11] | **no** | ACDC ED | 0.94 | GAN **trained on real image pairs** → violates zero-real + no-external-dep; out of bounds |
| UltimateSynth vs SynthSeg [S6] | yes | brain | +7 pts (physics>random) | confirms our **physics** direction; no cardiac number; we already do this |
| nnU-Net real [S10] | — | ACDC/M&M | RV .92–.96 | upper reference |

**No published pure-zero-real number exists on ACDC/M&M SAX** — the exact benchmark. We are in **unmapped
territory**, which is why the honest triad (real / synth-only / synth+DA) is itself the contribution.

## Verdict — 0.61 is in-band with the achievable pure-parametric zero-real ceiling

- The **only** comparable zero-real cardiac result on a hard real set is **~0.66** ([S12]). Our **0.61** is right
  next to it — not dramatically below. The eye-catching high numbers are on an **easier target** (SynthSeg /
  MMWHS) or use a **real-trained generator** (XCAT-GAN), neither of which is our constraint.
- **This corroborates the session's exhaustion.** Every parametric micro-lever is spent: appearance/color
  (nk70), boundary/resolution (uw5p → nk70.2 combined config *collapsed* to 0.314), RV recall (we55/ru27/dbpp
  post-hoc + source), RV coverage/omission (egeh — deficit is a continuum, 3 hard omissions), shape-coverage
  (composite dice-neutral). Independent lines converging on the same place is what a ceiling looks like.
- **The owner's "SynthSeg reaches higher, don't accept 0.61" counter is weakened, not confirmed.** SynthSeg's
  higher number is a different, easier task; on a comparable hard cardiac set the zero-real ceiling is ~0.66, and
  we are there. The 49b7 residual is therefore mostly the **randomization tax + intrinsic multi-vendor-cine
  hardness**, not a large findable parametric inadequacy.

## What this decides

- **Stop chasing zero-real Dice with parametric micro-levers.** Lit + exhaustion agree we are near the
  pure-parametric ceiling for this hard regime. Further appearance/boundary/recall tweaks are measuring a
  near-ceiling method harder.
- **The deliverable is the honest, lit-contextualized triad** — 0.61 is not a failure; it is in-band with the one
  comparable published zero-real cardiac number (~0.66), while higher numbers use easier targets or real-trained
  generators. State this scope up front.
- **The additive directions are different value props, not more zero-real Dice:**
  1. **Digital-twin / inverse** (`ncph`) — controlled fidelity to a scan; the project's 2nd named direction, a
     distinct value prop, unaffected by the uncontrolled-diversity ceiling.
  2. **Synth-as-augmentation on top of real** (`pwih`) — the deployable question; no clean matched A/B yet.
  3. **`vpn5` learned shape prior** — the one *structural* (not micro-tweak) zero-real lever left, but the lit
     gives **no strong warrant** that shape closes the multi-vendor-cine gap (shape-coverage was dice-neutral,
     and the comparable learned-generation result [S12] is still ~0.66). Exploration, not a predicted win.

## Honesty / caveats

- No pure-zero-real ACDC/M&M SAX number exists, so "0.61 ≈ ceiling" is an **inference** from the nearest
  comparator ([S12] ~0.66, MSCMR) + convergent internal exhaustion, not a matched external benchmark.
- [S12]'s test set (MSCMR) is not ACDC/M&M; magnitudes aren't perfectly matched. The **direction** (a hard
  zero-real cardiac set lands in the low-0.6s) is the load-bearing fact, not the exact 0.66.
- UltimateSynth's physics gain is brain-only; a cardiac physics-vs-random head-to-head is unpublished (an open
  experiment, but we already sit on the physics side).
