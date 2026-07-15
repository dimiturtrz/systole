# d′→Dice proxy: over-separation is a real lever but a minor slice of the ceiling

**Date:** 2026-07-15 · **Task:** synth generation fidelity · **Beads:** 5xf6 (proxy scatter), m4cp (verdict), epic 2gk5

## TL;DR

Does driving synth's per-slice myo|cav separability d′ toward real's 2.65 — via **physically-derived
acquisition-resolution difficulty** (k-space PSF + partial-volume blur, swept as a domain-rand axis) —
raise zero-real Dice? **Yes, but only a little.** Two findings:

1. **The proxy holds within synth.** Across 4 zero-real `synth_main` arms (quick, seed 0, matched budget),
   lower measured d′ ⇒ higher frozen-test Dice: **Pearson r = −0.82**, monotonic, **+0.034 test / +0.039
   val** at the most aggressive regime, same direction on two independent real sets (frozen 642 + ACDC-val).
   Over-separation is a **real, physical, leak-free** lever — the difficulty comes from acquisition geometry
   (finite resolution), not a knob tuned to d′.
2. **But it's a minor slice of the ceiling.** The most aggressive regime reaches d′ 2.87 ≈ real's 2.65 —
   yet its Dice is 0.588, still **0.27 below** real-trained 0.854. **Matching d′ closes only ~12% of the
   zero-real gap.** Over-separation is not what caps zero-real synth; the dominant residual is elsewhere
   (shape-generation cost + appearance/domain gap).

## The scatter (procedural bg = the training config; d′ measured on it, matched)

| regime (kspace / pv_sigma) | measured d′ (proc bg) | test Dice | val Dice | myo Dice |
|---|---|---|---|---|
| real-trained (anchor) | 2.65 | **0.854** | — | ~0.80 |
| r1  0.7 / 0.6 (mild) | 3.73 | 0.552 | 0.592 | 0.459 |
| r2  0.5 / 1.0 | 3.26 | 0.567 | 0.610 | 0.488 |
| r3  0.4 / 1.5 (coarse) | 2.87 | 0.588 | 0.631 | 0.498 |
| baseline kspace0 (historical, matched) | 4.55 | 0.554 | — | — |

Pearson r(d′, Dice): **−0.82** over the synth arms; −0.63 including the real anchor (the anchor sits far
off the synth line — same d′, +0.27 Dice — which is finding 2). The lever is **nonlinear**: r1 (mild,
physically-faithful ~2mm resolution) is Dice-flat vs baseline; the gain only appears at r2/r3, where the
blur is aggressive (kspace 0.4 ≈ 3.75mm effective — the coarse edge of the resolution domain-rand axis,
low-field cine territory). So a *little* boundary softening does nothing; you have to push d′ most of the
way to real before Dice moves.

## Why this is the honest reading

- **d′ is a real lever, kept.** Monotonic + two-set agreement + r −0.82 is more than single-seed scatter
  (memory: single-seed cross-vendor noise ±0.01–0.03; the *trend* across 4 arms is the evidence, not any
  one delta). This is the **best single-lever zero-real dent measured on this project** — most generation
  levers (cohort size, bg mode, coverage, contrast) were Dice-neutral (memory: generation-ceiling-robust).
  It is physically-derived (resolution from geometry), self-balancing, and leak-safe — d′ was read out,
  never tuned to 2.65.
- **d′ is not the ceiling.** Equal-d′ synth (r3) still trails real by 0.27 Dice. So the over-separation
  hypothesis, while confirmed as *a* lever, is **rejected as the dominant cause** of the zero-real ceiling.
  The residual matches the known decomposition (memory synth-reporting-taxonomy): generate-vs-repaint shape
  cost (~0.12) + appearance/domain gap. Difficulty-matching fixes neither.

## Consequences

- **2gk5 (epic) verdict:** over-separation *attributed and measured* — a real +0.03 physical lever, ~12% of
  the gap. The ceiling is dented, not broken, by difficulty-matching. Epic's core question answered.
- **KEEP:** fold a **physically-constrained resolution regime** (kspace + pv_sigma swept across the physical
  range, ~r2-strength) into the standing synth generation as a difficulty ingredient — free, physical,
  leak-safe, monotonic ~+0.02–0.03. Filed as a follow-up (do NOT tune to d′; sweep the physical axis).
- **kopr (d′-curriculum):** the proxy held, so a d′-ordered curriculum (easy→hard by resolution) is live —
  but capped upside (~+0.03 headroom), so P3, not the main thrust.
- **The main lever is elsewhere:** the 0.27 residual at matched d′ says the dominant zero-real gap is
  shape-generation + appearance/domain — pursue augmentation (real+synth) / inverse digital-twin, not more
  difficulty-matching. That is the epic's pivot.
- d′ stays a **readout**. Nothing was tuned to it; the physical regimes were swept, Dice measured, trend read.
