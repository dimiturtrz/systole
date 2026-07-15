# Within-class spread is structural, not thermal — the noise knob was a fake texture stand-in (nk70.1)

**Date:** 2026-07-15 · **Lane:** synth appearance-correctness epic (nk70) · **Task:** nk70.1

## Question

After the composition+inflow fixes (zero-real Dice 0.554→0.613), synth myo|cav per-slice d′ overshot to
0.69× real — mild *under*-separation, driven by too-high within-class spread. The ticket hypothesised the
source was **across-image z-score variance** from the new torso composition (each image normalised
differently → myo z varies image-to-image → high pooled σ). Diagnose data-side (variance mode, no
training), then fix the dominant term physically.

## Finding — the ticket premise is refuted, twice

**1. The excess is within-slice, not across-image.** Post-composition myo per-class σ:

| myo σ | real | synth (noise=0.05) |
|---|---|---|
| pooled (across + within) | 0.511 | 0.669 |
| per-slice (within-image) | 0.279 | **0.552** (≈2×) |

Decompose: across-image σ = √(pooled² − per_slice²) → real **0.428**, synth **0.378**. Synth's
across-image variance is *below* real — composition/z-variance is fine. The entire excess is **within-slice**,
and the knob-off attribution is unambiguous: turning noise off drops per-slice myo σ by 0.501 (0.552→0.051).
It is the **noise** term.

**2. The noise magnitude was physically absurd.** `noise=0.05` is a flat absolute Rician σ (paint units,
pre-z-score) — a magic constant. Against the bSSFP signal it implies **myocardial SNR ≈ 1.1** (S_myo=0.056
paint units / 0.05). Real cardiac cine bSSFP myocardial SNR is **~35–65 at 1.5T** (S1/S5/S6;
`research/deep_dives/2026-07-15_cardiac_bssfp_snr.md`). The knob was ~60× too noisy.

**3. At physical SNR, thermal noise is negligible — so real's within-class texture is STRUCTURAL.**
Setting σ = S_myo/SNR at any physical SNR (>~30) makes thermal contribution ~0.0009 paint units. Synth
within-slice then collapses to the structural floor:

| per-slice σ | real | synth @ snr=45 |
|---|---|---|
| RV | 0.616 | 0.194 |
| LV-myo | 0.279 | 0.199 |
| LV-cav | 0.75 | 0.170 |

All heart classes fall **under** real; myo|cav per-slice d′ = 6.06 = **2.29× real** (now *over*-separated —
synth too clean). The flat noise had been a **fake stand-in for structural texture** — unphysical difficulty
masking that synth is structurally too smooth (same family as the rejected over-blur hack).

**Class-dependent signature the flat knob could never reproduce:** real within-slice σ is blood **>** myo
(cav 0.75, RV 0.616 ≫ myo 0.279). That ordering is a **flow / papillary-trabecular** signature in the blood
pools, not thermal (thermal is class-independent). A single additive σ cannot produce it.

## The kill — noise=0.05 is load-bearing AUGMENTATION, not fidelity

Built the SNR-parameterised refactor (`snr=45`, σ = S_myo/snr, leak-free) and ran the zero-real arm
(synth_main --quick) against the 0.613 baseline. It **collapses**, across every class:

| zero-real Dice | baseline (noise=0.05) | snr=45 (physical) | Δ |
|---|---|---|---|
| VAL mean | 0.676 | **0.395** | −0.28 |
| TEST mean | 0.613 | **0.385** | −0.23 |
| TEST myo / cav | — | 0.337 / 0.483 | — |

−0.23 is not a dip, it's a collapse. Mechanism: at physical SNR the effective heart-region noise is ~0.001
(negligible), so synth heart regions go **pristine**; the model overfits to clean synth and fails on the
textured real images. **The `noise=0.05` knob was the model's only source of within-class heart texture —
serving as essential AUGMENTATION (robustness to real texture), not as a fidelity claim.** Its two roles —
fake-fidelity (wrong) and texture-augmentation (load-bearing) — were conflated in one knob. Removing it to be
"physically correct" strips the augmentation and tanks generalisation.

This is the *physically-correct-method-regresses* case (owner directive): the mechanism (physical SNR) is
sound but **mis-applied in isolation** — it removed a load-bearing augmentation without the physical
replacement. The regression is the QUESTION answered, not a verdict on the physics.

## Action — reverted, re-scoped

- **Reverted** the SNR refactor; `noise=0.05` stays (working 0.613 pipeline intact, non-regressed). It is
  documented now as an empirical **texture-augmentation stand-in**, not thermal fidelity.
- **Re-scoped nk70.1**: the fix is NOT "remove fake noise" — it is "supply the within-class texture real has
  through PHYSICAL structural mechanisms, THEN drop thermal to physical SNR and verify Dice holds." The
  structural texture (leak-free, geometry-derived): **partial-volume from finite resolution** (6–8 mm slices
  average trabeculae/blood into myo; in-plane PSF) → **uw5p**; **papillary/flow structure** in the blood
  pools (which also produces the real blood>myo within-slice ordering flat noise can't) → **fi33**. nk70.1
  becomes structural-texture-first, tightly coupled to uw5p/fi33 — not a hand-off. Only once physical texture
  reproduces the real within-class spread (and holds Dice) does the SNR-param thermal correction ship.

## The follow-up that killed the whole thesis

The re-scope said: supply the structural texture physically (pv + trabec), *then* lower noise. Built both
(uw5p `pv_sigma`, fi33 trabecular-PV) and ran the combined arm — physical texture + noise lowered:

| zero-real TEST | baseline | noise↓ only | noise↓ + pv + trabec + kspace |
|---|---|---|---|
| mean | 0.613 | 0.385 | **0.314** |

Physical texture **does not recover** the collapse. So noise=0.05's role is **not** the structural
within-class texture (which pv/trabec reproduce in the per-slice σ statistic) — it is **high-frequency
pixel-level augmentation** (robustness to real texture) that structured PV/trabeculae do not provide.
**The within-class-sd / d′ correctness target is refuted as a Dice lever**: the noise-driven over-spread IS
load-bearing augmentation, and you cannot lower it toward real's σ without losing generalisation, physical
replacement or not.

## Honesty

d′ is a readout, never the target — and here it turned out not to be a Dice lever at all. Two training arms
(noise↓ −0.23, noise↓+texture −0.30) settle it, single-seed but ≫ the ±0.02 noise floor and consistent
across val+test. What survives from this thread: **level/mean** correctness (composition, which fixed a big
myo miss and did help), not **spread** correctness. See `nk70` epic verdict.
