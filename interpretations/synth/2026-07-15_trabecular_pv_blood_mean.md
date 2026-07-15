# Papillary/trabecular partial-volume: the leak-free blood-mean lever — and a fidelity-bg footgun (fi33)

**Date:** 2026-07-15 · **Lane:** synth appearance-correctness epic (nk70) · **Task:** fi33

## Mechanism

Papillary muscles and trabeculae are **myocardium-signal structures inside the blood pools**. On a
short-axis cine slice they partial-volume with blood → darker, heterogeneous blood. Modelled as: a cited
blood-pool volume fraction (`trabec_lv` / `trabec_rv`) of blood pixels mixes toward the myo signal,
spatially-correlated at a ~3 mm trabecular scale. One physical cause, two matched effects — **lowers blood
mean** and **adds within-class blood texture**. Cited fraction: **LV 11.9 ± 5.6 %** (JCMR; deep-dive
`2026-07-15_papillary_trabecular_volume_fractions.md`). RV is more trabeculated but has no clean cavity-%
citation (only 44.6 ml/m² absolute) → anatomical ~2× LV estimate, flagged.

## The footgun that nearly inverted the finding

`SynthFidelity` builds a default `TrainCfg()`, whose default `SynthCfg.bg` is **`PartitionBg`** (partitions
the *real* per-slice intensity for background). But zero-real training (`synth_main`) uses **`ProceduralBg`
torso composition**. Background sets the whole-FOV histogram → the per-image z-score → **every heart-class
level**. Measured at the wrong bg, the blood-mean gap **reverses sign**:

| z-mean | real | synth @ PartitionBg (wrong) | synth @ ProceduralBg (correct) |
|---|---|---|---|
| RV | 1.543 | 0.672 (too **dark**) | 2.056 (too **bright**) |
| LV-cav | 1.630 | 0.696 (too **dark**) | 2.032 (too **bright**) |

At PartitionBg the trabecular darkening looked *harmful* (synth already too dark); at the correct
ProceduralBg it is the **right** direction. Same knob, opposite verdict — from bg alone. **Rule:** any
`synth-fidelity` run analysing the zero-real generator must pass `--set synth.bg.mode=procedural`.
Paint-driven *relative* knob effects are ~robust to bg; *absolute* z-means/sds are not.

## Validation (correct bg)

Trabecular-PV lowers the blood mean toward real:

| z-mean | real | trabec OFF | trabec ON (rv .25 / lv .12) |
|---|---|---|---|
| RV | 1.543 | 2.056 | **1.813** (closes ~48 %) |
| LV-cav | 1.630 | 2.032 | 1.970 (~15 %; LV fraction small) |

Correct direction, cited, leak-free — the ticket's +2.04 → +1.66 lever, working. Texture: RV rises strongly
(overshoots at trabec_rv 0.25 on procedural bg → the RV fraction needs *physical* calibration, not tuning to
the target); LV-cav rises moderately.

## Scope / remaining

- **Built + gate-clean:** trabecular-PV paint mechanism, default OFF (turned on with noise lowered at the
  nk70.2 gate, per the coupling in `2026-07-15_within_class_spread_structural.md`).
- **Remaining (fi33):** the LV-cav mean residual (+1.97 vs +1.63) and LV-cav within-class texture are
  **flow**, not trabeculae — the flow-dephasing half, which needs a leak-free velocity→signal-loss
  parameterisation (the gradient first moment isn't cleanly available from headers) — the harder physics,
  deferred. RV trabecular fraction needs a physical calibration argument.
- **Verdict on Dice** belongs to nk70.2: turn on trabecular-PV (blood) + pv_sigma (myo, uw5p) + lower noise
  to physical SNR, in one arm; check the physical texture replaces the noise augmentation and Dice holds.

## Honesty

Diagnostics only (no training). The bg footgun means every fidelity number this session that wasn't from a
training arm was on PartitionBg and must be re-read at procedural — the nk70.1 kill (a `synth_main` training
arm, correct bg) stands; the uw5p pv direction (paint-driven) stands; absolute per-slice numbers were
PartitionBg. Trabecular-PV's blood-mean direction is validated at the correct bg; its Dice payoff is
unproven pending the combined gate.
