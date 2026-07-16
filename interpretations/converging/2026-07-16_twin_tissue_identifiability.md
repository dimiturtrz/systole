# Digital-twin, tissue direction: identifiability verdict

**Date:** 2026-07-16 · **Scope:** inverse / digital-twin (direction D) · **Beads:** ncph / 5ev5 · **Status:**
measured verdict (synthetic recovery + real ACDC)

The project's 2nd named direction is the INVERSE/twin: given a real scan + its seg, fit the generator's physics
params that regenerate it (interpretable qMRI / harmonization). `ixea` proved single-frame *acquisition* (TR/flip)
is not identifiable from the heart (2 tissue levels + an affine). This extends the fit to per-class *tissue*
contrast (T2 + relative PD) over the same-session {ED, ES} set (`Inverse.fit_tissue`, `python -m core.data twin
--mode tissue`), and measures the identifiability directly.

## The mechanism + the switch

The fit exposes an `absolute` switch that is the whole story:
- **calibrated** (`absolute=True`) — loss on raw signal levels. The 2 heart levels are 2 absolute constraints →
  tissue params **recoverable**. Synthetic round-trip (plant T2 off the prior, recover): **margin 0.23** (params
  move off the prior toward truth), recon exact.
- **uncalibrated** (`absolute=False`) — loss on the CONTRAST after removing one joint affine, which is what real
  cardiac MRI (arbitrary receiver gain) forces. 2 levels + a 2-dof affine leave **0 contrast dof** → the fit
  **collapses to the literature prior**. Synthetic: **margin 0.001**.

The `margin` (max |log-deviation| off the literature prior) is the readout: high = data pins the params, ~0 =
the prior fully determines them.

## Real ACDC (patient001, {ED, ES})

| regime | recon_loss | margin | read |
|---|---|---|---|
| **uncalibrated** (correct for real) | 0.315 | **0.082** | params ≈ prior (blood T2 244/256 vs prior 250; myo 40.1 vs 40) — data barely moves them |
| calibrated (wrong for real) | 0.975 | 2.524 | garbage — chasing the arbitrary gain drives T2 to 374/550, recon *worse* |

Two facts:
1. **The tissue-level twin is under-determined on standard cine.** Under the correct uncalibrated treatment the
   margin is 0.082 — an order below the calibrated-synthetic 0.23; the fit returns ~the literature prior. The twin
   as a *qMRI product* needs calibrated or varied-flip/TR (real qMRI) input, which ACDC/M&M don't provide.
2. **The flat forward model leaves a large residual** (recon_loss 0.315, not ~0). Real cardiac has within-tissue
   STRUCTURE — papillary muscle, flow, partial volume — that a 2-level constant-per-class render omits. This is
   the same "synth is too clean" gap seen forward-side (over-separation / within-class spread), now measured from
   the inverse.

## Verdict

- **Tissue qMRI twin from standard cine: blocked by identifiability**, confirming + quantifying ixea. Not a
  parameter-recovery product on ACDC/M&M — it returns the prior. Would need a qMRI protocol (varied flip/TR, same
  session) or absolute calibration.
- **The FIT operator is correct and reusable** (`fit_tissue`, differentiable, converges; the `absolute` switch is
  the identifiability instrument). It is the right substrate IF calibrated/multi-acquisition input arrives.
- **The forward-model residual (0.315) is the more useful signal** — it says the physical generator lacks
  within-tissue structure, corroborating the forward-side finding from the opposite direction.
- Twin's remaining *usable* value on this data is **harmonization** (re-render at canonical acquisition), which is
  a 2-level affine remap here — a separate, lower-novelty use; not pursued now.

## Honesty / caveats

- One real case (patient001); the margin is a single-case readout, but the mechanism (2 levels + affine → 0 dof)
  is structural, not sample-dependent — the synthetic test locks it deterministically.
- ED+ES share receiver gain but also share acquisition, so they add no new contrast equation — K=2 here is about
  the shared-scale discipline (`5ev5`), not disentangling T1 vs T2 (which needs varied flip/TR).
