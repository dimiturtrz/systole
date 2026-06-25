# Known diversity factors — registry

The cross-vendor / cross-centre gap is a sum of **physical variation sources**. Each one can be
handled by **two opposing forces**, and they share the same underlying physics model:

- **Augment** (add it) — generate the variation, train on all versions → model learns invariance.
- **Normalize** (strip it) — estimate the variation, remove it → model never sees it (train AND test).

**The duality:** the forward model that *generates* a factor is the inverse of the estimator that
*removes* it. Write the physics once, run it both directions. (N4 = inverse of bias-field generation;
histogram match-to-vendor = same machinery as match-to-reference.)

**Rule: per factor, pick ONE direction.** Stripping AND augmenting the same axis = adding back what
you removed. Strip when the factor is pure nuisance and reliably estimable; diversify when stripping is
unreliable or discards signal.

## Registry

| Factor | Physical cause | Augment (add) | Normalize (strip) | Current call | Status |
|---|---|---|---|---|---|
| Per-volume brightness/scale | acquisition gain, windowing | gamma/contrast (have) | z-score per volume (have) | normalize (z-score) + coarse augment | ✅ both, crude |
| Coil sensitivity / B1 | receive coil geometry → smooth spatial brightness ramp | bias field (Tier 1 / jp1) | N4 bias correction | **undecided** — A/B strip vs diversify | ⬜ neither yet |
| Vendor intensity distribution | vendor T1-weighting, flip-angle, recon LUT | random gamma + histogram retarget-to-vendor (Tier 1) | histogram standardization (Nyúl, → ref) = harmonization `qfz` | diversify (qfz parked: vendors level in-domain) | ⬜ augment side open |
| Noise floor / distribution | magnitude operation on complex signal | Rician noise (Tier 1) — we use plain Gaussian (wrong shape) | denoise / variance-stabilize | diversify (Rician) | ⬜ |
| k-space artifacts | corrupted/missing k-space points, motion during acq | ghosting, spike (Tier 1) | de-ghost / artifact reject | diversify (rare, hard to strip) | ⬜ |
| Geometry / orientation | patient pose, FOV, slice prescription | flip/rotate/scale (have) | resample to common grid (have, spacing-aware) | both | ✅ |
| Contrast space (whole) | everything above, jointly | SynthSeg label→random-intensity (Tier 2) | — (can't strip into a canonical contrast cheaply) | diversify (contrast-agnostic) | ⬜ Tier 2 |
| Full acquisition physics | sequence params TR/TE/flip, tissue T1/T2/PD | CMRsim Bloch sim (Tier 3) | — | diversify (maximalist) | ⬜ Tier 3 |

## Notes
- **Spacing/orientation** is the one factor where we already do both cleanly (resample = strip;
  affine aug = diversify) — proof the pattern works.
- **Bias field** is the highest-value undecided row: build the smooth-field model once for jp1, get N4
  almost free, A/B the two directions on this single axis even while full harmonization (`qfz`) stays
  parked on evidence (in-domain M&M-2 vendors are level — no minority-vendor deficit).
- **Harmonization (`qfz`)** is the normalize column of the "vendor intensity distribution" row —
  deprioritized, but the augment side of the same row (histogram retarget) is live in Tier 1.

Linked: ROADMAP synth-data tiers; `bd cardiac-seg-{chm,jp1,bgc,276,qfz}`;
`research/deep_dives/2026-06-24_mri-sim-libs-eval.md`.
