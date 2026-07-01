# cardioseg/preprocessing/normalization — domain shift, handled at the right layer

The central obstacle to a cardiac-MRI model trained on one set generalizing to another is **domain
shift**: the train and deployment image distributions differ. MRI makes it acute because intensity is
**uncalibrated** — unlike CT's Hounsfield units, a cine-MRI pixel value has no absolute meaning (it
depends on coil gain, recon auto-scaling, sequence weighting), so the same tissue reads differently
across scanners. This package is how we name that variance and decide what to do with each kind.

## The ML framing (so the vocabulary is standard)

- **Domain shift** (a.k.a. dataset / distribution shift) — the umbrella. Decomposes (Moreno-Torres 2012)
  into **covariate shift** (P(x) changes, P(y|x) stays — *the scanner/intensity case*: images differ,
  the anatomy→label relationship doesn't), **label/prior shift** (P(y) changes — disease mix per site),
  and **concept shift** (P(y|x) changes — e.g. annotation-protocol differences).
- **Acquisition shift** — the MRI-specific name for scanner/protocol-induced covariate shift; also
  called **scanner / site / vendor effects** or **batch effects** (term borrowed from genomics).
- **Domain generalization (DG)** — training to generalize to *unseen* domains (vs *domain adaptation*,
  which targets a known one). The M&Ms challenge is a multi-vendor DG benchmark; so is our split.
- Task-irrelevant variance is **nuisance variation**. The whole job: separate signal from nuisance.

**Three strategy families attack it — and they are the two forces below:**
| family | what it does | force |
|---|---|---|
| **harmonization / normalization / standardization** (ComBat, Nyúl, N4, z-score) | *remove* the nuisance variance | strip |
| **data augmentation / domain randomization** | *add* variance so the model learns invariance | diversify |
| **domain adaptation** | adapt to a specific known target | (not our setting — we hold targets out) |

## The principle: knowable → correct; unknowable → normalize/diversify

You don't pick physical-vs-statistical by taste. **If the variance is knowable** (a number we can parse,
or a field we can estimate from the image) → remove it at its source. **If it's unknowable** (no
parameter, no anchor) → normalize it statistically, or learn invariance to it via augmentation. Same
question, two answers. Knowability tiers map 1:1 to the right tool:

| tier | source | tool | reproducible? |
|---|---|---|---|
| header | NIfTI affine | read | ✓ deterministic |
| sidecar | Info.cfg / CSV / folders | **parse** | ✓ deterministic ← *the bulk* |
| paper | challenge paper | cite + `verified` flag | sourced, not regenerated |
| image-derived | the pixels | N4 / blood-pool ref / vol-curve | ✓ deterministic |
| (none) | — | statistical normalize / augment / accept | n/a |

## The variance-source taxonomy (5 buckets × 2 forces)

Each row is a source of variance; the shift-type names which P(·) it moves; the two columns are the
*knowable→correct* and *unknowable→normalize/diversify* handling.

| bucket | shift type | knowable → correct / parse | unknowable → normalize / diversify |
|---|---|---|---|
| **machine** (between scanners) | covariate | spacing (header), vendor/scanner/field/centre (sidecar+paper) | recon scale → z-score / Nyúl; contrast → augment |
| **scan** (between scans) | covariate | bias field → N4 (image) | receive gain → z-score |
| **patient** (between people) | covariate + label | age/sex/BSA (sidecar), heart locate+orient (image) | body habitus → augmentation |
| **temporal** (within cine) | — | ED/ES frames (sidecar / vol-curve) | residual motion → augment |
| **annotation** (between labelers) | concept | label convention (geom verify), papillary/basal rule (paper) | inter-observer noise → **irreducible LoA floor** |

The bottom-right is the honest endpoint: even two human experts disagree on EF, so our limits of
agreement **cannot beat the inter-observer floor**. We aim to reach it, not pretend past it.

## The duality: every factor, modeled both directions

The two forces are **duals** — the forward model that *generates* a factor is the inverse of the
estimator that *removes* it. Write the physics once, run it both ways:
- bias field: a smooth-field *generator* (augment) vs **N4**, its *estimator* (strip)
- vendor intensity: histogram *retarget-to-vendor* (augment) vs *match-to-reference* = harmonization (strip)

**Rule — per factor, pick ONE direction.** Stripping *and* augmenting the same axis = adding back what
you just removed (wasteful, sometimes harmful). Strip when the factor is pure nuisance and reliably
estimable; diversify when stripping is unreliable or discards signal.

### Factor registry

| factor | physical cause | augment (add) | normalize (strip) | current call | status |
|---|---|---|---|---|---|
| per-volume brightness/scale | acquisition gain, windowing | gamma/contrast (have) | z-score per volume (have) | both, crude | ✅ |
| coil sensitivity / B1 | receive coil → smooth brightness ramp | bias field (T1) | N4 bias correction (have, opt-in) | **test-neutral on our sample** (augment + strip both regressed on a bias-mild test that lacks the axis); N4 **retained opt-in** for bias-heavy real-world data we can't measure — NOT "dead" | ◐ test-bound |
| vendor intensity dist. | T1-weighting, flip angle, recon LUT | random gamma + histogram retarget (T1) | histogram standardization (Nyúl) = harmonization `qfz` | diversify (qfz parked: vendors level in-domain) | ⬜ aug side |
| noise floor / distribution | magnitude op on complex signal | Rician noise (T1) — we use plain Gaussian | denoise / variance-stabilize | diversify (Rician) | ⬜ |
| k-space artifacts | corrupted k-space, motion in acq | ghosting, spike (T1) | de-ghost / artifact reject | diversify (rare, hard to strip) | ⬜ |
| geometry / orientation | pose, FOV, slice prescription | flip/rotate/scale (have) | resample to common grid (have) | both | ✅ |
| contrast space (whole) | everything above, jointly | SynthSeg label→random-intensity (T2) | — (no cheap canonical contrast) | diversify (contrast-agnostic) | ⬜ T2 |
| full acquisition physics | sequence TR/TE/flip, tissue T1/T2/PD | CMRsim Bloch sim (T3) | — | diversify (maximalist) | ⬜ T3 |

**Geometry/orientation** is the proof the pattern works — we already do *both* cleanly (resample strips,
affine aug diversifies). **Bias field** is the row we're actively A/B-ing in both directions (Findings below).

## Findings — the augmentation wave hit a model-class / recipe floor (measured)

We pushed the *diversify* side hard and it **did not move the unseen-vendor gap** — and we measured *why*,
which is the result:

1. **Bias-field augmentation regressed.** Adding a smooth bias-field perturbation (p=0.3) made the
   unseen-vendor (Canon) result slightly *worse* — Dice 0.839→0.827, ECE 0.172→0.192, error-detection
   AUPRC 0.600→0.563. "Confidently wrong," not "honestly uncertain" → a self-inflicted distribution
   mismatch (the perturbation was redundant with existing contrast aug + ran post-z-score).
2. **The reducible headroom is small.** Uncertainty decomposition (BALD): on Canon only ~31% of the
   uncertainty is epistemic (reducible) via the weak TTA estimate; a **4-seed deep ensemble** puts it
   lower (~15–18%) — the seeds *agree*, so more of the same recipe / more aug can't help.
3. **Deep-ensembling buys nothing** on unseen vendors (Canon Dice +0.000, GE +0.006 — within noise),
   confirming low headroom.

4. **The strip dual (N4) regressed too — ON OUR TEST.** Normalizing the bias factor (N4, default
   params) made it *worse*, EF most of all: ACDC EF 6.5→7.3%, Canon 11.9→15.0%, GE 11.3→13.5% (Dice
   ~flat, −0.006 on the test vendors). Both directions of the bias-field factor hurt here — augment
   *and* strip.
   **BUT read this honestly (the finite-test caveat):** ACDC/M&Ms are relatively clean / vendor-
   post-processed → mild bias field. Our held-out set (ACDC/Canon/GE) **does not contain the
   bias-field axis**, so "N4 regressed" means *"our test doesn't reward N4,"* NOT *"N4 is useless."*
   N4 is a physically-valid normalization (RF-coil / B1 inhomogeneity is a real artifact); on a
   bias-heavy scanner in the wild — which the effectively-infinite real distribution surely contains
   — it could be essential, and we **cannot measure that** with what we have. So the taxonomy entry is
   **"strip vs diversify: test-neutral on our sample, N4 RETAINED as opt-in for bias-heavy real-world
   data we can't represent"** — not "resolved: neither / dead." Off by default (`n4=True` to enable),
   kept + documented precisely because finite-test-neutrality ≠ real-world-uselessness.

**Conclusion:** the cross-vendor gap is **not reducible by augmentation, same-recipe ensembling, or bias
normalization** — these experiments exhausted our current recipe. The real reducible lever is a
**stronger model class**: nnU-Net (50ep/1fold/2D floor, same split) achieves Canon EF 2.6% and GE EF
4.3% vs ours 11.9% / 11.3% — demonstrating the gap was largely model-class epistemic, not an
irreducible aleatoric floor. (Aleatoric / inter-observer limits are never conclusively proven here —
only bounded below by inter-observer LoA.) A same-recipe ensemble is structurally blind to this; the
duality experiment (augment ↔ strip the same factor) is itself the result: both hurt *on this test*.
Scope it honestly — our held-out set is bias-mild and finite; bias-field normalization (N4) stays a
valid, retained opt-in for the bias-heavy acquisitions the real (effectively-infinite) distribution
contains but our test can't. "Settled" = settled *for closing the cross-vendor gap on our data*, not
"N4 is worthless." Tracked: `bd cardiac-seg-{jp1,chm}`; runs logged in MLflow
(gen / aug_bias / n4 / seeds comparable on canonical axes).

## The diversify force — two distinct families: augmentation vs generation

The *diversify* column splits into two genuinely different things (don't conflate them): **augmentation**
*perturbs real images*; **synthetic generation** *invents images from labels*. Anatomy is always real
ACDC labels (none invents new hearts). Full plan + sim-lib evaluation in
[`research/deep_dives/2026-06-24_mri-sim-libs-eval.md`](../../../research/deep_dives/2026-06-24_mri-sim-libs-eval.md);
tracked `bd cardiac-seg-{chm,jp1,bgc,276}`.

```
AUGMENTATION  real image  ──perturb──>          T1  (physics transforms)        -> training/augment.py
GENERATION    real labels ──paint random──>     T2  synth-appearance (SynthSeg)  -> own module (future)
              real labels ──simulate physics──> T3  synth-physics (CMRsim/Bloch) -> own module (future)
```
- **T1 — augmentation** (`jp1`, recommended first): bias-field + Rician + k-space into
  `cardioseg/training/augment.py`; histogram-match into this package (it needs a vendor reference, so
  it's harmonization, not a random batch op). M&Ms credits intensity aug + histogram matching for the
  vendor gap (Zeng 2021 ≈ 0.905 LV Dice unseen-vendor). Cheapest, highest evidence.
- **T2 — generation** (`bgc`): SynthSeg-style per-label intensity randomization → a contrast-**agnostic**
  model. A *generator*, not augment.py — gets its own module.
- **T3 — generation** (`276`): CMRsim Bloch sim from per-tissue T1/T2/PD; physically grounded, no
  seg-aug precedent. Also its own module.

Augmentation (T1) stays small (a handful of transforms in one file). Generation (T2/T3) is a separate
concern that would grow its own home; it is **not** part of `augment.py` or this package.

## Per-dataset coverage — what each ships vs what we fetch
Legend: **AUTO** parse shipped file · **WEB** fetch from paper (cited) · **ABSENT** → image-derived / unknowable fallback.

| field | ACDC | M&M-2 | M&Ms-1 |
|---|---|---|---|
| in-plane res / slice | AUTO (hdr) | AUTO | AUTO |
| vendor | WEB (Siemens) | AUTO (csv) | AUTO (csv) |
| scanner model | WEB (Aera/Trio) | AUTO (csv) | WEB |
| field strength | WEB; per-pt ABSENT | AUTO (csv) | WEB |
| centre | AUTO (1, Dijon) | WEB (3) | AUTO (csv) |
| pathology | AUTO (Info.cfg) | AUTO (csv) | AUTO (csv) |
| age / sex | ABSENT | ABSENT | AUTO (csv) |
| height+weight (BSA) | AUTO (Info.cfg) | ABSENT | AUTO (csv) |
| ED/ES | AUTO (Info.cfg) | AUTO (files) | AUTO (csv) |
| label convention | AUTO (geom) | AUTO (geom) | AUTO (geom) |
| papillary / basal rule | WEB (shared ✓) | WEB | WEB |
| official split | AUTO (folders) | WEB | AUTO (folders) |
| intensity scale | ABSENT → z-score | ABSENT | ABSENT |
| inter-observer | ABSENT → floor | ABSENT | ABSENT |

**Finding (2026-06-20 research):** ACDC / M&Ms-1 / M&Ms-2 **share** the LV-cavity convention (papillary
+ trabeculae included; M&Ms-2 "follows ACDC standards") → the cross-dataset EF bias is *not* a
label-protocol mismatch; it's domain/intensity. See
`research/deep_dives/2026-06-20_dataset-acquisition-and-conventions.md` and
`research/deep_dives/2026-06-21_intensity-normalization-and-harmonization.md`.

## Reproducibility — data in data, source in repo, no artifacts
- Datasets + sidecars + extracted reference values live **out-of-repo** (`<data>/raw/<ds>/meta/`).
- **This package = source only** (parsers + transforms); it commits no data artifacts. The metadata
  view is *regenerated* by parsing shipped sidecars (deterministic); cached files are gitignored.
- **Provenance** per value: `{value, source, by: auto|paper, verified}` — the bulk is `by: auto`
  (parsed); the paper layer is `verified`-gated, unverified stays visibly unverified. Absence →
  per-scan fallback, never a silent default.

## Status / contents
- **AUTO parser** ✅ — `data/mri/{acdc,mnm2,mnms1}.py` `meta()` parses sidecars → `_source` per field.
- **Paper layer** ✅ — `sources.yaml`: cited WEB/paper constants, each `{value, source, verified}`.
- **persist** ✅ — `persist.py` merges AUTO + paper → `<data>/raw/<ds>/meta/<ds>.yaml`; `load_meta()` with
  per-scan fallback. `python -m cardioseg.preprocessing.normalization.persist`.
- **fetch_sources** ✅ — `fetch_sources.sh`: public challenge-page pulls + a manifest of gated sources.
- **N4** ✅ — `n4.py` (bias correction; params in `DataCfg.n4_params`, recorded in config.json).
- **Nyúl histogram standardization** ✅ built (`core/preprocessing/nyul.py`, `DataCfg.nyul`; standard
  fit to `reference/nyul.yaml`) — **measured NULL**: 0.857 vs 0.864 real-only cross-vendor Dice (−0.007,
  within noise; EF unchanged). z-score + heavy aug already capture the harmonizable variance; the
  residual cross-vendor gap is model-class (nnU-Net closes EF), not intensity-distribution. Off by
  default; kept as a tested, reusable tool. · **synth-aug T1/T2/T3** ⬜ (see tiers above).

Referenced from the main [README](../../../README.md#domain-shift--normalization).
