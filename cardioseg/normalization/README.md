# cardioseg/normalization — inter-scanner variance, handled at the right layer

MRI intensity is **uncalibrated** — unlike CT's Hounsfield units, a cine-MRI pixel value has no
absolute meaning (it depends on coil gain, recon auto-scaling, sequence weighting). So the same
tissue reads differently across scanners, and **inter-scanner variance is the core obstacle** to a
model trained on one set generalizing to another. This package organizes how we attack it.

## The principle: knowable → correct; unknowable → normalize

You don't pick "physical vs statistical" by taste. **If the variance is knowable** (a number we can
parse, or a field we can estimate from the image) → remove it at its source. **If it's unknowable**
(no parameter, no anchor) → normalize it statistically (or learn invariance via augmentation). Same
question, two answers.

Three knowability tiers map 1:1 to the right tool:

| tier | source | tool | reproducible? |
|---|---|---|---|
| header | NIfTI affine | read | ✓ deterministic |
| sidecar | Info.cfg / CSV / folders | **parse** | ✓ deterministic ← *the bulk* |
| paper | challenge paper | LLM/manual + `verified` flag | sourced, not regenerated |
| image-derived | the pixels | N4 / blood-pool ref / vol-curve | ✓ deterministic |
| (none) | — | statistical / augmentation / accept | n/a |

## The 5 buckets × 2 axes

| bucket | knowable → correct/parse | unknowable → normalize |
|---|---|---|
| **machine** (between scanners) | spacing (header), vendor/scanner/field/centre (sidecar+paper) | recon scale → z-score / Nyúl |
| **scan** (between scans) | bias field → N4 (image) | receive gain → z-score |
| **patient** (between people) | age/sex/BSA (sidecar), heart locate+orient (image) | body habitus → augmentation |
| **temporal** (within cine) | ED/ES frames (sidecar / vol-curve) | residual motion |
| **annotation** (between labelers) | label convention (geom verify), papillary/basal rule (paper) | inter-observer noise → **irreducible LoA floor** |

The bottom-right is the honest endpoint: even two human experts disagree on EF, so our limits of
agreement **cannot beat the inter-observer floor**. We aim to reach it, not pretend past it.

## Per-dataset coverage — what each ships vs what we fetch

Legend: **AUTO** parse shipped file · **WEB** fetch from paper (cited) · **LLM** present but needs
restructuring · **ABSENT** → image-derived or unknowable fallback.

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
| official split | AUTO (folders) | LLM (readme→struct) | AUTO (folders) |
| intensity scale | ABSENT → z-score | ABSENT | ABSENT |
| inter-observer | ABSENT → floor | ABSENT | ABSENT |

**Finding (2026-06-20 research):** ACDC / M&Ms-1 / M&Ms-2 **share** the LV-cavity convention
(papillary + trabeculae included; M&Ms-2 "follows ACDC standards") → the cross-dataset EF bias is
*not* a label-protocol mismatch; it's domain/intensity. See
`research/deep_dives/2026-06-20_dataset-acquisition-and-conventions.md`.

## Reproducibility — data in data, source in repo, no artifacts

- **Datasets + their metadata sidecars + extracted reference values** live **out-of-repo**
  (`D:/data/volumetric/mri/`), per-dataset (`raw/<ds>/meta/`) + cross-dataset (`common/normalization/`,
  may be empty = "we don't have it" → graceful fallback).
- **This package = source only** (parsers + transforms). It **commits no data artifacts**: the
  metadata view is *regenerated* by parsing shipped sidecars (deterministic); anything cached in the
  repo tree is gitignored. The repo describes *how* (parser + cited sources), the data stays the data.
- **Provenance** per value: `{value, source, by: auto|llm|human, verified}`. The bulk is `by: auto`
  (parsed). A small paper-extracted layer is `by: llm/human` with `verified` gating — unverified is
  visibly unverified. Absence → per-scan fallback, never a silent default.
- **Splits**: honor each dataset's *official* split where possible (comparability with published
  numbers); deviations documented.

## Status / contents
- **AUTO parser** ✅ — `data/mri/{acdc,mnm2,mnms1}.py` `meta()` parses the shipped sidecars
  (Info.cfg / CSVs) → acquisition + demographics with per-field `_source`.
- **Paper layer** ✅ — `sources.yaml`: the cited WEB/paper-tier constants (ACDC vendor/field, …),
  each `{value, source, verified}`; unverified fields stay visibly unverified.
- **persist** ✅ — `persist.py` merges AUTO + paper → `<data>/raw/<ds>/meta/<ds>.yaml`
  (`{value, source, by, verified}` per field; regenerable, out-of-repo) + `load_meta()` with
  per-scan fallback. `python -m cardioseg.normalization.persist`.
- **fetch_sources** ✅ — `fetch_sources.sh`: public challenge-page pulls + a manifest of the
  paywalled/register-gated sources to fetch manually.
- **N4** ✅ — `n4.py` (bias correction). **Nyúl standardization** ⬜ — spec.

Referenced from the main [README](../../README.md#data-normalization).
