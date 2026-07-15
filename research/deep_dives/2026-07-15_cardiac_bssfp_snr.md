# Cardiac Cine bSSFP SNR at 1.5T and 3T

**Date**: 2026-07-15
**Status**: settled
**Supersedes**: (none)

## TL;DR

Myocardial SNR at 3T exceeds 1.5T by 48% for bSSFP (blood pool +30%), driven primarily by intrinsic field gain (~1.66-1.80×) but offset by acquisition trade-offs (banding artifacts, SAR constraints). Clinical bSSFP SNR is adequate for cine imaging at both field strengths (typical blood pool ~17-29 range), though 3T requires tighter field-shimming and shorter TR to manage homogeneity sensitivity.

## Question

What are reported signal-to-noise ratios (SNR) and blood-pool SNR values for cardiac cine bSSFP at 1.5T versus 3T? How does SNR scale between field strengths, and what methodological factors affect the measurements?

## Findings

### Myocardial and Blood SNR at 1.5T vs 3T

- **SSFP myocardial SNR at 3T increased 48% relative to 1.5T** [S1]; **blood pool SNR increased 30%** [S1]. This is measured on equivalent flip-angle-optimized sequences (FLASH for comparison showed smaller gains: 19% myocardium, 13% blood) [S1].
- **Contrast-to-noise ratio (CNR) increased 22%** from 1.5T to 3T for SSFP, more favorable than FLASH (which showed smaller CNR gains) [S1].
- **Intrinsic SNR (ISNR, field-only, independent of tissue relaxation)** scales as B₀²: measured ratios are **1.66–1.80× at 3T vs 1.5T**, **2.36× at 4T** [S2]. This sets the physical ceiling; actual clinical SNR gains are lower due to acquisition adjustments.

### Clinical Blood-Pool SNR Range

- **Blood pool SNR in pediatric cardiac bSSFP: ~28.8 ± 3.7** [S4]; in adults, reported range **~17.0–29** depending on flip angle and field strength [S4].
- **Post-contrast myocardial SNR: 145 ± 107** (abbreviated protocol at unspecified field, likely 3T); pre-contrast standard bSSFP **95 ± 45** [S5]. This demonstrates that post-contrast protocols can substantially elevate signal.
- **Myocardium at 0.35T peaked at SNR 11.0** (at 70° flip); extrapolating upward, 1.5T at optimal flip yields ~3–3.5× higher SNR; **0.35T blood pool peaked at SNR 40.5** (at 130° flip) [S6].

### SNR Scaling and Coil Dependence

- **Whole-body coil receiver (coronary SSFP):** 87% SNR gain from 1.5T to 3T [S3]; **phased-array cardiac coil: 53% SNR gain** [S3]. Coil type substantially modifies realized SNR, not just tissue.
- **Theoretical SNR gain from 1.5T to 3T is 2× (100%)**; actual gains (30–48%) are **sublinear due to acquisition trade-offs** (shorter TR to reduce banding artifact, lower flip angles to manage SAR at 3T) [S2, S7].

### SNR Methodology and Measurement Caveats

- **SNR measurement methods:** Two validated approaches are (1) multiple-acquisition method (repeated scans) and (2) difference method; excellent agreement between them [S8]. Parallel imaging SNR degrades with g-factor (noise amplification during reconstruction) [S8].
- **SNR highly sensitive to:** acquisition parameters (flip angle, TR, TE, bandwidth), field homogeneity, coil positioning, regional placement of ROI [S7]. Studies report CNR variance 9.4–86% across protocols, reflecting parameter sensitivity.
- **SSFP SNR depends on:** tissue T₁/T₂ ratio, voxel volume, and cumulative sampling time (efficiency), **NOT directly on TR or bandwidth if scan time is held constant** [S9]. Optimal flip angles differ by field (e.g., 56° at 1.5T vs 70° at 0.35T for max contrast) [S6].

### Field Inhomogeneity Trade-off

- **bSSFP is sensitive to B₀ inhomogeneity**, which worsens with higher field strength. At 3T, shorter TR and increased shimming are required to suppress banding artifacts [S7]. This mandates reduced spatial resolution or higher receiver bandwidth, offsetting potential SNR gains [S7].
- **Image quality at 3T more variable than 1.5T**, with increased susceptibility artifacts and local brightening despite SNR gains [S3].

## Open Questions

- **Specific SNR values at 3T for clinical retrospectively-gated cine bSSFP** (retrospective gating and temporal averaging effects on observed SNR not deeply characterized across field strengths in the literature sampled).
- **Whether parallel imaging (GRAPPA, SENSE) SNR scaling differs between 1.5T and 3T** for cardiac cine — only general parallel-imaging methodology found, not field-specific cardiac data.
- **SNR in real clinical pipelines (with motion correction, temporal filtering, post-processing)** — most measurements are on raw reconstructed data.

## Sources

- [S1] "Cardiac cine MR-imaging at 3T: FLASH vs SSFP" — PubMed 16891230 (2006) — **primary source for 48% myocardium / 30% blood / 22% CNR gains.**
- [S2] "The Intrinsic Signal-to-Noise Ratio in Human Cardiac Imaging at 1.5, 3, and 4 T" — PMC2896425 (in vivo ISNR mapping; B₀-dependent scaling).
- [S3] "Three-dimensional breathhold SSFP coronary MRA: a comparison between 1.5T and 3.0T" — PubMed 16028242 (whole-body coil 87%, phased-array 53% SNR gains; image quality variability at 3T).
- [S4] "Ventricular function assessment using an ultrafast spoiled gradient echo sequence with an intravascular blood pool contrast agent in pediatric patients" — PLOS One (blood pool SNR ~28.8 ± 3.7 range).
- [S5] "Abbreviated Cardiac MRI Protocol with Post-contrast Cine SSFP: Reproducibility, Contrast-to-noise Ratio, and Energy Savings" — Journal of Cardiovascular Magnetic Resonance (post-contrast SNR 145±107, pre-contrast 95±45).
- [S6] "Cardiac balanced steady-state free precession MRI at 0.35 T: a comparison study with 1.5 T" — QIMS (flip-angle optimization; myocardium SNR 11.0 at 0.35T, ~3–3.5× higher at 1.5T).
- [S7] "Cardiovascular magnetic resonance at 3.0T: Current state of the art" — PMC2964699 (field inhomogeneity trade-off, CNR variance, parameter sensitivity).
- [S8] "Practical approaches to the evaluation of signal-to-noise ratio performance with parallel imaging: application with cardiac imaging and a 32-channel cardiac coil" — PubMed 16088885 (Reeder/Kellman; SNR measurement methodology, g-factor, parallel-imaging degradation).
- [S9] "Signal-to-Noise Ratio Behavior of Steady-State Free Precession" — PMC2396310 (SSFP SNR mathematical form: ∝ T₂/T₁ · ΔV · T_s', independent of TR/BW if efficiency constant; optimal flip angle examples).
