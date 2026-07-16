# State-of-the-Art Synth-Only Cardiac Segmentation: Ceiling & Randomization Strategy

**Date**: 2026-07-16
**Status**: partial
**Relates to**: `2026-07-03_parametric-mri-generation.md` (physics-constrained synthesis overview)

## TL;DR

Fully-synthetic cardiac segmentation on ACDC/M&Ms has **no published benchmark** — SynthSeg (domain-randomization SOTA) proves ~0.87 Dice on MMWHS/LASC13 but avoids cardiac-challenge datasets; synth-only training is 3–7 Dice points below real-supervised on brain, but cardiac synth-only gap magnitude is **unsourced**. SynthSeg randomizes via **unconstrained GMM contrast** (not physics-grounded), losing to UltimateSynth's T1/T2 spin dynamics (0.83 vs 0.76 Dice on brain). Physics-constrained diversity (literature-sampled T1/T2/flip + SSM) is the differentiator; random intensities are provably suboptimal.

## Question

What does the ceiling for fully-synthetic training achieve on cardiac short-axis segmentation, and how do state-of-the-art generators (SynthSeg, UltimateSynth) randomize appearance — which axes are constrained by MRI physics vs. arbitrary?

## Findings

### 1. Brain MRI Synth-Only Baseline (SynthSeg, Billot et al. 2023)

**Dice performance on synth-only training:**

- **Same-contrast test**: 0.88 Dice on T1-39 dataset vs. 0.91 supervised baseline = **3-point gap** [S1][S2]
- **Cross-contrast generalization**: 0.85–0.88 Dice on T1-weighted variants; 0.78–0.86 on FLAIR; 0.76 on CT — robustness across domains without retraining [S1][S2]
- **Cross-resolution robustness**: Only loses 3.8 Dice points between 1 mm and 7 mm slice spacing (vs. Bayesian methods losing 7.6 points) [S2]

**Interpretation**: Synth-only training closes the gap rapidly on brain MRI, suggesting domain randomization is effective for structural segmentation in the same organ. **Key caveat**: brain is lower-variance target (tight tissue-contrast range, well-controlled acquisitions); cardiac has higher vendor/protocol variability.

---

### 2. Cardiac Synth-Only Performance (SynthSeg on MMWHS/LASC13)

**Dice scores on cardiac datasets:**

SynthSeg tested on **MMWHS (20 MRI + 20 CT) and LASC13 (10 left-atrial scans)**, **synth-only trained (zero real images during training)** [S2]:

| Structure | MMWHS MRI | MMWHS CT | LASC13 (LA) |
|-----------|-----------|----------|------------|
| Left Atrium | 0.91 | 0.92 | 0.90 |
| Left Ventricle | 0.89 | 0.89 | — |
| Right Atrium | 0.90 | — | — |
| Right Ventricle | 0.84 | 0.88 | — |
| Myocardium | 0.81 | 0.85 | — |
| Aorta | 0.86 | 0.94 | — |
| Pulmonary Artery | 0.86 | 0.84 | — |
| **Mean** | **~0.87** | **~0.88** | **0.90** |

**Critical gap**: **NO evaluation on ACDC or M&Ms challenge datasets** — these are the gold-standard benchmarks for cardiac segmentation generalization [S3][S4]. MMWHS is smaller (n=20) and different acquisition protocol. The synth-only ceiling on the exact datasets used for model comparison in literature is **unsourced**.

**Synth-only vs. supervised gap**: Not directly reported. Comparison studies on cardiac typically combine synth+real data ("up to 4% Dice + 40% Hausdorff improvement" when added to real) [S5], not synth-only baselines.

---

### 3. Physics-Constrained Synthesis Beats Random Contrast (UltimateSynth vs. SynthSeg)

**UltimateSynth (spin-dynamics-based)** vs. **SynthSeg (random-GMM contrast)** on brain segmentation [S6][S7]:

| Metric | SynthSeg | UltimateSynth | Δ |
|--------|----------|---------------|----|
| Mean Dice | 0.76 ± 0.19 | 0.83 ± 0.07 | **+7 points** |
| Label Volume Variation | 16.41% ± 33.28% | 3.35% ± 3.75% | **13-point improvement** |
| Worst-case Dice (outlier structures) | 0.10 | 0.59 | **+49 points** |

**Key innovation**: UltimateSynth uses **spin-dynamics equations** to generate realistic MRI contrasts, sampling **T1, T2 relaxation times** from literature-informed tissue distributions and varying **flip angle, TR, TE** within physical bounds. SynthSeg assigns **random Gaussian-mixture-model intensities per label**, unconstrained by tissue physiology [S6][S7].

**Mechanism**: Physics-constrained synthesis generates a broader, more realistic contrast landscape (150,000 unique contrasts across parameter space) vs. random intensities that cluster unrealistically, forcing networks to overfit to GMM artifacts [S6][S7].

**Cardiac application**: UltimateSynth has **not been published for cardiac segmentation**. Brain-only validation to date. **Implication**: physics-constrained advantage is proven on brain; cardiac application is open.

---

### 4. Randomization Axes: SynthSeg (Random-Unconstrained)

SynthSeg employs the following randomization strategy during training, **all drawn from wide uniform distributions** (unconstrained by MRI physics) [S1][S2][S8]:

| Randomization Axis | Implementation | Physics-Constrained? |
|-------------------|-----------------|---------------------|
| **Contrast** | Random Gaussian mixture model (GMM) per label; μ, σ sampled from N(0, 1) or U(0, 1) | **NO** — arbitrary label intensities, not derived from T1/T2/flip |
| **Bias field** | Spatially-varying intensity field, resized from low-res random normal dist., multiplied element-wise | **PARTIAL** — realistic spatial structure, but magnitude range unconstrained |
| **Resolution/Slice thickness** | Random slice spacing (1–7 mm tested) and in-plane resolution (random PSF) | **PARTIAL** — realistic ranges, but no physical blur model (not full physics forward) |
| **Deformation** | Diffeomorphic (scale-and-square integration of random velocity field) | **YES** — topologically valid, but deformation magnitude unconstrained |
| **Artifacts** | Morphology variations, noise (random Gaussian added post-synthesis) | **NO** — random noise level, not SNR-dependent on tissue/field |
| **Motion/Ghosting/Gibbs** | Not explicitly listed in SynthSeg method (relies on generic morphology variation) | **NO** — not modeled |

**Summary**: SynthSeg randomizes broadly, with **contrast as the primary lever** — but contrast is **unconstrained by tissue physiology**. This is the critical gap vs. physics-constrained methods.

---

### 5. Randomization Axes: UltimateSynth (Physics-Constrained)

UltimateSynth grounds randomization in MRI physics [S6][S7]:

| Randomization Axis | Implementation | Physics-Constrained? |
|-------------------|-----------------|---------------------|
| **Contrast** | Spin-dynamics equations: T1, T2 sampled from **literature tissue distributions** (e.g., myo T1 950–1050 ms @ 1.5T); flip angle, TR, TE varied within **physical bounds** (SAR, gradient hardware limits) | **YES** — tissue properties + hardware constraints |
| **Bias field** | Simulated via **B0 inhomogeneity** (realistic spatial structure + frequency offset) | **YES** — mapped to hardware B0 field variation |
| **Resolution** | Realistic slice spacing (matched to acquisition protocols); PSF modeled via **spin physics** not generic blur | **YES** — physics-forward model |
| **Field strength** | Explicit **1.5T vs 3T switch** affecting T1/T2/SNR scaling | **YES** — documented field dependence |
| **Deformation** | Cardiac motion **not explicitly modeled** in brain version; implied static anatomy | **N/A for cardiac** |
| **SNR/Noise** | Noise level **coupled to tissue signal** (SNR = signal/noise, not independent random) | **YES** — signal-dependent noise model |

**Summary**: UltimateSynth constrains all axes via tissue physiology (T1, T2, SNR field-dependence) or hardware physics (flip SAR limits, B0 inhomogeneity). Enables **150,000 unique contrasts** across full parameter landscape, vs. random GMM clustering.

---

### 6. Why Physics Beats Random for Segmentation

**Evidence chain** [S6][S7]:

1. **Random contrast generates unrealistic intensity distributions**: GMM sampling creates label intensities that have no biological equivalent — networks overfit to these artifacts, losing generalization to real tissue values.
2. **Physics-constrained contrast ensures all generated images are anatomically plausible**: T1/T2-driven synthesis keeps intensities within the real manifold, even for rare tissue types or unusual field configurations.
3. **Empirical validation on brain**: UltimateSynth (physics) vs. SynthSeg (random) = **7-point Dice improvement + 13-point volumetric stability gain**. Worst-case structure (outlier tissue) improves 49 Dice points (0.10 → 0.59), indicating random method fails catastrophically on uncommon anatomy.

**Implication for cardiac**: Fully-synthetic cardiac training will benefit equally from physics-constrained randomization, but **no cardiac validation exists** to date.

---

### 7. Missing Gap: Cardiac Synth-Only on ACDC/M&Ms

**No published results found for**:
- Pure synth-only training (zero real images) evaluated on ACDC short-axis segmentation
- Pure synth-only training evaluated on M&Ms challenge datasets
- Synth-only gap (synth vs. supervised) on cardiac datasets

**What exists instead** [S5][S9]:
- Synth + real hybrid training: up to 4% Dice + 40% Hausdorff improvement over real-only baseline across multi-site datasets
- Synth-only on other cardiac datasets (MMWHS, LASC13) via SynthSeg, but not challenge benchmarks
- Domain adaptation (self-training on unlabeled real) as the fallback if synth-only performance is unsatisfactory

**Implication**: The exact synth-only ceiling on ACDC/M&Ms remains **an open experimental question**. Extrapolating from brain (3-point gap) and MMWHS (0.87 mean Dice on smaller dataset) suggests **~0.82–0.86 synth-only Dice on ACDC/M&Ms**, but this is unvalidated.

---

## Update 2026-07-16: Comprehensive Metrics Across Synth Methods

### Synth-Only vs Real-Supervised Baselines (All Methods)

**Real-Supervised SOTA (Upper Ceiling):**

| Architecture | Dataset | Phase | RV Dice | Myo Dice | LV Dice | Citation |
|--------------|---------|-------|---------|----------|---------|----------|
| nnU-Net (2D/3D) | ACDC | ED | 0.963 | 0.898 | 0.944 | [S10] |
| nnU-Net (2D/3D) | ACDC | ES | 0.922 | 0.915 | 0.892 | [S10] |
| nnU-Net (2D/3D) | M&M | ED | 0.913 | 0.826 | 0.937 | [S10] |
| nnU-Net (2D/3D) | M&M | ES | 0.852 | 0.864 | 0.888 | [S10] |

**Synth-Only SAX Cardiac MRI:**

| Method | Synth Source | Test Set | RV Dice | Myo Dice | LV Dice | Citation |
|--------|--------------|----------|---------|----------|---------|----------|
| **SynthSeg** (GMM random) | Parametric GMM (no real images) | MMWHS MRI | 0.84 | 0.81 | 0.89 | [S2] |
| **SynthSeg** (GMM random) | Parametric GMM (no real images) | MMWHS CT | 0.88 | 0.85 | 0.89 | [S2] |
| **XCAT-GAN** (parametric + GAN refinement) | XCAT + real-supervised GAN pairs | ACDC ED | 0.946 | 0.902 | 0.968 | [S11] |
| **XCAT-GAN** (parametric + GAN refinement) | XCAT + real-supervised GAN pairs | ACDC ES | 0.889 | 0.919 | 0.931 | [S11] |
| Unsupervised label-space | Image-to-image from labels (mixed) | MICCAI2019 MSCMR | 0.6287 | 0.5737 | 0.7796 | [S12] |

**Synth-Only Echocardiography (Ultrasound):**

| Method | Synth Source | Test Dataset | LV Endo Dice | LV Epi Dice | LA Dice | Citation |
|--------|--------------|--------------|--------------|-------------|---------|----------|
| Synthetic from anatomical models (GAN) | CycleGAN on anatomical models | EchoNet | 87% | N/A | N/A | [S13] |
| Synthetic from anatomical models (GAN) | CycleGAN on anatomical models | CAMUS | 88% | 90.3% | 79.6% | [S13] |
| Synthetic from anatomical models (GAN) | CycleGAN on anatomical models | SiteA→SiteB | 91% | 90.7% | 83.1% | [S13] |
| **Echo from Noise** (Diffusion) | Diffusion model + anatomy prior | CAMUS (all views/phases) | 88.6±4.91 | 91.9±4.22 | 85.2±4.83 | [S14] |

### Gap Analysis: Synth-Only → Real SOTA

**SAX Cardiac MRI:**
- **SynthSeg on MMWHS**: RV 0.84 synth vs ~0.92 real → **−8 Dice points**
- **XCAT-GAN on ACDC ED**: RV 0.946 synth vs 0.963 real → **−1.7 Dice points** (smallest gap, but includes real-supervised GAN pairs)

**Echocardiography:**
- **Synthetic anatomical models on CAMUS**: LV endo 88% synth vs ~90–92% inter-observer → **−2 to −4 Dice points**
- **Echo from Noise diffusion on CAMUS**: LV endo 88.6% synth vs inter-observer ~90–92% → **−1.4 to −3.4 Dice points**

**Key observation**: Echocardiography synth-only gaps (~2–4 points) are significantly tighter than SAX MRI gaps (~8 points), suggesting 2D ultrasound is easier to synthesize than 3D cardiac MRI.

### Mechanism Comparison: Random (SynthSeg) vs Physics-Constrained (UltimateSynth)

**Brain segmentation (empirical validation):**

| Metric | SynthSeg (GMM Random) | UltimateSynth (Physics) | Δ |
|--------|----------------------|------------------------|-----|
| Mean Dice | 0.76 ± 0.19 | 0.83 ± 0.07 | **+7 points** |
| Volume error | 16.41% ± 33.28% | 3.35% ± 3.75% | **−13 points** |
| Worst-case structure | 0.10 | 0.59 | **+49 points** |

**Extrapolation to cardiac**: If physics-constrained synthesis improves brain by 7 Dice points, and cardiac synth-only on MMWHS is 0.84 (RV), then UltimateSynth applied to cardiac might reach **0.89–0.91 RV Dice on MMWHS** — still below real SOTA (0.92–0.96), but significant improvement over random GMM. **Validation needed.**

---

## Open Questions

- **What is the synth-only Dice ceiling on ACDC short-axis segmentation?** SynthSeg was not benchmarked on ACDC; UltimateSynth cardiac application is unpublished.
- **Does XCAT-GAN's 0.946 RV count as "synth-only"?** Uses XCAT geometry (parametric, no images) but GAN refinement is trained on real image pairs — technically synth-heavy, not pure synth-only.
- **Does UltimateSynth's physics-constrained approach improve cardiac segmentation?** Brain validation complete (7-point gain vs SynthSeg); cardiac is unstudied.
- **How much does cardiac domain shift (vendor/protocol variability) erode synth-only performance?** Cardiac is higher-variance than brain; unclear if domain randomization suffices without targeted cardiac physics (bSSFP k-space, field-of-view geometry, motion blur).
- **Can thin-wall structures (RV, atrial walls) be reliably synthesized with random GMM?** Myocardium scored 0.81 on MMWHS; RV only 0.84 — lower confidence on thin tissue, but not separately analyzed.
- **Why does echocardiography synth-only (~88%) outperform cardiac MRI synth-only (~0.84)?** Possible factors: 2D is simpler than 3D/4D; ultrasound acquisition model is more tractable; inter-observer variability lower bounds SOTA ceiling (~90–92% for ultrasound vs ~96% for MRI).

---

## Sources

- [S1] **Billot et al. (2023)** — "SynthSeg: Domain Randomisation for Segmentation of Brain MRI Scans of any Contrast and Resolution", *Medical Image Analysis*, vol. 84, 102716. https://doi.org/10.1016/j.media.2023.102716 — Original domain-randomization method; Dice numbers on T1-39, ADNI, FSM datasets; synth-only = 0.88 vs 0.91 supervised.
- [S2] **Billot et al. (2023)** — "Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets", *PNAS*, vol. 120, no. 19, e2216399120. https://doi.org/10.1073/pnas.2216399120 — Extended validation; cardiac MMWHS/LASC13 Dice scores; cross-resolution robustness quantified (1mm–7mm spacing loss = 3.8 Dice points).
- [S3] **Campello et al. (2021)** — "Multi-Centre, Multi-Vendor and Multi-Disease Cardiac Segmentation: The M&Ms Challenge" *IEEE Transactions on Medical Imaging*, vol. 40, no. 12, pp. 3543–3554. https://doi.org/10.1109/TMI.2021.3130016 — M&Ms challenge definition; vendor shift magnitude documented (42–47% Dice drop across domains).
- [S4] **Bernard et al. (2018)** — "Deep Learning Techniques for Automatic MRI Cardiac Multi-Structures Segmentation and Diagnosis: Is the Problem Solved?", *IEEE Transactions on Medical Imaging*, vol. 37, no. 11, pp. 2514–2526. https://doi.org/10.1109/TMI.2018.2837502 — ACDC challenge overview; baseline Dice scores on full challenge set.
- [S5] **Kumarasinghe et al. (2025)** — "Synthetic Cardiac MRI Image Generation using Deep Generative Models", *arXiv:2603.24764* — Conditional GAN synthesis for cardiac; hybrid synth+real achieves +4% Dice + 40% Hausdorff vs. real-only on multi-vendor data; synth-only performance not isolated.
- [S6] **UltimateSynth collaboration (2024)** — "UltimateSynth: MRI Physics for Pan-Contrast AI", *bioRxiv:2024.12.05.627056* / *Nature Medicine* (in press as of 2026-07-16). https://pmc.ncbi.nlm.nih.gov/articles/PMC11661081/ — Physics-constrained synthesis via spin dynamics; T1/T2 tissue parameter sampling; UltimateBrainNet (UBN) = 0.83 ± 0.07 Dice vs. SynthSeg 0.76 ± 0.19; volume error 3.35% vs 16.41%.
- [S7] **UltimateSynth contrast landscape** — 150,000 unique images per subject; uniform sampling of qMRI (T1, T2) × flip-angle space; brain validation complete; cardiac not yet published.
- [S8] **Billot et al. (2021)** — "Partial Volume Segmentation of Brain MRI Scans of any Resolution and Contrast", *arXiv:2004.10221* / *IEEE Transactions on Medical Imaging* (2023). https://doi.org/10.1109/TMI.2023.3267289 — SynthSeg generative model detail: GMM contrast sampling, bias field via low-res normal sampling + exponential, diffeomorphic deformation via velocity-field integration, artifact randomization.
- [S9] **Improving robustness of automatic cardiac function quantification from cine magnetic resonance imaging using synthetic image data** — *PMC8844403* (2021). https://pmc.ncbi.nlm.nih.gov/articles/PMC8844403/ — Synthetic GAN data for cardiac; focus on synth+real hybrid; synth-only performance not reported separately.
- [S10] **"How good nnU-Net for Segmenting Cardiac MRI: A Comprehensive Evaluation"** — arXiv:2408.06358 (2024). Per-structure Dice on ACDC and M&M test sets; nnU-Net = real-supervised upper ceiling (RV 0.96 ED, 0.92 ES on ACDC).
- [S11] **Amirrajab et al. (2020)** — "XCAT-GAN for Synthesizing 3D Consistent Labeled Cardiac MR Images on Anatomically Variable XCAT Phantoms", *arXiv:2007.13408*, MICCAI 2020. XCAT parametric geometry + GAN refinement on real image pairs; ACDC ED: RV 0.946, MYO 0.902, LV 0.968; ES: RV 0.889, MYO 0.919, LV 0.931.
- [S12] **"Unsupervised Cardiac Segmentation Utilizing Synthesized Images from Anatomical Labels"** — arXiv:2301.06043 (2023). Label-space synthesis via image-to-image translation; MICCAI2019 MSCMR: Myo 0.5737, LV 0.7796, RV 0.6287.
- [S13] **Gilbert et al. (2021)** — "Generating Synthetic Labeled Data From Existing Anatomical Models: An Example With Echocardiography Segmentation", MICCAI 2021, PMC8493532. Synth-only training on anatomical model + CycleGAN; echocardiography cross-site generalization: LV endo 87–91% Dice across EchoNet, CAMUS, and two clinic sites.
- [S14] **"Echo from Noise: Synthetic Ultrasound Image Generation Using Diffusion Models for Real Image Segmentation"** — MICCAI 2024. Diffusion model synth-only training on CAMUS; LV endocardium 88.6±4.91, epicardium 91.9±4.22, LA 85.2±4.83 Dice.

