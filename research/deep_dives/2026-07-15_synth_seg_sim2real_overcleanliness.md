# Synthetic-Data Cleanliness as a Sim2Real Bottleneck in Medical Image Segmentation

**Date**: 2026-07-15  
**Status**: partial  
**Supersedes**: None

## TL;DR

Synthetic-data "over-cleanliness" (excessive class separability / intensity separation) is **recognized but not primary** in medical-imaging literature. The "reality gap" is a well-named failure mode in robotics and computer vision; medical imaging frames it as domain shift / distribution gap instead. Deliberate techniques to add realism via hardness-curriculum, learned appearance refinement (GANs/diffusion), and label-smoothing exist, but they target visual realism, not specifically separability. No validated metric predicts downstream segmentation Dice from synthetic realism alone.

## Question

Is "synthetic data is too clean → poor sim2real transfer" recognized as a named, quantified failure mode in medical-imaging and general computer-vision literature? What techniques deliberately degrade synthetic realism / increase training difficulty, and do they improve real-domain performance? What tools measure separability / difficulty as predictors of segmentation success?

## Findings

### 1. Named Failure Modes: "Reality Gap" vs. "Domain Gap"

- **Robotics & CV**: The **"reality gap"** is a recognized, named failure mode in robotics sim2real transfer and simulation-based training. It is defined as the discrepancy between the simulator's visual appearance and physics vs. the real world, covering texture, lighting, backgrounds, and (critically) **task difficulty**. [S1] The term is extensively used and is not considered a new observation (sim2real terminology is established since ~2010s). [S2]
- **Medical Imaging**: The equivalent is framed as **"domain gap"** or **"distribution shift"** between synthetic and real images. Papers do not typically use the word "cleanliness" but do discuss **"intensity overlap,"** **"boundary artifacts,"** and **"oversimplification."** [S3] The gap is attributed to acquisition device differences, protocols, and scanner vendor variation, less commonly to inherent synthetic-generation simplification.
- **Key distinction**: Computer vision acknowledges that synthetic (game-engine or simulator) images are *inherently* cleaner (sharper edges, fewer artifacts, less noise). Medical imaging focuses on statistical distribution mismatch rather than visual "cleanliness."

### 2. Quantified Performance Gap: Synthetic-Only Underperformance

- **Cardiac MRI segmentation**: Models trained solely on synthetic data underperform real-trained baselines. Dice gaps cited as **~0.56 (synth-only) vs. ~0.85 (real-trained)** in one echocardiography study, with synth-trained scoring **-5.19% Dice** on real test sets. [S4]
- **Mixed synthetic+real training**: Combining synthetic and real data during training yields **4% Dice improvement and 40% Hausdorff distance improvement** over real-only baselines on multi-vendor cardiac datasets. [S4] This suggests synthetic data is useful if paired with real, but synthetic-only struggles with domain shift.
- **Not fully explained by shape/texture**: Your context notes shape and texture coverage ruled out; the cited literature does not isolate **intensity separability** as the driver vs. other factors (e.g., noise distribution, edge blur, scanner artifact patterns).

### 3. SynthSeg & Domain Randomization: Limitations Not Explicitly Stated

- **SynthSeg (Billot et al., 2021)**: The landmark brain-MRI synthetic-training work reports "unparalleled generalization compared with supervised CNNs, state-of-the-art domain adaptation, and Bayesian segmentation," testing across 5,000 scans and six modalities without retraining. [S5] However, the paper **does not discuss limitations** of domain randomization or the possibility that synthetic images are "too easy."
- **Domain randomization strategy**: SynthSeg trains on synthetic images with **fully randomized contrast, resolution, and intensity** drawn from uninformative uniform priors, intentionally forcing domain-agnostic features. [S6] The approach does not attempt to match real-image difficulty; instead, it assumes diversity is sufficient.
- **Cardiac extension missing**: No parallel SynthSeg-equivalent for cardiac MRI found in this search; cardiac work relies on ad-hoc architectures + domain adaptation rather than pure domain-randomized synthetic training.

### 4. Curriculum Learning & Hardness: Techniques to Deliberately Add Difficulty

- **Diffusion Curriculum (DisCL, 2024)**: Directly addresses the synthetic-to-real difficulty mismatch. The method:
  - Identifies hard samples in original (real) images as guidance
  - Generates a **full spectrum of synthetic-to-real images** by varying diffusion guidance strength
  - Uses curriculum strategy to select training data that **maximizes expected progress**, avoiding easy synthetic samples that overfit the model [S7]
  - Reported results on iWildCam and ImageNet-LT (not medical), but the framing is explicit: **hardness-based curricula improve real-world transfer.**

- **Style Curriculum Learning (2021)**: Trains segmentation models using an **"easy-to-hard"** mode, generating curriculum samples via style transfer. The approach **gradually focuses on complex and adversarial style samples** to boost robustness. [S8] Applied to M&Ms cardiac challenge dataset, achieving "significant improvements in segmentation accuracy" on unknown distributions. No specific Dice numbers quoted in abstract.

- **Hard Mask Prioritization**: Research emphasizes that generating diverse "hard" synthetic samples (measured via **mask-level global hardness** quantification) prevents overfitting to easy samples and improves downstream generalization. [S7] However, hard samples customized to one downstream model limit broader applicability.

### 5. Learned Appearance Refinement: GAN & Diffusion-Based Methods

- **SimGAN-style refiner networks** (Learning from Simulated and Unsupervised Images, Shrivastava et al., 2016 lineage): A **refiner neural network** maps synthetic images to realistic images via:
  - **Adversarial loss**: Fools a discriminator into classifying refined images as real
  - **Self-regularization**: Minimizes pixel-level difference between synthetic and refined, preserving annotations
  - Applications: autonomous driving, object segmentation, pose estimation. [S9] **No medical-imaging results quoted in available abstracts.**

- **CycleGAN for medical images**: Performs unsupervised image-to-image translation between domains (synthetic→real appearance). Applied to X-ray angiography, CT-ultrasound translation, and polyp synthesis. [S10] Preserves segmentation labels during style transfer. **Specific performance gains not quantified in search results.**

- **Diffusion-based synthetic-to-real refinement** (FLUX + REGEN, 2026): Uses diffusion models to enhance geometry and materials (FLUX.2-4B Klein), then applies distribution-matching image-to-image translation (REGEN) to align synthetic images with real-dataset characteristics. Results on VKITTI2: **CMMD distribution distance reduced from 3.734 to 1.781**, matching or exceeding standard diffusion-alone approaches. [S11] **Not tested on medical images; focuses on game-engine synthetic data.**

- **Cardiac pathology diffusion synthesis**: Recent work (LeFusion, 2024) synthesizes myocardial pathology on cardiac MRI via lesion-focused diffusion models. Diffusion-generated images show **2.9–3.8 percentage point Dice improvements** over baseline segmentation augmentation (2025 study). [S12] However, this is augmentation *on real data*, not synthetic-to-real training.

### 6. Distribution-Distance Metrics: Do They Predict Segmentation Performance?

- **FID (Fréchet Inception Distance)**: The de facto standard for assessing distributional properties of synthetic images. Embeds real and synthetic images in Inception Net features, computes Wasserstein-2 distance assuming multivariate Gaussians. [S13] **However, FID does NOT reliably predict downstream segmentation Dice.** Multiple sources show:
  - Decreasing FID does not guarantee improved Dice [S13]
  - Low-FID synthetic datasets can actually *degrade* segmentation accuracy [S13]
  - In retinal-image synthesis, lower FID did not translate to improved segmentation when synthetic data used for augmentation [S13]
  - Log-normal or task-specific non-monotonic relationships exist between FID and Dice, with intermediate FID sometimes optimal [S13]

- **Ideal Observer / Signal Detection**: Theoretically grounded approach using **Hotelling observer** (uses all statistical information) or **ideal Bayesian observer** to measure task performance. [S14] A 2-AFC (two-alternative forced-choice) study at ~50% observer accuracy implies synthetic and real distributions are visually indistinguishable. [S14] **Not yet applied systematically in medical segmentation to predict Dice.**

- **Embedded Characteristic Score (ECS)**: Proposed 2025 approach for assessing distributional fidelity of synthetic chest X-rays. Uses characteristic function theory to evaluate whether synthetic and real distributions remain properly distinguished. [S15] Moves beyond pixel-level or standard feature metrics, but results and predictive power for segmentation not yet available.

- **Consensus finding**: No validated metric currently predicts downstream segmentation Dice from synthetic realism without training downstream models. [S13] Distribution-gap metrics are useful for generation quality but not diagnostic for transfer performance.

### 7. Domain Adaptation vs. Domain Randomization: When You Have Unlabeled Real Data

- **Transformation-Invariant Self-Training (TI-ST, 2023)**: Leverages unlabeled real images to adapt synthetic-trained models. Filters unreliable pseudo-labels at the pixel level by requiring "transformation-invariant detections" (predictions stable under image transforms). [S16] Addresses large distribution gaps between synthetic and real, particularly across scanner vendors.
- **Consensus**: Self-training / pseudo-labeling can reduce the gap when unlabeled real data is available. [S16] Your context implies zero unlabeled real data; domain adaptation is not applicable.
- **Key trade-off**: Domain randomization (maximize diversity at train time) vs. domain adaptation (exploit real data at inference/fine-tune time). Medical papers increasingly use **hybrid approaches**: train on synthetic + diverse augmentation, then self-train on unlabeled real if available. [S16]

### 8. Adjacent Insights: What Limits Synthetic-Only Training?

- **Intensity histogram mismatch**: Realistic synthetic images must match real intensity distributions, noise profiles, and bias-field patterns. Papers address this via:
  - Gaussian blur for smooth lesion boundaries (kernel 3×3, σ=1.0) [S17]
  - Realistic bias-field, Gibbs noise, intensity scaling augmentations [S17]
  - But none frame this as "cleanliness" prevention; rather, as *artifact realism* to match pathology.

- **Physics-based generation with randomization**: bSSFP cardiac MRI simulation exists (physics of off-resonance, T1/T2 relaxation, flip-angle variation) but literature focuses on *pulse-sequence parameters* (field strength, resolution, acceleration), not on controlling final image "difficulty." [S18]

- **Textbooks are silent**: No medical-imaging paper uses the term "synthetic data is too clean" or "over-separated classes." The equivalent phrasing is "domain shift," "distribution gap," or "low visual realism."

## Open Questions

1. **Is "over-separability" the bottleneck or a symptom?** Your d'=4.5 (synth) vs. d'=2.65 (real) measurement is striking. No paper in this search quantifies class separability as a *predictor* of downstream Dice without training. Could your measurement directly inform curriculum weighting?

2. **Why no diffusion-curriculum on cardiac MRI yet?** DisCL (hardness spectrum generation) was published 2024 but applied only to image classification, not medical segmentation. Cardiac MRI seems like a natural testbed—have you prototyped this?

3. **Can you measure "effective difficulty" of synthetic training on a held-out pool of real images without labels?** E.g., early-stopping performance of a pretrained encoder on a real-image clustering task might predict transferability. No paper found using this proxy.

4. **Will matched separability (deliberately reducing d' in synth) hurt domain-specific generalization?** If you blur boundaries or inject intensity overlap, does the model trade real-domain Dice for robustness to vendor variation? Unknown in current literature.

## Sources

- [S1] Object Detection Using Sim2Real Domain Randomization for Robotic Applications (arXiv:2208.04171, 2022)
- [S2] Bridging Sim2Real Gaps in Industrial Inspection with Domain Randomization (Medium, 2025) + Lil'Log "Domain Randomization for Sim2Real Transfer" (blog, 2019)
- [S3] Domain Generalization for Medical Image Analysis: A Review (arXiv:2310.08598, 2023)
- [S4] Pathology Synthesis of 3D-Consistent Cardiac MR Images using 2D VAEs and GANs (arXiv:2209.04223); Synthetic Boost: Leveraging Synthetic Data for Enhanced Vision-Language Segmentation in Echocardiography (arXiv:2309.12829, 2023)
- [S5] SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining (arXiv:2107.09559, Billot et al., 2021)
- [S6] Synthetic data in generalizable, learning-based neuroimaging (Imaging Neuroscience, MIT Press; PMC11752692, 2024)
- [S7] Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion (arXiv:2410.13674, 2024); FreeMask: Synthetic Images with Dense Annotations Make Stronger Segmentation Models (arXiv:2310.15160, 2023)
- [S8] Style Curriculum Learning for Robust Medical Image Segmentation (arXiv:2108.00402, Sheikh et al., 2021)
- [S9] Improving the Realism of Synthetic Images (Apple ML Research); Object Detection using Domain Randomization and Generative Adversarial Refinement of Synthetic Images (arXiv:1805.11778, 2018)
- [S10] S-CycleGAN: Semantic Segmentation Enhanced CT-Ultrasound Image-to-Image Translation for Robotic Ultrasonography (arXiv:2406.01191, 2024); CycleGAN for style transfer in X-ray angiography (Springer Nature, 2019)
- [S11] A Hybrid Approach for Closing the Sim2Real Appearance Gap in Game Engine Synthetic Datasets (arXiv:2605.02291, 2026)
- [S12] Multi-modal MRI synthesis with conditional latent diffusion models for data augmentation in tumor segmentation (ScienceDirect, 2025); Diffusion Model-Based Augmentation Using Asymmetric Attention Mechanisms for Cardiac MRI Images (Diagnostics 15(16), 2024)
- [S13] GANs for Medical Image Synthesis: An Empirical Study (arXiv:2105.05318); A Pragmatic Note on Evaluating Generative Models with Fréchet Inception Distance for Retinal Image Synthesis (arXiv:2502.17160, 2025); Benchmarking the Alignment of Data-Quality Metrics, Human Judgment and Land-Cover Segmentation Performance (arXiv:2606.25128, 2026)
- [S14] Gaussian Function Model for Task-Specific Evaluation in Medical Imaging (PMC12920941, 2024); Observer-study-based approaches to quantitatively evaluate the realism of synthetic medical images (PMC10411234, 2023)
- [S15] Assessing the Distributional Fidelity of Synthetic Chest X-rays using the Embedded Characteristic Score (arXiv:2501.00744, 2025)
- [S16] Domain Adaptation for Medical Image Segmentation using Transformation-Invariant Self-Training (arXiv:2307.16660, 2023); Image Translation-Based Unsupervised Cross-Modality Domain Adaptation for Medical Image Segmentation (arXiv:2502.15193, 2025)
- [S17] Medical image data augmentation: techniques, comparisons and interpretations (PMC10027281, 2023)
- [S18] Dynamic cardiac MRI with high spatiotemporal resolution using accelerated spiral-out and spiral-in/out bSSFP pulse sequences (Springer Nature, 2024)
