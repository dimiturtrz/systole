# Cardiac MRI Segmentation Landscape: OSS, Benchmarks, Domain Gaps & Browser Inference

**Date**: 2026-06-22
**Status**: partial
**Supersedes**: none

## TL;DR

nnU-Net dominates production cardiac MRI segmentation with ~91.6% Dice on ACDC; MONAI and TotalSegmentator (CT-focused) are established OSS frameworks. Cross-vendor generalization remains unsolved—vendor-specific Siemens→Philips drops ~42% (0.89→0.47 Dice). Browser-based inference (OHIF v3.10 + Cornerstone3D 2.0, ONNX Runtime) just landed in 2024–2025; MRI physics simulators (JEMRIS, MRiLab, KomaMRI) support cardiac motion but lack high-fidelity synthetic data pipelines.

## Question

What is the state of open-source cardiac MRI segmentation tooling, benchmark performance, cross-vendor robustness, browser-side inference capabilities, and MRI physics simulation for cardiac applications?

## Findings

### 1. Dominant OSS Frameworks

- **nnU-Net** [S1] is the gold-standard baseline for cardiac MRI segmentation. It achieved **91.61% average Dice on ACDC** [S2], with 2D variants outperforming 3D due to MRI's high in-plane, low through-plane resolution. nnU-Net automatically adapts preprocessing and network configuration to dataset characteristics [S3]. It is the backbone of TotalSegmentator and the de-facto choice for M&Ms winners.
- **MONAI** (Medical Open Network for AI) [S4] is an open-source PyTorch-based framework offering modular deep-learning primitives for medical imaging. Recent work (Sept 2025) compares MONAI models (DenseUNet, 3D UNet, UNETR, SegResNet, Attention UNet) on left-ventricle segmentation tasks [S5]. MONAI lacks cardiac-specific pretrained models but provides a standardized training harness.
- **TotalSegmentator** [S6] is nnU-Net-based and segments 117 anatomical structures in CT/MR. It includes two cardiac models: TSTOTAL (1.5mm voxel spacing, all 117 structures) and TSHC (submillimeter resolution, cardiac anatomy focused) [S7]. However, **TotalSegmentator is primarily CT-trained**; cardiac MRI support is limited.
- **Cardiac-specific repos** (e.g., `github.com/MarioProjects/MnMsCardiac`) publish M&Ms challenge solutions but are typically single-project implementations, not maintained frameworks [S8].

GitHub star counts not disclosed in searches; visit repos directly.

### 2. M&M and M&Ms-2 Benchmarks & Leaderboard Results

- **M&Ms Challenge (2020/2021)** [S9]: 375 patients, four vendors (Siemens, Philips, GE, Canon), split 150 train (two vendors) / 25 unannotated (third vendor). nnU-Net achieved best results overall. Participants used intensity-based augmentation and domain adaptation (CycleGAN, feature alignment) to bridge vendor gaps [S10].
- **M&Ms-2 Challenge (2023/2024)** [S11]: 360 cases, three vendors, nine scanners, short-axis + long-axis 4-chamber views, 160 annotated training + 40 validation. nnU-Net again achieved best overall performance [S12]. Focus shifted to Right Ventricle (RV) segmentation and multi-view heterogeneity.
- **ACDC Benchmark (Automated Cardiac Diagnosis Challenge)** [S13]: 300 images (200 train, 100 test), established as the canonical cardiac MRI segmentation benchmark. Top-performing methods achieved **92.25% Dice** (recent GCASCADE variant), outperforming TransUNet (89.71%) and Attention U-Net. Recent CardioSAM (topology-aware decoder) achieved **93.39% Dice** [S2], suggesting nnU-Net is no longer state-of-the-art on ACDC, though it remains production standard.
- **CMRxRecon2024 Challenge** [S14]: Focused on **reconstruction** (not segmentation) from accelerated k-space data. Top-5 methods significantly outperformed GRAPPA and SENSE baselines on SSIM, PSNR, NMSE metrics. Multi-vendor, multi-view k-space dataset spanning 1.5T and 3T.

### 3. Cross-Vendor Domain Generalization: Active Research, Not Solved

- **Generalization gap is substantial**: When a model trained on Siemens data is tested on Philips without fine-tuning, Dice drops from ~0.89 to ~0.47—a **47% relative drop** [S15]. Similar gaps reported across GE and Canon data [S16].
- **Approaches attempted** [S17]:
  - CycleGAN-based image-level domain adaptation + feature-level MSE alignment [S18]
  - Multi-sequence training (train on multiple MRI protocols) improves inductive bias for unseen domains [S19]
  - Domain generalization (learning domain-agnostic features from multiple source vendors) via loss reweighting and disentangled representations [S20]
- **M&Ms still the benchmark** for cross-vendor robustness. Post-challenge work shows one model achieved **74.8% average Dice on M&Ms without fine-tuning**, with stable RV performance but small myocardium drop [S21]. This is well below in-domain performance, indicating the problem remains.
- **Not a closed problem**: Recent papers (2024–2025) continue proposing domain adaptation and generalization methods; no published universal solution exists.

### 4. Browser-Based Inference: Recently Landed

- **OHIF v3.10 (April 2025)** [S22] introduced **Local AI Enhanced Segmentation** — AI models run entirely in-browser, no server needed. This is a major shift from v3.9 (Nov 2024, Cornerstone3D 2.0 only).
- **Cornerstone3D 2.0** [S23] supports multiple input volumes per GPU texture (background + segmentation overlay independently) and integrates ONNX Runtime for local inference.
- **ONNX + Cornerstone workflow** [S24]: Export trained segmentation model to ONNX → deploy with ONNX Runtime in OHIF plugin → wrap in FastAPI if needed for reading-room integration. Example: CT segmentation exported to ONNX, wrapped in FastAPI, called from OHIF to generate DICOM SEG masks.
- **OHIF-AI fork** [S25] extends OHIF with SAM2, nnInteractive, MedSAM2, VoxTell (VLM) for interactive segmentation via visual prompts (points, scribbles, bounding boxes) or text.
- **Cardiac-specific in-browser tools**: No dedicated browser-based cardiac segmentation tools found in search (Sept 2025). OHIF + Cornerstone3D provide the infrastructure; cardiac models must be ported to ONNX separately.
- **VTK.js** [S26] now supports multi-volume 3D rendering (segmentation + source image on separate GPU textures), enabling 3D visualization alongside segmentation—relevant for cardiac 4D cine workflows.

### 5. MRI Physics Simulators

- **JEMRIS** [S27]: Versatile, open-source, multi-purpose MRI simulator. Supports **complex motion** (cardiac, respiratory) and **cardiovascular MR (CMR) sequences** [S28]. GUI + code-based experiment definition. Actively maintained.
- **MRiLab** [S29]: GPU-accelerated numerical MRI simulator with interactive GUI. Supports **multi-dimensional multiple spin species simulations**. Optimized for real-time prototyping. Mature, widely used in education.
- **KomaMRI.jl** [S30]: Julia-based, open-source, GPU-accelerated MRI simulator with **arbitrary motion support** (published 2024). Supports cardiovascular MR and MR angiography simulations [S31].
- **CMRsim** [S32]: Purpose-built for **CMR simulations with realistic motion**. Limited documentation in open search results.
- **Limitations**: None of these simulators expose high-fidelity synthetic cardiac MRI datasets suitable for pre-training segmentation models. Existing synthetic-to-real transfer in cardiac MRI remains niche. nnU-Net, MONAI, and other production models train on real clinical data (ACDC, M&Ms) due to sim-to-real domain gap [S33].

### 6. EF / Volume Estimation from Segmentation

- **Volume from segmentation is standard**: End-diastolic (EDV) and end-systolic (ESV) volumes derived from LV segmentation mask → ejection fraction (EF) = (EDV − ESV) / EDV [S34].
- **nnU-Net accuracy** [S35]: Mean absolute differences vs. reference are **EDV ±2.9 mL, ESV ±3.5 mL, EF ±2.6%** on real-time CMR at rest. Dice coefficients 0.94 (LV), 0.90 (RV), 0.89 (myocardium).
- **End-to-end EF estimation** (without explicit segmentation): Vision Transformer–based models can regress EF directly from cine-MRI video, avoiding intermediate segmentation [S36]. Less common in production; segmentation remains the standard.

## Open Questions

- **Can domain-generalization methods (vs. adaptation) close the vendor gap to <10% Dice loss?** Recent papers claim 65–75% generalization, still far from clinical utility.
- **Are there published cardiac-MRI-specific ONNX models ready for browser deployment?** OHIF v3.10 supports the infrastructure; no cardiac models found in search.
- **Do synthetic MRI simulations (JEMRIS, KomaMRI) + sim-to-real transfer close the performance gap vs. real-data training?** No published benchmarks found; likely still open.
- **Can M&Ms-2 leaderboard results be accessed post-challenge?** Search found challenge announcement but not final leaderboard standings (possible paywall).
- **What is the adoption of browser-based inference (OHIF v3.10) in clinical radiology workflows?** Only released April 2025; adoption/validation unclear.

## Sources

- [S1] GitHub - MIC-DKFZ/nnUNet — https://github.com/mic-dkfz/nnunet
- [S2] CardioSAM: Topology-Aware Decoder Design for High-Precision Cardiac MRI Segmentation — https://arxiv.org/html/2604.03313
- [S3] How good nnU-Net for Segmenting Cardiac MRI: A Comprehensive Evaluation — https://arxiv.org/abs/2408.06358
- [S4] Medical Open Network for Artificial Intelligence (MONAI) — https://siim.org/resource/medical-open-network-for-artificial-intelligence-monai/
- [S5] Comparative Analysis of MONAI Models for Left Ventricle Segmentation on EMIDEC Dataset — https://medium.com/@arehman.bscs22seecs/comparative-analysis-of-monai-models-for-left-ventricle-segmentation-on-emidec-dataset-f6066e8d082f
- [S6] GitHub - wasserth/TotalSegmentator — https://github.com/wasserth/TotalSegmentator
- [S7] Automatic Segmentation of Cardiovascular Structures on Chest CT Data Sets: An Update of the TotalSegmentator — https://www.ejradiology.com/article/S0720-048X(25)00092-0/fulltext
- [S8] GitHub - MarioProjects/MnMsCardiac: 4th place solution of M&Ms Challenge — https://github.com/MarioProjects/MnMsCardiac
- [S9] Multi-Centre, Multi-Vendor and Multi-Disease Cardiac Segmentation: The M&Ms Challenge — https://ieeexplore.ieee.org/document/9458279/
- [S10] Multi-Centre, Multi-Vendor, and Multi-Disease Cardiac Image Segmentation — https://www.researchgate.net/publication/352494774_Multi_Centre_Multi_Vendor_and_Multi_Disease_Cardiac_Segmentation_The_MMs_Challenge
- [S11] M&Ms-2 Challenge — https://www.ub.edu/mnms-2/
- [S12] Deep Learning Segmentation of the Right Ventricle in Cardiac MRI: The M&Ms Challenge — https://www.researchgate.net/publication/370071927_Deep_Learning_Segmentation_of_the_Right_Ventricle_in_Cardiac_MRI_The_Mms_Challenge
- [S13] Frontiers | Deep Learning for Cardiac Image Segmentation: A Review — https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2020.00025/full
- [S14] CMRxRecon2024: A Multi-Modality, Multi-View K-Space Dataset Boosting Universal Machine Learning for Accelerated Cardiac MRI — https://arxiv.org/pdf/2406.19043
- [S15] Studying Robustness of Semantic Segmentation under Domain Shift in cardiac MRI — https://arxiv.org/pdf/2011.07592
- [S16] Domain Adaptation for Medical Image Analysis: A Survey — https://arxiv.org/pdf/2102.09508
- [S17] The Domain Shift Problem of Medical Image Segmentation — https://arxiv.org/pdf/1910.13681
- [S18] Studying Robustness of Semantic Segmentation Under Domain Shift in Cardiac MRI — https://link.springer.com/chapter/10.1007/978-3-030-68107-4_24
- [S19] A Domain-Shift Invariant CNN Framework for Cardiac MRI Segmentation Across Unseen Domains — https://pmc.ncbi.nlm.nih.gov/articles/PMC10501982/
- [S20] Disentangled Representations for Domain-generalized Cardiac Segmentation — https://arxiv.org/pdf/2008.11514
- [S21] Domain generalization in deep learning for contrast-enhanced imaging — https://arxiv.org/pdf/2110.07360
- [S22] OHIF Viewer v3.10 with Local AI Enhanced Segmentation and More — https://ohif.org/newsletters/2025-04-09-ohif%20viewer%20v3.10%20with%20local%20ai%20enhanced%20segmentation%20and%20more--release-note3p10
- [S23] cornerstone3D/CHANGELOG.md — https://github.com/cornerstonejs/cornerstone3D/blob/main/CHANGELOG.md
- [S24] Radiology AI in one page: models, datasets, CPU VLMs, Hugging Face, safe deployment — https://x.com/drpiyushENT/status/1974539639668051968
- [S25] GitHub - CCI-Bonn/OHIF-AI: OHIF DICOM Web viewer with server-side AI integration — https://github.com/CCI-Bonn/OHIF-AI
- [S26] VTK.js v35 Release — https://www.kitware.com/vtk-js-v35-release/
- [S27] JEMRIS: Welcome to the the JEMRIS project — https://www.jemris.org/
- [S28] Towards Modality- and Sampling-Universal Learning Strategies for Accelerating Cardiovascular Imaging: Summary of the CMRxRecon2024 Challenge — https://arxiv.org/pdf/2503.03971
- [S29] MRILab – GPU accelerated MRI simulator – Open Source Imaging — https://www.opensourceimaging.org/project/mrilab-gpu-accelerated-mri-simulator/
- [S30] KomaMRI.jl: An Open-Source Framework for General MRI Simulations with GPU Acceleration — https://arxiv.org/pdf/2301.02702
- [S31] Versatile and Highly Efficient MRI Simulation of Arbitrary Motion in KomaMRI — https://onlinelibrary.wiley.com/doi/10.1002/mrm.70145
- [S32] An Open-Source Framework for General MRI Simulations — https://arxiv.org/pdf/2301.02702
- [S33] Efficient cardiac MRI multi-structure segmentation for cardiovascular assessment with limited annotation by integrating data-level and network-level consistency — https://www.nature.com/articles/s41746-026-02475-y
- [S34] Left Ventricle Segmentation and Volume Estimation on Cardiac MRI using Deep Learning — https://arxiv.org/pdf/1809.06247
- [S35] Assessment of deep learning segmentation for real-time free-breathing cardiac magnetic resonance imaging at rest and under exercise stress — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10866998/
- [S36] Hierarchical Vision Transformers for Cardiac Ejection Fraction Estimation — https://arxiv.org/pdf/2304.00177
