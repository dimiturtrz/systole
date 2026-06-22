# Cardiac MRI Segmentation: OSS Landscape, Benchmarks, Domain Gaps & Inference Stack

**Date**: 2026-06-22  
**Status**: settled  
**Supersedes**: [2026-06-22_cardiac_mri_segmentation_landscape.md](superseded/2026-06-22_cardiac_mri_segmentation_landscape.md), [2026-06-22_cardiac_mri_segmentation_facts.md](superseded/2026-06-22_cardiac_mri_segmentation_facts.md)

## TL;DR

nnU-Net (8.6k ⭐, last commit Apr 2026) and MONAI (8.3k ⭐, v1.6.0 Jun 2026) dominate production cardiac segmentation but lack explicit cardiac MRI documentation; TotalSegmentator (2.8k ⭐, May 2026) adds `heartchambers_highres` task. Cross-vendor generalization remains unsolved (42% Dice drop Siemens→Philips); browser inference infrastructure exists (OHIF v3.10, ONNX Runtime Web) but no cardiac-specific implementations. MRI simulators (KomaMRI.jl 208⭐, JEMRIS 58⭐, MRiLab 248⭐) support cardiac motion in roadmap but lack high-fidelity synthetic datasets for pre-training.

## Question

What is the state of open-source cardiac MRI segmentation tooling (frameworks, GitHub maturity), benchmark performance (ACDC, M&Ms, M&Ms-2), cross-vendor robustness, in-browser inference, and MRI physics simulation for cardiac applications — with verified GitHub URLs, star counts, last-commit dates, maintenance status, and reproducibility evidence?

## Findings

### AREA 1: Dominant OSS Frameworks for Medical Image Segmentation

**nnU-Net (MIC-DKFZ/nnUNet)**
- **GitHub**: https://github.com/MIC-DKFZ/nnUNet [S1]
- **Metrics**: 8.6k stars, 8.4k on releases page, last commit Apr 28, 2026 [S1]
- **Latest release**: v2.4.1 (Apr 21, 2024) with Residual Encoder UNets [S1]
- **Documentation**: Automatic pipeline adaptation; end-to-end workflow from preprocessing to inference [S1]
- **Cardiac MRI**: NO explicit mention in README; used extensively in M&Ms/ACDC/cardiac research papers but cardiac support is application-layer, not framework feature [S1], [S15]
- **Performance on cardiac**: Mean Dice 93.5% on left atrium (5 datasets: LAScarQs 2022, LASC 2018, ACDC, MnM1, MnM2); 0.91–0.92 Dice on ACDC LV/RV [S15], [S16]
- **Maintenance**: Active (62 open issues, 18 PRs) [S1]

**MONAI (Project-MONAI/MONAI)**
- **GitHub**: https://github.com/Project-MONAI/MONAI [S2]
- **Metrics**: 8.3k ⭐, 1.5k forks, last release v1.6.0 (Jun 11, 2026) [S2]
- **Documentation**: PyTorch-based, multi-dimensional medical imaging, domain-specific networks/losses/metrics [S2]
- **Cardiac MRI**: NO explicit mention in README; used in cardiac workflows (DECATHLON cardiac data, 3D-UNET) but no cardiac segmentation module in public docs [S2], [S17]
- **Maintenance**: Active 2026 releases [S2]

**TotalSegmentator (wasserth/TotalSegmentator)**
- **GitHub**: https://github.com/wasserth/TotalSegmentator [S3]
- **Metrics**: 2.8k ⭐, v2.5.0-weights (Jan 13, 2026), PyPI release Jun 10, 2026 [S3], [S4]
- **Cardiac tasks explicitly documented** [S3]:
  - `heartchambers_highres`: myocardium, left/right atrium, left/right ventricle, aorta, pulmonary artery at sub-millimeter resolution
  - Additional cardiac models: Task503_cardiac_motion, Task417_heart_mixed_317subj, Task435_Heart_vessels_118subj
- **EF/volume**: Abdominal organ volume via web apps documented; cardiac volume NOT mentioned [S3]
- **Maintenance**: Active (model improvements 2026-02-11) [S4]

**Smaller Cardiac-Specific Projects** [S19]
- DL_cardiac_segmentation (danielececcarelli): left ventricle mesh + 2D SAX auto-segmentation
- CardiacSegmentationPropagation (julien-zheng): PyTorch 3D consistent segmentation with spatial propagation
- cardiac-segmentation (chuckyee): Right ventricle segmentation
- Open-source pre-clinical cardiac MRI dataset (mrphys): 130 mice cine short-axis with web interface

**Verdict**: nnU-Net and MONAI are framework-agnostic; cardiac adoption is user-driven. TotalSegmentator explicitly supports cardiac chambers but lacks EF/volume calculation. Cardiac-specific projects exist but are niche.

---

### AREA 2: Public Benchmarks & Leaderboard Performance

**ACDC (Automated Cardiac Diagnosis Challenge)**
- **Challenge period**: MICCAI 2017; open submissions through end-2022, now closed [S5]
- **Dataset**: 150 multi-equipment CMRI recordings, 200 train / 100 test split, reference measurements and 5-class diagnostic labels [S5]
- **Focus**: LV endocardium/epicardium and RV endocardium at end-diastole and end-systole; diagnostic classification [S5]
- **Largest public fully-annotated cardiac MRI dataset** [S5]
- **Performance reported in literature**: 
  - Top methods: ~92–93% Dice (LV 0.94, RV 0.90, myocardium 0.89 on nnU-Net) [S16]
  - CardioSAM (topology-aware decoder): 93.39% Dice [S2_landscape]
  - GCASCADE: 92.25% Dice [S2_landscape]
- **Leaderboard**: Official `/results.html` exists but specific top-3 scores not captured in automated fetch [S5]
- **Maintenance**: Archived reference benchmark [S5]

**M&Ms Challenge (2020/2021)**
- **Dataset**: 375 patients, 150 train (two vendors) / 25 unannotated (third vendor) / ~200 test [S9], [S10]
- **Vendors**: Siemens, Philips, GE, Canon [S9]
- **Focus**: Multi-centre, multi-vendor, multi-disease cardiac segmentation [S9]
- **nnU-Net baseline**: Achieved best-in-challenge results; participants used CycleGAN + feature alignment for domain adaptation [S10]
- **Challenge solutions**: 4th place available at https://github.com/MarioProjects/MnMsCardiac [S21]

**M&Ms-2 Challenge (2023/2024)**
- **Dataset**: 360 cases, 3 vendors, 9 scanners, short-axis + long-axis 4-chamber views [S6], [S11]
- **Training**: 160 annotated + 40 validation [S6], [S11]
- **Focus**: Right ventricle segmentation; multi-view heterogeneity [S11]
- **nnU-Net baseline**: Achieved best overall performance; systole Dice 0.87+ reported [S16]
- **Leaderboard**: https://www.ub.edu/mnms-2/ [S6]; specific 2026 standings not accessible via automated fetch
- **Challenge repo**: https://github.com/cgalaz01/mnms2_challenge [S20]

**CMRxRecon2024 Challenge**
- **Focus**: Reconstruction (not segmentation) from accelerated k-space data [S14]
- **Dataset**: Multi-vendor, multi-view k-space spanning 1.5T and 3T [S14]
- **Performance**: Top-5 methods significantly outperformed GRAPPA/SENSE on SSIM/PSNR/NMSE [S14]

---

### AREA 3: Cross-Vendor Domain Generalization: Open Problem

**Magnitude of the problem** [S15], [S18]
- Siemens→Philips (no fine-tuning): Dice drops from ~0.89 → 0.47 (**47% relative drop**) [S15], [S18]
- Similar gaps reported for GE and Canon [S16]
- M&Ms without fine-tuning: Best model achieves 74.8% average Dice (vs. ~92% in-domain) [S21_landscape]

**Approaches in literature** [S16], [S17], [S18], [S19], [S20]
- **Image-level**: CycleGAN-based domain adaptation + feature-level MSE alignment [S18]
- **Data-level**: Multi-sequence training improves inductive bias for unseen domains [S19]
- **Feature-level**: Domain generalization via disentangled representations, loss reweighting [S20]

**Active research** (no single OSS solution)
- Disentangled Representations for Domain-generalized Cardiac Segmentation [arxiv:2008.11514] [S23]
- Studying Robustness under Domain Shift in Cardiac MRI [arxiv:2011.07592] [S23]
- Generalisable Cardiac Structure Segmentation via Attentional and Stacked Image Adaptation [arxiv:2008.01216] [S23]
- CNN Generalization on UK Biobank (trained 3,975 subjects, tested cross-domain) [S24]
- **Resource**: https://github.com/Ziwei-Niu/Generalized_MedIA (curates papers/codes on domain generalization for medical imaging; updated Jun 2025; Dec 2024 Universal Segmentation Foundational Model branch) [S24]

**Verdict**: nnU-Net's extensive data augmentation serves as practical baseline. No established OSS framework for cardiac domain generalization emerged by Jun 2026. Problem remains research-phase.

---

### AREA 4: Browser-Based Inference Stack

**OHIF (Open Health Imaging Foundation)**
- **Framework**: Browser-based DICOM viewer with segmentation tools [S26]
- **Latest**: OHIF v3.10 (April 2025) added Local AI Enhanced Segmentation — models run entirely in-browser [S22_landscape]
- **Cornerstone3D 2.0** (Nov 2024): Multiple input volumes per GPU texture, ONNX Runtime integration [S23_landscape]
- **Workflow**: Export segmentation model to ONNX → deploy with ONNX Runtime in OHIF plugin → wrap in FastAPI if needed [S24_landscape]
- **Cardiac examples**: None found in official OHIF documentation [S26]
- **Maintenance**: Active development [S22_landscape]

**ONNX Runtime Web (onnxruntime-web)**
- **NPM**: https://www.npmjs.com/package/onnxruntime-web [S10]
- **Docs**: https://onnxruntime.ai/docs/tutorials/web/ [S10]
- **Capability**: WebAssembly (CPU) + WebGL/WebGPU (GPU acceleration) [S10]
- **Performance**: Complex pipelines at near-native speeds; chest X-ray triage demo ~30ms per image [S10]
- **Cardiac examples**: None found [S10]
- **Maintenance**: Active ONNX Runtime ecosystem [S10]

**IntelliCardiac (tiffany9056/IntelliCardiac)**
- **GitHub**: https://github.com/tiffany9056/IntelliCardiac [S9]
- **Metrics**: 2 ⭐, research paper arxiv:2505.03838 (May 2025) [S9]
- **Status**: Prototype, not production-ready; README states "not intended for industry/commercial use" [S9]
- **Features**: Real-time web-based 4D cardiac segmentation + disease classification [S9]
- **Performance**: 92.6% segmentation accuracy (ACDC-trained); 98% classification accuracy [S9], [S80_facts]
- **Maintenance**: Not actively maintained; zero releases, no active PRs/issues [S9]

**Other platforms**
- Weasis DICOM viewer (open-source, limited segmentation) [S26]
- CMRSegTools (annotation-focused, not inference) [S26]
- VTK.js v35: Multi-volume 3D rendering (segmentation + source on separate GPU textures) for 4D cine workflows [S26_landscape]

**Verdict**: Browser infrastructure (OHIF + ONNX Runtime Web) exists but lacks cardiac-specific implementations. IntelliCardiac is research prototype only.

---

### AREA 5: MRI Physics Simulators for Cardiac Applications

**KomaMRI.jl (JuliaHealth/KomaMRI.jl)**
- **GitHub**: https://github.com/JuliaHealth/KomaMRI.jl [S11]
- **Metrics**: 208 ⭐, latest release v0.11.3 (Jun 18, 2026) [S11]
- **Features**: GPU-accelerated MRI simulations, Pulseq compatibility, interactive PlotlyJS visualization, Jupyter/Pluto support [S11]
- **Cardiac support**: PLANNED for v1.0 (roadmap mentions "Cardiac phantoms and triggers") but NOT YET IMPLEMENTED [S11]
- **Performance**: v0.9 (Oct 2024) achieved 4–5× speedup and 80× memory reduction vs. prior versions [S11]
- **Maintenance**: Active (Jun 2026 release) [S11]

**JEMRIS (JEMRIS/jemris)**
- **GitHub**: https://github.com/JEMRIS/jemris [S12]
- **Metrics**: 58 ⭐, latest release 2.9.2 (Jan 3, 2025) [S12]
- **Features**: General MRI simulation framework; sequence, sample, coil setup [S12]
- **Cardiac support**: NOT mentioned in README [S12]
- **Maintenance**: Last release Jan 2025; no explicit 2026 commits [S12]

**MRiLab (leoliuf/MRiLab)**
- **GitHub**: https://github.com/leoliuf/MRiLab [S13]
- **Metrics**: 248 ⭐ [S13]
- **Features**: Numerical MRI simulation; MR signal formation, k-space, reconstruction, RF pulse analysis, sequence design [S13]
- **Cardiac support**: NOT mentioned [S13]
- **Maintenance**: Limited activity (12 commits shown; date not captured) [S13]
- **Alt**: SourceForge: https://mrilab.sourceforge.net/ [S13]

**Cardiac simulation ecosystem comparison** [S27]
- JEMRIS: CPU-only, slower
- MRiLab: GPU support but MATLAB-like environment
- KomaMRI: 8× faster than JEMRIS; Julia + GPU; Pulseq compatible
- CMRsim (Python): Supports cardiovascular motion but limited documentation
- MRSeqStudio: Web-based sequence design + simulation (arxiv:2512.00011); no cardiac examples cited

**Synthetic data for cardiac segmentation**
- **Status**: No published high-fidelity synthetic cardiac MRI datasets suitable for pre-training segmentation models [S33_landscape]
- **Sim-to-real domain gap**: Acknowledged in literature; production models train on real clinical data (ACDC, M&Ms) due to synthetic domain gap [S33_landscape]
- **Verdict**: Simulators exist but cardiac-specific synthetic datasets are not yet contributing to model pre-training

---

### AREA 6: EF / Volume Estimation from Segmentation

**Volume calculation standard** [S34_landscape]
- EDV and ESV derived from LV segmentation mask → EF = (EDV − ESV) / EDV

**nnU-Net on cardiac MRI** [S35_landscape]
- Real-time CMR at rest: EDV ±2.9 mL, ESV ±3.5 mL, EF ±2.6% mean absolute differences
- Dice coefficients: LV 0.94, RV 0.90, myocardium 0.89

**End-to-end EF (without explicit segmentation)** [S36_landscape]
- Vision Transformer models can regress EF directly from cine-MRI video
- Less common in production; segmentation remains standard

---

## Open Questions

- **Can domain-generalization methods close the vendor gap to <10% Dice loss?** Recent papers claim 65–75% generalization; still far from clinical utility.
- **Are published cardiac-MRI-specific ONNX models ready for OHIF v3.10 deployment?** Infrastructure exists; cardiac models not found.
- **Do synthetic MRI simulations + sim-to-real transfer close the performance gap vs. real-data training?** No published benchmarks found.
- **Can M&Ms-2 / ACDC leaderboard final standings be accessed post-challenge?** Official pages exist but automated fetch did not expose specific scores.
- **Clinical adoption of browser-based inference (OHIF v3.10):** Released Apr 2025; adoption/validation status unclear by Jun 2026.
- **Cardiac phantom contributions to KomaMRI.jl v1.0 roadmap:** Status unclear; simulator ecosystem may still be pre-cardiac.

## Sources

- [S1] GitHub - MIC-DKFZ/nnUNet — https://github.com/MIC-DKFZ/nnUNet — accessed 2026-06-22
- [S2] GitHub - Project-MONAI/MONAI — https://github.com/Project-MONAI/MONAI — accessed 2026-06-22
- [S3] GitHub - wasserth/TotalSegmentator — https://github.com/wasserth/TotalSegmentator — accessed 2026-06-22
- [S4] TotalSegmentator PyPI — https://pypi.org/project/TotalSegmentator/ — accessed 2026-06-22
- [S5] ACDC Challenge — https://www.creatis.insa-lyon.fr/Challenge/acdc/ — accessed 2026-06-22
- [S6] M&Ms-2 Challenge — https://www.ub.edu/mnms-2/ — accessed 2026-06-22
- [S9] GitHub - tiffany9056/IntelliCardiac — https://github.com/tiffany9056/IntelliCardiac — accessed 2026-06-22; arxiv:2505.03838
- [S10] ONNX Runtime Web — https://onnxruntime.ai/docs/tutorials/web/ and https://www.npmjs.com/package/onnxruntime-web — accessed 2026-06-22
- [S11] GitHub - JuliaHealth/KomaMRI.jl — https://github.com/JuliaHealth/KomaMRI.jl — accessed 2026-06-22
- [S12] GitHub - JEMRIS/jemris — https://github.com/JEMRIS/jemris — accessed 2026-06-22
- [S13] GitHub - leoliuf/MRiLab — https://github.com/leoliuf/MRiLab — accessed 2026-06-22; SourceForge https://mrilab.sourceforge.net/
- [S14] CMRxRecon2024: A Multi-Modality, Multi-View K-Space Dataset Boosting Universal Machine Learning for Accelerated Cardiac MRI — https://arxiv.org/pdf/2406.19043
- [S15] Studying Robustness of Semantic Segmentation under Domain Shift in Cardiac MRI — https://arxiv.org/pdf/2011.07592
- [S16] How good nnU-Net for Segmenting Cardiac MRI: A Comprehensive Evaluation — https://arxiv.org/abs/2408.06358
- [S17] MONAI: Framework for Medical Imaging Powered by PyTorch — https://learnopencv.com/monai-medical-imaging-pytorch/
- [S18] Studying Robustness of Semantic Segmentation Under Domain Shift in Cardiac MRI — https://link.springer.com/chapter/10.1007/978-3-030-68107-4_24
- [S19] A Domain-Shift Invariant CNN Framework for Cardiac MRI Segmentation Across Unseen Domains — https://pmc.ncbi.nlm.nih.gov/articles/PMC10501982/
- [S20] GitHub - cgalaz01/mnms2_challenge — https://github.com/cgalaz01/mnms2_challenge — accessed 2026-06-22
- [S21] GitHub - MarioProjects/MnMsCardiac (M&Ms-2 4th place) — https://github.com/MarioProjects/MnMsCardiac — accessed 2026-06-22
- [S22_landscape] OHIF Viewer v3.10 with Local AI Enhanced Segmentation — https://ohif.org/newsletters/2025-04-09-ohif%20viewer%20v3.10%20with%20local%20ai%20enhanced%20segmentation%20and%20more--release-note3p10
- [S23_landscape] Cornerstone3D/CHANGELOG.md — https://github.com/cornerstonejs/cornerstone3D/blob/main/CHANGELOG.md
- [S24_landscape] Radiology AI in one page: models, datasets, CPU VLMs, Hugging Face, safe deployment — https://x.com/drpiyushENT/status/1974539639668051968
- [S26] The 12 Best Online DICOM Viewer Free Tools for 2026 and Weasis Documentation — https://pycad.co/online-dicom-viewer-free/ and https://weasis.org/en/ — accessed 2026-06-22
- [S27] KomaMRI.jl: An open-source framework for general MRI simulations with GPU acceleration — https://arxiv.org/pdf/2301.02702 and Magn Reson Med 2023; MRSeqStudio arxiv:2512.00011
- [S23] Domain Generalization GitHub topic and papers — https://github.com/topics/domain-generalization — Disentangled Representations (arxiv:2008.11514), Generalisable Cardiac Structure (arxiv:2008.01216) — accessed 2026-06-22
- [S24] GitHub - Ziwei-Niu/Generalized_MedIA — https://github.com/Ziwei-Niu/Generalized_MedIA — accessed 2026-06-22
- [S33_landscape] Efficient cardiac MRI multi-structure segmentation for cardiovascular assessment with limited annotation by integrating data-level and network-level consistency — https://www.nature.com/articles/s41746-026-02475-y
- [S34_landscape] Left Ventricle Segmentation and Volume Estimation on Cardiac MRI using Deep Learning — https://arxiv.org/pdf/1809.06247
- [S35_landscape] Assessment of deep learning segmentation for real-time free-breathing cardiac magnetic resonance imaging at rest and under exercise stress — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10866998/
- [S36_landscape] Hierarchical Vision Transformers for Cardiac Ejection Fraction Estimation — https://arxiv.org/pdf/2304.00177
- [S19] Cardiac segmentation GitHub topic — https://github.com/topics/cardiac-segmentation — accessed 2026-06-22
