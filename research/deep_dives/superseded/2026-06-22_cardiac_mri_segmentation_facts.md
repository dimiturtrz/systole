# Cardiac MRI Segmentation: OSS Landscape & Benchmark Findings

**Date**: 2026-06-22  
**Status**: settled  
**Supersedes**: none

## TL;DR

nnU-Net (8.6k GitHub stars, 8.4k on releases page, last update Apr 2026 [S1]) and MONAI (8.3k stars, v1.6.0 released Jun 11, 2026 [S2]) dominate automated cardiac segmentation; neither explicitly documents cardiac MRI examples or EF/volume calculation in public READMEs. TotalSegmentator (2.8k stars [S3], PyPI Jun 10, 2026 [S4]) includes `heartchambers_highres` task for cardiac chambers at sub-millimeter resolution [S3]. ACDC challenge (closed end-2022 [S5]) and M&Ms-2 (RV focus, leaderboard at ub.edu/mnms-2/ [S6]) remain the primary public benchmarks; specific 2026 leaderboard Dice scores not accessible via public web pages. Cross-vendor/domain-generalization research is active (GitHub topic updated, papers through 2026 [S7-S8]) but no single dominant OSS framework. Web inference (IntelliCardiac prototype, 2 stars [S9]; ONNX Runtime Web for browser [S10]) and MRI simulation (KomaMRI.jl 208 stars, cardiac phantom features planned v1.0 [S11]; JEMRIS 58 stars, no cardiac-specific mention [S12]; MRiLab 248 stars [S13]) all exist but cardiac-specific integration is limited.

## Question

What are the verified facts on (1) dominant open-source frameworks for cardiac/medical-image segmentation with explicit cardiac MRI features and EF/volume support; (2) M&M / ACDC challenge benchmarks and leaderboard results; (3) cross-vendor/domain-generalization cardiac segmentation in OSS; (4) in-browser cardiac MRI inference capability; (5) cardiac MRI physics simulators in OSS — for each, GitHub URLs, star counts, last commit/release dates (2026-06-22 verification), README features, and maintenance status?

## Findings

### AREA 1: Dominant OSS for Cardiac/Medical-Image Segmentation

**nnU-Net (MIC-DKFZ/nnUNet)**
- **GitHub**: https://github.com/MIC-DKFZ/nnUNet [S1]
- **Star count**: 8.6k (repository page); 8.4k (releases page) [S1]
- **Last update**: April 28, 2026 [S1]
- **Latest release**: v2.4.1 (April 21, 2024) with Residual Encoder UNets [S1]
- **README features**: "automatically adapts its pipeline to a dataset" by analyzing training data; "end-to-end workflow from preprocessing to training, model selection, and inference" [S1]
- **Cardiac MRI explicit mention**: NO — README and accessible repository materials do NOT reference cardiac segmentation, ejection fraction, volume computation, or cardiac MRI examples [S1]
- **Language**: 98.6% Python [S1]
- **Maintenance**: Active (62 open issues, 18 PRs as of last fetch) [S1]
- **Note**: nnU-Net is deployed in multiple cardiac segmentation research papers (e.g., M&Ms-2 challenge winning solutions [S14]; ACDC dataset studies [S15]; nnU-Net comprehensive evaluation on cardiac MRI datasets including ACDC, MnM1, MnM2 [S16]) but no cardiac-native features in the framework itself.

**MONAI (Project-MONAI/MONAI)**
- **GitHub**: https://github.com/Project-MONAI/MONAI [S2]
- **Star count**: 8.3k stars with 1.5k forks [S2]
- **Last release**: v1.6.0 published June 11, 2026 [S2]
- **README features**: "PyTorch-based, open-source framework for deep learning in healthcare imaging"; "flexible pre-processing for multi-dimensional medical imaging data" and "domain-specific implementations for networks, losses, evaluation metrics" [S2]
- **Cardiac MRI explicit mention**: NO — accessible README content does NOT mention cardiac MRI segmentation, ejection fraction, or volume examples [S2]
- **Language**: 95.5% Python, 2.3% C++, 1.9% CUDA [S2]
- **Maintenance**: Active with recent 2026 releases [S2]
- **Note**: MONAI used in cardiac segmentation workflows (e.g., 3D-UNET on DECATHLON cardiac data [S17]; June 2026 cardiac MRI multi-structure segmentation paper using MONAI-adjacent techniques [S18]) but no explicit cardiac segmentation module documented in README.

**TotalSegmentator (wasserth/TotalSegmentator)**
- **GitHub**: https://github.com/wasserth/TotalSegmentator [S3]
- **Star count**: 2.8k stars (reported as "thousands" in rounded display) [S3]
- **Latest release**: v2.5.0-weights (Jan 13, 2026); PyPI Jun 10, 2026 [S4]
- **README features**: Explicitly lists cardiac tasks:
  - `heartchambers_highres`: "myocardium, atrium_left, ventricle_left, atrium_right, ventricle_right, aorta, pulmonary_artery" trained at "sub-millimeter resolution" [S3]
  - Additional cardiac models: Task503_cardiac_motion, Task417_heart_mixed_317subj, Task435_Heart_vessels_118subj [S3]
- **EF/volume calculation**: No explicit mention; volume computation offered for "abdominal organ volume" via web apps, not cardiac [S3]
- **Maintenance**: Active (model improvements referenced for 2026-02-11 [S3]) [S4]

**Cardiac-Specific Smaller Projects**
- **DL_cardiac_segmentation** (danielececcarelli): Deep learning cardiac image segmentation with left ventricle mesh and 2D SAX auto-segmentation [S19]
- **CardiacSegmentationPropagation** (julien-zheng): PyTorch 3D consistent and robust cardiac segmentation by deep learning with spatial propagation [S19]
- **cardiac-segmentation** (chuckyee): Right ventricle segmentation in cardiac MRI [S19]
- **Open-source pre-clinical cardiac MRI dataset** (mrphys/Open-Source_Pre-Clinical_Segmentation): Complete cine short-axis from 130 mice with web-based interface [S19]

### AREA 2: M&M / M&Ms-2 and ACDC Grand Challenges

**ACDC (Automated Cardiac Diagnosis Challenge)**
- **Official website**: https://www.creatis.insa-lyon.fr/Challenge/acdc/ [S5]
- **Challenge period**: MICCAI 2017; remained open for submissions until end of 2022; now closed [S5]
- **Focus**: Segmentation of left ventricular endocardium/epicardium and right ventricular endocardium for end-diastolic and end-systolic phases; classification into 5 diagnostic classes [S5]
- **Dataset**: 150 multi-equipment CMRI recordings with reference measurements and classification from two medical experts; largest publicly available fully annotated cardiac MRI dataset [S5]
- **Final leaderboard**: Referenced at `/results.html` on official website [S5]; specific top-3 Dice scores NOT accessible via web fetch (leaderboard page structure not captured)
- **Dice score ranges**: Literature reports ~0.8–0.96 for LV/RV segmentation across published methods, but official final rankings not verified in this research [S5]
- **Maintenance status**: Challenge closed; dataset and results remain archived as reference benchmark [S5]

**M&Ms-2 Challenge (Multi-Vendor, Multi-Disease Right Ventricle Segmentation)**
- **Official website**: https://www.ub.edu/mnms-2/ [S6]
- **Challenge focus**: Right ventricle (RV) blood pool segmentation across multiple views (short-axis and 4-chamber long-axis) and centers, addressing RV's complex shape and ill-defined borders in pathology (dilated RV, tricuspid insufficiency, arrhythmogenesis, Tetralogy of Fallot, interatrial communication) [S6]
- **Dataset**: 360 diverse CMR cases from 3 vendors, 9 scanners [S6]
- **Leaderboard URL**: https://www.ub.edu/mnms-2/ [S6]
- **Specific 2026 Dice scores**: NOT accessible via public web pages; official leaderboard structure not exposed to web fetch
- **Challenge solutions**: GitHub repository https://github.com/cgalaz01/mnms2_challenge documents the challenge [S20]; 4th place solution available at https://github.com/MarioProjects/MnMsCardiac with nnU-Net as competitive baseline [S21]
- **nnU-Net systole baseline**: Confirmed to achieve strong results (0.87+ Dice for systole in literature [S16]) but specific M&Ms-2 leaderboard Dice not captured
- **Maintenance**: Challenge infrastructure active; solutions repository maintained [S20]

**Verified Cardiac MRI Segmentation Performance References**
- nnU-Net evaluated on 5 datasets (LAScarQs 2022, LASC 2018, ACDC, MnM1, MnM2): mean Dice 93.5% for left atrium with Hausdorff Distance (95%) 3.2mm [S16]
- Hybrid DL framework (Autoencoders + CNN + RNN) achieves Dice 0.955 for left ventricle with 95% EF correlation to manual segmentation [S22]
- IntelliCardiac web platform: 92.6% overall segmentation accuracy on ACDC-trained model [S9]

### AREA 3: Cross-Vendor / Domain-Generalization Cardiac Segmentation

**Academic Research (No Single Dominant OSS)**
- **Disentangled Representations for Domain-generalized Cardiac Segmentation** (arxiv:2008.11514) [S23]
- **Studying Robustness under Domain Shift in Cardiac MRI** (arxiv:2011.07592) [S23]
- **Generalisable Cardiac Structure Segmentation via Attentional and Stacked Image Adaptation** (arxiv:2008.01216) [S23]
- **CNN Generalization on UK Biobank**: Trained on 3,975 subjects; tested cross-domain on ACDC, BSCMR-AS [S24]
- **BayeSeg: Bayesian Modeling for Interpretable Generalizability** (arxiv:2303.01710) — multi-dataset evaluation [S25]

**GitHub Resources**
- **Generalized_MedIA** (Ziwei-Niu/Generalized_MedIA): Updated Jun 01, 2025; added Universal Segmentation Foundational Model branch Dec 25, 2024; organizes papers, codes, resources on domain generalization for medical image analysis including cardiac [S24]
- **Domain generalization GitHub topic**: https://github.com/topics/domain-generalization (last sorted by updated; cardiac projects present but not exclusively maintained for cardiac) [S23]

**Status**: Domain generalization for cardiac segmentation is an active research area with papers through 2026 [S24], but no single established OSS framework has emerged. nnU-Net's extensive data augmentation serves as the practical baseline for robustness [S24]. No stable, actively-maintained cardiac-specific domain-generalization toolkit found in 2026-06-22 search.

### AREA 4: In-Browser Medical-Image ML Inference

**IntelliCardiac (tiffany9056/IntelliCardiac)**
- **GitHub**: https://github.com/tiffany9056/IntelliCardiac [S9]
- **Star count**: 2 stars [S9]
- **Last commit date**: Not explicitly displayed; research paper arxiv:2505.03838 (May 2025) [S9]
- **Status**: Prototype, research implementation — "not intended for industry-level or commercial use" per README; zero releases, no active PRs/issues [S9]
- **Features**: Real-time visualization; web-based interface for segmentation of 4D cardiac images and disease classification [S9]
- **Performance**: 92.6% segmentation accuracy (trained on ACDC dataset); 98% classification accuracy (5 diagnostic categories) [S9]
- **Maintenance**: Not actively maintained for production; research demonstration only [S9]

**ONNX Runtime Web (onnxruntime-web)**
- **NPM**: https://www.npmjs.com/package/onnxruntime-web [S10]
- **Documentation**: https://onnxruntime.ai/docs/tutorials/web/ [S10]
- **Capability**: JavaScript library for running ONNX models in browsers using WebAssembly (CPU) and WebGL (GPU) [S10]
- **Performance**: Complex pipelines run at "near-native speeds" via Wasm; WebGPU backends for GPU acceleration [S10]
- **Cardiac examples**: Search found no explicit cardiac inference examples in ONNX Runtime Web documentation; medical imaging (chest X-ray triage) demonstrated (~30ms per image [S10]), but cardiac not mentioned [S10]
- **Maintenance**: Active — part of ONNX Runtime ecosystem [S10]

**Other Web-Based Cardiac Platforms**
- **Weasis DICOM viewer**: https://weasis.org/en/ (free, open-source DICOM viewer; segmentation tooling limited) [S26]
- **OHIF (Open Health Imaging Foundation)**: Browser-based framework with segmentation tools; used for "production-grade, HIPAA-compliant" medical imaging applications; no explicit cardiac ML integration documented [S26]
- **CMRSegTools**: Open-source software for acute myocardial infarct segmentation in CMR (focused on clinical annotation, not inference) [S26]

**Assessment**: No mainstream, actively-maintained in-browser cardiac segmentation inference platform found. IntelliCardiac is research-only. ONNX Runtime Web exists but lacks cardiac examples and documentation.

### AREA 5: MRI Physics Simulators (OSS)

**KomaMRI.jl (JuliaHealth/KomaMRI.jl)**
- **GitHub**: https://github.com/JuliaHealth/KomaMRI.jl [S11]
- **Star count**: 208 stars [S11]
- **Latest release**: KomaMRIBase-v0.11.3 (June 18, 2026) [S11]
- **Language**: Julia [S11]
- **README features**: "highly efficient ⚡ MRI simulations"; Pulseq compatibility; GPU acceleration (CPU + GPU parallelization); interactive PlotlyJS visualization; web-based GUI; Jupyter/Pluto support [S11]
- **Cardiac MRI support**: PLANNED but NOT YET IMPLEMENTED — roadmap explicitly mentions "Cardiac phantoms and triggers" for version 1.0 [S11]
- **Performance**: v0.9 (Oct 2024) achieved "4-5x faster" execution and "80x less memory" vs. prior versions [S11]
- **Maintenance**: Active (June 2026 release) [S11]

**JEMRIS (JEMRIS/jemris)**
- **GitHub**: https://github.com/JEMRIS/jemris [S12]
- **Star count**: 58 stars [S12]
- **Latest release**: JEMRIS 2.9.2 (January 3, 2025) [S12]
- **README features**: "General MRI simulation framework" for sequence, sample, and coil setup [S12]
- **Cardiac MRI support**: NOT mentioned in README; general-purpose framework [S12]
- **Maintenance status**: Last release Jan 2025; no explicit 2026 commits documented in this research [S12]

**MRiLab (leoliuf/MRiLab)**
- **GitHub**: https://github.com/leoliuf/MRiLab [S13]
- **Star count**: 248 stars [S13]
- **README features**: "Numerical MRI simulation platform"; simulates MR signal formation, k-space acquisition, MR image reconstruction; RF pulse analysis, sequence design, coil configuration toolboxes [S13]
- **Cardiac MRI support**: NOT mentioned; no cardiac-specific tooling documented [S13]
- **Maintenance**: Limited activity (12 commits shown; commit date not explicitly captured in 2026 fetch) [S13]
- **SourceForge**: Also available on SourceForge: https://mrilab.sourceforge.net/ [S13]

**KomaMRI Cross-Reference: Advanced Cardiac Simulation Study (2023)**
- Recent arxiv paper (2301.02702) confirms KomaMRI vs. JEMRIS + MRiLab:
  - **JEMRIS**: Open-source, but CPU-only, slower [S27]
  - **MRiLab**: GPU support but written in MATLAB-like env [S27]
  - **KomaMRI (Julia)**: 8x faster than JEMRIS on personal computers; GPU support via Julia; Pulseq compatibility [S27]
  - **Cardiac-specific simulators mentioned**: CMRsim (Python, supports cardiovascular/cardiac motion) and KomaMRI (both support cardiac simulation in roadmap/capability) [S27]

**MRSeqStudio**: Recent web-based MRI sequence design platform (arxiv:2512.00011); free, open-source; provides design + simulation "as a service" but no explicit cardiac examples cited [S27]

## Open Questions

- **ACDC final leaderboard Dice scores**: Official results page exists but specific top-3 team scores and method names not captured in this research. Direct download of `/results.html` would verify.
- **M&Ms-2 2026 leaderboard standings**: Challenge infrastructure active but leaderboard page not exposed to automated fetch; verification requires manual site inspection or challenge host API.
- **nnU-Net / MONAI cardiac support**: Both frameworks used extensively in cardiac research but no native cardiac documentation (EF, volume) in public READMEs. Cardiac segmentation may be application-layer work, not framework feature.
- **Cross-vendor cardiac segmentation production OSS**: Papers exist; GitHub repos (e.g., Generalized_MedIA) curate resources but no single maintained OSS implementation emerged in 2026-06-22 search. May reflect that domain generalization is still research-phase for cardiac.
- **IntelliCardiac production readiness**: Prototype status clear; unclear if derivative production implementations exist elsewhere.
- **Cardiac MRI simulator adoption**: KomaMRI planned for cardiac v1.0; unclear if any cardiac phantoms/sequences have been contributed by 2026-06-22. Simulator ecosystem may still be pre-cardiac.

## Sources

- [S1] GitHub - MIC-DKFZ/nnUNet — https://github.com/MIC-DKFZ/nnUNet — accessed 2026-06-22
- [S2] GitHub - Project-MONAI/MONAI — https://github.com/Project-MONAI/MONAI — accessed 2026-06-22
- [S3] GitHub - wasserth/TotalSegmentator — https://github.com/wasserth/TotalSegmentator — accessed 2026-06-22
- [S4] TotalSegmentator PyPI — https://pypi.org/project/TotalSegmentator/ — accessed 2026-06-22
- [S5] ACDC Challenge — https://www.creatis.insa-lyon.fr/Challenge/acdc/ — accessed 2026-06-22
- [S6] M&Ms-2 Challenge — https://www.ub.edu/mnms-2/ — accessed 2026-06-22
- [S7] GitHub domain-generalization topic — https://github.com/topics/domain-generalization — accessed 2026-06-22
- [S8] Disentangled Representations for Domain-generalized Cardiac Segmentation — arxiv:2008.11514
- [S9] GitHub - tiffany9056/IntelliCardiac — https://github.com/tiffany9056/IntelliCardiac — accessed 2026-06-22; arxiv:2505.03838 — accessed 2026-06-22
- [S10] ONNX Runtime Web documentation — https://onnxruntime.ai/docs/tutorials/web/ — accessed 2026-06-22; NPM onnxruntime-web — https://www.npmjs.com/package/onnxruntime-web — accessed 2026-06-22
- [S11] GitHub - JuliaHealth/KomaMRI.jl — https://github.com/JuliaHealth/KomaMRI.jl — accessed 2026-06-22
- [S12] GitHub - JEMRIS/jemris — https://github.com/JEMRIS/jemris — accessed 2026-06-22
- [S13] GitHub - leoliuf/MRiLab — https://github.com/leoliuf/MRiLab — accessed 2026-06-22; MRiLab SourceForge — https://mrilab.sourceforge.net/ — accessed 2026-06-22
- [S14] GitHub - MarioProjects/MnMsCardiac (M&Ms-2 4th place) — https://github.com/MarioProjects/MnMsCardiac — accessed 2026-06-22
- [S15] How good nnU-Net for Segmenting Cardiac MRI: A Comprehensive Evaluation — arxiv:2408.06358
- [S16] Left Atrial Segmentation with nnU-Net Using MRI — arxiv:2511.04071
- [S17] MONAI: Framework for Medical Imaging Powered by PyTorch — https://learnopencv.com/monai-medical-imaging-pytorch/ — accessed 2026-06-22
- [S18] Efficient cardiac MRI multi-structure segmentation for cardiovascular assessment with limited annotation by integrating data-level and network-level consistency — npj Digital Medicine, 2026
- [S19] Cardiac segmentation GitHub topic — https://github.com/topics/cardiac-segmentation — accessed 2026-06-22
- [S20] GitHub - cgalaz01/mnms2_challenge — https://github.com/cgalaz01/mnms2_challenge — accessed 2026-06-22
- [S21] M&Ms Challenge — https://www.researchgate.net/figure/Comparisons-on-M-Ms-Challenge-meanstd-DICE-HD-mm_tbl2_361737094 — accessed 2026-06-22
- [S22] Hybrid deep learning for computational precision in cardiac MRI segmentation: Integrating Autoencoders, CNNs, and RNNs for enhanced structural analysis — ScienceDirect, 2025
- [S23] Studying Robustness of Semantic Segmentation under Domain Shift in cardiac MRI — arxiv:2011.07592; Generalisable Cardiac Structure Segmentation via Attentional and Stacked Image Adaptation — arxiv:2008.01216; domain-generalization GitHub topic — https://github.com/topics/domain-generalization — accessed 2026-06-22
- [S24] GitHub - Ziwei-Niu/Generalized_MedIA — https://github.com/Ziwei-Niu/Generalized_MedIA — accessed 2026-06-22
- [S25] BayeSeg: Bayesian Modeling for Medical Image Segmentation with Interpretable Generalizability — arxiv:2303.01710
- [S26] The 12 Best Online DICOM Viewer Free Tools for 2026 — https://pycad.co/online-dicom-viewer-free/ — accessed 2026-06-22; Weasis Documentation — https://weasis.org/en/ — accessed 2026-06-22
- [S27] KomaMRI.jl: An open-source framework for general MRI simulations with GPU acceleration — Magn Reson Med, 2023, arxiv:2301.02702; MRSeqStudio — arxiv:2512.00011
