# Research Notice Board

## Open Questions

(none pending)

## Settled Findings

| Topic | Deep-Dive | TL;DR |
|-------|-----------|-------|
| Cardiac MRI segmentation landscape (OSS, benchmarks, domain gaps, browser inference, simulators) | `2026-06-22_cardiac-segmentation-oss-landscape-unified.md` | nnU-Net (8.6k⭐) + MONAI (8.3k⭐) dominate; cross-vendor gap unsolved (47% Dice drop Siemens→Philips); browser stack (OHIF v3.10) exists but no cardiac implementations; MRI simulators roadmapped for cardiac but no high-fidelity synthetic datasets yet |
| CardioSAM triage (run-on-our-data + provenance gate) | `2026-06-30_cardiosam_triage.md` | NO-GO. Real preprint (arXiv:2604.03313, Ujjwal Jain, single-author, no venue/peer review); claims mean ACDC Dice 93.39% (in-dist, self-reported); CODE GATE = NONE (no repo/weights/license); NO cross-vendor eval; PSO buzzword red flag. Not runnable, not a trustworthy baseline. |
| MRI vendor acquisition params for cine-bSSFP synth (`systole`) | `2026-07-01_mri-vendor-acquisition-params.md` | TR/TE ~vendor-invariant (~3ms/~1.3ms, gradient-limited); FLIP is the real vendor+field axis, SAR-capped ~80° @1.5T / ~50° @3T (PubMed 26509846). ACDC=Siemens Aera 1.5T + Trio 3T; M&Ms data ~1.5T-dominant. Proposed `acquisition.yaml` block (per-vendor TR/TE + per-field flip, all verified:false; Canon extrapolated=LOW). Inflow: fresh-blood boost f=min(1, v·TR/thk), sweep f∈[0,0.6]. Papers document vendor+field but NOT sequence timing — DICOM mining needed to verify. |
| Cardiac anatomy generation SOTA (SSM, parametric models, learned generative, topology) | `2026-07-01_cardiac-anatomy-generation-sota.md` | Three tiers: (1) parametric (MRXCAT2.0, UK Biobank HeartSSM) — transparent anatomy, biophysical ground truth, manual pathology params; (2) learned (VAE-GAN, diffusion) — realistic diversity, mode collapse risk, large dataset needed; (3) hybrid (parametric+learned) — complementary strengths. Topology (RV-myo-LV validity) unsolved end-to-end; TPM Loss (2025) mitigates 93% violations. Whole-FOV (cardiac+lung+liver) unintegrated. Recommend parametric-first (MRXCAT2.0) for MVP + shape-prior constraints + post-hoc mesh validation. |

## Progress Log

- **2026-06-22**: Initial landscape survey complete. Five research angles covered: OSS frameworks, M&M benchmarks, domain generalization gap (Siemens→Philips ~42% Dice drop), browser-based inference (OHIF v3.10 April 2025), MRI simulators (JEMRIS, MRiLab, KomaMRI). ~25 tool calls, all findings cited.
- **2026-07-01**: Cardiac anatomy generation SOTA deep-dive complete. Three parametric tiers (SSM/biomechanical/learned), evidence on topology preservation gaps, ready-to-use assets (MRXCAT2.0, UK Biobank HeartSSM, SynthSeg cardiac). Parametric-vs-learned tradeoff analyzed: parametric = transparent + controllable pathology, learned = realistic diversity but mode collapse. Recommendation: parametric-first MVP (MRXCAT2.0) with shape-prior constraints. Caller re-verified 5 load-bearing citations (Bai arXiv:2603.28711, MRXCAT2.0 JCMR 2023, TPM Loss arXiv:2503.07874, DG review 2310.08598, pathology-synth VAE-GAN 2209.04223) — all real, specifics accurate. Added [S19] Rodero/KCL SSM cohort (Zenodo, CC-BY-4.0, PCA mode-extrapolated ±SD "extreme" four-chamber meshes) — most directly-reusable coverage-by-construction asset, missed in first pass. **Action**: systole v1 anatomy generation decision (parametric vs hybrid, MRXCAT2.0 + Rodero integration start) due before v1 MVP freeze.
