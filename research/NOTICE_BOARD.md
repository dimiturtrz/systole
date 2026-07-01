# Research Notice Board

## Open Questions

(none pending)

## Settled Findings

| Topic | Deep-Dive | TL;DR |
|-------|-----------|-------|
| Cardiac MRI segmentation landscape (OSS, benchmarks, domain gaps, browser inference, simulators) | `2026-06-22_cardiac-segmentation-oss-landscape-unified.md` | nnU-Net (8.6k⭐) + MONAI (8.3k⭐) dominate; cross-vendor gap unsolved (47% Dice drop Siemens→Philips); browser stack (OHIF v3.10) exists but no cardiac implementations; MRI simulators roadmapped for cardiac but no high-fidelity synthetic datasets yet |
| CardioSAM triage (run-on-our-data + provenance gate) | `2026-06-30_cardiosam_triage.md` | NO-GO. Real preprint (arXiv:2604.03313, Ujjwal Jain, single-author, no venue/peer review); claims mean ACDC Dice 93.39% (in-dist, self-reported); CODE GATE = NONE (no repo/weights/license); NO cross-vendor eval; PSO buzzword red flag. Not runnable, not a trustworthy baseline. |
| MRI vendor acquisition params for cine-bSSFP synth (`systole`) | `2026-07-01_mri-vendor-acquisition-params.md` | TR/TE ~vendor-invariant (~3ms/~1.3ms, gradient-limited); FLIP is the real vendor+field axis, SAR-capped ~80° @1.5T / ~50° @3T (PubMed 26509846). ACDC=Siemens Aera 1.5T + Trio 3T; M&Ms data ~1.5T-dominant. Proposed `acquisition.yaml` block (per-vendor TR/TE + per-field flip, all verified:false; Canon extrapolated=LOW). Inflow: fresh-blood boost f=min(1, v·TR/thk), sweep f∈[0,0.6]. Papers document vendor+field but NOT sequence timing — DICOM mining needed to verify. |

## Progress Log

- **2026-06-22**: Initial landscape survey complete. Five research angles covered: OSS frameworks, M&M benchmarks, domain generalization gap (Siemens→Philips ~42% Dice drop), browser-based inference (OHIF v3.10 April 2025), MRI simulators (JEMRIS, MRiLab, KomaMRI). ~25 tool calls, all findings cited.
