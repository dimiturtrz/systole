# Research Notice Board

## Open Questions

(none pending)

## Settled Findings

| Topic | Deep-Dive | TL;DR |
|-------|-----------|-------|
| Cardiac MRI segmentation landscape (OSS, benchmarks, domain gaps, browser inference, simulators) | `2026-06-22_cardiac-segmentation-oss-landscape-unified.md` | nnU-Net (8.6k⭐) + MONAI (8.3k⭐) dominate; cross-vendor gap unsolved (47% Dice drop Siemens→Philips); browser stack (OHIF v3.10) exists but no cardiac implementations; MRI simulators roadmapped for cardiac but no high-fidelity synthetic datasets yet |

## Progress Log

- **2026-06-22**: Initial landscape survey complete. Five research angles covered: OSS frameworks, M&M benchmarks, domain generalization gap (Siemens→Philips ~42% Dice drop), browser-based inference (OHIF v3.10 April 2025), MRI simulators (JEMRIS, MRiLab, KomaMRI). ~25 tool calls, all findings cited.
