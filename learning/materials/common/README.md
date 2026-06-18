# learning/materials/common

**Cross-modality** theory — applies to cardiac imaging regardless of MRI / CT / echo.
Kept separate from `mri/` so the CT and echo lanes reuse it instead of duplicating.

- [cardiac-anatomy-and-cycle.md](cardiac-anatomy-and-cycle.md) — B1: chambers, myocardium, the cardiac cycle, ED/ES.
- [ejection-fraction.md](ejection-fraction.md) — B2: EF formula, volumes, normal ranges, clinical thresholds, related metrics.
- [G_geometry-and-volumetry.md](G_geometry-and-volumetry.md) — B3 + the geometry thread: voxel→volume, Simpson's method, meshing, wall thickness, papillary convention.
- [M_segmentation-theory.md](M_segmentation-theory.md) — C2: U-Net, nnU-Net, 2D vs 3D, loss, augmentation, patient-level splits.
- [E_evaluation-theory.md](E_evaluation-theory.md) — Dice/Jaccard, Hausdorff/HD95, ASSD, volume & EF agreement (Bland-Altman), calibration, failure analysis.

Modality-specific physics + datasets live in [../mri/](../mri/) (and later `../ct/`, `../echo/`).
Scope and "are we done?" bar: [../field-map.md](../field-map.md).

Grounded in [../../../research/deep_dives/](../../../research/deep_dives/) (esp.
`2026-06-17_application-curriculum-and-gaps.md` and `…_cardiac-mri-ef-foundations.md`);
medical facts flagged where unverified.
