# Notice Board — cardiac-seg research

| Date | Deep-dive | Status | Open questions |
|---|---|---|---|
| 2026-06-17 | [Cardiac MRI + EF Foundations](deep_dives/2026-06-17_cardiac-mri-ef-foundations.md) | settled | EF MAE for nnU-Net on ACDC test set (PDF unreadable); papillary inclusion rule confirmation; multi-vendor Dice gap |
| 2026-06-17 | [Application Curriculum & Gaps](deep_dives/2026-06-17_application-curriculum-and-gaps.md) | partial | ACDC label integers **RESOLVED 2026-06-18** (verified geometrically on real masks: 0 bg / 1 RV / 2 myo / 3 LV-cav); ACDC basal/apical slice convention in eval script; nnU-Net ACDC per-class Dice (leaderboard not fetched); Gibbs EF impact quantitative study not found |
| 2026-06-18 | [DL Segmentation + Computational Geometry Curriculum](deep_dives/2026-06-18_ml-geometry-application-curriculum.md) | grounded | UF EEL6935 PDF didn't fetch (ordering from abstract); CS231n/fast.ai exact syllabi not vetted; Bland-Altman canonical URL not pinned |
| 2026-06-20 | [Dataset acquisition + annotation conventions](deep_dives/2026-06-20_dataset-acquisition-and-conventions.md) | grounded | **PAPILLARY RESOLVED**: ACDC/M&Ms-1/M&Ms-2 share the convention (papillary+trabeculae IN LV cavity; M&Ms-2 "follows ACDC standards") → NOT the EF-bias source, bias is domain/intensity. ACDC scanners = Siemens Aera 1.5T + Trio Tim 3T (Dijon). M&Ms-1 = 375/6 centres ES-DE-CA/4 vendors incl Canon. Open: M&Ms-1 per-vendor·field counts (TMI tables); ACDC per-patient 1.5T/3T split not public |
