# Papillary Muscle and Trabecular Volume Fractions in Cardiac Anatomy

**Date**: 2026-07-15
**Status**: settled
**Supersedes**: (none)

## TL;DR

In normal adults, papillary + trabecular structures comprise 11.9 ± 5.6% of LV cavity volume and 20.2 ± 4.3% of LV mass [S1]. Normal LV trabeculation alone measures 24.8 ± 7.1% of total LV myocardial volume [S2]. RV trabeculated volumes are comparable to LV (males 44.6 ± 11.4 ml/m² RV vs. 43.1 ± 8.7 ml/m² LV, indexed to BSA) [S5]. Cavity volume differences with inclusion/exclusion of papillary/trabecular structures reach 25–68% (EDV–ESV) in disease states [S3].

## Question

Four quantitative research targets:
1. LV papillary muscle + trabecular mass as % of LV blood-pool cavity volume (normal adults)?
2. RV equivalent fractions (RV heavily trabeculated; moderator band, coarse trabeculae)?
3. Short-axis slice area-fraction measurements of papillary/trabecular structures?
4. Cavity volume delta: measured LV/RV cavity volume WITH vs. WITHOUT papillary/trabecular inclusion?

## Findings

### LV Papillary Muscle + Trabecular Volume as Fraction of Cavity

- **Normal adults (n=288)**: Papillary muscle and trabeculae account for **11.9 ± 5.6% of LV end-diastolic volume (EDV)** [S1]. Sex-based difference: males 14.1 ± 5.0% vs. females 9.9 ± 5.3% (p < 0.001) [S1]. Age-related increase from ~9.4% in subjects in their twenties to ~13.8% in their sixties [S1].
- **As % of LV myocardial mass**: 20.2 ± 4.3% in normal controls [S1], showing no significant sex difference when expressed as mass fraction [S1].
- **Normal LV trabeculation alone** (excluding papillary muscles): 24.8 ± 7.1% of total LV myocardial volume in healthy controls [S2]. In disease: dilated cardiomyopathy 30.3 ± 14.3%, isolated LVNC 42.6 ± 13.8% [S2].
- **HCM patients**: Papillary muscle mass 6.6 ± 2.5 g/m² (clinical HCM), 4.0 ± 0.9 g/m² (subclinical mutation carriers) [S4].

### RV Trabecular Fractions and Architecture

- **RV trabeculated volumes (indexed to BSA) in healthy subjects**: Males 44.6 ± 11.4 ml/m², Females 36.4 ± 9.6 ml/m² [S5]. RV volumes were significantly larger in males and decreased with age (p < 0.001) [S5].
- **LV trabeculated volumes (indexed to BSA) for comparison**: Males 43.1 ± 8.7 ml/m², Females 38.1 ± 5.9 ml/m² [S5]. RV and LV trabeculated volumes show similar magnitudes but distinct quantification in healthy subjects [S5].
- **Non-compacted-to-compacted ratio (NC/C)** in RV and LV: Mean NC/C ratio 1.9 ± 0.7 in females, 2.1 ± 0.8 in males; 23.9% of healthy subjects had NC/C ratios exceeding 2.3, conventionally considered pathological [S5].
- **Septomarginal trabeculation (SMT, RV moderator band complex)**: Identified as the most anterior trabeculation from interventricular septum toward apex; mass derived from volume using myocardial density 1.05 g/cm³; quantified via manual contour tracing on CMR [S6]. No specific percentage of RV cavity given in literature searched.

### Short-Axis Area-Fraction Measurements

- Short-axis cine images are the standard for papillary muscle contour tracing; anterolateral and posteromedial papillary muscles contoured separately on each slice containing papillary tissue [S7]. Hypertrophied posteromedial papillary muscle example: 11 mm in short-axis measurement [S7].
- No published area-fraction percentages for short-axis slices found in peer-reviewed literature. Papillary muscle quantification typically done via **volumetric summation** (3D stack integration) rather than 2D area fractions; alternative **threshold-based automatic exclusion** (morphological operations) used in automated segmentation pipelines [S7].

### Cavity Volume Deltas (Inclusion/Exclusion)

- **In HCM patients**: Including vs. excluding papillary muscles and trabeculae produced **25% difference in end-diastolic volume (EDV) and 68% in end-systolic volume (ESV)** [S3]. LV mass difference: 35 ± 10 g or 20 ± 4% [S3].
- **In HOCM patients (n=74)**: Using "mask method" (PTMs included in myocardium) vs. conventional (excluded):
  - LV EDV: 45.5% decrease [S8]
  - LV ESV: 128.9% decrease [S8]
  - LVEF: increased from 64.3% to 77.2% (+20.1 percentage points) [S8]
  - LV mass: increased by 26.9% [S8]
  - Gender variation: LVMI increased 30.2% in women, 25.7% in men [S8]
- **In advanced systolic dysfunction**: Exclusion of papillary/trabecular muscles from cavity volume yields **17% higher indexed LV mass, 20% lower indexed LV diastolic volume, 13% higher LV ejection fraction** compared to standard inclusion [S9].
- **Volume change example (absolute)**: EDV shifts from 149 ml to 160 ml (11 ml change), ESV from 48 ml to 54 ml (6 ml change) by removing papillary/trabecular from cavity [S9]. LV mass: 157 ± 71 g (with inclusion) vs. 141 ± 62 g (without) [S9].

## Open Questions

- **Short-axis area fractions**: Published literature does not quantify papillary/trabecular area as % of slice cross-section; volumetric (3D) quantification dominates.
- **RV percentage of cavity**: Most RV studies quantify absolute volume (ml/m²) indexed to BSA; percentage-of-cavity framing rare in RV literature (unlike LV 11.9% standard).
- **Papillary muscle volume alone (vs. combined with trabeculae)**: Most CMR studies report combined papillary + trabecular fraction; isolated papillary-only percentages scarce.
- **Moderator band specific volume**: No published percentage-of-RV-cavity for moderator band in healthy subjects found.
- **Age-dependent fractions in RV**: RV trabeculation data limited; LV age-dependence (9.4%→13.8%) not yet validated for RV.

## Sources

- [S1] **Effect of papillary muscle and trabeculae on left ventricular function analysis via computed tomography: A cross-sectional study** — PMC10659619 (2024) — Normal control cohort n=288; 11.9 ± 5.6% EDV, 20.2 ± 4.3% LVM fractions.
- [S2] **Quantification of left ventricular trabeculae using cardiovascular magnetic resonance for the diagnosis of left ventricular non-compaction: evaluation of trabecular volume and refined semi-quantitative criteria** — PMC4855408 (JCMR 2016) — Normal healthy controls 24.8 ± 7.1% trabeculated LV volume; DCM 30.3 ± 14.3%; isolated LVNC 42.6 ± 13.8%.
- [S3] **Effect of Papillary Muscles and Trabeculae on Left Ventricular Measurement Using Cardiovascular Magnetic Resonance Imaging in Patients with Hypertrophic Cardiomyopathy** — PMC4296277 (Korean J Radiol. 2015) — HCM: 25% EDV, 68% ESV difference between inclusion/exclusion; 35 ± 10 g or 20 ± 4% LV mass delta.
- [S4] **Impact of the papillary muscles on cardiac magnetic resonance image analysis of important left ventricular parameters in hypertrophic cardiomyopathy** — PMC4840113 (2016) — HCM papillary muscle mass 6.6 ± 2.5 g/m², LVEF relative increase 4.5 ± 1.8%, LV mass increase 8.7 ± 2.6%.
- [S5] **Assessment of left and right ventricular trabeculation and non-compacted myocardium in a large selected healthy reference population** — PMC4043455 (2013) — RV trabeculated volume males 44.6 ± 11.4 ml/m², females 36.4 ± 9.6 ml/m²; LV males 43.1 ± 8.7 ml/m², females 38.1 ± 5.9 ml/m²; NC/C 1.9 ± 0.7 (F), 2.1 ± 0.8 (M).
- [S6] **Pulmonary hypertension: role of septomarginal trabeculation and moderator band complex assessed by cardiac magnetic resonance imaging** — PMC7860696 (2020) — SMT quantification methodology; mass from volume (1.05 g/cm³); manual contour tracing on CMR.
- [S7] **Magnetic resonance imaging of the papillary muscles of the left ventricle: normal anatomy, variants, and abnormalities** — PMC6702502 (Insights Imaging 2019) — Short-axis contour methodology; separate anterolateral/posteromedial papillary muscle tracing; threshold-based automatic exclusion in pipelines.
- [S8] **Papillary and Trabecular Muscles Have Substantial Impact on Quantification of Left Ventricle in Patients with Hypertrophic Obstructive Cardiomyopathy** — PMC9407152 (2022) — HOCM n=74: PTM mass 47.9 ± 18.7 g (26.9% of LVM); mask method: EDV −45.5%, ESV −128.9%, LVEF +20.1 pp, LV mass +26.9%; gender: LVMI +30.2% women, +25.7% men.
- [S9] **Left ventricular papillary muscles and trabeculae are significant determinants of cardiac MRI volumetric measurements: Effects on clinical standards in patients with advanced systolic dysfunction** — PubMed 17698216 (2007) — Advanced systolic dysfunction: indexed LV mass +17%, indexed diastolic volume −20%, LVEF +13% by exclusion of PTM; absolute volumes EDV 149→160 ml, ESV 48→54 ml; LV mass 157±71 vs. 141±62 g.
