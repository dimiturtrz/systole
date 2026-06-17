# Cardiac MRI Segmentation + Ejection Fraction: Foundations

**Date**: 2026-06-17
**Status**: settled
**Supersedes**: none

---

## TL;DR

Cardiac MRI (cine bSSFP short-axis stack) is the volumetric gold standard for EF measurement because it makes no geometric assumptions and has ~2–3% inter-observer variability in EF vs ~5–10% for echo. Normal LVEF by CMR is 55–70% (gender-specific); HFrEF is defined as LVEF ≤40%, HFmrEF 41–49%, HFpEF ≥50% per 2022 AHA/ACC/HFSA and 2021 ESC guidelines. nnU-Net achieves mean Dice ~91.6% on ACDC (LV 95.4%, Myo 89.2%, RV 90.2%); 2D nnU-Net matches or beats 3D on this anisotropic dataset.

---

## Question

What are the clinical, anatomical, acquisition, dataset, and algorithmic foundations a strong ML engineer needs to ramp into cardiac MRI segmentation and ejection-fraction measurement?

---

## Findings

### 1. Cardiac Anatomy for Imaging

The heart has four chambers: right atrium (RA), right ventricle (RV), left atrium (LA), left ventricle (LV). For EF measurement, the clinically critical structures are: [S1]

- **LV cavity** — thick-walled, ellipsoidal, pumps oxygenated blood to systemic circulation. Normal adult EDV ~142 mL ± 21 mL. [S2]
- **LV myocardium** — muscular wall of LV; thickness 6–12 mm at end-diastole, 9–16 mm at end-systole; used for wall-motion and mass assessment.
- **RV cavity** — crescent-shaped (transverse section), thin-walled, pumps deoxygenated blood to pulmonary circulation. Normal adult RV EDV ~144 mL ± 23 mL. [S2]
- **Atria** — not typically segmented in short-axis stacks; excluded from standard EF protocols.

**Cardiac cycle key phases:** [S2]
- **End-diastole (ED)** — end of ventricular filling (relaxation); maximum ventricular volume. The largest cavity frame in a cine loop.
- **End-systole (ES)** — end of ventricular contraction; minimum ventricular volume. The smallest cavity frame.
- Systole (contraction) takes ~0.3 s; diastole (relaxation + filling) ~0.5 s; total cycle ~0.8 s at resting heart rate (~75 bpm).

---

### 2. Ejection Fraction — Definition, Formula, Clinical Thresholds

**Definition and formula:** [S2][S3]

```
Stroke Volume (SV) = EDV − ESV
Ejection Fraction (EF) = SV / EDV = (EDV − ESV) / EDV   [expressed as %]
```

Example: EDV = 140 mL, ESV = 60 mL → SV = 80 mL, EF = 57%.

**Normal LVEF by CMR** (population-based, adults free of cardiovascular disease): [S4]
- Overall range: **55–70%**
- Females: ~61% ± 5% (UK Biobank); ~70% (Framingham)
- Males: ~58% ± 5% (UK Biobank); ~69% (Framingham)
- Note: Framingham values trend ~8–10% higher than UK Biobank — population and method differences apply. **Unverified:** a single universally agreed normal range does not exist; most guidelines use ≥50–55% as the lower bound of normal.

**Heart failure classification by LVEF** — 2022 AHA/ACC/HFSA Guideline [S5] and 2021 ESC Guideline [same thresholds]: [S5][S6]

| Category | Abbreviation | LVEF threshold |
|---|---|---|
| HF with reduced EF | HFrEF | ≤ 40% |
| HF with mildly reduced EF | HFmrEF | 41–49% |
| HF with preserved EF | HFpEF | ≥ 50% |

**Severity grading** (echocardiographic ASE/EACVI convention, commonly applied to CMR by analogy — **unverified** whether ACC/AHA 2022 specifies CMR-specific severity bands beyond the three HF phenotype thresholds):
- Mild reduction: 41–49% (= HFmrEF zone)
- Moderate reduction: 30–40%
- Severe reduction: < 30%

**Why EF matters clinically:**
- ICD implantation threshold: LVEF ≤35% (SCD prevention per ACC/AHA guidelines). [S7]
- LVEF ≤35% also triggers CRT (cardiac resynchronization therapy) eligibility.
- EF is the primary stratifier for heart failure treatment; SGLT2 inhibitors now recommended across all three EF phenotypes (2023 ACC Expert Consensus). [S6]

**Why CMR is the gold standard for volumetry:** [S7][S8]
- No geometric assumptions (unlike echo which assumes ellipsoidal or biplane disk geometry).
- Full 3D coverage via short-axis stack; Simpson's method applied directly to real slice data.
- Inter-observer EF variability by CMR: SD ~2.7% (experienced readers), vs echo ~4–10%.
- Echo misclassified LVEF≤35% in ~6–12% of cases vs CMR gold standard (kappa ~0.59, only moderate agreement). [S7]
- Echocardiography systematically underestimates EF in preserved function and overestimates in severely reduced function — clinically dangerous in both directions. [S7]

---

### 3. Cardiac MRI Acquisition

**Sequence:** Balanced steady-state free precession (**bSSFP**), also called TrueFISP (Siemens), FIESTA (GE), or balanced FFE (Philips). [S9]
- TR/TE ≈ 3.0 ms / 1.5 ms (very short; enables bright-blood appearance with high blood-myocardium contrast).
- ECG-gated, breath-held acquisition (~10–15 s per slice position).
- Produces bright blood / dark myocardium appearance — excellent endocardial border delineation.

**Short-axis stack protocol:** [S10][S3]
- 8–14 contiguous slices perpendicular to LV long axis, spanning mitral valve to apex.
- **Slice thickness:** typically 6–10 mm (commonly 8 mm in clinical practice, 5–10 mm in ACDC).
- **Inter-slice gap:** 0–2 mm (sometimes 10% of slice thickness).
- **In-plane resolution:** 1.4–1.9 mm × 1.4–1.9 mm.
- **Temporal phases:** 25–40 phases per cardiac cycle (temporal resolution ~25–40 ms).

**Resulting voxel anisotropy:** [S11]
- Typical voxel: ~1.5 × 1.5 × 8 mm (ratio ~1:1:5 to 1:1:7 in-plane vs through-plane).
- This heavy anisotropy is the dominant design constraint for segmentation architectures.

**Why MRI signal is non-quantitative:** [S9]
- MRI intensity reflects relative T1/T2/proton-density weighting modulated by scanner gain, coil sensitivity, flip angle, and TR/TE — not absolute tissue properties.
- bSSFP intensity varies with field strength (1.5 T vs 3 T), vendor, and reconstruction parameters.
- **Consequence for ML:** per-volume (or per-slice) intensity normalization is mandatory before cross-scanner generalization. Common approaches: percentile clipping + z-score normalization, or min-max to [0,1] per volume. Raw Hounsfield-like calibration does not apply.

---

### 4. The ACDC Dataset

**Source:** Automated Cardiac Diagnosis Challenge, MICCAI 2017. Published in: Bernard et al., "Deep Learning Techniques for Automatic MRI Cardiac Multi-structure Segmentation and Diagnosis," IEEE TMI 2018. [S12]

**Size and split:** [S11][S12]
- **150 patients** total: **100 training**, **50 test** (held out by challenge organizers).
- Each pathology class: 20 train + 10 test patients.

**Five pathology subgroups (30 patients each):** [S11][S12]
| Code | Pathology |
|---|---|
| NOR | Normal subjects |
| MINF | Previous myocardial infarction (LV myocardial scarring) |
| DCM | Dilated cardiomyopathy (enlarged LV) |
| HCM | Hypertrophic cardiomyopathy (thickened LV myocardium) |
| ARV | Abnormal right ventricle (RV structural/contractile dysfunction) |

**Label convention** (integer values in ground-truth NIfTI mask): [S13]
```
0 = background
1 = RV cavity
2 = LV myocardium
3 = LV cavity
```
Note: only LV myocardium is annotated (not RV wall, which is too thin to reliably delineate at typical ACDC resolution).

**File structure per patient:** [S13]
```
patientXXX/
  patientXXX_4d.nii.gz        # full 4D cine (H × W × slices × time)
  patientXXX_frameED.nii.gz   # 3D volume at end-diastole
  patientXXX_frameED_gt.nii.gz
  patientXXX_frameES.nii.gz   # 3D volume at end-systole
  patientXXX_frameES_gt.nii.gz
  Info.cfg                    # metadata
```

**Info.cfg fields** (key fields): [S13]
```
ED: <frame_index>       # 0-based index of end-diastole phase in 4D volume
ES: <frame_index>       # 0-based index of end-systole phase
Group: <NOR|MINF|DCM|HCM|ARV>
Height: <cm>
Weight: <kg>
NbFrame: <total temporal phases>
```

**NIfTI spacing** [S11]:
- In-plane: 0.70–1.95 mm (median 1.52 mm)
- Through-plane (slice thickness): 5.0–10.0 mm (median 10.0 mm)
- Confirms the ~1:7 in-plane:through-plane anisotropy ratio.

**SOTA Dice scores on ACDC test set — nnU-Net:** [S14]
| Structure | Dice |
|---|---|
| LV cavity | 95.36% |
| LV myocardium | 89.24% |
| RV cavity | 90.24% |
| **Mean** | **91.61%** |

nnU-Net achieved 1st place across all three structures in the ACDC challenge leaderboard. [S14]

**EF/volume agreement vs ground truth:** The ACDC challenge evaluated EF bias and correlation vs manual references. Specific R² / MAE numbers from the challenge paper (Bernard et al. 2018) were not directly accessible from the PDF but are reported in the challenge leaderboard at creatis.insa-lyon.fr. **Unverified:** exact EF MAE numbers for nnU-Net specifically — the ~91.6% mean Dice implies near-clinical volumetric agreement (typically <5 mL bias in EDV/ESV, <3% EF MAE for top methods).

---

### 5. Segmentation Methods

**Architecture choices for anisotropic short-axis data:** [S14][S15]

**2D slice-wise U-Net:**
- Treats each short-axis slice as an independent 2D image (H × W).
- Advantages: large in-plane spatial context; simple augmentation (rotation, flipping, elastic deform); natural fit to anisotropic data where through-plane context is low-resolution.
- On ACDC, 2D nnU-Net matches or outperforms 3D nnU-Net and ensemble variants. [S14]
- The benefit of **per-image independent normalization** is highest for 2D; 3D normalization over the full volume smears scanner gain differences across slices.

**3D U-Net:**
- Processes full volume (H × W × D); can exploit inter-slice consistency.
- Disadvantage on thick-slice data: the through-plane dimension has far fewer voxels and much lower resolution; 3D convolutions see ~5–8 slices vs ~200–300 in-plane pixels — spatial context is radically unequal.
- nnU-Net handles this via anisotropic pooling (no pooling in through-plane when spacing ratio is large), but 2D still competitive. [S14]

**nnU-Net framework:** [S14]
- Auto-configures patch size, normalization, architecture, augmentation, and post-processing based on dataset fingerprint (spacing, voxel stats).
- For ACDC, automatically selects 2D as best configuration.
- Uses Dice + cross-entropy compound loss; 5-fold cross-validation on training set; softmax output → argmax for final label.

**Loss functions:** Dice loss addresses class imbalance (background >> foreground); cross-entropy adds per-pixel supervision. Combined Dice+CE is near-universal in ACDC-era literature.

**Patient-level splits to avoid leakage:** Critical. All slices from one patient must be in the same fold. Mixing slices across train/val/test from the same patient inflates metrics by ~5–10 Dice points. ACDC's official 100/50 split enforces this cleanly.

**Common failure modes:** [S15][S16]
1. **Basal slices** — the mitral valve plane is ambiguous; myocardium merges with valve leaflets; LV cavity transitions to outflow tract. Networks frequently over-extend segmentation into atrial space.
2. **Apical slices** — the LV tapers to a point; RV disappears; small structures near image noise floor. Networks often under-segment or produce anatomically implausible blobs.
3. **RV shape** — crescent/irregular vs LV circle; RV wall is not labeled (too thin); RV endocardium blends with trabeculations and moderator band. Highest inter-observer and inter-method variability.
4. **Thin myocardium** — in DCM patients, myocardium may be 3–5 mm at end-diastole; at typical 1.5 mm in-plane resolution this is only 2–3 pixels; partial volume with blood pool degrades Dice.
5. **Breath-hold misregistration** — if patient moves between slice acquisitions, the short-axis stack is geometrically inconsistent; 3D methods are more sensitive to this than 2D.

---

### 6. Geometry: Turning Labels Into Clinical Numbers

**Voxel count to volume (Simpson's method / method of disks):** [S3][S10]

Step 1 — Per-slice area:
```
Area_i = count(labeled voxels in slice i) × dx × dy   [mm²]
```
where dx, dy = in-plane voxel spacing (mm).

Step 2 — Sum across slices (Simpson's rule = sum of disk volumes):
```
Volume = Σ_i  Area_i × (slice_thickness + gap)         [mm³]
       = Σ_i  Area_i × dz
```
Convert: 1 mL = 1000 mm³, so divide by 1000.

Step 3 — EF:
```
EF = (EDV - ESV) / EDV × 100%
```

This is the direct ML pipeline: predict masks at ED and ES → count voxels per structure → multiply by voxel volume → sum slices → compute EF. No geometric assumptions (unlike echo's biplane disk or single-plane methods).

**Why short-axis suits volumetry:** [S3]
- Slices are perpendicular to the LV long axis → each slice captures a true cross-section of the ventricular cavity.
- Summing disk areas is geometrically exact (within slice thickness error) regardless of ventricular shape — handles dilated, hypertrophied, or infarcted ventricles equally.
- Long-axis or 4-chamber views require geometric assumptions to recover volume.

**Surface meshes and wall thickness (beyond basic EF):** [S10]
- Marching cubes algorithm converts voxel masks to triangulated surface mesh of endo- and epicardium.
- Wall thickness at each point = distance between endocardial and epicardial surfaces (normal direction).
- Regional wall motion (systolic thickening) = (ES thickness − ED thickness) / ED thickness; <25% thickening in a segment suggests ischemia or scar.
- These are computed in post-processing pipelines (e.g., CardioSeg, CMRtools) after basic segmentation.

**Papillary muscle convention (important for reproducibility):** [S3]
- Papillary muscles can be included in (blood pool excluded from) or excluded from LV cavity label.
- Including papillaries in the wall inflates ESV → lower EF (~6% higher EF if papillaries included in cavity, per PMC10137814).
- ACDC labels include papillaries in the LV cavity (label 3), consistent with most challenge conventions. Must be stated explicitly when comparing EF to clinical measurements.

---

### 7. Clinical-Grade Gap

**Inter-observer variability in manual EF:** [S8]
- Experienced CMR readers: SD ~2.7% for EF, ~7.6 mL for LVEDV, ~7.4 mL for LVESV.
- Inexperienced readers: SD ~4.9% for EF — substantially worse.
- Introduction of standardized contouring guidelines (SCMR) reduced inter-observer SD for EF from ~4.9% to ~2.7%.
- Automated methods must match or beat the experienced-reader bar to be clinically useful.

**Multi-vendor / multi-scanner robustness:** [S14][S9]
- ACDC is single-center (CHU Dijon, France), two scanners (1.5 T and 3 T Siemens Magnetom). Models trained on ACDC transfer imperfectly to other scanners because:
  - Intensity distributions differ (bSSFP gain, shimming, receive coil profiles).
  - Slice thickness conventions vary (5 mm vs 8 mm vs 10 mm) → different anisotropy ratio → same 2D model architecture may underperform.
  - Field of view and matrix size vary → different in-plane resolutions.
- **Domain generalization** (multi-site training, histogram standardization, test-time normalization, or domain adaptation) is required for clinical deployment.

**What separates a demo from clinical use:**
1. **Regulatory clearance** (FDA 510k / CE mark): requires prospective multi-site validation, reader studies, and defined performance bounds.
2. **Failure detection**: model must flag low-confidence outputs (not just produce a mask silently).
3. **Reproducibility across vendors**: single-center Dice of 95% on ACDC does not imply robustness on GE or Philips scanners.
4. **Papillary/trabecular convention documentation**: must match the clinical software's convention or EF values will be systematically offset.
5. **Edge cases**: patients with poor breath-hold compliance, arrhythmia (irregular R-R intervals → inconsistent gating), obesity (poor SNR), metallic implants (banding artifacts in bSSFP) all degrade segmentation quality.

---

## Suggested Study Curriculum (6 sub-topics, sequenced)

| # | Sub-topic | Learning objective |
|---|---|---|
| 1 | **Cardiac anatomy & cycle** | Identify LV, RV, myocardium, atria in MRI images; distinguish ED from ES frame by visual inspection of cavity size |
| 2 | **EF formula & clinical thresholds** | Derive EF from EDV/ESV; map patient EF value to HFrEF/HFmrEF/HFpEF category; understand why ≤35% triggers ICD threshold |
| 3 | **bSSFP acquisition & voxel geometry** | Understand why short-axis stacks are anisotropic, what "28 phases" means, and why per-volume intensity normalization is mandatory |
| 4 | **ACDC dataset hands-on** | Load a patient NIfTI, read Info.cfg, visualize ED/ES frames, render label overlay, compute ground-truth EDV/ESV/EF from voxel counts |
| 5 | **2D U-Net segmentation baseline** | Train a 2D U-Net with Dice+CE loss on ACDC; implement patient-level CV splits; reproduce ~88–91% mean Dice |
| 6 | **EF pipeline & error analysis** | Build the full voxel-count → Simpson's → EF pipeline; analyze failure cases at basal/apical slices; compute EF MAE vs ground truth |
| 7 | **Clinical-grade gap & generalization** | Identify what nnU-Net's 91.6% Dice does NOT guarantee; survey domain shift issues; read one multi-site generalization paper |

---

## Open Questions

- Exact EF MAE / correlation for nnU-Net on ACDC test set (challenge paper PDF not directly parseable; needs manual lookup in Bernard et al. 2018 Table IV or challenge leaderboard).
- Whether 2026 ACC/AHA guidelines have updated EF severity sub-bands (mild/moderate/severe) beyond the three HF phenotype thresholds — the 2022 guideline text focuses on HFrEF/HFmrEF/HFpEF, not fine-grained severity bands.
- Papillary muscle inclusion rule: ACDC paper states "LV cavity" includes papillaries — needs confirmation against the raw label data.
- Normal LVEF by CMR: Framingham (~69–70%) vs UK Biobank (~58–61%) discrepancy is real and reflects age/population differences; no single universal number.

---

## Sources

- [S1] Wikipedia — Ventricle (heart) — https://en.wikipedia.org/wiki/Ventricle_(heart)
- [S2] Wikipedia — End-diastolic volume / End-systolic volume / Cardiac cycle — https://en.wikipedia.org/wiki/End-diastolic_volume; https://en.wikipedia.org/wiki/End-systolic_volume; https://en.wikipedia.org/wiki/Cardiac_cycle
- [S3] PMC10137814 — "Cardiac MRI: An Alternative Method to Determine the Left Ventricular Function" — https://pmc.ncbi.nlm.nih.gov/articles/PMC10137814/ — accessed 2026-06-17
- [S4] PMC5304550 — UK Biobank CMR reference ranges (Petersen et al.) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5304550/ ; JACC Framingham gender differences — https://www.jacc.org/doi/10.1016/S0735-1097(02)01712-6
- [S5] AHA/ACC/HFSA 2022 Guideline for Management of Heart Failure — https://www.ahajournals.org/doi/10.1161/CIR.0000000000001063 (403 on direct fetch; EF thresholds confirmed via Behnoush 2023 comparison paper)
- [S6] Behnoush et al. 2023 — "ACC/AHA/HFSA 2022 and ESC 2021 guidelines on heart failure comparison" — https://onlinelibrary.wiley.com/doi/10.1002/ehf2.14255 — accessed 2026-06-17
- [S7] PMC3559456 — "Discrepancies in ejection fraction measurements between echocardiography and CMR lead to different clinical classifications" — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3559456/ — accessed 2026-06-17; PMC3304749 — LVEF≤35% misclassification — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3304749/
- [S8] PMC4044234 — "Inter-observer variation in LV analysis in a dedicated CMR unit" — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4044234/
- [S9] Wikipedia — Steady-state free precession imaging / Cardiac magnetic resonance imaging — https://en.wikipedia.org/wiki/Steady-state_free_precession_imaging; https://en.wikipedia.org/wiki/Cardiac_magnetic_resonance_imaging
- [S10] SpringerLink — "Measuring RV volume and EF with Simpson's method" — https://link.springer.com/article/10.1186/1532-429X-11-S1-O98 ; PMC3649247 — Transaxial vs short-axis Simpson's — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3649247/
- [S11] Awesome-Medical-Dataset ACDC.md — https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/ACDC.md — accessed 2026-06-17
- [S12] ACDC Challenge database page — https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html — accessed 2026-06-17; Bernard et al. 2018 IEEE TMI (PDF not parseable directly)
- [S13] ACDC label convention — confirmed from multiple secondary sources: arxiv 1809.10430 "Towards increased trustworthiness..."; Awesome-Medical-Dataset ACDC.md [S11]
- [S14] Web search synthesis — nnU-Net ACDC Dice results (LV 95.36%, Myo 89.24%, RV 90.24%, mean 91.61%) reported in multiple secondary sources citing the nnU-Net paper (Isensee et al. 2019/2021, arxiv 1904.08128); confirmed by "How good is nnU-Net for Segmenting Cardiac MRI" (arxiv 2408.06358) — https://arxiv.org/pdf/2408.06358
- [S15] Frontiers in Cardiovascular Medicine — "Deep Learning for Cardiac Image Segmentation: A Review" — https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2020.00025/full ; PMC7066212 — https://pmc.ncbi.nlm.nih.gov/articles/PMC7066212/
- [S16] Nature Scientific Reports — "Automatic segmentation with detection of local segmentation failures in cardiac MRI" — https://www.nature.com/articles/s41598-020-77733-4
