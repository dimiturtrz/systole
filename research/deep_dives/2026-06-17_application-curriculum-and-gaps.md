# Application-Level MRI/Cardiac Curriculum: Topics, Gaps, and Technical Reference

**Date**: 2026-06-17
**Status**: partial
**Supersedes**: none

## TL;DR

An ML engineer with MRI physics foundations needs ~8 additional topic clusters to do cardiac segmentation → EF → evaluation rigorously. The most load-bearing gaps are: bSSFP cine sequence mechanics + ECG gating, the artifact zoo (especially bSSFP dark banding and partial volume at apices), Simpson's method + papillary inclusion convention, and why Dice alone misleads on clinical EF accuracy. ACDC label encoding (0=bg, 1=RV cavity, 2=LV myocardium, 3=LV cavity) is the community convention but is NOT explicitly numeric-confirmed in the official docs — inferred from standard usage across published papers.

---

## 1. Completeness Checklist from Reference Curricula

Sources: RAD229 syllabus [S1], IMAIOS e-MRI 16-chapter index [S2].

### RAD229 Topics (labeled by module)

| Module | Topic | Tag |
|--------|-------|-----|
| 1A–1D | B0 field, B1 pulses, Bloch equations, T1/T2 relaxation | **already known** |
| 2A–2D | Gradients, slice selection, k-space, phase/frequency encoding | **already known** |
| 3A–4C | Matrix/EPG simulation tools | out-of-scope |
| 5A–5C | Gradient non-linearity, eddy currents, Maxwell fields | useful (know they exist) |
| 6A–6C | Motion artifacts, motion compensation, motion encoding | **essential** |
| 7A–7C | SNR, noise, multi-channel coils | useful |
| 8A–8B | Spin-echo signals | out-of-scope |
| 9A–9D | Balanced-SSFP (bSSFP) and spoiled sequences | **essential** |
| 10A–10C | Echo Planar Imaging | out-of-scope |
| 11A–11B | Radial/Spiral sequences | out-of-scope |
| 12A–12C | Gradient waveform design | out-of-scope |
| 13A–13B | Sampling and timing | useful |
| 14A–15C | Magnetization preparation, diffusion | out-of-scope |

### IMAIOS e-MRI Chapters

| Ch | Topic | Tag |
|----|-------|-----|
| 1 | NMR basics | already known |
| 2 | Instrumentation & safety | useful (coil types for SNR) |
| 3 | NMR signal & contrast | already known |
| 4 | Spatial encoding | already known |
| 5 | Image formation / k-space | already known |
| 6 | Sequences (SE, GRE, EPI, FSE) | useful (bSSFP is GRE family) |
| 7 | Water/fat imaging (STIR, fat-sat) | useful (affects bSSFP artifacts) |
| 8 | Contrast agents | out-of-scope |
| 9 | Magnetization transfer | out-of-scope |
| 10 | Image quality & artifacts | **essential** |
| 11 | Parallel MR imaging (GRAPPA, SENSE) | useful |
| 12 | MR angiography & flow | useful (flow artifacts) |
| 13 | Cardiac MRI | **essential** |
| 14 | Cerebral perfusion | out-of-scope |
| 15 | Diffusion / DTI | out-of-scope |
| 16 | fMRI, MRS, ultra-high field | out-of-scope |

**Summary**: User has covered ~Ch 1–5 of IMAIOS and RAD229 modules 1A–2D. Remaining essential material: RAD229 6A–6C, 9A–9D; IMAIOS Ch 10, 13.

---

## 2. Cardiac MRI — bSSFP Cine, ECG Gating, Acquisition

Source: IMAIOS Ch 13 overview [S2], bSSFP literature [S3], ACDC dataset paper (Bernard et al. 2018) [S4], bSSFP acquisition parameter references [S5].

### bSSFP Cine Sequence

- **What**: Balanced Steady-State Free Precession (also called TrueFISP/FIESTA/b-FFE by vendor). A gradient-echo variant where gradient moments across TR are zeroed, yielding signal proportional to T2/T1 rather than pure T2* [S3].
- **Why cardiac**: High blood-myocardium contrast (bright blood), fast readout per TR (~3–5 ms), enabling temporal resolution ~30–50 ms per cardiac phase [S5].
- **Typical parameters** (1.5T/3T clinical standard):
  - TR/TE: ~4/2 ms
  - In-plane resolution: 1.25–1.68 mm²/pixel [S4, S5]
  - Slice thickness: 5–8 mm (ACDC: "5 mm or sometimes 8 mm" with 5 mm gaps) [S4]
  - Temporal resolution: 28–40 frames cover full cardiac cycle; ~30–45 ms/frame [S4, S5]
  - FOV: ~320–340 mm
  - Acceleration: GRAPPA ×2 typical
- **Anisotropy**: In-plane ~1.5 mm, through-plane ~8 mm → ratio ~1:5 to 1:7. This makes 3D models suboptimal; 2D slice-by-slice models dominate. nnU-Net auto-detects this and defaults to 2D or pseudo-2D [S6].

### ECG Gating

- **Prospective gating**: Acquisition triggered by R-wave. Collects data during a defined window after the R-wave. Misses end-diastole (last frames before next R-wave) — typically 10–30 ms gap. Safer for arrhythmia patients [S2].
- **Retrospective gating**: Acquires continuously while recording ECG. After acquisition, k-space lines are sorted back to cardiac phases by timing relative to R-wave. Covers full cardiac cycle including end-diastole; better for EF computation. Standard for cine. [S2, S3].
- **Segmented k-space**: Each cardiac phase is filled across multiple heartbeats. Typically 10–16 phase-encode lines acquired per heartbeat ("views per segment"). Full k-space for one slice requires ~10–16 heartbeats → ~10–16 s breath-hold per slice [S5].

### Short-Axis Stack Prescription

- Long-axis (4-chamber, 2-chamber) views used to define the mitral valve plane and apex.
- Short-axis stack prescribed perpendicular to the LV long axis, from base (mitral valve) to apex.
- Typically 10–15 slices for whole-heart coverage [S5].
- Each slice acquired in a separate breath-hold (~10–15 s); total exam ~15–20 min for cine stack.

### ED / ES Frame Identification

- **End-diastole (ED)**: Frame with largest LV cavity; typically the first frame post-R-wave in retrospective gating.
- **End-systole (ES)**: Frame with smallest LV cavity; occurs ~300–400 ms after R-wave at rest.
- In ACDC: both ED and ES frames are annotated with ground-truth segmentations [S4].

---

## 3. MRI Artifacts Affecting Cardiac Segmentation / EF

Sources: IMAIOS Ch 10 [S2], RadioGraphics cardiac artifacts [S7], bSSFP off-resonance literature [S3, S8].

### 3a. Motion / Ghosting — HIGH IMPACT on EF

- **What**: Phase-encode ghosting from periodic motion (cardiac, respiratory). Appears as repeated ghost copies of the heart displaced in the phase-encode direction [S2].
- **EF effect**: Blurs myocardial borders → boundary detection errors. Can be mistaken for pericardial effusion or thin myocardium. Respiratory ghosts can overlap LV cavity.
- **Mitigation**: ECG gating (cardiac), breath-hold (respiratory), navigator echoes.
- **Severity**: High if breath-hold not maintained; moderate with standard protocol.

### 3b. bSSFP Off-Resonance Dark Banding — MODERATE–HIGH IMPACT

- **What**: bSSFP has periodic signal nulls at off-resonance frequencies spaced at 1/TR Hz (~250 Hz at TR=4 ms). Local B0 inhomogeneity shifts spins into nulls → dark bands crossing the image [S3, S8].
- **Where**: Near diaphragm-lung interface, near pericardial fat, at 3T (worse than 1.5T). Bands can cross myocardium directly.
- **EF effect**: Dark band crossing myocardium mimics wall thinning or creates false low signal → segmentation model may misclassify myocardium as background → underestimates myocardial mass; if band crosses LV cavity, cavity boundary is misplaced → EF error.
- **Mitigation**: Shimming, frequency scouting, phase cycling (acquire at multiple frequency offsets and combine) [S8].

### 3c. Partial Volume Effect (PVE) — HIGH IMPACT at APICES

- **What**: At apical slices, the LV tapers. A single 8 mm thick slice may contain both myocardium and surrounding tissue within the same voxel → averaged signal, blurred border [S2, literature].
- **EF effect**: Apical volume is systematically mis-measured. Because apical slices contribute disproportionately to EDV/ESV difference, PVE at apex can bias both EDV and ESV similarly (partially cancelling in EF) or create net EF overestimation if ES apical cavity is falsely enlarged. Thin right ventricular myocardium (3–5 mm) is almost entirely subject to PVE at standard resolutions [S9].
- **Mitigation**: Thinner slices (5 mm preferred over 8 mm). In ACDC, 5 mm is stated as primary, 8 mm as occasional [S4].

### 3d. Gibbs / Truncation Ringing — LOW–MODERATE IMPACT

- **What**: Finite k-space truncation produces oscillatory ringing at sharp interfaces (blood-myocardium boundary). Manifests as parallel bright/dark bands near borders [S2].
- **EF effect**: Can artificially thicken or thin apparent myocardial wall; introduces ~1–2 pixel uncertainty at boundary. Low-frequency ringing can shift apparent cavity area slightly.
- **Mitigation**: Apodization (k-space windowing) at cost of blurring. Usually minor vs. motion artifacts.

### 3e. Bias Field / Intensity Inhomogeneity (B1) — LOW IMPACT on EF

- **What**: Receive coil sensitivity variation creates smooth signal intensity gradient across image — center brighter than periphery (surface coil effect) [S2].
- **EF effect**: Intensity-based thresholding methods fail. Deep learning models trained on normalized data are relatively robust if normalization is consistent. Does not affect geometry directly. Can affect model generalization across different coil configurations.
- **Mitigation**: N4 bias field correction (standard preprocessing). nnU-Net applies z-score normalization per case [S6].

### 3f. Aliasing / Wrap-around — CONTEXT-DEPENDENT IMPACT

- **What**: FOV smaller than object in phase-encode direction wraps the overhanging anatomy onto the opposite side of the image [S2].
- **EF effect**: Chest wall or liver can fold onto cardiac region. Rare with adequate FOV planning, but seen in large patients or suboptimal protocol.
- **Mitigation**: Increase FOV, use saturation bands, parallel imaging oversampling.

### 3g. Flow Artifacts — LOW–MODERATE IMPACT

- **What**: Moving blood can cause signal variations in bSSFP due to inflow and through-plane flow effects. At high velocities (aortic valve, mitral valve), signal voids appear [S2, S3].
- **EF effect**: Signal void near valves can be misinterpreted as myocardium → base-of-heart segmentation errors. Particularly affects RV outflow tract and LV outflow tract at ES.
- **Mitigation**: bSSFP is relatively flow-robust vs. spin-echo, but not immune. Temporal averaging is not applicable (each frame is a snapshot).

### Priority ranking for EF accuracy:
1. Motion/ghosting (ruins entire frame if severe)
2. bSSFP dark banding (spatially focal but can bisect key structures)
3. Partial volume effect at apices (systematic, hard to fix without thinner slices)
4. Flow artifacts at base (basal slice plane prescription errors)
5. Gibbs ringing, bias field (minor for DL models)

---

## 4. Cardiac Anatomy & Cycle

Sources: standard anatomy; ACDC challenge [S4]; IMAIOS e-Anatomy [S10].

### Chambers & Structures

- **Right ventricle (RV)**: Thin-walled (~3–5 mm), crescent-shaped in short axis, anterior. Pumps to pulmonary circulation (low pressure). ACDC label: **1**.
- **LV myocardium**: Thick muscle wall (~8–12 mm at ED, ~12–16 mm at ES), surrounds LV cavity. ACDC label: **2**.
- **Left ventricle (LV) cavity**: Blood pool, roughly circular in short axis. Pumps to systemic circulation (high pressure). ACDC label: **3**.
- **Background**: Everything else (RV myocardium not separately labeled in ACDC, pericardium, lungs, liver). ACDC label: **0**.

**UNCERTAINTY**: The numeric encoding 0=bg, 1=RV, 2=myo, 3=LV is the universal community convention in published ACDC papers and code repositories, but the official CREATIS page and Bernard 2018 PDF did not yield the explicit integer table in these fetches. This encoding is cited from published code and papers referencing ACDC [S4, S11] and should be verified against the NIfTI mask files directly.

### Cardiac Cycle

- **Systole**: Ventricular contraction. Begins at QRS complex (R-wave), ends ~300–500 ms later. LV volume decreases from EDV to ESV.
- **Diastole**: Ventricular filling. LV volume increases from ESV back to EDV.
- **End-diastole (ED)**: Maximum volume, maximum myocardial wall area in short axis; typically first frame after R-wave.
- **End-systole (ES)**: Minimum volume, maximum wall thickening; thickest myocardium, smallest cavity.
- **Papillary muscles**: Arise from LV endocardium, protrude into LV cavity. By convention in ACDC and CMR guidelines, papillary muscles are included in the LV cavity (label 3), not in myocardium. This matters: papillary + trabecular volume = ~12% EDV [S12]. Exclusion inflates LV volume by ~12% and raises EF.

---

## 5. Cardiac Function Metrics

Sources: AHA EF guidelines [S13], ScienceDirect overview [S14].

### Core EF Formula

```
EF (%) = (EDV - ESV) / EDV × 100
       = SV / EDV × 100
```

Where:
- **EDV** = End-Diastolic Volume (mL)
- **ESV** = End-Systolic Volume (mL)
- **SV** = Stroke Volume = EDV − ESV (mL)

### Normal Ranges (LV)

| Metric | Normal | Mildly reduced | Reduced (HF) |
|--------|--------|----------------|--------------|
| LVEF | 52–72% (M), 54–74% (F) | 41–49% | <40% |
| EDV (indexed) | ~70–155 mL/m² | — | varies |
| ESV (indexed) | ~20–60 mL/m² | — | varies |

- **HFrEF** (Heart Failure with reduced EF): EF < 40%
- **HFmrEF** (mildly reduced): EF 41–49%
- **HFpEF** (preserved EF): EF ≥ 50% but with diastolic dysfunction [S13]

### Cardiac Output

CO (L/min) = SV × Heart Rate. Normal: 4–8 L/min. Not directly measurable from a single ED/ES frame pair; requires HR from ECG.

### LV Mass

LV mass = myocardial volume × 1.05 g/mL (myocardial density). Myocardial volume = (epicardial volume − endocardial volume) summed across slices. Requires epicardial contour (not provided in ACDC ground truth — ACDC labels myocardium directly as a region, so epicardial boundary is the outer boundary of label 2).

Normal LV mass indexed: 49–115 g/m² (M), 43–95 g/m² (F).

### Strain (awareness only)

Myocardial strain measures regional deformation (%) using feature tracking or tagging sequences. Circumferential strain (Ecc, normal ~−20%) and longitudinal strain (GLS, normal ~−20%) are important clinical markers. Not computed from standard cine segmentation — requires dedicated sequences or post-processing. Out of scope for ACDC EF task but worth knowing exists.

---

## 6. Geometry / Volumetry

Sources: Simpson's method literature [S12, S15], nnU-Net [S6].

### Voxel Count → Volume

```
Volume (mL) = N_voxels × dx × dy × dz / 1000
```

where dx, dy = in-plane resolution (mm), dz = slice thickness (mm), divide by 1000 to convert mm³ → mL.

For ACDC typical voxels: 1.5 × 1.5 × 8 mm = 18 mm³/voxel = 0.018 mL/voxel.

### Simpson's Method (Summation of Disks)

The standard CMR volumetry method:

```
V = Σᵢ Aᵢ × dz
```

Where Aᵢ = cross-sectional cavity area at slice i (mm²), dz = slice thickness (mm). Summed over all short-axis slices containing the structure. This is exactly what voxel counting implements when each slice contributes area = N_pixels_in_slice × dx × dy [S15].

- Requires consistent basal plane definition (include/exclude basal slice when < 50% of area is myocardium is a common convention).
- Apical cap: the apex beyond the most apical slice is approximated as a cone or ignored; this introduces systematic underestimation of volume (~5–10 mL).

### Marching Cubes / Surface Meshing

For 3D surface reconstruction and wall thickness measurement. Not required for EF from short-axis stack — Simpson's method is standard. Marching cubes is used if 3D visualization or surface-based metrics (Hausdorff distance) are needed.

### Wall Thickness

Measured as shortest distance from endocardial to epicardial surface at each point. Normal LV wall: 6–12 mm at ED. In HCM: ≥ 15 mm (≥ 30 mm in severe cases). Requires both LV cavity and LV myocardium labels.

### Papillary Muscle Convention & EF Effect

Per ACDC and CMR community standards: papillary muscles are included in LV cavity (label 3), not myocardium. This follows the standard of including trabeculations in the cavity.
- Excluding papillary muscles (treating them as myocardium) → smaller measured LV cavity → lower EDV, lower ESV → EF may shift by 2–5% absolute [S12].
- For reproducibility: be consistent across ED and ES, and match training-set convention.

---

## 7. Segmentation Theory (Medical)

Sources: U-Net (Ronneberger 2015) [S16], nnU-Net (Isensee, Nature Methods 2021) [S6], segmentation survey literature.

### U-Net

- Encoder-decoder with skip connections. Encoder: successive conv+pool stages, builds feature hierarchy. Decoder: upsampling + concatenation of matching encoder feature maps. Skip connections preserve spatial detail lost in pooling [S16].
- Trained with pixel-wise cross-entropy or Dice loss. Works well with small datasets (original paper: 30 training images).
- Standard input: 2D slices or 3D patches.

### nnU-Net (No-New-Net)

- Self-configuring framework: reads dataset fingerprint (spacing, intensity distribution, image size) and automatically sets: patch size, batch size, network depth, normalization, augmentation, loss function, postprocessing [S6].
- On 23 medical segmentation challenges: matched or exceeded specialized state-of-the-art without any manual architecture changes.
- **On ACDC specifically**: nnU-Net defaults to 2D configuration because of high anisotropy (8 mm slice / 1.5 mm in-plane ≈ 1:5 ratio). nnU-Net targets in-plane resampling to ~1.56 × 1.56 mm [S6, S17].
- **2D vs 3D on anisotropic data**: 3D convolutions across 8 mm slices vs. 1.5 mm in-plane treat through-plane and in-plane equally → suboptimal. 2D models process each slice independently and outperform naive 3D on highly anisotropic cardiac data [S17]. 2.5D (multi-slice input) is a compromise.

### Loss Functions

- **Dice loss**: 1 − 2|P∩G|/(|P|+|G|). Handles class imbalance by normalizing to foreground. Differentiable approximation used during training.
- **Cross-entropy**: Pixel-wise log-likelihood. Sensitive to class imbalance (background >> foreground in cardiac MRI).
- **Compound loss (Dice + CE)**: Standard in nnU-Net and most modern cardiac segmentation. Gets stability of CE + class-balance of Dice.
- **Boundary/Hausdorff-inspired losses**: Explicitly penalize surface distance; less common but improves HD metrics.

### Augmentation (nnU-Net defaults)

Random rotation, scaling, elastic deformation, Gaussian noise, Gaussian blur, brightness/contrast adjustment, gamma correction, mirroring. Applied online during training.

### Class Imbalance

Background dominates the image. Pure CE loss → model biased toward background. Dice loss or compound loss mitigates this. Per-class weighting also used.

### Train/Val/Test Split — Critical

- **Patient-level split, not slice-level**. All slices from one patient must be in the same fold. Slice-level split → data leakage (model sees nearly identical slices from same patient in both train and val) → inflated metrics.
- ACDC standard: 100 training patients (70/10/20 or cross-validation across 100), 50 held-out test patients submitted to server.
- 5-fold cross-validation at patient level is the nnU-Net default.

---

## 8. Evaluation Theory (Medical)

Sources: segmentation metrics literature [S18, S19], Bland-Altman [S20], ACDC challenge results [S4].

### Overlap Metrics

**Dice Similarity Coefficient (DSC)**:
```
DSC = 2|P∩G| / (|P| + |G|)
```
Range [0,1]. 1 = perfect. For cardiac structures, top methods achieve DSC ~0.93–0.96 (LV cavity), ~0.88–0.92 (LV myocardium), ~0.88–0.92 (RV cavity) on ACDC [S4].

**Jaccard / IoU**:
```
IoU = |P∩G| / |P∪G| = DSC / (2 - DSC)
```
Monotone with Dice; less commonly reported in cardiac literature.

### Surface / Boundary Metrics

**Hausdorff Distance (HD)**:
```
HD(P,G) = max(max_{p∈∂P} min_{g∈∂G} d(p,g), max_{g∈∂G} min_{p∈∂P} d(p,g))
```
Worst-case surface error. Highly sensitive to outliers (single misplaced voxel can dominate).

**HD95**: 95th percentile of the directed Hausdorff distances. Removes top-5% outliers. Standard in medical segmentation. Reported in mm. For cardiac structures, good methods: HD95 ~2–5 mm [S18].

**ASSD (Average Symmetric Surface Distance)**:
```
ASSD = (Σ_{p∈∂P} min_{g∈∂G} d(p,g) + Σ_{g∈∂G} min_{p∈∂P} d(p,g)) / (|∂P| + |∂G|)
```
Mean surface error in mm. More sensitive to systematic boundary offsets than HD95. Also called MASD.

### Volume & EF Agreement

**Why Dice alone misleads for clinical EF**:
1. Dice is scale-normalized: a model that gets LV cavity Dice = 0.95 on large healthy hearts may still have 5–10 mL volume error, which corresponds to 3–5% EF error for small hearts (HCM, HFrEF with low EDV).
2. Dice does not decompose by cardiac phase: a model can have high mean Dice but systematically underestimate ES cavity → biased EF estimate even when ED is perfect.
3. Dice does not penalize smooth systematic boundary offsets: a contour shifted by 1 mm inward throughout can yield DSC > 0.9 but biases volume by ~5–10 mL [S19].

**Bland-Altman Analysis** (method agreement):
- Plot: y-axis = (Method A − Method B) for each patient; x-axis = mean of A and B.
- Reports: **bias** (mean difference), **limits of agreement** (bias ± 1.96 SD).
- Clinical threshold for LV EF: bias < 2–3%, limits of agreement < ±5–8% EF for clinical equivalence [S20].
- Also apply to EDV, ESV, SV, LV mass separately.

**MAE (Mean Absolute Error)**: simpler scalar. Report in mL for volumes, % for EF.

**Pearson correlation**: often reported but misleading — high correlation with large bias is common. Bland-Altman is preferred.

### Calibration / Uncertainty

For DL segmentation: MC dropout or ensemble spread gives per-pixel uncertainty. Useful for flagging cases where prediction is unreliable (basis for QC in clinical deployment). Not required for ACDC benchmark but good practice.

### Failure Analysis

- Always report worst-case (lowest Dice per case) and examine which pathology group fails.
- ACDC pathology split (NOR/MINF/DCM/HCM/ARV × 30 cases each) enables per-class breakdown.
- HCM: thick wall → RV/LV boundary harder. ARV: thin RV wall, high PVE. MINF: scar tissue alters texture.
- Report: failure histogram, worst-5% cases, qualitative inspection.

---

## 9. Clinical-Grade Gap

Sources: domain shift literature [S21, S22].

### Multi-Vendor / Multi-Scanner Domain Shift

- ACDC uses 2 field strengths (1.5T, 3T) from one institution (CHU Dijon, France). A model trained on ACDC may fail on:
  - Different scanners (Siemens / Philips / GE have different noise characteristics, k-space trajectories, reconstruction filters)
  - Different protocols (different slice thickness, TR, TE)
  - Different patient populations (obesity, pediatric, ethnic variation in cardiac size)
- Empirically: Dice drops 5–15% absolute when applying ACDC-trained models to out-of-distribution scanners [S21].
- **M&M Challenge** (Multi-Centre Multi-Vendor) is the standard benchmark for generalization.
- Mitigations: domain adaptation (adversarial training, instance normalization), data augmentation simulating scanner variability, test-time adaptation.

### Why a Demo ≠ Clinical Tool

1. **Regulatory**: CE marking (EU) or FDA 510(k)/De Novo clearance required. Requires prospective clinical validation studies, not retrospective benchmarks.
2. **Ground truth quality**: ACDC annotations are single-expert. Clinical tools require inter-rater variability assessment and expert consensus.
3. **Edge cases**: pacemakers, congenital heart disease, post-surgical anatomy, poor image quality — all underrepresented in ACDC.
4. **EF threshold decisions**: a model with mean EF MAE = 5% may misclassify HFrEF (<40%) as HFmrEF (41–49%) in individual patients, with treatment implications.
5. **Audit trail**: clinical tools need explainability, failure modes documentation, and continuous monitoring.

---

## Open Questions

- Exact integer label encoding (0/1/2/3) for ACDC NIfTI files: **not directly confirmed from official docs in this research pass** — verify by `np.unique(mask_array)` on actual data and cross-reference with the challenge's evaluation script.
- ACDC evaluation script conventions for basal/apical slice inclusion/exclusion in volume computation — check official challenge code.
- Whether RV myocardium has any sub-label in ACDC (it does not — RV myocardium is folded into background label 0; only RV cavity is labeled).
- Exact nnU-Net ACDC leaderboard numbers (Dice per class) — accessible at creatis.insa-lyon.fr/Challenge/acdc/results.html (not fetched this pass).
- Gibbs ringing effect size on EF — only qualitative evidence found; quantitative study not located.

---

## Sources

- [S1] RAD229 Lecture Notes / Syllabus — https://web.stanford.edu/class/rad229/Notes.html — accessed 2026-06-17
- [S2] IMAIOS e-MRI Curriculum — https://www.imaios.com/en/e-mri — accessed 2026-06-17
- [S3] bSSFP off-resonance dark banding, joint suppression paper — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11663763/ — accessed 2026-06-17
- [S4] ACDC dataset page, CREATIS — https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html — accessed 2026-06-17
- [S5] Dynamic cardiac MRI bSSFP parameters, PMC — https://pmc.ncbi.nlm.nih.gov/articles/PMC10667461/ — accessed 2026-06-17
- [S6] nnU-Net Nature Methods 2021, Isensee et al. — https://www.nature.com/articles/s41592-020-01008-z — accessed 2026-06-17
- [S7] Artifacts at Cardiac MRI, RadioGraphics — https://pubs.rsna.org/doi/10.1148/rg.230200 — accessed 2026-06-17
- [S8] Banding-free bSSFP frequency modulation — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7048216/ — accessed 2026-06-17
- [S9] Partial volume in cardiac MRI, PLOS ONE — https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0114760 — accessed 2026-06-17
- [S10] IMAIOS Cardiac MRI artifacts chapter — https://www.imaios.com/en/e-mri/image-quality-and-artifacts — accessed 2026-06-17
- [S11] ACDC dataset, Awesome-Medical-Dataset — https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/ACDC.md — accessed 2026-06-17
- [S12] Papillary muscle & Simpson's method, ECR Journal — https://www.ecrjournal.com/articles/measuring-left-ventricular-ejection-fraction-techniques-and-potential-pitfalls — accessed 2026-06-17
- [S13] AHA EF reference — https://www.heart.org/en/health-topics/heart-failure/diagnosing-heart-failure/ejection-fraction-heart-failure-measurement — accessed 2026-06-17
- [S14] Ejection fraction overview — https://en.wikipedia.org/wiki/Ejection_fraction — accessed 2026-06-17
- [S15] Simpson's disk summation, johnsonfrancis.org — https://johnsonfrancis.org/professional/modified-simpsons-rule-for-lvef/ — accessed 2026-06-17
- [S16] U-Net overview, UW-Madison ML Nexus — https://uw-madison-datascience.github.io/ML-X-Nexus/Toolbox/Models/UNET.html — accessed 2026-06-17
- [S17] nnU-Net cardiac evaluation — https://arxiv.org/html/2408.06358v1 — accessed 2026-06-17
- [S18] Segmentation metrics pitfalls — https://arxiv.org/html/2410.02630v2 — accessed 2026-06-17
- [S19] Metrics revolutions biomedical segmentation — https://arxiv.org/html/2410.02630v1 — accessed 2026-06-17
- [S20] Bland-Altman analysis guide — https://innolitics.com/articles/bland-altman-analysis-best-practices-faqs-and-examples/ — accessed 2026-06-17
- [S21] Domain-shift invariant CNN framework cardiac — https://pmc.ncbi.nlm.nih.gov/articles/PMC10501982/ — accessed 2026-06-17
- [S22] Studying robustness domain shift cardiac MRI — https://arxiv.org/abs/2011.07592 — accessed 2026-06-17
