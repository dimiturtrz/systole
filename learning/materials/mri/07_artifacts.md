# 07 · Artifacts — as segmentation/EF failure modes

Framed application-first: each artifact = a way the model mis-segments or the EF comes
out wrong. This is the lens for **failure analysis** (Phase D), grounded in real cases.

## Priority for EF accuracy
1. **Motion / ghosting** — ruins whole frames if severe
2. **bSSFP dark banding** — focal but can bisect key structures
3. **Partial volume at apices** — systematic, hard to fix without thinner slices
4. **Flow at the base** — basal-plane errors
5. **Gibbs ringing, bias field** — minor for DL models

## 1. Motion / ghosting — HIGH
Periodic motion (cardiac, respiratory) → **ghost copies** of the heart displaced along
the phase-encode axis. **Effect:** blurred myocardial borders → boundary errors;
respiratory ghosts can overlap the LV cavity; can mimic effusion or thin wall.
**Mitigation:** ECG gating, breath-hold, navigators. (Failed breath-hold = worst case.)

## 2. bSSFP off-resonance dark banding — MODERATE–HIGH
bSSFP has periodic **signal nulls** at off-resonance frequencies spaced **1/TR Hz**
(~250 Hz at TR=4 ms). Local B₀ inhomogeneity pushes spins into a null → **dark bands**
across the image, worse at **3T** and near diaphragm/lung & pericardial fat.
**Effect:** a band over myocardium mimics **wall thinning** (→ myocardium misclassified
as background, low LV mass); over the cavity → misplaced boundary → EF error.
**Mitigation:** shimming, frequency scouting, phase cycling.

## 3. Partial volume effect (PVE) — HIGH at apices
A voxel straddles two tissues → averaged intensity, blurred border. Worst where the LV
**tapers (apex)** and for the **thin RV wall (3–5 mm)**, which is near-entirely
partial-volumed at 8 mm slices. **Effect:** apical volume mis-measured; biases EDV/ESV
(can partly cancel in EF, or net-bias EF if ES apex is falsely enlarged).
**Mitigation:** thinner slices (5 mm > 8 mm). Inherent failure mode to *report*.

## 4. Flow artifacts — LOW–MODERATE
Moving blood (valves, outflow tracts) → signal variation / voids in bSSFP. **Effect:**
a void near a valve can be read as myocardium → **basal-slice** segmentation errors
(LVOT/RVOT at ES). bSSFP is relatively flow-robust but not immune.

## 5. Gibbs / truncation ringing — LOW–MODERATE
Finite k-space (we cut high frequencies) → **ripples at sharp edges**
(blood–myocardium). **Effect:** ~1–2 px boundary uncertainty; slight apparent
wall thick/thin. **Mitigation:** apodization (at the cost of blur). *(EF effect size:
qualitative only — no quantitative study found.)*

## 6. Bias field / intensity inhomogeneity (B1) — LOW for DL
Coil sensitivity → smooth intensity gradient (center brighter). **Effect:** breaks
intensity-threshold methods; DL models are fairly robust **if normalization is
consistent**; mainly a **cross-scanner generalization** risk. **Mitigation:** N4 bias
correction; per-case z-score normalization (nnU-Net default).

## 7. Aliasing / wrap-around — context-dependent
FOV < object along phase-encode → overhanging anatomy **folds** onto the opposite side.
**Effect:** chest wall/liver folded over the heart. Rare with good FOV; seen in large
patients. **Mitigation:** larger FOV, saturation bands, oversampling.

---
**How to use this:** don't memorize — in Phase D, when a case fails, match the failure
to one of these (which slice? base/apex? a band? a ghost?) and **report the mechanism**.
That diagnosis is the senior signal. *(Cross-scanner domain shift — the biggest
real-world failure — is in [../common/E_evaluation-theory.md](../common/E_evaluation-theory.md).)*
