# Glossary — plain-language reference

Terms used across this repo, grouped. One line each, no assumed knowledge.

## Heart anatomy
- **LV** — left ventricle. The big pump; pushes oxygenated blood to the whole body. High pressure, thick wall.
- **RV** — right ventricle. The smaller pump; pushes blood to the lungs. Low pressure, thin wall, crescent-shaped.
- **myocardium / myo** — heart *muscle* (the wall). The LV myo is the thick ring we segment.
- **LV cavity / blood pool** — the blood-filled space *inside* the LV. Its volume is what EF is computed from.
- **endocardium / epicardium** — inner / outer surface of the myocardium (cavity-side / lung-side).
- **papillary muscles** — little muscle nubs poking into the LV cavity. By convention counted as *cavity*, not muscle.
- **base / apex** — top of the heart (near the valves) / bottom tip. Short-axis slices run base → apex.

## Cardiac cycle & function
- **ED — end-diastole** — heart relaxed and *fullest* of blood. Largest LV cavity. (ACDC gives this frame.)
- **ES — end-systole** — heart squeezed and *emptiest*. Smallest LV cavity.
- **EDV / ESV** — end-diastolic / end-systolic *volume* of the LV cavity (in mL).
- **SV — stroke volume** — blood ejected per beat = EDV − ESV.
- **EF — ejection fraction** = (EDV − ESV) / EDV, as a %. *The* number: fraction of blood pumped out per beat. ~55–70% normal; <40% = heart failure.
- **pathology groups** (ACDC, 30 patients each):
  - **NOR** normal · **DCM** dilated (big floppy weak LV, low EF) · **HCM** hypertrophic (thick stiff wall) · **MINF** prior heart attack (scar) · **ARV** abnormal right ventricle.

## Imaging
- **MRI** — magnetic-resonance imaging. No radiation; great soft-tissue contrast.
- **bSSFP cine** — the MRI sequence used for cardiac movies; bright blood, ~30 frames over a heartbeat.
- **short-axis** — slicing the heart in cross-sections perpendicular to its long axis (the round "donut" views).
- **voxel** — a 3D pixel. Has a physical size in mm.
- **spacing** — physical size of a voxel, e.g. 1.56 × 1.56 × 10 mm.
- **anisotropy** — voxels not cube-shaped. Ours ~6–7× (in-plane fine, slices coarse) → we model in 2D.
- **DICOM** — clinical image format: one file per slice, huge header, patient PII. What scanners produce.
- **NIfTI** (`.nii.gz`) — research format: whole 3D/4D volume in one file, de-identified. What ACDC ships.
- **affine** — 4×4 matrix in the NIfTI header mapping voxel index → physical mm position (spacing + orientation).
- **z-score normalize** — rescale intensities to mean 0 / std 1. Needed because MRI values are uncalibrated.

## Segmentation & ML
- **segmentation** — labeling every voxel with a class. Our output.
- **mask / label map** — the per-voxel class image. ACDC classes: 0 bg, 1 RV, 2 LV-myo, 3 LV-cavity.
- **class** — one label value (e.g. class 3 = LV cavity).
- **U-Net** — the standard segmentation neural network (encoder shrinks, decoder grows back, skip connections keep detail).
- **2D vs 3D** — process each slice alone (2D) vs the whole volume (3D). Anisotropy → 2D wins here.
- **augmentation** — random flips/rotations/intensity tweaks on training data so the model generalizes.
- **patient-level split** — all of one patient's slices stay in train *or* val, never both. Else **leakage** (near-identical neighbour slices) fakes good scores.
- **epoch** — one full pass over the training data.

## Evaluation metrics
- **Dice** — overlap score between predicted and true mask, 0–1 (1 = perfect). The main seg metric.
- **Hausdorff (HD / HD95)** — worst-case boundary distance in mm (how far off the contour is). Catches errors Dice hides.
- **MAE** — mean absolute error. We report EF MAE in % (avg |predicted EF − true EF|).
- **failure analysis** — looking at the *worst* cases, not the average. Which patients/pathologies break.
- **SOTA — state of the art** — best published result on a benchmark. ⚠️ see caveat below.

## The "SOTA" caveat (read this)
ACDC is **one hospital, one-ish scanner type, curated** — a *homogeneous* dataset. Matching published
ACDC Dice numbers means "competent on this clean benchmark", **not** "clinical-grade" or "state of the art
in the real world." A model can score high here and still fail on a different scanner/vendor/site
(domain shift). So in this repo we say **"comparable to published ACDC results"**, never "we are SOTA."
The honest gap — multi-vendor robustness, validation — is the hard 80% (see ROADMAP / research notes).
