# MRXCAT2.0 full-use — the gate: what the phantom actually gives us

**Date:** 2026-07-17 · **Scope:** cross-task (zero-real generation lane, MRXCAT source) · **Beads:**
cardiac-seg-hpy / hpy.1 (3etx) / hpy.3 (u1vz) · **Status:** gate decided; ssmfov training arm §4

Redefined `hpy` asked for *full* use of MRXCAT2.0 — fetch → generate → pathology-swept + whole-FOV
pools → wire as a composite source → zero-real train → triad. Running the bundled tool + probing its
output settles what is actually achievable, and it is much narrower than the goal assumed.

## §1 — the tool runs; the label path is usable now

External checkout works (pinned `9f396a9`, gitignored `external/mrxcat2`). The personalization pipeline
**has run**: `outputData/Patient1/XCAT_warped_vti/Image_01..20.vti` (a biophysically-LV-warped SAX
phantom, 20 cardiac phases) + tissue maps ship as proof. Re-running from scratch needs `cv2` (trivial)
plus, for a *new* background, gated XCAT-generation artifacts (`XCAT_output/Image_log`) — my label-only
warper run aborted there. The **grayscale texturizer weights (LAXsegnet `.pth`) are absent**, so
MRXCAT's own MR image can't be reproduced — but we never wanted it (our painter does color; the module
consumes label `.vti` only).

Both bundled label sources load + remap cleanly through our adapter (`Mrxcat.load_vti_labels` →
`to_canonical`): the `Background` XCAT volume (18 slices × 20 phases) and the `Patient1` warped SAX.

## §2 — RV is real (the gate's decision question)

RV is our weakest class; the worry was a degenerate/static XCAT RV template that could regress it.
**Refuted — RV is a real deforming shape.** Across the 20 cardiac phases of the background phantom, RV
blood swings **30.7k → 50.9k px (51 % of mean)**, footprint +30 %, centroid tracks through z (7.5→10.5)
and x (259→284), **in phase with the LV** (both peak φ5, trough φ12) = genuine ED/ES motion. The warped
Patient1 SAX shows the same (RV 4.3k→6.2k px across phases).

**Caveat from the warper source** (`myMorphingFunctions.warpImages3D`): RV is added as **anchored control
points** (`pts1 == pts2 == ptsRV`) — only the LV is biophysically personalized; RV rides the LV-driven
field pinned to the XCAT template. So RV **beats** realistically but its **cross-patient shape is
template-locked**. MRXCAT adds LV-pathology + whole-FOV + motion, **not RV shape-diversity** → it is not
a fix for the RV deficit (which is closed anyway, `rv-deficit…continuum`).

## §3 — the value is already banked; the pathology sweep is gated out

The decisive finding: **hpy's infrastructure is already built and its principal value already shipped.**

- MRXCAT pools already on disk (`D:/data/…/processed/mrxcat/`): `pool_example` (198, heart-only),
  `pool_fov`/`pool_fov_tight` (214, 7-class whole-FOV), **`pool_ssmfov` (10,488 — our Rodero hearts
  composited into XCAT whole-FOV torso context)**.
- `mri_physics.TORSO_BG` (air .293 / lung .129 / liver .048 / muscle .53) was **derived from `pool_fov`**
  via `Mrxcat.torso_fractions`. That torso composition **is the 0.554 → 0.613 zero-real composition win**
  already in production synth (`ceiling-attribution`). MRXCAT's biggest realized contribution is banked.

- **Pathology sweep (hpy.2) is not achievable from the bundle.** It ships **one background + one
  personalized patient** — no pathology mesh library. New anatomies need either the gated Duke XCAT
  binary (new backgrounds) or a library of patient LV meshes to warp (biophysical personalization);
  neither is available. The DCM/HCM/infarct sweep premise is **killed** unless a mesh library is sourced.

So the only genuinely-untested MRXCAT lever is §4: does the **structured** whole-FOV context of
`pool_ssmfov` beat the **statistically-matched procedural** bg (whose fractions came from this phantom)?

## §4 — the ssmfov zero-real arm (bd hpy.3)

Zero-real on the new `synth_fov` split (`pool_ssmfov` anatomy + `FovBg` whole-FOV painter), quick 40ep,
seed 0. val = ACDC real, test = the locked 642 real set — identical to `synth_main`, so the number is
directly comparable to the banked zero-real **0.613 test / 0.676 val**.

| Arm | bg | val mean | **test mean** | test RV / myo / cav | EF MAE |
|-----|----|----------|---------------|---------------------|--------|
| synth_main (banked) | procedural (randomized) | 0.676 | **0.613** | — / 0.54 / — | ~11–15% |
| **synth_fov** (this) | mrxcat FovBg (structured whole-FOV) | 0.586 | **0.538** (−0.075) | 0.472 / **0.404** / 0.738 | 15.6% |

(zero-real, quick 40ep, seed 0, early-stop @ ep41 best val 0.586; identical val/test to synth_main.)

**Verdict — structured whole-FOV bg LOST (−0.075, well beyond the ±0.02–0.03 single-seed floor).** The
pre-registered negative fired. Myo regressed hardest (0.54 → 0.404): the fixed XCAT torso layout crowds
the myo boundary and lets the model lean on spurious fixed-context cues instead of invariant heart cues.

This is the **`nk70`/`xmcf` lesson a third time**: for zero-real generalization, *randomized diversity
beats structured realism*. The procedural bg's value was never the torso *realism* — it was the
randomized field forcing invariant-cue learning; the statistical composition (`TORSO_BG`, already
distilled from this same phantom) is the part that helped. Placing hearts in one real XCAT context
**removes** diversity and regresses. Single-seed but the drop is 2.5–3× the noise floor and the
mechanism is the thrice-confirmed one — no multi-seed warranted to confirm a negative.

## What is kept vs killed

- **KEPT** — the MRXCAT adapter + pools + `TORSO_BG` derivation (all correct, already paying off through
  composition); the `synth_fov` split (the clean whole-FOV arm, kept as the documented negative +
  reusable if a mesh library ever lands).
- **KILLED** — (1) structured whole-FOV bg as a zero-real lever (§4: −0.075); use the randomized
  procedural field, not FovBg placement. (2) the pathology-sweep pool ambition (hpy.2): not reachable
  from the bundled data (one background + one patient); needs gated XCAT or an external mesh library.
- **Bottom line** — MRXCAT is **capability-complete for what it can give us** (whole-FOV torso
  composition, already banked into production), **not a new Dice lever**. `hpy` closes here.
