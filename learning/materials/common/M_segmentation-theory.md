# C2 · Segmentation theory (medical)

The model side. You (ML background) know much of this; below is what's specific to
medical/cardiac segmentation.

## U-Net
Encoder–decoder with **skip connections**:
- **Encoder:** conv + downsample stages → feature hierarchy (what's here).
- **Decoder:** upsample + concatenate the matching encoder maps → recover spatial
  detail (where it is). Skips preserve resolution lost in pooling.
- Per-pixel output → softmax → argmax to a label map. Works with **small datasets**
  (original: 30 images) — relevant for ACDC's 100 patients.

## nnU-Net ("no-new-net") — the default to beat
A **self-configuring** framework: reads the dataset fingerprint (spacing, intensity,
sizes) and auto-sets patch size, normalization, augmentation, architecture depth, loss,
post-processing. Matched/beat specialized SOTA on 23 challenges with **no manual
architecture tuning**. **The honest baseline** for ACDC — beating bespoke models is
hard; the value-add is the *analysis*, not a fancier net.
- On ACDC it **defaults to 2D** (anisotropy ~1:5–1:7) and resamples in-plane to
  ~1.56 mm. *(Per nnU-Net papers; treat as reported.)*

## 2D vs 3D on anisotropic data
Short-axis voxels are ~1.5 mm in-plane but ~8 mm through-plane. A 3D conv treats both
equally → wastes capacity on the coarse axis. So:
- **2D (slice-wise)** is the strong, cheap default on ACDC — large in-plane context,
  simple augmentation, independent per-slice normalization.
- **2.5D** (a few adjacent slices as input channels) = middle ground.
- **3D** helps only when through-plane resolution is decent. Start 2D; try 3D if it earns it.

## Loss
- **Dice loss** `1 − 2|P∩G|/(|P|+|G|)` — handles class imbalance by normalizing to
  foreground.
- **Cross-entropy** — per-pixel; sensitive to imbalance (background ≫ foreground).
- **Dice + CE (compound)** — the near-universal cardiac default (CE stability +
  Dice balance). nnU-Net uses it.
- Boundary/Hausdorff-inspired losses — penalize surface error; help HD metrics, less common.

## Augmentation (nnU-Net defaults)
Rotation, scaling, elastic deformation, Gaussian noise/blur, brightness/contrast,
gamma, mirroring — applied online. Cheap robustness; helps small datasets.

## Splits — the one that bites
**Split by PATIENT, never by slice.** All slices of a patient go in the same fold.
Slice-level splitting leaks near-identical neighboring slices across train/val →
**inflated metrics** (can be +5–10 Dice of fake gains).
- ACDC: 100 train patients (use patient-level k-fold; nnU-Net default = 5-fold), 50
  held-out test (server-scored).

## Class imbalance
Background dominates. Pure CE → background-biased. Dice/compound loss or per-class
weighting mitigates.
