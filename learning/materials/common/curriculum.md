# Curriculum — analysis stack (cross-modality)

The **modality-agnostic** half of the pipeline: turn images into trustworthy
numbers. Reused identically for MRI, CT, echo — which is why it lives in `common/`,
not under a modality. Counterpart to [../mri/curriculum.md](../mri/curriculum.md)
(MRI physics, Stack A); this is **Stack B** in [../field-map.md](../field-map.md).

Sits on top of [../foundations/](../foundations/) (the maths). Theory before code;
hands-on logs in [../../](../../) (`<date>_<topic>.md`). Status: ✅ done · 🔄 doing
· ⬜ planned. Grounded in
[../../../research/deep_dives/2026-06-18_ml-geometry-application-curriculum.md](../../../research/deep_dives/2026-06-18_ml-geometry-application-curriculum.md).

---

## Phase M — Modelling / DL segmentation (the model side)
The learned function image → per-voxel class. Theory stub:
[segmentation-theory.md](segmentation-theory.md). Hands-on log:
[../../2026-06-18_ml-segmentation.md](../../2026-06-18_ml-segmentation.md).

- **M1 · CNN fundamentals** 🔄 — convolution, pooling, stride/padding, receptive
  field, feature hierarchy, translation equivariance.
- **M2 · U-Net architecture** ⬜ — encoder/decoder, skip connections, why it beats a
  plain CNN here (Ronneberger 2015).
- **M3 · Losses** ⬜ — Dice, cross-entropy, compound Dice+CE, class imbalance,
  boundary losses.
- **M4 · Training & augmentation** ⬜ — optimizer, LR, epochs, what a gradient step
  does, augmentation set, overfitting.
- **M5 · Splits / leakage** ✅(theory) — patient-level split, k-fold CV.
- **M6 · 2D vs 3D vs 2.5D** ✅(theory) — the anisotropy decision.
- **M7 · nnU-Net baseline** ⬜ — the self-configuring "default to beat" (Isensee
  2021); the *Revisited* validation caveat (MICCAI 2024).

## Phase G — Geometry / measurement (the number side)
Per-voxel labels → clinical scalar + 3D model. Theory stub:
[geometry-and-volumetry.md](geometry-and-volumetry.md). Hands-on log:
[../../2026-06-18_geometry-volumetry.md](../../2026-06-18_geometry-volumetry.md).

- **G1 · voxel → volume / Simpson's** ✅(theory) — spacing, anisotropy, mm³→mL,
  stack-of-disks.
- **G2 · marching cubes** ⬜ — mask → triangle surface mesh (Lorensen & Cline 1987).
- **G3 · mesh processing** ⬜ — surface area, smoothing, decimation, wall thickness
  (endo↔epi surface distance). Libraries: VTK / PyVista / trimesh.
- **G4 · distance / Hausdorff** ⬜ — distance transforms; HD/HD95 as geometry.
- **G5 · registration** ⬜ — ED↔ES (rigid→affine→deformable); SimpleITK.
- **G6 · shape models / 3D anatomical modelling** ⬜ (awareness) — atlases, PCA on meshes.

## Phase E — Evaluation & validation rigor
Theory stub: [evaluation-theory.md](evaluation-theory.md).
- **E1 · overlap + boundary metrics** ✅(theory) — Dice/Jaccard, Hausdorff/HD95/ASSD.
- **E2 · metric pitfalls** ⬜ — Metrics Reloaded (one metric misleads).
- **E3 · function agreement** ⬜ — Bland-Altman (bias + limits of agreement) for EF;
  *not* MAE. (Fix the README wording.)
- **E4 · calibration / uncertainty** ⬜ — MC-dropout / ensembles; flag bad cases.
- **E5 · domain shift / clinical-grade gap** ⬜ — M&Ms multi-vendor; why ACDC over-states.

---

## Reference curricula (benchmark our coverage against these)
Real courses that treat this stack properly; we cover the practitioner subset
(use the libraries, own the judgment) — see [../field-map.md](../field-map.md).

**DL for medical image segmentation:**
- UF **EEL6935 — Deep Learning in Medical Image Analysis** (Shao):
  [syllabus](https://www.ece.ufl.edu/wp-content/uploads/syllabi/Fall%202023/EEL6935_Deep_Learning_Med_Image_Shao_Fall_2023.pdf) — closest ordering to ours.
- Purdue **Applied Medical Image Processing & Analysis**; CMU **Methods in MIA**;
  Duke **Machine Learning and Imaging** ([deepimaging.github.io](https://deepimaging.github.io/)).
- **MONAI tutorials** ([Project-MONAI/tutorials](https://github.com/Project-MONAI/tutorials)) — the framework we use; nnU-Net runner.
- Papers: U-Net (Ronneberger 2015); nnU-Net (Isensee 2021, *Nature Methods*); nnU-Net Revisited (MICCAI 2024).
- General DL: Stanford **CS231n**, **fast.ai**. *(Standard; exact syllabi not vetted.)*

**Computational geometry / meshes:**
- UC Berkeley **CS294-74 — Mesh Generation & Geometry Processing** (Shewchuk):
  [site](https://people.eecs.berkeley.edu/~jrs/mesh/) — marching cubes → Delaunay → mesh processing.
- Marching cubes: Lorensen & Cline 1987; [Stanford note](https://graphics.stanford.edu/~mdfisher/MarchingCubes.html).
- Libraries: **VTK** ([vtk.org](https://vtk.org/)), **PyVista** ([docs](https://docs.pyvista.org/)), trimesh, **SimpleITK** (registration).

**Evaluation:** Metrics Reloaded (Maier-Hein); Bland-Altman 1986; **M&Ms** multi-vendor challenge ([ub.edu/mnms](https://www.ub.edu/mnms/)).

*(Full-depth courses; we take the subset our lane needs — not building algorithms
from scratch. One sub-topic at a time; learner decides when to move on.)*
