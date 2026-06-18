# Application Curriculum: DL Segmentation + Computational Geometry (the "analysis stack")

**Date**: 2026-06-18
**Status**: grounded (real references fetched/searched 2026-06-18)
**Purpose**: ground a learning curriculum for the *analysis* half of the cardiac
pipeline — deep-learning segmentation + computational geometry + evaluation —
benchmarked against real university courses, the way the MRI track is benchmarked
against Stanford RAD229. Audience: experienced ML engineer (audio background),
MRI physics already covered, using libraries (not building from scratch).

## TL;DR
The analysis stack has three teachable layers: (1) **foundations / maths** —
linear algebra, calculus/optimization, probability (Deisenroth is the canonical
text); (2) **DL segmentation** — CNN → U-Net → nnU-Net, loss, training, evaluation
(real courses: UF EEL6935, Purdue, CMU; framework: MONAI); (3) **computational
geometry** — voxel→volume, marching cubes, mesh processing, registration (real
course: Berkeley CS294-74 Shewchuk; libraries: VTK/PyVista/SimpleITK). University
medical-image-analysis courses consistently sequence: imaging basics → CNN →
segmentation (U-Net 2D/3D) → registration → generative/advanced. Geometry is a
*separate* discipline (mesh generation / geometry processing) the medical courses
mostly assume; the target job calls it out explicitly (VTK, meshes, comp-geom),
so it deserves first-class treatment here.

---

## 1. Foundations / maths (the layer under everything)

A practitioner needs working fluency (not proofs) in three areas. Canonical, free,
widely-cited text: **Deisenroth, Faisal & Ong, *Mathematics for Machine Learning***
(Cambridge, free PDF) [M1] — explicitly structured as linear algebra → calculus →
probability, "for ML." Companion: **DeepLearning.AI / Math Academy** specializations
[M2][M3], and the **dair-ai/Mathematics-for-ML** resource list [M4].

Where each is *load-bearing in our pipeline* (this is the angle that makes it stick):
- **Linear algebra** — vectors/matrices, matrix multiply, eigen/SVD, **affine
  transforms**. Directly: the NIfTI **affine** (voxel→mm), spacing/resampling,
  image rotation in augmentation, convolution as a (sparse, shared) linear operator.
- **Calculus / optimization** — gradients, chain rule (= backprop), gradient
  descent, convexity, learning rate. Directly: how `loss.backward()` + Adam update
  the U-Net weights; why LR matters; why loss curves look like they do.
- **Probability / statistics** — distributions, likelihood, **cross-entropy**,
  expectation/variance, estimation, agreement statistics. Directly: the CE half of
  Dice+CE loss; softmax as a distribution; Bland-Altman / limits-of-agreement for
  EF; cross-validation variance.

Scope: working understanding + intuition, not measure theory. The point is to read
a paper's loss/method section and know *why*, and to debug training/eval.

## 2. Deep-learning segmentation (the model side)

### Real university courses (benchmark for ordering)
- **UF EEL6935 — Deep Learning in Medical Image Analysis** (Wei Shao) [S1]. Ordering:
  medical-imaging basics → CNNs → image classification → **segmentation (2D & 3D
  U-Net, single- and multi-class)** → transformers → **image registration** → GANs
  → image-to-image translation → super-resolution → diffusion. *This is the closest
  match to our lane's ordering.*
- **Purdue — Applied Medical Image Processing and Analysis** [S2]: imaging physics →
  enhancement → **segmentation → registration → visualization** (tools: Python, 3D
  Slicer, ITK, ImageJ).
- **CMU — Methods in Medical Image Analysis** (Galeotti) [S3]; **Duke — Machine
  Learning and Imaging** (deepimaging.github.io) [S4]; **VU Amsterdam — Deep
  Learning for Medical Image Analysis** [S5]. All teach CNN → segmentation as the spine.

### The core topic chain (what to actually know)
1. **CNN fundamentals** — convolution, pooling, stride, padding, **receptive field**,
   feature hierarchy, translation equivariance.
2. **U-Net** (Ronneberger et al. 2015) [S6] — encoder (contract) / decoder (expand)
   / **skip connections** (recover spatial detail). Works on small datasets.
3. **Losses** — Dice, cross-entropy, **compound Dice+CE** (cardiac default),
   boundary/Hausdorff-inspired losses, class imbalance.
4. **Training mechanics** — optimizer (Adam/SGD), LR/schedule, epochs, batch,
   **augmentation** (flip/rotate/scale/intensity/elastic), overfitting, early stop.
5. **Data discipline** — **patient-level splits** (no slice leakage), k-fold CV.
6. **2D vs 3D vs 2.5D** — the anisotropy decision (cardiac short-axis → 2D).
7. **nnU-Net** (Isensee et al. 2021, *Nature Methods*) [S7] — self-configuring
   baseline; the "default to beat." Caveat: **nnU-Net Revisited** (MICCAI 2024) [S8]
   argues for rigorous validation and that many "improvements" don't hold up.

### Framework
- **MONAI** [S9] — PyTorch-based medical-imaging library (transforms, networks,
  losses, metrics). Official **Project-MONAI/tutorials** incl. nnU-Net runner [S9].
  MICCAI Educational Initiative explainer for nnU-Net [S10].
- General DL grounding (cite real, full-depth): Stanford CS231n (CNNs for visual
  recognition), fast.ai. *(Named as standard; not re-vetted this pass — UNVERIFIED
  exact current syllabi.)*

## 3. Computational geometry (the measurement / modeling side)

The medical-DL courses mostly *assume* this; it's its own discipline. The target
job emphasizes it (VTK, meshes, comp-geom), so treat it first-class.

### Real course (benchmark)
- **UC Berkeley CS294-74 — Mesh Generation and Geometry Processing** (Jonathan
  Shewchuk) [G1]. Topic spine (fetched): **isosurface extraction (marching cubes +
  variants, dual contouring, the marching-cubes ambiguity)** → simplicial complexes
  / manifolds / triangulations → Delaunay triangulations → mesh generation (Delaunay
  refinement, advancing front, octree) → mesh data structures → **mesh processing
  (smoothing, simplification via quadric error, parametrization)**. Core reading:
  Bern & Eppstein survey; Shewchuk's robustness notes.
- Also: RPI CSCI 4560/6560 [G2], UIUC (Erickson) [G3] — classical comp-geom (convex
  hull, Voronoi, Delaunay) — broader than we need.

### The core topic chain (what we actually use)
1. **Voxel grid → physical volume** — spacing, anisotropy, mm³→mL; **Simpson's
   method** = stack-of-disks (Σ area × thickness) = what voxel-counting implements.
2. **Marching cubes** (Lorensen & Cline 1987) [G4] — isosurface extraction; binary
   mask → triangle surface mesh. The mdfisher/Stanford note is a clean explainer [G5].
3. **Mesh processing** — surface area, **smoothing** (Laplacian/Taubin),
   **decimation/simplification** (quadric error), **wall thickness** = surface-to-
   surface distance (endo↔epi).
4. **Distance transforms / Hausdorff** — boundary distance; HD/HD95 as geometry
   (not overlap). SciPy `distance_transform_edt` (what our `evaluate.hausdorff` uses).
5. **Registration** — align ED↔ES (and later cross-modality); rigid → affine →
   deformable. Standard tool: **SimpleITK** (registration notebooks) [G6].
6. **Statistical shape models / 3D anatomical modeling** — awareness level (atlases,
   PCA on meshes); the rung above EF measurement.

### Libraries (the job names these)
- **VTK** (Visualization Toolkit) [G7] — the workhorse for isosurfacing + rendering
  + mesh ops; the "VTK Textbook"/User's Guide.
- **PyVista** [G8] — Pythonic VTK wrapper (much easier API).
- **trimesh** — lightweight mesh analysis; **scikit-image** `marching_cubes` (what
  our `viz.py` uses now); **SimpleITK** — I/O, registration, distance maps.

## 4. Evaluation & validation rigor (medical)

- **Overlap**: Dice, Jaccard/IoU. **Boundary**: Hausdorff, **HD95**, ASSD/MASD.
- **Pitfalls**: **Metrics Reloaded** (Maier-Hein et al.) [E1] — why a single metric
  misleads; pick metrics by failure mode.
- **Function agreement**: **Bland-Altman** (bias + limits-of-agreement) for EF/EDV/
  ESV — *not* the same as MAE (a current sloppiness in our README to fix) [E2].
- **Calibration / uncertainty**: MC-dropout / ensembles; flag unreliable cases.
- **Generalization / domain shift**: **M&Ms challenge** (multi-centre, multi-vendor)
  [E3] — the clinical-grade gap; ACDC (single-centre) over-states real performance.
- **Cross-validation** — patient-level k-fold; report variance, not one split.

## 5. Proposed phased curriculum (mirrors the MRI track's A→D)

Modality-AGNOSTIC (reused for CT/echo), so it sits in the cross-modality area, not
under `mri/`. Recommended structure:

- **Phase F — Foundations / maths** (new `materials/foundations/`): F1 linear
  algebra · F2 calculus & optimization · F3 probability & statistics. Each tied to
  where it appears in the pipeline.
- **Phase M — Modelling / DL segmentation** (deepen `materials/common/segmentation-
  theory.md` + hands-on logs): M1 CNN fundamentals · M2 U-Net architecture · M3 loss
  · M4 training & augmentation · M5 splits/leakage · M6 2D/3D · M7 nnU-Net baseline.
- **Phase G — Geometry / measurement** (deepen `materials/common/geometry-and-
  volumetry.md`): G1 voxel→volume/Simpson's · G2 marching cubes · G3 mesh processing
  & wall thickness · G4 distance/Hausdorff · G5 registration · G6 shape models (aware).
- **Phase E — Evaluation** (deepen `materials/common/evaluation-theory.md`): metrics,
  pitfalls, Bland-Altman, domain shift.

**In scope** (library-using practitioner): use U-Net/nnU-Net via MONAI; use marching
cubes/VTK/PyVista; understand the maths well enough to choose losses/metrics, debug
training, and reason about error. **Out of scope**: deriving mesh-generation
algorithms, writing a CUDA conv kernel, measure-theoretic probability, building a
scanner. We *use* the toolbox and own the *judgment*.

## 6. Map to the target job (coverage check)
| Job requirement | Curriculum home |
|---|---|
| PyTorch, deep learning models | Phase M (+ Foundations) |
| 3D segmentation / reconstruction | Phase M + Phase G |
| computational geometry | Phase G |
| process 3D meshes / VTK | Phase G (VTK/PyVista) |
| NumPy / SciPy | Foundations + threaded throughout |
| validation / robustness / accuracy | Phase E |
| optimize algorithms | Foundations (optimization) + M/G |

---

## Sources
- [M1] Deisenroth, Faisal, Ong — *Mathematics for Machine Learning* — https://mml-book.github.io/ — 2026-06-18
- [M2] DeepLearning.AI — Mathematics for ML & Data Science — https://www.deeplearning.ai/courses/mathematics-for-machine-learning-and-data-science-specialization/ — 2026-06-18
- [M3] Math Academy — Mathematics for ML — https://www.mathacademy.com/courses/mathematics-for-machine-learning — 2026-06-18
- [M4] dair-ai — Mathematics-for-ML — https://github.com/dair-ai/Mathematics-for-ML — 2026-06-18
- [S1] UF EEL6935 — Deep Learning in Medical Image Analysis (Shao) — https://www.ece.ufl.edu/wp-content/uploads/syllabi/Fall%202023/EEL6935_Deep_Learning_Med_Image_Shao_Fall_2023.pdf — 2026-06-18
- [S2] Purdue — Applied Medical Image Processing and Analysis — https://engineering.purdue.edu/online/courses/applied-medical-image-processing-and-analysis — 2026-06-18
- [S3] CMU — Methods in Medical Image Analysis — http://biglab.ri.cmu.edu/galeotti/methods_course/medical_image_analysis_course_2018/syllabus.html — 2026-06-18
- [S4] Duke — Machine Learning and Imaging — https://deepimaging.github.io/ — 2026-06-18
- [S5] VU Amsterdam — Deep Learning for Medical Image Analysis — https://research.vu.nl/en/courses/deep-learning-for-medical-image-analysis-2/ — 2026-06-18
- [S6] Ronneberger et al. 2015 — U-Net — https://arxiv.org/abs/1505.04597 — 2026-06-18
- [S7] Isensee et al. 2021 — nnU-Net, Nature Methods — https://www.nature.com/articles/s41592-020-01008-z — 2026-06-18
- [S8] nnU-Net Revisited (MICCAI 2024) — https://link.springer.com/chapter/10.1007/978-3-031-72114-4_47 — 2026-06-18
- [S9] Project-MONAI/tutorials (incl. nnU-Net) — https://github.com/Project-MONAI/tutorials — 2026-06-18
- [S10] MICCAI Educational Initiative — nnU-Net explainer — https://medium.com/miccai-educational-initiative/nnu-net-the-no-new-unet-for-automatic-segmentation-8d655f3f6d2a — 2026-06-18
- [G1] UC Berkeley CS294-74 — Mesh Generation & Geometry Processing (Shewchuk) — https://people.eecs.berkeley.edu/~jrs/mesh/ — 2026-06-18
- [G2] RPI CSCI 4560/6560 — Computational Geometry — https://www.cs.rpi.edu/~cutler/classes/computationalgeometry/F23/syllabus.php — 2026-06-18
- [G3] UIUC — Computational Geometry (Erickson) — https://jeffe.cs.illinois.edu/teaching/compgeom/ — 2026-06-18
- [G4] Lorensen & Cline 1987 — Marching Cubes (orig. paper) — https://dl.acm.org/doi/10.1145/37402.37422 — 2026-06-18
- [G5] Stanford/mdfisher — Marching Cubes note — https://graphics.stanford.edu/~mdfisher/MarchingCubes.html — 2026-06-18
- [G6] SimpleITK — registration notebooks — https://simpleitk.org/ — 2026-06-18
- [G7] VTK — Visualization Toolkit — https://vtk.org/ — 2026-06-18
- [G8] PyVista — https://docs.pyvista.org/ — 2026-06-18
- [E1] Maier-Hein et al. — Metrics Reloaded — https://arxiv.org/abs/2206.01653 — 2026-06-18
- [E2] Bland-Altman analysis (method agreement) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2votes/ — UNVERIFIED exact URL; canonical ref Bland & Altman 1986, Lancet — 2026-06-18
- [E3] M&Ms Challenge — multi-centre multi-vendor cardiac — https://www.ub.edu/mnms/ — 2026-06-18

## Open / UNVERIFIED
- UF EEL6935 PDF did not fetch (server returned dept homepage); topic ordering taken
  from the search-result abstract, not the PDF body. Re-fetch to confirm exact weeks.
- CS231n / fast.ai named as standard DL grounding but exact current syllabi not vetted.
- [E2] Bland-Altman canonical citation is the 1986 Lancet paper; exact stable URL not pinned.
