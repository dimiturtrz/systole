# PLAN — systole (what & why)

The design and direction. Ordered milestones/status live in [`ROADMAP.md`](ROADMAP.md); live
granular tasks in `bd` (`bd ready`); the synthesis architecture in detail in
[`core/data/dynamic/GENERATION.md`](../core/data/dynamic/GENERATION.md).

## Thesis
One bounded, honestly-evaluated capability — **segment the heart → measure cardiac function
(ejection fraction) → show where it fails** — pursued across three modalities (MRI → CT → echo), in a
rhythm of *understand the data → look at the data → solve a bounded problem*. The transferable claim
is **cross-vendor / cross-centre generalization measured honestly**: the diversity-in-training
asymmetry (multi-vendor → generalizes; single-centre → collapses) is the headline result.

## Synthesis — the differentiated thread
Beyond training on real images, systole **generates** its training data, and inverts the generator to
read a scan's physics. Two things make this distinctive: it's *physics-based* (bSSFP painted from
labels, not a black-box GAN), and it's a **composite of generation sources**.

- **One generative process (analysis-by-synthesis).** `anatomy → tissue → acquisition → framing →
  image`, a differentiable DAG. Every pipeline is an operation on it: **SAMPLE** (generate), **FIT**
  (invert), **FIX** (condition). Full spec: [`GENERATION.md`](../core/data/dynamic/GENERATION.md).
- **Composite all-synthetic dataset.** No single generator reaches the whole real manifold, so we
  compose a *portfolio of sources* — each enters the DAG at a different point with a different control
  degree, unioned to cover more than any one: fully-parametric (highest control), SSM meshes (Rodero),
  label-space pathology deformation (covers the DCM/HCM/RV tail the healthy SSM misses), MRXCAT
  (whole-torso), learned prior (future).
- **Two directions.** *Uncontrolled* → diversity for training (diversity > fidelity: the most
  physically-accurate synth trains *worse*). *Controlled* → an inverse/parametric **digital-twin**:
  fit the generator's params to a real recording, recover interpretable qMRI-like values, reconstruct
  (fidelity is the objective here). Same engine, opposite objectives.
- **Measured honestly.** Report `real / synth-only / synth+real` as a triad — that comparison IS the
  result, not a leaderboard number. Coverage is measured per factor (shape / color / framing) and per
  source, so we see which source fills which gap.

## Principles
- **No magic constants** — every number derived (public → our-data stats → simulate). Physics >
  statistics.
- **Diversity > fidelity** for training; **fidelity** for the inverse twin.
- **No dependencies** for this artifact; **domain generalization** is the north star.
- **Honest eval** — Dice + surface metrics (HD95/ASSD) *and* clinical metrics (EF MAE, Bland–Altman),
  stratified failure by pathology/vendor.

## The grid — three modalities, one story
Three modalities × three steps, all converging on cardiac function (EF) — one story, not three.

| Modality | Theory | Data viz | Problem |
|---|---|---|---|
| **MRI** (ACDC + M&M-2 + M&Ms-1) | acquisition physics ([mri-sim](../mri-sim/)), short-axis geometry | [cardioview](../cardioview/) 3D viewer + held-out EF | seg LV/myo/RV → EF, cross-dataset (DG) |
| **CT** (MM-WHS) | HU calibration, CTA | EDA | whole-heart / chamber seg |
| **echo** (CAMUS) | ultrasound, 2D+t | EDA | LV seg → EF, Simpson's biplane |

## The geometry thread (cross-cutting)
Computational geometry turns per-voxel labels into a clinical number, and recurs in every cell:
voxel count → physical volume (mm³ → mL) → EF; marching-cubes surface mesh per chamber (live in
cardioview); myocardial wall thickness; Simpson's biplane (echo); spacing/resampling to a common
grid + anisotropy; ED↔ES (and cross-modality) correspondence.

## Structure & how it's driven
- **Structure:** `cardioseg/` (pipeline) + `cardioview/` (viewer) + `mri-sim/` (acquisition). A
  modality/piece is added only when it's real — no speculative folders. Data lives out of the repo
  (licensing + size).
- **Circuit:** field understanding drives the work — research (`research/`) → theory writeup
  (`learning/<date>_<topic>.md`) → quiz on demand → sharpen the next step. Build log = git history.
- **Canonical docs:** architecture → `GENERATION.md`; domain-shift/normalization taxonomy →
  `cardioseg/preprocessing/normalization/README.md`; numbers → `cardioseg/RESULTS.json`.
