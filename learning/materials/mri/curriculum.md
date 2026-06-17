# Curriculum — MRI lane

Ordered study path, **fundamentals first**: how MRI works → how it images the heart →
the heart & the measurement → the data & the task → hands-on. Theory before code.

Status: ✅ done · 🔄 doing · ⬜ planned. Grounded in
[../../../research/deep_dives/](../../../research/deep_dives/).

---

## Phase A — How MRI works (the imaging itself)

### A1 · MRI physics from scratch ✅
Covered in depth (machine + protons + method + k-space + image):
- [01_machine-physics.md](01_machine-physics.md) — magnet, gradients, RF, cooling, the "sandwich".
- [02_proton-physics.md](02_proton-physics.md) — spin, Larmor, resonance, energy scales, T1/T2.
- [03_work-principle.md](03_work-principle.md) — slice select, encoding, the echo, the pulse sequence.
- [04_k-space.md](04_k-space.md) — the raw data grid.
- [05_to-image.md](05_to-image.md) — Fourier → image, + acquisition timing.
*(Quiz pending — say "quiz me" to lock it in.)*

### A2 · From MRI to *cardiac* MRI ✅
Imaging a moving organ: bSSFP cine, ECG gating (prospective/retrospective), segmented
k-space, short-axis stack, ED/ES, anisotropy → 2D. → [06_cardiac-mri.md](06_cardiac-mri.md).

### A3 · Artifacts (failure modes) ✅
Motion/ghosting, bSSFP banding, partial volume, flow, Gibbs, bias field, aliasing — as
segmentation/EF failure modes. → [07_artifacts.md](07_artifacts.md). *(Applied for real
in Phase D failure analysis.)*

---

## Phase B — The heart & the measurement ✅ (cross-modality → `../common/`)
- **B1 · Cardiac anatomy & cycle** ✅ → [../common/cardiac-anatomy-and-cycle.md](../common/cardiac-anatomy-and-cycle.md)
- **B2 · Ejection fraction** ✅ → [../common/ejection-fraction.md](../common/ejection-fraction.md)
- **B3 · Geometry & volumetry** ✅ → [../common/geometry-and-volumetry.md](../common/geometry-and-volumetry.md)

## Phase C — The data & the task ✅
- **C1 · The ACDC dataset** ✅ → [08_acdc-dataset.md](08_acdc-dataset.md)
- **C2 · Segmentation theory** ✅ → [../common/segmentation-theory.md](../common/segmentation-theory.md)
- **Evaluation theory** ✅ → [../common/evaluation-theory.md](../common/evaluation-theory.md)

## Phase D — Hands-on (code) ⬜  ← only thing left
EDA → 2D U-Net baseline → EF vs GT → **failure analysis** (apply A3 artifacts) →
clinical-grade gap. Maps to beads `mri-eda → mri-model → mri-ef`.
**Theory is now prepared; the remaining work is doing D with honest evaluation.**
First D step: verify ACDC label encoding with `np.unique` on a real mask.

---

## Reference curricula (benchmarks — compare our coverage against these)
Real external courses that treat MRI properly. We benchmark against the **relevant
subset** (acquisition → encoding → k-space → image + cardiac), **not** the whole
thing — see [../field-map.md](../field-map.md) for what's in/out of our scope.

**Engineering / signals depth (closest to our style):**
- Stanford **RAD229 — MRI Signals & Sequences** (Hargreaves/Ennis):
  [site](https://web.stanford.edu/class/rad229) · [code](https://github.com/mribri999/MRSignalsSeqs) · [YouTube](https://www.youtube.com/channel/UCJgAoFeFMKQ-f1XVPrFBslQ)
- Books: Nishimura, *Principles of MRI* (Stanford EE369); Bernstein et al., *Handbook of MRI Pulse Sequences*.

**Physics / biomedical:**
- edX — *Fundamentals of Biomedical Imaging: MRI* (EPFL, Gruetter):
  [Class Central](https://www.classcentral.com/course/edx-fundamentals-of-biomedical-imaging-magnetic-resonance-imaging-mri-7058)
- MRI course hub: [Class Central — MRI](https://www.classcentral.com/subject/mri)

**Clinician / interactive:**
- [IMAIOS e-MRI](https://www.imaios.com/en/e-mri); Book: McRobbie, *MRI: From Picture to Proton*.

*(Not individually vetted; real, well-known resources. These are full-depth — we
cover only the subset our lane needs.)*

---
*One sub-topic at a time; I decide when to move on. A1 is done in depth — A2 (cardiac)
is next.*
