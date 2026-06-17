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

### A2 · From MRI to *cardiac* MRI 🔄
Imaging a moving organ: ECG gating, cine, bSSFP, short-axis stack, anisotropy.
Partially covered (timing/cine bridge in [05_to-image.md](05_to-image.md)); needs its
own writeup. IMAIOS e-MRI Ch.13 = Cardiac MRI.

---

## Phase B — The heart & the measurement
- **B1 · Cardiac anatomy & the cardiac cycle** ⬜ — chambers, myocardium, ED/ES.
- **B2 · Ejection fraction — what & why** ⬜ — `EF=(EDV−ESV)/EDV`; CMR as gold standard.
- **B3 · From image to number — geometry** ⬜ — voxel→volume, Simpson's method.

## Phase C — The data & the task (leads toward code)
- **C1 · The ACDC dataset** ⬜ — structure, labels, format, splits.
- **C2 · Segmentation, in theory** ⬜ — U-Net; 2D vs 3D given anisotropy; leakage.

## Phase D — Hands-on (code)
EDA → 2D U-Net baseline → EF pipeline → failure analysis → clinical-grade gap.
Maps to beads `mri-eda → mri-model → mri-ef`. *Not before the theory above.*

---
*One sub-topic at a time; I decide when to move on. A1 is done in depth — A2 (cardiac)
is next.*
