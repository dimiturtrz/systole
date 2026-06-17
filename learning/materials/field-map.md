# Field map & coverage — what to learn, and how much

Purpose: a **yardstick**. Where medical-imaging knowledge lives, **which track we're
aiming at**, and a checklist to answer "have we covered enough?" without guessing.
Concrete course links to benchmark against live in each modality's `curriculum.md`
(e.g. [mri/curriculum.md](mri/curriculum.md) → *Reference curricula*).

## Where this knowledge lives (tracks, by focus & depth)
| Track | Who | MRI-physics depth | Focus | (vs us) |
|---|---|---|---|---|
| **Clinical radiology** | radiologists | moderate | image **interpretation**, safety, protocol choice | we don't do diagnosis |
| **MRI technologist** | techs | operational | **running** the scanner, positioning, params | we don't operate hardware |
| **Medical physics** | physicists | **deep, quantitative** | pulse sequences, reconstruction, QA, safety | deeper than we need |
| **BME / EE imaging** | engineers | deep, systems | acquisition, encoding, **reconstruction**, hardware | source of our physics view |
| **Medical image computing ← US** | ML / CV | **physics-enough** + analysis | datasets, **segmentation, evaluation, clinical metrics** (MICCAI) | our lane |

What we studied in A1 = the **medical-physics / imaging-engineering** *view* of MRI,
taken only to the depth that explains the data. Our actual deliverable lives in
**medical image computing**.

## Our target competency (the bar) — two stacks
**A) Imaging-physics-enough** (per modality): acquisition principle · contrast
mechanism · why intensity is (un)calibrated · geometry/units/spacing · key artifacts ·
dataset conventions. *Enough to understand the data and its failure modes — NOT to
build a scanner or design pulse sequences.*

**B) Analysis stack** (the real work): data loading + normalization · segmentation
(architecture, 2D vs 3D, leakage) · evaluation (Dice, Hausdorff, **failure analysis**,
calibration) · clinical metric (**EF**) + agreement vs ground truth · the
clinical-grade gap.

## Coverage checklist (tick as we go) — ✅ done · ⬜ pending
**Per modality** (MRI ✅ theory · CT ⬜ · echo ⬜):
- [x] acquisition principle — MRI ✅ (`mri/01,02,06`)
- [x] contrast mechanism + intensity (un)calibration — MRI ✅ (`mri/02`)
- [x] geometry / units / spacing + key artifacts — MRI ✅ (`mri/07`)
- [x] dataset conventions — MRI/ACDC ✅ (`mri/08`)
**Cross-cutting (theory done — reused by all modalities):**
- [x] geometry: voxel→volume→EF, Simpson's, meshing (`common/geometry-and-volumetry`)
- [x] segmentation theory: U-Net; 2D/3D; leakage (`common/segmentation-theory`)
- [x] evaluation: Dice / Hausdorff / failure / calibration (`common/evaluation-theory`)
- [x] clinical metric (EF) + validation gap (`common/ejection-fraction`, `evaluation-theory`)

**MRI status:** **theory complete** — physics (A1), cardiac (A2), artifacts, dataset,
+ all cross-cutting analysis theory. **Remaining = Phase D hands-on (code)**, where the
artifact/eval theory gets applied to real ACDC data. CT/echo: theory pending (reuse `common/`).

## "Are we done?" rule
For our lane, **enough** = Stack A at the depth that **explains the data's failure
modes**, plus Stack B **executed with honest evaluation**. We are **not** chasing
medical-physics or radiology completeness — that's explicitly out of scope. When we
deliberately stop short of a track's full depth, **say so** (honesty) rather than
implying full mastery.

---
*Tracks sketched from knowledge (a map, not gospel); the verifiable benchmarks are the
real course links in each `curriculum.md`. Refine with a research pass if we want exact
syllabus topic-lists to tick against.*
