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

## Coverage checklist (tick as we go)
**Per modality** — MRI · CT · echo:
- [ ] acquisition principle
- [ ] contrast mechanism + intensity (un)calibration
- [ ] geometry / units / spacing + key artifacts
- [ ] dataset conventions
**Cross-cutting:**
- [ ] geometry: voxel→volume→EF, Simpson's, meshing
- [ ] segmentation theory (U-Net; 2D/3D; leakage)
- [ ] evaluation: Dice / Hausdorff / failure analysis / calibration
- [ ] clinical metric (EF) + validation gap

**MRI status:** acquisition + contrast + geometry-of-imaging ✅ (A1). Cardiac
specifics (A2) 🔄. MRI dataset/seg/eval ⬜.

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
