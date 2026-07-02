# ROADMAP — systole (order & status)

Time-ordered execution view. The *what & why* is in [`PLAN.md`](PLAN.md); the synthesis architecture
in [`GENERATION.md`](../core/data/dynamic/GENERATION.md); live granular tasks in `bd` (`bd ready`).

Status: ✅ done · 🔄 doing · ⬜ planned.

## Where it stands (2026-07)
**MRI lane delivered, past Gate 1.** Trained Siemens+Philips (M&M-2 + M&Ms-1, **495 labelled**,
2 vendors); val = ACDC-150 (centre/protocol shift); test = Canon+GE (unseen vendors, held out).
- **Segmentation generalizes** — val ACDC mean Dice **0.88** (ED+ES); unseen Canon+GE both **0.839**
  (n=78). Reverse (single-centre ACDC → multi-vendor) collapses to **0.70** (RV 0.85→0.59). The
  diversity-in-training asymmetry is the headline.
- **EF improved, not clinical** — val ACDC EF MAE **6.5%**, bias −5.6%, LoA [−20.1, +8.9]. Good masks,
  fragile derived number.
- **Shipped:** [cardioview](../cardioview/) viewer + in-browser ONNX seg, [mri-sim](../mri-sim/),
  surface metrics (HD95/ASSD), error-distribution plots.

## EF paths — weak number → usable (effort order)
1. ✅ Postprocess (largest-CC) — EF MAE 9.4→8.2%, HD RV 191→59 mm.
2. ✅ Test-time augmentation (flip averaging).
3. ✅ Heavy aug + early stop + multi-source pooling — RV Dice 0.84→0.88, EF MAE 8.2→6.5%. nnU-Net
   still leads unseen-vendor EF (Canon 2.6% / GE 4.3% vs ours 11.9/11.3) — the gap is **model-class
   epistemic**, not the floor; we trade it for a 57× smaller deployable ONNX model.
4. ⬜ Cross-scanner harmonization — **deprioritized by evidence** (in-domain vendors already level;
   `bd cardiac-seg-qfz`).
5. ⬜ Bias calibration (held-out linear EF correction).
6. ⬜ Stronger seg + SOTA benchmark — nnU-Net done; benchmark CardioSAM cross-vendor (`bd …-0h7`).
7. ⬜ Eval rigor — 5-fold CV (`bd …-4ev`); UQ / calibration flags (`bd …-iq7`).

## Synthesis thread — status
Design in [`PLAN.md`](PLAN.md) / [`GENERATION.md`](../core/data/dynamic/GENERATION.md). Forward engine
(SAMPLE) is fully built; the composite sources and the inverse (FIT) are the live frontier.
- ✅ **Physics painter** — bSSFP from tissue params, swept acquisition, corruption chain; whole-FOV bg
  (flat / procedural / partition / hybrid strategies).
- ✅ **Tier-1 augmentation of real images** — **concluded, negative**: bias-field aug *regressed* the
  unseen-vendor gap; 4-seed ensemble shows only ~15–18% reducible headroom → the gap is aleatoric +
  model-class. ([normalization README](../cardioseg/preprocessing/normalization/README.md))
- ✅ **SSM anatomy source** (Rodero) — voxelizer + 1000-mesh pool; zero-real generation ~0.56
  cross-vendor (probe's "walled" verdict was wrong once confounds fixed).
- ✅ **Label-space pathology source** — DCM/HCM/abnormal-RV deformation; closes the shape-coverage
  tail (DCM 0.70→0.96, all groups ~0.9+). Composite Dice **neutral** (~0.57): coverage-in-descriptors
  ≠ Dice; the residual is shape *fidelity* (boundary detail) + color, not coverage.
- 🔄 **Inverse / digital-twin (FIT)** — operator built (differentiable render + fit); one-frame heart
  fit is degenerate (2 tissues + uncalibrated intensity). Needs multi-acquisition, shared scale.
- ⬜ **MRXCAT source** (whole-torso), **learned shape prior**, **color-axis source** (the binding
  constraint above ~0.68).

## Resolved
- **Machine axis tested** (n=78, two vendors agree) — Canon+GE both 0.839; leave-one-vendor-out CV
  still open (`bd cardiac-seg-bsz`).

## Gates
- **Gate 1 — MRI presentable** ✅ (done + exceeded: domain generalization, multi-vendor).
- **Gate 2 — EF you could defend** 🔄 — close cross-dataset EF, add eval rigor, honest gap writeup.
- **Gate 3 — CT lane** ⬜ (MM-WHS, reuse pipeline).
- **Gate 4 — echo lane** ⬜ (CAMUS, Simpson's biplane).

---
*Build log = git history. Theory writeups → `learning/`; research → `research/`.*
