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
5. ✅ Bias calibration (held-out linear EF correction) — fit `ef_corr = 1.10·ef_pred + 2.1` on VAL,
   apply once to TEST (`python -m cardioseg.evaluation ef_calibrate`). Removes the systematic EF bias:
   val MAE 7.1→5.4 / bias −6.4→0.0; unseen-vendor **transfers** — Canon 9.9→5.4 / GE 11.0→7.4 MAE,
   bias −10→−3. The bias is vendor-systematic, so one linear fit carries OOD (residual −3pp = the
   correction under-shoots the larger OOD shift; post-hoc calibration stays domain-shift-limited).
   Source-level levers (differentiable vol-consistency loss, Kaggle EF-only weak supervision) each buy
   ~1.5–2pp EF MAE Dice-free and stack with calibration — see
   `interpretations/ef/2026-07-15_ef_defensibility.md`.
6. ⬜ Stronger seg + SOTA benchmark — nnU-Net done; benchmark CardioSAM cross-vendor (`bd …-0h7`).
7. 🔄 Eval rigor — **EF now carries a bootstrap 95% CI** (`qhdm`): every reported MAE/bias has a
   percentile-bootstrap error bar (`Measure.bootstrap_ef_ci`, in `ef_calibrate`). The CIs make the story
   defensible — GE bias CI [−5.3, −1.1] excludes 0 (residual OOD bias real), Canon MAE CI [2.2, 9.7]
   exposes n=9 as underpowered (nnU-Net's 2.6 falls *inside* it → "indistinguishable at n=9", not worse);
   GE gap vs nnU-Net (4.3 below our [6.1, 8.9]) is real + owned. See
   `interpretations/ef/2026-07-15_ef_defensibility.md` §4. Still open: 5-fold retrain CV (`bd …-4ev`,
   adds training variance); UQ / calibration flags (`bd …-iq7`).

## Synthesis thread — status
Design in [`PLAN.md`](PLAN.md) / [`GENERATION.md`](../core/data/dynamic/GENERATION.md). Forward engine
(SAMPLE) is fully built; the composite sources and the inverse (FIT) are the live frontier.
- ✅ **Physics painter** — bSSFP from tissue params, swept acquisition, corruption chain; whole-FOV bg
  (flat / procedural / partition / hybrid strategies).
- ✅ **Tier-1 augmentation of real images** — **concluded, negative**: bias-field aug *regressed* the
  unseen-vendor gap; 4-seed ensemble shows only ~15–18% reducible headroom → the gap is aleatoric +
  model-class. ([normalization README](../cardioseg/preprocessing/normalization/README.md))
- ✅ **SSM anatomy source** (Rodero) — voxelizer + 1000-mesh pool; zero-real generation ~0.56
  cross-vendor (probe's "walled" verdict was wrong once confounds fixed; **ceiling re-confirmed 0.559**,
  2-seed, post-bugfix). Physical **inflow** is default, but its earlier "+0.054 / RV +0.095" was measured
  with the blood-velocity lever DEAD (v stuck [0,1) cm/s, fixed `cab7326`) — **retracted**. Re-verified
  post-fix (bd mdem): the effect is small (~+0.02, at the noise floor), and the gain is in myo+cav, not
  RV. Cine blood IS inflow-enhanced (keep it), but it isn't the headline lever it looked.
- ✅ **Myo weak-link diagnosed** (`bd b6tb`) — geometry ruled out (synth wall thickness matches real,
  3.82 vs 4.21 px); it's a contrast/**separability** gap (within-slice myo|blood d′ 0.54 vs real 2.67).
  Brightness levers are d′-invariant (variance scales with signal) — why every cheap lever was
  Dice-neutral. Separability is *learnable* (net gets myo at .50), so the gap-to-real is **fidelity,
  not a bug**; chasing it down trades away the DR breadth generalization needs. Not knob-fixable.
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
- **Gate 2 — EF you could defend** ✅ — the three criteria are met: cross-dataset EF closed
  (calibration transfers OOD), eval rigor added (bootstrap 95% CI on every EF MAE/bias), honest gap
  writeup done (`interpretations/ef/2026-07-15_ef_defensibility.md`). Optional deepening remains (5-fold
  retrain CV, UQ flags — `bd …-4ev`/`…-iq7`), but the defensibility bar (a number with a CI + owned gap)
  is cleared.
- **Gate 3 — CT lane** ⬜ (MM-WHS, reuse pipeline).
- **Gate 4 — echo lane** ⬜ (CAMUS, Simpson's biplane).

---
*Build log = git history. Theory writeups → `learning/`; research → `research/`.*
