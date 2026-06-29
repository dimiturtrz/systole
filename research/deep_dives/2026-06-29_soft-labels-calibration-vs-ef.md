# Soft-label segmentation: a calibration win, not an EF fix (2026-06-29)

**Question.** Boundary voxels are partial-volume *mixes*; hard 0/1 labels are a lie. Does training on
**soft (probabilistic) boundary targets** — the honest representation — do better? Specifically (a)
calibration, (b) EF (the standing weakness)?

**Principle that shaped it.** The loss stays a correct, balanced, *general* objective — never fit to a
specific need (rejected the `7oe` volume-penalty term on this ground). Soft labels qualify: a *more
honest target*, uniform over all boundaries, not a symptom patch. Fix via better labels, not loss
engineering. Keep the representation soft as far down the pipeline as possible; collapse to hard only
at the leaf (Dice/mask), read **volume from the soft probabilities** (expected volume, collapse-never).

## Setup
- `soften()`: one-hot → per-class Gaussian blur (σ voxels) → renormalize. Soft at boundaries, hard in
  interiors. σ a principled boundary-uncertainty width, not tuned to EF. (`AugCfg.soft_label_sigma`)
- `SoftDiceCE`: soft Dice + soft-CE = −(target·log_softmax).sum(C).mean(). MONAI DiceCELoss can't be
  used — its CE term argmaxes the target, collapsing soft labels.
- Trained the flagship recipe with σ=1.0 → `runs/soft`. Eval on the ACDC val split (n=150), GT is hard.
- `expected_volume_ml` = Σ blood-prob within the largest-CC gate (the late-collapse EF readout).

## Result (ACDC val, n=150)
| metric | `gen` (hard) | `soft` (σ=1.0) | Δ |
|---|---|---|---|
| ECE | 0.0925 | **0.0806** | **−13% (better calibrated)** |
| Dice mean | 0.9022 | 0.9000 | −0.2% (noise) |
| EF MAE (hard readout) | 6.50% | 6.47% | ~0 |
| EF bias (hard readout) | −5.6% | −5.9% | ~0 |
| EF bias (expected-vol readout) | −4.7% | −5.0% | ~0 |

## Reads
1. **Soft labels improve calibration** (ECE −13%) at **equal Dice and equal EF**. The overconfident
   model (temp T≈2.49) becomes better-calibrated — provable, the headline win. Matches the principle:
   honest targets, clean loss.
2. **Soft labels do NOT fix EF.** Soft *training* left the EF bias unchanged. The boundary it learns is
   still ~hard, and EF is bounded by **resolution** (a 20 mL ES cavity on 10 mm slices = 2–3 slices →
   partial-volume the labels can't encode), not by the label representation. This is the **5th
   independent angle** all landing on resolution/model-class as the EF wall (after: aug, nnU-Net
   rebaseline, CMRxMotion, data-balance measurement).
3. **Expected-volume is a small free readout gain (~1 pp EF bias) on ANY model** — the CC-gated soft sum
   counts boundary voxels as fractions, shrinking the over-filled ES cavity more than ED. Measured *vs
   GT* → a real, verifiable improvement (not the unprovable-vs-true-volume kind). Earlier I wrongly
   called this "unverifiable" — agreement-with-GT *is* measurable, and it improves.

## Verdict
Soft labels = a genuine, honest **calibration** improvement, shippable as a validated option
(`soft_label_sigma`); not an accuracy or EF upgrade. EF remains resolution-bound — the only lever that
moves it is finer-spacing / 3D (model-class), now triangulated five ways. Expected-volume is worth
adopting as the default EF readout (principled, free ~1 pp).

EF levers ledger (all considered, all bounded):
| lever | verdict |
|---|---|
| volume-penalty loss (`7oe`) | rejected on principle (don't fit loss to need) |
| data-balance | limited (small-ES-cavity physiologically rare + 10 mm floor) |
| partial-volume correction | unverifiable vs truth (GT also PV-limited) |
| soft-label training | calibration ✓, EF null (this doc) |
| expected-volume readout | small free EF gain (~1 pp), adopt |
| **finer-spacing / 3D (model-class)** | **the only thing that lifts the ceiling — proven by nnU-Net** |
