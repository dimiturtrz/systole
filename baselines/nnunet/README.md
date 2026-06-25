# nnU-Net baseline (quarantined)

The **SOTA reference**, kept deliberately at arm's length. nnU-Net is the
self-configuring default-to-beat for medical segmentation — so we run it as a
*baseline*, not a dependency.

**Why it lives here, not in `cardioseg/`:** nnU-Net owns its own world (its
`nnUNet_raw` data format, preprocessing, training, 5-fold ensemble). Folding it
into the pipeline would drag a duplicate dataset + a heavy framework through the
clean modular code. So it's isolated: **its framework + deps never enter
`cardioseg`'s.** The only bridges are two thin scripts:

- `convert.py` — our data → nnU-Net's `nnUNet_raw` format (reuses cardioseg loaders).
- `score.py` — nnU-Net's predicted masks → **cardioseg's eval layer** (same Dice /
  HD95 / EF we score our own model with).

That's the architecture in one folder: **the segmenter is a commodity; the
measurement + evaluation is the value, and it's the same regardless of segmenter.**
Our own clean 2D U-Net stays the *deployable* model (it ONNX-exports in one line for
cardioview's in-browser inference — nnU-Net can't, cleanly). nnU-Net is here for the
ceiling number, not for shipping.

## Run flow
nnU-Net is installed in a **separate env** (not a cardioseg dependency):
```bash
pip install nnunetv2          # its own env
# nnU-Net's three dirs, derived from cardioseg's path config -> <data>/nnunet/ (never the data
# root, machine-independent). Exports nnUNet_raw / _preprocessed / _results:
source baselines/nnunet/env.sh

# 1. our data -> nnU-Net format: builds Dataset029_BATTERY (train+val pool Tr, held-out acdc+canon Ts)
python -m baselines.nnunet.convert --id 29

# 2.+3. preprocess -> train (50ep/fold0) -> predict held-out -> score via cardioseg eval:
bash baselines/nnunet/run_battery.sh
#   step-by-step equivalent:
#   nnUNetv2_plan_and_preprocess -d 29 --verify_dataset_integrity
#   nnUNetv2_train 29 2d 0                       # ... folds 0..4 for the full ensemble
#   nnUNetv2_predict -i $nnUNet_raw/Dataset029_BATTERY/imagesTs -o <pred_dir> -d 29 -c 2d
#   python -m baselines.nnunet.score --pred <pred_dir> --manifest <.../ts_manifest.json> \
#       --out baselines/nnunet/results.json   # single source read by cardioseg/evaluation/results.py
```
Data + `raw/preprocessed/results/pred` all live **outside the repo**, namespaced under
`<data>/nnunet/` (derived from `paths.yaml` / `CARDIAC_DATA` via `cardioseg.config`) — nothing
heavy gets committed, and nnU-Net's dirs default into the namespace, never the data root.

## Published ceiling (context, before we even run it)
nnU-Net on ACDC is a solved benchmark — top-of-leaderboard, **mean Dice ~0.91**
(LV-cav ~0.93–0.96, myo ~0.88–0.92, RV ~0.88–0.92). So this baseline mostly
*confirms* a known ceiling and gives the "I can operate it + score it through my own
pipeline" credential — it is **not** a discovery.

## Results — generalization split (ACDC-150 held-out axis)
> **Provisional:** nnU-Net was trained on the **old split** (ACDC as test axis, 564-subject train —
> M&M-2 + M&Ms-1 ex-Canon, all four vendors). The current split uses ACDC as val (early stopping +
> calibration), Canon+GE as the unseen-vendor test, and narrows training to Siemens+Philips (495
> subjects). Re-run on the new split is pending — numbers below reflect the old configuration.

Both trained on the **same pooled cloud** (M&M-2 + M&Ms-1 ex-Canon, 564 labelled), tested on the
held-out **ACDC-150** axis, **scored by `cardioseg.evaluation`** — apples-to-apples.

<!-- results:nnucompare -->
| segmenter | mean Dice | LV-cav | myo | RV | EF MAE | notes |
|---|---|---|---|---|---|---|
| our 2D U-Net (+ heavy aug + early stop + largest-CC + TTA) | 0.884 | 0.916 | 0.860 | 0.875 | 6.5% | deployable / ONNX |
| **nnU-Net** (50 ep, 1 fold) | **0.912** | **0.948** | **0.876** | **0.911** | **5.6%** | baseline / not deployed |
| Δ (nnU-Net − ours) | +2.8 | +3.2 | +1.6 | +3.6 | -0.9 | |
<!-- /results:nnucompare -->

<sub>All Dice/HD95 pool ED+ES (both phases) — same yardstick both rows.</sub>

**Efficiency:** ours **1.6 M params / ~0.8 GFLOPs** vs nnU-Net **92 M / ~19 GFLOPs** (single forward,
fvcore; nnU-Net at its 256×320 patch, inference adds tiling + TTA) — **~57× fewer params, ~23× fewer
FLOPs**, for ~2–3 Dice points less on ACDC. That's the deployable trade made quantitative.

**Be frank — this is nnU-Net under-powered, not at full strength.** What we ran vs its real recipe:
- **1 fold**, not the **5-fold ensemble** (the headline nnU-Net result averages 5 models).
- **50 epochs**, not **1000**.
- **2D only** — no `3d_fullres` / `3d_cascade` (it auto-picks the best config; we skipped that).
- no postprocessing/config selection step.

So **0.912 is nnU-Net's floor here**; the full 1000-epoch × 5-fold + TTA recipe would pull further
ahead. Even at this floor it leads by **~2.8 Dice points on ACDC** (0.912 vs 0.884) and on unseen-vendor
Canon (0.876 vs 0.84). **EF is roughly level** (6.5 vs 5.6%). We report the floor honestly and didn't
chase the ceiling — it's a lot more compute for a model we wouldn't deploy anyway (92 M, no clean ONNX).
The baseline's job is *"I can operate + score SOTA through my own pipeline,"* **not** *"I matched it."*

**EF agreement (Bland–Altman):** ours bias −5.6%, 95% LoA [−20.1, +8.9]; nnU-Net bias **−4.2%**,
LoA **[−19.3, +10.8]** — closely matched. Both still *underpredict* (negative bias), so part of
the cross-domain EF shift is intrinsic (calibration), not just model quality.

**Read:** nnU-Net leads ~2–3 Dice points across structures (its main margins on LV-cav +3.0 and RV
+2.9), at **50 epochs / 1 fold** (its floor; the full recipe goes higher). Our largest-CC + TTA still
gives a tighter LV-cav/myo boundary (HD95 2.1 vs 3.3 / 2.9 mm), and the clinical number (EF) is
near-identical (6.5 vs 5.6%). (Both rows are each model's deployable output, ED+ES, same eval.)

*The gap is the honest price of a deployable, ONNX-exportable, fully-understood model —
and it names the remaining levers to close it on our clean U-Net while staying exportable
(TTA already applied): **instance norm, finer target spacing (1.23 mm), heavier augmentation,
longer training.** The baseline is a roadmap, not just a credential.*
