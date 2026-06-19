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
export nnUNet_raw=D:/data/nnUNet_raw nnUNet_preprocessed=D:/data/nnUNet_pre nnUNet_results=D:/data/nnUNet_res

# 1. our data -> nnU-Net format (uses cardioseg loaders; ED/ES -> one case each)
python -m baselines.nnunet.convert --dataset acdc --out $nnUNet_raw --id 27

# 2. nnU-Net does the rest (its recipe: fingerprint -> plan -> train 5 folds)
nnUNetv2_plan_and_preprocess -d 27 --verify_dataset_integrity
nnUNetv2_train 27 2d 0   # ... folds 0..4 for the ensemble
nnUNetv2_predict -i <held_out_images> -o <pred_dir> -d 27 -c 2d

# 3. score nnU-Net's masks with OUR eval layer (the bridge back)
python -m baselines.nnunet.score --pred <pred_dir> --gt $nnUNet_raw/Dataset027_ACDC/labelsTr
```
Data + `nnUNet_raw/preprocessed/results` all live **outside the repo** (under `D:/data`),
same as the rest — nothing heavy gets committed.

## Published ceiling (context, before we even run it)
nnU-Net on ACDC is a solved benchmark — top-of-leaderboard, **mean Dice ~0.91**
(LV-cav ~0.93–0.96, myo ~0.88–0.92, RV ~0.88–0.92). So this baseline mostly
*confirms* a known ceiling and gives the "I can operate it + score it through my own
pipeline" credential — it is **not** a discovery.

## Results — M&M-2 → ACDC (the project's generalization setup)
Both trained on multi-vendor M&M-2, tested on the held-out single-centre ACDC (100
patients / 200 frames), **scored by `cardioseg.evaluation`** — apples-to-apples.

| segmenter | mean Dice | LV-cav | myo | RV | EF MAE | notes |
|---|---|---|---|---|---|---|
| our 2D U-Net (+ largest-CC + TTA) | 0.885 | 0.939 | 0.855 | 0.862 | 7.9% | deployable / ONNX |
| **nnU-Net** (50 ep, 1 fold) | **0.909** | 0.947 | 0.871 | **0.908** | **5.5%** | baseline / not deployed |
| gain | +2.4 | +0.8 | +1.6 | **+4.6** | **−2.4** | |

**EF agreement (Bland–Altman):** ours bias −7.3%, 95% LoA [−34, +19]; nnU-Net bias **−4.1%**,
LoA **[−17.8, +9.7]** — roughly **half the spread** and less bias. Both still *underpredict*
(negative bias), so part of the cross-domain EF shift is intrinsic (calibration), not just model
quality — but nnU-Net's tighter masks cut the random error a lot.

**Read:** nnU-Net wins everything at only **50 epochs / 1 fold** (its floor — the full
1000-epoch × 5-fold + TTA recipe goes higher). Biggest gains where it matters most:
**RV +4.6** (the thin, domain-fragile structure the simple model is weakest on) and
**EF MAE 7.9 → 5.5%** (better masks cut the systematic volume bias, so the *clinical
number* improves, not just Dice). It hits **0.909 on ACDC trained on M&M-2** — near the
in-domain ceiling (~0.91), cross-domain. (Both rows are each model's deployable output
scored by the same eval; ours includes largest-CC + TTA, nnU-Net its own.)

*The gap is the honest price of a deployable, ONNX-exportable, fully-understood model —
and it names the remaining levers to close it on our clean U-Net while staying exportable
(TTA already applied): **instance norm, finer target spacing (1.23 mm), heavier augmentation,
longer training.** The baseline is a roadmap, not just a credential.*
