# Model Card — nnU-Net baseline (cardiac short-axis, generalization split)

An external **SOTA reference** for the cardioseg 2D U-Net: nnU-Net v2 trained on the *same*
generalization split and scored by the *same* eval layer, so the comparison is apples-to-apples.
This is a benchmark/reference model, not the project's shipped model — see
[`cardioseg/MODEL_CARD.md`](../../cardioseg/MODEL_CARD.md) for that.

## Model details
- **Framework:** nnU-Net v2 (2.8.0), config `2d`, **fold 0**, trainer `nnUNetTrainer_50epochs`.
- **Self-configured:** nnU-Net plans its own architecture, resampling, and **per-image z-score
  normalization** from the dataset fingerprint — fed RAW volumes (via cardioseg adapters), not our
  preprocessed store, so it runs its full native pipeline.
- **Dataset:** `Dataset029_BATTERY` (built by `baselines/nnunet/convert.py`).
- **This is a FLOOR, not the ceiling:** 50 epochs / single fold / no ensemble. nnU-Net's full recipe
  (1000 epochs × 5-fold + TTA) would score higher (`cardiac-seg-yp3`).

## Intended use
- External SOTA reference point to position the cardioseg model honestly.
- **Out of scope:** not clinical, not a device — public benchmark data only (same envelope as the
  main model). Not the artifact the web demo / pipeline uses.

## Training & evaluation data

> **Provisional:** this run used the **old split** — ACDC as the held-out test axis, 564-subject
> train (M&M-2 + M&Ms-1 ex-Canon, all four vendors including GE). The current split uses ACDC as
> val (early stopping + calibration), Canon+GE as the unseen-vendor test, and narrows training to
> Siemens+Philips (495 subjects). Re-run on the new split is pending.

Split as run (old configuration):
- **Train+val (nnU-Net does its own CV):** M&M-2 + M&Ms-1 ex-Canon, labelled — 564 subjects / 1128
  ED+ES cases (all four vendors: Siemens, Philips, GE; Canon excluded).
- **Held out:** ACDC-150 (centre/protocol shift) + Canon-9 (unseen vendor) — 159 subjects / 318
  cases. Per-case axis tagged in `ts_manifest.json`; scored per axis by `baselines/nnunet/score.py`.

## Performance (scored by cardioseg.evaluation, same yardstick)

**ACDC-150 (centre/protocol shift):**

| structure | Dice | HD95 (mm) |
|---|---|---|
| LV cavity | 0.948 | 3.3 |
| LV myocardium | 0.876 | 2.9 |
| RV cavity | 0.911 | 5.1 |
| **mean** | **0.912** | |

EF vs GT: **MAE 5.6%**, bias −4.2%, 95% LoA [−19.3, +10.8] (n=150).

**Canon-9 (unseen vendor, old split):** mean Dice **0.876**; EF MAE 2.3% but **n=9 → not a reliable
EF number** (M&Ms-1 withholds most Testing GT; only 9 Canon cases are labelled). GE was in training
on the old split — it moves to the unseen-vendor test on the new split (pending re-run).

**Pooled (159):** mean Dice 0.907, EF MAE 5.4%, bias −4.1%.

## vs the cardioseg 2D U-Net (same split, same eval)

<!-- results:cardcompare -->
| ACDC-150 | nnU-Net (50ep/fold0) | this model |
|---|---|---|
| mean Dice | 0.912 | 0.884 |
| EF MAE / bias | 5.6% / -4.2% | 6.5% / -5.6% |
| Canon-9 Dice | 0.876 | 0.84 |
<!-- /results:cardcompare -->

- Both pool ED+ES. nnU-Net leads by **~2.8 Dice points** on ACDC (0.912 vs 0.884) and on unseen-vendor
  Canon (0.876 vs 0.84), even at this floor setting. **EF is roughly level** (5.6% vs 6.5%). cardioseg
  is ~2–3 Dice points under the nnU-Net floor at ~57× fewer parameters.
- **One offset:** cardioseg's boundary is *tighter* (LV-cav HD95 2.1 vs 3.3 mm, myo 2.1 vs 2.9) — from its
  largest-CC + TTA postprocessing, which **this nnU-Net run did not apply**. So the HD95 gap is a
  postproc artifact, not a modelling result; read Dice/EF as the fair head-to-head.

## Limitations
- **Floor, not ceiling** (50ep/1fold/no-ensemble) — the true nnU-Net SOTA bar is higher (`yp3`).
- **No postprocessing/TTA** in this run → inflated boundary metrics relative to a tuned nnU-Net.
- Unseen-vendor test on old split: Canon **n=9** only — Dice signal only, EF is noise. New split adds GE n=69 as a second unseen-vendor leg (pending re-run).
- Same data-distribution caveats as the main model: public-benchmark performance ≠ deployment.

## Reproduce
```bash
pip install -e .[nnunet]
python -m baselines.nnunet.convert --id 29            # build Dataset029_BATTERY from the store
conda run -n <env> bash baselines/nnunet/run_battery.sh   # preprocess -> train -> predict -> score
```
Numbers above: `Dataset029_BATTERY`, fold 0, `nnUNetTrainer_50epochs`, scored via `score.py --manifest`.
