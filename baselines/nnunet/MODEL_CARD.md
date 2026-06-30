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

- **Train+val (nnU-Net does its own CV):** Siemens+Philips from M&M-2+M&Ms-1 — **495 labelled
  subjects** (GE excluded from train).
- **Val:** ACDC-150 (centre/protocol shift; used by the main model for early stopping + calibration —
  not a nnU-Net test axis here).
- **Held out (test):** Canon-9 + GE-69 = **78 subjects** (two fully unseen vendors). Per-case axis
  tagged in `ts_manifest.json`; scored per axis by `baselines/nnunet/score.py`.

## Performance (scored by cardioseg.evaluation, same yardstick)

**Unseen-vendor test — Canon (n=9):**

| structure | Dice | HD95 (mm) |
|---|---|---|
| LV cavity | 0.892 | 5.9 |
| LV myocardium | 0.829 | 8.1 |
| RV cavity | 0.878 | 6.8 |
| **mean** | **0.866** | |

EF vs GT: **MAE 2.6%**, bias −1.4%, 95% LoA [−8.2, +5.4] (n=9).

**Unseen-vendor test — GE (n=69):**

| structure | Dice | HD95 (mm) |
|---|---|---|
| LV cavity | 0.914 | 5.1 |
| LV myocardium | 0.842 | 4.7 |
| RV cavity | 0.877 | 5.8 |
| **mean** | **0.878** | |

EF vs GT: **MAE 4.3%**, bias +0.9%, 95% LoA [−11.7, +13.5] (n=69).

## vs the cardioseg 2D U-Net (same split, same eval)

<!-- results:cardcompare -->
| unseen-vendor (held out) | nnU-Net (50ep/fold0) | this model |
|---|---|---|
| Canon mean Dice | 0.866 | 0.836 |
| GE mean Dice | 0.878 | 0.838 |
| Canon / GE EF MAE | **2.6 / 4.3%** | 12.1 / 11.5% |
<!-- /results:cardcompare -->

Both pool ED+ES. nnU-Net leads by **~3–4 Dice points** on unseen-vendor Canon (0.866 vs 0.836) and
GE (0.878 vs 0.838), even at this floor setting. **EF gap is dramatic** — nnU-Net Canon 2.6% vs
cardioseg 12.1%; GE 4.3% vs 11.5%. The EF gap is model-class epistemic (reducible by a stronger
model): a stronger segmenter substantially closes it. cardioseg trades that for ONNX portability at
~57× fewer parameters.

- **One offset:** cardioseg's boundary is *tighter* on LV-cav HD95 (2.1 vs 5.9 mm Canon) — from its
  largest-CC + TTA postprocessing, which **this nnU-Net run did not apply**. Read Dice/EF as the fair
  head-to-head; HD95 gap is a postproc artifact.

## Limitations
- **Floor, not ceiling** (50ep/1fold/no-ensemble) — the true nnU-Net SOTA bar is higher (`yp3`).
- **No postprocessing/TTA** in this run → inflated boundary metrics relative to a tuned nnU-Net.
- Canon **n=9** is thin (M&Ms-1 withholds most Testing GT); Dice signal is real but EF at n=9 is
  noisy. GE n=69 is the reliable unseen-vendor leg.
- Same data-distribution caveats as the main model: public-benchmark performance ≠ deployment.

## Reproduce
```bash
pip install -e .[nnunet]
python -m baselines.nnunet.convert --id 29            # build Dataset029_BATTERY from the store
conda run -n <env> bash baselines/nnunet/run_battery.sh   # preprocess -> train -> predict -> score
```
Numbers above: `Dataset029_BATTERY`, fold 0, `nnUNetTrainer_50epochs`, scored via `score.py --manifest`.
