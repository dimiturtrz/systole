#!/usr/bin/env bash
# nnU-Net battery baseline: preprocess -> train (50ep/fold0) -> predict -> per-axis score.
# Run inside the cardioseg env:  conda run -n pytorch_training_env bash baselines/nnunet/run_battery.sh
# (convert first: python -m baselines.nnunet.convert --id 29)
set -e

# config-derived nnU-Net dirs under <data>/nnunet/ (never the data root) — see env.sh
source "$(dirname "$0")/env.sh"
DS="$nnUNet_raw/Dataset029_BATTERY"
PRED="$NN/pred/Dataset029"

echo "=== plan + preprocess ==="
nnUNetv2_plan_and_preprocess -d 29 --verify_dataset_integrity

echo "=== train (2d, fold 0, 50 epochs) ==="
nnUNetv2_train 29 2d 0 -tr nnUNetTrainer_50epochs

echo "=== predict held-out Ts (acdc + canon) ==="
nnUNetv2_predict -i "$DS/imagesTs" -o "$PRED" -d 29 -c 2d -f 0 -tr nnUNetTrainer_50epochs

echo "=== score (per battery axis) ==="
cd /d/personal_projects/cardiac-seg
python -m baselines.nnunet.score --pred "$PRED" --gt "$DS/labelsTs" --manifest "$DS/ts_manifest.json"
echo "=== NNUNET BATTERY DONE ==="
