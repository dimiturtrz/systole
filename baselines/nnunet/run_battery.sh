#!/usr/bin/env bash
# nnU-Net battery baseline: preprocess -> train (50ep/fold0) -> predict -> per-axis score.
# Run inside the cardioseg env:  uv run bash baselines/nnunet/run_battery.sh
# (convert first: python -m baselines.nnunet.convert --id 29)
set -e

# config-derived nnU-Net dirs under <data>/nnunet/ (never the data root) — see env.sh
source "$(dirname "$0")/env.sh"
DS="$nnUNet_raw/Dataset029_BATTERY"
PRED="$NN/pred/Dataset029"

PREP="$nnUNet_preprocessed/Dataset029_BATTERY"
# Always re-train + re-predict fresh (the actual baseline work). But PREPROCESS only when the split
# changed — it's I/O-heavy (~12min on a 9p/WSL mount) and pointless to redo for identical data.
rm -rf "$nnUNet_results/Dataset029_BATTERY" "$PRED"
# Fingerprint the raw dataset (case lists + dataset.json). This is what makes skipping SAFE: a
# different split -> different cases -> mismatch -> rebuild, so a prior split's splits_final.json
# can't leak (the original reason this wiped unconditionally). Force with FORCE_PREPROCESS=1.
FP="$PREP/.battery_fingerprint"
NEWFP=$( (cat "$DS/dataset.json"; ls "$DS/imagesTr" "$DS/imagesTs" | sort) | sha1sum | cut -d' ' -f1 )
if [ "${FORCE_PREPROCESS:-0}" != "1" ] && [ -f "$FP" ] && [ "$(cat "$FP" 2>/dev/null)" = "$NEWFP" ]; then
    echo "=== preprocessed cache matches this split (fingerprint $NEWFP) — skip plan+preprocess ==="
else
    echo "=== split changed/absent -> clean + plan + preprocess ==="
    rm -rf "$PREP"
    nnUNetv2_plan_and_preprocess -d 29 --verify_dataset_integrity
    echo "$NEWFP" > "$FP"
fi

echo "=== train (2d, fold 0, 50 epochs) ==="
nnUNetv2_train 29 2d 0 -tr nnUNetTrainer_50epochs

echo "=== predict held-out Ts (unseen vendors: Canon + GE) ==="
nnUNetv2_predict -i "$DS/imagesTs" -o "$PRED" -d 29 -c 2d -f 0 -tr nnUNetTrainer_50epochs

echo "=== score (per battery axis) ==="
cd "$(cd "$(dirname "$0")/../.." && pwd)"   # repo root, portable (git-bash /d, WSL /mnt/d, native linux)
# SCORE_OUT overridable so cross-platform verification runs don't clobber the committed results.json.
python -m baselines.nnunet.score --pred "$PRED" --gt "$DS/labelsTs" --manifest "$DS/ts_manifest.json" \
    --out "${SCORE_OUT:-baselines/nnunet/results.json}"   # single source read by cardioseg/evaluation/results.py
echo "=== NNUNET BATTERY DONE ==="
