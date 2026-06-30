#!/usr/bin/env bash
# Export nnU-Net's three dirs under the cardiac-data namespace (<data>/nnunet/), DERIVED from
# cardioseg's path config (paths.yaml / CARDIAC_DATA) — so artifacts land in the namespace on any
# machine and never dump at the data root. Source it inside the cardioseg conda env:
#   source baselines/nnunet/env.sh
NN=$(python -c "from core.config import data_root; print(data_root('nnunet'))")
export NN
export nnUNet_raw="$NN/raw"
export nnUNet_preprocessed="$NN/preprocessed"
export nnUNet_results="$NN/results"
echo "nnU-Net namespace: $NN"
