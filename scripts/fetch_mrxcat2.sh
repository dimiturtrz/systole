#!/usr/bin/env bash
# Fetch the MRXCAT2.0 phantom tool for the OFFLINE mrxcat generation lane (not needed for train/eval).
# Public ETH repo, paper-cited MIT (Buoso et al, JCMR 25:25, 2023). systole only ADAPTS its .vti label
# output (core/data/dynamic/mrxcat.py) — the tool is an EXTERNAL CHECKOUT kept in the gitignored
# external/, never vendored/redistributed. Idempotent; pins a commit for reproducibility.
set -euo pipefail

URL="https://gitlab.ethz.ch/ibt-cmr-public/mrxcat-2.0.git"
PIN="9f396a998f435525b234a304c502238ad5955fb2"
DEST="$(cd "$(dirname "$0")/.." && pwd)/external/mrxcat2"

if [ ! -d "$DEST/.git" ]; then
    echo "cloning MRXCAT2.0 -> $DEST"
    git clone "$URL" "$DEST"
fi
git -C "$DEST" checkout -q "$PIN"
echo "MRXCAT2.0 ready at $DEST (pinned $PIN)"
echo "NB: the full XCAT torso backgrounds are Duke-licensed (obtain separately); the bundled example"
echo "    runs with runXCAT=False / use_texturizer=False and needs no MATLAB/XCAT."
