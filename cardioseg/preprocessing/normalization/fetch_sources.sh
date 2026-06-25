#!/usr/bin/env bash
# WEB tier of the normalization scheme: pull the PUBLIC dataset-info sources into
# <data>/raw/<ds>/meta/sources/, and list the paywalled/register-gated ones you must fetch yourself.
# Source-only — pulled files live out-of-repo (gitignored data dir). Run in the cardioseg env:
#   bash cardioseg/normalization/fetch_sources.sh
set -e
DATA=$(python -c "from cardioseg.config import data_root; print(data_root('raw'))")
echo "data root: $DATA"

pull() {  # pull <dataset> <url> <outfile>
  local ds="$1" url="$2" out="$3"
  mkdir -p "$DATA/$ds/meta/sources"
  echo "  $ds <- $url"
  curl -fsSL "$url" -o "$DATA/$ds/meta/sources/$out" || echo "    (failed — fetch manually)"
}

# --- public challenge pages (acquisition descriptions) ---
pull acdc  "https://www.creatis.insa-lyon.fr/Challenge/acdc/databasesTraining.html" acdc_challenge.html
pull mnm2  "https://www.ub.edu/mnms-2/"                                              mnm2_challenge.html
pull mnms1 "https://www.ub.edu/mnms/"                                               mnms1_challenge.html

# --- paywalled / register-gated: pull yourself (manifest) ---
mkdir -p "$DATA"
cat > "$DATA/SOURCES_MANUAL.md" <<'MAN'
# Sources to fetch manually (paywalled / register-gated; not auto-pullable)
- ACDC data + Info.cfg sidecars : register @ creatis.insa-lyon.fr/Challenge/acdc
- M&M-2 + dataset_information.csv : register @ ub.edu/mnms-2
- M&Ms-1 + CSV (demographics)    : register @ ub.edu/mnms
- ACDC paper (vendor / field)    : Bernard et al. 2018, IEEE TMI (paywalled) -> cardioseg/normalization/sources.yaml
- M&Ms paper (per-vendor field)  : Campello et al. 2021, IEEE TMI (paywalled) -> sources.yaml (field_T still unverified)
MAN
echo "wrote $DATA/SOURCES_MANUAL.md"
echo "done. persist parsed+paper meta with:  python -m cardioseg.normalization.persist"
