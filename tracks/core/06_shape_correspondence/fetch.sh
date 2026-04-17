#!/usr/bin/env bash
# Download and extract the TACO shape-correspondence dataset (Zenodo 14066437).
# ~120 MB zip → ~600 MB extracted (80 OFF meshes + 420 GT correspondence .mat).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_ROOT/data/core_06_shape/taco"

if [[ -d "$DATA_DIR/offs" && -d "$DATA_DIR/gt_matches" ]]; then
    echo "[c6-fetch] Already present: $DATA_DIR"
    ls "$DATA_DIR/offs" | wc -l | awk '{print "[c6-fetch]   "$1" OFF meshes"}'
    ls "$DATA_DIR/gt_matches" | wc -l | awk '{print "[c6-fetch]   "$1" GT files"}'
    exit 0
fi

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

URL="https://zenodo.org/api/records/14066437/files/taco-dataset.zip/content"
echo "[c6-fetch] Downloading TACO from Zenodo..."
curl -sL --connect-timeout 30 -o taco-dataset.zip "$URL"

echo "[c6-fetch] Extracting..."
unzip -q taco-dataset.zip
rm -f taco-dataset.zip

echo "[c6-fetch] Done. Installed to $DATA_DIR"
