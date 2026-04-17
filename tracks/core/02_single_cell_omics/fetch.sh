#!/usr/bin/env bash
# Download 10x PBMC 10k Multiome (ATAC + GEX from same cells) —
# a standard cross-modality alignment benchmark. ~200 MB.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_ROOT/data/core_02_sc_omics"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"
if [[ -s pbmc_10k_multiome.h5 ]]; then
    echo "[c2-fetch] Already present: $DATA_DIR/pbmc_10k_multiome.h5"
    exit 0
fi
URL="https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5"
echo "[c2-fetch] Downloading PBMC 10k Multiome (~200 MB)..."
curl -sSL -o pbmc_10k_multiome.h5 "$URL"
echo "[c2-fetch] Done. $DATA_DIR/pbmc_10k_multiome.h5"
