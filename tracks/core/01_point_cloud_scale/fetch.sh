#!/usr/bin/env bash
# Download ModelNet40 from Princeton and extract for the point-cloud scale
# benchmark.  ~440 MB zip; extraction produces ~400 MB of .off mesh files.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_ROOT/data/core_01_point_cloud"

mkdir -p "$DATA_DIR"

ZIP="$DATA_DIR/ModelNet40.zip"
EXTRACT_DIR="$DATA_DIR/ModelNet40"

# Download only if not already present
if [[ -s "$ZIP" ]]; then
    echo "[c1-fetch] cached $ZIP"
else
    echo "[c1-fetch] downloading ModelNet40.zip (~440 MB) ..."
    curl -sSL -o "$ZIP" "https://modelnet.cs.princeton.edu/ModelNet40.zip"
    echo "[c1-fetch] download complete: $(du -sh "$ZIP" | cut -f1)"
fi

# Extract only if target directory doesn't exist yet
if [[ -d "$EXTRACT_DIR" ]]; then
    echo "[c1-fetch] already extracted at $EXTRACT_DIR"
else
    echo "[c1-fetch] extracting ..."
    unzip -q "$ZIP" -d "$DATA_DIR"
    echo "[c1-fetch] extraction complete."
fi

# Verify at least 10 .off files per target class
echo "[c1-fetch] verifying target classes ..."
for cls in airplane car lamp table sofa; do
    count=$(find "$EXTRACT_DIR/$cls/train" -name "*.off" 2>/dev/null | wc -l)
    if [[ "$count" -lt 10 ]]; then
        echo "[c1-fetch] ERROR: class '$cls' has only $count .off files (expected >=10)" >&2
        exit 1
    fi
    echo "[c1-fetch]   $cls/train: $count .off files  OK"
done

echo "[c1-fetch] done."
