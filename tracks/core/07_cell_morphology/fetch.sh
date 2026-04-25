#!/usr/bin/env bash
# C7 — download SWC files for cells listed in stage_{a,b}_manifest.txt.
# Manifests are pinned cell-ID lists; populate them once via
# `build_manifests.py` before running this script.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_ROOT/data/core_07_cell_morphology"
mkdir -p "$DATA_DIR/swc/stage_a" "$DATA_DIR/swc/stage_b"

fetch_neuromorpho() {
    local manifest="$1" outdir="$2"
    while IFS=$'\t' read -r name cls archive; do
        [[ "$name" == "neuron_name" || -z "$name" || "$name" == \#* ]] && continue
        local out="$outdir/${name}.swc"
        [[ -s "$out" ]] && continue
        # SWC URL uses lowercased archive directory under dableFiles/
        local archive_lc="$(echo "$archive" | tr '[:upper:]' '[:lower:]')"
        if ! curl -fsSL --retry 3 --max-time 60 -o "$out" \
                "https://neuromorpho.org/dableFiles/${archive_lc}/CNG%20version/${name}.CNG.swc"; then
            echo "[c7-fetch] WARN: missing $name (archive=$archive)"; rm -f "$out"
        fi
    done < "$manifest"
}

fetch_allen() {
    # Allen requires resolving each specimen_id → well_known_file_id via
    # the Specimen JSON include — the simpler endpoint returns 404. Done
    # in fetch_allen.py with thread pool for speed (~10 min for ~700).
    env -u PYTHONPATH micromamba run -n c7_morph python \
        "$SCRIPT_DIR/fetch_allen.py"
}

fetch_neuromorpho "$SCRIPT_DIR/stage_a_manifest.txt" "$DATA_DIR/swc/stage_a"
fetch_allen       "$SCRIPT_DIR/stage_b_manifest.txt" "$DATA_DIR/swc/stage_b"

# 80 % manifest-survival sanity gate
for stage in stage_a stage_b; do
    expected=$(grep -cv -E '^(neuron_name|specimen_id|#|$)' \
               "$SCRIPT_DIR/${stage}_manifest.txt")
    actual=$(ls "$DATA_DIR/swc/$stage"/*.swc 2>/dev/null | wc -l)
    threshold=$(( expected * 80 / 100 ))
    if (( expected > 0 && actual < threshold )); then
        echo "[c7-fetch] ERROR: $stage has $actual/$expected (need >= $threshold)"
        exit 1
    fi
done
echo "[c7-fetch] done."
