#!/usr/bin/env bash
# C2 M_samples sweep — how does torchgw's per-iter cost-matrix
# sample count affect FOSCTTM at various N? Uses cisTopic
# preprocessing (cached). Fixed ε = 5e-3, max_iter = 300.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_PY="$REPO_ROOT/tracks/core/02_single_cell_omics/run.py"
OUT="$REPO_ROOT/results/c2_msamples"
mkdir -p "$OUT"

# Only torchgw variants use M_samples; precomputed is the clean one
# (takes an external SCOT-style cost matrix; not confounded by
# internal distance-mode choice).
SOLVERS=(torchgw-precomputed)
N_LIST=(2000 5000)
M_LIST=(80 160 320 640 1280 2560 5120)
SEEDS=(0 1 2)

run_cell () {
    local solver="$1"; local n="$2"; local m="$3"; local seed="$4"
    local tag="M${m}"
    local out_json="$OUT/c2_ms__${solver}__n${n}__seed${seed}__${tag}.json"
    if [[ -s "$out_json" ]]; then
        python -c "
import json
try:
    d=json.load(open('$out_json'))
    import sys; sys.exit(0 if d.get('status') != 'fail' and d.get('metrics',{}).get('task') else 1)
except Exception: import sys; sys.exit(1)" && { echo "[c2-ms] cached $solver N=$n M=$m seed=$seed"; return 0; } || true
    fi
    echo "[c2-ms] $solver N=$n M=$m seed=$seed"
    python "$RUN_PY" \
        --solver "$solver" --seed "$seed" \
        --n-cells "$n" --out "$OUT" --atac-method cistopic \
        --epsilon 5e-3 --tag "$tag" \
        2>&1 | grep -E "^\[C2\]" || true
    local src="$OUT/core_02_single_cell_omics__${solver}__n${n}__seed${seed}__${tag}.json"
    [[ -s "$src" ]] && mv "$src" "$out_json" || true
}

# Patch: we need M_samples as a CLI flag; call run.py with --M
# Actually run.py doesn't expose M_samples, so we'll call a small
# Python wrapper instead (in parallel with this shell script).
echo "[c2-ms] shell dispatch unsupported — using Python runner instead"
exec python "$REPO_ROOT/scripts/experiments/run_c2_msamples_sweep.py"
