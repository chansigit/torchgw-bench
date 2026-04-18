#!/usr/bin/env bash
# C2 benchmark with LDA ATAC preprocessing — does it close the gap to
# SCOT+ literature (FOSCTTM = 0.12)?  Fixed ε = 5e-3 (sweet spot from
# the ε-sweep). 5 solvers × N ∈ {1000, 2000, 5000} × 3 seeds.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_PY="$REPO_ROOT/tracks/core/02_single_cell_omics/run.py"
OUT="$REPO_ROOT/results/c2_sc_lda"
mkdir -p "$OUT"

SOLVERS=(
    "torchgw-landmark"
    "torchgw-dijkstra"
    "torchgw-precomputed"
    "pot-entropic-gpu"
    "pot-exact-gpu"
)
N_CELLS=(1000 2000 5000)
SEEDS=(0 1 2)
EPS=5e-3

run_cell () {
    local solver="$1"; local n="$2"; local seed="$3"
    local out_json="$OUT/core_02_single_cell_omics__${solver}__n${n}__seed${seed}.json"
    if [[ -s "$out_json" ]]; then
        local ok
        ok="$(python -c "
import json
try:
    d=json.load(open('$out_json'))
    print('1' if d.get('status') != 'fail' and d.get('metrics',{}).get('task') else '0')
except Exception: print('0')")"
        [[ "$ok" == "1" ]] && { echo "[c2-lda] cached $solver N=$n seed=$seed"; return 0; }
    fi
    echo "[c2-lda] $solver N=$n seed=$seed"
    # pot-exact has no epsilon
    local eps_flag="--epsilon $EPS"
    if [[ "$solver" == "pot-exact-gpu" ]]; then eps_flag=""; fi
    python "$RUN_PY" --solver "$solver" --seed "$seed" \
        --n-cells "$n" --out "$OUT" --atac-method lda $eps_flag \
        2>&1 | grep -E "^\[C2\]" || true
}

for seed in "${SEEDS[@]}"; do
    for n in "${N_CELLS[@]}"; do
        for solver in "${SOLVERS[@]}"; do
            run_cell "$solver" "$n" "$seed"
        done
    done
done
echo "[c2-lda] Done. Results in $OUT"
