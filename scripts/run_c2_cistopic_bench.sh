#!/usr/bin/env bash
# C2 benchmark with cisTopic ATAC preprocessing (literature-matching).
# Fixed ε=5e-3. torchgw uses M_samples = N/2 (see c2_msamples_sweep).
# 5 solvers × N ∈ {1000, 2000, 5000} × 3 seeds.
# Assumes embeddings already cached.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_PY="$REPO_ROOT/tracks/core/02_single_cell_omics/run.py"
OUT="$REPO_ROOT/results/c2_sc_cistopic"
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
    h = d.get('hyperparams', {})
    M = h.get('M_samples', 80)
    # Require: status ok + torchgw cells have tuned M (>= 1000 and != 80)
    is_torchgw = d.get('solver','').startswith('torchgw')
    ok = d.get('status') != 'fail' and d.get('metrics',{}).get('task')
    if is_torchgw and M < 1000: ok = False
    print('1' if ok else '0')
except Exception: print('0')")"
        [[ "$ok" == "1" ]] && { echo "[c2-cis] cached $solver N=$n seed=$seed"; return 0; }
    fi
    local m_flag=""
    if [[ "$solver" == torchgw-* ]]; then
        # M = max(1000, 3N/4) capped at N — keeps torchgw above the
        # unstable M/N ~ 50 % regime seen in c2_msamples_sweep.
        local m_val=$(( n * 3 / 4 ))
        (( m_val < 1000 )) && m_val=1000
        (( m_val > n )) && m_val=$n
        m_flag="--M-samples $m_val"
    fi
    local eps_flag="--epsilon $EPS"
    [[ "$solver" == "pot-exact-gpu" ]] && eps_flag=""
    echo "[c2-cis] $solver N=$n seed=$seed  $m_flag"
    python "$RUN_PY" --solver "$solver" --seed "$seed" \
        --n-cells "$n" --out "$OUT" --atac-method cistopic \
        $eps_flag $m_flag \
        2>&1 | grep -E "^\[C2\]" || true
}

for seed in "${SEEDS[@]}"; do
    for n in "${N_CELLS[@]}"; do
        for solver in "${SOLVERS[@]}"; do
            run_cell "$solver" "$n" "$seed"
        done
    done
done
echo "[c2-cis] Done. Results in $OUT"
