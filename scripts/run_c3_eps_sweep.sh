#!/usr/bin/env bash
# C3 epsilon sweep: how does entropic regularisation strength affect the
# two ε-regularised FGW solvers (pot-entropic-gpu, pot-bapg-gpu)?
# Fixed N=4000, K=5000, max_iter=100, force-full, 3 seeds.
#
# Also sweeps torchgw-landmark (uses ε for its Sinkhorn inner solver) to
# see whether the torchgw family is equally epsilon-sensitive.
#
# Output: results/c3_eps/*.json (one per (solver, epsilon, seed)).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_PY="$REPO_ROOT/tracks/core/03_branched/run.py"

OUT="$REPO_ROOT/results/c3_eps"
N_SRC=4000
N_TGT=5000
MAX_ITER=100
SEEDS=(0 1 2)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)     OUT="$2"; shift 2 ;;
        --n)       N_SRC="$2"; shift 2 ;;
        --n-tgt)   N_TGT="$2"; shift 2 ;;
        --max-iter) MAX_ITER="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$OUT"

SOLVERS=(
    "torchgw-landmark"
    "torchgw-dijkstra"
    "torchgw-precomputed"
    "pot-entropic-gpu"
    "pot-bapg-gpu"
)

# Cross four orders of magnitude: 5e-4, 5e-3, 5e-2, 5e-1.
EPSILONS=(5e-4 5e-3 5e-2 5e-1)

cell_path () {
    local solver="$1"; local eps="$2"; local seed="$3"
    echo "$OUT/core_03_branched__${solver}__n${N_SRC}k${N_TGT}__seed${seed}__eps${eps}.json"
}

run_cell () {
    local solver="$1"; local eps="$2"; local seed="$3"
    local out_json; out_json="$(cell_path "$solver" "$eps" "$seed")"
    if [[ -s "$out_json" ]]; then
        local ok
        ok="$(python -c "
import json
try:
    d = json.load(open('$out_json'))
    print('1' if d.get('status') != 'fail' and d.get('metrics',{}).get('task') else '0')
except Exception:
    print('0')")"
        if [[ "$ok" == "1" ]]; then
            echo "[c3-eps] cached: solver=$solver eps=$eps seed=$seed"
            return 0
        fi
    fi
    echo "[c3-eps] solver=$solver eps=$eps seed=$seed max_iter=$MAX_ITER"
    python "$RUN_PY" \
        --solver "$solver" \
        --seed "$seed" \
        --out "$OUT" \
        --n-source "$N_SRC" \
        --n-target "$N_TGT" \
        --max-iter "$MAX_ITER" \
        --epsilon "$eps" \
        --force-full \
        --tag "eps${eps}" \
        2>&1 | grep -E "^\[C3\]" || true
}

echo "[c3-eps] === epsilon sweep at N=$N_SRC, K=$N_TGT, max_iter=$MAX_ITER ==="
for seed in "${SEEDS[@]}"; do
    for solver in "${SOLVERS[@]}"; do
        for eps in "${EPSILONS[@]}"; do
            run_cell "$solver" "$eps" "$seed"
        done
    done
done

echo "[c3-eps] Done. Results in $OUT"
echo "[c3-eps] To plot: python scripts/experiments/make_c3_eps_plot.py"
