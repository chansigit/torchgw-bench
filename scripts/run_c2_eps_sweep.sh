#!/usr/bin/env bash
# C2 epsilon sensitivity sweep — does ε tuning close the gap to literature?
# Focus on two ε-regularised solvers at N=2000 × 3 seeds.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_PY="$REPO_ROOT/tracks/core/02_single_cell_omics/run.py"
OUT="$REPO_ROOT/results/c2_eps"
mkdir -p "$OUT"

N=2000
SEEDS=(0 1 2)

TORCHGW_EPS=(5e-3 5e-2 5e-1 1.0)
POT_EPS=(5e-4 5e-3 5e-2 5e-1)

run_cell () {
    local solver="$1"; local eps="$2"; local seed="$3"
    local tag="eps${eps}"
    local out_json="$OUT/core_02_single_cell_omics__${solver}__n${N}__seed${seed}__${tag}.json"
    if [[ -s "$out_json" ]]; then
        local ok
        ok="$(python -c "
import json
try:
    d=json.load(open('$out_json'))
    print('1' if d.get('status') != 'fail' and d.get('metrics',{}).get('task') else '0')
except Exception: print('0')")"
        [[ "$ok" == "1" ]] && { echo "[c2-eps] cached $solver eps=$eps seed=$seed"; return 0; }
    fi
    echo "[c2-eps] $solver eps=$eps seed=$seed"
    python "$RUN_PY" --solver "$solver" --seed "$seed" \
        --n-cells "$N" --out "$OUT" \
        --epsilon "$eps" --tag "$tag" 2>&1 | grep -E "^\[C2\]" || true
}

for seed in "${SEEDS[@]}"; do
    for eps in "${TORCHGW_EPS[@]}"; do
        run_cell torchgw-precomputed "$eps" "$seed"
    done
    for eps in "${POT_EPS[@]}"; do
        run_cell pot-entropic-gpu "$eps" "$seed"
    done
    # pot-exact baseline (no eps) — reuse from c2_sc if present, else run
    pot_json="$OUT/core_02_single_cell_omics__pot-exact-gpu__n${N}__seed${seed}.json"
    if [[ ! -s "$pot_json" ]]; then
        src_json="$REPO_ROOT/results/c2_sc/core_02_single_cell_omics__pot-exact-gpu__n${N}__seed${seed}.json"
        if [[ -s "$src_json" ]]; then
            cp "$src_json" "$pot_json"
            echo "[c2-eps] copied baseline pot-exact seed=$seed"
        else
            python "$RUN_PY" --solver pot-exact-gpu --seed "$seed" \
                --n-cells "$N" --out "$OUT" 2>&1 | grep -E "^\[C2\]" || true
        fi
    fi
done
echo "[c2-eps] Done. Results in $OUT"
