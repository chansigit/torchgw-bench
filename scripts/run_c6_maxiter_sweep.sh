#!/usr/bin/env bash
# C6 max_iter stress test for torchgw-dijkstra on TACO.
# Fixed pairs × max_iter sweep × 3 seeds, --force-full to disable
# early-stop. pot-exact-gpu baseline rerun at max_iter=500 for reference.
#
# Goal: see whether cranking torchgw-dijkstra's max_iter on shape
# correspondence closes the gap to POT, or confirms the gap is
# algorithmic.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_PY="$REPO_ROOT/tracks/core/06_shape_correspondence/run.py"

OUT="$REPO_ROOT/results/c6_maxiter"
N=2000
SEEDS=(0 1 2)
mkdir -p "$OUT"

PAIRS=(
    "cat0,cat1"
    "horse0,horse5"
    "david0,david1"
)
ITERS=(100 300 1000 2000 5000)
K_VALUES=(5 10 20 40)

run_cell () {
    local solver="$1"; local pair="$2"; local mi="$3"; local k="$4"; local seed="$5"
    local src; local tgt
    src="${pair%,*}"; tgt="${pair#*,}"
    local tag
    if [[ -z "$k" ]]; then tag="iter${mi}"
    else tag="iter${mi}_k${k}"; fi
    local out_json="$OUT/core_06_shape_correspondence__${solver}__${src}_${tgt}__n${N}__seed${seed}__${tag}.json"
    if [[ -s "$out_json" ]]; then
        local ok
        ok="$(python -c "
import json
try:
    d=json.load(open('$out_json'))
    print('1' if d.get('status') != 'fail' and d.get('metrics',{}).get('task') else '0')
except Exception: print('0')")"
        [[ "$ok" == "1" ]] && { echo "[c6-mi] cached $solver $pair $tag s=$seed"; return 0; }
    fi
    echo "[c6-mi] $solver $pair $tag seed=$seed"
    local k_flag=""
    if [[ -n "$k" ]]; then k_flag="--k $k"; fi
    python "$RUN_PY" \
        --solver "$solver" \
        --seed "$seed" \
        --out "$OUT" \
        --pair "$pair" \
        --n-source "$N" \
        --n-target "$N" \
        --max-iter "$mi" \
        --force-full \
        --tag "$tag" \
        $k_flag \
        2>&1 | grep -E "^\[C6\]" || true
}

echo "[c6-mi] === torchgw-dijkstra max_iter sweep (k=5 default) ==="
for seed in "${SEEDS[@]}"; do
    for pair in "${PAIRS[@]}"; do
        for mi in "${ITERS[@]}"; do
            run_cell torchgw-dijkstra "$pair" "$mi" "" "$seed"
        done
    done
done

echo "[c6-mi] === torchgw-dijkstra k sweep (max_iter=500 fixed) ==="
for seed in "${SEEDS[@]}"; do
    for pair in "${PAIRS[@]}"; do
        for k in "${K_VALUES[@]}"; do
            run_cell torchgw-dijkstra "$pair" 500 "$k" "$seed"
        done
    done
done

# POT-exact baseline at the same pairs/seeds, fixed max_iter=500
echo "[c6-mi] === pot-exact-gpu baseline (max_iter=500) ==="
for seed in "${SEEDS[@]}"; do
    for pair in "${PAIRS[@]}"; do
        run_cell pot-exact-gpu "$pair" 500 "" "$seed"
    done
done

echo "[c6-mi] Done. Results in $OUT"
