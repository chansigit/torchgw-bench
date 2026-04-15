#!/usr/bin/env bash
# C3 anytime / Pareto sweep: for each GPU solver, sweep max_iter at fixed N
# with early-stop disabled. Produces time-vs-rho Pareto trajectories.
#
# Output: results/c3_anytime/*.json (one file per (solver, max_iter, seed)).
#
# Usage:
#     bash scripts/run_c3_anytime.sh                  # default N=4000, 3 seeds
#     bash scripts/run_c3_anytime.sh --n 10000        # larger N
#     bash scripts/run_c3_anytime.sh --out /tmp/foo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_PY="$REPO_ROOT/tracks/core/03_branched/run.py"

OUT="$REPO_ROOT/results/c3_anytime"
N_SRC=4000
N_TGT=5000
SEEDS=(0 1 2)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)   OUT="$2"; shift 2 ;;
        --n)     N_SRC="$2"; shift 2 ;;
        --n-tgt) N_TGT="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$OUT"

# GPU-only solver set (CPU variants excluded from anytime study).
SOLVERS=(
    "torchgw-landmark"
    "torchgw-dijkstra"
    "torchgw-precomputed"
    "pot-entropic-gpu"
    "pot-exact-gpu"
    "pot-bapg-gpu"
)

# max_iter grid — geometric so low end has fine resolution for the steep
# early-gains region, high end caps solver-default max.
ITERS=(5 10 20 50 100 200 500)

cell_path () {
    local solver="$1"; local mi="$2"; local seed="$3"
    echo "$OUT/core_03_branched__${solver}__n${N_SRC}k${N_TGT}__seed${seed}__iter${mi}.json"
}

read_rho () {  # echo tail_arclen_spearman from a JSON file, or "nan"
    local f="$1"
    python -c "import json,sys;\
d=json.load(open('$f'));\
r=d.get('metrics',{}).get('task',{}).get('tail_arclen_spearman');\
print(r if r is not None else 'nan')" 2>/dev/null || echo "nan"
}

run_cell () {
    local solver="$1"; local mi="$2"; local seed="$3"
    local out_json; out_json="$(cell_path "$solver" "$mi" "$seed")"
    if [[ -s "$out_json" ]]; then
        # Only honor cache if the prior run succeeded (status != fail).
        local cached_ok
        cached_ok="$(python -c "
import json
try:
    d = json.load(open('$out_json'))
    print('1' if d.get('status') != 'fail' and d.get('metrics',{}).get('task') else '0')
except Exception:
    print('0')")"
        if [[ "$cached_ok" == "1" ]]; then
            echo "[c3-any] cached: solver=$solver max_iter=$mi seed=$seed"
            return 0
        fi
    fi
    echo "[c3-any] solver=$solver  max_iter=$mi  seed=$seed  (N=$N_SRC K=$N_TGT)"
    python "$RUN_PY" \
        --solver "$solver" \
        --seed "$seed" \
        --out "$OUT" \
        --n-source "$N_SRC" \
        --n-target "$N_TGT" \
        --max-iter "$mi" \
        --force-full \
        --tag "iter$mi" \
        2>&1 | grep -E "^\[C3\]" || true
}

# Saturation rule: if the last two cells of this (solver, seed) both hit
# rho >= 0.999 AND |delta| < 0.0005, skip remaining higher max_iter values.
saturated () {
    local prev="$1"; local cur="$2"
    python -c "
prev, cur = $prev, $cur
import math
if math.isnan(prev) or math.isnan(cur): print('0')
elif prev >= 0.999 and cur >= 0.999 and abs(cur-prev) < 5e-4: print('1')
else: print('0')"
}

echo "[c3-any] === anytime Pareto sweep at N=$N_SRC, K=$N_TGT (smart early-stop) ==="
for seed in "${SEEDS[@]}"; do
    for solver in "${SOLVERS[@]}"; do
        prev_rho="nan"
        for mi in "${ITERS[@]}"; do
            run_cell "$solver" "$mi" "$seed"
            cur_rho="$(read_rho "$(cell_path "$solver" "$mi" "$seed")")"
            if [[ "$(saturated "$prev_rho" "$cur_rho")" == "1" ]]; then
                echo "[c3-any] SATURATED at max_iter=$mi (rho=$cur_rho, prev=$prev_rho); skipping higher iters for $solver seed=$seed"
                break
            fi
            prev_rho="$cur_rho"
        done
    done
done

echo ""
echo "[c3-any] Done. Results in $OUT"
echo "[c3-any] To plot: python scripts/experiments/make_c3_anytime_plot.py --in $OUT"
