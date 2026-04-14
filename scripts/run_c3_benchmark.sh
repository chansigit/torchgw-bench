#!/usr/bin/env bash
# Drive the C3 Y-fork benchmark sweep:
#   E1 — solver shootout at N=400, K=500 across 5 seeds, all 3 solvers.
#   E2 — multi-scale wall/memory at N in {400, 1000, 4000, 10000, 20000}
#        across 3 seeds. POT auto-skips above the memory guard threshold.
#
# Output: results/c3_benchmark/*.json (one file per (solver, scale, seed)).
#
# Usage:
#     bash scripts/run_c3_benchmark.sh                 # full E1 + E2
#     bash scripts/run_c3_benchmark.sh --quick         # E1 only
#     bash scripts/run_c3_benchmark.sh --out /tmp/foo  # custom output dir

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_PY="$REPO_ROOT/tracks/core/03_branched/run.py"

OUT="$REPO_ROOT/results/c3_benchmark"
QUICK=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)   OUT="$2"; shift 2 ;;
        --quick) QUICK=1; shift ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$OUT"

SOLVERS=("torchgw-landmark" "torchgw-fused" "pot-fused")

declare -A SCALES
SCALES[400]="500"
SCALES[1000]="1200"
SCALES[2000]="2500"
SCALES[3000]="3700"
SCALES[4000]="5000"
SCALES[7500]="9000"
SCALES[10000]="12000"
SCALES[20000]="25000"

run_cell () {
    local solver="$1"
    local n_src="$2"
    local n_tgt="$3"
    local seed="$4"
    echo "[c3-bench] solver=$solver  N=$n_src K=$n_tgt  seed=$seed"
    python "$RUN_PY" \
        --solver "$solver" \
        --seed "$seed" \
        --out "$OUT" \
        --n-source "$n_src" \
        --n-target "$n_tgt" \
        2>&1 | grep -E "^\[C3\]" || true
}

echo "[c3-bench] === E1: solver shootout at N=400, K=500, 5 seeds ==="
for seed in 0 1 2 3 4; do
    for solver in "${SOLVERS[@]}"; do
        run_cell "$solver" 400 500 "$seed"
    done
done

if [[ "$QUICK" -eq 1 ]]; then
    echo "[c3-bench] --quick selected; skipping E2 scale sweep."
    echo "[c3-bench] Results in $OUT"
    exit 0
fi

echo "[c3-bench] === E2: scale sweep, 3 seeds per cell ==="
for n_src in 1000 2000 3000 4000 7500 10000 20000; do
    n_tgt="${SCALES[$n_src]}"
    for seed in 0 1 2; do
        for solver in "${SOLVERS[@]}"; do
            run_cell "$solver" "$n_src" "$n_tgt" "$seed"
        done
    done
done

echo ""
echo "[c3-bench] Done. Results in $OUT"
echo "[c3-bench] To plot: python scripts/experiments/make_c3_benchmark_plots.py"
