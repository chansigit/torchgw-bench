#!/usr/bin/env bash
# Run C1 Foundation track at all Phase 1b scales and regenerate tier_core.md.
#
# Usage:
#     bash scripts/run_scale_sweep.sh [--out RESULTS_DIR] [--seed N] [--quick]
#
# Options:
#     --out DIR    directory to write JSON records (default: results/)
#     --seed N     random seed (default: 0)
#     --quick      only run 400x500 and 4000x5000 (skip 10k and 20k)
#
# Uses the active Python environment. Does NOT require mamba/conda.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_PY="$REPO_ROOT/tracks/core/01_foundation/run.py"

OUT="$REPO_ROOT/results"
SEED=0
QUICK=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)   OUT="$2"; shift 2 ;;
        --seed)  SEED="$2"; shift 2 ;;
        --quick) QUICK=1; shift ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUT"

# Scale definitions: "N_source K_target"
SCALES=("400 500" "4000 5000" "10000 12000" "20000 25000")
if [[ "$QUICK" -eq 1 ]]; then
    SCALES=("400 500" "4000 5000")
fi

SOLVERS=("torchgw-landmark" "pot-entropic")

echo "[sweep] Starting C1 scale sweep  seed=$SEED  out=$OUT"
echo "[sweep] Scales: ${SCALES[*]}"
echo ""

for scale in "${SCALES[@]}"; do
    N=$(echo "$scale" | awk '{print $1}')
    K=$(echo "$scale" | awk '{print $2}')
    for solver in "${SOLVERS[@]}"; do
        echo "[sweep] N=$N K=$K  solver=$solver"
        python "$RUN_PY" \
            --solver "$solver" \
            --seed "$SEED" \
            --out "$OUT" \
            --n-source "$N" \
            --n-target "$K" \
            2>&1 | grep -E "^\[C1\]" || true
    done
done

echo ""
echo "[sweep] Done. Regenerate report with:"
echo "    python $SCRIPT_DIR/make_report.py --format docs --results $OUT --out $REPO_ROOT/docs/ --tier core"
