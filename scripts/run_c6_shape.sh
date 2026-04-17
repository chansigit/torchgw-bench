#!/usr/bin/env bash
# C6 TACO shape-correspondence benchmark: 5 GPU solvers × ~18 TACO pairs
# × 3 seeds at N=2000 vertices subsampled per mesh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_PY="$REPO_ROOT/tracks/core/06_shape_correspondence/run.py"
DATA_ROOT="$REPO_ROOT/data/core_06_shape/taco"

OUT="$REPO_ROOT/results/c6_shape"
N_SRC=2000
N_TGT=2000
SEEDS=(0 1 2)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)   OUT="$2"; shift 2 ;;
        --n)     N_SRC="$2"; N_TGT="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done
mkdir -p "$OUT"

SOLVERS=(
    "torchgw-landmark"
    "torchgw-dijkstra"
    "torchgw-precomputed"
    "pot-entropic-gpu"
    "pot-exact-gpu"
)

# Pick a few representative pairs per animal class (first 2 pairs listed
# in pairs.txt for each class). This gives us broad class coverage
# without blowing up the cell count.
PAIRS=$(DATA_ROOT="$DATA_ROOT" python - <<'PY'
import os, re
from pathlib import Path
data = Path(os.environ["DATA_ROOT"])
pairs = [l.strip() for l in (data/"pairs.txt").read_text().splitlines() if l.strip()]
by_class = {}
for p in pairs:
    a, b = p.split(",")
    cls = re.match(r"[a-z]+", a).group()
    by_class.setdefault(cls, []).append(p)
for cls in sorted(by_class):
    for p in by_class[cls][:2]:
        print(p)
PY
)

echo "[c6-bench] Pairs ($(echo "$PAIRS" | wc -l) total):"
echo "$PAIRS" | sed 's/^/  /'

run_cell () {
    local solver="$1"; local pair="$2"; local seed="$3"
    local src; local tgt
    src="${pair%,*}"; tgt="${pair#*,}"
    local out_json="$OUT/core_06_shape_correspondence__${solver}__${src}_${tgt}__n${N_SRC}__seed${seed}.json"
    if [[ -s "$out_json" ]]; then
        local ok
        ok="$(python -c "
import json
try:
    d=json.load(open('$out_json'))
    print('1' if d.get('status') != 'fail' and d.get('metrics',{}).get('task') else '0')
except Exception: print('0')")"
        if [[ "$ok" == "1" ]]; then
            echo "[c6-bench] cached: $solver $pair seed=$seed"
            return 0
        fi
    fi
    echo "[c6-bench] $solver  $pair  seed=$seed"
    python "$RUN_PY" \
        --solver "$solver" \
        --seed "$seed" \
        --out "$OUT" \
        --pair "$pair" \
        --n-source "$N_SRC" \
        --n-target "$N_TGT" \
        2>&1 | grep -E "^\[C6\]" || true
}

for seed in "${SEEDS[@]}"; do
    for solver in "${SOLVERS[@]}"; do
        while IFS= read -r pair; do
            [[ -z "$pair" ]] && continue
            run_cell "$solver" "$pair" "$seed"
        done <<< "$PAIRS"
    done
done

echo "[c6-bench] Done. Results in $OUT"
echo "[c6-bench] To plot: python scripts/experiments/make_c6_shape_plot.py"
