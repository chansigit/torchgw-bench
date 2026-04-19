#!/usr/bin/env bash
# C5 word-embedding cross-lingual alignment benchmark.
#
# Sweeps: pairs en-es, en-fi; N ∈ {2000, 5000, 10000}; seeds 0,1,2; 5 solvers.
# torchgw uses M = max(1000, 3N/4) capped at N.
#
# ε NOTE: paper (Alvarez-Melis & Jaakkola 2018) uses ε=5e-5, but POT 0.9.6's
# entropic_gromov_wasserstein does not converge at that scale (Sinkhorn warnings,
# P@1-CSLS≈0). Empirically ε=5e-4 is the C5 sweet spot (P@1-CSLS=0.45 at N=2000
# en-es). We use ε=5e-4 as default for BOTH entropic solvers throughout this bench.
#
# Total cells: 2 pairs × 3 N × 3 seeds × 5 solvers = 90 cells.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_PY="$REPO_ROOT/tracks/core/05_word_embedding/run.py"
OUT="$REPO_ROOT/results/c5_word_embedding"
mkdir -p "$OUT"

SOLVERS=(
    "torchgw-landmark"
    "torchgw-dijkstra"
    "torchgw-precomputed"
    "pot-entropic-gpu"
    "pot-exact-gpu"
)
PAIRS=("en-es" "en-fi")
N_WORDS=(2000 5000 10000)
SEEDS=(0 1 2)
# ε=5e-4: C5 operating point (NOT the paper's 5e-5 which fails to converge in POT 0.9.6)
EPS=5e-4

run_cell () {
    local solver="$1"; local pair="$2"; local n="$3"; local seed="$4"
    local out_json="$OUT/core_05_word_embedding__${solver}__${pair}__n${n}__seed${seed}.json"

    # Cache-skip: require status ok + torchgw cells have tuned M (>= 1000).
    # Also skip cells pre-marked as TimeoutError (too slow to run).
    if [[ -s "$out_json" ]]; then
        local ok
        ok="$(python3 -c "
import json
try:
    d=json.load(open('$out_json'))
    h = d.get('hyperparams', {})
    M = h.get('M_samples', 80)
    is_torchgw = d.get('solver','').startswith('torchgw')
    status = d.get('status','')
    err = d.get('error','') or ''
    # Accept: status ok with task metrics, or pre-marked TimeoutError skip
    if status == 'fail' and 'TimeoutError' in err:
        ok = True  # intentional skip
    else:
        ok = status != 'fail' and d.get('metrics',{}).get('task')
    if is_torchgw and M < 1000 and 'TimeoutError' not in err: ok = False
    print('1' if ok else '0')
except Exception: print('0')")"
        [[ "$ok" == "1" ]] && { echo "[c5] cached $solver $pair N=$n seed=$seed"; return 0; }
    fi

    local m_flag=""
    if [[ "$solver" == torchgw-* ]]; then
        # M = max(1000, 3N/4) capped at N
        local m_val=$(( n * 3 / 4 ))
        (( m_val < 1000 )) && m_val=1000
        (( m_val > n )) && m_val=$n
        m_flag="--M-samples $m_val"
    fi

    local eps_flag="--epsilon $EPS"
    [[ "$solver" == "pot-exact-gpu" ]] && eps_flag=""

    echo "[c5] running $solver $pair N=$n seed=$seed  $m_flag"
    python3 "$RUN_PY" --solver "$solver" --seed "$seed" \
        --pair "$pair" --n-words "$n" --out "$OUT" \
        $eps_flag $m_flag \
        2>&1 | grep -E "^\[C5\]" || true
}

for seed in "${SEEDS[@]}"; do
    for n in "${N_WORDS[@]}"; do
        for pair in "${PAIRS[@]}"; do
            for solver in "${SOLVERS[@]}"; do
                run_cell "$solver" "$pair" "$n" "$seed" || \
                    echo "[c5] FAILED $solver $pair N=$n seed=$seed — continuing"
            done
        done
    done
done

echo "[c5] Done. Results in $OUT"
