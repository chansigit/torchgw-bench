#!/usr/bin/env bash
# C1 point-cloud scalability benchmark.
#
# Sweeps: shape_class=airplane; instance_idx ∈ {0,1,2}; N ∈ {10000,20000,50000,100000};
#         seeds ∈ {0,1,2}; N-conditional solver list (see Step 2 below).
# torchgw variants: M = max(1000, 3N/4) capped at N.
# Cost: kNN-hop geodesic with k=200 (sweet spot at N=10k+; tradeoff vs k=400
# which gave P@1=0.66 but Dijkstra at k=400 takes 5min/cell at N=10k).
# ε=5e-4 (k=200 sweet spot from grid sweep — gives torchgw-precomp P@1≈0.53
# at N=10k; smaller than C5 word embedding's 5e-4 by accident).
#
# N-conditional solver list:
#   N ≤ 20k: all 7 solvers (5 standard + 2 lowrank)
#   N > 20k: 4 torchgw-only solvers (POT + precomputed self-skip at OOM risk)
#
# Estimated cells:
#   N=10k,20k: 3 inst × 3 seeds × 7 solvers = 63 cells each → 126 total
#   N=50k,100k: 3 inst × 3 seeds × 4 solvers = 36 cells each → 72 total
#   Grand total: ~198 cells (3–15 hours depending on solver speed)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_PY="$REPO_ROOT/tracks/core/01_point_cloud_scale/run.py"
OUT="$REPO_ROOT/results/c1_point_cloud_scale"
LOG="$REPO_ROOT/logs/c1_bench.log"
mkdir -p "$OUT" "$REPO_ROOT/logs"

# Synthetic asymmetric-spiral task (no "instance" concept — spiral shape is fixed;
# seed varies rotation matrix + optional noise).  Keep INSTANCES=(0) for loop
# compatibility but run.py ignores instance_idx when --data=spiral.
INSTANCES=(0)
N_POINTS=(10000 20000 50000 100000)
SEEDS=(0 1 2)
# ε=5e-2: kNN-hop operating point (NOT the C5 word-embedding 5e-4)
EPS=5e-3

run_cell () {
    local solver="$1"; local inst="$2"; local n="$3"; local seed="$4"
    local out_json="$OUT/core_01_point_cloud_scale__${solver}__spiral_noise0.0__n${n}__seed${seed}.json"

    # Cache-skip: require status ok + torchgw cells have tuned M (>= 1000).
    # Also accept skipped_oom_risk (intentional N-conditional skip) and TimeoutError.
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
    if status == 'skipped_oom_risk':
        ok = True  # intentional N-conditional skip
    elif status == 'fail' and 'TimeoutError' in err:
        ok = True  # intentional skip
    else:
        ok = status != 'fail' and d.get('metrics',{}).get('task')
    if is_torchgw and M < 1000 and 'TimeoutError' not in err and status != 'skipped_oom_risk': ok = False
    print('1' if ok else '0')
except Exception: print('0')")"
        [[ "$ok" == "1" ]] && { echo "[c1] cached $solver N=$n seed=$seed inst=$inst"; return 0; }
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

    local rank_flag=""
    [[ "$solver" == torchgw-lowrank-* ]] && rank_flag="--lowrank-rank 20"

    echo "[c1] running $solver N=$n seed=$seed  $m_flag $rank_flag"
    # 60 min per-cell wall cap — some dijkstra cells approach this at N=100k
    timeout 3600 python3 "$RUN_PY" --solver "$solver" --seed "$seed" \
        --data spiral --n-points "$n" --out "$OUT" \
        $eps_flag $m_flag $rank_flag \
        2>&1 | grep -E "^\[C1\]" || true
}

for seed in "${SEEDS[@]}"; do
    for n in "${N_POINTS[@]}"; do
        # N-conditional solver list.  torchgw-lowrank-dijkstra dropped: it's
        # dominated by torchgw-lowrank-landmark on quality AND is ~100× slower
        # at large N (e.g. N=20k took 3541s vs 32s for lowrank-landmark).
        if (( n <= 20000 )); then
            SOLVERS=(pot-entropic-gpu pot-exact-gpu \
                     torchgw-landmark torchgw-dijkstra torchgw-precomputed \
                     torchgw-lowrank-landmark)
        else
            # N>20k: POT/precomputed self-skip (OOM).  Keep only the variants
            # that have a chance of fitting on an 80GB H100 with contention.
            SOLVERS=(torchgw-landmark torchgw-dijkstra torchgw-lowrank-landmark)
        fi
        for inst in "${INSTANCES[@]}"; do
            for solver in "${SOLVERS[@]}"; do
                run_cell "$solver" "$inst" "$n" "$seed" || \
                    echo "[c1] FAILED $solver N=$n seed=$seed inst=$inst — continuing"
            done
        done
    done
done

echo "[c1] Done. Results in $OUT"
