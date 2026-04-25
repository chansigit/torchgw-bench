#!/usr/bin/env bash
# C7 Stage A sweep: NeuroMorpho hand-picked subset (~300 cells, 3 classes).
# Idempotent — existing JSONs are skipped.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT="$REPO_ROOT/results/c7_cell_morphology"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$OUT" "$LOG_DIR"

SOLVERS=(cajal-native pot-entropic-gpu pot-exact-gpu torchgw-precomputed)
N_VALUES=(50 200 500 1000)
SEEDS=(0 1 2)

for solver in "${SOLVERS[@]}"; do
    for n in "${N_VALUES[@]}"; do
        # spec §5: pot-exact-gpu skipped above N=200 (CG memory)
        if [[ "$solver" == "pot-exact-gpu" && "$n" -gt 200 ]]; then continue; fi
        for seed in "${SEEDS[@]}"; do
            tag="${solver}__stage_a__n${n}__seed${seed}"
            json="$OUT/core_07_cell_morphology__${tag}.json"
            if [[ -s "$json" ]]; then
                echo "[c7-A] skip done: $tag"; continue
            fi
            echo "[c7-A] running $tag"
            env -u PYTHONPATH micromamba run -n c7_morph python \
                "$REPO_ROOT/tracks/core/07_cell_morphology/run.py" \
                --stage A --solver "$solver" --n-per-cell "$n" \
                --seed "$seed" --out "$OUT" \
                2>&1 | tee -a "$LOG_DIR/c7_stage_a.log"
        done
    done
done
echo "[c7-A] done."
