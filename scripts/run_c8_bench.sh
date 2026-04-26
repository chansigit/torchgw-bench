#!/usr/bin/env bash
# C8 brain-alignment bench: 3 resolutions × 4 solvers × 66 pairs × 3 seeds.
# Idempotent — existing JSONs are skipped. Skip rules per spec §8:
#   - pot-entropic-fgw: dense FGW OOM on fsaverage6/7, only fsaverage5
#   - torchgw-balanced: per C1 finding, OOM at fsaverage7
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT="$REPO_ROOT/results/c8_brain_alignment"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$OUT" "$LOG_DIR"

RESOLUTIONS=(fsaverage5 fsaverage6 fsaverage7)
SOLVERS=(fugw-native pot-entropic-fgw torchgw-balanced torchgw-unbalanced)
SEEDS=(0 1 2)

# Parse subject IDs from manifest (12 subjects)
mapfile -t SUBJECTS < <(awk -F'\t' '!/^#/ && !/^subject_id/ && NF>=3 {print $1}' \
                       "$REPO_ROOT/tracks/core/08_brain_alignment/manifest.txt")
N=${#SUBJECTS[@]}
echo "[c8] $N subjects, $((N*(N-1)/2)) pairs per (resolution, solver, seed) cell"

for resolution in "${RESOLUTIONS[@]}"; do
    for solver in "${SOLVERS[@]}"; do
        # spec §8: pot-entropic-fgw OOM at fsaverage6+; skip cleanly
        if [[ "$solver" == "pot-entropic-fgw" && "$resolution" != "fsaverage5" ]]; then
            echo "[c8] skip $solver @ $resolution (spec §8: OOM expected)"; continue
        fi
        # spec §8: torchgw-balanced expected to OOM at fsaverage7 per C1
        if [[ "$solver" == "torchgw-balanced" && "$resolution" == "fsaverage7" ]]; then
            echo "[c8] skip $solver @ $resolution (spec §8: C1 30k ceiling)"; continue
        fi
        for seed in "${SEEDS[@]}"; do
            for ((i=0; i<N; i++)); do
                for ((j=i+1; j<N; j++)); do
                    pair="${SUBJECTS[i]}__${SUBJECTS[j]}"
                    json="$OUT/core_08_brain__${solver}__${resolution}__${pair}__seed${seed}.json"
                    if [[ -s "$json" ]]; then
                        continue
                    fi
                    echo "[c8] running $solver $resolution $pair seed$seed"
                    env -u PYTHONPATH micromamba run -n c8_brain python \
                        "$REPO_ROOT/tracks/core/08_brain_alignment/run.py" \
                        --resolution "$resolution" --solver "$solver" \
                        --pair "$pair" --seed "$seed" --out "$OUT" \
                        2>&1 | tee -a "$LOG_DIR/c8_bench.log"
                done
            done
        done
    done
done
echo "[c8] done."
