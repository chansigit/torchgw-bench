#!/usr/bin/env bash
# Iterate all tracks in a given tier, activate each track's declared conda env,
# and run its run.py with the requested solver and seed. Results are written
# into ../results/ (relative to the repo root).
#
# Usage:
#     bash scripts/run_tier.sh <tier> [--solvers "s1 s2 ..."] [--seed N]
#
# Defaults:
#     --solvers "torchgw-landmark pot-entropic"
#     --seed 0
#
# Phase 1 notes:
# - Only the C1 foundation track is landed, so running this with tier=core
#   will simply run C1.
# - Gallery tracks are skipped entirely: Gallery outputs are notebooks, not
#   JSON records, and thus do not participate in run_tier.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

TIER="${1:-}"
if [[ -z "$TIER" ]]; then
    echo "Usage: $0 <tier> [--solvers \"s1 s2 ...\"] [--seed N]" >&2
    exit 2
fi
shift

SOLVERS="torchgw-landmark pot-entropic"
SEED=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --solvers) SOLVERS="$2"; shift 2 ;;
        --seed)    SEED="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [[ "$TIER" == "gallery" ]]; then
    echo "[run_tier] Gallery tier does not support scripted runs (notebooks only)."
    exit 0
fi

TIER_DIR="$REPO_ROOT/tracks/$TIER"
if [[ ! -d "$TIER_DIR" ]]; then
    echo "[run_tier] tier directory not found: $TIER_DIR" >&2
    exit 1
fi

# Source conda for 'conda activate'
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

mkdir -p "$REPO_ROOT/results"

for track_dir in "$TIER_DIR"/*/; do
    track_name="$(basename "$track_dir")"
    run_py="$track_dir/run.py"
    env_yaml="$track_dir/env.yaml"

    if [[ ! -f "$run_py" ]]; then
        echo "[run_tier] skipping $TIER/$track_name (no run.py)"
        continue
    fi

    # Parse 'env:' value from the track's env.yaml (first non-comment "env:" line)
    env_short=""
    if [[ -f "$env_yaml" ]]; then
        env_short="$(grep -E '^env:' "$env_yaml" | head -n1 | awk '{print $2}')"
    fi
    env_short="${env_short:-base}"
    conda_env="tgwbench-$env_short"

    echo "[run_tier] === $TIER/$track_name (env=$conda_env) ==="
    conda activate "$conda_env"

    for solver in $SOLVERS; do
        echo "[run_tier]   solver=$solver seed=$SEED"
        python "$run_py" --solver "$solver" --seed "$SEED" --out "$REPO_ROOT/results/" || {
            echo "[run_tier]   WARN: $TIER/$track_name solver=$solver exited non-zero"
        }
    done

    conda deactivate
done

echo "[run_tier] Done. Regenerate the report with:"
echo "    python scripts/make_report.py --format docs --results results/ --out docs/ --tier $TIER"
