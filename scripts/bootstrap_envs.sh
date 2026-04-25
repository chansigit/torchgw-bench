#!/usr/bin/env bash
# Bootstrap all conda envs declared under envs/*.yaml, then pip-editable-install
# torchgw into each env. Re-running is idempotent: existing envs are updated.
#
# Usage:
#     TORCHGW_SRC=/path/to/torchgw bash scripts/bootstrap_envs.sh
#
# Defaults:
#     TORCHGW_SRC defaults to ../sgw (sibling directory of torchgw-bench)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TORCHGW_SRC="${TORCHGW_SRC:-$REPO_ROOT/../sgw}"

if [[ ! -d "$TORCHGW_SRC" ]]; then
    echo "[bootstrap] ERROR: TORCHGW_SRC '$TORCHGW_SRC' does not exist." >&2
    echo "[bootstrap] Set TORCHGW_SRC to the absolute path of your torchgw clone." >&2
    exit 1
fi

if ! command -v mamba >/dev/null 2>&1; then
    echo "[bootstrap] ERROR: 'mamba' not found on PATH. Install mambaforge first." >&2
    exit 1
fi


for f in "$REPO_ROOT"/envs/*.yaml; do
    name="$(basename "$f" .yaml)"
    env_name="tgwbench-$name"

    if mamba env list | awk '{print $1}' | grep -qx "$env_name"; then
        echo "[bootstrap] Updating existing env: $env_name"
        mamba env update -f "$f" -n "$env_name"
    else
        echo "[bootstrap] Creating env: $env_name"
        mamba env create -f "$f" -n "$env_name"
    fi

    echo "[bootstrap] Installing torchgw (editable) into $env_name from $TORCHGW_SRC"
    mamba run -n "$env_name" pip install -e "$TORCHGW_SRC"
done

echo "[bootstrap] Done. Created/updated envs:"
mamba env list | grep '^tgwbench-' || true

# C7 cell morphology — isolated to keep CAJAL's POT pin off C2/C3/C5/C6
if ! micromamba env list | grep -q '^c7_morph '; then
    micromamba env create -f tracks/core/07_cell_morphology/env.yaml -y
fi
