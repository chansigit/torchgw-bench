#!/usr/bin/env python
"""Plot the C3 epsilon sweep — how does ε affect tail ρ and wall time?

Reads results/c3_eps/*.json (one per (solver, epsilon, seed)) and produces
c3_eps_sweep.png — a 2-panel figure:

  left:  tail ρ vs epsilon (log-x)
  right: wall_s_solve vs epsilon (log-log)

One line per solver, shaded ±1σ across seeds. Fixed max_iter=100 with
force-full enabled.

Usage:
    python scripts/experiments/make_c3_eps_plot.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = REPO / "results" / "c3_eps"
FIG_DIR = REPO / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SOLVER_COLOR = {
    "torchgw-landmark":    "#08306b",
    "torchgw-dijkstra":    "#2171b5",
    "torchgw-precomputed": "#6baed6",
    "pot-entropic-gpu":    "#d94801",
    "pot-bapg-gpu":        "#fd8d3c",
}
SOLVER_LABEL = {
    "torchgw-landmark":    "torchgw-landmark (ε=Sinkhorn inner)",
    "torchgw-dijkstra":    "torchgw-dijkstra (ε=Sinkhorn inner)",
    "torchgw-precomputed": "torchgw-precomputed (ε=Sinkhorn inner)",
    "pot-entropic-gpu":    "POT-entropic (GPU)",
    "pot-bapg-gpu":        "POT-BAPG (GPU, fp64)",
}
SOLVER_ORDER = list(SOLVER_LABEL)


def load(results_dir: Path) -> list[dict]:
    out = []
    for p in sorted(results_dir.glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("status") in ("fail", "skip"):
            continue
        out.append(d)
    return out


def aggregate(recs: list[dict]) -> dict:
    bucket: dict[tuple[str, float], list[tuple[float, float]]] = defaultdict(list)
    for r in recs:
        solver = r["solver"]
        eps = r["hyperparams"].get("epsilon")
        if eps is None:
            continue
        wall = float(r["metrics"]["efficiency"]["wall_s_solve"])
        rho = r["metrics"]["task"]["tail_arclen_spearman"]
        if rho is None or not np.isfinite(rho):
            continue
        bucket[(solver, float(eps))].append((wall, float(rho)))

    out: dict[str, list[tuple[float, float, float, float, float]]] = defaultdict(list)
    for (solver, eps), pairs in bucket.items():
        arr = np.asarray(pairs)
        out[solver].append((eps, arr[:, 0].mean(), arr[:, 0].std(),
                             arr[:, 1].mean(), arr[:, 1].std()))
    for s in out:
        out[s].sort(key=lambda t: t[0])
    return out


def plot(agg: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for solver in SOLVER_ORDER:
        if solver not in agg:
            continue
        trace = np.asarray(agg[solver])
        eps, mw, sw, mr, sr = trace.T
        color = SOLVER_COLOR[solver]
        label = SOLVER_LABEL[solver]

        axes[0].errorbar(eps, mr, yerr=sr, fmt="o-", color=color, label=label,
                          markersize=6, linewidth=1.6, capsize=2)
        axes[1].errorbar(eps, mw, yerr=sw, fmt="o-", color=color, label=label,
                          markersize=6, linewidth=1.6, capsize=2)

    axes[0].set_xscale("log")
    axes[0].set_xlabel("epsilon (entropic regularisation)")
    axes[0].set_ylabel("tail Spearman ρ")
    axes[0].set_title("Quality vs ε")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend(fontsize=8, loc="lower left")

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("epsilon (log)")
    axes[1].set_ylabel("wall_s_solve (s, log)")
    axes[1].set_title("Cost vs ε")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.suptitle("C3 epsilon sweep — N=4000, max_iter=100, force-full, 3 seeds",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"[eps] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out", type=Path, default=FIG_DIR / "c3_eps_sweep.png")
    args = ap.parse_args()

    recs = load(args.in_dir)
    if not recs:
        raise SystemExit(f"no records in {args.in_dir}")
    agg = aggregate(recs)
    plot(agg, args.out)


if __name__ == "__main__":
    main()
