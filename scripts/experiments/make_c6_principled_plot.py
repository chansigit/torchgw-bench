#!/usr/bin/env python
"""Principled C6 evaluation visuals.

Two panels:
  left:  CDF of normalised geodesic error (supervised, task-aligned)
  right: CDF of pair distortion (unsupervised, GW-native)

Both are 'fraction of samples ≤ τ' curves — equivalent to the
shape-matching community's Princeton benchmark curve. We deliberately
do NOT label the y-axis 'accuracy' — it is a CDF of an error metric,
not a classification score.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results" / "c6_principled"
FIG_DIR = REPO / "docs" / "figures"

SOLVER_ORDER = ["torchgw-landmark", "torchgw-dijkstra", "torchgw-precomputed",
                 "pot-entropic-gpu", "pot-exact-gpu"]
SOLVER_COLOR = {
    "torchgw-landmark":    "#08306b",
    "torchgw-dijkstra":    "#2171b5",
    "torchgw-precomputed": "#6baed6",
    "pot-entropic-gpu":    "#d94801",
    "pot-exact-gpu":       "#f16913",
}
SOLVER_LABEL = {
    "torchgw-landmark":    "torchgw-landmark (ε=5e-2)",
    "torchgw-dijkstra":    "torchgw-dijkstra (ε=5e-2)",
    "torchgw-precomputed": "torchgw-precomputed (ε=5e-2)",
    "pot-entropic-gpu":    "POT-entropic (GPU)",
    "pot-exact-gpu":       "POT-exact (GPU)",
}


def load_cells():
    cells = []
    for p in sorted(RES.glob("*.json")):
        if p.name.startswith("_"):
            continue
        cells.append(json.loads(p.read_text()))
    return cells


def pool_mean_metrics(cells, metric_key):
    """Return {solver: [per-cell metric value]}. Each cell already has
    a per-cell scalar; we gather across cells (each cell = 1 pair ×
    1 seed) to CDF over all cells."""
    out = {s: [] for s in SOLVER_ORDER}
    for c in cells:
        s = c["solver"]
        out[s].append(c["metrics"][metric_key])
    return out


def plot_cdf(ax, data: dict, xlabel: str, title: str, xmax=None):
    for s in SOLVER_ORDER:
        vals = np.sort(np.asarray(data[s]))
        if vals.size == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, drawstyle="steps-post",
                 color=SOLVER_COLOR[s], linewidth=1.8,
                 label=SOLVER_LABEL[s])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of (pair, seed) cells ≤ x")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if xmax is not None:
        ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower right")


def main():
    cells = load_cells()
    geo = pool_mean_metrics(cells, "mean_geo_err_norm")
    pd = pool_mean_metrics(cells, "pair_distortion_mean")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_cdf(axes[0], geo,
              xlabel="normalised geodesic error (mean per cell)",
              title="Supervised: geodesic error CDF (lower → left = better)",
              xmax=0.6)
    plot_cdf(axes[1], pd,
              xlabel="pair distortion (mean per cell)",
              title="Unsupervised GW-native: pair distortion CDF (lower → left = better)",
              xmax=0.2)
    fig.suptitle("C6 TACO — principled evaluation · 18 pairs × 3 seeds × 5 solvers",
                  fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / "c6_principled_eval.png"
    fig.savefig(out, dpi=170)
    print(f"[c6-plot] wrote {out}")

    # Print summary table
    summary = json.loads((RES / "_summary.json").read_text())
    print(f"\n{'solver':<22} {'geo_err_mean':>13} {'pair_dist_mean':>16}")
    for s in SOLVER_ORDER:
        d = summary[s]
        print(f"{s:<22} {d['geo_err_mean']:>13.4f} {d['pair_dist_mean']:>16.4f}")


if __name__ == "__main__":
    main()
