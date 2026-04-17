#!/usr/bin/env python
"""Plot the C6 TACO shape-correspondence benchmark.

Reads results/c6_shape/*.json (one per (solver, pair, seed)) and produces
c6_shape_benchmark.png — two panels:

  left:  accuracy curve (% matches within τ × diameter), mean across
         pairs/seeds per solver
  right: grouped bar chart of mean normalised geodesic error per solver
         across pairs (median and mean annotated)

Usage:
    python scripts/experiments/make_c6_shape_plot.py
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
DEFAULT_RESULTS = REPO / "results" / "c6_shape"
FIG_DIR = REPO / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SOLVER_COLOR = {
    "torchgw-landmark":    "#08306b",
    "torchgw-dijkstra":    "#2171b5",
    "torchgw-precomputed": "#6baed6",
    "pot-entropic-gpu":    "#d94801",
    "pot-exact-gpu":       "#f16913",
}
SOLVER_LABEL = {
    "torchgw-landmark":    "torchgw-landmark",
    "torchgw-dijkstra":    "torchgw-dijkstra",
    "torchgw-precomputed": "torchgw-precomputed",
    "pot-entropic-gpu":    "POT-entropic (GPU)",
    "pot-exact-gpu":       "POT-exact (GPU)",
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


def aggregate(recs: list[dict]):
    """Return {solver: {'errs': [per-run mean_norm_err], 'curves': [per-run accuracy curve]}}"""
    out = defaultdict(lambda: {"errs": [], "curves": [], "walls": []})
    for r in recs:
        s = r["solver"]
        t = r["metrics"]["task"]
        e = float(t["mean_err_normalised"])
        out[s]["errs"].append(e)
        out[s]["curves"].append([(float(tau), float(f)) for tau, f in t["accuracy_curve"]])
        out[s]["walls"].append(float(r["metrics"]["efficiency"]["wall_s_total"]))
    return out


def plot(agg, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: accuracy curves, mean ± σ across runs per solver
    for s in SOLVER_ORDER:
        if s not in agg:
            continue
        curves = agg[s]["curves"]
        if not curves:
            continue
        taus = np.asarray([c[0] for c in curves[0]])
        mat = np.asarray([[f for _, f in c] for c in curves])
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        axes[0].plot(taus, mean, "o-", color=SOLVER_COLOR[s], label=SOLVER_LABEL[s],
                      linewidth=1.8, markersize=5)
        axes[0].fill_between(taus, mean - std, mean + std, color=SOLVER_COLOR[s], alpha=0.15)
    axes[0].set_xlabel("geodesic error threshold τ (× diameter)")
    axes[0].set_ylabel("fraction of correspondences ≤ τ")
    axes[0].set_title("Accuracy curve — mean ± σ over pairs × seeds")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].set_xlim(0, 0.25)
    axes[0].set_ylim(0, 1)

    # Right: bar chart of mean normalised error per solver
    means, stds, labels, colors = [], [], [], []
    for s in SOLVER_ORDER:
        if s not in agg:
            continue
        errs = np.asarray(agg[s]["errs"])
        means.append(errs.mean()); stds.append(errs.std())
        labels.append(SOLVER_LABEL[s]); colors.append(SOLVER_COLOR[s])
    x = np.arange(len(labels))
    axes[1].bar(x, means, yerr=stds, color=colors, capsize=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=90, fontsize=9, ha="center")
    axes[1].set_ylabel("normalised mean geodesic error\n(lower is better)")
    axes[1].set_title("Aggregate error per solver")
    axes[1].grid(True, alpha=0.3, axis="y")
    # Random baseline ≈ 0.5 * diameter. Annotate.
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    axes[1].text(len(labels) - 0.5, 0.51, "random", color="gray", fontsize=8, ha="right")

    fig.suptitle("C6 TACO shape correspondence — N=2000, pure GW, 5 solvers",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"[c6-plot] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out", type=Path, default=FIG_DIR / "c6_shape_benchmark.png")
    args = ap.parse_args()

    recs = load(args.in_dir)
    if not recs:
        raise SystemExit(f"no records in {args.in_dir}")
    agg = aggregate(recs)
    plot(agg, args.out)


if __name__ == "__main__":
    main()
