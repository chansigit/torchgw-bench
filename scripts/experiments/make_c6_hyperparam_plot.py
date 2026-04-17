#!/usr/bin/env python
"""Plot the C6 torchgw-dijkstra hyperparameter sensitivity study.

Reads results/c6_maxiter/*.json and produces c6_hyperparam_sweep.png —
two panels:

  left:  mean normalised geodesic error vs max_iter (at k=5)
  right: mean normalised geodesic error vs k (at max_iter=500)

Horizontal dashed line on both panels: pot-exact-gpu baseline (the
target torchgw is trying to close).
"""
from __future__ import annotations
import json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results" / "c6_maxiter"
FIG_DIR = REPO / "docs" / "figures"


def load_vals():
    iter_rows: dict[int, list[float]] = defaultdict(list)
    k_rows: dict[int, list[float]] = defaultdict(list)
    pot_vals: list[float] = []
    for f in RES.glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("status") != "ok":
            continue
        e = float(d["metrics"]["task"]["mean_err_normalised"])
        name = f.name
        if "pot-exact-gpu" in name:
            pot_vals.append(e)
            continue
        m_k = re.search(r"iter(\d+)_k(\d+)", name)
        if m_k:
            mi, k = int(m_k.group(1)), int(m_k.group(2))
            if k == 5:
                iter_rows[mi].append(e)
            if mi == 500:
                k_rows[k].append(e)
            continue
        m_plain = re.search(r"__iter(\d+)\.json$", name)
        if m_plain:
            mi = int(m_plain.group(1))
            iter_rows[mi].append(e)
    return iter_rows, k_rows, pot_vals


def main():
    iter_rows, k_rows, pot_vals = load_vals()
    pot_mean = float(np.mean(pot_vals)) if pot_vals else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # max_iter panel
    ax = axes[0]
    mis = sorted(iter_rows)
    means = [np.mean(iter_rows[m]) for m in mis]
    stds = [np.std(iter_rows[m]) for m in mis]
    ax.errorbar(mis, means, yerr=stds, fmt="o-", color="#2171b5",
                 label="torchgw-dijkstra", capsize=3, linewidth=1.6, markersize=6)
    ax.axhline(pot_mean, color="#f16913", linestyle="--", linewidth=1.2,
                label=f"pot-exact-gpu (baseline = {pot_mean:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("max_iter (log)")
    ax.set_ylabel("normalised mean geodesic error")
    ax.set_title("torchgw-dijkstra · max_iter sweep (k=5)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)

    # k panel
    ax = axes[1]
    ks = sorted(k_rows)
    means = [np.mean(k_rows[k]) for k in ks]
    stds = [np.std(k_rows[k]) for k in ks]
    ax.errorbar(ks, means, yerr=stds, fmt="o-", color="#2171b5",
                 label="torchgw-dijkstra", capsize=3, linewidth=1.6, markersize=6)
    ax.axhline(pot_mean, color="#f16913", linestyle="--", linewidth=1.2,
                label=f"pot-exact-gpu (baseline = {pot_mean:.3f})")
    ax.set_xlabel("k (kNN for geodesic graph)")
    ax.set_ylabel("normalised mean geodesic error")
    ax.set_title("torchgw-dijkstra · k sweep (max_iter=500)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("C6 TACO — torchgw-dijkstra hyperparameter sensitivity\n"
                  "3 pairs × 3 seeds, N=2000, force-full",
                  fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "c6_hyperparam_sweep.png"
    fig.savefig(out, dpi=170)
    print(f"[c6-hp] wrote {out}")


if __name__ == "__main__":
    main()
