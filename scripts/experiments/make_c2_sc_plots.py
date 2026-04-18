#!/usr/bin/env python
"""C2 benchmark plots: FOSCTTM vs N (scale sweep) and FOSCTTM vs ε
(sensitivity sweep) on 10x PBMC Multiome.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SC_DIR = REPO / "results" / "c2_sc_lda"   # LDA preprocessing is canonical
EPS_DIR = REPO / "results" / "c2_eps"
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
    "torchgw-precomputed": "torchgw-precomputed (SCOT cost)",
    "pot-entropic-gpu":    "POT-entropic (GPU)",
    "pot-exact-gpu":       "POT-exact (GPU)",
}
SOLVER_ORDER = list(SOLVER_LABEL)


def load_all(results_dir: Path):
    records = []
    for p in results_dir.glob("*.json"):
        d = json.loads(p.read_text())
        if d.get("status") != "ok":
            continue
        records.append(d)
    return records


def plot_scale():
    recs = load_all(SC_DIR)
    # group by (solver, N) → list of FOSCTTM values
    foscttm_by = defaultdict(list)
    wall_by = defaultdict(list)
    for r in recs:
        s = r["solver"]; n = r["dataset"]["n_cells"]
        foscttm_by[(s, n)].append(r["metrics"]["task"]["foscttm"])
        wall_by[(s, n)].append(r["metrics"]["efficiency"]["wall_s_total"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    Ns = sorted({n for (_, n) in foscttm_by})

    for s in SOLVER_ORDER:
        ys_f = [np.mean(foscttm_by[(s, n)]) for n in Ns]
        stds_f = [np.std(foscttm_by[(s, n)]) for n in Ns]
        ys_w = [np.mean(wall_by[(s, n)]) for n in Ns]
        axes[0].errorbar(Ns, ys_f, yerr=stds_f, fmt="o-", color=SOLVER_COLOR[s],
                          label=SOLVER_LABEL[s], capsize=3, linewidth=1.6,
                          markersize=6)
        axes[1].plot(Ns, ys_w, "o-", color=SOLVER_COLOR[s],
                      label=SOLVER_LABEL[s], linewidth=1.6, markersize=6)
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[0].text(Ns[-1] * 0.92, 0.505, "random", color="gray", fontsize=8)
    axes[0].axhline(0.12, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
    axes[0].text(Ns[-1] * 0.8, 0.13, "SCOT+ literature", color="green", fontsize=8)
    axes[0].set_xscale("log"); axes[0].set_xlabel("N cells (log)")
    axes[0].set_ylabel("FOSCTTM (lower = better)")
    axes[0].set_title("Quality vs scale — 3 seeds per cell")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 0.75)

    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("N cells (log)")
    axes[1].set_ylabel("wall_s_total (log)")
    axes[1].set_title("Cost vs scale")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend(fontsize=8)

    fig.suptitle("C2 PBMC Multiome — cross-modality GW alignment", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / "c2_sc_benchmark.png"
    fig.savefig(out, dpi=170)
    print(f"[c2-plot] wrote {out}")


def plot_eps():
    recs = load_all(EPS_DIR)
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        s = r["solver"]
        eps = r["hyperparams"].get("epsilon")
        if eps is None:
            continue
        by[s][float(eps)].append(r["metrics"]["task"]["foscttm"])

    # pot-exact baseline (no eps)
    baseline = [r["metrics"]["task"]["foscttm"] for r in recs
                 if r["solver"] == "pot-exact-gpu"]
    pot_exact_mean = np.mean(baseline) if baseline else None

    fig, ax = plt.subplots(figsize=(8, 5))
    for s in ["torchgw-precomputed", "pot-entropic-gpu"]:
        eps_vals = sorted(by[s].keys())
        means = [np.mean(by[s][e]) for e in eps_vals]
        stds = [np.std(by[s][e]) for e in eps_vals]
        ax.errorbar(eps_vals, means, yerr=stds, fmt="o-",
                     color=SOLVER_COLOR[s], label=SOLVER_LABEL[s],
                     capsize=3, linewidth=1.8, markersize=7)
    if pot_exact_mean is not None:
        ax.axhline(pot_exact_mean, color=SOLVER_COLOR["pot-exact-gpu"],
                    linestyle="--", linewidth=1.3,
                    label=f"POT-exact baseline = {pot_exact_mean:.3f}")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(5e-4, 0.505, "random", color="gray", fontsize=8)
    ax.axhline(0.12, color="green", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(5e-4, 0.13, "SCOT+ literature", color="green", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("ε (entropic regularisation)")
    ax.set_ylabel("FOSCTTM (lower = better)")
    ax.set_title("C2 ε sensitivity · N=2000, 3 seeds")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 0.6)
    fig.tight_layout()
    out = FIG_DIR / "c2_sc_eps.png"
    fig.savefig(out, dpi=170)
    print(f"[c2-plot] wrote {out}")


def main():
    plot_scale()
    plot_eps()


if __name__ == "__main__":
    main()
