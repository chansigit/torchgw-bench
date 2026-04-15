#!/usr/bin/env python
"""Plot the C3 anytime Pareto sweep.

Reads results/c3_anytime/*.json (one record per (solver, max_iter, seed))
and produces one figure:

  c3_anytime_pareto.png — wall_s_solve (x, log) vs quality (y).
  Two panels:
    - left: tail Spearman ρ
    - right: 1 - tail ρ (log scale)
  One line per solver, points = max_iter steps (connected), shaded band
  = ±1σ across seeds. Pareto-optimal solvers sit bottom-right (left panel
  high y; right panel low y).

Usage:
    python scripts/experiments/make_c3_anytime_plot.py
    python scripts/experiments/make_c3_anytime_plot.py --in path/to/dir
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
DEFAULT_RESULTS = REPO / "results" / "c3_anytime"
FIG_DIR = REPO / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SOLVER_COLOR = {
    "torchgw-landmark":    "#08306b",
    "torchgw-dijkstra":    "#2171b5",
    "torchgw-precomputed": "#6baed6",
    "pot-entropic-gpu":    "#d94801",
    "pot-exact-gpu":       "#f16913",
    "pot-bapg-gpu":        "#fd8d3c",
}
SOLVER_LABEL = {
    "torchgw-landmark":    "torchgw-landmark",
    "torchgw-dijkstra":    "torchgw-dijkstra",
    "torchgw-precomputed": "torchgw-precomputed",
    "pot-entropic-gpu":    "POT-entropic (GPU)",
    "pot-exact-gpu":       "POT-exact (GPU)",
    "pot-bapg-gpu":        "POT-BAPG (GPU)",
}
SOLVER_ORDER = list(SOLVER_LABEL)


def load_records(results_dir: Path) -> list[dict]:
    records = []
    for p in sorted(results_dir.glob("*.json")):
        d = json.loads(p.read_text())
        if d.get("status") == "skip" or d.get("status") == "fail":
            continue
        records.append(d)
    return records


def aggregate(records: list[dict]) -> dict:
    """Return {solver: [(max_iter, mean_wall, std_wall, mean_rho, std_rho), ...]}
    sorted by max_iter."""
    bucket: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for r in records:
        solver = r["solver"]
        mi = int(r["hyperparams"].get("max_iter", -1))
        if mi <= 0:
            continue
        wall = float(r["metrics"]["efficiency"]["wall_s_solve"])
        rho = r["metrics"]["task"]["tail_arclen_spearman"]
        if rho is None or not np.isfinite(rho):
            continue
        bucket[(solver, mi)].append((wall, float(rho)))

    out: dict[str, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for (solver, mi), pairs in bucket.items():
        arr = np.asarray(pairs)  # [n_seeds, 2]
        walls = arr[:, 0]
        rhos = arr[:, 1]
        out[solver].append((mi, walls.mean(), walls.std(), rhos.mean(), rhos.std()))
    for solver in out:
        out[solver].sort(key=lambda t: t[0])
    return out


def plot_pareto(agg: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for solver in SOLVER_ORDER:
        if solver not in agg:
            continue
        trace = np.asarray(agg[solver])  # [n_iters, 5]
        mi, mw, sw, mr, sr = trace.T

        color = SOLVER_COLOR[solver]
        label = SOLVER_LABEL[solver]

        # Left panel: tail-ρ vs wall_s_solve
        ax = axes[0]
        ax.errorbar(mw, mr, xerr=sw, yerr=sr,
                    fmt="o-", color=color, label=label,
                    markersize=5, linewidth=1.5, capsize=2, alpha=0.9)
        # Annotate endpoints with their max_iter
        for x, y, m in zip(mw, mr, mi):
            if m in (int(mi.min()), int(mi.max())):
                ax.annotate(f"{int(m)}", (x, y),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=7, color=color)

        # Right panel: 1-ρ (log) vs wall_s_solve
        ax = axes[1]
        gap = np.clip(1.0 - mr, 1e-5, None)
        ax.plot(mw, gap, "o-", color=color, label=label,
                markersize=5, linewidth=1.5, alpha=0.9)

    axes[0].set_xscale("log")
    axes[0].set_xlabel("wall_s_solve (s, log)")
    axes[0].set_ylabel("tail Spearman ρ")
    axes[0].set_title("Quality vs compute — raw ρ")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("wall_s_solve (s, log)")
    axes[1].set_ylabel("1 − tail ρ  (log)")
    axes[1].set_title("Quality gap vs compute — Pareto view")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.suptitle("C3 anytime Pareto — tail ρ vs solve time (points: max_iter ∈ {5..500})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"[anytime] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out", type=Path, default=FIG_DIR / "c3_anytime_pareto.png")
    args = ap.parse_args()

    records = load_records(args.in_dir)
    if not records:
        raise SystemExit(f"no records in {args.in_dir}")
    agg = aggregate(records)
    plot_pareto(agg, args.out)


if __name__ == "__main__":
    main()
