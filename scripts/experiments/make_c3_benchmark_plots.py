#!/usr/bin/env python
"""Plot the C3 Y-fork benchmark sweep (E1 + E2).

Reads results/c3_benchmark/*.json (one record per (solver, scale, seed)
written by tracks/core/03_branched/run.py) and produces two figures:

  1. e1_solver_shootout.png — N=400, K=500: grouped bar chart with
     mean ± std (over seeds) for branch_accuracy, backbone-ρ, tail-ρ,
     wall_s. One bar group per solver.

  2. e2_scale_sweep.png — wall_s and gpu_peak_gb as a function of N for
     each solver, with shaded ±1σ bands across seeds. Log-log axes.
     Skip records (POT memory guard) appear as missing points.

Usage:
    python scripts/experiments/make_c3_benchmark_plots.py
    python scripts/experiments/make_c3_benchmark_plots.py --results path/to/dir
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
DEFAULT_RESULTS = REPO / "results" / "c3_benchmark"
FIG_DIR = REPO / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 6 FGW variants — torchgw family in blues, POT family in reds.
SOLVER_COLOR = {
    "torchgw-landmark":    "#1f4e8c",  # deep blue
    "torchgw-dijkstra":    "#4a90d9",  # medium blue
    "torchgw-precomputed": "#7cb5ec",  # light blue
    "pot-entropic":        "#8c1a1a",  # dark red
    "pot-exact":           "#d62728",  # medium red
    "pot-bapg":            "#f39c7c",  # light salmon
}
SOLVER_LABEL = {
    "torchgw-landmark":    "torchgw · landmark (kNN+LM geodesic)",
    "torchgw-dijkstra":    "torchgw · dijkstra (kNN geodesic)",
    "torchgw-precomputed": "torchgw · precomputed (Euclidean)",
    "pot-entropic":        "POT · entropic FGW (Sinkhorn)",
    "pot-exact":           "POT · exact FGW (conditional gradient)",
    "pot-bapg":            "POT · BAPG FGW",
}
SOLVER_ORDER = [
    "torchgw-landmark", "torchgw-dijkstra", "torchgw-precomputed",
    "pot-entropic", "pot-exact", "pot-bapg",
]


def load_records(results_dir: Path) -> list[dict]:
    records = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            records.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return records


def _group(records: list[dict], n_src_filter: int | None = None
            ) -> dict[tuple[str, int], list[dict]]:
    """Group records by (solver, n_source). Filter to ok-status only."""
    out: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in records:
        if r.get("status") != "ok":
            continue
        n_src = int(r.get("dataset", {}).get("n_source", -1))
        if n_src_filter is not None and n_src != n_src_filter:
            continue
        solver = str(r.get("solver", "?"))
        out[(solver, n_src)].append(r)
    return out


def _metric(r: dict, dotted_path: str, default=None):
    cur = r
    for k in dotted_path.split("."):
        cur = cur.get(k, {}) if isinstance(cur, dict) else {}
    return cur if not isinstance(cur, dict) else default


# ---- Figure 1: E1 solver shootout ---------------------------------------

def make_e1_figure(records: list[dict]) -> Path:
    grp = _group(records, n_src_filter=400)

    metrics = [
        # ylim chosen to include negative values, since pure GW can flip
        # orientation at small N — we want the failure to be visible.
        ("branch_accuracy",       "metrics.task.branch_accuracy",        "branch accuracy",  (0.0, 1.05),  "linear"),
        ("backbone_arclen_rho",   "metrics.task.main_arclen_spearman",   r"backbone Spearman $\rho$", (-1.1, 1.1), "linear"),
        ("tail_arclen_rho",       "metrics.task.tail_arclen_spearman",   r"tail Spearman $\rho$",     (-1.1, 1.1), "linear"),
        ("wall_s",                "metrics.efficiency.wall_s",           "wall (s)",          None,         "log"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.2))

    bar_w = 0.7
    x = np.arange(len(SOLVER_ORDER))

    for ax_idx, (key, path, label, ylim, yscale) in enumerate(metrics):
        ax = axes[ax_idx]
        means, stds = [], []
        for solver in SOLVER_ORDER:
            recs = grp.get((solver, 400), [])
            vals = [
                float(_metric(r, path, default=float("nan"))) for r in recs
            ]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=0)))
            else:
                means.append(float("nan"))
                stds.append(0.0)
        colors = [SOLVER_COLOR[s] for s in SOLVER_ORDER]
        ax.bar(x, means, bar_w, yerr=stds, color=colors, edgecolor="black",
               linewidth=0.6, capsize=4, error_kw=dict(lw=1.0, ecolor="#222"))
        ax.set_xticks(x)
        ax.set_xticklabels([s.split("·")[0].strip()
                             for s in (SOLVER_LABEL[s] for s in SOLVER_ORDER)],
                            rotation=15, fontsize=8.5, ha="right")
        ax.set_title(label, fontsize=11)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if yscale == "log":
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25, linestyle=":")
        # Reference lines for the rho panels: zero (sign boundary) and ±0.95
        if "Spearman" in label:
            ax.axhline(0.0, color="#888", lw=0.7, linestyle="-", alpha=0.6)
            ax.axhline(0.95, color="#2ca02c", lw=0.7, linestyle=":", alpha=0.7)
            ax.axhline(-0.95, color="#2ca02c", lw=0.7, linestyle=":", alpha=0.7)
        # Annotate values above bars (or below for negative bars)
        for xi, m, s in zip(x, means, stds):
            if not np.isfinite(m):
                continue
            if yscale == "log":
                ax.text(xi, m * 1.06 + s, f"{m:.3f}", ha="center",
                        va="bottom", fontsize=8.5, color="#222")
                continue
            yspan = ((ylim[1] - ylim[0]) if ylim else 1.0)
            if m >= 0:
                ax.text(xi, m + s + 0.02 * yspan, f"{m:+.3f}",
                        ha="center", va="bottom", fontsize=8.5, color="#222")
            else:
                ax.text(xi, m - s - 0.02 * yspan, f"{m:+.3f}",
                        ha="center", va="top", fontsize=8.5, color="#222")

    # Single legend below the row
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=SOLVER_COLOR[s], ec="black", lw=0.6)
        for s in SOLVER_ORDER
    ]
    fig.legend(legend_handles, [SOLVER_LABEL[s] for s in SOLVER_ORDER],
               loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("E1 — Solver shootout on C3 Y-fork (N=400, K=500, 5 seeds)",
                 fontsize=12.5, fontweight="bold", y=1.00)
    fig.tight_layout()
    out = FIG_DIR / "e1_solver_shootout.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ---- Figure 2: E2 scale sweep -------------------------------------------

def make_e2_figure(records: list[dict]) -> Path:
    # Group by (solver, n_source) collapsing seeds
    by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in records:
        if r.get("status") != "ok":
            continue
        n_src = int(r.get("dataset", {}).get("n_source", -1))
        solver = str(r.get("solver", "?"))
        by_key[(solver, n_src)].append(r)

    scales = sorted({n for (_s, n) in by_key.keys()})

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # Wall time
    ax = axes[0]
    for solver in SOLVER_ORDER:
        xs, ys, errs = [], [], []
        for n in scales:
            recs = by_key.get((solver, n), [])
            walls = [float(_metric(r, "metrics.efficiency.wall_s",
                                     default=float("nan"))) for r in recs]
            walls = [w for w in walls if np.isfinite(w)]
            if walls:
                xs.append(n)
                ys.append(float(np.mean(walls)))
                errs.append(float(np.std(walls, ddof=0)))
        if xs:
            xs_a = np.asarray(xs, dtype=float)
            ys_a = np.asarray(ys, dtype=float)
            er_a = np.asarray(errs, dtype=float)
            ax.plot(xs_a, ys_a, "-o", color=SOLVER_COLOR[solver],
                    label=SOLVER_LABEL[solver], lw=1.8, ms=6,
                    markeredgecolor="black", markeredgewidth=0.5)
            ax.fill_between(xs_a, np.maximum(ys_a - er_a, 1e-3),
                             ys_a + er_a, color=SOLVER_COLOR[solver],
                             alpha=0.18, linewidth=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N (source size)")
    ax.set_ylabel("wall (s)")
    ax.set_title("Wall time vs scale", fontsize=11)
    ax.grid(True, which="both", alpha=0.25, linestyle=":")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)

    # GPU peak
    ax = axes[1]
    for solver in SOLVER_ORDER:
        xs, ys, errs = [], [], []
        for n in scales:
            recs = by_key.get((solver, n), [])
            mems = [_metric(r, "metrics.efficiency.gpu_peak_gb") for r in recs]
            mems = [float(m) for m in mems if m is not None and np.isfinite(float(m))]
            if mems:
                xs.append(n)
                ys.append(float(np.mean(mems)))
                errs.append(float(np.std(mems, ddof=0)))
        if xs:
            xs_a = np.asarray(xs, dtype=float)
            ys_a = np.asarray(ys, dtype=float)
            er_a = np.asarray(errs, dtype=float)
            ax.plot(xs_a, ys_a, "-o", color=SOLVER_COLOR[solver],
                    label=SOLVER_LABEL[solver], lw=1.8, ms=6,
                    markeredgecolor="black", markeredgewidth=0.5)
            ax.fill_between(xs_a, np.maximum(ys_a - er_a, 1e-4),
                             ys_a + er_a, color=SOLVER_COLOR[solver],
                             alpha=0.18, linewidth=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N (source size)")
    ax.set_ylabel("GPU peak memory (GB)")
    ax.set_title("GPU peak memory vs scale (CPU solvers omitted)", fontsize=11)
    ax.grid(True, which="both", alpha=0.25, linestyle=":")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)

    fig.suptitle("E2 — Scale sweep on C3 Y-fork (3 seeds per scale; "
                 "POT skipped above N=5000 via memory guard)",
                 fontsize=12.5, fontweight="bold", y=1.00)
    fig.tight_layout()
    out = FIG_DIR / "e2_scale_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def make_torchgw_vs_pot_figure(records: list[dict]) -> Path:
    """Focused torchgw-vs-POT comparison across N: quality (backbone-ρ,
    tail-ρ) AND cost (wall time, GPU memory) in a single 2×2 publication
    figure. Shaded bands show ±1σ across seeds at each scale.
    """
    # Aggregate: (solver, n_source) -> per-metric list across seeds
    by_key: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in records:
        if r.get("status") != "ok":
            continue
        n = int(r.get("dataset", {}).get("n_source", -1))
        solver = str(r.get("solver", "?"))
        by_key[(solver, n)].append(r)

    scales = sorted({n for (_s, n) in by_key.keys()})

    def _series(solver: str, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs, ys, es = [], [], []
        for n in scales:
            vals = []
            for r in by_key.get((solver, n), []):
                v = _metric(r, path, default=None)
                if v is None:
                    continue
                try:
                    vf = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(vf):
                    vals.append(vf)
            if vals:
                xs.append(n)
                ys.append(float(np.mean(vals)))
                es.append(float(np.std(vals, ddof=0)))
        return np.asarray(xs, float), np.asarray(ys, float), np.asarray(es, float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    panel_spec = [
        # (ax_idx, path, ylabel, yscale, ylim, threshold)
        ((0, 0), "metrics.efficiency.wall_s",           "wall (s)",               "log",    None,         None),
        ((0, 1), "metrics.efficiency.gpu_peak_gb",      "GPU peak memory (GB)",   "log",    None,         None),
        ((1, 0), "metrics.task.main_arclen_spearman",   r"backbone Spearman $\rho$", "linear", (-1.1, 1.1), 0.95),
        ((1, 1), "metrics.task.tail_arclen_spearman",   r"tail Spearman $\rho$",     "linear", (-1.1, 1.1), 0.95),
    ]

    for (r, c), path, ylabel, yscale, ylim, thr in panel_spec:
        ax = axes[r, c]
        any_data = False
        for solver in SOLVER_ORDER:
            xs, ys, es = _series(solver, path)
            if len(xs) == 0:
                continue
            any_data = True
            ax.plot(xs, ys, "-o", color=SOLVER_COLOR[solver],
                    label=SOLVER_LABEL[solver], lw=1.8, ms=6,
                    markeredgecolor="black", markeredgewidth=0.5)
            lo = ys - es
            hi = ys + es
            if yscale == "log":
                lo = np.maximum(lo, 1e-4)
            ax.fill_between(xs, lo, hi, color=SOLVER_COLOR[solver],
                            alpha=0.18, linewidth=0)
        ax.set_xscale("log")
        if yscale == "log":
            ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(*ylim)
        if thr is not None:
            ax.axhline(thr, color="#2ca02c", lw=0.8, linestyle=":",
                       alpha=0.8, zorder=0)
            ax.axhline(-thr, color="#2ca02c", lw=0.8, linestyle=":",
                       alpha=0.8, zorder=0)
            ax.axhline(0.0, color="#888", lw=0.6, linestyle="-",
                       alpha=0.5, zorder=0)
        ax.set_xlabel("N (source size)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25, linestyle=":")
        if any_data and (r, c) == (0, 0):
            ax.legend(loc="upper left", fontsize=8.5, frameon=False)

    fig.suptitle("torchgw vs POT across scale — quality (bottom row) and cost "
                 "(top row), mean ± σ over 3–5 seeds",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    out = FIG_DIR / "torchgw_vs_pot.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot C3 benchmark sweep")
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = ap.parse_args()

    records = load_records(args.results)
    if not records:
        raise SystemExit(f"no records found in {args.results}/")

    print(f"[plot] loaded {len(records)} records from {args.results}/")
    p1 = make_e1_figure(records)
    print(f"[plot] wrote {p1}")
    p2 = make_e2_figure(records)
    print(f"[plot] wrote {p2}")
    p3 = make_torchgw_vs_pot_figure(records)
    print(f"[plot] wrote {p3}")


if __name__ == "__main__":
    main()
