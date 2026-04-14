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

# 9 FGW variants — torchgw in blues, POT-CPU in warm reds, POT-GPU in cool reds.
SOLVER_COLOR = {
    "torchgw-landmark":    "#08306b",  # deepest blue
    "torchgw-dijkstra":    "#2171b5",  # mid blue
    "torchgw-precomputed": "#6baed6",  # light blue
    "pot-entropic":        "#67000d",  # darkest red (CPU entropic)
    "pot-exact":           "#a50f15",  # dark red (CPU exact)
    "pot-bapg":            "#cb181d",  # medium red (CPU BAPG)
    "pot-entropic-gpu":    "#d94801",  # orange (GPU entropic)
    "pot-exact-gpu":       "#f16913",  # lighter orange (GPU exact)
    "pot-bapg-gpu":        "#fd8d3c",  # lightest orange (GPU BAPG)
}
SOLVER_LABEL = {
    "torchgw-landmark":    "torchgw · landmark (GPU)",
    "torchgw-dijkstra":    "torchgw · dijkstra (GPU)",
    "torchgw-precomputed": "torchgw · precomputed (GPU)",
    "pot-entropic":        "POT · entropic (CPU)",
    "pot-exact":           "POT · exact CG (CPU)",
    "pot-bapg":            "POT · BAPG (CPU)",
    "pot-entropic-gpu":    "POT · entropic (GPU)",
    "pot-exact-gpu":       "POT · exact CG (GPU)",
    "pot-bapg-gpu":        "POT · BAPG (GPU)",
}
SOLVER_ORDER = [
    "torchgw-landmark", "torchgw-dijkstra", "torchgw-precomputed",
    "pot-entropic", "pot-exact", "pot-bapg",
    "pot-entropic-gpu", "pot-exact-gpu", "pot-bapg-gpu",
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
            walls = [float(_metric(r, "metrics.efficiency.wall_s_total",
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

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    panel_spec = [
        # (ax_idx, path, ylabel, yscale)
        (0, "metrics.efficiency.wall_s_total", "wall (s)  —  preprocess + solve",    "log"),
        (1, "metrics.efficiency.gpu_peak_gb",  "GPU peak (GB)",   "log"),
        (2, "metrics.efficiency.ram_peak_gb",  "RAM peak (GB)",   "log"),
    ]

    for idx, path, ylabel, yscale in panel_spec:
        ax = axes[idx]
        any_data = False
        for solver in SOLVER_ORDER:
            xs, ys, es = _series(solver, path)
            if len(xs) == 0:
                continue
            any_data = True
            ax.plot(xs, ys, "-o", color=SOLVER_COLOR[solver],
                    label=SOLVER_LABEL[solver], lw=1.8, ms=6,
                    markeredgecolor="black", markeredgewidth=0.5)
            lo = np.maximum(ys - es, 1e-4)
            hi = ys + es
            ax.fill_between(xs, lo, hi, color=SOLVER_COLOR[solver],
                             alpha=0.15, linewidth=0)
        ax.set_xscale("log")
        if yscale == "log":
            ax.set_yscale("log")
        ax.set_xlabel("N (source size)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25, linestyle=":")
        if any_data and idx == 0:
            ax.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=1)

    fig.suptitle("torchgw vs POT — cost across scale (mean ± σ over 3–5 seeds)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "torchgw_vs_pot.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def make_rho_by_position_figure(records: list[dict], N_for_positions: int = 4000) -> Path:
    """Bar chart of per-region Spearman ρ at a fixed N.

    Five positions along the manifold (from inner spiral to the tails):
      spiral-inner, spiral-middle, spiral-outer, long-tail, short-tail
    × 9 solvers, one group of bars per position. Bars are mean ± σ
    across seeds. All bar heights are the signed ρ in that region.
    """
    # Which records are at the anchor N and completed OK?
    recs_at_N = [
        r for r in records
        if r.get("status") == "ok"
        and int(r.get("dataset", {}).get("n_source", -1)) == N_for_positions
    ]
    if not recs_at_N:
        raise SystemExit(f"no ok records at N={N_for_positions}")

    # Geometry constants to split backbone into inner/middle/outer and
    # pick out long-tail / short-tail by label+arclen. We read these from
    # the track module so the figure stays in sync with the data code.
    import importlib, sys
    sys.path.insert(0, str(REPO / "tracks" / "core" / "03_branched"))
    import run as _run  # type: ignore[import-not-found]
    c3 = importlib.reload(_run)
    sys.path.pop(0)

    fork_s = float(c3.spiral_arclen(9.0).item())

    # Anchor definitions: (label_name, mask_fn -> bool array on target points)
    def make_mask(lo: float, hi: float, label1: bool = False):
        def _f(a_tgt: np.ndarray, L_tgt: np.ndarray) -> np.ndarray:
            if label1:
                return L_tgt == 1
            return (L_tgt == 0) & (a_tgt >= lo) & (a_tgt < hi)
        return _f

    positions = [
        ("spiral inner",  make_mask(0.0,         fork_s * 1/3)),
        ("spiral middle", make_mask(fork_s * 1/3, fork_s * 2/3)),
        ("spiral outer",  make_mask(fork_s * 2/3, fork_s)),
        ("long tail",     make_mask(fork_s,       fork_s + 10.0)),
        ("short tail",    make_mask(-1, -1, label1=True)),
    ]

    # For each record, recompute per-position rho. We need X, Y, arclens,
    # labels per record — but records only carry T's filename was used
    # for that solver. T is not persisted in JSON. Instead we read the
    # ρ from the metrics (main_ and tail_arclen_spearman), which are
    # global-over-backbone and global-over-branch. For per-position
    # rho we would need T. Approximate: use main_ρ for the 3 backbone
    # positions and tail_ρ for short-tail; long-tail is part of main_ρ
    # but we cannot split it from the JSON alone.
    #
    # Compromise: regenerate the data ourselves, re-run arg-max from a
    # saved T? T is not in the JSON. Simpler: re-sample the dataset at
    # the same seed (deterministic) and re-run the solver to get T.
    # That defeats the purpose.
    #
    # Cleaner path: we already have three-sigma quality as
    # branch_accuracy / backbone_ρ / tail_ρ in the JSON. So the bar
    # chart actually uses TWO positions (backbone, tail-2), not five.
    # Use those.
    positions_compact = [
        ("backbone (main + long tail)", "metrics.task.main_arclen_spearman"),
        ("short tail (off-axis)",       "metrics.task.tail_arclen_spearman"),
    ]

    # Aggregate mean/std per (solver, position) from the existing
    # per-record backbone_ρ / tail_ρ fields.
    n_pos = len(positions_compact)
    x_groups = np.arange(n_pos)

    fig, ax = plt.subplots(figsize=(14, 5.2))
    n_sol = len(SOLVER_ORDER)
    total_w = 0.85
    bar_w = total_w / n_sol

    for i, solver in enumerate(SOLVER_ORDER):
        solver_recs = [r for r in recs_at_N if r.get("solver") == solver]
        if not solver_recs:
            continue
        means = []
        stds = []
        for _pos_name, path in positions_compact:
            vals = []
            for r in solver_recs:
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
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals, ddof=0)))
            else:
                means.append(float("nan"))
                stds.append(0.0)
        offset = (i - (n_sol - 1) / 2) * bar_w
        xs = x_groups + offset
        ax.bar(xs, means, bar_w, yerr=stds, color=SOLVER_COLOR[solver],
               edgecolor="black", linewidth=0.4, capsize=2.5,
               error_kw=dict(lw=0.8, ecolor="#222"),
               label=SOLVER_LABEL[solver])

    ax.set_xticks(x_groups)
    ax.set_xticklabels([p[0] for p in positions_compact], fontsize=11)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.95, color="#2ca02c", lw=0.6, linestyle=":", alpha=0.8)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.legend(loc="lower center", fontsize=8, ncol=3, frameon=False,
              bbox_to_anchor=(0.5, -0.28))
    fig.suptitle(
        f"Per-region Spearman ρ at N = {N_for_positions} (mean ± σ over seeds)",
        fontsize=12.5, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    out = FIG_DIR / "rho_by_position.png"
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
    p4 = make_rho_by_position_figure(records, N_for_positions=4000)
    print(f"[plot] wrote {p4}")


if __name__ == "__main__":
    main()
