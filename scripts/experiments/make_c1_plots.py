#!/usr/bin/env python
"""C1 point-cloud scalability plots.

Produces 3 headline figures and a summary JSON:
  docs/figures/c1_scalability_wall.png
  docs/figures/c1_scalability_quality.png
  docs/figures/c1_scalability_memory.png
  results/c1_point_cloud_scale/c1_summary.json

Also prints a markdown summary table to stdout.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "results" / "c1_point_cloud_scale"
FIG_DIR = REPO / "docs" / "figures"
SUMMARY_OUT = DATA_DIR / "c1_summary.json"

FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Solver display order & colour palette
# POT solvers → warm (red/orange), torchgw solvers → cool (blue/green)
# ---------------------------------------------------------------------------
SOLVER_ORDER = [
    "pot-exact-gpu",
    "pot-entropic-gpu",
    "torchgw-precomputed",
    "torchgw-landmark",
    "torchgw-lowrank-landmark",
    "torchgw-dijkstra",
    "torchgw-lowrank-dijkstra",
]

SOLVER_COLORS = {
    "pot-exact-gpu":            "#d62728",   # red
    "pot-entropic-gpu":         "#ff7f0e",   # orange
    "torchgw-precomputed":      "#1f77b4",   # blue
    "torchgw-landmark":         "#17becf",   # teal
    "torchgw-lowrank-landmark": "#2ca02c",   # green
    "torchgw-dijkstra":         "#9467bd",   # purple
    "torchgw-lowrank-dijkstra": "#8c564b",   # brown
}

SOLVER_LABELS = {
    "pot-exact-gpu":            "POT-exact",
    "pot-entropic-gpu":         "POT-entropic",
    "torchgw-precomputed":      "TorchGW-precomp",
    "torchgw-landmark":         "TorchGW-landmark",
    "torchgw-lowrank-landmark": "TorchGW-lowrank-lm",
    "torchgw-dijkstra":         "TorchGW-dijkstra",
    "torchgw-lowrank-dijkstra": "TorchGW-lowrank-dij",
}

N_VALUES = [10_000, 20_000, 50_000, 100_000]
N_LABELS = {10_000: "10k", 20_000: "20k", 50_000: "50k", 100_000: "100k"}

H100_GB = 80.0  # GPU ceiling annotation

# ---------------------------------------------------------------------------
# Load all JSONs
# ---------------------------------------------------------------------------

def load_all() -> list[dict]:
    records = []
    for p in sorted(DATA_DIR.glob("*.json")):
        try:
            records.append(json.loads(p.read_text()))
        except Exception:
            pass
    return records


def parse_oom_alloc_gb(error: str | None) -> float | None:
    """Parse 'Tried to allocate X GiB' from OOM error string."""
    if not error:
        return None
    m = re.search(r"Tried to allocate ([0-9.]+)\s*GiB", error)
    if m:
        return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Aggregate: mean ± std across seeds per (solver, N)
# ---------------------------------------------------------------------------

def aggregate(records: list[dict]) -> dict:
    """Return nested dict: agg[solver][N] = dict of aggregated stats."""
    raw: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        solver = r["solver"]
        n = r["dataset"]["n_points"]
        raw[solver][n].append(r)

    agg: dict[str, dict[int, dict]] = {}
    for solver in raw:
        agg[solver] = {}
        for n in raw[solver]:
            cells = raw[solver][n]
            ok_cells = [c for c in cells if c["status"] == "ok"]
            fail_cells = [c for c in cells if c["status"] == "fail"]

            entry: dict = {
                "n": n,
                "n_ok": len(ok_cells),
                "n_fail": len(fail_cells),
                "n_total": len(cells),
                "all_failed": len(ok_cells) == 0 and len(cells) > 0,
            }

            if ok_cells:
                walls = [c["metrics"]["efficiency"]["wall_solve_s"] for c in ok_cells]
                spears = [abs(c["metrics"]["task"]["arclen_spearman"]) for c in ok_cells]
                gpus = [c["metrics"]["efficiency"]["gpu_peak_gb"] for c in ok_cells]

                entry["wall_mean"] = float(np.mean(walls))
                entry["wall_std"] = float(np.std(walls))
                entry["spearman_mean"] = float(np.mean(spears))
                entry["spearman_std"] = float(np.std(spears))
                entry["gpu_mean"] = float(np.mean(gpus))
                entry["gpu_std"] = float(np.std(gpus))
            else:
                entry["wall_mean"] = None
                entry["wall_std"] = None
                entry["spearman_mean"] = None
                entry["spearman_std"] = None
                entry["gpu_mean"] = None
                entry["gpu_std"] = None

            # OOM allocation size from first fail cell with parseable error
            oom_sizes = []
            for c in fail_cells:
                gb = parse_oom_alloc_gb(c.get("error"))
                if gb is not None:
                    oom_sizes.append(gb)
            entry["oom_alloc_gb"] = float(np.mean(oom_sizes)) if oom_sizes else None

            agg[solver][n] = entry

    return agg


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _setup_fig(title: str, xlabel: str, ylabel: str,
               figsize=(9, 5.5), logx=True, logy=True):
    fig, ax = plt.subplots(figsize=figsize)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    # Annotate N tick labels
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels([N_LABELS[n] for n in N_VALUES], fontsize=10)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    return fig, ax


def _plot_solver_line(ax, solver: str, n_agg: dict[int, dict],
                      y_key_mean: str, y_key_std: str | None,
                      marker="o", fail_marker=True):
    """Plot mean±std line for one solver; mark last-good N with ✗ if fails after."""
    color = SOLVER_COLORS[solver]
    label = SOLVER_LABELS[solver]

    ns_sorted = sorted(n_agg.keys())
    xs, ys, yes = [], [], []
    last_ok_n = None
    last_ok_y = None
    has_fail_after = False

    for n in ns_sorted:
        entry = n_agg[n]
        y = entry.get(y_key_mean)
        if y is not None and not entry["all_failed"]:
            xs.append(n)
            ys.append(y)
            ye = entry.get(y_key_std) or 0.0
            yes.append(ye)
            last_ok_n = n
            last_ok_y = y
        else:
            # This N failed (or all_failed)
            if last_ok_n is not None:
                has_fail_after = True

    if not xs:
        return

    yerr = np.array(yes)
    valid_err = yerr > 0
    if valid_err.any():
        ax.errorbar(xs, ys, yerr=yerr, fmt=f"{marker}-",
                    color=color, label=label,
                    capsize=3, linewidth=1.6, markersize=6)
    else:
        ax.plot(xs, ys, f"{marker}-", color=color, label=label,
                linewidth=1.6, markersize=6)

    # Mark the ceiling: ✗ at the last successful N
    if fail_marker and has_fail_after and last_ok_n is not None:
        ax.plot(last_ok_n, last_ok_y, "x", color=color,
                markersize=12, markeredgewidth=2.5, zorder=5)


# ---------------------------------------------------------------------------
# Figure 1: Wall-solve time vs N (log-log)
# ---------------------------------------------------------------------------

def fig_wall(agg: dict, out_path: Path):
    fig, ax = _setup_fig(
        title="Wall-solve time vs N — scalability ceiling by solver",
        xlabel="N (points)",
        ylabel="Wall-solve time (s)",
        logx=True, logy=True,
    )

    for solver in SOLVER_ORDER:
        if solver not in agg:
            continue
        _plot_solver_line(ax, solver, agg[solver],
                          y_key_mean="wall_mean", y_key_std="wall_std",
                          fail_marker=True)

    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"[c1-plots] wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: |Spearman| vs N
# ---------------------------------------------------------------------------

def fig_quality(agg: dict, out_path: Path):
    fig, ax = _setup_fig(
        title="Correspondence quality (|Spearman|) vs N — ceiling + degradation pattern",
        xlabel="N (points)",
        ylabel="|Spearman rank correlation|",
        logx=True, logy=False,
    )
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.9, alpha=0.6,
               label="perfect (1.0)")

    for solver in SOLVER_ORDER:
        if solver not in agg:
            continue
        _plot_solver_line(ax, solver, agg[solver],
                          y_key_mean="spearman_mean", y_key_std="spearman_std",
                          fail_marker=True)

    ax.legend(fontsize=9, loc="lower left", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"[c1-plots] wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: GPU peak memory vs N (log-log)
# ---------------------------------------------------------------------------

def fig_memory(agg: dict, out_path: Path):
    fig, ax = _setup_fig(
        title="GPU peak memory vs N — O(N²) scaling wall",
        xlabel="N (points)",
        ylabel="GPU peak memory (GB)",
        logx=True, logy=True,
    )

    # Annotate H100 80 GB ceiling
    ax.axhline(H100_GB, color="firebrick", linestyle="--", linewidth=1.2,
               alpha=0.75, label="H100 80 GB ceiling")
    ax.text(N_VALUES[0] * 1.05, H100_GB * 1.03, "H100 80 GB",
            color="firebrick", fontsize=9, va="bottom")

    # Draw reference O(N²) guide line through the first OK data point
    # of the most memory-hungry successful solver (torchgw-precomputed)
    ref_solver = "torchgw-precomputed"
    if ref_solver in agg:
        ref_entries = [(n, e) for n, e in agg[ref_solver].items()
                       if e.get("gpu_mean") is not None]
        if ref_entries:
            ref_n, ref_e = sorted(ref_entries)[0]
            ref_y = ref_e["gpu_mean"]
            ns_ref = np.array([1e4, 1e5])
            ys_ref = ref_y * (ns_ref / ref_n) ** 2
            ax.plot(ns_ref, ys_ref, "k:", linewidth=1.0, alpha=0.5,
                    label="O(N²) guide")

    for solver in SOLVER_ORDER:
        if solver not in agg:
            continue
        color = SOLVER_COLORS[solver]
        label = SOLVER_LABELS[solver]
        ns_sorted = sorted(agg[solver].keys())
        xs, ys = [], []
        last_ok_n = None
        last_ok_y = None
        has_fail_after = False

        for n in ns_sorted:
            entry = agg[solver][n]
            y = entry.get("gpu_mean")
            if y is not None and not entry["all_failed"]:
                xs.append(n)
                ys.append(y)
                last_ok_n = n
                last_ok_y = y
            else:
                # Try OOM attempted allocation size
                oom_gb = entry.get("oom_alloc_gb")
                if oom_gb is not None:
                    ax.plot(n, oom_gb, "^", color=color,
                            markersize=8, alpha=0.55, zorder=4)
                if last_ok_n is not None:
                    has_fail_after = True

        if not xs:
            continue

        ax.plot(xs, ys, "o-", color=color, label=label,
                linewidth=1.6, markersize=6)

        if has_fail_after and last_ok_n is not None and last_ok_y is not None:
            ax.plot([last_ok_n], [last_ok_y], "x", color=color,
                    markersize=12, markeredgewidth=2.5, zorder=5)

    ax.legend(fontsize=9, loc="upper left", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"[c1-plots] wrote {out_path}")


# ---------------------------------------------------------------------------
# Markdown summary table
# ---------------------------------------------------------------------------

def print_markdown_tables(agg: dict):
    solvers_present = [s for s in SOLVER_ORDER if s in agg]
    n_cols = [n for n in N_VALUES]

    def fmt_cell_spearman(entry):
        if entry is None:
            return "—"
        if entry["all_failed"] or entry.get("spearman_mean") is None:
            return "FAIL"
        mean = entry["spearman_mean"]
        std = entry["spearman_std"]
        return f"{mean:.3f}±{std:.3f}"

    def fmt_cell_wall(entry):
        if entry is None:
            return "—"
        if entry["all_failed"] or entry.get("wall_mean") is None:
            return "FAIL"
        mean = entry["wall_mean"]
        std = entry["wall_std"]
        if mean >= 3600:
            return f"FAIL"
        unit = "s"
        return f"{mean:.1f}±{std:.1f}{unit}"

    header = "| Solver | " + " | ".join(f"N={N_LABELS[n]}" for n in n_cols) + " |"
    sep = "|---|" + "---|" * len(n_cols)

    print("\n### |Spearman| (quality)\n")
    print(header)
    print(sep)
    for solver in solvers_present:
        row_parts = [SOLVER_LABELS[solver]]
        for n in n_cols:
            entry = agg[solver].get(n)
            row_parts.append(fmt_cell_spearman(entry))
        print("| " + " | ".join(row_parts) + " |")

    print("\n### Wall-solve time (efficiency)\n")
    print(header)
    print(sep)
    for solver in solvers_present:
        row_parts = [SOLVER_LABELS[solver]]
        for n in n_cols:
            entry = agg[solver].get(n)
            row_parts.append(fmt_cell_wall(entry))
        print("| " + " | ".join(row_parts) + " |")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    records = load_all()
    print(f"[c1-plots] loaded {len(records)} JSONs from {DATA_DIR}")

    agg = aggregate(records)

    # Save summary JSON
    # Convert int keys to str for JSON serialisation
    summary_serialisable = {
        solver: {str(n): entry for n, entry in ns.items()}
        for solver, ns in agg.items()
    }
    SUMMARY_OUT.write_text(json.dumps(summary_serialisable, indent=2))
    print(f"[c1-plots] wrote {SUMMARY_OUT}")

    # Produce figures
    fig_wall(agg, FIG_DIR / "c1_scalability_wall.png")
    fig_quality(agg, FIG_DIR / "c1_scalability_quality.png")
    fig_memory(agg, FIG_DIR / "c1_scalability_memory.png")

    # Print markdown tables
    print_markdown_tables(agg)


if __name__ == "__main__":
    main()
