#!/usr/bin/env python
"""Generate C5 word-embedding benchmark figures.

Figure 1  docs/figures/c5_bench.png
    Grouped bars: P@1-CSLS by solver × N, one panel per language pair.
    Paper targets annotated with a note that paper used 20k+Procrustes.

Figure 2  docs/figures/c5_msamples.png
    P@1-CSLS vs M (log-x) for torchgw-precomputed at en-es N=5000.
    pot-entropic-gpu baseline as dashed horizontal line.
    Mirrors c2_msamples_sweep.png styling.
"""
from __future__ import annotations
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO / "results" / "c5_word_embedding"
MS_SUMMARY = REPO / "results" / "c5_msamples" / "_summary.json"
FIG_DIR = REPO / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = ["en-es", "en-fi"]
# Paper targets (Alvarez-Melis & Jaakkola 2018, 20k words + Procrustes init)
PAPER_TARGETS = {"en-es": 0.81, "en-fi": 0.28}

SOLVER_ORDER = [
    "pot-exact-gpu",
    "pot-entropic-gpu",
    "torchgw-precomputed",
    "torchgw-dijkstra",
    "torchgw-landmark",
]
SOLVER_LABELS = {
    "pot-exact-gpu":       "POT-exact",
    "pot-entropic-gpu":    "POT-entropic",
    "torchgw-precomputed": "torchgw-precomp",
    "torchgw-dijkstra":    "torchgw-dijkstra",
    "torchgw-landmark":    "torchgw-landmark",
}
SOLVER_COLORS = {
    "pot-exact-gpu":       "#1f77b4",
    "pot-entropic-gpu":    "#ff7f0e",
    "torchgw-precomputed": "#2ca02c",
    "torchgw-dijkstra":    "#d62728",
    "torchgw-landmark":    "#9467bd",
}
N_LIST = [2000, 5000, 10000]


# ---------------------------------------------------------------------------
# Figure 1: grouped bar chart
# ---------------------------------------------------------------------------

def load_bench_data() -> dict:
    """Returns {pair: {solver: {N: (mean, std)}}}"""
    data: dict = {p: {s: {} for s in SOLVER_ORDER} for p in PAIRS}
    files = list(BENCH_DIR.glob("core_05_word_embedding__*.json"))
    if not files:
        warnings.warn(f"No bench JSONs found in {BENCH_DIR}")
        return data

    agg: dict = defaultdict(list)
    for fp in files:
        try:
            d = json.loads(fp.read_text())
        except Exception as e:
            warnings.warn(f"Could not parse {fp.name}: {e}")
            continue
        if d.get("status") == "fail":
            warnings.warn(f"Failed cell: {fp.name}")
            continue
        task = d.get("metrics", {}).get("task", {})
        p1 = task.get("p1_csls") or task.get("test_p1_csls")
        if p1 is None:
            warnings.warn(f"No p1_csls in {fp.name}")
            continue
        solver = d["solver"]
        pair = d.get("dataset", {}).get("pair") or d.get("subset", "").split("_")[0]
        n = d.get("dataset", {}).get("n_words")
        if pair not in PAIRS or solver not in SOLVER_ORDER or n not in N_LIST:
            continue
        agg[(pair, solver, n)].append(float(p1))

    for (pair, solver, n), vals in agg.items():
        arr = np.array(vals)
        data[pair][solver][n] = (float(arr.mean()), float(arr.std()))

    return data


def make_bench_figure(data: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    x = np.arange(len(N_LIST))
    n_solvers = len(SOLVER_ORDER)
    bar_w = 0.75 / n_solvers
    offsets = np.linspace(-(n_solvers - 1) * bar_w / 2,
                           (n_solvers - 1) * bar_w / 2, n_solvers)

    for ax_idx, pair in enumerate(PAIRS):
        ax = axes[ax_idx]
        any_data = False
        for si, solver in enumerate(SOLVER_ORDER):
            means, stds = [], []
            for n in N_LIST:
                if n in data[pair][solver]:
                    m, s = data[pair][solver][n]
                    means.append(m)
                    stds.append(s)
                else:
                    means.append(np.nan)
                    stds.append(0.0)
            if any(not np.isnan(m) for m in means):
                any_data = True
            ax.bar(x + offsets[si], means, bar_w,
                   yerr=stds, capsize=3,
                   color=SOLVER_COLORS[solver], alpha=0.85,
                   label=SOLVER_LABELS[solver], error_kw={"elinewidth": 1.1})

        # Paper target annotation
        tgt = PAPER_TARGETS[pair]
        ax.axhline(tgt, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(len(N_LIST) - 0.05, tgt + 0.01,
                f"Paper target {tgt}\n(20k + Procrustes)",
                ha="right", va="bottom", fontsize=8, color="black", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in N_LIST])
        ax.set_xlabel("N (vocabulary size)")
        ax.set_ylabel("P@1-CSLS")
        ax.set_title(f"Pair: {pair}")
        ax.set_ylim(0, max(tgt * 1.25, 0.15))
        ax.grid(True, alpha=0.3, axis="y")
        if ax_idx == 0:
            ax.legend(fontsize=8, loc="upper left")
        if not any_data:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")

    fig.suptitle(
        "C5 Word-Embedding Cross-Lingual Alignment — P@1-CSLS by Solver × N\n"
        r"(ε=5×10$^{-4}$; paper targets used 20k words + Procrustes init)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "c5_bench.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"[c5-plot] wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2: M_samples sweep
# ---------------------------------------------------------------------------

def make_msamples_figure():
    if not MS_SUMMARY.exists():
        warnings.warn(f"M_samples summary not found: {MS_SUMMARY}")
        return

    data = json.loads(MS_SUMMARY.read_text())
    torchgw_rows = [r for r in data if r["solver"] == "torchgw-precomputed"]
    pot_rows = [r for r in data if r["solver"] == "pot-entropic-gpu"]

    if not torchgw_rows:
        warnings.warn("No torchgw rows in M_samples summary")
        return

    torchgw_rows.sort(key=lambda r: r["M"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    color_tgw = "#2ca02c"
    color_pot = "#ff7f0e"

    # Left: P@1-CSLS vs M
    ax = axes[0]
    xs = [r["M"] for r in torchgw_rows]
    ys = [r["p1_csls_mean"] for r in torchgw_rows]
    es = [r["p1_csls_std"] for r in torchgw_rows]
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color=color_tgw,
                label="torchgw-precomputed", capsize=3, linewidth=1.6, markersize=6)

    if pot_rows:
        # Aggregate across seeds if multiple
        pot_mean = np.mean([r["p1_csls_mean"] for r in pot_rows])
        ax.axhline(pot_mean, color=color_pot, linestyle="--",
                   linewidth=1.2, alpha=0.7,
                   label=f"POT-entropic baseline: {pot_mean:.3f}")
        ax.text(xs[-1] * 1.05 if xs else 5000, pot_mean,
                f"POT: {pot_mean:.3f}",
                color=color_pot, fontsize=8, va="center")

    ax.set_xscale("log")
    ax.set_xlabel("M_samples (torchgw per-iter cost rows)")
    ax.set_ylabel("P@1-CSLS")
    ax.set_title(f"Quality vs M_samples\n(en-es N={torchgw_rows[0].get('n', 5000)}, "
                 r"3 seeds, ε=5×10$^{-4}$)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(bottom=0)

    # Right: wall_s vs M
    ax = axes[1]
    ys_wall = [r["wall_s_mean"] for r in torchgw_rows]
    ax.plot(xs, ys_wall, "o-", color=color_tgw,
            label="torchgw-precomputed", linewidth=1.6, markersize=6)
    if pot_rows:
        pot_wall = np.mean([r["wall_s_mean"] for r in pot_rows])
        ax.axhline(pot_wall, color=color_pot, linestyle="--",
                   linewidth=1.2, alpha=0.7,
                   label=f"POT-entropic: {pot_wall:.1f}s")
        ax.text(xs[-1] * 1.05 if xs else 5000, pot_wall,
                f"POT: {pot_wall:.1f}s",
                color=color_pot, fontsize=8, va="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("M_samples")
    ax.set_ylabel("wall_s (log)")
    ax.set_title("Cost vs M_samples")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper left")

    fig.suptitle(
        "C5 torchgw M_samples sweep — per-iter cost-row sampling knob\n"
        "(en-es, N=5000, 3 seeds)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIG_DIR / "c5_msamples.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"[c5-plot] wrote {out}")


# ---------------------------------------------------------------------------

def main():
    print("[c5-plot] building bench figure ...", flush=True)
    data = load_bench_data()
    make_bench_figure(data)

    print("[c5-plot] building M_samples figure ...", flush=True)
    make_msamples_figure()

    print("[c5-plot] done.", flush=True)


if __name__ == "__main__":
    main()
