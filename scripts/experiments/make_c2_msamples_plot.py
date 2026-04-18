#!/usr/bin/env python
"""Plot C2 M_samples sweep.

Two panels:
  left: FOSCTTM vs M (log-x) for N ∈ {2000, 5000}, with POT baseline
  right: wall_s vs M (log-log) for same solvers.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
SUMMARY = REPO / "results" / "c2_msamples" / "_summary.json"
FIG_DIR = REPO / "docs" / "figures"

COLOR_BY_N = {2000: "#08306b", 5000: "#d94801"}


def main():
    data = json.loads(SUMMARY.read_text())
    by_n = defaultdict(list)
    baselines = {}
    for r in data:
        if r["solver"] == "pot-entropic-gpu":
            baselines[r["n"]] = r
        else:
            by_n[r["n"]].append(r)

    for n in by_n:
        by_n[n].sort(key=lambda x: x["M"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: FOSCTTM vs M
    ax = axes[0]
    for n in sorted(by_n):
        xs = [r["M"] for r in by_n[n]]
        ys = [r["foscttm_mean"] for r in by_n[n]]
        es = [r["foscttm_std"] for r in by_n[n]]
        color = COLOR_BY_N[n]
        ax.errorbar(xs, ys, yerr=es, fmt="o-", color=color,
                     label=f"torchgw-precomputed N={n}",
                     capsize=3, linewidth=1.6, markersize=6)
        # POT baseline as horizontal dashed line
        b = baselines[n]
        ax.axhline(b["foscttm_mean"], color=color, linestyle="--",
                    linewidth=1.1, alpha=0.55)
        ax.text(xs[-1] * 1.05, b["foscttm_mean"],
                 f"POT N={n}: {b['foscttm_mean']:.3f}",
                 color=color, fontsize=8, va="center")
    ax.axhline(0.12, color="green", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.text(xs[0] * 0.9, 0.125, "SCOT+ literature 0.12",
             color="green", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("M_samples (torchgw per-iter cost rows)")
    ax.set_ylabel("FOSCTTM (lower = better)")
    ax.set_title("Quality vs M_samples (3 seeds, cisTopic, ε=5e-3)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0.10, 0.55)

    # Right: wall_s vs M
    ax = axes[1]
    for n in sorted(by_n):
        xs = [r["M"] for r in by_n[n]]
        ys = [r["wall_s_mean"] for r in by_n[n]]
        color = COLOR_BY_N[n]
        ax.plot(xs, ys, "o-", color=color,
                 label=f"torchgw-precomputed N={n}",
                 linewidth=1.6, markersize=6)
        b = baselines[n]
        ax.axhline(b["wall_s_mean"], color=color, linestyle="--",
                    linewidth=1.1, alpha=0.55)
        ax.text(xs[-1] * 1.05, b["wall_s_mean"],
                 f"POT N={n}: {b['wall_s_mean']:.1f}s",
                 color=color, fontsize=8, va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("M_samples")
    ax.set_ylabel("wall_s (log)")
    ax.set_title("Cost vs M_samples")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper left")

    fig.suptitle("C2 torchgw M_samples sweep — sampled-GW's per-iter cost rows knob",
                  fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / "c2_msamples_sweep.png"
    fig.savefig(out, dpi=170)
    print(f"[c2-ms-plot] wrote {out}")


if __name__ == "__main__":
    main()
