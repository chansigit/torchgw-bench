#!/usr/bin/env python
"""Generate figures for the GW symmetry-breaking experiment report.

Figures produced:
  1. datasets.png — side-by-side visualisation of C1 / C2 (θ-coloured) / C3 datasets
  2. matchings.png — argmax(T) correspondences for C1 (forward), C1-large (reverse),
                     C2-fused (forward), C3-branched (forward)
  3. spearman_bar.png — |rho| and signed rho across tracks/scales

Outputs go into docs/figures/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIG_DIR = REPO / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Add tracks to sys.path so we can import their run.py modules directly.
sys.path.insert(0, str(REPO / "tracks" / "core" / "01_foundation"))
import run as c1  # type: ignore[import-not-found]
sys.path.pop(0)

sys.path.insert(0, str(REPO / "tracks" / "core" / "02_foundation_fused"))
import importlib
import run as _c2_mod  # type: ignore[import-not-found]
c2 = importlib.reload(_c2_mod)
sys.path.pop(0)

sys.path.insert(0, str(REPO / "tracks" / "core" / "03_branched"))
import run as _c3_mod  # type: ignore[import-not-found]
c3 = importlib.reload(_c3_mod)
sys.path.pop(0)


# ---- Figure 1: dataset visualisation ------------------------------------

def make_datasets_figure() -> Path:
    fig = plt.figure(figsize=(14, 8))

    # C1 source (2D spiral)
    X1, a1 = c1.sample_spiral(n=400, seed=0)
    ax = fig.add_subplot(2, 3, 1)
    sc = ax.scatter(X1[:, 0], X1[:, 1], c=a1, cmap="viridis", s=10)
    ax.set_title("C1 source: 2D spiral (θ ∈ [0, 9])")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="θ", shrink=0.8)

    # C1 target (3D swiss roll)
    Y1, b1 = c1.sample_swiss_roll(n=500, seed=1)
    ax = fig.add_subplot(2, 3, 2, projection="3d")
    ax.scatter(Y1[:, 0], Y1[:, 2], Y1[:, 1], c=b1, cmap="viridis", s=10)
    ax.set_title("C1 target: 3D Swiss roll")
    ax.view_init(elev=20, azim=-60)

    # C2: same as C1 but annotated with FGW feature
    ax = fig.add_subplot(2, 3, 3)
    sc = ax.scatter(X1[:, 0], X1[:, 1], c=a1, cmap="viridis", s=10)
    ax.set_title("C2: identical data,\nθ used as FGW feature")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="θ feature", shrink=0.8)

    # C3 source (spiral + tangential tail)
    X3, a3, L3 = c3.sample_branched_spiral(n=400, seed=0)
    ax = fig.add_subplot(2, 3, 4)
    ax.scatter(X3[L3 == 0, 0], X3[L3 == 0, 1], c="steelblue", s=10, label="main")
    ax.scatter(X3[L3 == 1, 0], X3[L3 == 1, 1], c="crimson", s=14, marker="s",
               label="tail")
    ax.set_title("C3 source: spiral + tangential tail\n(at outer end, θ=9)")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8)

    # C3 target (swiss roll + tail)
    Y3, b3, L3t = c3.sample_branched_swiss_roll(n=500, seed=1)
    ax = fig.add_subplot(2, 3, 5, projection="3d")
    ax.scatter(Y3[L3t == 0, 0], Y3[L3t == 0, 2], Y3[L3t == 0, 1],
               c="steelblue", s=10, label="main")
    ax.scatter(Y3[L3t == 1, 0], Y3[L3t == 1, 2], Y3[L3t == 1, 1],
               c="crimson", s=14, marker="s", label="tail")
    ax.set_title("C3 target: Swiss roll + tail")
    ax.view_init(elev=20, azim=-60)
    ax.legend(loc="upper right", fontsize=8)

    # Empty panel annotating the logic
    ax = fig.add_subplot(2, 3, 6)
    ax.axis("off")
    ax.text(0.05, 0.5,
            "C1: symmetric geometry\n"
            "    → pure GW has two optima\n"
            "    (forward / reverse)\n\n"
            "C2: same geometry + θ feature\n"
            "    → FGW breaks tie via feature W\n\n"
            "C3: asymmetric geometry (single tail)\n"
            "    → pure GW forward-only",
            fontsize=11, family="monospace", verticalalignment="center")

    fig.suptitle("Three tracks, three approaches to the GW orientation problem",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    out = FIG_DIR / "datasets.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ---- Figure 2: matchings (argmax correspondences) -----------------------

def _run_c1(n_src: int, n_tgt: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, a = c1.sample_spiral(n=n_src, seed=seed)
    Y, b = c1.sample_swiss_roll(n=n_tgt, seed=seed + 1)
    out = c1.run_torchgw_landmark(X, Y, seed=seed)
    return X, a, Y, b, out["T"]


def _run_c2(n_src: int, n_tgt: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, a = c2.sample_spiral(n=n_src, seed=seed)
    Y, b = c2.sample_swiss_roll(n=n_tgt, seed=seed + 1)
    out = c2.run_torchgw_fused(X, Y, a, b, seed=seed)
    return X, a, Y, b, out["T"]


def _run_c3(n_src: int, n_tgt: int, seed: int) -> tuple:
    X, a, Ls = c3.sample_branched_spiral(n=n_src, seed=seed)
    Y, b, Lt = c3.sample_branched_swiss_roll(n=n_tgt, seed=seed + 1)
    out = c3.run_torchgw_landmark(X, Y, seed=seed)
    return X, a, Ls, Y, b, Lt, out["T"]


def _scatter_match(ax, src_pts_2d, src_angles, tgt_angles, T, title, sample=60):
    """Draw source as 2D scatter coloured by θ; overlay lines to matched tgt θ.
    Because target is 3D, we'll just draw source coloured by MATCHED θ for the
    alignment indicator."""
    matched_theta = tgt_angles[np.argmax(T, axis=1)]
    sc = ax.scatter(src_pts_2d[:, 0], src_pts_2d[:, 1],
                     c=matched_theta, cmap="plasma", s=12, vmin=0, vmax=9)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    return sc


def make_matchings_figure() -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # C1 small: usually forward
    print("  [fig2] running C1 N=400 K=500...")
    X, a, Y, b, T = _run_c1(400, 500, seed=0)
    matched = b[np.argmax(T, axis=1)]
    from scipy.stats import spearmanr
    rho = spearmanr(a, matched).statistic
    sc0 = _scatter_match(axes[0], X, a, b, T,
                          f"C1 (N=400): pure GW\nSpearman = {rho:+.3f}")

    # C1 large: sometimes reverse (force by running seed=0 at 10k×12k)
    print("  [fig2] running C1 N=10000 K=12000... (~30s on GPU)")
    X, a, Y, b, T = _run_c1(10000, 12000, seed=0)
    matched = b[np.argmax(T, axis=1)]
    rho = spearmanr(a, matched).statistic
    sc1 = _scatter_match(axes[1], X, a, b, T,
                          f"C1 (N=10k): pure GW\nSpearman = {rho:+.3f}")

    # C2: FGW (N=400)
    print("  [fig2] running C2 FGW N=400 K=500...")
    X, a, Y, b, T = _run_c2(400, 500, seed=0)
    matched = b[np.argmax(T, axis=1)]
    rho = spearmanr(a, matched).statistic
    sc2 = _scatter_match(axes[2], X, a, b, T,
                          f"C2 FGW (N=400): θ feature\nSpearman = {rho:+.3f}")

    # C3: branched (N=400)
    print("  [fig2] running C3 branched N=400 K=500...")
    X, a, Ls, Y, b, Lt, T = _run_c3(400, 500, seed=0)
    matched_theta = b[np.argmax(T, axis=1)]
    from scipy.stats import spearmanr
    main_mask = (Ls == 0)
    rho_main = spearmanr(a[main_mask], matched_theta[main_mask]).statistic
    sc3 = axes[3].scatter(X[:, 0], X[:, 1], c=matched_theta,
                          cmap="plasma", s=12, vmin=0, vmax=9.8)
    # Mark tail points
    axes[3].scatter(X[Ls == 1, 0], X[Ls == 1, 1], facecolors="none",
                    edgecolors="crimson", s=45, linewidths=1.2,
                    marker="s", label="tail points")
    axes[3].set_title(f"C3 tailed (N=400): pure GW\nmain-Spearman = {rho_main:+.3f}",
                      fontsize=11)
    axes[3].set_aspect("equal")
    axes[3].legend(loc="lower left", fontsize=8)

    for ax in axes:
        ax.set_xlabel("source x")
    axes[0].set_ylabel("source y")

    cbar = fig.colorbar(sc0, ax=axes, orientation="horizontal",
                         shrink=0.6, pad=0.15, aspect=40)
    cbar.set_label("matched target θ (via argmax T)")

    fig.suptitle("Source points coloured by matched target θ — "
                 "forward matching = colour ramp aligned with spiral progress",
                 fontsize=12, y=1.03)
    out = FIG_DIR / "matchings.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ---- Figure 3: spearman comparison --------------------------------------

def make_spearman_bar_figure() -> Path:
    """Compare |rho| and signed rho across the three tracks + C1 at 10k."""
    import json
    import glob

    # Run fresh to fill in values (cheap except C1-10k already cached above)
    results = []

    # C1 N=400
    X, a = c1.sample_spiral(n=400, seed=0)
    Y, b = c1.sample_swiss_roll(n=500, seed=1)
    T = c1.run_torchgw_landmark(X, Y, seed=0)["T"]
    matched = b[np.argmax(T, axis=1)]
    from scipy.stats import spearmanr
    signed_rho = float(spearmanr(a, matched).statistic)
    results.append(("C1\nN=400", signed_rho))

    # C1 N=10k (same as fig 2 but re-run — cheap-ish)
    X, a = c1.sample_spiral(n=10000, seed=0)
    Y, b = c1.sample_swiss_roll(n=12000, seed=1)
    T = c1.run_torchgw_landmark(X, Y, seed=0)["T"]
    matched = b[np.argmax(T, axis=1)]
    signed_rho = float(spearmanr(a, matched).statistic)
    results.append(("C1\nN=10k", signed_rho))

    # C2 N=400
    X, a = c2.sample_spiral(n=400, seed=0)
    Y, b = c2.sample_swiss_roll(n=500, seed=1)
    T = c2.run_torchgw_fused(X, Y, a, b, seed=0)["T"]
    matched = b[np.argmax(T, axis=1)]
    signed_rho = float(spearmanr(a, matched).statistic)
    results.append(("C2 FGW\nN=400", signed_rho))

    # C3 N=400 (main-branch only)
    X, a, Ls = c3.sample_branched_spiral(n=400, seed=0)
    Y, b, Lt = c3.sample_branched_swiss_roll(n=500, seed=1)
    T = c3.run_torchgw_landmark(X, Y, seed=0)["T"]
    matched = b[np.argmax(T, axis=1)]
    main_mask = (Ls == 0)
    signed_rho = float(spearmanr(a[main_mask], matched[main_mask]).statistic)
    results.append(("C3 tailed\nN=400", signed_rho))

    labels = [r[0] for r in results]
    signed = [r[1] for r in results]
    abs_vals = [abs(x) for x in signed]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, signed, width, label="signed ρ",
                    color=["#4575b4" if s >= 0 else "#d73027" for s in signed])
    bars2 = ax.bar(x + width / 2, abs_vals, width, label="|ρ|",
                    color="#fdae61", alpha=0.75)

    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axhline(0.95, color="green", linestyle=":", linewidth=0.8,
               label="threshold 0.95")
    ax.axhline(-0.95, color="green", linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spearman ρ on θ")
    ax.set_title("Orientation symmetry: signed ρ flips at C1 N=10k; C2/C3 are stable")
    ax.set_ylim(-1.1, 1.2)
    ax.legend(loc="lower right")

    # Annotate values
    for b_, v in zip(bars1, signed):
        ax.annotate(f"{v:+.3f}", xy=(b_.get_x() + b_.get_width() / 2,
                                       v + (0.03 if v >= 0 else -0.08)),
                    ha="center", fontsize=9)

    fig.tight_layout()
    out = FIG_DIR / "spearman_bar.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ---- Figure 4: C3 zoom -- dedicated branched-dataset visualisation ------

def make_c3_zoom_figure() -> Path:
    """Four panels zooming into C3: source + target side-by-side with labels,
    plus matched-θ colouring and branch-accuracy overlay."""
    from scipy.stats import spearmanr

    X, a, Ls = c3.sample_branched_spiral(n=400, seed=0)
    Y, b, Lt = c3.sample_branched_swiss_roll(n=500, seed=1)
    T = c3.run_torchgw_landmark(X, Y, seed=0)["T"]
    matched_theta = b[np.argmax(T, axis=1)]
    matched_label = Lt[np.argmax(T, axis=1)]

    main_mask = (Ls == 0)
    rho_main = float(spearmanr(a[main_mask], matched_theta[main_mask]).statistic)
    branch_acc = float(np.mean(Ls == matched_label))

    fig = plt.figure(figsize=(16, 5))

    # Panel 1: C3 source labelled
    ax = fig.add_subplot(1, 4, 1)
    ax.scatter(X[Ls == 0, 0], X[Ls == 0, 1], c="steelblue", s=12, label="main")
    ax.scatter(X[Ls == 1, 0], X[Ls == 1, 1], c="crimson", s=18, marker="s",
               label="tail")
    ax.set_title("C3 source (2D)\nspiral + tangential tail at θ=9")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=9)

    # Panel 2: C3 target labelled (3D)
    ax = fig.add_subplot(1, 4, 2, projection="3d")
    ax.scatter(Y[Lt == 0, 0], Y[Lt == 0, 2], Y[Lt == 0, 1],
               c="steelblue", s=10, label="main")
    ax.scatter(Y[Lt == 1, 0], Y[Lt == 1, 2], Y[Lt == 1, 1],
               c="crimson", s=16, marker="s", label="tail")
    ax.set_title("C3 target (3D)\nSwiss roll + tail")
    ax.view_init(elev=20, azim=-60)
    ax.legend(loc="upper right", fontsize=9)

    # Panel 3: source coloured by matched target θ
    ax = fig.add_subplot(1, 4, 3)
    sc = ax.scatter(X[:, 0], X[:, 1], c=matched_theta, cmap="plasma",
                     s=14, vmin=0, vmax=9.8)
    ax.scatter(X[Ls == 1, 0], X[Ls == 1, 1], facecolors="none",
               edgecolors="crimson", s=50, linewidths=1.4, marker="s")
    ax.set_title(f"matched target θ\nmain-Spearman = {rho_main:+.4f}")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="matched target θ", shrink=0.8)

    # Panel 4: label-match (green) vs label-mismatch (red)
    ax = fig.add_subplot(1, 4, 4)
    correct = (Ls == matched_label)
    ax.scatter(X[correct, 0], X[correct, 1], c="#2ca02c", s=12,
               label=f"label matches ({correct.sum()})")
    ax.scatter(X[~correct, 0], X[~correct, 1], c="#d62728", s=22, marker="x",
               label=f"label mismatch ({(~correct).sum()})")
    ax.set_title(f"branch label propagation\nbranch_accuracy = {branch_acc:.4f}")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=9)

    fig.suptitle("C3 — single tangential tail at θ=9 + pure GW: deterministic forward matching",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "c3_detail.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("[fig] Generating datasets.png ...")
    p1 = make_datasets_figure()
    print(f"  → {p1}")
    print("[fig] Generating matchings.png ...")
    p2 = make_matchings_figure()
    print(f"  → {p2}")
    print("[fig] Generating spearman_bar.png ...")
    p3 = make_spearman_bar_figure()
    print(f"  → {p3}")
    print("[fig] Generating c3_detail.png ...")
    p4 = make_c3_zoom_figure()
    print(f"  → {p4}")
    print("\nAll figures written to", FIG_DIR)
