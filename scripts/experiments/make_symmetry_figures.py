#!/usr/bin/env python
"""Generate figures for the GW symmetry-breaking experiment report.

Figures:
  1. datasets.png       — clean 2x2 dataset showcase (C1/C2 and C3 tracks).
  2. solver_effects.png — 3x2 solver-comparison grid: rows = (dataset, scale),
                          cols = pure GW | FGW.
  3. spearman_bar.png   — signed vs abs Spearman across key conditions.
  4. c3_detail.png      — track-specific 4-panel zoom for C3.

All panels share a consistent visual language (two colour maps, one per
figure family; matching axis limits; same dpi).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[2]
FIG_DIR = REPO / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Import track modules (reload to pick up any concurrent edits)
import importlib

sys.path.insert(0, str(REPO / "tracks" / "core" / "01_foundation"))
import run as _c1_mod  # type: ignore[import-not-found]
c1 = importlib.reload(_c1_mod)
sys.path.pop(0)

sys.path.insert(0, str(REPO / "tracks" / "core" / "02_foundation_fused"))
import run as _c2_mod  # type: ignore[import-not-found]
c2 = importlib.reload(_c2_mod)
sys.path.pop(0)

sys.path.insert(0, str(REPO / "tracks" / "core" / "03_branched"))
import run as _c3_mod  # type: ignore[import-not-found]
c3 = importlib.reload(_c3_mod)
sys.path.pop(0)


# ---- Visual style --------------------------------------------------------

DATA_CMAP = "viridis"  # dataset panels — depicts the natural per-point scalar
MATCH_CMAP = "plasma"  # solver-effect panels — depicts the matched scalar
XY_LIM = 1.8           # 2D axis limit; covers both simple spiral and Y-fork
TITLE_SIZE = 12
FIG_DPI = 130
# 3D view: a 3/4 perspective that shows the spiral pattern AND the
# vertical extrusion. Swiss roll spans ~3 units in xy but only 1 in z,
# so the default cube aspect compresses the spiral; we stretch the z
# axis with set_box_aspect (see _apply_3d_view) and keep elev modest.
VIEW_3D = dict(elev=30, azim=-60)
BOX_ASPECT_3D = (1.6, 1.6, 1.0)


def _apply_3d_view(ax) -> None:
    """Set the standard 3D camera and box aspect for swiss-roll panels."""
    ax.view_init(elev=VIEW_3D["elev"], azim=VIEW_3D["azim"])
    try:
        ax.set_box_aspect(BOX_ASPECT_3D)
    except (AttributeError, NotImplementedError):
        pass  # older matplotlib; fall back to default cube


def _add_correspondence_lines(
    fig, ax2d, ax3d,
    src_xy: np.ndarray, tgt_xyz_display: np.ndarray,
    src_s: np.ndarray, tgt_s: np.ndarray,
    n_lines: int = 12, cmap_name: str = DATA_CMAP,
    alpha: float = 0.6, linewidth: float = 1.2,
) -> None:
    """Draw `n_lines` across-panel lines connecting source points and their
    parameter-matched target points. Coloured by the shared parameter.

    tgt_xyz_display must already be in the (x, y, z) order used by the 3D
    scatter call (i.e. for a swiss roll plotted as scatter(Y[:,0], Y[:,2],
    Y[:,1]), pass np.stack((Y[:,0], Y[:,2], Y[:,1]), axis=1)).
    """
    from matplotlib.patches import ConnectionPatch
    from mpl_toolkits.mplot3d import proj3d
    import matplotlib.colors as mcolors

    # Force the 3D projection matrix to be current so proj_transform is valid.
    fig.canvas.draw()

    s_min = float(min(src_s.min(), tgt_s.min()))
    s_max = float(max(src_s.max(), tgt_s.max()))
    # Anchor values evenly spanning the parameter range, avoiding the very ends
    anchors = np.linspace(s_min, s_max, n_lines + 2)[1:-1]
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=s_min, vmax=s_max)

    for s in anchors:
        i = int(np.argmin(np.abs(src_s - s)))
        j = int(np.argmin(np.abs(tgt_s - s)))
        x2, y2 = float(src_xy[i, 0]), float(src_xy[i, 1])
        xyz = tgt_xyz_display[j]
        x3p, y3p, _ = proj3d.proj_transform(
            float(xyz[0]), float(xyz[1]), float(xyz[2]), ax3d.get_proj(),
        )
        con = ConnectionPatch(
            xyA=(x2, y2), coordsA=ax2d.transData,
            xyB=(x3p, y3p), coordsB=ax3d.transData,
            color=cmap(norm(s)), alpha=alpha, linewidth=linewidth,
            zorder=5,
        )
        con.set_clip_on(False)  # allow line to cross axes boundaries
        fig.add_artist(con)


def _get_rho(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    val = getattr(r, "statistic", None)
    if val is None:
        val = r.correlation  # type: ignore[attr-defined]
    return float(np.asarray(val, dtype=float).item())


# ---- Figure 1: dataset showcase -----------------------------------------

def make_datasets_figure() -> Path:
    """Clean 2x2 showcase:
        row 0: C1/C2 — simple spiral  | simple Swiss roll   (θ-coloured)
        row 1: C3    — Y-fork spiral  | Y-fork Swiss roll   (arclen-coloured)
    Tail-2 points in C3 carry a red outline; all other points are filled.
    """
    # C1 / C2 dataset
    X1, a1 = c1.sample_spiral(n=400, seed=0)
    Y1, b1 = c1.sample_swiss_roll(n=500, seed=1)
    # C3 dataset (arclens, with label 1 = tail 2)
    X3, a3, L3 = c3.sample_branched_spiral(n=400, seed=0)
    Y3, b3, L3t = c3.sample_branched_swiss_roll(n=500, seed=1)
    arclen_max = float(max(a3.max(), b3.max()))

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, wspace=0.18, hspace=0.22)

    # (0, 0) C1 / C2 source — 2D spiral. No colorbar here; we share one
    # colorbar per row (placed on the 3D panel's right) so the space between
    # the 2D and 3D panels stays clear for correspondence lines.
    ax_c12_src = fig.add_subplot(gs[0, 0])
    ax_c12_src.scatter(X1[:, 0], X1[:, 1], c=a1, cmap=DATA_CMAP, s=14,
                        vmin=0, vmax=9)
    ax_c12_src.set_title("C1 / C2 source — simple spiral (2D)",
                          fontsize=TITLE_SIZE)
    ax_c12_src.set_xlabel("x"); ax_c12_src.set_ylabel("y")
    ax_c12_src.set_xlim(-XY_LIM, XY_LIM); ax_c12_src.set_ylim(-XY_LIM, XY_LIM)
    ax_c12_src.set_aspect("equal")

    # (0, 1) C1 / C2 target — 3D Swiss roll
    ax_c12_tgt = fig.add_subplot(gs[0, 1], projection="3d")
    Y1_display = np.stack((Y1[:, 0], Y1[:, 2], Y1[:, 1]), axis=1)
    sc = ax_c12_tgt.scatter(Y1_display[:, 0], Y1_display[:, 1],
                              Y1_display[:, 2], c=b1, cmap=DATA_CMAP, s=12,
                              vmin=0, vmax=9)
    ax_c12_tgt.set_title("C1 / C2 target — simple Swiss roll (3D)",
                          fontsize=TITLE_SIZE)
    ax_c12_tgt.set_xlim(-XY_LIM, XY_LIM); ax_c12_tgt.set_ylim(-XY_LIM, XY_LIM)
    ax_c12_tgt.set_zlim(0, 1.0)  # z is uniform in [0, 1]
    _apply_3d_view(ax_c12_tgt)
    plt.colorbar(sc, ax=ax_c12_tgt, shrink=0.7, label="θ (rad)", pad=0.08)

    # Backbone-vs-tail masks (label 0 = main + long tail, label 1 = short tail).
    # The split between main spiral and long tail happens at the spiral's own
    # arc length at θ=9 (≈5.85 units).
    fork_s = float(c3.spiral_arclen(9.0).item())
    src_main = (L3 == 0) & (a3 <= fork_s + 1e-6)
    src_long = (L3 == 0) & (a3 > fork_s + 1e-6)
    src_short = (L3 == 1)
    tgt_main = (L3t == 0) & (b3 <= fork_s + 1e-6)
    tgt_long = (L3t == 0) & (b3 > fork_s + 1e-6)
    tgt_short = (L3t == 1)

    # (1, 0) C3 source — 2D Y-fork spiral. No colorbar (shared with 3D panel).
    ax_c3_src = fig.add_subplot(gs[1, 0])
    ax_c3_src.scatter(X3[src_main, 0], X3[src_main, 1], c=a3[src_main],
                       cmap=DATA_CMAP, s=14, vmin=0, vmax=arclen_max,
                       marker="o", label="main spiral")
    ax_c3_src.scatter(X3[src_long, 0], X3[src_long, 1], c=a3[src_long],
                       cmap=DATA_CMAP, s=28, vmin=0, vmax=arclen_max,
                       marker="^", label="long tail")
    ax_c3_src.scatter(X3[src_short, 0], X3[src_short, 1], c=a3[src_short],
                       cmap=DATA_CMAP, s=28, vmin=0, vmax=arclen_max,
                       marker="s", label="short tail")
    ax_c3_src.set_title("C3 source — spiral + Y-fork (2D)",
                         fontsize=TITLE_SIZE)
    ax_c3_src.set_xlabel("x"); ax_c3_src.set_ylabel("y")
    ax_c3_src.set_xlim(-XY_LIM, XY_LIM); ax_c3_src.set_ylim(-XY_LIM, XY_LIM)
    ax_c3_src.set_aspect("equal")
    ax_c3_src.legend(loc="lower left", fontsize=9)

    # (1, 1) C3 target — 3D Y-fork Swiss roll
    ax_c3_tgt = fig.add_subplot(gs[1, 1], projection="3d")
    Y3_display = np.stack((Y3[:, 0], Y3[:, 2], Y3[:, 1]), axis=1)
    sc = ax_c3_tgt.scatter(Y3_display[tgt_main, 0], Y3_display[tgt_main, 1],
                             Y3_display[tgt_main, 2], c=b3[tgt_main],
                             cmap=DATA_CMAP, s=12, vmin=0, vmax=arclen_max,
                             marker="o", label="main spiral")
    ax_c3_tgt.scatter(Y3_display[tgt_long, 0], Y3_display[tgt_long, 1],
                       Y3_display[tgt_long, 2], c=b3[tgt_long],
                       cmap=DATA_CMAP, s=22, vmin=0, vmax=arclen_max,
                       marker="^", label="long tail")
    ax_c3_tgt.scatter(Y3_display[tgt_short, 0], Y3_display[tgt_short, 1],
                       Y3_display[tgt_short, 2], c=b3[tgt_short],
                       cmap=DATA_CMAP, s=22, vmin=0, vmax=arclen_max,
                       marker="s", label="short tail")
    ax_c3_tgt.set_title("C3 target — Swiss roll + Y-fork (3D)",
                         fontsize=TITLE_SIZE)
    ax_c3_tgt.set_xlim(-XY_LIM, XY_LIM); ax_c3_tgt.set_ylim(-XY_LIM, XY_LIM)
    ax_c3_tgt.set_zlim(0, 1.0)
    _apply_3d_view(ax_c3_tgt)
    ax_c3_tgt.legend(loc="upper right", fontsize=8)
    plt.colorbar(sc, ax=ax_c3_tgt, shrink=0.7, label="geodesic arclen",
                  pad=0.08)

    # Ground-truth correspondence lines (same parameter → matching target).
    _add_correspondence_lines(fig, ax_c12_src, ax_c12_tgt,
                                X1, Y1_display, a1, b1, n_lines=12)
    _add_correspondence_lines(fig, ax_c3_src, ax_c3_tgt,
                                X3, Y3_display, a3, b3, n_lines=12)

    fig.suptitle("Datasets — each point coloured by its natural 1D parameter "
                  "(θ for the simple spiral, geodesic arclen for the Y-fork)",
                  fontsize=14, y=0.995)
    out = FIG_DIR / "datasets.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---- Figure 2: solver effects on each dataset ---------------------------

def _scatter_match(ax, X, a_src, b_tgt, T, title: str, vmax: float) -> None:
    matched = b_tgt[np.argmax(T, axis=1)]
    # Spearman on up to 2000 points to keep it fast at N=10k
    if len(a_src) > 2000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(a_src), 2000, replace=False)
        rho = _get_rho(a_src[idx], matched[idx])
    else:
        rho = _get_rho(a_src, matched)
    ax.scatter(X[:, 0], X[:, 1], c=matched, cmap=MATCH_CMAP, s=8,
               vmin=0, vmax=vmax)
    ax.set_title(f"{title}\nSpearman = {rho:+.4f}", fontsize=TITLE_SIZE)
    ax.set_xlim(-XY_LIM, XY_LIM); ax.set_ylim(-XY_LIM, XY_LIM)
    ax.set_aspect("equal")


def make_solver_effects_figure() -> Path:
    """3x2 grid: rows = (dataset, scale), cols = pure GW | FGW.
    Source 2D points coloured by matched target θ/arclen. Same cmap in all
    panels; per-row vmax follows the row's natural scalar range.
    """
    # --- Data for all rows -----------------------------------------------
    print("  [fig2] N=400 runs (C1, C2 on simple spiral) ...")
    X_s, a_s = c1.sample_spiral(n=400, seed=0)
    Y_s, b_s = c1.sample_swiss_roll(n=500, seed=1)
    T_pure_400 = c1.run_torchgw_landmark(X_s, Y_s, seed=0)["T"]
    T_fused_400 = c2.run_torchgw_fused(X_s, Y_s, a_s, b_s, seed=0)["T"]

    print("  [fig2] N=10000 runs (C1 pure + C2 fused at scale) ...")
    X_l, a_l = c1.sample_spiral(n=10000, seed=0)
    Y_l, b_l = c1.sample_swiss_roll(n=12000, seed=1)
    T_pure_10k = c1.run_torchgw_landmark(X_l, Y_l, seed=0)["T"]
    T_fused_10k = c2.run_torchgw_fused(X_l, Y_l, a_l, b_l, seed=0)["T"]

    print("  [fig2] C3 N=400 runs (pure vs FGW on Y-fork) ...")
    X3, a3, _ = c3.sample_branched_spiral(n=400, seed=0)
    Y3, b3, _ = c3.sample_branched_swiss_roll(n=500, seed=1)
    T_c3_pure = c3.run_torchgw_landmark(X3, Y3, seed=0)["T"]
    T_c3_fused = c3.run_torchgw_fused(X3, Y3, a3, b3, seed=0)["T"]

    arclen_max = float(max(a3.max(), b3.max()))

    fig, axes = plt.subplots(3, 2, figsize=(12, 17))
    fig.subplots_adjust(wspace=0.15, hspace=0.48, top=0.92, right=0.88)

    # Row 1: C1/C2, N=400, vmax = 9 (θ range)
    _scatter_match(axes[0, 0], X_s, a_s, b_s, T_pure_400,
                    "C1 / C2 spiral, N=400\npure GW  (torchgw-landmark)", 9.0)
    _scatter_match(axes[0, 1], X_s, a_s, b_s, T_fused_400,
                    "C1 / C2 spiral, N=400\nFGW  (torchgw-fused, θ feature)", 9.0)

    # Row 2: C1/C2 at N=10k — the orientation flip case
    _scatter_match(axes[1, 0], X_l, a_l, b_l, T_pure_10k,
                    "C1 / C2 spiral, N=10k\npure GW  (orientation may flip!)", 9.0)
    _scatter_match(axes[1, 1], X_l, a_l, b_l, T_fused_10k,
                    "C1 / C2 spiral, N=10k\nFGW  (θ feature locks orientation)", 9.0)

    # Row 3: C3 Y-fork
    _scatter_match(axes[2, 0], X3, a3, b3, T_c3_pure,
                    "C3 Y-fork, N=400\npure GW  (torchgw-landmark)", arclen_max)
    _scatter_match(axes[2, 1], X3, a3, b3, T_c3_fused,
                    "C3 Y-fork, N=400\nFGW  (arclen feature)", arclen_max)

    # Axis labels only on the outside
    for ax in axes[:, 0]:
        ax.set_ylabel("source y")
    for ax in axes[-1, :]:
        ax.set_xlabel("source x")

    # Shared colourbars per row (same vmax per row). Using ax=axes[row, :]
    # lets matplotlib take the room from the right-hand gutter reserved by
    # right=0.88 in subplots_adjust.
    for row_idx, label in enumerate(("matched θ", "matched θ", "matched arclen")):
        fig.colorbar(
            axes[row_idx, 1].collections[0], ax=axes[row_idx, :],
            orientation="vertical", fraction=0.04, pad=0.02, shrink=0.85,
            label=label,
        )

    fig.suptitle(
        "Solver effects — source points coloured by argmax-matched target scalar.\n"
        "Forward match → colour ramp aligned with source progression.",
        fontsize=14, y=0.975,
    )
    out = FIG_DIR / "solver_effects.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---- Figure 3: Spearman bar comparison ----------------------------------

def make_spearman_bar_figure() -> Path:
    """Compare signed and |rho| across key conditions."""
    results: list[tuple[str, float]] = []

    # C1 N=400
    X, a = c1.sample_spiral(n=400, seed=0)
    Y, b = c1.sample_swiss_roll(n=500, seed=1)
    T = c1.run_torchgw_landmark(X, Y, seed=0)["T"]
    results.append(("C1\nN=400", _get_rho(a, b[np.argmax(T, axis=1)])))

    # C1 N=10k (the flip)
    X, a = c1.sample_spiral(n=10000, seed=0)
    Y, b = c1.sample_swiss_roll(n=12000, seed=1)
    T = c1.run_torchgw_landmark(X, Y, seed=0)["T"]
    matched = b[np.argmax(T, axis=1)]
    results.append(("C1\nN=10k", _get_rho(a, matched)))

    # C2 N=400
    X, a = c2.sample_spiral(n=400, seed=0)
    Y, b = c2.sample_swiss_roll(n=500, seed=1)
    T = c2.run_torchgw_fused(X, Y, a, b, seed=0)["T"]
    results.append(("C2 FGW\nN=400", _get_rho(a, b[np.argmax(T, axis=1)])))

    # C3 N=400 main arc
    X3, a3, L3 = c3.sample_branched_spiral(n=400, seed=0)
    Y3, b3, L3t = c3.sample_branched_swiss_roll(n=500, seed=1)
    T3 = c3.run_torchgw_fused(X3, Y3, a3, b3, seed=0)["T"]
    matched3 = b3[np.argmax(T3, axis=1)]
    results.append(("C3 FGW\nN=400",
                     _get_rho(a3[L3 == 0], matched3[L3 == 0])))

    labels = [r[0] for r in results]
    signed = [r[1] for r in results]
    abs_vals = [abs(x) for x in signed]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, signed, width, label="signed ρ",
           color=["#4575b4" if s >= 0 else "#d73027" for s in signed])
    ax.bar(x + width / 2, abs_vals, width, label="|ρ|",
           color="#fdae61", alpha=0.75)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axhline(0.95, color="green", linestyle=":", linewidth=0.8,
               label="threshold 0.95")
    ax.axhline(-0.95, color="green", linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Orientation stability across tracks — only C1 at N=10k flips")
    ax.set_ylim(-1.1, 1.2)
    ax.legend(loc="lower right")
    for xi, v in zip(x, signed):
        ax.annotate(f"{v:+.3f}", xy=(xi - width / 2,
                                      v + (0.03 if v >= 0 else -0.08)),
                    ha="center", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "spearman_bar.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---- Figure 4: C3 deep-dive ---------------------------------------------

def make_c3_zoom_figure() -> Path:
    """Four panels zooming into C3 with FGW (torchgw-fused, arclen feature)."""
    X, a, Ls = c3.sample_branched_spiral(n=400, seed=0)
    Y, b, Lt = c3.sample_branched_swiss_roll(n=500, seed=1)
    T = c3.run_torchgw_fused(X, Y, a, b, seed=0)["T"]
    matched_scalar = b[np.argmax(T, axis=1)]
    matched_label = Lt[np.argmax(T, axis=1)]

    main_mask = (Ls == 0)
    tail_mask = (Ls == 1)
    rho_main = _get_rho(a[main_mask], matched_scalar[main_mask])
    rho_tail = _get_rho(a[tail_mask], matched_scalar[tail_mask])
    branch_acc = float(np.mean(Ls == matched_label))

    fork_s = float(c3.spiral_arclen(9.0).item())
    main_arc = (Ls == 0) & (a <= fork_s + 1e-6)
    long_t = (Ls == 0) & (a > fork_s + 1e-6)
    short_t = (Ls == 1)
    main_arc_t = (Lt == 0) & (b <= fork_s + 1e-6)
    long_t_tgt = (Lt == 0) & (b > fork_s + 1e-6)
    short_t_tgt = (Lt == 1)

    fig = plt.figure(figsize=(16, 5))

    ax = fig.add_subplot(1, 4, 1)
    ax.scatter(X[main_arc, 0], X[main_arc, 1], c="steelblue", s=12,
               label="main spiral")
    ax.scatter(X[long_t, 0], X[long_t, 1], c="mediumseagreen", s=18,
               marker="^", label="long tail")
    ax.scatter(X[short_t, 0], X[short_t, 1], c="crimson", s=18, marker="s",
               label="short tail")
    ax.set_title("C3 source (2D): asymmetric Y-fork")
    ax.set_xlim(-XY_LIM, XY_LIM); ax.set_ylim(-XY_LIM, XY_LIM)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=9)

    ax = fig.add_subplot(1, 4, 2, projection="3d")
    ax.scatter(Y[main_arc_t, 0], Y[main_arc_t, 2], Y[main_arc_t, 1],
               c="steelblue", s=10, label="main spiral")
    ax.scatter(Y[long_t_tgt, 0], Y[long_t_tgt, 2], Y[long_t_tgt, 1],
               c="mediumseagreen", s=14, marker="^", label="long tail")
    ax.scatter(Y[short_t_tgt, 0], Y[short_t_tgt, 2], Y[short_t_tgt, 1],
               c="crimson", s=14, marker="s", label="short tail")
    ax.set_xlim(-XY_LIM, XY_LIM); ax.set_ylim(-XY_LIM, XY_LIM)
    ax.set_zlim(0, 1.0)
    ax.set_title("C3 target (3D)")
    _apply_3d_view(ax)
    ax.legend(loc="upper right", fontsize=8)

    ax = fig.add_subplot(1, 4, 3)
    vmax = float(b.max())
    sc = ax.scatter(X[main_arc, 0], X[main_arc, 1],
                     c=matched_scalar[main_arc], cmap=MATCH_CMAP,
                     s=14, vmin=0, vmax=vmax, marker="o")
    ax.scatter(X[long_t, 0], X[long_t, 1], c=matched_scalar[long_t],
               cmap=MATCH_CMAP, vmin=0, vmax=vmax, s=22, marker="^")
    ax.scatter(X[short_t, 0], X[short_t, 1], c=matched_scalar[short_t],
               cmap=MATCH_CMAP, vmin=0, vmax=vmax, s=22, marker="s")
    ax.set_title(f"matched target arclen (FGW)\n"
                  f"main-Spearman = {rho_main:+.4f}\n"
                  f"tail-Spearman = {rho_tail:+.4f}")
    ax.set_xlim(-XY_LIM, XY_LIM); ax.set_ylim(-XY_LIM, XY_LIM)
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, shrink=0.8, label="matched geodesic arclen")

    ax = fig.add_subplot(1, 4, 4)
    correct = (Ls == matched_label)
    ax.scatter(X[correct, 0], X[correct, 1], c="#2ca02c", s=12,
               label=f"label matches ({int(correct.sum())})")
    ax.scatter(X[~correct, 0], X[~correct, 1], c="#d62728", s=22, marker="x",
               label=f"label mismatch ({int((~correct).sum())})")
    ax.set_title(f"branch label propagation\nbranch_accuracy = {branch_acc:.4f}")
    ax.set_xlim(-XY_LIM, XY_LIM); ax.set_ylim(-XY_LIM, XY_LIM)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=9)

    fig.suptitle("C3 deep-dive — asymmetric Y-fork with FGW (arclen feature)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "c3_detail.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("[fig] datasets.png ...")
    print(f"  → {make_datasets_figure()}")
    print("[fig] solver_effects.png ...")
    print(f"  → {make_solver_effects_figure()}")
    print("[fig] spearman_bar.png ...")
    print(f"  → {make_spearman_bar_figure()}")
    print("[fig] c3_detail.png ...")
    print(f"  → {make_c3_zoom_figure()}")
    print("\nAll figures written to", FIG_DIR)
