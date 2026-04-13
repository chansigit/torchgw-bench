#!/usr/bin/env python
"""Generate figures for the C3 Y-fork GW-alignment experiment.

Focus: one dataset, the asymmetric Y-fork. Source is the 3D Swiss roll
(with Y-fork); target is the 2D spiral (with Y-fork). The Y-fork alone is
enough to break GW's forward/reverse orientation ambiguity, so we no
longer feature the simple-spiral / FGW-on-θ track from earlier drafts.

Figures produced:
  1. datasets.png       — 3D source + 2D target with bundled correspondence lines.
  2. solver_effects.png — side-by-side pure GW vs FGW (arclen feature) on C3.
  3. spearman_bar.png   — backbone and branch Spearman under the two solvers.
  4. c3_detail.png      — 4-panel deep dive on C3 FGW.
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

import importlib

sys.path.insert(0, str(REPO / "tracks" / "core" / "03_branched"))
import run as _c3_mod  # type: ignore[import-not-found]
c3 = importlib.reload(_c3_mod)
sys.path.pop(0)


# ---- Visual style --------------------------------------------------------

DATA_CMAP = "viridis"
MATCH_CMAP = "plasma"
XY_LIM = 1.8
TITLE_SIZE = 12
FIG_DPI = 130
VIEW_3D = dict(elev=30, azim=-60)
BOX_ASPECT_3D = (1.5, 1.5, 1.4)  # taller z so the swiss roll has presence
SURFACE_ALPHA = 0.35  # opaque enough to read the swirl through the scatter
TAIL_STRIP_ALPHA = 0.30


def _apply_3d_view(ax) -> None:
    ax.view_init(elev=VIEW_3D["elev"], azim=VIEW_3D["azim"])
    try:
        ax.set_box_aspect(BOX_ASPECT_3D)
    except (AttributeError, NotImplementedError):
        pass


def _overlay_swiss_roll_surface(
    ax3d,
    r_min: float = 0.3, r_max: float = 1.0, theta_max: float = 9.0,
    z_max: float = 1.0, alpha: float = SURFACE_ALPHA,
    cmap_name: str = DATA_CMAP,
    vmax: "float | None" = None,
) -> None:
    """Draw a shaded clean Swiss-roll surface underneath the scatter cloud.
    Axis order matches scatter(Y[:,0], Y[:,2], Y[:,1]) = (spiral_x, spiral_y, height).
    """
    from matplotlib.colors import LightSource, Normalize

    theta = np.linspace(0, theta_max, 120)
    z = np.linspace(0, z_max, 14)
    T, Z = np.meshgrid(theta, z)
    R = r_min + (r_max - r_min) * T / theta_max
    Xs = R * np.cos(T)
    Ys = R * np.sin(T)

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=0, vmax=(vmax if vmax is not None else theta_max))
    face_rgb = cmap(norm(T))[..., :3]
    ls = LightSource(azdeg=315, altdeg=35)
    shaded = ls.shade_rgb(face_rgb, Z, blend_mode="soft", vert_exag=0.5)

    ax3d.plot_surface(
        Xs, Ys, Z,
        facecolors=shaded, alpha=alpha, antialiased=True,
        shade=False, rstride=1, cstride=1,
        linewidth=0, edgecolor="none",
    )


def _overlay_tail_strip(
    ax3d,
    base_xy: "tuple[float, float]", direction_xy: "tuple[float, float]",
    length: float, z_max: float = 1.0,
    color=(0.85, 0.9, 0.35), alpha: float = TAIL_STRIP_ALPHA,
) -> None:
    """Thin extruded strip for a straight tail, same axis order as scatter."""
    bx, by = base_xy
    dx, dy = direction_xy
    s = np.linspace(0, length, 10)
    z = np.linspace(0, z_max, 6)
    S, Z = np.meshgrid(s, z)
    Xs = bx + S * dx
    Ys = by + S * dy
    facecolor = np.broadcast_to(np.asarray(color, dtype=float), (*Xs.shape, 3))
    ax3d.plot_surface(
        Xs, Ys, Z, facecolors=facecolor, alpha=alpha, antialiased=True,
        shade=False, rstride=1, cstride=1, linewidth=0, edgecolor="none",
    )


def _get_rho(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    val = getattr(r, "statistic", None)
    if val is None:
        val = r.correlation  # type: ignore[attr-defined]
    return float(np.asarray(val, dtype=float).item())


# ---- Bundle helpers ------------------------------------------------------

def _add_bundled_lines(
    fig, ax_src, ax_tgt,
    src_coords_display: np.ndarray, tgt_xy: np.ndarray,
    bundles: "list[tuple[np.ndarray, np.ndarray, object]]",
    alpha: float = 0.55, linewidth: float = 1.1,
) -> None:
    """Draw bundles of lines from 3D source points to 2D target points.

    src_coords_display: (N, 3) array in the order passed to ax_src.scatter
                         (for a swiss roll plotted as
                          scatter(Y[:,0], Y[:,2], Y[:,1]), pass that stack).
    tgt_xy:             (K, 2) array of 2D target points.
    bundles: list of (src_indices, tgt_indices, colour). lines connect
             them pairwise.
    """
    from matplotlib.patches import ConnectionPatch
    from mpl_toolkits.mplot3d import proj3d

    fig.canvas.draw()
    for src_idx, tgt_idx, color in bundles:
        for i, j in zip(src_idx, tgt_idx):
            xyz = src_coords_display[int(i)]
            x3p, y3p, _ = proj3d.proj_transform(
                float(xyz[0]), float(xyz[1]), float(xyz[2]), ax_src.get_proj(),
            )
            x2, y2 = float(tgt_xy[int(j), 0]), float(tgt_xy[int(j), 1])
            con = ConnectionPatch(
                xyA=(x3p, y3p), coordsA=ax_src.transData,
                xyB=(x2, y2), coordsB=ax_tgt.transData,
                color=color, alpha=alpha, linewidth=linewidth,
                zorder=5,
            )
            con.set_clip_on(False)
            fig.add_artist(con)


def _draw_schematic(ax) -> None:
    """Schematic of the 3D-source → 2D-target Y-fork alignment task.

    The 3D source is drawn as a stacked "ribbon" (bottom + top spirals
    connected by vertical sticks) so it visually reads as 3D. The 2D
    target is drawn as a single flat curve. The three regions (main
    spiral / long tail / short tail) are colour-coded the same way the
    cluster legend is, so the same colour means the same region in every
    place it appears.
    """
    import matplotlib.patches as mpatches

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # --- Title bar -------------------------------------------------------
    ax.text(0.5, 0.97, "Experiment design", ha="center", va="top",
            fontsize=14, fontweight="bold")
    ax.text(0.5, 0.90, "Asymmetric Y-fork alignment   (3D → 2D)",
            ha="center", va="top", fontsize=10, style="italic", color="#444")

    # Region palette — same hues as the cluster legend below
    COLOR_MAIN = "#3b528b"   # viridis-low (spiral backbone)
    COLOR_LONG = "#5dc863"   # viridis-mid (long tail)
    COLOR_SHORT = "#fde725"  # viridis-high (short tail)

    # --- Geometry of the mini Y-fork (shared between 3D and 2D sketch) --
    theta_main = np.linspace(0.5, 5.4, 80)
    r_main = 0.018 + 0.022 * theta_main / 5.4
    # Main spiral, end point and tangent
    end_phi = theta_main[-1]
    end_offset = np.array([r_main[-1] * np.cos(end_phi),
                            r_main[-1] * np.sin(end_phi)])
    tang_dir = np.array([np.cos(end_phi) * (0.022 / 5.4)
                          - r_main[-1] * np.sin(end_phi),
                          np.sin(end_phi) * (0.022 / 5.4)
                          + r_main[-1] * np.cos(end_phi)])
    tang_dir = tang_dir / (np.linalg.norm(tang_dir) + 1e-9)
    rad_dir = np.array([np.cos(end_phi), np.sin(end_phi)])
    cross = tang_dir[0] * rad_dir[1] - tang_dir[1] * rad_dir[0]
    sign = -1.0 if cross > 0 else 1.0
    fork_angle = sign * np.pi / 6
    short_dir = np.array([
        tang_dir[0] * np.cos(fork_angle) - tang_dir[1] * np.sin(fork_angle),
        tang_dir[0] * np.sin(fork_angle) + tang_dir[1] * np.cos(fork_angle),
    ])
    s_long = np.linspace(0, 0.045, 12)
    s_short = np.linspace(0, 0.022, 8)

    def _layout_curves(cx: float, cy: float):
        sx_m = cx + r_main * np.cos(theta_main)
        sy_m = cy + r_main * np.sin(theta_main)
        end_x = cx + end_offset[0]
        end_y = cy + end_offset[1]
        sx_l = end_x + s_long * tang_dir[0]
        sy_l = end_y + s_long * tang_dir[1]
        sx_s = end_x + s_short * short_dir[0]
        sy_s = end_y + s_short * short_dir[1]
        return sx_m, sy_m, sx_l, sy_l, sx_s, sy_s

    # --- 3D source box (left) -------------------------------------------
    src = mpatches.FancyBboxPatch(
        (0.04, 0.42), 0.36, 0.36,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        edgecolor="#1f4e79", facecolor="#e7f0fa", linewidth=1.5,
    )
    ax.add_patch(src)
    ax.text(0.22, 0.74, "3D source", ha="center", va="top",
            fontsize=11, fontweight="bold", color="#1f4e79")
    ax.text(0.22, 0.69, "Y-fork Swiss roll", ha="center", va="top",
            fontsize=9, color="#1f4e79")

    cx, cy = 0.18, 0.55
    sx_m, sy_m, sx_l, sy_l, sx_s, sy_s = _layout_curves(cx, cy)
    # 3D illusion: skew the "top layer" up-right and connect with sticks
    dx, dy = 0.005, 0.038
    sx_m_t, sy_m_t = sx_m + dx, sy_m + dy
    sx_l_t, sy_l_t = sx_l + dx, sy_l + dy
    sx_s_t, sy_s_t = sx_s + dx, sy_s + dy
    # Bottom curves (solid)
    ax.plot(sx_m, sy_m, color=COLOR_MAIN, lw=1.2, zorder=2)
    ax.plot(sx_l, sy_l, color=COLOR_LONG, lw=1.6, zorder=2)
    ax.plot(sx_s, sy_s, color=COLOR_SHORT, lw=1.6, zorder=2)
    # Vertical sticks (semi-transparent) — sampled points
    for arr_b, arr_t, arr_b_y, arr_t_y, color, idxs in (
        (sx_m, sx_m_t, sy_m, sy_m_t, COLOR_MAIN,
         np.linspace(0, len(sx_m) - 1, 9).astype(int)),
        (sx_l, sx_l_t, sy_l, sy_l_t, COLOR_LONG, [0, 5, 11]),
        (sx_s, sx_s_t, sy_s, sy_s_t, COLOR_SHORT, [0, 4, 7]),
    ):
        for i in idxs:
            ax.plot([arr_b[i], arr_t[i]], [arr_b_y[i], arr_t_y[i]],
                    color=color, lw=0.5, alpha=0.55, zorder=1)
    # Top curves (lighter)
    ax.plot(sx_m_t, sy_m_t, color=COLOR_MAIN, lw=1.0, alpha=0.65, zorder=2)
    ax.plot(sx_l_t, sy_l_t, color=COLOR_LONG, lw=1.3, alpha=0.65, zorder=2)
    ax.plot(sx_s_t, sy_s_t, color=COLOR_SHORT, lw=1.3, alpha=0.65, zorder=2)
    # z arrow indicator
    ax.annotate("", xy=(0.075, 0.605), xytext=(0.075, 0.495),
                 arrowprops=dict(arrowstyle="->", lw=0.9, color="#666"))
    ax.text(0.082, 0.55, "z", ha="left", va="center", fontsize=8,
            color="#666", style="italic")
    ax.text(0.22, 0.45, "N = 4000   (extruded along z)", ha="center",
            va="top", fontsize=9, style="italic", color="#1f4e79")

    # --- 2D target box (right) ------------------------------------------
    tgt = mpatches.FancyBboxPatch(
        (0.60, 0.42), 0.36, 0.36,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        edgecolor="#a85d00", facecolor="#fff3e0", linewidth=1.5,
    )
    ax.add_patch(tgt)
    ax.text(0.78, 0.74, "2D target", ha="center", va="top",
            fontsize=11, fontweight="bold", color="#a85d00")
    ax.text(0.78, 0.69, "Y-fork spiral", ha="center", va="top",
            fontsize=9, color="#a85d00")

    cx, cy = 0.74, 0.55
    sx_m2, sy_m2, sx_l2, sy_l2, sx_s2, sy_s2 = _layout_curves(cx, cy)
    # Single flat curves — no extrusion
    ax.plot(sx_m2, sy_m2, color=COLOR_MAIN, lw=1.5)
    ax.plot(sx_l2, sy_l2, color=COLOR_LONG, lw=1.8)
    ax.plot(sx_s2, sy_s2, color=COLOR_SHORT, lw=1.8)
    # Tiny x/y axes inside the box to emphasise flatness
    ax.annotate("", xy=(0.66, 0.46), xytext=(0.625, 0.46),
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="#888"))
    ax.annotate("", xy=(0.625, 0.495), xytext=(0.625, 0.46),
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="#888"))
    ax.text(0.665, 0.452, "x", fontsize=7, color="#888", style="italic")
    ax.text(0.617, 0.495, "y", fontsize=7, color="#888", style="italic")
    ax.text(0.78, 0.45, "K = 5000   (flat)", ha="center", va="top",
            fontsize=9, style="italic", color="#a85d00")

    # --- Algorithm arrow -------------------------------------------------
    ax.annotate("", xy=(0.60, 0.60), xytext=(0.40, 0.60),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#222"))
    ax.text(0.50, 0.66, "GW / FGW", ha="center", fontsize=10,
            fontweight="bold")
    ax.text(0.50, 0.555, "(arclen feature)", ha="center", fontsize=8,
            style="italic", color="#444")

    # --- Region legend (same colours as the 3D / 2D mini sketches) ------
    ax.text(0.5, 0.34, "Regions (same colour = same region everywhere)",
            ha="center", fontsize=10, fontweight="bold")
    ax.scatter(0.10, 0.27, marker="o", s=55, c=COLOR_MAIN,
                edgecolors="black", linewidths=0.6)
    ax.text(0.13, 0.27, "spiral inner / middle / outer  (backbone, label 0)",
            va="center", fontsize=9)
    ax.scatter(0.10, 0.20, marker="^", s=70, c=COLOR_LONG,
                edgecolors="black", linewidths=0.6)
    ax.text(0.13, 0.20, "long tail  (tangent extension, label 0)",
            va="center", fontsize=9)
    ax.scatter(0.10, 0.13, marker="s", s=65, c=COLOR_SHORT,
                edgecolors="black", linewidths=0.6)
    ax.text(0.13, 0.13, "short tail  (off-axis branch, label 1)",
            va="center", fontsize=9)

    ax.text(0.5, 0.04,
            "Metric: signed Spearman ρ on geodesic arclen,  per region",
            ha="center", fontsize=8, style="italic", color="#444")


def _bundle_closest(
    src_s: np.ndarray, tgt_s: np.ndarray, anchor: float, n: int,
    src_mask: "np.ndarray | None" = None,
    tgt_mask: "np.ndarray | None" = None,
) -> "tuple[np.ndarray, np.ndarray]":
    src_vals = np.where(src_mask, src_s, np.inf) if src_mask is not None else src_s
    tgt_vals = np.where(tgt_mask, tgt_s, np.inf) if tgt_mask is not None else tgt_s
    si = np.argsort(np.abs(src_vals - anchor))[:n]
    ti = np.argsort(np.abs(tgt_vals - anchor))[:n]
    si = si[np.argsort(src_s[si])]
    ti = ti[np.argsort(tgt_s[ti])]
    return si, ti


# ---- Figure 1: dataset showcase (3D source → 2D target) -----------------

def make_datasets_figure() -> Path:
    """2x3 cluster grid. Top-left cell is empty (reserved for a schematic).
    Top row right two cells: long tail / short tail clusters.
    Bottom row: three spiral clusters (inner, middle, outer).

    Each non-empty cell is a tight pair of (3D source, 2D target). Every
    panel shows the *full* arclen-coloured cloud as small semi-transparent
    dots; the 10-point cluster pops with the region's marker (○ / ▲ / ■),
    a black edge, and full viridis colour. Lines link the 10 source points
    to their 10 parameter-matched target points.
    """
    N_SRC, N_TGT = 4000, 5000
    X, a_src, L_src = c3.sample_branched_swiss_roll(n=N_SRC, seed=0)
    Y, a_tgt, L_tgt = c3.sample_branched_spiral(n=N_TGT, seed=1)
    arclen_max = float(max(a_src.max(), a_tgt.max()))
    fork_s = float(c3.spiral_arclen(9.0).item())
    X_display = np.stack((X[:, 0], X[:, 2], X[:, 1]), axis=1)

    src_main = (L_src == 0) & (a_src <= fork_s + 1e-6)
    src_long = (L_src == 0) & (a_src > fork_s + 1e-6)
    src_short = (L_src == 1)
    tgt_main = (L_tgt == 0) & (a_tgt <= fork_s + 1e-6)
    tgt_long = (L_tgt == 0) & (a_tgt > fork_s + 1e-6)
    tgt_short = (L_tgt == 1)

    import matplotlib.colors as mcolors
    data_cmap = plt.get_cmap(DATA_CMAP)
    norm = mcolors.Normalize(vmin=0, vmax=arclen_max)

    base_x = float(np.cos(9.0)); base_y = float(np.sin(9.0))
    (d1x, d1y), (d2x, d2y) = c3._asymmetric_tail_directions(9.0, np.pi / 6)

    # 2x3 layout. Cell (0, 0) intentionally left empty.
    clusters_layout = [
        # (gs_row, gs_col, anchor, src_mask, tgt_mask, marker, label)
        (0, 1, fork_s + 0.6, src_long,  tgt_long,  "^", "long tail (≈ fork + 0.6)"),
        (0, 2, fork_s + 0.3, src_short, tgt_short, "s", "short tail (≈ fork + 0.3)"),
        (1, 0, 0.6,           src_main, tgt_main,  "o", "spiral inner (≈ 0.6)"),
        (1, 1, 2.5,           src_main, tgt_main,  "o", "spiral middle (≈ 2.5)"),
        (1, 2, 4.5,           src_main, tgt_main,  "o", "spiral outer (≈ 4.5)"),
    ]
    n_per_cluster = 10

    fig = plt.figure(figsize=(19, 14))
    outer_gs = GridSpec(2, 3, figure=fig, wspace=0.13, hspace=0.20,
                         left=0.03, right=0.93, top=0.93, bottom=0.04)

    # Top-left cell: schematic of the experiment
    ax_schematic = fig.add_subplot(outer_gs[0, 0])
    _draw_schematic(ax_schematic)

    last_sc = None
    for (gs_row, gs_col, anchor, sm, tm, marker, label) in clusters_layout:
        # Each cell is itself a tight 1x2 (3D source, 2D target) sub-grid
        cell_gs = outer_gs[gs_row, gs_col].subgridspec(1, 2, wspace=0.03)
        ax_src = fig.add_subplot(cell_gs[0, 0], projection="3d")
        ax_tgt = fig.add_subplot(cell_gs[0, 1])

        # Pick the 10 source / 10 target points closest to the anchor
        src_dist = np.where(sm, np.abs(a_src - anchor), np.inf)
        tgt_dist = np.where(tm, np.abs(a_tgt - anchor), np.inf)
        src_idx = np.argsort(src_dist)[:n_per_cluster]
        tgt_idx = np.argsort(tgt_dist)[:n_per_cluster]
        src_idx = src_idx[np.argsort(a_src[src_idx])]
        tgt_idx = tgt_idx[np.argsort(a_tgt[tgt_idx])]

        # --- 3D source panel ---------------------------------------------
        _overlay_swiss_roll_surface(ax_src, vmax=arclen_max, alpha=0.16)
        _overlay_tail_strip(ax_src, (base_x, base_y), (d1x, d1y), length=1.2,
                              alpha=0.18)
        _overlay_tail_strip(ax_src, (base_x, base_y), (d2x, d2y), length=0.6,
                              alpha=0.18)
        # All points coloured by arclen, small + semi-transparent
        ax_src.scatter(
            X_display[:, 0], X_display[:, 1], X_display[:, 2],
            c=a_src, cmap=DATA_CMAP, s=2.0, alpha=0.18,
            vmin=0, vmax=arclen_max, edgecolors="none", depthshade=True,
        )
        # Highlighted cluster: region marker, full colour, black edge
        ax_src.scatter(
            X_display[src_idx, 0], X_display[src_idx, 1], X_display[src_idx, 2],
            c=a_src[src_idx], cmap=DATA_CMAP, s=70, vmin=0, vmax=arclen_max,
            marker=marker, edgecolors="black", linewidths=0.7,
        )
        ax_src.set_xlim(-XY_LIM, XY_LIM); ax_src.set_ylim(-XY_LIM, XY_LIM)
        ax_src.set_zlim(0, 1.0)
        ax_src.set_xticks([-1, 0, 1])
        ax_src.set_yticks([-1, 0, 1])
        ax_src.set_zticks([0, 1])
        ax_src.tick_params(labelsize=7)
        _apply_3d_view(ax_src)

        # --- 2D target panel ---------------------------------------------
        ax_tgt.scatter(
            Y[:, 0], Y[:, 1],
            c=a_tgt, cmap=DATA_CMAP, s=2.0, alpha=0.20,
            vmin=0, vmax=arclen_max, edgecolors="none",
        )
        sc = ax_tgt.scatter(
            Y[tgt_idx, 0], Y[tgt_idx, 1], c=a_tgt[tgt_idx], cmap=DATA_CMAP,
            s=70, vmin=0, vmax=arclen_max, marker=marker,
            edgecolors="black", linewidths=0.7,
        )
        last_sc = sc
        ax_tgt.set_xlim(-XY_LIM, XY_LIM); ax_tgt.set_ylim(-XY_LIM, XY_LIM)
        ax_tgt.set_aspect("equal")
        ax_tgt.set_xticks([-1, 0, 1])
        ax_tgt.set_yticks([-1, 0, 1])
        ax_tgt.set_xlabel(""); ax_tgt.set_ylabel("")
        ax_tgt.tick_params(labelsize=8)

        # Cluster label centred above the (source, target) pair
        bbox_src = ax_src.get_position()
        bbox_tgt = ax_tgt.get_position()
        cx = (bbox_src.x0 + bbox_tgt.x1) / 2
        cy = max(bbox_src.y1, bbox_tgt.y1) + 0.005
        fig.text(cx, cy, label, ha="center", va="bottom",
                  fontsize=11, fontweight="bold")

        # 10 connection lines, all coloured by the cluster's anchor arclen
        bundle_color = data_cmap(norm(float(anchor)))
        bundles = [(src_idx, tgt_idx, bundle_color)]
        _add_bundled_lines(fig, ax_src, ax_tgt, X_display, Y, bundles,
                             alpha=0.7, linewidth=1.1)

    # Shared colorbar on the right
    if last_sc is not None:
        cbar_ax = fig.add_axes((0.945, 0.10, 0.012, 0.78))
        cbar = fig.colorbar(last_sc, cax=cbar_ax, label="geodesic arclen")
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        f"Five source clusters → matched target clusters · "
        f"3D Swiss roll N={N_SRC} → 2D spiral K={N_TGT}",
        fontsize=13, y=0.97,
    )
    out = FIG_DIR / "datasets.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---- Figure 2: solver effects on C3 --------------------------------------

def make_solver_effects_figure() -> Path:
    """2 panels: pure GW vs FGW on the C3 Y-fork task (3D → 2D).
    Target 2D points coloured by argmax-matched source arclen.
    """
    X, a_src, L_src = c3.sample_branched_swiss_roll(n=4000, seed=0)
    Y, a_tgt, L_tgt = c3.sample_branched_spiral(n=5000, seed=1)
    arclen_max = float(max(a_src.max(), a_tgt.max()))
    fork_s = float(c3.spiral_arclen(9.0).item())

    T_pure = c3.run_torchgw_landmark(X, Y, seed=0)["T"]
    T_fgw = c3.run_torchgw_fused(X, Y, a_src, a_tgt, seed=0)["T"]

    # For each target point, matched source arclen via argmax over rows
    def _per_target_matched_src(T):
        return a_src[np.argmax(T, axis=0)]

    matched_pure = _per_target_matched_src(T_pure)
    matched_fgw = _per_target_matched_src(T_fgw)

    # Backbone / branch split for Spearman (label 0 vs label 1)
    def _rho_backbone(T):
        matched = _per_target_matched_src(T)
        m = (L_tgt == 0)
        return _get_rho(a_tgt[m], matched[m])

    def _rho_tail(T):
        matched = _per_target_matched_src(T)
        m = (L_tgt == 1)
        return _get_rho(a_tgt[m], matched[m])

    def _branch_acc(T):
        matched_label = L_src[np.argmax(T, axis=0)]
        return float(np.mean(L_tgt == matched_label))

    rho_bb_pure = _rho_backbone(T_pure); rho_t_pure = _rho_tail(T_pure)
    rho_bb_fgw = _rho_backbone(T_fgw); rho_t_fgw = _rho_tail(T_fgw)
    acc_pure = _branch_acc(T_pure); acc_fgw = _branch_acc(T_fgw)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def _panel(ax, matched, title):
        main_mask = (L_tgt == 0) & (a_tgt <= fork_s + 1e-6)
        long_mask = (L_tgt == 0) & (a_tgt > fork_s + 1e-6)
        short_mask = (L_tgt == 1)
        sc = ax.scatter(Y[main_mask, 0], Y[main_mask, 1],
                         c=matched[main_mask], cmap=MATCH_CMAP, s=14,
                         vmin=0, vmax=arclen_max, marker="o")
        ax.scatter(Y[long_mask, 0], Y[long_mask, 1], c=matched[long_mask],
                    cmap=MATCH_CMAP, vmin=0, vmax=arclen_max, s=26, marker="^")
        ax.scatter(Y[short_mask, 0], Y[short_mask, 1], c=matched[short_mask],
                    cmap=MATCH_CMAP, vmin=0, vmax=arclen_max, s=26, marker="s")
        ax.set_xlim(-XY_LIM, XY_LIM); ax.set_ylim(-XY_LIM, XY_LIM)
        ax.set_aspect("equal")
        ax.set_xlabel("target x"); ax.set_ylabel("target y")
        ax.set_title(title, fontsize=TITLE_SIZE)
        return sc

    sc0 = _panel(axes[0], matched_pure,
                   f"pure GW (torchgw-landmark)\n"
                   f"branch_acc={acc_pure:.3f}  "
                   f"backbone-ρ={rho_bb_pure:+.3f}  tail-ρ={rho_t_pure:+.3f}")
    _panel(axes[1], matched_fgw,
            f"FGW (torchgw-fused, arclen feature)\n"
            f"branch_acc={acc_fgw:.3f}  "
            f"backbone-ρ={rho_bb_fgw:+.3f}  tail-ρ={rho_t_fgw:+.3f}")

    fig.colorbar(sc0, ax=axes, orientation="vertical", fraction=0.03,
                  pad=0.02, shrink=0.85, label="matched source arclen")
    fig.suptitle("Target 2D points coloured by the argmax-matched source arclen "
                 "(forward match → colour ramp aligned with target's own arclen)",
                 fontsize=13, y=0.98)
    out = FIG_DIR / "solver_effects.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---- Figure 3: Spearman bar comparison ----------------------------------

def make_spearman_bar_figure() -> Path:
    X, a_src, L_src = c3.sample_branched_swiss_roll(n=4000, seed=0)
    Y, a_tgt, L_tgt = c3.sample_branched_spiral(n=5000, seed=1)

    T_pure = c3.run_torchgw_landmark(X, Y, seed=0)["T"]
    T_fgw = c3.run_torchgw_fused(X, Y, a_src, a_tgt, seed=0)["T"]

    def _rho_region(T, region_mask_tgt):
        matched_src = a_src[np.argmax(T, axis=0)]
        return _get_rho(a_tgt[region_mask_tgt], matched_src[region_mask_tgt])

    bb_mask = (L_tgt == 0)
    t_mask = (L_tgt == 1)
    vals = [
        ("pure GW\nbackbone", _rho_region(T_pure, bb_mask)),
        ("pure GW\ntail",      _rho_region(T_pure, t_mask)),
        ("FGW\nbackbone",      _rho_region(T_fgw, bb_mask)),
        ("FGW\ntail",          _rho_region(T_fgw, t_mask)),
    ]

    labels = [v[0] for v in vals]
    signed = [v[1] for v in vals]
    abs_vals = [abs(v) for v in signed]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, signed, width, label="signed ρ",
           color=["#4575b4" if s >= 0 else "#d73027" for s in signed])
    ax.bar(x + width / 2, abs_vals, width, label="|ρ|",
           color="#fdae61", alpha=0.75)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axhline(0.95, color="green", linestyle=":", linewidth=0.8,
               label="threshold 0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Spearman ρ on arclen")
    ax.set_title("C3 Y-fork (3D → 2D) — FGW closes the backbone/tail gap of pure GW")
    ax.set_ylim(-0.1, 1.15)
    ax.legend(loc="lower right")
    for xi, v in zip(x, signed):
        ax.annotate(f"{v:+.3f}", xy=(xi - width / 2, v + 0.02),
                    ha="center", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "spearman_bar.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---- Figure 4: C3 deep-dive ---------------------------------------------

def make_c3_zoom_figure() -> Path:
    X, a_src, L_src = c3.sample_branched_swiss_roll(n=4000, seed=0)
    Y, a_tgt, L_tgt = c3.sample_branched_spiral(n=5000, seed=1)
    arclen_max = float(max(a_src.max(), a_tgt.max()))
    fork_s = float(c3.spiral_arclen(9.0).item())
    T = c3.run_torchgw_fused(X, Y, a_src, a_tgt, seed=0)["T"]

    # For each target point, find argmax-matched source
    matched_src_scalar = a_src[np.argmax(T, axis=0)]
    matched_src_label = L_src[np.argmax(T, axis=0)]

    main_mask_t = (L_tgt == 0) & (a_tgt <= fork_s + 1e-6)
    long_mask_t = (L_tgt == 0) & (a_tgt > fork_s + 1e-6)
    short_mask_t = (L_tgt == 1)

    rho_bb = _get_rho(a_tgt[L_tgt == 0], matched_src_scalar[L_tgt == 0])
    rho_t = _get_rho(a_tgt[L_tgt == 1], matched_src_scalar[L_tgt == 1])
    branch_acc = float(np.mean(L_tgt == matched_src_label))

    main_mask_s = (L_src == 0) & (a_src <= fork_s + 1e-6)
    long_mask_s = (L_src == 0) & (a_src > fork_s + 1e-6)
    short_mask_s = (L_src == 1)

    X_display = np.stack((X[:, 0], X[:, 2], X[:, 1]), axis=1)

    fig = plt.figure(figsize=(16, 5))

    # Panel 1: 3D source with labels
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    _overlay_swiss_roll_surface(ax, vmax=arclen_max)
    base_x = float(np.cos(9.0)); base_y = float(np.sin(9.0))
    (d1x, d1y), (d2x, d2y) = c3._asymmetric_tail_directions(9.0, np.pi / 6)
    _overlay_tail_strip(ax, (base_x, base_y), (d1x, d1y), length=1.2)
    _overlay_tail_strip(ax, (base_x, base_y), (d2x, d2y), length=0.6)
    ax.scatter(X_display[main_mask_s, 0], X_display[main_mask_s, 1],
               X_display[main_mask_s, 2], c="steelblue", s=12,
               label="main spiral", edgecolors="black", linewidths=0.25)
    ax.scatter(X_display[long_mask_s, 0], X_display[long_mask_s, 1],
               X_display[long_mask_s, 2], c="mediumseagreen", s=18,
               marker="^", label="long tail",
               edgecolors="black", linewidths=0.25)
    ax.scatter(X_display[short_mask_s, 0], X_display[short_mask_s, 1],
               X_display[short_mask_s, 2], c="crimson", s=18,
               marker="s", label="short tail",
               edgecolors="black", linewidths=0.25)
    ax.set_xlim(-XY_LIM, XY_LIM); ax.set_ylim(-XY_LIM, XY_LIM)
    ax.set_zlim(0, 1.0)
    _apply_3d_view(ax)
    ax.set_title("Source (3D): Y-fork Swiss roll")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 2: 2D target with labels
    ax = fig.add_subplot(1, 4, 2)
    ax.scatter(Y[main_mask_t, 0], Y[main_mask_t, 1], c="steelblue", s=12,
               label="main spiral")
    ax.scatter(Y[long_mask_t, 0], Y[long_mask_t, 1], c="mediumseagreen",
               s=18, marker="^", label="long tail")
    ax.scatter(Y[short_mask_t, 0], Y[short_mask_t, 1], c="crimson",
               s=18, marker="s", label="short tail")
    ax.set_title("Target (2D): Y-fork spiral")
    ax.set_xlim(-XY_LIM, XY_LIM); ax.set_ylim(-XY_LIM, XY_LIM)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=9)

    # Panel 3: target coloured by matched source arclen
    ax = fig.add_subplot(1, 4, 3)
    sc = ax.scatter(Y[main_mask_t, 0], Y[main_mask_t, 1],
                     c=matched_src_scalar[main_mask_t], cmap=MATCH_CMAP,
                     s=14, vmin=0, vmax=arclen_max, marker="o")
    ax.scatter(Y[long_mask_t, 0], Y[long_mask_t, 1],
               c=matched_src_scalar[long_mask_t], cmap=MATCH_CMAP,
               vmin=0, vmax=arclen_max, s=22, marker="^")
    ax.scatter(Y[short_mask_t, 0], Y[short_mask_t, 1],
               c=matched_src_scalar[short_mask_t], cmap=MATCH_CMAP,
               vmin=0, vmax=arclen_max, s=22, marker="s")
    ax.set_title(f"matched source arclen (FGW)\n"
                  f"backbone-ρ = {rho_bb:+.4f}\n"
                  f"tail-ρ = {rho_t:+.4f}")
    ax.set_xlim(-XY_LIM, XY_LIM); ax.set_ylim(-XY_LIM, XY_LIM)
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, shrink=0.8, label="matched source arclen")

    # Panel 4: label match propagation
    ax = fig.add_subplot(1, 4, 4)
    correct = (L_tgt == matched_src_label)
    ax.scatter(Y[correct, 0], Y[correct, 1], c="#2ca02c", s=12,
               label=f"label matches ({int(correct.sum())})")
    ax.scatter(Y[~correct, 0], Y[~correct, 1], c="#d62728", s=22, marker="x",
               label=f"label mismatch ({int((~correct).sum())})")
    ax.set_title(f"branch label propagation\nbranch_accuracy = {branch_acc:.4f}")
    ax.set_xlim(-XY_LIM, XY_LIM); ax.set_ylim(-XY_LIM, XY_LIM)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=9)

    fig.suptitle("C3 deep-dive — 3D source → 2D target alignment under FGW (arclen feature)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "c3_detail.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def make_schematic_files() -> "tuple[Path, Path]":
    """Save the experiment schematic standalone in both SVG and PNG."""
    out_svg = FIG_DIR / "schematic.svg"
    out_png = FIG_DIR / "schematic.png"
    for path, fmt in ((out_svg, "svg"), (out_png, "png")):
        fig, ax = plt.subplots(figsize=(7, 5.4))
        _draw_schematic(ax)
        fig.savefig(path, format=fmt, bbox_inches="tight")
        plt.close(fig)
    return out_svg, out_png


if __name__ == "__main__":
    print("[fig] schematic.svg / .png ...")
    p_svg, p_png = make_schematic_files()
    print(f"  → {p_svg}\n  → {p_png}")
    print("[fig] datasets.png ...")
    print(f"  → {make_datasets_figure()}")
    print("[fig] solver_effects.png ...")
    print(f"  → {make_solver_effects_figure()}")
    print("[fig] spearman_bar.png ...")
    print(f"  → {make_spearman_bar_figure()}")
    print("[fig] c3_detail.png ...")
    print(f"  → {make_c3_zoom_figure()}")
    print("\nAll figures written to", FIG_DIR)
