#!/usr/bin/env python
"""Visualise solver predictions on TACO pairs.

For each of 3 representative pairs, run torchgw-dijkstra and pot-exact-gpu
at N=800 subsampled vertices and draw three (src → tgt) panels per row:

    column 1: ground-truth correspondence (oracle)
    column 2: torchgw-dijkstra argmax prediction
    column 3: pot-exact-gpu argmax prediction

Source panel is shared across columns (same subsample, same colored
sample points). Correspondence lines in each column go from src dots
to the predicted (or GT) target positions.

Output: docs/figures/c6_mapping_viz.png
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib import colormaps
from mpl_toolkits.mplot3d import proj3d

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tracks" / "core" / "06_shape_correspondence"))
import run  # type: ignore[import-not-found]

FIG_DIR = REPO / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("cat0", "cat1"),
    ("horse0", "horse5"),
    ("david0", "david1"),
]
N_VERTS = 800
N_LINES = 18


def _camera(ax, cls: str, V: np.ndarray) -> None:
    if cls in ("cat", "dog", "wolf", "horse"):
        ax.view_init(elev=12, azim=-60)
    elif cls == "centaur":
        ax.view_init(elev=12, azim=-75)
    else:
        ax.view_init(elev=12, azim=-90)
    ranges = V.ptp(axis=0)
    ax.set_box_aspect(ranges / ranges.max())
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for name in "xyz":
        getattr(ax, f"{name}axis").pane.set_visible(False)
    ax.grid(False)
    margins = 0.02 * (V.max(axis=0) - V.min(axis=0))
    ax.set_xlim(V[:, 0].min() - margins[0], V[:, 0].max() + margins[0])
    ax.set_ylim(V[:, 1].min() - margins[1], V[:, 1].max() + margins[1])
    ax.set_zlim(V[:, 2].min() - margins[2], V[:, 2].max() + margins[2])


def _proj(ax, V: np.ndarray) -> np.ndarray:
    xs, ys, _ = proj3d.proj_transform(V[:, 0], V[:, 1], V[:, 2], ax.get_proj())
    return ax.transData.transform(np.column_stack([xs, ys]))


def main():
    methods = [
        ("GT (oracle)", "gt"),
        ("torchgw-dijkstra", "torchgw"),
        ("pot-exact-gpu", "pot"),
    ]

    fig = plt.figure(figsize=(15, 12))
    outer = GridSpec(len(PAIRS), len(methods), figure=fig,
                      hspace=0.18, wspace=0.12)

    panel_info = []
    for r, (src_name, tgt_name) in enumerate(PAIRS):
        # Load + subsample once per pair; reuse across methods.
        V_src_full, _, V_tgt_full, _, gt_full = run.load_taco_pair(src_name, tgt_name)
        V_src, V_tgt, gt = run.subsample_pair(
            V_src_full, V_tgt_full, gt_full,
            N_VERTS, N_VERTS, seed=0,
        )
        # Solve
        out_tg = run.run_torchgw_dijkstra(V_src, V_tgt, seed=0, max_iter=300)
        out_pot = run.run_pot_exact_gpu(V_src, V_tgt, seed=0, max_iter=500)
        T_tg = out_tg["T"]
        T_pot = out_pot["T"]
        pred_tg = T_tg.argmax(axis=1)
        pred_pot = T_pot.argmax(axis=1)

        cls = re.match(r"[a-z]+", src_name).group()  # type: ignore[union-attr]
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(V_src.shape[0], N_LINES, replace=False)
        zs = V_src[sample_idx, 2]
        z_norm = (zs - zs.min()) / (zs.ptp() + 1e-9)
        colors = colormaps["viridis"](z_norm)

        for c, (label, method) in enumerate(methods):
            inner = outer[r, c].subgridspec(1, 2, wspace=0.0)
            ax_src = fig.add_subplot(inner[0, 0], projection="3d")
            ax_tgt = fig.add_subplot(inner[0, 1], projection="3d")

            # Background scatter
            ax_src.scatter(*V_src.T, s=1.0, c="0.78", linewidths=0, alpha=0.45)
            ax_tgt.scatter(*V_tgt.T, s=1.0, c="0.78", linewidths=0, alpha=0.45)

            # Sample dots on src (same for every column)
            ax_src.scatter(*V_src[sample_idx].T, s=34, c=colors,
                            edgecolors="black", linewidths=0.4, zorder=5)

            # Targets chosen per method
            if method == "gt":
                tgt_idx = gt[sample_idx]
            elif method == "torchgw":
                tgt_idx = pred_tg[sample_idx]
            else:
                tgt_idx = pred_pot[sample_idx]

            ax_tgt.scatter(*V_tgt[tgt_idx].T, s=34, c=colors,
                            edgecolors="black", linewidths=0.4, zorder=5)

            _camera(ax_src, cls, V_src)
            _camera(ax_tgt, cls, V_tgt)
            if r == 0:
                ax_src.set_title(f"{label}\n{src_name}", fontsize=10, pad=-4)
            else:
                ax_src.set_title(src_name, fontsize=10, pad=-4)
            ax_tgt.set_title(tgt_name, fontsize=10, pad=-4)

            panel_info.append((ax_src, ax_tgt, sample_idx, tgt_idx,
                                V_src, V_tgt, colors))

    # After all draws, overlay correspondence lines
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for ax_src, ax_tgt, s_idx, t_idx, V_src, V_tgt, colors in panel_info:
        src_f = inv.transform(_proj(ax_src, V_src[s_idx]))
        tgt_f = inv.transform(_proj(ax_tgt, V_tgt[t_idx]))
        for ps, pt, col in zip(src_f, tgt_f, colors):
            line = mlines.Line2D([ps[0], pt[0]], [ps[1], pt[1]],
                                   transform=fig.transFigure,
                                   color=col, alpha=0.6,
                                   linewidth=0.9, zorder=1)
            fig.add_artist(line)

    fig.suptitle("C6 TACO — solver predictions vs ground truth  (N=800, 18 sampled correspondences)",
                 fontsize=13, y=0.96)
    out = FIG_DIR / "c6_mapping_viz.png"
    fig.savefig(out, dpi=170)
    print(f"[c6-mapping] wrote {out}")


if __name__ == "__main__":
    main()
