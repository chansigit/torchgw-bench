#!/usr/bin/env python
"""Visualise the TACO dataset: a grid of source→target mesh pairs with
ground-truth correspondence lines. Uses GridSpec to tighten the
within-pair gap so correspondence lines are short and legible.

Output: docs/figures/c6_taco_dataset.png
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

DATA_ROOT = REPO / "data" / "core_06_shape" / "taco"


def _first_pair_per_class() -> list[tuple[str, str]]:
    pairs_txt = DATA_ROOT / "pairs.txt"
    all_pairs = [l.strip() for l in pairs_txt.read_text().splitlines() if l.strip()]
    by_class: dict[str, tuple[str, str]] = {}
    for p in all_pairs:
        a, b = p.split(",")
        cls = re.match(r"[a-z]+", a).group()  # type: ignore[union-attr]
        by_class.setdefault(cls, (a, b))
    return [by_class[c] for c in sorted(by_class)]


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
    # Tighten axis limits to the data with a small margin
    margins = 0.02 * (V.max(axis=0) - V.min(axis=0))
    ax.set_xlim(V[:, 0].min() - margins[0], V[:, 0].max() + margins[0])
    ax.set_ylim(V[:, 1].min() - margins[1], V[:, 1].max() + margins[1])
    ax.set_zlim(V[:, 2].min() - margins[2], V[:, 2].max() + margins[2])


def _proj(ax, V: np.ndarray) -> np.ndarray:
    xs, ys, _ = proj3d.proj_transform(V[:, 0], V[:, 1], V[:, 2], ax.get_proj())
    return ax.transData.transform(np.column_stack([xs, ys]))


def main():
    pairs = _first_pair_per_class()
    n_lines = 12
    ncols = 3
    nrows = 3

    # 3×3 pairs, each pair = 2 subpanels sharing a close gap. Use 3
    # top-level columns with nested 2-col gridspec inside each; outer
    # wspace creates inter-pair gap, inner uses tiny wspace.
    fig = plt.figure(figsize=(15, 12))
    outer = GridSpec(nrows, ncols, figure=fig, hspace=0.08, wspace=0.18)
    cell_of_pair = [(0, 1), (0, 1), (0, 1)]  # unused, we'll nest

    panel_info = []
    for k, (src, tgt) in enumerate(pairs):
        r, c = divmod(k, ncols)
        inner = outer[r, c].subgridspec(1, 2, wspace=0.0)
        ax_src = fig.add_subplot(inner[0, 0], projection="3d")
        ax_tgt = fig.add_subplot(inner[0, 1], projection="3d")
        try:
            V_src, _, V_tgt, _, gt = run.load_taco_pair(src, tgt)
        except FileNotFoundError:
            ax_src.text2D(0.5, 0.5, "missing", transform=ax_src.transAxes,
                           ha="center")
            continue
        cls = re.match(r"[a-z]+", src).group()  # type: ignore[union-attr]

        rng = np.random.default_rng(0)
        n_bg = min(V_src.shape[0], 2500)
        n_bg_t = min(V_tgt.shape[0], 2500)
        bg_s = rng.choice(V_src.shape[0], n_bg, replace=False)
        bg_t = rng.choice(V_tgt.shape[0], n_bg_t, replace=False)
        ax_src.scatter(*V_src[bg_s].T, s=0.8, c="0.78", linewidths=0, alpha=0.5)
        ax_tgt.scatter(*V_tgt[bg_t].T, s=0.8, c="0.78", linewidths=0, alpha=0.5)

        valid = np.flatnonzero(gt >= 0)
        if valid.size < n_lines:
            continue
        line_src = rng.choice(valid, n_lines, replace=False)
        line_tgt = gt[line_src]
        zs = V_src[line_src, 2]
        z_norm = (zs - zs.min()) / (zs.ptp() + 1e-9)
        colors = colormaps["viridis"](z_norm)

        ax_src.scatter(*V_src[line_src].T, s=30, c=colors,
                        edgecolors="black", linewidths=0.4, zorder=5)
        ax_tgt.scatter(*V_tgt[line_tgt].T, s=30, c=colors,
                        edgecolors="black", linewidths=0.4, zorder=5)

        _camera(ax_src, cls, V_src)
        _camera(ax_tgt, cls, V_tgt)
        ax_src.set_title(src, fontsize=11, pad=-4)
        ax_tgt.set_title(tgt, fontsize=11, pad=-4)
        panel_info.append((ax_src, ax_tgt, line_src, line_tgt, V_src, V_tgt, colors))

    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for ax_src, ax_tgt, src_idx, tgt_idx, V_src, V_tgt, colors in panel_info:
        src_f = inv.transform(_proj(ax_src, V_src[src_idx]))
        tgt_f = inv.transform(_proj(ax_tgt, V_tgt[tgt_idx]))
        for ps, pt, col in zip(src_f, tgt_f, colors):
            line = mlines.Line2D([ps[0], pt[0]], [ps[1], pt[1]],
                                   transform=fig.transFigure,
                                   color=col, alpha=0.55,
                                   linewidth=0.9, zorder=1)
            fig.add_artist(line)

    fig.suptitle("TACO dataset — representative pairs with GT correspondence",
                 fontsize=14, y=0.97)
    out = FIG_DIR / "c6_taco_dataset.png"
    fig.savefig(out, dpi=170)
    print(f"[c6-viz] wrote {out}")


if __name__ == "__main__":
    main()
