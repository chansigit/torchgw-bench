#!/usr/bin/env python
"""Visualise the TACO dataset: a grid of source→target mesh pairs with
ground-truth correspondence lines drawn for a small random sample of
vertices. Uses only pairs that exist as forward GT files.

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
import matplotlib.cm as cm
import matplotlib.lines as mlines

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tracks" / "core" / "06_shape_correspondence"))
import run  # type: ignore[import-not-found]

FIG_DIR = REPO / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = REPO / "data" / "core_06_shape" / "taco"


def _first_pair_per_class() -> list[tuple[str, str]]:
    """Pick one representative pair per animal class from pairs.txt."""
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


def _proj_to_display(ax, V: np.ndarray) -> np.ndarray:
    """Project 3D world coords to 2D display pixel coords using
    mpl_toolkits.mplot3d.proj3d (the canonical path matplotlib uses
    internally for scatter3d)."""
    from mpl_toolkits.mplot3d import proj3d
    xs, ys, _zs = proj3d.proj_transform(V[:, 0], V[:, 1], V[:, 2],
                                           ax.get_proj())
    return ax.transData.transform(np.column_stack([xs, ys]))


def main():
    pairs = _first_pair_per_class()[:9]   # up to 9 classes
    n_show = 9
    pairs = pairs[:n_show]
    ncols = 3
    nrows = 3
    fig = plt.figure(figsize=(15, 13))
    axes = fig.subplots(nrows, ncols * 2, subplot_kw={"projection": "3d"})
    axes = axes.reshape(nrows, ncols, 2)

    seed = 0
    n_lines = 20
    panel_info = []   # (row, col, src_name, tgt_name, gt, V_src, V_tgt, src_idx)

    for k, (src_name, tgt_name) in enumerate(pairs):
        r, c = divmod(k, ncols)
        ax_src, ax_tgt = axes[r, c, 0], axes[r, c, 1]
        try:
            V_src, _, V_tgt, _, gt = run.load_taco_pair(src_name, tgt_name)
        except FileNotFoundError:
            ax_src.text2D(0.5, 0.5, "missing", transform=ax_src.transAxes,
                           ha="center")
            ax_tgt.axis("off")
            continue

        cls = re.match(r"[a-z]+", src_name).group()  # type: ignore[union-attr]

        rng = np.random.default_rng(seed)
        n_bg = min(V_src.shape[0], 2500)
        n_bg_t = min(V_tgt.shape[0], 2500)
        bg_src_idx = rng.choice(V_src.shape[0], n_bg, replace=False)
        bg_tgt_idx = rng.choice(V_tgt.shape[0], n_bg_t, replace=False)
        ax_src.scatter(*V_src[bg_src_idx].T, s=0.9, c="0.78",
                        linewidths=0, alpha=0.55)
        ax_tgt.scatter(*V_tgt[bg_tgt_idx].T, s=0.9, c="0.78",
                        linewidths=0, alpha=0.55)

        # valid source indices (GT ≥ 0)
        valid = np.flatnonzero(gt >= 0)
        if valid.size < n_lines:
            continue
        line_src_idx = rng.choice(valid, n_lines, replace=False)
        line_tgt_idx = gt[line_src_idx]
        zs = V_src[line_src_idx, 2]
        z_norm = (zs - zs.min()) / (zs.ptp() + 1e-9)
        colors = cm.get_cmap("viridis")(z_norm)

        ax_src.scatter(*V_src[line_src_idx].T, s=28, c=colors,
                        edgecolors="black", linewidths=0.35, zorder=5)
        ax_tgt.scatter(*V_tgt[line_tgt_idx].T, s=28, c=colors,
                        edgecolors="black", linewidths=0.35, zorder=5)

        _camera(ax_src, cls, V_src)
        _camera(ax_tgt, cls, V_tgt)
        ax_src.set_title(src_name, fontsize=10, pad=0)
        ax_tgt.set_title(tgt_name, fontsize=10, pad=0)

        panel_info.append((ax_src, ax_tgt, line_src_idx, line_tgt_idx,
                            V_src, V_tgt, colors))

    # Now that 3D views are set, compute display coords and draw bundles
    fig.canvas.draw()
    for ax_src, ax_tgt, src_idx, tgt_idx, V_src, V_tgt, colors in panel_info:
        src_px = _proj_to_display(ax_src, V_src[src_idx])
        tgt_px = _proj_to_display(ax_tgt, V_tgt[tgt_idx])
        inv = fig.transFigure.inverted()
        src_f = inv.transform(src_px)
        tgt_f = inv.transform(tgt_px)
        for ps, pt, col in zip(src_f, tgt_f, colors):
            line = mlines.Line2D([ps[0], pt[0]], [ps[1], pt[1]],
                                   transform=fig.transFigure,
                                   color=col, alpha=0.35,
                                   linewidth=0.6, zorder=1)
            fig.add_artist(line)

    fig.suptitle("TACO dataset — representative pairs with GT correspondence",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / "c6_taco_dataset.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"[c6-viz] wrote {out}")


if __name__ == "__main__":
    main()
