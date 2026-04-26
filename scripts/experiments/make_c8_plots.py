#!/usr/bin/env python
"""C8 plots: quality vs resolution, wall vs resolution, survival bars."""
from __future__ import annotations
import json, pathlib
import matplotlib.pyplot as plt
import numpy as np

SOLVER_ORDER = ["fugw-native", "pot-entropic-fgw",
                "torchgw-balanced", "torchgw-unbalanced"]
SOLVER_COLOR = {
    "fugw-native":         "#444",
    "pot-entropic-fgw":    "#1f77b4",
    "torchgw-balanced":    "#9467bd",
    "torchgw-unbalanced":  "#d62728",
}
RES_ORDER = ["fsaverage5", "fsaverage6", "fsaverage7"]
RES_NV = {"fsaverage5": 10242, "fsaverage6": 40962, "fsaverage7": 163842}


def _load(results_dir: pathlib.Path) -> list[dict]:
    out = []
    for p in sorted(results_dir.glob("core_08_brain__*.json")):
        d = json.load(open(p))
        out.append(d)
    return out


def _by(records, key_path: tuple, only_ok: bool = True):
    bins: dict = {}
    for r in records:
        if only_ok and r.get("status") != "ok":
            continue
        v = r
        for k in key_path:
            v = v.get(k) if isinstance(v, dict) else None
        if v is None: continue
        bins.setdefault((r["solver"], r["resolution"]), []).append(float(v))
    return bins


def _plot_metric(records, key_path, ylabel, title, out_path, log_y=False):
    bins = _by(records, key_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    for solver in SOLVER_ORDER:
        xs, ys, errs = [], [], []
        for res in RES_ORDER:
            if (solver, res) not in bins: continue
            v = bins[(solver, res)]
            xs.append(RES_NV[res]); ys.append(np.mean(v)); errs.append(np.std(v))
        if xs:
            ax.errorbar(xs, ys, yerr=errs, marker="o", label=solver,
                        color=SOLVER_COLOR[solver])
    ax.set_xscale("log"); ax.set_xlabel("# vertices (per hemisphere)")
    if log_y: ax.set_yscale("log")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def _plot_survival(records, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.18
    x = np.arange(len(RES_ORDER))
    for i, solver in enumerate(SOLVER_ORDER):
        rates = []
        for res in RES_ORDER:
            cells = [r for r in records if r["solver"] == solver
                     and r["resolution"] == res]
            if not cells:
                rates.append(0.0); continue
            ok = sum(1 for r in cells if r.get("status") == "ok")
            rates.append(ok / len(cells))
        ax.bar(x + i * width - 1.5 * width, rates, width=width,
               label=solver, color=SOLVER_COLOR[solver])
    ax.set_xticks(x); ax.set_xticklabels(RES_ORDER); ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of pairs run successfully")
    ax.set_title("C8 — solver survival across resolutions")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    repo = pathlib.Path(__file__).resolve().parents[2]
    rdir = repo / "results" / "c8_brain_alignment"
    figdir = repo / "docs" / "figures"; figdir.mkdir(exist_ok=True, parents=True)

    records = _load(rdir)
    if not records:
        raise SystemExit(f"no records in {rdir}")

    _plot_metric(records, ("metrics", "func_corr_holdout_mean"),
                 "Held-out functional Pearson r",
                 "C8 — alignment quality vs resolution",
                 figdir / "c8_quality.png")
    _plot_metric(records, ("metrics", "retrieval_top1"),
                 "Retrieval accuracy (top-1)",
                 "C8 — contrast retrieval vs resolution",
                 figdir / "c8_retrieval.png")
    _plot_metric(records, ("efficiency", "wall_s_total"),
                 "Wall time (s)",
                 "C8 — per-pair wall vs resolution",
                 figdir / "c8_wall.png", log_y=True)
    _plot_survival(records, figdir / "c8_survival.png")


if __name__ == "__main__":
    main()
