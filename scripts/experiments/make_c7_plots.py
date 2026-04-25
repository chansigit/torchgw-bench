#!/usr/bin/env python
"""C7 plots: quality vs N_per_cell, wall vs N_per_cell, per-pair latency."""
from __future__ import annotations
import argparse, glob, json, pathlib
import matplotlib.pyplot as plt
import numpy as np

SOLVER_ORDER = ["cajal-native", "pot-entropic-gpu", "pot-exact-gpu",
                "torchgw-precomputed"]
SOLVER_COLOR = {
    "cajal-native":         "#444",
    "pot-entropic-gpu":     "#1f77b4",
    "pot-exact-gpu":        "#9467bd",
    "torchgw-precomputed":  "#d62728",
}


def _load(stage: str, results_dir: pathlib.Path) -> list[dict]:
    out = []
    for p in sorted(results_dir.glob(f"core_07*{stage}*.json")):
        d = json.load(open(p))
        if d.get("status") != "ok":
            continue
        out.append(d)
    return out


def _aggregate(records: list[dict], metric_path: tuple[str, ...]):
    """Return dict[(solver, n_per_cell)] = (mean, std) over seeds."""
    bins: dict = {}
    for r in records:
        v = r
        for k in metric_path:
            v = v.get(k) if isinstance(v, dict) else None
        if v is None:
            continue
        bins.setdefault((r["solver"], r["n_per_cell"]), []).append(float(v))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in bins.items()}


def _plot_metric(records, metric_path, ylabel, title, out_path, log_y=False):
    agg = _aggregate(records, metric_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    n_values = sorted({n for (_, n) in agg.keys()})
    for solver in SOLVER_ORDER:
        xs, ys, errs = [], [], []
        for n in n_values:
            if (solver, n) not in agg: continue
            m, s = agg[(solver, n)]
            xs.append(n); ys.append(m); errs.append(s)
        if xs:
            ax.errorbar(xs, ys, yerr=errs, marker="o",
                        label=solver, color=SOLVER_COLOR[solver])
    ax.set_xscale("log")
    if log_y: ax.set_yscale("log")
    ax.set_xlabel("N_per_cell")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B"])
    args = ap.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[2]
    rdir = repo / "results" / "c7_cell_morphology"
    figdir = repo / "docs" / "figures"; figdir.mkdir(exist_ok=True, parents=True)
    stage = f"stage_{args.stage.lower()}"

    recs = _load(stage, rdir)
    if not recs:
        raise SystemExit(f"no records for {stage} in {rdir}")

    _plot_metric(recs, ("metrics", "ARI_ward"),
                 "ARI (Ward, vs ground truth)",
                 f"C7 {stage} — clustering quality vs sample size",
                 figdir / f"c7_{stage}_ari.png")
    _plot_metric(recs, ("metrics", "knn_acc_k5"),
                 "kNN accuracy (LOO, k=5)",
                 f"C7 {stage} — kNN type recovery vs sample size",
                 figdir / f"c7_{stage}_knn.png")
    _plot_metric(recs, ("efficiency", "wall_full_matrix_s"),
                 "Full-matrix wall (s)",
                 f"C7 {stage} — full N×N GW wall vs sample size",
                 figdir / f"c7_{stage}_wall.png", log_y=True)
    _plot_metric(recs, ("efficiency", "wall_per_pair_ms"),
                 "Per-pair wall (ms)",
                 f"C7 {stage} — per-pair GW latency vs sample size",
                 figdir / f"c7_{stage}_per_pair.png", log_y=True)

    # UMAP figure for the highest-N seed-0 of each solver
    from umap import UMAP
    n_top = max({r["n_per_cell"] for r in recs})
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, solver in zip(axes, SOLVER_ORDER):
        candidates = [r for r in recs if r["solver"] == solver
                      and r["n_per_cell"] == n_top and r["seed"] == 0]
        if not candidates:
            ax.set_title(f"{solver}\n(no data)"); ax.axis("off"); continue
        npy = pathlib.Path(rdir / (
            f"core_07_cell_morphology__{solver}__{stage}"
            f"__n{n_top}__seed0.npy"
        ))
        if not npy.exists():
            ax.set_title(f"{solver}\n(matrix not saved)"); ax.axis("off"); continue
        D = np.load(npy)
        emb = UMAP(metric="precomputed", random_state=0,
                   n_neighbors=min(15, D.shape[0] - 1)).fit_transform(D)
        labels = candidates[0]["classes"]
        # rebuild y by reading the manifest:
        manifest = repo / "tracks" / "core" / "07_cell_morphology" / f"{stage}_manifest.txt"
        cls_to_int = {c: i for i, c in enumerate(labels)}
        y = []
        for line in open(manifest):
            line = line.strip()
            if not line or line.startswith("neuron_name") or line.startswith("specimen_id"):
                continue
            cls = line.split("\t")[1]
            y.append(cls_to_int[cls])
        y = np.asarray(y[:emb.shape[0]])
        ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=8, alpha=0.8)
        ax.set_title(f"{solver}, N={n_top}"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"C7 {stage} — UMAP of GW distance matrices")
    fig.tight_layout()
    fig.savefig(figdir / f"c7_{stage}_umap.png", dpi=150)
    print(f"wrote {figdir / f'c7_{stage}_umap.png'}")


if __name__ == "__main__":
    main()
