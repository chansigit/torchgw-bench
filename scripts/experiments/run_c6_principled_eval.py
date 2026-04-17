#!/usr/bin/env python
"""C6 principled evaluation: mean/median normalised geodesic error
(supervised, task-aligned) + pair distortion (unsupervised, GW-native
cross-check). No 'accuracy', no top-k retrieval. One evaluation pass
across all 18 v1 benchmark pairs × 3 seeds × 5 solvers.

Writes:
    results/c6_principled/<record>.json
    results/c6_principled/_summary.json  (aggregate table)
"""
from __future__ import annotations
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tracks" / "core" / "06_shape_correspondence"))
import run  # type: ignore[import-not-found]

DATA = REPO / "data" / "core_06_shape" / "taco"
OUT_DIR = REPO / "results" / "c6_principled"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N = 2000
SEEDS = [0, 1, 2]
N_PAIR_SAMPLES = 5000


def first_two_pairs_per_class() -> list[tuple[str, str]]:
    lines = [l.strip() for l in (DATA / "pairs.txt").read_text().splitlines() if l.strip()]
    by_class: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for p in lines:
        a, b = p.split(",")
        cls = re.match(r"[a-z]+", a).group()  # type: ignore[union-attr]
        by_class[cls].append((a, b))
    return [p for cls in sorted(by_class) for p in by_class[cls][:2]]


def pair_distortion(T, D_src, D_tgt, seed=0, n_samples=N_PAIR_SAMPLES):
    rng = np.random.default_rng(seed)
    n = D_src.shape[0]
    pred = T.argmax(axis=1)
    Dn_src = D_src / D_src.max()
    Dn_tgt = D_tgt / D_tgt.max()
    i = rng.integers(0, n, n_samples)
    j = rng.integers(0, n, n_samples)
    mask = i != j
    i, j = i[mask], j[mask]
    err = np.abs(Dn_src[i, j] - Dn_tgt[pred[i], pred[j]])
    return float(err.mean()), float(np.median(err))


SOLVERS = [
    ("torchgw-landmark",    lambda V_s, V_t, s: run.run_torchgw_landmark(
        V_s, V_t, seed=s, epsilon=5e-2, max_iter=300)),
    ("torchgw-dijkstra",    lambda V_s, V_t, s: run.run_torchgw_dijkstra(
        V_s, V_t, seed=s, epsilon=5e-2, max_iter=300)),
    ("torchgw-precomputed", lambda V_s, V_t, s: run.run_torchgw_precomputed(
        V_s, V_t, seed=s, epsilon=5e-2, max_iter=300)),
    ("pot-entropic-gpu",    lambda V_s, V_t, s: run.run_pot_entropic_gpu(
        V_s, V_t, seed=s, max_iter=100)),
    ("pot-exact-gpu",       lambda V_s, V_t, s: run.run_pot_exact_gpu(
        V_s, V_t, seed=s, max_iter=500)),
]


def main():
    pairs = first_two_pairs_per_class()
    print(f"[c6-eval] {len(pairs)} pairs × {len(SEEDS)} seeds × {len(SOLVERS)} solvers "
           f"= {len(pairs) * len(SEEDS) * len(SOLVERS)} cells")
    records = []
    t_start = time.perf_counter()

    for i, (src, tgt) in enumerate(pairs):
        for seed in SEEDS:
            V_src_f, _, V_tgt_f, _, gt_f = run.load_taco_pair(src, tgt)
            V_src, V_tgt, gt = run.subsample_pair(
                V_src_f, V_tgt_f, gt_f, N, N, seed=seed,
            )
            D_src = run.knn_geodesic_matrix(V_src)
            D_tgt = run.knn_geodesic_matrix(V_tgt)
            diam = D_tgt.max()

            for solver_name, solver_fn in SOLVERS:
                cell_path = OUT_DIR / f"{solver_name}__{src}_{tgt}__seed{seed}.json"
                if cell_path.exists():
                    rec = json.loads(cell_path.read_text())
                    records.append(rec)
                    continue
                try:
                    out = solver_fn(V_src, V_tgt, seed)
                except Exception as e:
                    print(f"  FAIL {solver_name} {src}-{tgt} seed={seed}: {e}")
                    continue
                T = out["T"]
                pred = T.argmax(axis=1)
                err_norm = (D_tgt[pred, gt] / diam)
                pd_mean, pd_med = pair_distortion(T, D_src, D_tgt, seed=seed)
                rec = {
                    "solver": solver_name,
                    "pair": f"{src},{tgt}",
                    "seed": seed,
                    "metrics": {
                        "mean_geo_err_norm":   float(err_norm.mean()),
                        "median_geo_err_norm": float(np.median(err_norm)),
                        "pair_distortion_mean":   pd_mean,
                        "pair_distortion_median": pd_med,
                    },
                    "efficiency": {
                        "wall_s_total":    out["wall_s_total"],
                        "wall_s_solve":    out["wall_s_solve"],
                        "gpu_peak_gb":     out["gpu_peak_gb"],
                    },
                    "hyperparams": out["hyperparams"],
                }
                cell_path.write_text(json.dumps(rec, indent=2))
                records.append(rec)
                print(f"  [{i+1:2d}/{len(pairs)}] {src:>10}→{tgt:<10} seed={seed} "
                       f"{solver_name:<22}  geo_err={rec['metrics']['mean_geo_err_norm']:.3f} "
                       f"pair_dist={pd_mean:.3f}")

    elapsed = time.perf_counter() - t_start

    # Aggregate
    by_solver: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_solver[r["solver"]].append(r)

    print(f"\n[c6-eval] === SUMMARY (elapsed {elapsed:.0f}s) ===")
    print(f"{'solver':<22} {'geo_err_mean':>13} {'geo_err_med':>12} "
           f"{'pair_dist_mean':>15} {'pair_dist_med':>14}")
    summary = {}
    order = ["torchgw-landmark", "torchgw-dijkstra", "torchgw-precomputed",
              "pot-entropic-gpu", "pot-exact-gpu"]
    for s in order:
        rows = by_solver[s]
        arr = np.asarray([[
            r["metrics"]["mean_geo_err_norm"],
            r["metrics"]["median_geo_err_norm"],
            r["metrics"]["pair_distortion_mean"],
            r["metrics"]["pair_distortion_median"],
        ] for r in rows])
        summary[s] = {
            "n": len(rows),
            "geo_err_mean":      float(arr[:, 0].mean()),
            "geo_err_median":    float(arr[:, 1].mean()),
            "pair_dist_mean":    float(arr[:, 2].mean()),
            "pair_dist_median":  float(arr[:, 3].mean()),
        }
        d = summary[s]
        print(f"{s:<22} {d['geo_err_mean']:>13.4f} {d['geo_err_median']:>12.4f} "
               f"{d['pair_dist_mean']:>15.4f} {d['pair_dist_median']:>14.4f}")
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
