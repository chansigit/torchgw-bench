#!/usr/bin/env python
"""Neighborhood-preservation metrics for C6 — the GW-native way of
evaluating shape correspondence (no GT needed).

For each (src, tgt) pair, build kNN-geodesic D_src and D_tgt, run
torchgw-dijkstra and pot-exact-gpu, then report:

  1. Pair distortion: for K randomly sampled source pairs (i,j),
     |D_src(i,j) - D_tgt(argmax T[i], argmax T[j])| / max(D_src.max(), D_tgt.max())
     - Mean and median over the sample
  2. k-NN preservation: for each source vertex i, take its k nearest
     source neighbors S_i. Map every vertex via argmax T. Compute
     |argmax(S_i) ∩ kNN_target(argmax(T[i]))| / k.
     - Mean over source vertices

Lower distortion = better. Higher kNN preservation = better.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tracks" / "core" / "06_shape_correspondence"))
import run  # type: ignore[import-not-found]

PAIRS = [("cat0", "cat1"), ("horse0", "horse5"), ("david0", "david1")]
N = 2000
SEEDS = [0, 1, 2]
K_NN = 10
N_PAIR_SAMPLES = 5000


def pair_distortion(T, D_src, D_tgt, n_samples=N_PAIR_SAMPLES, seed=0):
    """Sample n_samples (i,j) source pairs; compute |D_src(i,j) -
    D_tgt(T[i], T[j])| (in the same units, both normalised to [0,1] by
    diameter)."""
    rng = np.random.default_rng(seed)
    n = D_src.shape[0]
    pred = T.argmax(axis=1)
    diam_src = D_src.max(); diam_tgt = D_tgt.max()
    Dn_src = D_src / diam_src
    Dn_tgt = D_tgt / diam_tgt
    i = rng.integers(0, n, n_samples)
    j = rng.integers(0, n, n_samples)
    mask = i != j
    i, j = i[mask], j[mask]
    d_s = Dn_src[i, j]
    d_t = Dn_tgt[pred[i], pred[j]]
    err = np.abs(d_s - d_t)
    return float(err.mean()), float(np.median(err))


def knn_preservation(T, V_src, V_tgt, k=K_NN):
    """For each src vertex i, find its k nearest source neighbors. Map
    all to target via argmax T. Check overlap with kNN_target(T[i])."""
    pred = T.argmax(axis=1)
    nn_src = NearestNeighbors(n_neighbors=k + 1).fit(V_src)
    _, src_neighbors = nn_src.kneighbors(V_src)
    src_neighbors = src_neighbors[:, 1:]  # drop self
    nn_tgt = NearestNeighbors(n_neighbors=k + 1).fit(V_tgt)
    _, tgt_neighbors = nn_tgt.kneighbors(V_tgt)
    tgt_neighbors = tgt_neighbors[:, 1:]
    # For each source vertex i, mapped neighbors are pred[src_neighbors[i]]
    mapped = pred[src_neighbors]                # [n_src, k]
    expected = tgt_neighbors[pred]              # [n_src, k]
    # Overlap fraction per row
    overlaps = []
    for m_row, e_row in zip(mapped, expected):
        overlaps.append(len(set(m_row.tolist()) & set(e_row.tolist())) / k)
    return float(np.mean(overlaps))


def main():
    rows = {"torchgw-dijkstra": [], "pot-exact-gpu": []}
    for pair in PAIRS:
        for seed in SEEDS:
            V_src_full, _, V_tgt_full, _, gt_full = run.load_taco_pair(*pair)
            V_src, V_tgt, _ = run.subsample_pair(V_src_full, V_tgt_full,
                                                    gt_full, N, N, seed=seed)
            D_src = run.knn_geodesic_matrix(V_src)
            D_tgt = run.knn_geodesic_matrix(V_tgt)
            out_tg = run.run_torchgw_dijkstra(V_src, V_tgt, seed=seed, max_iter=300)
            out_pot = run.run_pot_exact_gpu(V_src, V_tgt, seed=seed, max_iter=500)
            for name, out in [("torchgw-dijkstra", out_tg), ("pot-exact-gpu", out_pot)]:
                T = out["T"]
                pd_mean, pd_med = pair_distortion(T, D_src, D_tgt, seed=seed)
                knn = knn_preservation(T, V_src, V_tgt, k=K_NN)
                rows[name].append({
                    "pair_distortion_mean": pd_mean,
                    "pair_distortion_median": pd_med,
                    "knn10_preservation": knn,
                })
                print(f"  {pair[0]:>9}→{pair[1]:<9} seed={seed} {name:<18} "
                       f"distort_mean={pd_mean:.3f} distort_med={pd_med:.3f} "
                       f"knn10_pres={knn:.3f}")

    print()
    print(f"{'metric':<25} {'torchgw-dij':>14} {'pot-exact':>14}")
    for k in ["pair_distortion_mean", "pair_distortion_median", "knn10_preservation"]:
        tg = np.mean([r[k] for r in rows["torchgw-dijkstra"]])
        pt = np.mean([r[k] for r in rows["pot-exact-gpu"]])
        print(f"{k:<25} {tg:>14.4f} {pt:>14.4f}")


if __name__ == "__main__":
    main()
