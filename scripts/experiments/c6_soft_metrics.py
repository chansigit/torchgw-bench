#!/usr/bin/env python
"""Probe whether the argmax-based metric unfairly penalises diffuse
transport plans. For each of the 3 representative pairs, rerun
torchgw-dijkstra and pot-exact-gpu and report:

  1. argmax mean_err_normalised         (current metric)
  2. top-k hit rate at k=1,5,10,50      (is GT among top-k of T[i]?)
  3. transport-weighted accuracy@τ:     (Σ_j T[i,j] 1[D_tgt(j, gt_i) ≤ τ])
  4. soft geodesic error:               Σ_j T[i,j] D_tgt(j, gt_i)

If torchgw is competitive under (2)/(3)/(4) while losing under (1),
the argmax metric was unfair. If it still loses → algorithmic gap.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tracks" / "core" / "06_shape_correspondence"))
import run  # type: ignore[import-not-found]

PAIRS = [("cat0", "cat1"), ("horse0", "horse5"), ("david0", "david1")]
N = 2000
SEEDS = [0, 1, 2]
TAUS = (0.05, 0.10, 0.25)


def metrics(T: np.ndarray, D_tgt: np.ndarray, gt: np.ndarray):
    diam = D_tgt.max()
    pred = T.argmax(axis=1)
    err_norm = D_tgt[pred, gt] / diam

    # top-k hit rates
    sorted_idx = np.argsort(-T, axis=1)
    topk = {}
    for k in (1, 5, 10, 50):
        topk[k] = float((sorted_idx[:, :k] == gt[:, None]).any(axis=1).mean())

    # transport-weighted accuracy
    # normalize each row of T to sum to 1
    Tn = T / (T.sum(axis=1, keepdims=True) + 1e-30)
    D_norm = D_tgt[:, gt].T / diam  # [n_src, n_tgt]; D_norm[i, j] = norm geo dist tgt[j] → tgt[gt[i]]
    soft_err_n = float((Tn * D_norm).sum(axis=1).mean())
    soft_acc = {}
    for tau in TAUS:
        hit = (D_norm <= tau).astype(np.float32)   # [n_src, n_tgt]
        soft_acc[tau] = float((Tn * hit).sum(axis=1).mean())

    return {
        "argmax_err": float(err_norm.mean()),
        "top1": topk[1], "top5": topk[5], "top10": topk[10], "top50": topk[50],
        "soft_err": soft_err_n,
        "soft_acc_005": soft_acc[0.05],
        "soft_acc_010": soft_acc[0.10],
        "soft_acc_025": soft_acc[0.25],
    }


def main():
    cache = {}
    agg = {"torchgw-dijkstra": [], "pot-exact-gpu": []}
    for pair in PAIRS:
        for seed in SEEDS:
            key = (pair, seed)
            V_src_full, _, V_tgt_full, _, gt_full = run.load_taco_pair(*pair)
            V_src, V_tgt, gt = run.subsample_pair(
                V_src_full, V_tgt_full, gt_full, N, N, seed=seed,
            )
            D_tgt = run.knn_geodesic_matrix(V_tgt)
            cache[key] = (V_src, V_tgt, gt, D_tgt)
            out_tg = run.run_torchgw_dijkstra(V_src, V_tgt, seed=seed, max_iter=300)
            out_pot = run.run_pot_exact_gpu(V_src, V_tgt, seed=seed, max_iter=500)
            m_tg = metrics(out_tg["T"], D_tgt, gt)
            m_pot = metrics(out_pot["T"], D_tgt, gt)
            agg["torchgw-dijkstra"].append(m_tg)
            agg["pot-exact-gpu"].append(m_pot)
            print(f"  {pair[0]:>9}→{pair[1]:<9} seed={seed}  "
                   f"tg argmax={m_tg['argmax_err']:.3f} top10={m_tg['top10']:.2f} "
                   f"soft_acc@0.05={m_tg['soft_acc_005']:.3f}  "
                   f"pot argmax={m_pot['argmax_err']:.3f} top10={m_pot['top10']:.2f} "
                   f"soft_acc@0.05={m_pot['soft_acc_005']:.3f}")

    print()
    print(f"{'metric':<20} {'torchgw-dij':>14} {'pot-exact':>14}")
    keys = ["argmax_err", "top1", "top5", "top10", "top50",
             "soft_err", "soft_acc_005", "soft_acc_010", "soft_acc_025"]
    for k in keys:
        tg = np.mean([r[k] for r in agg["torchgw-dijkstra"]])
        pt = np.mean([r[k] for r in agg["pot-exact-gpu"]])
        print(f"{k:<20} {tg:>14.4f} {pt:>14.4f}")


if __name__ == "__main__":
    main()
