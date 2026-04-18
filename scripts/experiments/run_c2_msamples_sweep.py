#!/usr/bin/env python
"""C2 M_samples sweep — how does torchgw-precomputed's per-iter cost-
matrix sample count affect FOSCTTM at various N? Uses cisTopic
preprocessing (cached). Fixed ε=5e-3, max_iter=300. Writes one JSON
per cell, plus _summary.json.

Also runs pot-entropic-gpu once per (N, seed) as a baseline line.
"""
from __future__ import annotations
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tracks" / "core" / "02_single_cell_omics"))
import run  # type: ignore[import-not-found]

CACHE = REPO / "data" / "core_02_sc_omics" / "embeddings_n_comps50_atac_cistopic.npz"
OUT = REPO / "results" / "c2_msamples"
OUT.mkdir(parents=True, exist_ok=True)

N_LIST = [2000, 5000]
M_LIST = [80, 160, 320, 640, 1280, 2560, 5000]
SEEDS = [0, 1, 2]


def cell_path(solver, n, seed, tag):
    return OUT / f"c2_ms__{solver}__n{n}__seed{seed}__{tag}.json"


def run_torchgw(V_rna, V_atac, n, seed, M, eps=5e-3):
    t0 = time.perf_counter()
    out = run.run_torchgw_precomputed(
        V_rna, V_atac, seed=seed, epsilon=eps,
        M_samples=M, max_iter=300,
    )
    wall = time.perf_counter() - t0
    return out["T"], wall, out["gpu_peak_gb"]


def run_pot(V_rna, V_atac, n, seed, eps=5e-3):
    t0 = time.perf_counter()
    out = run.run_pot_entropic_gpu(V_rna, V_atac, seed=seed, epsilon=eps,
                                      max_iter=100)
    wall = time.perf_counter() - t0
    return out["T"], wall, out["gpu_peak_gb"]


def main():
    print(f"[c2-ms] loading cached cisTopic embedding from {CACHE.name}",
           flush=True)
    z = np.load(CACHE)
    V_rna_full, V_atac_full = z["V_rna"], z["V_atac"]

    total = len(SEEDS) * len(N_LIST) * (len(M_LIST) + 1)
    k = 0
    for seed in SEEDS:
        for n in N_LIST:
            Vr, Va, _ = run.subsample_cells(V_rna_full, V_atac_full, n, seed)
            Vr = run.l2_normalize(Vr)
            Va = run.l2_normalize(Va)

            for M in M_LIST:
                k += 1
                tag = f"M{M}"
                p = cell_path("torchgw-precomputed", n, seed, tag)
                if p.exists():
                    print(f"[c2-ms] {k}/{total} cached {p.name}", flush=True)
                    continue
                T, wall, gpu = run_torchgw(Vr, Va, n, seed, M)
                f = run.foscttm(T, Vr, Va)
                rec = {
                    "solver": "torchgw-precomputed",
                    "n": n, "seed": seed, "M": M,
                    "foscttm": float(f),
                    "wall_s": float(wall),
                    "gpu_peak_gb": gpu,
                }
                p.write_text(json.dumps(rec, indent=2))
                print(f"[c2-ms] {k}/{total} n={n} M={M} seed={seed}  "
                       f"FOSCTTM={f:.3f}  wall={wall:.1f}s", flush=True)

            # POT baseline for this (n, seed)
            k += 1
            p = cell_path("pot-entropic-gpu", n, seed, "baseline")
            if p.exists():
                print(f"[c2-ms] {k}/{total} cached {p.name}", flush=True)
                continue
            T, wall, gpu = run_pot(Vr, Va, n, seed)
            f = run.foscttm(T, Vr, Va)
            rec = {
                "solver": "pot-entropic-gpu",
                "n": n, "seed": seed, "M": None,
                "foscttm": float(f),
                "wall_s": float(wall),
                "gpu_peak_gb": gpu,
            }
            p.write_text(json.dumps(rec, indent=2))
            print(f"[c2-ms] {k}/{total} n={n} POT baseline seed={seed}  "
                   f"FOSCTTM={f:.3f}  wall={wall:.1f}s", flush=True)

    # Summary
    agg = defaultdict(list)
    for p in OUT.glob("c2_ms__*.json"):
        d = json.loads(p.read_text())
        agg[(d["solver"], d.get("n"), d.get("M"))].append(d)
    summary = []
    for (s, n, m), rows in agg.items():
        arr = np.asarray([(r["foscttm"], r["wall_s"]) for r in rows])
        summary.append({
            "solver": s, "n": n, "M": m,
            "foscttm_mean": float(arr[:, 0].mean()),
            "foscttm_std":  float(arr[:, 0].std()),
            "wall_s_mean":  float(arr[:, 1].mean()),
            "n_seeds":      len(rows),
        })
    (OUT / "_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[c2-ms] done. summary written.", flush=True)


if __name__ == "__main__":
    main()
