#!/usr/bin/env python
"""C7 cell-morphology benchmark — one (stage, solver, N_per_cell, seed) cell."""
from __future__ import annotations
import argparse
import datetime as _dt
import json
import os
import pathlib
import socket
import sys
import time
import numpy as np

TRACK = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TRACK))

import swc_io  # noqa: F401  (kept importable for symmetry with other tracks)
import intracell
import eval as track_eval
import solvers


def _read_manifest(stage: str) -> tuple[list[pathlib.Path], np.ndarray, list[str]]:
    repo = TRACK.parents[2]
    manifest = TRACK / f"{stage}_manifest.txt"
    swc_dir = repo / "data" / "core_07_cell_morphology" / "swc" / stage
    paths: list[pathlib.Path] = []
    lbls: list[str] = []
    with open(manifest) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("neuron_name") or line.startswith("specimen_id"):
                continue
            parts = line.split("\t")
            name, cls = parts[0], parts[1]  # 3rd column (archive) is fetch.sh-only
            p = swc_dir / f"{name}.swc"
            if p.exists():
                paths.append(p); lbls.append(cls)
    classes_sorted = sorted(set(lbls))
    cls_to_int = {c: i for i, c in enumerate(classes_sorted)}
    y = np.asarray([cls_to_int[c] for c in lbls], dtype=np.int64)
    return paths, y, classes_sorted


def _peak_rss_gb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**30
    except ImportError:
        return float("nan")


def _gw_full_matrix(solver: str, D_list, *, epsilon, M_samples, seed):
    """Build the N×N pairwise GW matrix.

    `cajal-native` calls CAJAL's parallel batch routine to measure its
    multiprocessing-augmented end-to-end speed. The other three solvers
    have no batched-small-GW API and loop pairs serially — apples-to-
    oranges on full-matrix wall, apples-to-apples on per-pair wall.
    """
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    n_cells = len(D_list)
    n_pairs = n_cells * (n_cells - 1) // 2
    pair_walls: list[float] = []

    if solver == "cajal-native":
        t0 = time.perf_counter()
        M = solvers.gw_full_matrix_cajal(D_list)
        wall_total = time.perf_counter() - t0
        # CAJAL returns the full matrix; impute per-pair as wall_total/n_pairs.
        pair_walls = [wall_total / max(n_pairs, 1)] * n_pairs
    else:
        M = np.zeros((n_cells, n_cells), dtype=np.float64)
        t0 = time.perf_counter()
        for i in range(n_cells):
            for j in range(i + 1, n_cells):
                out = solvers.gw_pair(solver, D_list[i], D_list[j],
                                      epsilon=epsilon, M_samples=M_samples, seed=seed)
                M[i, j] = M[j, i] = out["gw"]
                pair_walls.append(out["wall_s"])
        wall_total = time.perf_counter() - t0

    gpu_peak = (torch.cuda.max_memory_allocated() / 2**30
                if torch.cuda.is_available() else None)
    return M, {
        "wall_full_matrix_s": float(wall_total),
        "wall_per_pair_ms":   float(np.mean(pair_walls) * 1000),
        "gpu_peak_gb":        gpu_peak,
        "cpu_peak_gb":        _peak_rss_gb(),
        "n_pairs":            n_pairs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B"])
    ap.add_argument("--solver", required=True, choices=[
        "cajal-native", "pot-entropic-gpu", "pot-exact-gpu", "torchgw-precomputed",
    ])
    ap.add_argument("--n-per-cell", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epsilon", type=float, default=5e-3)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    if args.solver == "pot-exact-gpu" and args.n_per_cell > 200:
        print(f"[c7] skip pot-exact-gpu at n_per_cell={args.n_per_cell} > 200")
        return
    if args.solver == "torchgw-precomputed" and args.n_per_cell > 200:
        # smoke showed ~0.45 s/pair at N=50 → ~10 s/pair at N=1000;
        # serial 45 000-pair loop = 5+ days per (seed). Skip per spec §10.
        print(f"[c7] skip torchgw-precomputed at n_per_cell={args.n_per_cell} > 200 "
              f"(serial-loop infeasible for many-tiny-GW; see spec §10 caveat 1)")
        return

    stage = f"stage_{args.stage.lower()}"
    paths, y, classes = _read_manifest(stage)
    cache_dir = (TRACK.parents[2] / "results" / "c7_cell_morphology"
                 / "_intracell_cache" / stage)

    rec = {
        "track": "core/07_cell_morphology",
        "stage": stage, "solver": args.solver,
        "n_per_cell": args.n_per_cell, "seed": args.seed,
        "epsilon": args.epsilon,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "n_cells": len(paths), "classes": classes,
        "status": "ok", "error": None,
        "metrics": {}, "efficiency": {},
    }
    out_file = args.out / (
        f"core_07_cell_morphology__{args.solver}__{stage}"
        f"__n{args.n_per_cell}__seed{args.seed}.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        D_list = [intracell.compute_intracell(p, args.n_per_cell, cache_dir)
                  for p in paths]
        M, eff = _gw_full_matrix(args.solver, D_list,
                                 epsilon=args.epsilon, M_samples=None, seed=args.seed)
        ev = track_eval.eval_distance_matrix(M, y, k_classes=len(classes), knn_k=5)
        rec["metrics"]    = ev
        rec["efficiency"] = eff
        # save the full distance matrix only for seed 0 (UMAP figures use this)
        if args.seed == 0:
            np.save(out_file.with_suffix(".npy"), M)
    except Exception as e:
        rec["status"] = "fail"; rec["error"] = f"{type(e).__name__}: {e}"

    with open(out_file, "w") as fh:
        json.dump(rec, fh, indent=2, default=str)
    print(f"[c7] wrote {out_file} (status={rec['status']})")


if __name__ == "__main__":
    main()
