#!/usr/bin/env python
"""C5 M_samples sweep — torchgw-precomputed at pair=en-es, N=5000.

Sweeps M ∈ {80, 320, 1280, 3750, 5000} across seeds 0,1,2.
Plus one pot-entropic-gpu baseline row per seed (no M).
Fixed ε=5e-4 (C5 operating point; NOT 5e-5 which doesn't converge in POT 0.9.6).
Writes per-cell JSONs + _summary.json under results/c5_msamples/.
"""
from __future__ import annotations
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tracks" / "core" / "05_word_embedding"))
import run  # type: ignore[import-not-found]

DATA_ROOT = REPO / "data" / "core_05_word_embedding"
OUT = REPO / "results" / "c5_msamples"
OUT.mkdir(parents=True, exist_ok=True)

PAIR = "en-es"
N = 5000
SEEDS = [0, 1, 2]
M_LIST = [80, 320, 1280, 3750, 5000]
# ε=5e-4: C5 operating point (NOT the paper's 5e-5 which fails in POT 0.9.6)
EPS = 5e-4


def cell_path(solver: str, seed: int, tag: str) -> Path:
    return OUT / f"c5_ms__{solver}__n{N}__seed{seed}__{tag}.json"


def load_vectors():
    """Load and return (words_src, V_src, words_tgt, V_tgt) for PAIR at N words."""
    lang_src, lang_tgt = PAIR.split("-")
    vec_src = DATA_ROOT / "vectors" / f"wiki.{lang_src}.vec"
    vec_tgt = DATA_ROOT / "vectors" / f"wiki.{lang_tgt}.vec"
    print(f"[c5-ms] loading {vec_src.name} N={N}", flush=True)
    words_src, V_src = run._io.read_fasttext(str(vec_src), N)
    print(f"[c5-ms] loading {vec_tgt.name} N={N}", flush=True)
    words_tgt, V_tgt = run._io.read_fasttext(str(vec_tgt), N)
    print(f"[c5-ms] V_src={V_src.shape}  V_tgt={V_tgt.shape}", flush=True)
    return words_src, V_src, words_tgt, V_tgt


def compute_p1_csls(T: np.ndarray, V_src: np.ndarray, V_tgt: np.ndarray,
                    words_src: list, words_tgt: list, gold: dict) -> float:
    proj = run._eval.barycentric_project(T, V_tgt)
    scores = run._eval.precision_at_k_csls(proj, V_tgt, words_src, words_tgt, gold, ks=(1,))
    return scores[1]


def main():
    words_src, V_src, words_tgt, V_tgt = load_vectors()

    # Build cost matrices (same as run.py main)
    print("[c5-ms] building cosine cost matrices", flush=True)
    C_src, C_tgt = run.build_cost_matrices(V_src, V_tgt)

    # Load train dict (0-5000).  The test dict (5000-6500) has zero coverage
    # in our N=5000 vocabulary (only top-5000 words loaded), so we always
    # evaluate against the train dict here — consistent with run.py's fix.
    dict_train = DATA_ROOT / "dicts" / f"{PAIR}.0-5000.txt"
    gold = run._io.read_muse_dict(str(dict_train))
    print(f"[c5-ms] train dict has {len(gold)} entries", flush=True)

    total = len(SEEDS) * (len(M_LIST) + 1)
    k = 0
    for seed in SEEDS:
        # ---- torchgw-precomputed sweep over M ----------------------------
        for M in M_LIST:
            k += 1
            tag = f"M{M}"
            p = cell_path("torchgw-precomputed", seed, tag)
            if p.exists():
                print(f"[c5-ms] {k}/{total} cached {p.name}", flush=True)
                continue

            t0 = time.perf_counter()
            result = run.run_torchgw_precomputed(
                V_src, V_tgt, seed=seed, epsilon=EPS,
                M_samples=M, max_iter=300,
                C_src=C_src, C_tgt=C_tgt,
            )
            wall = time.perf_counter() - t0
            p1 = compute_p1_csls(result["T"], V_src, V_tgt, words_src, words_tgt, gold)
            rec = {
                "solver": "torchgw-precomputed",
                "pair": PAIR, "n": N, "seed": seed, "M": M,
                "p1_csls": float(p1),
                "wall_s": float(wall),
                "gpu_peak_gb": result.get("gpu_peak_gb"),
            }
            p.write_text(json.dumps(rec, indent=2))
            print(f"[c5-ms] {k}/{total} M={M} seed={seed}  "
                  f"P@1-CSLS={p1:.4f}  wall={wall:.1f}s", flush=True)

        # ---- pot-entropic-gpu baseline -----------------------------------
        k += 1
        p = cell_path("pot-entropic-gpu", seed, "baseline")
        if p.exists():
            print(f"[c5-ms] {k}/{total} cached {p.name}", flush=True)
            continue

        t0 = time.perf_counter()
        result = run.run_pot_entropic_gpu(C_src, C_tgt, seed=seed, epsilon=EPS, max_iter=100)
        wall = time.perf_counter() - t0
        p1 = compute_p1_csls(result["T"], V_src, V_tgt, words_src, words_tgt, gold)
        rec = {
            "solver": "pot-entropic-gpu",
            "pair": PAIR, "n": N, "seed": seed, "M": None,
            "p1_csls": float(p1),
            "wall_s": float(wall),
            "gpu_peak_gb": result.get("gpu_peak_gb"),
        }
        p.write_text(json.dumps(rec, indent=2))
        print(f"[c5-ms] {k}/{total} POT baseline seed={seed}  "
              f"P@1-CSLS={p1:.4f}  wall={wall:.1f}s", flush=True)

    # ---- summary ---------------------------------------------------------
    agg = defaultdict(list)
    for fp in OUT.glob("c5_ms__*.json"):
        d = json.loads(fp.read_text())
        agg[(d["solver"], d.get("n"), d.get("M"))].append(d)

    summary = []
    for (s, n, m), rows in agg.items():
        arr = np.asarray([(r["p1_csls"], r["wall_s"]) for r in rows])
        summary.append({
            "solver": s, "n": n, "M": m,
            "p1_csls_mean": float(arr[:, 0].mean()),
            "p1_csls_std":  float(arr[:, 0].std()),
            "wall_s_mean":  float(arr[:, 1].mean()),
            "n_seeds":      len(rows),
        })

    (OUT / "_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[c5-ms] done. summary written to", OUT / "_summary.json", flush=True)


if __name__ == "__main__":
    main()
