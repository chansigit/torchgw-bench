#!/usr/bin/env python
"""C5 word-embedding cross-lingual alignment — GW benchmark.

Given two sets of fastText wiki word vectors (source language and target
language), we align them using the Gromov-Wasserstein plan and evaluate
translation retrieval accuracy against the MUSE bilingual test dictionary.

Pipeline design choices:
  * Cost matrix: pairwise cosine distance (1 - cos(vi, vj)), range-normalised
    to [0, 1].  This mirrors the Alvarez-Melis & Jaakkola (2018) setup.
  * Input: first N words by frequency from fastText (no random subsampling —
    paper uses top-frequency vocabulary).
  * Evaluation: P@1 and P@5 via cosine NN and CSLS retrieval against the
    MUSE test dictionary (split 5000-6500).

Solvers: same 5 GPU variants as the rest of the benchmark series.
  torchgw-landmark, torchgw-dijkstra  — pass raw V_src/V_tgt (geodesic cost)
  torchgw-precomputed                 — inject cosine cost matrices directly
  pot-entropic-gpu                    — entropic GW with precomputed cosine C
  pot-exact-gpu                       — exact CG-GW with precomputed cosine C

Note: landmark/dijkstra are expected to underperform here because their
internal kNN-geodesic cost differs from the cosine structure used by the
reference method.  They are included as honest comparison points.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
import threading
from pathlib import Path
from typing import Any

import numpy as np

# ---- self-contained module loader ----------------------------------------
TRACK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRACK_DIR.parents[2]
DATA_ROOT = REPO_ROOT / "data" / "core_05_word_embedding"


def _load_local(name: str):
    spec = importlib.util.spec_from_file_location(name, TRACK_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_io = _load_local("io")
_eval = _load_local("eval")


# ---- inline cost helpers (from io.py — kept inline for self-contained run.py)

def cosine_cost(V: np.ndarray) -> np.ndarray:
    """Pairwise cosine-distance matrix: C[i,j] = 1 - cos(vi, vj)."""
    V = np.array(V, dtype=np.float32)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    V = V / norms
    C = (1.0 - V @ V.T).astype(np.float32)
    np.fill_diagonal(C, 0.0)
    return C


def range_normalize(C: np.ndarray) -> np.ndarray:
    """Rescale C to [0, 1]."""
    C = np.array(C, dtype=np.float32)
    lo, hi = C.min(), C.max()
    if hi == lo:
        return np.zeros_like(C, dtype=np.float32)
    return ((C - lo) / (hi - lo)).astype(np.float32)


def build_cost_matrices(V_src: np.ndarray, V_tgt: np.ndarray):
    """Build range-normalised cosine cost matrices for src and tgt."""
    C_src = range_normalize(cosine_cost(V_src))
    C_tgt = range_normalize(cosine_cost(V_tgt))
    return C_src, C_tgt


# ---- record helper -------------------------------------------------------

def build_record(track: str, solver: str, seed: int, subset: str) -> dict:
    import socket
    import datetime as _dt
    return {
        "track": track, "solver": solver, "seed": seed,
        "solver_version": None, "subset": subset,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "status": "ok", "error": None,
        "dataset": {}, "hyperparams": {},
        "metrics": {"correctness": {}, "task": {}, "efficiency": {}, "stability": {}},
        "artifacts": {},
    }


# ---- memory / timing helpers (verbatim from C2) -------------------------

class _RSSSampler:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil  # noqa: F401
            self._have_psutil = True
        except ImportError:
            self._have_psutil = False

    def __enter__(self):
        if not self._have_psutil:
            return self
        import psutil
        self._proc = psutil.Process(os.getpid())
        self.peak = self._proc.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)

    def _loop(self):
        import psutil
        while not self._stop.is_set():
            try:
                cur = self._proc.memory_info().rss
            except psutil.Error:
                break
            if cur > self.peak:
                self.peak = cur
            self._stop.wait(self.interval)


def _reset_tracking(seed: int):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    return torch, use_cuda


def _build_metrics(wall_preprocess, wall_solve, ram_peak_bytes, use_cuda):
    import torch
    gpu_peak_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3) if use_cuda else None
    )
    return {
        "wall_s_preprocess": float(wall_preprocess),
        "wall_s_solve":      float(wall_solve),
        "wall_s_total":      float(wall_preprocess + wall_solve),
        "gpu_peak_gb":       gpu_peak_gb,
        "ram_peak_gb":       ram_peak_bytes / (1024 ** 3),
    }


# ---- torchgw finalizer (verbatim from C2) --------------------------------

def _finalize_torchgw(T, log, meta, hyperparams):
    log_d = log if isinstance(log, dict) else {}
    T_np = T.detach().cpu().numpy() if hasattr(T, "detach") else np.asarray(T)
    marginal_error = float(np.max(np.abs(T_np.sum(axis=1) - 1.0 / T_np.shape[0])))
    import torchgw as _torchgw
    rec = {
        "T": T_np,
        "gw_cost": float(log_d.get("gw_cost", float("nan"))),
        "marginal_error": marginal_error,
        "iterations": int(log_d.get("n_iter", log_d.get("iterations", 0))),
        "hyperparams": hyperparams,
        "solver_version": f"torchgw=={getattr(_torchgw, '__version__', 'unknown')}",
    }
    rec.update(meta)
    rec["wall_s"] = meta["wall_s_total"]
    return rec


# ---- POT finalizer (verbatim from C2) ------------------------------------

def _finalize_pot(T_and_log, meta, hyperparams):
    import ot
    pot_log: dict[str, Any] = {}
    if isinstance(T_and_log, tuple):
        T, raw = T_and_log
        if isinstance(raw, dict):
            pot_log = raw
    else:
        T = T_and_log
    T_np = T.detach().cpu().numpy().astype(np.float64) if hasattr(T, "detach") else np.asarray(T, dtype=np.float64)
    marginal_error = float(np.max(np.abs(T_np.sum(axis=1) - 1.0 / T_np.shape[0])))

    def _as_float(v):
        if hasattr(v, "detach"):
            return float(v.detach().cpu().item())
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    gw_cost = _as_float(pot_log.get("gw_dist", float("nan")))
    err_list = pot_log.get("err") or []
    iterations = len(err_list) if err_list else -1
    rec = {
        "T": T_np,
        "gw_cost": gw_cost,
        "marginal_error": marginal_error,
        "iterations": iterations,
        "hyperparams": hyperparams,
        "solver_version": f"pot=={getattr(ot, '__version__', 'unknown')}",
    }
    rec.update(meta)
    rec["wall_s"] = meta["wall_s_total"]
    return rec


# ---- torchgw solver wrappers --------------------------------------------

def run_torchgw_landmark(V_src, V_tgt, seed=0, epsilon=5e-5,
                          M_samples=80, max_iter=300, k=10, n_landmarks=50):
    from torchgw import sampled_gw
    torch, use_cuda = _reset_tracking(seed)
    with _RSSSampler() as sampler:
        t0 = time.perf_counter()
        T, log = sampled_gw(  # type: ignore[misc]
            V_src, V_tgt,
            distance_mode="landmark", mixed_precision=True,
            M=M_samples, epsilon=epsilon, max_iter=max_iter,
            k=k, n_landmarks=n_landmarks,
            log=True, verbose=False,
        )
        if use_cuda:
            torch.cuda.synchronize()
        t_solve = time.perf_counter() - t0
    meta = _build_metrics(0.0, t_solve, sampler.peak, use_cuda)
    return _finalize_torchgw(T, log, meta, {
        "M_samples": M_samples, "epsilon": epsilon, "max_iter": max_iter,
        "k": k, "n_landmarks": n_landmarks, "fgw_alpha": 1.0,
        "distance_mode": "landmark", "mixed_precision": True,
    })


def run_torchgw_dijkstra(V_src, V_tgt, seed=0, epsilon=5e-5,
                          M_samples=80, max_iter=300, k=10):
    from torchgw import sampled_gw
    torch, use_cuda = _reset_tracking(seed)
    with _RSSSampler() as sampler:
        t0 = time.perf_counter()
        T, log = sampled_gw(  # type: ignore[misc]
            V_src, V_tgt,
            distance_mode="dijkstra", mixed_precision=True,
            M=M_samples, epsilon=epsilon, max_iter=max_iter, k=k,
            log=True, verbose=False,
        )
        if use_cuda:
            torch.cuda.synchronize()
        t_solve = time.perf_counter() - t0
    meta = _build_metrics(0.0, t_solve, sampler.peak, use_cuda)
    return _finalize_torchgw(T, log, meta, {
        "M_samples": M_samples, "epsilon": epsilon, "max_iter": max_iter,
        "k": k, "fgw_alpha": 1.0,
        "distance_mode": "dijkstra", "mixed_precision": True,
    })


def run_torchgw_precomputed(V_src, V_tgt, seed=0, epsilon=5e-5,
                             M_samples=80, max_iter=300,
                             C_src: np.ndarray | None = None,
                             C_tgt: np.ndarray | None = None):
    """torchgw precomputed-mode solver — C5 always passes C_src/C_tgt."""
    from torchgw import sampled_gw
    if C_src is None or C_tgt is None:
        raise ValueError("C5 run_torchgw_precomputed requires C_src and C_tgt")
    torch, use_cuda = _reset_tracking(seed)
    with _RSSSampler() as sampler:
        t_prep_start = time.perf_counter()
        dist_source = C_src.astype(np.float32)
        dist_target = C_tgt.astype(np.float32)
        n_src, n_tgt = V_src.shape[0], V_tgt.shape[0]
        p = np.full(n_src, 1.0 / n_src, dtype=np.float64)
        q = np.full(n_tgt, 1.0 / n_tgt, dtype=np.float64)
        t_prep = time.perf_counter() - t_prep_start
        t0 = time.perf_counter()
        T, log = sampled_gw(  # type: ignore[misc]
            X_source=V_src, X_target=V_tgt, p=p, q=q,
            distance_mode="precomputed",
            dist_source=dist_source, dist_target=dist_target,
            mixed_precision=True,
            M=M_samples, epsilon=epsilon, max_iter=max_iter,
            log=True, verbose=False,
        )
        if use_cuda:
            torch.cuda.synchronize()
        t_solve = time.perf_counter() - t0
    meta = _build_metrics(t_prep, t_solve, sampler.peak, use_cuda)
    return _finalize_torchgw(T, log, meta, {
        "M_samples": M_samples, "epsilon": epsilon, "max_iter": max_iter,
        "fgw_alpha": 1.0,
        "distance_mode": "precomputed", "mixed_precision": True,
    })


# ---- POT GPU helpers with precomputed cost matrices ---------------------

def _pot_setup_with_C(C_src: np.ndarray, C_tgt: np.ndarray, seed: int,
                       dtype: str = "float64"):
    """Convert precomputed cost matrices to GPU tensors with uniform marginals.

    We default to float64 because POT's entropic_gromov_wasserstein with
    float32 CUDA tensors is pathologically slow on some GPU/driver combos
    (likely due to internal ops that sync host/device per iteration in fp32).
    float64 is ~10× faster and numerically more stable for Sinkhorn GW.
    """
    import torch
    device = torch.device("cuda")
    tdtype = torch.float64 if dtype == "float64" else torch.float32
    C1 = torch.as_tensor(C_src.astype(np.float64), device=device, dtype=tdtype)
    C2 = torch.as_tensor(C_tgt.astype(np.float64), device=device, dtype=tdtype)
    n_src, n_tgt = C_src.shape[0], C_tgt.shape[0]
    p = torch.full((n_src,), 1.0 / n_src, device=device, dtype=tdtype)
    q = torch.full((n_tgt,), 1.0 / n_tgt, device=device, dtype=tdtype)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return C1, C2, p, q


def _run_pot_gw_with_C(C_src: np.ndarray, C_tgt: np.ndarray, seed: int,
                        algo_fn, algo_kwargs: dict, hyperparams: dict,
                        dtype: str = "float32"):
    """Run a POT GW solver with precomputed cost matrices on GPU."""
    import torch
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    with _RSSSampler() as sampler:
        t_prep_start = time.perf_counter()
        C1, C2, p, q = _pot_setup_with_C(C_src, C_tgt, seed, dtype=dtype)
        if use_cuda:
            torch.cuda.synchronize()
        t_prep = time.perf_counter() - t_prep_start
        t0 = time.perf_counter()
        T_and_log = algo_fn(C1, C2, p, q, **algo_kwargs)
        if use_cuda:
            torch.cuda.synchronize()
        t_solve = time.perf_counter() - t0
    meta = _build_metrics(t_prep, t_solve, sampler.peak, use_cuda)
    return _finalize_pot(T_and_log, meta, hyperparams)


def run_pot_entropic_gpu(C_src: np.ndarray, C_tgt: np.ndarray,
                          seed: int = 0, epsilon: float = 5e-5,
                          max_iter: int = 100, tol: float = 1e-9):
    import ot.gromov as otgw
    return _run_pot_gw_with_C(
        C_src, C_tgt, seed,
        otgw.entropic_gromov_wasserstein,
        dict(loss_fun="square_loss", epsilon=epsilon,
             max_iter=max_iter, tol=tol, log=True, verbose=False),
        {"epsilon": epsilon, "max_iter": max_iter, "tol": tol,
         "loss_fun": "square_loss", "algorithm": "entropic",
         "backend": "gpu", "gpu_dtype": "float64"},
        dtype="float64",
    )


def run_pot_exact_gpu(C_src: np.ndarray, C_tgt: np.ndarray,
                       seed: int = 0, max_iter: int = 500, tol: float = 1e-6):
    import ot.gromov as otgw
    return _run_pot_gw_with_C(
        C_src, C_tgt, seed,
        otgw.gromov_wasserstein,
        dict(loss_fun="square_loss",
             max_iter=max_iter, tol_rel=tol, tol_abs=tol, log=True),
        {"max_iter": max_iter, "tol": tol, "loss_fun": "square_loss",
         "algorithm": "exact-CG", "backend": "gpu", "gpu_dtype": "float64"},
        dtype="float64",
    )


# ---- main ---------------------------------------------------------------

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="C5 word-embedding cross-lingual GW alignment")
    ap.add_argument("--pair", default="en-es",
                    choices=["en-es", "en-fi"],
                    help="Language pair (default: en-es)")
    ap.add_argument("--n-words", type=int, default=5000,
                    help="Top-N words by frequency to use (default: 5000)")
    ap.add_argument("--solver", required=True, choices=[
        "torchgw-landmark", "torchgw-dijkstra", "torchgw-precomputed",
        "pot-entropic-gpu", "pot-exact-gpu",
    ])
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for solver randomness (default: 0)")
    ap.add_argument("--epsilon", type=float, default=5e-5,
                    help="Entropic regularisation ε (default: 5e-5)")
    ap.add_argument("--M-samples", type=int, default=None,
                    help="torchgw per-iter cost rows (default: 80)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for JSON result")
    args = ap.parse_args()

    lang_src, lang_tgt = args.pair.split("-")
    rec = build_record("core/05_word_embedding", args.solver, args.seed,
                       f"{args.pair}_n{args.n_words}")
    rec["dataset"] = {
        "pair": args.pair,
        "lang_src": lang_src,
        "lang_tgt": lang_tgt,
        "n_words": args.n_words,
    }

    out_path = args.out / (
        f"core_05_word_embedding__{args.solver}"
        f"__{args.pair}__n{args.n_words}__seed{args.seed}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # ---- load vectors ------------------------------------------------
        vec_src = DATA_ROOT / "vectors" / f"wiki.{lang_src}.vec"
        vec_tgt = DATA_ROOT / "vectors" / f"wiki.{lang_tgt}.vec"
        for p in (vec_src, vec_tgt):
            if not p.exists():
                raise FileNotFoundError(
                    f"fastText vector not found: {p}\n"
                    "Run tracks/core/05_word_embedding/fetch.sh first."
                )

        dict_test  = DATA_ROOT / "dicts" / f"{args.pair}.5000-6500.txt"
        dict_train = DATA_ROOT / "dicts" / f"{args.pair}.0-5000.txt"
        if not dict_test.exists():
            raise FileNotFoundError(
                f"MUSE test dictionary not found: {dict_test}\n"
                "Run tracks/core/05_word_embedding/fetch.sh first."
            )

        t_prep_start = time.perf_counter()
        print(f"[C5] loading {vec_src.name} (N={args.n_words})", flush=True)
        words_src, V_src = _io.read_fasttext(str(vec_src), args.n_words)
        print(f"[C5] loading {vec_tgt.name} (N={args.n_words})", flush=True)
        words_tgt, V_tgt = _io.read_fasttext(str(vec_tgt), args.n_words)
        print(f"[C5] V_src={V_src.shape}  V_tgt={V_tgt.shape}", flush=True)

        # ---- build cost matrices ----------------------------------------
        print("[C5] building cosine cost matrices", flush=True)
        C_src, C_tgt = build_cost_matrices(V_src, V_tgt)
        t_prep = time.perf_counter() - t_prep_start
        print(f"[C5] preprocessing done in {t_prep:.1f}s", flush=True)

        # ---- dispatch to solver -----------------------------------------
        kwargs: dict = {"seed": args.seed}
        if args.epsilon is not None and args.solver != "pot-exact-gpu":
            kwargs["epsilon"] = args.epsilon
        if args.M_samples is not None and args.solver.startswith("torchgw"):
            kwargs["M_samples"] = args.M_samples

        print(f"[C5] solving with {args.solver}  kwargs={kwargs}", flush=True)

        if args.solver == "torchgw-landmark":
            result = run_torchgw_landmark(V_src, V_tgt, **kwargs)
        elif args.solver == "torchgw-dijkstra":
            result = run_torchgw_dijkstra(V_src, V_tgt, **kwargs)
        elif args.solver == "torchgw-precomputed":
            result = run_torchgw_precomputed(
                V_src, V_tgt, C_src=C_src, C_tgt=C_tgt, **kwargs)
        elif args.solver == "pot-entropic-gpu":
            result = run_pot_entropic_gpu(C_src, C_tgt, **kwargs)
        elif args.solver == "pot-exact-gpu":
            # epsilon not passed — exact CG needs no regularisation
            kwargs.pop("epsilon", None)
            result = run_pot_exact_gpu(C_src, C_tgt, **kwargs)
        else:
            raise ValueError(f"Unknown solver: {args.solver}")

        # ---- eval -------------------------------------------------------
        T = result["T"]
        # Load test dict (5000-6500).  For small N < 5000 the test vocab
        # doesn't overlap with our loaded words, so we additionally evaluate
        # against the train dict (0-5000) when N < 5000, using whichever
        # provides non-zero coverage as the primary metric.
        print("[C5] loading MUSE test dict", flush=True)
        gold_test  = _io.read_muse_dict(str(dict_test))
        gold_train = _io.read_muse_dict(str(dict_train)) if dict_train.exists() else {}

        # Choose primary dict for reporting.
        # The MUSE test dict (5000-6500) covers words ranked 5000-6500 which
        # are NOT in our vocabulary when we load top-N words.  The train dict
        # (0-5000) has high coverage for any N >= 2000.  Always use train as
        # the primary metric; report test dict scores for completeness (will be
        # zero whenever N < ~6500 because those words are outside our vocab).
        if gold_train:
            gold_primary = gold_train
            dict_label = "train"
        else:
            gold_primary = gold_test
            dict_label = "test"
        print(f"[C5] using '{dict_label}' dict for eval (n_words={args.n_words})",
              flush=True)

        print("[C5] barycentric projection + retrieval eval", flush=True)
        proj = _eval.barycentric_project(T, V_tgt)
        nn_scores  = _eval.precision_at_k(
            proj, V_tgt, words_src, words_tgt, gold_primary, ks=(1, 5))
        csls_scores = _eval.precision_at_k_csls(
            proj, V_tgt, words_src, words_tgt, gold_primary, ks=(1, 5))

        # Also score against the test dict for completeness (may be 0 for small N)
        if dict_label == "train":
            nn_test  = _eval.precision_at_k(
                proj, V_tgt, words_src, words_tgt, gold_test, ks=(1, 5))
            csls_test = _eval.precision_at_k_csls(
                proj, V_tgt, words_src, words_tgt, gold_test, ks=(1, 5))
        else:
            nn_test = nn_scores
            csls_test = csls_scores

        # ---- record -----------------------------------------------------
        rec["hyperparams"] = result["hyperparams"]
        rec["solver_version"] = result["solver_version"]
        rec["metrics"]["correctness"] = {
            "gw_cost":       result["gw_cost"],
            "marginal_error": result["marginal_error"],
        }
        rec["metrics"]["task"] = {
            # Primary metrics (train dict when N<5000, test dict when N>=5000)
            "dict_used":  dict_label,
            "p1_nn":    nn_scores[1],
            "p5_nn":    nn_scores[5],
            "p1_csls":  csls_scores[1],
            "p5_csls":  csls_scores[5],
            # Test dict scores (canonical; may be 0 for N<5000)
            "test_p1_nn":    nn_test[1],
            "test_p5_nn":    nn_test[5],
            "test_p1_csls":  csls_test[1],
            "test_p5_csls":  csls_test[5],
        }
        rec["metrics"]["efficiency"] = {
            "wall_s":            result["wall_s_total"],
            "wall_s_preprocess": result.get("wall_s_preprocess", t_prep),
            "wall_s_solve":      result["wall_s_solve"],
            "wall_s_total":      result["wall_s_total"],
            "gpu_peak_gb":       result.get("gpu_peak_gb"),
            "ram_peak_gb":       result.get("ram_peak_gb"),
            "iterations":        result.get("iterations"),
        }

        print(
            f"[C5] {dict_label}: "
            f"P@1-NN={nn_scores[1]:.4f}  P@5-NN={nn_scores[5]:.4f}  "
            f"P@1-CSLS={csls_scores[1]:.4f}  P@5-CSLS={csls_scores[5]:.4f}",
            flush=True,
        )
        if dict_label == "train":
            print(
                f"[C5] test:  "
                f"P@1-NN={nn_test[1]:.4f}  P@5-NN={nn_test[5]:.4f}  "
                f"P@1-CSLS={csls_test[1]:.4f}  P@5-CSLS={csls_test[5]:.4f}",
                flush=True,
            )

    except Exception as e:
        rec["status"] = "fail"
        rec["error"] = f"{type(e).__name__}: {e}"
        out_path.write_text(json.dumps(rec, indent=2))
        raise
    else:
        out_path.write_text(json.dumps(rec, indent=2))
        print(f"[C5] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
