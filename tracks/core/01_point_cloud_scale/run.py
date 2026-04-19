#!/usr/bin/env python
"""C1 point-cloud scalability benchmark — GW alignment of rotated ModelNet40 shapes.

Given a ModelNet40 .off file, we downsample to N points, apply a random rotation
to get a target cloud, then align source -> target via GW and evaluate correspondence.

Pipeline design choices:
  * Cost matrix: **kNN-graph hop-count geodesic** (NOT raw Euclidean).  Smoke
    tests at N=1000 showed raw Euclidean cost (dense, mean ~125) gives
    torchgw P@1 ≈ 0 regardless of epsilon — the dense Gaussian-like distribution
    of pairwise distances has insufficient SNR for sampled_gw's MC gradient
    (same pathology as C5 cosine cost).  kNN-hop cost (sparse + long-tailed)
    matches the C2 recipe that lets torchgw work; partial recovery observed
    (torchgw P@1 ~ 0.4 vs POT-exact ~ 0.98 at N=1000).
  * GT correspondence is identity: source[i] <-> target[i] (FPS + rotation).
  * Evaluation: P@1 (correspondence_accuracy), P@5 (correspondence_recall_at_5),
    Chamfer distance (via barycentric projection).

Solvers: 7 GPU variants.
  torchgw-landmark, torchgw-dijkstra          — pass raw point clouds (geodesic cost)
  torchgw-precomputed                         — inject Euclidean cost matrices
  pot-entropic-gpu, pot-exact-gpu             — POT with precomputed Euclidean C
  torchgw-lowrank-landmark                    — low-rank GW, landmark geodesic
  torchgw-lowrank-dijkstra                    — low-rank GW, dijkstra geodesic

N-conditional skip:
  N > 20000: skip pot-entropic-gpu, pot-exact-gpu, torchgw-precomputed (OOM risk).
  N > 50000: warn but proceed for torchgw-landmark, torchgw-dijkstra.
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
DATA_ROOT = REPO_ROOT / "data" / "core_01_point_cloud"


def _load_local(name: str):
    spec = importlib.util.spec_from_file_location(name, TRACK_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_io   = _load_local("io")
_pair = _load_local("pair")
_eval = _load_local("eval")


# ---- cost-matrix construction -------------------------------------------

def build_cost_matrices(P_src: np.ndarray, P_tgt: np.ndarray, k: int = 20):
    """kNN-graph hop-count geodesic cost matrices for source and target.

    Pipeline (matches C2 SCOT recipe):
      1. Build symmetric kNN connectivity graph (k=20 default) on each cloud.
      2. Run unweighted Dijkstra → integer hop-count distance matrix.
      3. Replace any inf (disconnected components) with 1.5× max finite.
      4. Normalize each by its max → both in [0, 1].

    Why not raw Euclidean: dense pairwise Euclidean cost (mean ~125, std ~63
    for ModelNet airplane) has near-Gaussian distribution; torchgw sampled MC
    gradient has SNR < sqrt(2 ln N) → P@1 ≈ 0 regardless of ε (same pathology
    as C5 cosine cost).  kNN-hop cost is sparse + long-tailed, matching the
    structure where torchgw works (cf. C2 single-cell).

    Returns (C_src, C_tgt) both float32 in [0, 1].
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

    def _knn_hop(P: np.ndarray) -> np.ndarray:
        G = kneighbors_graph(P, n_neighbors=k, mode="connectivity",
                             include_self=False)
        G = ((G + G.T) > 0).astype(np.float32)  # symmetrize
        D = scipy_dijkstra(G, unweighted=True, directed=False).astype(np.float32)
        finite = D[~np.isinf(D)]
        if finite.size and np.isinf(D).any():
            D[np.isinf(D)] = float(finite.max()) * 1.5
        m = float(D.max())
        return (D / m).astype(np.float32) if m > 0 else D

    return _knn_hop(P_src), _knn_hop(P_tgt)


def _normalize_cost_mean(C: np.ndarray) -> np.ndarray:
    """Mean-normalize a cost matrix: C / C.mean() so typical entry ≈ 1.
    Used by POT callers to put epsilon on the natural [0,1] scale."""
    C = np.array(C, dtype=np.float32)
    m = float(C.mean())
    return C if m == 0.0 else (C / m).astype(np.float32)


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


# ---- memory / timing helpers ---------------------------------------------

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


# ---- torchgw finalizer ---------------------------------------------------

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


# ---- POT finalizer -------------------------------------------------------

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

def run_torchgw_landmark(P_src, P_tgt, seed=0, epsilon=5e-3,
                          M_samples=80, max_iter=300, k=10, n_landmarks=50):
    from torchgw import sampled_gw
    torch, use_cuda = _reset_tracking(seed)
    with _RSSSampler() as sampler:
        t0 = time.perf_counter()
        T, log = sampled_gw(  # type: ignore[misc]
            P_src, P_tgt,
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
        "k": k, "n_landmarks": n_landmarks, "fgw_alpha": 0.0,
        "distance_mode": "landmark", "mixed_precision": True,
    })


def run_torchgw_dijkstra(P_src, P_tgt, seed=0, epsilon=5e-3,
                          M_samples=80, max_iter=300, k=10):
    from torchgw import sampled_gw
    torch, use_cuda = _reset_tracking(seed)
    with _RSSSampler() as sampler:
        t0 = time.perf_counter()
        T, log = sampled_gw(  # type: ignore[misc]
            P_src, P_tgt,
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
        "k": k, "fgw_alpha": 0.0,
        "distance_mode": "dijkstra", "mixed_precision": True,
    })


def run_torchgw_precomputed(P_src, P_tgt, seed=0, epsilon=5e-3,
                             M_samples=80, max_iter=300,
                             C_src: np.ndarray | None = None,
                             C_tgt: np.ndarray | None = None):
    """torchgw precomputed-mode solver — injects precomputed Euclidean cost matrices."""
    from torchgw import sampled_gw
    if C_src is None or C_tgt is None:
        raise ValueError("run_torchgw_precomputed requires C_src and C_tgt")
    torch, use_cuda = _reset_tracking(seed)
    with _RSSSampler() as sampler:
        t_prep_start = time.perf_counter()
        # Mean-normalize for numerical stability (torchgw precomputed mode
        # passes the matrix directly into Sinkhorn iterations)
        dist_source = _normalize_cost_mean(C_src)
        dist_target = _normalize_cost_mean(C_tgt)
        n_src, n_tgt = P_src.shape[0], P_tgt.shape[0]
        p = np.full(n_src, 1.0 / n_src, dtype=np.float64)
        q = np.full(n_tgt, 1.0 / n_tgt, dtype=np.float64)
        t_prep = time.perf_counter() - t_prep_start
        t0 = time.perf_counter()
        T, log = sampled_gw(  # type: ignore[misc]
            X_source=P_src, X_target=P_tgt, p=p, q=q,
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
        "fgw_alpha": 0.0,
        "distance_mode": "precomputed", "mixed_precision": True,
    })


# ---- lowrank solver wrappers --------------------------------------------

def run_torchgw_lowrank_landmark(P_src, P_tgt, seed=0, epsilon=5e-3,
                                  M_samples=80, max_iter=300, rank=20):
    from torchgw import sampled_lowrank_gw
    torch, use_cuda = _reset_tracking(seed)
    with _RSSSampler() as sampler:
        t_prep_start = time.perf_counter()
        p = np.full(P_src.shape[0], 1.0 / P_src.shape[0], dtype=np.float64)
        q = np.full(P_tgt.shape[0], 1.0 / P_tgt.shape[0], dtype=np.float64)
        t_prep = time.perf_counter() - t_prep_start
        t0 = time.perf_counter()
        T, log = sampled_lowrank_gw(
            X_source=P_src, X_target=P_tgt, p=p, q=q,
            distance_mode="landmark",
            fgw_alpha=0.0, M=M_samples, epsilon=epsilon,
            max_iter=max_iter, rank=rank,
            mixed_precision=True, log=True,
        )
        if use_cuda:
            torch.cuda.synchronize()
        t_solve = time.perf_counter() - t0
    meta = _build_metrics(t_prep, t_solve, sampler.peak, use_cuda)
    return _finalize_torchgw(T, log, meta, {
        "M_samples": M_samples, "epsilon": epsilon, "max_iter": max_iter,
        "rank": rank, "distance_mode": "landmark", "lowrank": True,
        "mixed_precision": True,
    })


def run_torchgw_lowrank_dijkstra(P_src, P_tgt, seed=0, epsilon=5e-3,
                                  M_samples=80, max_iter=300, rank=20):
    from torchgw import sampled_lowrank_gw
    torch, use_cuda = _reset_tracking(seed)
    with _RSSSampler() as sampler:
        t_prep_start = time.perf_counter()
        p = np.full(P_src.shape[0], 1.0 / P_src.shape[0], dtype=np.float64)
        q = np.full(P_tgt.shape[0], 1.0 / P_tgt.shape[0], dtype=np.float64)
        t_prep = time.perf_counter() - t_prep_start
        t0 = time.perf_counter()
        T, log = sampled_lowrank_gw(
            X_source=P_src, X_target=P_tgt, p=p, q=q,
            distance_mode="dijkstra",
            fgw_alpha=0.0, M=M_samples, epsilon=epsilon,
            max_iter=max_iter, rank=rank,
            mixed_precision=True, log=True,
        )
        if use_cuda:
            torch.cuda.synchronize()
        t_solve = time.perf_counter() - t0
    meta = _build_metrics(t_prep, t_solve, sampler.peak, use_cuda)
    return _finalize_torchgw(T, log, meta, {
        "M_samples": M_samples, "epsilon": epsilon, "max_iter": max_iter,
        "rank": rank, "distance_mode": "dijkstra", "lowrank": True,
        "mixed_precision": True,
    })


# ---- POT GPU helpers with precomputed cost matrices ---------------------

def _pot_setup_with_C(C_src: np.ndarray, C_tgt: np.ndarray, seed: int,
                       dtype: str = "float64"):
    """Convert precomputed cost matrices to GPU tensors with uniform marginals.

    We default to float64 because POT's entropic_gromov_wasserstein with
    float32 CUDA tensors is pathologically slow on this GPU/driver combo
    (same issue as C5 — internal ops sync host/device per iteration in fp32).
    float64 is much faster and numerically stable for Sinkhorn GW.
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
                        dtype: str = "float64"):
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
                          seed: int = 0, epsilon: float = 5e-3,
                          max_iter: int = 100, tol: float = 1e-9):
    import ot.gromov as otgw
    # Mean-normalize so epsilon is on same scale as cost (~1 after normalization)
    C_src_n = _normalize_cost_mean(C_src)
    C_tgt_n = _normalize_cost_mean(C_tgt)
    return _run_pot_gw_with_C(
        C_src_n, C_tgt_n, seed,
        otgw.entropic_gromov_wasserstein,
        dict(loss_fun="square_loss", epsilon=epsilon,
             max_iter=max_iter, tol=tol, log=True, verbose=False),
        {"epsilon": epsilon, "max_iter": max_iter, "tol": tol,
         "loss_fun": "square_loss", "algorithm": "entropic",
         "backend": "gpu", "gpu_dtype": "float64",
         "cost_normalized": "mean"},
        dtype="float64",
    )


def run_pot_exact_gpu(C_src: np.ndarray, C_tgt: np.ndarray,
                       seed: int = 0, max_iter: int = 500, tol: float = 1e-6):
    import ot.gromov as otgw
    # Mean-normalize for numerical stability
    C_src_n = _normalize_cost_mean(C_src)
    C_tgt_n = _normalize_cost_mean(C_tgt)
    return _run_pot_gw_with_C(
        C_src_n, C_tgt_n, seed,
        otgw.gromov_wasserstein,
        dict(loss_fun="square_loss",
             max_iter=max_iter, tol_rel=tol, tol_abs=tol, log=True),
        {"max_iter": max_iter, "tol": tol, "loss_fun": "square_loss",
         "algorithm": "exact-CG", "backend": "gpu", "gpu_dtype": "float64",
         "cost_normalized": "mean"},
        dtype="float64",
    )


# ---- main ---------------------------------------------------------------

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="C1 point-cloud scalability GW benchmark")
    ap.add_argument("--shape-class", default="airplane",
                    choices=["airplane", "car", "lamp", "table", "sofa"],
                    help="ModelNet40 shape class (default: airplane)")
    ap.add_argument("--instance-idx", type=int, default=0,
                    help="Instance index 0-based (file is 1-indexed: 0 -> 0001.off)")
    ap.add_argument("--n-points", type=int, default=5000,
                    help="Number of points after FPS downsampling (default: 5000)")
    ap.add_argument("--solver", required=True, choices=[
        "pot-entropic-gpu", "pot-exact-gpu",
        "torchgw-landmark", "torchgw-dijkstra", "torchgw-precomputed",
        "torchgw-lowrank-landmark", "torchgw-lowrank-dijkstra",
    ])
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for FPS + rotation + solver randomness (default: 0)")
    ap.add_argument("--epsilon", type=float, default=5e-2,
                    help="Entropic regularisation ε (default: 5e-2 — kNN-hop "
                         "cost sweet spot per smoke test; smaller ε on this "
                         "structured cost makes torchgw collapse to uniform plan)")
    ap.add_argument("--M-samples", type=int, default=None,
                    help="torchgw per-iter cost rows (default: 80 in solver)")
    ap.add_argument("--lowrank-rank", type=int, default=20,
                    help="Rank for lowrank variants (default: 20)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for JSON result")
    args = ap.parse_args()

    # ---- instance path ---------------------------------------------------
    # ModelNet40 files are 1-indexed: --instance-idx 0 -> airplane_0001.off
    file_idx = args.instance_idx + 1
    shape_class = args.shape_class
    off_path = (DATA_ROOT / "ModelNet40" / shape_class / "train" /
                f"{shape_class}_{file_idx:04d}.off")

    # ---- output path -----------------------------------------------------
    out_stem = (
        f"core_01_point_cloud_scale__{args.solver}"
        f"__{shape_class}_i{args.instance_idx}"
        f"__n{args.n_points}__seed{args.seed}"
    )
    out_path = args.out / f"{out_stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rec = build_record(
        "core/01_point_cloud_scale",
        args.solver,
        args.seed,
        f"{shape_class}_i{args.instance_idx}_n{args.n_points}",
    )
    rec["dataset"] = {
        "shape_class":    shape_class,
        "instance_idx":   args.instance_idx,
        "file_idx":       file_idx,
        "off_path":       str(off_path),
        "n_points":       args.n_points,
    }

    def write_json(r: dict):
        # Strip non-serializable T array before dumping
        r_dump = {k: v for k, v in r.items() if k != "T"}
        out_path.write_text(json.dumps(r_dump, indent=2))

    # ---- N-conditional OOM skip -----------------------------------------
    oom_solvers = {"pot-entropic-gpu", "pot-exact-gpu", "torchgw-precomputed"}
    if args.n_points > 20000 and args.solver in oom_solvers:
        rec["status"] = "skipped_oom_risk"
        rec["error"] = (
            f"N={args.n_points} > 20000, OOM risk; not attempted"
        )
        write_json(rec)
        print(f"[C1] skipped (OOM risk): {out_path}", flush=True)
        return

    large_solvers = {"torchgw-landmark", "torchgw-dijkstra"}
    if args.n_points > 50000 and args.solver in large_solvers:
        print(
            f"[C1] WARNING: N={args.n_points} > 50000 for {args.solver}; "
            "kNN graph at this scale is borderline — proceeding.",
            flush=True,
        )

    try:
        # ---- load and downsample -----------------------------------------
        if not off_path.exists():
            raise FileNotFoundError(
                f"ModelNet40 .off file not found: {off_path}\n"
                "Run data download / extraction for C1 first."
            )
        print(f"[C1] loading {off_path.name}", flush=True)
        P_raw = _io.read_off(str(off_path))
        print(f"[C1] raw shape: {P_raw.shape}", flush=True)

        # ---- make pair ---------------------------------------------------
        print(
            f"[C1] make_pair N={args.n_points} seed={args.seed}", flush=True
        )
        P_src, P_tgt, R_gt = _pair.make_pair(P_raw, args.n_points, seed=args.seed)
        print(
            f"[C1] P_src={P_src.shape}  P_tgt={P_tgt.shape}  R_gt={R_gt.shape}",
            flush=True,
        )

        # ---- build cost matrices (only when needed) ----------------------
        C_src: np.ndarray | None = None
        C_tgt: np.ndarray | None = None
        cost_solvers = {"pot-entropic-gpu", "pot-exact-gpu", "torchgw-precomputed"}
        if args.solver in cost_solvers:
            print("[C1] building kNN-hop geodesic cost matrices …", flush=True)
            t_cost0 = time.perf_counter()
            C_src, C_tgt = build_cost_matrices(P_src, P_tgt)
            print(
                f"[C1] cost matrices built in {time.perf_counter() - t_cost0:.1f}s",
                flush=True,
            )

        # ---- dispatch to solver -----------------------------------------
        kwargs: dict = {"seed": args.seed, "epsilon": args.epsilon}
        if args.M_samples is not None:
            kwargs["M_samples"] = args.M_samples

        print(
            f"[C1] solving with {args.solver}  kwargs={kwargs}", flush=True
        )

        if args.solver == "torchgw-landmark":
            result = run_torchgw_landmark(P_src, P_tgt, **kwargs)
        elif args.solver == "torchgw-dijkstra":
            result = run_torchgw_dijkstra(P_src, P_tgt, **kwargs)
        elif args.solver == "torchgw-precomputed":
            assert C_src is not None and C_tgt is not None
            result = run_torchgw_precomputed(
                P_src, P_tgt, C_src=C_src, C_tgt=C_tgt, **kwargs)
        elif args.solver == "torchgw-lowrank-landmark":
            result = run_torchgw_lowrank_landmark(
                P_src, P_tgt, rank=args.lowrank_rank, **kwargs)
        elif args.solver == "torchgw-lowrank-dijkstra":
            result = run_torchgw_lowrank_dijkstra(
                P_src, P_tgt, rank=args.lowrank_rank, **kwargs)
        elif args.solver == "pot-entropic-gpu":
            assert C_src is not None and C_tgt is not None
            result = run_pot_entropic_gpu(C_src, C_tgt, **kwargs)
        elif args.solver == "pot-exact-gpu":
            assert C_src is not None and C_tgt is not None
            kwargs.pop("epsilon", None)
            result = run_pot_exact_gpu(C_src, C_tgt, **kwargs)
        else:
            raise ValueError(f"Unknown solver: {args.solver}")

        # ---- eval -------------------------------------------------------
        T = result["T"]
        print("[C1] evaluating correspondences …", flush=True)
        p1  = _eval.correspondence_accuracy(T)
        p5  = _eval.correspondence_recall_at_k(T, k=5)
        proj = _eval.barycentric_project(T, P_tgt)
        cd  = _eval.chamfer_distance(proj, P_tgt)

        print(
            f"[C1] P@1={p1:.4f}  P@5={p5:.4f}  Chamfer={cd:.6f}",
            flush=True,
        )

        # ---- record -----------------------------------------------------
        rec["hyperparams"]    = result["hyperparams"]
        rec["solver_version"] = result["solver_version"]
        rec["metrics"]["correctness"] = {
            "gw_cost":        result["gw_cost"],
            "marginal_error": result["marginal_error"],
        }
        rec["metrics"]["task"] = {
            "correspondence_accuracy":  float(p1),
            "correspondence_recall_at_5": float(p5),
            "chamfer_distance":         float(cd),
            "n_points":                 int(args.n_points),
        }
        rec["metrics"]["efficiency"] = {
            "wall_preprocess_s": result.get("wall_s_preprocess", 0.0),
            "wall_solve_s":      result["wall_s_solve"],
            "gpu_peak_gb":       result.get("gpu_peak_gb"),
            "ram_peak_gb":       result.get("ram_peak_gb"),
        }

    except Exception as e:
        rec["status"] = "fail"
        rec["error"] = f"{type(e).__name__}: {e}"
        write_json(rec)
        raise
    else:
        write_json(rec)
        print(f"[C1] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
