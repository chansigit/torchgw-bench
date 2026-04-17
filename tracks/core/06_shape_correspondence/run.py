#!/usr/bin/env python
"""C6 TACO shape correspondence — pure GW matching across poses.

Given two TACO meshes of the same class in different poses, build a
Gromov-Wasserstein transport plan that maps source vertices to target
vertices, and evaluate against TACO's ground-truth vertex correspondence.

All solvers run pure GW (fgw_alpha = 1.0, no linear feature cost). The
v1 benchmark targets 5 GPU solvers at a fixed subsample size; scale
sweep and FGW-with-HKS features are deferred to v2.

Subsampling: we uniformly random-sample n_source vertices from the source
mesh and n_target vertices from the target, remapping the TACO ground
truth to the subsampled indices via nearest-vertex on the target cloud.

Structural distance: intrinsic geodesic distance on the mesh, computed
as Dijkstra shortest paths on a kNN graph over the subsampled point cloud
(sklearn kNN + scipy.sparse.csgraph.shortest_path). This matches the
structural prior that torchgw-dijkstra builds internally.
"""
from __future__ import annotations

import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Any

import numpy as np

TRACK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRACK_ROOT.parents[2]
DATA_ROOT = REPO_ROOT / "data" / "core_06_shape" / "taco"

# ---- record helper (mirrors 03_branched convention) ---------------------

def build_record(track: str, solver: str, seed: int, subset: str) -> dict:
    import socket
    import datetime as _dt
    return {
        "track": track,
        "solver": solver,
        "seed": seed,
        "solver_version": None,
        "subset": subset,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "status": "ok",
        "error": None,
        "dataset": {},
        "hyperparams": {},
        "metrics": {"correctness": {}, "task": {}, "efficiency": {}, "stability": {}},
        "artifacts": {},
    }


# ---- OFF mesh loader ----------------------------------------------------

def load_off(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an OFF / NOFF mesh file. Returns (V (n_v, 3), F (n_f, 3))."""
    with open(path, "r") as f:
        lines = f.readlines()
    header = lines[0].strip()
    if header not in ("OFF", "NOFF"):
        raise ValueError(f"unexpected OFF header in {path}: {header!r}")
    parts = lines[1].split()
    nv, nf = int(parts[0]), int(parts[1])
    V = np.empty((nv, 3), dtype=np.float32)
    for i in range(nv):
        V[i] = list(map(float, lines[2 + i].split()[:3]))
    F = np.empty((nf, 3), dtype=np.int64)
    for i in range(nf):
        F[i] = list(map(int, lines[2 + nv + i].split()[1:4]))
    return V, F


def load_taco_pair(src: str, tgt: str, data_root: Path = DATA_ROOT):
    """Load a TACO (src, tgt) pair. Ground truth Pi is 1-indexed in the
    .mat file; we return 0-indexed gt[i_src] = i_tgt_full. Because TACO
    meshes have different connectivities, the inverse-mapping case (only
    the reverse GT exists) may leave some source vertices without a GT;
    those are marked -1 in gt and must be filtered by the caller."""
    from scipy.io import loadmat
    V_src, F_src = load_off(data_root / "offs" / f"{src}.off")
    V_tgt, F_tgt = load_off(data_root / "offs" / f"{tgt}.off")
    fwd = data_root / "gt_matches" / f"{src}_{tgt}.mat"
    rev = data_root / "gt_matches" / f"{tgt}_{src}.mat"
    if fwd.exists():
        gt = loadmat(fwd)["Pi"].ravel().astype(np.int64) - 1
    elif rev.exists():
        gt_inv = loadmat(rev)["Pi"].ravel().astype(np.int64) - 1
        # Pi_inv[j_tgt] = i_src (1-indexed - 1). Invert: for each unique
        # src index hit, record the first target that mapped to it.
        gt = np.full(V_src.shape[0], -1, dtype=np.int64)
        for j_tgt, i_src in enumerate(gt_inv):
            if 0 <= i_src < V_src.shape[0] and gt[i_src] == -1:
                gt[i_src] = j_tgt
    else:
        raise FileNotFoundError(f"no gt for pair {src}-{tgt}")
    return V_src, F_src, V_tgt, F_tgt, gt


# ---- subsampling with GT remap ------------------------------------------

def subsample_pair(
    V_src: np.ndarray, V_tgt: np.ndarray, gt_full: np.ndarray,
    n_src: int, n_tgt: int, seed: int,
):
    """Uniform random subsample of both clouds. Remap gt to subsampled
    target indices via nearest neighbour when the exact GT target
    vertex wasn't kept. Returns (V_src_sub, V_tgt_sub, gt_sub)."""
    rng = np.random.default_rng(seed)
    n_src = min(n_src, V_src.shape[0])
    n_tgt = min(n_tgt, V_tgt.shape[0])
    idx_src = rng.choice(V_src.shape[0], n_src, replace=False)
    idx_tgt = rng.choice(V_tgt.shape[0], n_tgt, replace=False)
    # Map full_tgt_idx -> sub_tgt_idx; -1 if not in subsample.
    tgt_full_to_sub = -np.ones(V_tgt.shape[0], dtype=np.int64)
    tgt_full_to_sub[idx_tgt] = np.arange(n_tgt)

    V_src_sub = V_src[idx_src]
    V_tgt_sub = V_tgt[idx_tgt]
    # Nearest-neighbour fallback for GT targets not in subsample.
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(V_tgt_sub)
    gt_sub = np.empty(n_src, dtype=np.int64)
    for i, full_i in enumerate(idx_src):
        full_gt = gt_full[full_i]
        sub = tgt_full_to_sub[full_gt]
        if sub >= 0:
            gt_sub[i] = sub
        else:
            # nearest-in-subsample to the full GT target vertex
            gt_sub[i] = int(nn.kneighbors(V_tgt[full_gt:full_gt + 1])[1][0, 0])
    return V_src_sub, V_tgt_sub, gt_sub


# ---- geodesics on subsampled cloud --------------------------------------

def knn_geodesic_matrix(V: np.ndarray, k: int = 8) -> np.ndarray:
    """All-pairs geodesic distance on a kNN graph of V (Dijkstra). Used
    both as the structural cost for POT/precomputed and for computing
    geodesic error on the target side."""
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    nn = NearestNeighbors(n_neighbors=k + 1).fit(V)
    dists, idx = nn.kneighbors(V)
    n = V.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    data = dists[:, 1:].ravel().astype(np.float64)
    G = csr_matrix((data, (rows, cols)), shape=(n, n))
    G = G.maximum(G.T)  # symmetrise
    D = shortest_path(G, method="D", directed=False)
    if not np.all(np.isfinite(D)):
        finite_max = D[np.isfinite(D)].max()
        D[~np.isfinite(D)] = 2.0 * finite_max
    return D.astype(np.float32)


# ---- metrics ------------------------------------------------------------

def geodesic_error(T: np.ndarray, D_tgt: np.ndarray, gt: np.ndarray) -> dict:
    """Mean and median geodesic error of argmax(T) vs gt. Normalised by
    the target mesh's geodesic diameter."""
    pred = np.asarray(T).argmax(axis=1)
    diam = float(D_tgt.max())
    err = D_tgt[pred, gt]
    return {
        "mean_err_absolute": float(err.mean()),
        "median_err_absolute": float(np.median(err)),
        "mean_err_normalised": float(err.mean() / diam),
        "median_err_normalised": float(np.median(err) / diam),
        "diameter": diam,
    }


def match_accuracy_curve(
    T: np.ndarray, D_tgt: np.ndarray, gt: np.ndarray,
    thresholds=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25),
) -> list[tuple[float, float]]:
    """Fraction of predictions within threshold × diameter of GT."""
    pred = np.asarray(T).argmax(axis=1)
    diam = float(D_tgt.max())
    err_n = D_tgt[pred, gt] / diam
    return [(float(t), float((err_n <= t).mean())) for t in thresholds]


# ---- memory sampler (verbatim from 03_branched) -------------------------

class _RSSSampler:
    """Sample this process's RSS at fixed intervals; expose peak."""

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


# ---- torchgw solver wrappers (pure GW, no FGW feature) ------------------

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


def run_torchgw_landmark(
    V_src, V_tgt, seed=0, epsilon=5e-3, M_samples=80,
    max_iter=300, k=5, n_landmarks=50,
    min_iter_before_converge=None, tol=None,
):
    from torchgw import sampled_gw
    torch, use_cuda = _reset_tracking(seed)
    with _RSSSampler() as sampler:
        t_prep = 0.0  # distance build is internal
        t0 = time.perf_counter()
        extra = {}
        if min_iter_before_converge is not None:
            extra["min_iter_before_converge"] = min_iter_before_converge
        if tol is not None:
            extra["tol"] = tol
        T, log = sampled_gw(  # type: ignore[misc]
            V_src, V_tgt,
            distance_mode="landmark", mixed_precision=True,
            M=M_samples, epsilon=epsilon, max_iter=max_iter,
            k=k, n_landmarks=n_landmarks,
            log=True, verbose=False, **extra,
        )
        if use_cuda:
            torch.cuda.synchronize()
        t_solve = time.perf_counter() - t0
    meta = _build_metrics(t_prep, t_solve, sampler.peak, use_cuda)
    return _finalize_torchgw(T, log, meta, {
        "M_samples": M_samples, "epsilon": epsilon, "max_iter": max_iter,
        "k": k, "n_landmarks": n_landmarks, "fgw_alpha": 1.0,
        "distance_mode": "landmark", "mixed_precision": True,
    })


def run_torchgw_dijkstra(
    V_src, V_tgt, seed=0, epsilon=5e-3, M_samples=80,
    max_iter=300, k=5,
    min_iter_before_converge=None, tol=None,
):
    from torchgw import sampled_gw
    torch, use_cuda = _reset_tracking(seed)
    with _RSSSampler() as sampler:
        t_prep = 0.0
        t0 = time.perf_counter()
        extra = {}
        if min_iter_before_converge is not None:
            extra["min_iter_before_converge"] = min_iter_before_converge
        if tol is not None:
            extra["tol"] = tol
        T, log = sampled_gw(  # type: ignore[misc]
            V_src, V_tgt,
            distance_mode="dijkstra", mixed_precision=True,
            M=M_samples, epsilon=epsilon, max_iter=max_iter,
            k=k,
            log=True, verbose=False, **extra,
        )
        if use_cuda:
            torch.cuda.synchronize()
        t_solve = time.perf_counter() - t0
    meta = _build_metrics(t_prep, t_solve, sampler.peak, use_cuda)
    return _finalize_torchgw(T, log, meta, {
        "M_samples": M_samples, "epsilon": epsilon, "max_iter": max_iter,
        "k": k, "fgw_alpha": 1.0,
        "distance_mode": "dijkstra", "mixed_precision": True,
    })


def run_torchgw_precomputed(
    V_src, V_tgt, seed=0, epsilon=5e-3, M_samples=80, max_iter=300,
    min_iter_before_converge=None, tol=None,
):
    from torchgw import sampled_gw
    torch, use_cuda = _reset_tracking(seed)
    with _RSSSampler() as sampler:
        t_prep_start = time.perf_counter()
        # Use kNN geodesic rather than dense Euclidean — appropriate for
        # shape correspondence (intrinsic geometry).
        dist_source = knn_geodesic_matrix(V_src)
        dist_target = knn_geodesic_matrix(V_tgt)
        dist_source /= (dist_source.max() + 1e-12)
        dist_target /= (dist_target.max() + 1e-12)
        n_src, n_tgt = V_src.shape[0], V_tgt.shape[0]
        p = np.full(n_src, 1.0 / n_src, dtype=np.float64)
        q = np.full(n_tgt, 1.0 / n_tgt, dtype=np.float64)
        t_prep = time.perf_counter() - t_prep_start

        t0 = time.perf_counter()
        extra = {}
        if min_iter_before_converge is not None:
            extra["min_iter_before_converge"] = min_iter_before_converge
        if tol is not None:
            extra["tol"] = tol
        T, log = sampled_gw(  # type: ignore[misc]
            X_source=V_src, X_target=V_tgt, p=p, q=q,
            distance_mode="precomputed",
            dist_source=dist_source, dist_target=dist_target,
            mixed_precision=True,
            M=M_samples, epsilon=epsilon, max_iter=max_iter,
            log=True, verbose=False, **extra,
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


# ---- POT wrappers (GPU, pure GW) ----------------------------------------

def _pot_setup_gpu(V_src, V_tgt, seed, dtype: str = "float32"):
    """Build (C1, C2, p, q) as GPU torch tensors using kNN geodesic
    distances. Pure GW uses no linear cost matrix."""
    import torch
    device = torch.device("cuda")
    tdtype = torch.float64 if dtype == "float64" else torch.float32
    C1_np = knn_geodesic_matrix(V_src)
    C2_np = knn_geodesic_matrix(V_tgt)
    C1 = torch.as_tensor(C1_np ** 2, device=device, dtype=tdtype)
    C2 = torch.as_tensor(C2_np ** 2, device=device, dtype=tdtype)
    C1 /= (C1.max() + 1e-12)
    C2 /= (C2.max() + 1e-12)
    p = torch.full((V_src.shape[0],), 1.0 / V_src.shape[0], device=device, dtype=tdtype)
    q = torch.full((V_tgt.shape[0],), 1.0 / V_tgt.shape[0], device=device, dtype=tdtype)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return C1, C2, p, q


def _finalize_pot(T_and_log, meta, hyperparams):
    from typing import Any
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


def _run_pot_gw_gpu(V_src, V_tgt, seed, algo_fn, algo_kwargs, hyperparams,
                      dtype="float32"):
    import torch
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    with _RSSSampler() as sampler:
        t_prep_start = time.perf_counter()
        C1, C2, p, q = _pot_setup_gpu(V_src, V_tgt, seed, dtype=dtype)
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


def run_pot_entropic_gpu(V_src, V_tgt, seed=0, epsilon=5e-3, max_iter=100,
                           tol=1e-9):
    import ot.gromov as otgw
    return _run_pot_gw_gpu(
        V_src, V_tgt, seed,
        otgw.entropic_gromov_wasserstein,
        dict(loss_fun="square_loss", epsilon=epsilon,
              max_iter=max_iter, tol=tol, log=True, verbose=False),
        {"epsilon": epsilon, "max_iter": max_iter, "tol": tol,
         "loss_fun": "square_loss", "algorithm": "entropic",
         "backend": "gpu", "gpu_dtype": "float32"},
        dtype="float32",
    )


def run_pot_exact_gpu(V_src, V_tgt, seed=0, max_iter=500, tol=1e-6):
    import ot.gromov as otgw
    return _run_pot_gw_gpu(
        V_src, V_tgt, seed,
        otgw.gromov_wasserstein,
        dict(loss_fun="square_loss",
              max_iter=max_iter, tol_rel=tol, tol_abs=tol, log=True),
        {"max_iter": max_iter, "tol": tol, "loss_fun": "square_loss",
         "algorithm": "exact-CG", "backend": "gpu", "gpu_dtype": "float32"},
        dtype="float32",
    )


# ---- main ---------------------------------------------------------------

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="C6 TACO shape correspondence")
    ap.add_argument("--solver", required=True, choices=[
        "torchgw-landmark", "torchgw-dijkstra", "torchgw-precomputed",
        "pot-entropic-gpu", "pot-exact-gpu",
    ])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pair", required=True,
                    help="TACO pair as 'src,tgt' (e.g. 'cat0,cat1')")
    ap.add_argument("--n-source", type=int, default=2000)
    ap.add_argument("--n-target", type=int, default=2000)
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--force-full", action="store_true")
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    src_name, tgt_name = args.pair.split(",")
    rec = build_record("core/06_shape_correspondence", args.solver, args.seed, "full")
    rec["dataset"] = {
        "name": f"taco_{src_name}_to_{tgt_name}_n{args.n_source}",
        "source": src_name, "target": tgt_name,
        "n_source": args.n_source, "n_target": args.n_target,
    }
    _tag_part = f"__{args.tag}" if args.tag else ""
    out_path = args.out / (
        f"core_06_shape_correspondence__{args.solver}"
        f"__{src_name}_{tgt_name}__n{args.n_source}"
        f"__seed{args.seed}{_tag_part}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        V_src_full, _, V_tgt_full, _, gt_full = load_taco_pair(src_name, tgt_name)
        V_src, V_tgt, gt = subsample_pair(
            V_src_full, V_tgt_full, gt_full,
            args.n_source, args.n_target, args.seed,
        )
        # Precompute target geodesic matrix once (used for both metric calc
        # and — if applicable — the precomputed solver already built its
        # own). Keep a reference for the metric step.
        D_tgt_for_metric = knn_geodesic_matrix(V_tgt)

        solver_fns = {
            "torchgw-landmark":    run_torchgw_landmark,
            "torchgw-dijkstra":    run_torchgw_dijkstra,
            "torchgw-precomputed": run_torchgw_precomputed,
            "pot-entropic-gpu":    run_pot_entropic_gpu,
            "pot-exact-gpu":       run_pot_exact_gpu,
        }
        fn = solver_fns[args.solver]
        extra_kwargs: dict = {}
        if args.max_iter is not None:
            extra_kwargs["max_iter"] = args.max_iter
        if args.force_full:
            mi = args.max_iter if args.max_iter is not None else 500
            if args.solver.startswith("torchgw"):
                extra_kwargs["min_iter_before_converge"] = mi
                extra_kwargs["tol"] = 0.0
            else:
                extra_kwargs["tol"] = 0.0
        result = fn(V_src, V_tgt, seed=args.seed, **extra_kwargs)

        # Metrics
        geo = geodesic_error(result["T"], D_tgt_for_metric, gt)
        curve = match_accuracy_curve(result["T"], D_tgt_for_metric, gt)
        rec["hyperparams"] = result["hyperparams"]
        rec["solver_version"] = result["solver_version"]
        rec["metrics"]["correctness"] = {
            "gw_cost": result["gw_cost"],
            "marginal_error": result["marginal_error"],
        }
        rec["metrics"]["task"] = {
            **geo,
            "accuracy_curve": curve,
        }
        rec["metrics"]["efficiency"] = {
            "wall_s":            result["wall_s_total"],
            "wall_s_preprocess": result["wall_s_preprocess"],
            "wall_s_solve":      result["wall_s_solve"],
            "wall_s_total":      result["wall_s_total"],
            "gpu_peak_gb":       result["gpu_peak_gb"],
            "ram_peak_gb":       result["ram_peak_gb"],
            "iterations":        result["iterations"],
        }
    except Exception as e:
        rec["status"] = "fail"
        rec["error"] = f"{type(e).__name__}: {e}"
        out_path.write_text(json.dumps(rec, indent=2))
        raise
    else:
        out_path.write_text(json.dumps(rec, indent=2))
        print(f"[C6] wrote {out_path}")


if __name__ == "__main__":
    main()
