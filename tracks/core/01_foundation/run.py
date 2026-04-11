#!/usr/bin/env python
"""Track: core/01_foundation — spiral → Swiss roll GW alignment.

Phase 1 scope: N=400, K=500 only; solvers torchgw-landmark and pot-entropic.

This file is self-contained. It does NOT import from any sibling track or
from scripts/. Helper functions defined here are unit-tested by
tests/test_run.py via a sys.path hook in tests/conftest.py.

Note: this stub will be filled in by Tasks 5–10 of the Phase 1 plan. The
``__all__`` export list is deliberately omitted in the stub and will be
added in Task 10 once all helpers exist.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def sample_spiral(n: int, noise: float = 0.05, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """2D Archimedean spiral with Gaussian noise.

    Returns:
        points: (n, 2) float32 array
        angles: (n,) float64 array, the parameter used to generate each point
    """
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.3, 1.0, n)
    angles = np.linspace(0, 9, n)
    eps = rng.normal(size=(2, n)) * noise
    x = (radius + eps[0]) * np.cos(angles)
    y = (radius + eps[1]) * np.sin(angles)
    points = np.stack((x, y), axis=1).astype(np.float32)
    return points, angles


def sample_swiss_roll(n: int, noise: float = 0.05, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """3D Swiss roll parameterised by the same angular schedule as the spiral.

    Returns:
        points: (n, 3) float32 array
        angles: (n,) float64 array
    """
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.3, 1.0, n)
    angles = np.linspace(0, 9, n)
    eps = rng.normal(size=(2, n)) * noise
    x = (radius + eps[0]) * np.cos(angles)
    y = (radius + eps[1]) * np.sin(angles)
    z = rng.uniform(size=n)
    points = np.stack((x, z, y), axis=1).astype(np.float32)
    return points, angles


def arclen_spearman(T: np.ndarray, src_angles: np.ndarray, tgt_angles: np.ndarray) -> float:
    """Spearman rank correlation between source and matched-target arclengths.

    For each source row i, pick the target column j* = argmax_j T[i,j], then
    compute Spearman rho between src_angles and tgt_angles[j*] over all i.
    Perfect identity matching => 1.0; reverse matching => -1.0.
    """
    assert T.shape[0] == src_angles.shape[0]
    assert T.shape[1] == tgt_angles.shape[0]
    matched = tgt_angles[np.argmax(T, axis=1)]
    # scipy.stats.spearmanr returns a SignificanceResult (statistic/pvalue) in
    # modern scipy; older versions exposed it via .correlation. Using .statistic
    # with an attribute fallback keeps both working.
    result = spearmanr(src_angles, matched)
    rho = getattr(result, "statistic", None)
    if rho is None:
        rho = result.correlation  # type: ignore[attr-defined]
    return float(np.asarray(rho, dtype=float).item())


def get_host_info() -> dict:
    """Return a dict describing the current host's GPU, torch, CUDA, CPU."""
    info: dict = {"gpu": "cpu", "torch": "unknown", "cuda": None}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            # torch.version is a runtime submodule that isn't declared in the
            # official type stubs; fetch via vars() to avoid a pyright
            # reportAttributeAccessIssue on the literal attribute access.
            _version_mod = vars(torch).get("version")
            info["cuda"] = getattr(_version_mod, "cuda", None) if _version_mod else None
    except ImportError:
        pass
    try:
        import platform
        info["cpu"] = platform.processor() or platform.machine()
        info["hostname"] = platform.node()
        info["python"] = platform.python_version()
    except Exception:
        pass
    return info


def build_record(track: str, solver: str, seed: int, subset: str) -> dict:
    """Build an initial JSON record skeleton following CONVENTIONS.md.

    The caller fills in dataset, hyperparams, metrics, and artifacts after
    running the solver; on failure the caller flips status to 'fail' and sets
    'error' to the exception string.
    """
    return {
        "track": track,
        "solver": solver,
        "solver_version": None,
        "seed": seed,
        "subset": subset,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": get_host_info(),
        "status": "ok",
        "error": None,
        "dataset": {},
        "hyperparams": {},
        "metrics": {"correctness": {}, "task": {}, "efficiency": {}, "stability": {}},
        "artifacts": {},
    }


def run_torchgw_landmark(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = 0,
    epsilon: float = 5e-3,
    M: int = 80,
    max_iter: int = 300,
    k: int = 5,
    n_landmarks: int = 50,
) -> dict:
    """Run torchgw sampled_gw with landmark distance mode and mixed precision.

    k=5 (sparse kNN graph) is required for manifold-structured data like the
    2D spiral: a denser graph (k>=7) creates shortcuts between coils and
    destroys geodesic structure, collapsing Spearman from ~1.0 to ~0.74.

    Returns a dict with:
        T (ndarray): (N, K) transport plan
        gw_cost (float): reported from solver log
        marginal_error (float): max|T@1 - 1/N|
        wall_s (float): wall clock of the solver call
        gpu_peak_gb (float or None): torch.cuda.max_memory_allocated in GB
        iterations (int): outer GW iterations from solver log
        hyperparams (dict): echo of the chosen hyperparameters
        solver_version (str): torchgw version string
    """
    import torch
    from torchgw import sampled_gw
    import torchgw as _torchgw

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.perf_counter()
    # sampled_gw has no typed log=True overload in torchgw stubs, so the
    # return type is inferred as Tensor rather than tuple[Tensor, dict].
    T, log = sampled_gw(  # type: ignore[misc]
        X, Y,
        distance_mode="landmark",
        mixed_precision=True,
        M=M,
        epsilon=epsilon,
        max_iter=max_iter,
        k=k,
        n_landmarks=n_landmarks,
        log=True,
        verbose=False,
    )
    log_d: dict = log if isinstance(log, dict) else {}  # type: ignore[arg-type]
    if use_cuda:
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0

    T_np = T.detach().cpu().numpy() if hasattr(T, "detach") else np.asarray(T)
    marginal_error = float(np.max(np.abs(T_np.sum(axis=1) - 1.0 / T_np.shape[0])))
    gpu_peak_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3) if use_cuda else None
    )

    return {
        "T": T_np,
        "gw_cost": float(log_d.get("gw_cost", float("nan"))),
        "marginal_error": marginal_error,
        "wall_s": wall_s,
        "gpu_peak_gb": gpu_peak_gb,
        "iterations": int(log_d.get("n_iter", log_d.get("iterations", 0))),
        "hyperparams": {
            "M": M,
            "epsilon": epsilon,
            "max_iter": max_iter,
            "k": k,
            "n_landmarks": n_landmarks,
            "distance_mode": "landmark",
            "mixed_precision": True,
        },
        "solver_version": f"torchgw=={getattr(_torchgw, '__version__', 'unknown')}",
    }


def run_pot_entropic(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int = 0,
    epsilon: float = 5e-3,
    max_iter: int = 500,
) -> dict:
    """Run POT's entropic Gromov-Wasserstein on CPU with squared-Euclidean costs.

    Returns the same dict shape as run_torchgw_landmark for apples-to-apples
    downstream comparison.
    """
    import ot
    import ot.gromov as otgw

    C1 = np.asarray(ot.dist(X, X, metric="sqeuclidean"), dtype=np.float64)
    C2 = np.asarray(ot.dist(Y, Y, metric="sqeuclidean"), dtype=np.float64)
    C1 /= (C1.max() + 1e-12)
    C2 /= (C2.max() + 1e-12)

    p = np.full(X.shape[0], 1.0 / X.shape[0], dtype=np.float64)
    q = np.full(Y.shape[0], 1.0 / Y.shape[0], dtype=np.float64)

    np.random.seed(seed)

    t0 = time.perf_counter()
    T_and_log = otgw.entropic_gromov_wasserstein(
        C1, C2, p, q,
        loss_fun="square_loss",
        epsilon=epsilon,
        max_iter=max_iter,
        tol=1e-9,
        log=True,
        verbose=False,
    )
    pot_log: dict[str, Any] = {}
    if isinstance(T_and_log, tuple):
        T, _raw_log = T_and_log
        if isinstance(_raw_log, dict):
            pot_log = _raw_log
    else:
        T = T_and_log
    wall_s = time.perf_counter() - t0

    T_np = np.asarray(T, dtype=np.float64)
    marginal_error = float(np.max(np.abs(T_np.sum(axis=1) - p)))
    gw_cost = float(pot_log.get("gw_dist", float("nan")))
    err_list: list = pot_log.get("err") or []
    # Use -1 when "err" key is absent so unknown iteration count is explicit.
    iterations = len(err_list) if err_list else -1

    return {
        "T": T_np,
        "gw_cost": gw_cost,
        "marginal_error": marginal_error,
        "wall_s": wall_s,
        "gpu_peak_gb": None,
        "iterations": iterations,
        "hyperparams": {
            "epsilon": epsilon,
            "max_iter": max_iter,
            "loss_fun": "square_loss",
        },
        "solver_version": f"pot=={getattr(ot, '__version__', 'unknown')}",
    }


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="C1 Foundation track: spiral -> swiss roll GW alignment")
    ap.add_argument("--solver", required=True,
                    choices=["torchgw-landmark", "pot-entropic"],
                    help="Which solver to run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True, help="Directory to write the JSON record into")
    ap.add_argument("--subset", default="full", choices=["small", "full"])
    ap.add_argument("--n-source", type=int, default=400)
    ap.add_argument("--n-target", type=int, default=500)
    args = ap.parse_args()

    rec = build_record(
        track="core/01_foundation",
        solver=args.solver,
        seed=args.seed,
        subset=args.subset,
    )
    rec["dataset"] = {
        "name": f"spiral_{args.n_source}_swissroll_{args.n_target}",
        "n_source": args.n_source,
        "n_target": args.n_target,
        "source_dim": 2,
        "target_dim": 3,
    }

    out_path = args.out / f"core_01_foundation__{args.solver}__seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        X, src_angles = sample_spiral(args.n_source, seed=args.seed)
        Y, tgt_angles = sample_swiss_roll(args.n_target, seed=args.seed + 1)

        if args.solver == "torchgw-landmark":
            result = run_torchgw_landmark(X, Y, seed=args.seed)
        elif args.solver == "pot-entropic":
            result = run_pot_entropic(X, Y, seed=args.seed)
        else:
            raise ValueError(f"unknown solver: {args.solver}")

        # Pull hyperparams + version into the record
        rec["hyperparams"] = result["hyperparams"]
        rec["solver_version"] = result["solver_version"]

        # Fill metrics sub-dicts
        rec["metrics"]["correctness"] = {
            "gw_cost": result["gw_cost"],
            "marginal_error": result["marginal_error"],
        }
        rec["metrics"]["task"] = {
            "spearman_arclen": arclen_spearman(result["T"], src_angles, tgt_angles),
        }
        rec["metrics"]["efficiency"] = {
            "wall_s": result["wall_s"],
            "gpu_peak_gb": result["gpu_peak_gb"],
            "iterations": result["iterations"],
        }
    except Exception as e:
        rec["status"] = "fail"
        rec["error"] = f"{type(e).__name__}: {e}"
        out_path.write_text(json.dumps(rec, indent=2))
        raise
    else:
        out_path.write_text(json.dumps(rec, indent=2))
        print(f"[C1] wrote {out_path}")


if __name__ == "__main__":
    main()
