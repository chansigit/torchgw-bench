#!/usr/bin/env python
"""Track: core/02_foundation_fused — spiral → Swiss roll FGW alignment.

Uses Fused Gromov-Wasserstein with the arclength parameter θ as a per-point
feature. Breaks the forward/reverse symmetry of pure GW.

Self-contained; does not import from sibling tracks or from scripts/.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def sample_spiral(n: int, noise: float = 0.05, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.3, 1.0, n)
    angles = np.linspace(0, 9, n)
    eps = rng.normal(size=(2, n)) * noise
    x = (radius + eps[0]) * np.cos(angles)
    y = (radius + eps[1]) * np.sin(angles)
    points = np.stack((x, y), axis=1).astype(np.float32)
    return points, angles


def sample_swiss_roll(n: int, noise: float = 0.05, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
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
    """Signed Spearman rho (no abs) — FGW forces the forward solution."""
    assert T.shape[0] == src_angles.shape[0]
    assert T.shape[1] == tgt_angles.shape[0]
    matched = tgt_angles[np.argmax(T, axis=1)]
    result = spearmanr(src_angles, matched)
    rho = getattr(result, "statistic", None)
    if rho is None:
        rho = result.correlation  # type: ignore[attr-defined]
    return float(np.asarray(rho, dtype=float).item())


def get_host_info() -> dict:
    info: dict = {"gpu": "cpu", "torch": "unknown", "cuda": None}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
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


def pot_too_large(n_source: int, n_target: int, threshold: int = 5_000) -> bool:
    """POT's O(N²) cost matrices exceed memory guard when max(N,K) > threshold."""
    return max(n_source, n_target) > threshold


def _build_feature_cost(src_angles: np.ndarray, tgt_angles: np.ndarray) -> np.ndarray:
    """Build (n, k) squared-Euclidean feature cost matrix from 1D θ features, normalised."""
    import ot
    F_src = src_angles.reshape(-1, 1).astype(np.float64)
    F_tgt = tgt_angles.reshape(-1, 1).astype(np.float64)
    M = np.asarray(ot.dist(F_src, F_tgt, metric="sqeuclidean"), dtype=np.float64)
    M /= (M.max() + 1e-12)
    return M


def run_torchgw_fused(
    X: np.ndarray,
    Y: np.ndarray,
    src_angles: np.ndarray,
    tgt_angles: np.ndarray,
    seed: int = 0,
    epsilon: float = 5e-3,
    M_samples: int = 80,
    max_iter: int = 300,
    k: int = 5,
    n_landmarks: int = 50,
    fgw_alpha: float = 0.5,
) -> dict:
    """Run torchgw sampled_gw with fgw_alpha>0 and C_linear=feature cost matrix."""
    import torch
    from torchgw import sampled_gw
    import torchgw as _torchgw

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    torch.manual_seed(seed)
    np.random.seed(seed)

    M_feat = _build_feature_cost(src_angles, tgt_angles)

    t0 = time.perf_counter()
    T, log = sampled_gw(  # type: ignore[misc]
        X, Y,
        distance_mode="landmark",
        mixed_precision=True,
        M=M_samples,
        epsilon=epsilon,
        max_iter=max_iter,
        k=k,
        n_landmarks=n_landmarks,
        fgw_alpha=fgw_alpha,
        C_linear=M_feat,
        log=True,
        verbose=False,
    )
    log_d: dict = log if isinstance(log, dict) else {}  # type: ignore[arg-type]
    if use_cuda:
        torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0

    T_np = T.detach().cpu().numpy() if hasattr(T, "detach") else np.asarray(T)
    marginal_error = float(np.max(np.abs(T_np.sum(axis=1) - 1.0 / T_np.shape[0])))
    gpu_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if use_cuda else None

    return {
        "T": T_np,
        "gw_cost": float(log_d.get("gw_cost", float("nan"))),
        "marginal_error": marginal_error,
        "wall_s": wall_s,
        "gpu_peak_gb": gpu_peak_gb,
        "iterations": int(log_d.get("n_iter", log_d.get("iterations", 0))),
        "hyperparams": {
            "M_samples": M_samples,
            "epsilon": epsilon,
            "max_iter": max_iter,
            "k": k,
            "n_landmarks": n_landmarks,
            "fgw_alpha": fgw_alpha,
            "distance_mode": "landmark",
            "mixed_precision": True,
        },
        "solver_version": f"torchgw=={getattr(_torchgw, '__version__', 'unknown')}",
    }


def run_pot_fused(
    X: np.ndarray,
    Y: np.ndarray,
    src_angles: np.ndarray,
    tgt_angles: np.ndarray,
    seed: int = 0,
    epsilon: float = 5e-3,
    max_iter: int = 500,
    alpha: float = 0.5,
) -> dict:
    """Run POT entropic_fused_gromov_wasserstein on CPU."""
    import ot
    import ot.gromov as otgw

    C1 = np.asarray(ot.dist(X, X, metric="sqeuclidean"), dtype=np.float64)
    C2 = np.asarray(ot.dist(Y, Y, metric="sqeuclidean"), dtype=np.float64)
    C1 /= (C1.max() + 1e-12)
    C2 /= (C2.max() + 1e-12)

    M_feat = _build_feature_cost(src_angles, tgt_angles)

    p = np.full(X.shape[0], 1.0 / X.shape[0], dtype=np.float64)
    q = np.full(Y.shape[0], 1.0 / Y.shape[0], dtype=np.float64)

    np.random.seed(seed)

    t0 = time.perf_counter()
    T_and_log = otgw.entropic_fused_gromov_wasserstein(
        M_feat, C1, C2, p, q,
        loss_fun="square_loss",
        epsilon=epsilon,
        alpha=alpha,
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
    gw_cost = float(pot_log.get("fgw_dist", pot_log.get("gw_dist", float("nan"))))
    err_list: list = pot_log.get("err") or []
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
            "alpha": alpha,
            "loss_fun": "square_loss",
        },
        "solver_version": f"pot=={getattr(ot, '__version__', 'unknown')}",
    }


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="C2 Foundation-Fused track: spiral → swiss roll FGW")
    ap.add_argument("--solver", required=True,
                    choices=["torchgw-fused", "pot-fused"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subset", default="full", choices=["small", "full"])
    ap.add_argument("--n-source", type=int, default=400)
    ap.add_argument("--n-target", type=int, default=500)
    args = ap.parse_args()

    rec = build_record(
        track="core/02_foundation_fused",
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

    out_path = args.out / (
        f"core_02_foundation_fused__{args.solver}"
        f"__n{args.n_source}k{args.n_target}"
        f"__seed{args.seed}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.solver == "pot-fused" and pot_too_large(args.n_source, args.n_target):
        rec["status"] = "skip"
        rec["error"] = (
            f"skipped: POT O(N²) memory guard "
            f"(max(N,K)={max(args.n_source, args.n_target)} > 5000)"
        )
        out_path.write_text(json.dumps(rec, indent=2))
        print(f"[C2] skipped (POT memory guard) → {out_path}")
        return

    try:
        X, src_angles = sample_spiral(args.n_source, seed=args.seed)
        Y, tgt_angles = sample_swiss_roll(args.n_target, seed=args.seed + 1)

        if args.solver == "torchgw-fused":
            result = run_torchgw_fused(X, Y, src_angles, tgt_angles, seed=args.seed)
        elif args.solver == "pot-fused":
            result = run_pot_fused(X, Y, src_angles, tgt_angles, seed=args.seed)
        else:
            raise ValueError(f"unknown solver: {args.solver}")

        rec["hyperparams"] = result["hyperparams"]
        rec["solver_version"] = result["solver_version"]
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
        print(f"[C2] wrote {out_path}")


if __name__ == "__main__":
    main()
