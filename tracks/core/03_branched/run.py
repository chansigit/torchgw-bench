#!/usr/bin/env python
"""Track: core/03_branched — branched spiral → branched Swiss roll GW alignment.

Geometry is intrinsically non-symmetric: a side-branch at theta_branch breaks
the forward/reverse ambiguity that plagues pure GW on track 01.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import spearmanr


# ---- branched data generators -------------------------------------------

def sample_branched_spiral(
    n: int,
    branch_frac: float = 0.3,
    theta_branch: float = 6.0,
    branch_len: float = 0.4,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D Archimedean spiral with a perpendicular side-branch at theta_branch.

    Returns:
        points: (n, 2) float32
        angles: (n,) float64 — main-arc θ ∈ [0, 9] for main points;
                branch points get θ = theta_branch + s where s ∈ [0, branch_len]
        labels: (n,) int — 0 = main, 1 = branch
    """
    rng = np.random.default_rng(seed)
    n_branch = int(round(n * branch_frac))
    n_main = n - n_branch

    # Main spiral (same as C1)
    radius = np.linspace(0.3, 1.0, n_main)
    angles_main = np.linspace(0, 9, n_main)
    eps = rng.normal(size=(2, n_main)) * noise
    x_main = (radius + eps[0]) * np.cos(angles_main)
    y_main = (radius + eps[1]) * np.sin(angles_main)

    # Branch: starts at the spiral point with θ = theta_branch, extends along
    # the normal to the spiral tangent.
    # Spiral point at theta_branch:
    r_branch = 0.3 + (1.0 - 0.3) * (theta_branch / 9.0)
    base_x = r_branch * np.cos(theta_branch)
    base_y = r_branch * np.sin(theta_branch)
    # Tangent direction at theta_branch ≈ (-sin, cos); we use this as the
    # branch direction (perpendicular to the radial direction).
    dir_x = -np.sin(theta_branch)
    dir_y = np.cos(theta_branch)
    s = np.linspace(0.02, branch_len, n_branch) if n_branch > 0 else np.zeros(0)
    eps_b = rng.normal(size=(2, n_branch)) * noise
    x_branch = base_x + s * dir_x + eps_b[0]
    y_branch = base_y + s * dir_y + eps_b[1]
    angles_branch = theta_branch + s

    points = np.concatenate(
        (np.stack((x_main, y_main), axis=1),
         np.stack((x_branch, y_branch), axis=1)),
        axis=0,
    ).astype(np.float32)
    angles = np.concatenate((angles_main, angles_branch))
    labels = np.concatenate((np.zeros(n_main, dtype=np.int64),
                              np.ones(n_branch, dtype=np.int64)))
    return points, angles, labels


def sample_branched_swiss_roll(
    n: int,
    branch_frac: float = 0.3,
    theta_branch: float = 6.0,
    branch_len: float = 0.4,
    noise: float = 0.05,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D Swiss roll with a perpendicular branch at theta_branch.

    Returns:
        points: (n, 3) float32 — layout (x, z, y) as in C1 swiss_roll
        angles: (n,) float64
        labels: (n,) int — 0 = main, 1 = branch
    """
    rng = np.random.default_rng(seed)
    n_branch = int(round(n * branch_frac))
    n_main = n - n_branch

    # Main swiss roll (same layout as C1)
    radius = np.linspace(0.3, 1.0, n_main)
    angles_main = np.linspace(0, 9, n_main)
    eps = rng.normal(size=(2, n_main)) * noise
    x_main = (radius + eps[0]) * np.cos(angles_main)
    y_main = (radius + eps[1]) * np.sin(angles_main)
    z_main = rng.uniform(size=n_main)

    # Branch
    r_branch = 0.3 + (1.0 - 0.3) * (theta_branch / 9.0)
    base_x = r_branch * np.cos(theta_branch)
    base_y = r_branch * np.sin(theta_branch)
    dir_x = -np.sin(theta_branch)
    dir_y = np.cos(theta_branch)
    s = np.linspace(0.02, branch_len, n_branch) if n_branch > 0 else np.zeros(0)
    eps_b = rng.normal(size=(2, n_branch)) * noise
    x_branch = base_x + s * dir_x + eps_b[0]
    y_branch = base_y + s * dir_y + eps_b[1]
    z_branch = rng.uniform(size=n_branch)
    angles_branch = theta_branch + s

    # Note: C1 uses (x, z, y) layout
    pts_main = np.stack((x_main, z_main, y_main), axis=1)
    pts_branch = np.stack((x_branch, z_branch, y_branch), axis=1)
    points = np.concatenate((pts_main, pts_branch), axis=0).astype(np.float32)
    angles = np.concatenate((angles_main, angles_branch))
    labels = np.concatenate((np.zeros(n_main, dtype=np.int64),
                              np.ones(n_branch, dtype=np.int64)))
    return points, angles, labels


# ---- metrics ------------------------------------------------------------

def branch_accuracy(T: np.ndarray, src_labels: np.ndarray, tgt_labels: np.ndarray) -> float:
    """For each source row, pick target argmax; fraction of label matches."""
    assert T.shape[0] == src_labels.shape[0]
    assert T.shape[1] == tgt_labels.shape[0]
    matched = tgt_labels[np.argmax(T, axis=1)]
    return float(np.mean(src_labels == matched))


def main_arclen_spearman(
    T: np.ndarray,
    src_angles: np.ndarray,
    src_labels: np.ndarray,
    tgt_angles: np.ndarray,
    tgt_labels: np.ndarray,  # noqa: ARG001 — kept for symmetry with branch_accuracy
) -> float:
    """Signed Spearman rho on main-branch source points only (label == 0)."""
    del tgt_labels  # unused; accepted for interface symmetry
    main_mask = (src_labels == 0)
    if main_mask.sum() < 2:
        return float("nan")
    matched = tgt_angles[np.argmax(T, axis=1)]
    result = spearmanr(src_angles[main_mask], matched[main_mask])
    rho = getattr(result, "statistic", None)
    if rho is None:
        rho = result.correlation  # type: ignore[attr-defined]
    return float(np.asarray(rho, dtype=float).item())


# ---- host / record ------------------------------------------------------

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


# ---- solver wrapper (torchgw-landmark) ----------------------------------

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
    gpu_peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if use_cuda else None

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


# ---- main ---------------------------------------------------------------

def main() -> None:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="C3 Branched track: branched spiral → branched swiss roll")
    ap.add_argument("--solver", required=True, choices=["torchgw-landmark"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subset", default="full", choices=["small", "full"])
    ap.add_argument("--n-source", type=int, default=400)
    ap.add_argument("--n-target", type=int, default=500)
    ap.add_argument("--branch-frac", type=float, default=0.3)
    ap.add_argument("--theta-branch", type=float, default=6.0)
    ap.add_argument("--branch-len", type=float, default=0.4)
    args = ap.parse_args()

    rec = build_record(
        track="core/03_branched",
        solver=args.solver,
        seed=args.seed,
        subset=args.subset,
    )
    rec["dataset"] = {
        "name": f"branched_spiral_{args.n_source}_branched_swissroll_{args.n_target}",
        "n_source": args.n_source,
        "n_target": args.n_target,
        "source_dim": 2,
        "target_dim": 3,
        "branch_frac": args.branch_frac,
        "theta_branch": args.theta_branch,
        "branch_len": args.branch_len,
    }

    out_path = args.out / (
        f"core_03_branched__{args.solver}"
        f"__n{args.n_source}k{args.n_target}"
        f"__seed{args.seed}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        X, src_angles, src_labels = sample_branched_spiral(
            args.n_source, branch_frac=args.branch_frac,
            theta_branch=args.theta_branch, branch_len=args.branch_len,
            seed=args.seed,
        )
        Y, tgt_angles, tgt_labels = sample_branched_swiss_roll(
            args.n_target, branch_frac=args.branch_frac,
            theta_branch=args.theta_branch, branch_len=args.branch_len,
            seed=args.seed + 1,
        )

        if args.solver == "torchgw-landmark":
            result = run_torchgw_landmark(X, Y, seed=args.seed)
        else:
            raise ValueError(f"unknown solver: {args.solver}")

        rec["hyperparams"] = result["hyperparams"]
        rec["solver_version"] = result["solver_version"]
        rec["metrics"]["correctness"] = {
            "gw_cost": result["gw_cost"],
            "marginal_error": result["marginal_error"],
        }
        rec["metrics"]["task"] = {
            "branch_accuracy": branch_accuracy(result["T"], src_labels, tgt_labels),
            "main_arclen_spearman": main_arclen_spearman(
                result["T"], src_angles, src_labels, tgt_angles, tgt_labels,
            ),
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
        print(f"[C3] wrote {out_path}")


if __name__ == "__main__":
    main()
