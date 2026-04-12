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

def _spiral_tail_direction(theta_end: float, r_max: float = 1.0,
                            r_min: float = 0.3, theta_max: float = 9.0) -> tuple[float, float]:
    """Unit tangent to r(θ)·(cos θ, sin θ) at θ=theta_end, pointing outward in θ."""
    dr_dtheta = (r_max - r_min) / theta_max
    dx = dr_dtheta * np.cos(theta_end) - r_max * np.sin(theta_end)
    dy = dr_dtheta * np.sin(theta_end) + r_max * np.cos(theta_end)
    norm = float(np.sqrt(dx * dx + dy * dy))
    return float(dx / norm), float(dy / norm)


def sample_branched_spiral(
    n: int,
    branch_frac: float = 0.2,
    theta_tail_start: float = 9.0,
    tail_len: float = 0.8,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D spiral with a straight tail extending tangentially from its outer end.

    The main spiral is the C1 Archimedean spiral (θ ∈ [0, theta_tail_start]).
    The tail continues in the local-tangent direction at θ=theta_tail_start
    for a distance of `tail_len` units. This single-sided extension makes the
    manifold geometrically non-symmetric, breaking GW's forward/reverse tie.

    Returns:
        points: (n, 2) float32
        angles: (n,) float64 — main points use θ ∈ [0, theta_tail_start];
                tail points use θ = theta_tail_start + s, s ∈ (0, tail_len]
        labels: (n,) int — 0 = main, 1 = tail
    """
    rng = np.random.default_rng(seed)
    n_tail = int(round(n * branch_frac))
    n_main = n - n_tail

    # Main spiral: same parameterisation as C1
    radius = np.linspace(0.3, 1.0, n_main)
    angles_main = np.linspace(0.0, theta_tail_start, n_main)
    eps = rng.normal(size=(2, n_main)) * noise
    x_main = (radius + eps[0]) * np.cos(angles_main)
    y_main = (radius + eps[1]) * np.sin(angles_main)

    # Tail: line segment starting at the spiral's outer endpoint, pointing
    # along the local tangent at θ=theta_tail_start.
    base_x = 1.0 * np.cos(theta_tail_start)
    base_y = 1.0 * np.sin(theta_tail_start)
    dir_x, dir_y = _spiral_tail_direction(theta_tail_start)
    s = np.linspace(tail_len / max(n_tail, 1), tail_len, n_tail) if n_tail > 0 else np.zeros(0)
    eps_t = rng.normal(size=(2, n_tail)) * noise
    x_tail = base_x + s * dir_x + eps_t[0]
    y_tail = base_y + s * dir_y + eps_t[1]
    angles_tail = theta_tail_start + s

    points = np.concatenate(
        (np.stack((x_main, y_main), axis=1),
         np.stack((x_tail, y_tail), axis=1)),
        axis=0,
    ).astype(np.float32)
    angles = np.concatenate((angles_main, angles_tail))
    labels = np.concatenate((np.zeros(n_main, dtype=np.int64),
                              np.ones(n_tail, dtype=np.int64)))
    return points, angles, labels


def sample_branched_swiss_roll(
    n: int,
    branch_frac: float = 0.2,
    theta_tail_start: float = 9.0,
    tail_len: float = 0.8,
    noise: float = 0.05,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D Swiss roll with a tangential tail extending from its outer edge.

    Returns:
        points: (n, 3) float32 — layout (x, z, y) as in C1 swiss_roll
        angles: (n,) float64 — same semantics as sample_branched_spiral
        labels: (n,) int — 0 = main, 1 = tail
    """
    rng = np.random.default_rng(seed)
    n_tail = int(round(n * branch_frac))
    n_main = n - n_tail

    # Main Swiss roll (same layout as C1)
    radius = np.linspace(0.3, 1.0, n_main)
    angles_main = np.linspace(0.0, theta_tail_start, n_main)
    eps = rng.normal(size=(2, n_main)) * noise
    x_main = (radius + eps[0]) * np.cos(angles_main)
    y_main = (radius + eps[1]) * np.sin(angles_main)
    z_main = rng.uniform(size=n_main)

    # Tail: tangential extension at θ=theta_tail_start
    base_x = 1.0 * np.cos(theta_tail_start)
    base_y = 1.0 * np.sin(theta_tail_start)
    dir_x, dir_y = _spiral_tail_direction(theta_tail_start)
    s = np.linspace(tail_len / max(n_tail, 1), tail_len, n_tail) if n_tail > 0 else np.zeros(0)
    eps_t = rng.normal(size=(2, n_tail)) * noise
    x_tail = base_x + s * dir_x + eps_t[0]
    y_tail = base_y + s * dir_y + eps_t[1]
    z_tail = rng.uniform(size=n_tail)
    angles_tail = theta_tail_start + s

    # C1 layout: (x, z, y)
    pts_main = np.stack((x_main, z_main, y_main), axis=1)
    pts_tail = np.stack((x_tail, z_tail, y_tail), axis=1)
    points = np.concatenate((pts_main, pts_tail), axis=0).astype(np.float32)
    angles = np.concatenate((angles_main, angles_tail))
    labels = np.concatenate((np.zeros(n_main, dtype=np.int64),
                              np.ones(n_tail, dtype=np.int64)))
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
    ap.add_argument("--branch-frac", type=float, default=0.2)
    ap.add_argument("--theta-tail-start", type=float, default=9.0)
    ap.add_argument("--tail-len", type=float, default=0.8)
    args = ap.parse_args()

    rec = build_record(
        track="core/03_branched",
        solver=args.solver,
        seed=args.seed,
        subset=args.subset,
    )
    rec["dataset"] = {
        "name": f"tailed_spiral_{args.n_source}_tailed_swissroll_{args.n_target}",
        "n_source": args.n_source,
        "n_target": args.n_target,
        "source_dim": 2,
        "target_dim": 3,
        "branch_frac": args.branch_frac,
        "theta_tail_start": args.theta_tail_start,
        "tail_len": args.tail_len,
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
            theta_tail_start=args.theta_tail_start, tail_len=args.tail_len,
            seed=args.seed,
        )
        Y, tgt_angles, tgt_labels = sample_branched_swiss_roll(
            args.n_target, branch_frac=args.branch_frac,
            theta_tail_start=args.theta_tail_start, tail_len=args.tail_len,
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
