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

def _spiral_tangent(theta_end: float, r_max: float = 1.0, r_min: float = 0.3,
                     theta_max: float = 9.0) -> tuple[float, float]:
    """Unit tangent to r(θ)·(cos θ, sin θ) at θ=theta_end, pointing outward in θ."""
    dr_dtheta = (r_max - r_min) / theta_max
    dx = dr_dtheta * np.cos(theta_end) - r_max * np.sin(theta_end)
    dy = dr_dtheta * np.sin(theta_end) + r_max * np.cos(theta_end)
    norm = float(np.sqrt(dx * dx + dy * dy))
    return float(dx / norm), float(dy / norm)


def spiral_arclen(
    theta: float | np.ndarray,
    r_min: float = 0.3, r_max: float = 1.0, theta_max: float = 9.0,
) -> np.ndarray:
    """Arc length along the Archimedean spiral r(θ) = r_min + b·θ from 0 to θ.

    Closed-form: ∫₀^θ √(r² + b²) dθ with r = r_min + b·θ. Substituting u = r,
    du = b dθ gives ∫√(u² + b²) du / b, a standard integral:
        F(u) = ½ [u·√(u²+b²) + b²·ln(u + √(u²+b²))]
        arclen(θ) = [F(r_min + b·θ) - F(r_min)] / b
    """
    theta_arr = np.asarray(theta, dtype=np.float64)
    b = (r_max - r_min) / theta_max
    a = r_min

    def _F(u: np.ndarray) -> np.ndarray:
        return 0.5 * (u * np.sqrt(u * u + b * b)
                      + b * b * np.log(u + np.sqrt(u * u + b * b)))

    return (_F(a + b * theta_arr) - _F(np.asarray(a, dtype=np.float64))) / b


def _asymmetric_tail_directions(
    theta_tail_start: float, tail2_angle: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return unit vectors for the two (asymmetric) tails.

    Tail 1 goes along the spiral's local tangent at θ=theta_tail_start — the
    "curve continuation" direction. Tail 2 is tail 1 rotated by `tail2_angle`
    towards the outward radial direction (so tail 2 splays away from the
    spiral body rather than curling back onto it).
    """
    tx, ty = _spiral_tangent(theta_tail_start)
    rx = float(np.cos(theta_tail_start))
    ry = float(np.sin(theta_tail_start))
    # Sign convention: rotate tangent by an amount whose sign points toward
    # the outward radial. If cross(tangent, radial) > 0, radial is CCW from
    # tangent, so rotate by +tail2_angle; else rotate by -tail2_angle.
    cross = tx * ry - ty * rx
    rot = tail2_angle if cross > 0 else -tail2_angle
    c = float(np.cos(rot))
    s = float(np.sin(rot))
    d2x = tx * c - ty * s
    d2y = tx * s + ty * c
    return (tx, ty), (d2x, d2y)


def _build_branched_manifold(
    n: int,
    *,
    embed_3d: bool,
    branch_frac: float,
    theta_tail_start: float,
    tail1_len: float,
    tail2_len: float,
    tail2_angle: float,
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared backbone for spiral (2D) and Swiss-roll (3D) Y-fork datasets.

    Returns `(points, arclens, labels)`. The `arclens` field is the
    **geodesic distance from the spiral's inner end (θ=0)** along the
    manifold: it walks the spiral first, then continues along whichever
    tail the point lies on. Because tail 2 is shorter, the tail-2 arclen
    range `[arclen(9), arclen(9)+tail2_len]` is strictly contained in the
    tail-1 range `[arclen(9), arclen(9)+tail1_len]` — so a tail-1 point
    with s > tail2_len has no tail-2 counterpart at the same arclen, which
    is the signal an FGW feature term can exploit.
    """
    rng = np.random.default_rng(seed)
    n_tail_total = int(round(n * branch_frac))
    n_main = n - n_tail_total

    # Allocate tail points proportionally to length for uniform linear density.
    total_tail_len = tail1_len + tail2_len
    n_t1 = int(round(n_tail_total * tail1_len / max(total_tail_len, 1e-9)))
    n_t2 = n_tail_total - n_t1

    # Main spiral
    radius = np.linspace(0.3, 1.0, n_main)
    thetas_main = np.linspace(0.0, theta_tail_start, n_main)
    eps = rng.normal(size=(2, n_main)) * noise
    x_main = (radius + eps[0]) * np.cos(thetas_main)
    y_main = (radius + eps[1]) * np.sin(thetas_main)

    # Fork origin (outer spiral endpoint)
    base_x = 1.0 * np.cos(theta_tail_start)
    base_y = 1.0 * np.sin(theta_tail_start)
    (d1x, d1y), (d2x, d2y) = _asymmetric_tail_directions(theta_tail_start, tail2_angle)

    def _tail_points(n_points: int, dx: float, dy: float, tlen: float
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n_points <= 0:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        s = np.linspace(tlen / n_points, tlen, n_points)
        eps_t = rng.normal(size=(2, n_points)) * noise
        xs = base_x + s * dx + eps_t[0]
        ys = base_y + s * dy + eps_t[1]
        return xs, ys, s

    x1, y1, s1 = _tail_points(n_t1, d1x, d1y, tail1_len)
    x2, y2, s2 = _tail_points(n_t2, d2x, d2y, tail2_len)

    # Geodesic arc lengths from θ=0 along the manifold
    main_arclens = spiral_arclen(thetas_main)
    fork_arclen = float(spiral_arclen(theta_tail_start).item())
    tail1_arclens = fork_arclen + s1
    tail2_arclens = fork_arclen + s2

    # Labels: main + tail1 = 0 (the "backbone" arc); tail2 = 1 (the true branch)
    labels = np.concatenate((
        np.zeros(n_main, dtype=np.int64),
        np.zeros(n_t1, dtype=np.int64),
        np.ones(n_t2, dtype=np.int64),
    ))

    if embed_3d:
        z_main = rng.uniform(size=n_main)
        z1 = rng.uniform(size=n_t1)
        z2 = rng.uniform(size=n_t2)
        # C1 swiss-roll layout is (x, z, y)
        pts_main = np.stack((x_main, z_main, y_main), axis=1)
        pts_t1 = np.stack((x1, z1, y1), axis=1)
        pts_t2 = np.stack((x2, z2, y2), axis=1)
        points = np.concatenate((pts_main, pts_t1, pts_t2), axis=0).astype(np.float32)
    else:
        points = np.concatenate(
            (np.stack((x_main, y_main), axis=1),
             np.stack((x1, y1), axis=1),
             np.stack((x2, y2), axis=1)),
            axis=0,
        ).astype(np.float32)

    arclens = np.concatenate((main_arclens, tail1_arclens, tail2_arclens))
    return points, arclens, labels


def sample_branched_spiral(
    n: int,
    branch_frac: float = 0.3,
    theta_tail_start: float = 9.0,
    tail1_len: float = 1.2,
    tail2_len: float = 0.6,
    tail2_angle: float = np.pi / 6,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D spiral with an asymmetric Y-fork at the outer end (θ=9).

    Tail 1 extends along the spiral's local tangent (the "curve continuation"
    direction) for `tail1_len` units. Tail 2 is a shorter branch, rotated
    `tail2_angle` away from tail 1 toward the outward radial, with length
    `tail2_len < tail1_len`. Point counts are allocated proportionally to
    length so linear density is uniform.

    Labels: main spiral + tail 1 = 0 (a continuous "backbone" arc),
            tail 2 = 1 (the off-axis branch).

    Returns:
        points: (n, 2) float32
        arclens: (n,) float64 — **geodesic distance from the spiral's inner
                end (θ=0)** along the manifold. For main points it's the
                analytical arc length of r(θ)=0.3+0.0778·θ. For tail 1 /
                tail 2 points at parameter s, it's arclen(9) + s.
        labels: (n,) int — 0 = main+tail1 backbone, 1 = tail2 branch
    """
    return _build_branched_manifold(
        n, embed_3d=False,
        branch_frac=branch_frac, theta_tail_start=theta_tail_start,
        tail1_len=tail1_len, tail2_len=tail2_len, tail2_angle=tail2_angle,
        noise=noise, seed=seed,
    )


def sample_branched_swiss_roll(
    n: int,
    branch_frac: float = 0.3,
    theta_tail_start: float = 9.0,
    tail1_len: float = 1.2,
    tail2_len: float = 0.6,
    tail2_angle: float = np.pi / 6,
    noise: float = 0.05,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D Swiss roll with an asymmetric Y-fork — same shape as the 2D spiral
    but with an independent uniform z coordinate per point.

    See `sample_branched_spiral` for label / arclens conventions. Returns
    (points, arclens, labels).
    """
    return _build_branched_manifold(
        n, embed_3d=True,
        branch_frac=branch_frac, theta_tail_start=theta_tail_start,
        tail1_len=tail1_len, tail2_len=tail2_len, tail2_angle=tail2_angle,
        noise=noise, seed=seed,
    )


# ---- metrics ------------------------------------------------------------

def branch_accuracy(T: np.ndarray, src_labels: np.ndarray, tgt_labels: np.ndarray) -> float:
    """For each source row, pick target argmax; fraction of label matches."""
    assert T.shape[0] == src_labels.shape[0]
    assert T.shape[1] == tgt_labels.shape[0]
    matched = tgt_labels[np.argmax(T, axis=1)]
    return float(np.mean(src_labels == matched))


def _signed_spearman_on_subset(
    T: np.ndarray,
    src_coord: np.ndarray,
    src_mask: np.ndarray,
    tgt_coord: np.ndarray,
) -> float:
    if src_mask.sum() < 2:
        return float("nan")
    matched = tgt_coord[np.argmax(T, axis=1)]
    result = spearmanr(src_coord[src_mask], matched[src_mask])
    rho = getattr(result, "statistic", None)
    if rho is None:
        rho = result.correlation  # type: ignore[attr-defined]
    return float(np.asarray(rho, dtype=float).item())


def main_arclen_spearman(
    T: np.ndarray,
    src_arclens: np.ndarray,
    src_labels: np.ndarray,
    tgt_arclens: np.ndarray,
    tgt_labels: np.ndarray,  # noqa: ARG001
) -> float:
    """Signed Spearman rho on the backbone (main spiral + tail 1, label==0),
    using geodesic distance from the spiral start as the coordinate.
    """
    del tgt_labels
    return _signed_spearman_on_subset(T, src_arclens, src_labels == 0, tgt_arclens)


def tail_arclen_spearman(
    T: np.ndarray,
    src_arclens: np.ndarray,
    src_labels: np.ndarray,
    tgt_arclens: np.ndarray,
    tgt_labels: np.ndarray,  # noqa: ARG001
) -> float:
    """Signed Spearman rho on the off-axis branch (tail 2, label==1), using
    geodesic distance as the coordinate.
    """
    del tgt_labels
    return _signed_spearman_on_subset(T, src_arclens, src_labels == 1, tgt_arclens)


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


# ---- FGW solver using geodesic-distance feature -------------------------

def _build_feature_cost(
    src_arclens: np.ndarray, tgt_arclens: np.ndarray,
) -> np.ndarray:
    """Normalised (n_src, n_tgt) squared-Euclidean cost matrix over 1D arclens."""
    import ot
    F_src = src_arclens.reshape(-1, 1).astype(np.float64)
    F_tgt = tgt_arclens.reshape(-1, 1).astype(np.float64)
    M = np.asarray(ot.dist(F_src, F_tgt, metric="sqeuclidean"), dtype=np.float64)
    M /= (M.max() + 1e-12)
    return M


def run_torchgw_fused(
    X: np.ndarray,
    Y: np.ndarray,
    src_arclens: np.ndarray,
    tgt_arclens: np.ndarray,
    seed: int = 0,
    epsilon: float = 5e-3,
    M_samples: int = 80,
    max_iter: int = 300,
    k: int = 5,
    n_landmarks: int = 50,
    fgw_alpha: float = 0.5,
) -> dict:
    """torchgw sampled_gw in FGW mode with geodesic-distance feature.

    Uses the same landmark distance mode and hyperparameters as
    run_torchgw_landmark, plus an inter-domain feature cost built from each
    point's geodesic distance from the spiral start. Because tail 2 arclens
    live in a strict subset of tail 1 arclens (tail 2 is shorter), FGW can
    use the feature term to reject the "tail swap" that pure GW is prone to.
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

    M_feat = _build_feature_cost(src_arclens, tgt_arclens)

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


# ---- main ---------------------------------------------------------------

def main() -> None:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="C3 Branched track: branched spiral → branched swiss roll")
    ap.add_argument("--solver", required=True,
                    choices=["torchgw-landmark", "torchgw-fused"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subset", default="full", choices=["small", "full"])
    ap.add_argument("--n-source", type=int, default=400)
    ap.add_argument("--n-target", type=int, default=500)
    ap.add_argument("--branch-frac", type=float, default=0.3)
    ap.add_argument("--theta-tail-start", type=float, default=9.0)
    ap.add_argument("--tail1-len", type=float, default=1.2,
                    help="Length of tail 1 (along spiral tangent)")
    ap.add_argument("--tail2-len", type=float, default=0.6,
                    help="Length of tail 2 (splayed branch); must be < tail1-len")
    ap.add_argument("--tail2-angle", type=float, default=float(np.pi / 6),
                    help="Angle (rad) between tail 2 and tail 1; default 30°")
    args = ap.parse_args()

    rec = build_record(
        track="core/03_branched",
        solver=args.solver,
        seed=args.seed,
        subset=args.subset,
    )
    rec["dataset"] = {
        "name": f"yfork_spiral_{args.n_source}_yfork_swissroll_{args.n_target}",
        "n_source": args.n_source,
        "n_target": args.n_target,
        "source_dim": 2,
        "target_dim": 3,
        "branch_frac": args.branch_frac,
        "theta_tail_start": args.theta_tail_start,
        "tail1_len": args.tail1_len,
        "tail2_len": args.tail2_len,
        "tail2_angle": args.tail2_angle,
    }

    out_path = args.out / (
        f"core_03_branched__{args.solver}"
        f"__n{args.n_source}k{args.n_target}"
        f"__seed{args.seed}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        X, src_arclens, src_labels = sample_branched_spiral(
            args.n_source, branch_frac=args.branch_frac,
            theta_tail_start=args.theta_tail_start,
            tail1_len=args.tail1_len, tail2_len=args.tail2_len,
            tail2_angle=args.tail2_angle, seed=args.seed,
        )
        Y, tgt_arclens, tgt_labels = sample_branched_swiss_roll(
            args.n_target, branch_frac=args.branch_frac,
            theta_tail_start=args.theta_tail_start,
            tail1_len=args.tail1_len, tail2_len=args.tail2_len,
            tail2_angle=args.tail2_angle, seed=args.seed + 1,
        )

        if args.solver == "torchgw-landmark":
            result = run_torchgw_landmark(X, Y, seed=args.seed)
        elif args.solver == "torchgw-fused":
            result = run_torchgw_fused(X, Y, src_arclens, tgt_arclens, seed=args.seed)
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
                result["T"], src_arclens, src_labels, tgt_arclens, tgt_labels,
            ),
            "tail_arclen_spearman": tail_arclen_spearman(
                result["T"], src_arclens, src_labels, tgt_arclens, tgt_labels,
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
