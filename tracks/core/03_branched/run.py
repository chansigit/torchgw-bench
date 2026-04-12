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

def _two_tail_directions(
    theta_tail_start: float, fork_angle: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return unit vectors for two tails opening symmetrically around the outward
    radial at θ=theta_tail_start.

    On a tight spiral, the tangent at the outer end is nearly perpendicular to
    the radial direction, so rotating ± half the fork angle *around the tangent*
    can send one tail back towards the origin. Using the radial (outward)
    direction as the symmetry axis guarantees both tails move strictly outward
    (r increases), giving a V that opens away from the spiral.
    """
    rx = float(np.cos(theta_tail_start))
    ry = float(np.sin(theta_tail_start))
    half = fork_angle / 2.0
    c_neg, s_neg = float(np.cos(-half)), float(np.sin(-half))
    c_pos, s_pos = float(np.cos(half)), float(np.sin(half))
    dA_x = rx * c_neg - ry * s_neg
    dA_y = rx * s_neg + ry * c_neg
    dB_x = rx * c_pos - ry * s_pos
    dB_y = rx * s_pos + ry * c_pos
    return (dA_x, dA_y), (dB_x, dB_y)


def sample_branched_spiral(
    n: int,
    branch_frac: float = 0.3,
    theta_tail_start: float = 9.0,
    tail_len: float = 0.6,
    fork_angle: float = np.pi / 3,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D spiral with a Y-fork: two tails diverging from the outer end (θ=9).

    Main arc: C1 spiral on θ ∈ [0, theta_tail_start].
    Tail A: straight segment along the local tangent at θ=theta_tail_start.
    Tail B: straight segment at angle `fork_angle` from tail A (same origin).

    Both tails have the same length (`tail_len`). The two branches together
    hold `branch_frac * n` points, split evenly. The inner end (θ=0) is a
    single spiral terminus (no fork), so the two ends of the manifold have
    very different local geometries — this is what breaks GW's reversal tie.

    Returns:
        points: (n, 2) float32
        angles: (n,) float64 — main: θ ∈ [0, theta_tail_start];
                tail A: θ = theta_tail_start + s;
                tail B: θ = theta_tail_start + tail_len + s (offset for
                        monotone ordering when sorting)
        labels: (n,) int — 0 = main, 1 = tail (either A or B)
    """
    rng = np.random.default_rng(seed)
    n_tail_total = int(round(n * branch_frac))
    n_main = n - n_tail_total
    n_A = n_tail_total // 2
    n_B = n_tail_total - n_A

    # Main spiral: same parameterisation as C1
    radius = np.linspace(0.3, 1.0, n_main)
    angles_main = np.linspace(0.0, theta_tail_start, n_main)
    eps = rng.normal(size=(2, n_main)) * noise
    x_main = (radius + eps[0]) * np.cos(angles_main)
    y_main = (radius + eps[1]) * np.sin(angles_main)

    # Fork origin (outer spiral endpoint)
    base_x = 1.0 * np.cos(theta_tail_start)
    base_y = 1.0 * np.sin(theta_tail_start)
    (dA_x, dA_y), (dB_x, dB_y) = _two_tail_directions(theta_tail_start, fork_angle)

    def _tail_points(n_points: int, dx: float, dy: float
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n_points <= 0:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        s = np.linspace(tail_len / n_points, tail_len, n_points)
        eps_t = rng.normal(size=(2, n_points)) * noise
        xs = base_x + s * dx + eps_t[0]
        ys = base_y + s * dy + eps_t[1]
        return xs, ys, s

    xA, yA, sA = _tail_points(n_A, dA_x, dA_y)
    xB, yB, sB = _tail_points(n_B, dB_x, dB_y)
    angles_A = theta_tail_start + sA
    angles_B = theta_tail_start + tail_len + sB  # shift so sort order is stable

    points = np.concatenate(
        (np.stack((x_main, y_main), axis=1),
         np.stack((xA, yA), axis=1),
         np.stack((xB, yB), axis=1)),
        axis=0,
    ).astype(np.float32)
    angles = np.concatenate((angles_main, angles_A, angles_B))
    labels = np.concatenate((
        np.zeros(n_main, dtype=np.int64),
        np.ones(n_A, dtype=np.int64),
        np.ones(n_B, dtype=np.int64),
    ))
    return points, angles, labels


def sample_branched_swiss_roll(
    n: int,
    branch_frac: float = 0.3,
    theta_tail_start: float = 9.0,
    tail_len: float = 0.6,
    fork_angle: float = np.pi / 3,
    noise: float = 0.05,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """3D Swiss roll with a Y-fork at the outer edge (θ=9).

    Same construction as sample_branched_spiral but embedded in 3D with an
    independent uniform z coordinate per point.

    Returns:
        points: (n, 3) float32 — layout (x, z, y) as in C1 swiss_roll
        angles, labels: same semantics as sample_branched_spiral
    """
    rng = np.random.default_rng(seed)
    n_tail_total = int(round(n * branch_frac))
    n_main = n - n_tail_total
    n_A = n_tail_total // 2
    n_B = n_tail_total - n_A

    # Main Swiss roll
    radius = np.linspace(0.3, 1.0, n_main)
    angles_main = np.linspace(0.0, theta_tail_start, n_main)
    eps = rng.normal(size=(2, n_main)) * noise
    x_main = (radius + eps[0]) * np.cos(angles_main)
    y_main = (radius + eps[1]) * np.sin(angles_main)
    z_main = rng.uniform(size=n_main)

    # Fork origin
    base_x = 1.0 * np.cos(theta_tail_start)
    base_y = 1.0 * np.sin(theta_tail_start)
    (dA_x, dA_y), (dB_x, dB_y) = _two_tail_directions(theta_tail_start, fork_angle)

    def _tail_points(n_points: int, dx: float, dy: float
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if n_points <= 0:
            return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
        s = np.linspace(tail_len / n_points, tail_len, n_points)
        eps_t = rng.normal(size=(2, n_points)) * noise
        xs = base_x + s * dx + eps_t[0]
        ys = base_y + s * dy + eps_t[1]
        zs = rng.uniform(size=n_points)
        return xs, ys, zs, s

    xA, yA, zA, sA = _tail_points(n_A, dA_x, dA_y)
    xB, yB, zB, sB = _tail_points(n_B, dB_x, dB_y)
    angles_A = theta_tail_start + sA
    angles_B = theta_tail_start + tail_len + sB

    pts_main = np.stack((x_main, z_main, y_main), axis=1)
    pts_A = np.stack((xA, zA, yA), axis=1)
    pts_B = np.stack((xB, zB, yB), axis=1)
    points = np.concatenate((pts_main, pts_A, pts_B), axis=0).astype(np.float32)
    angles = np.concatenate((angles_main, angles_A, angles_B))
    labels = np.concatenate((
        np.zeros(n_main, dtype=np.int64),
        np.ones(n_A, dtype=np.int64),
        np.ones(n_B, dtype=np.int64),
    ))
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
    ap.add_argument("--theta-tail-start", type=float, default=9.0)
    ap.add_argument("--tail-len", type=float, default=0.6)
    ap.add_argument("--fork-angle", type=float, default=float(np.pi / 3))
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
        "tail_len": args.tail_len,
        "fork_angle": args.fork_angle,
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
            fork_angle=args.fork_angle, seed=args.seed,
        )
        Y, tgt_angles, tgt_labels = sample_branched_swiss_roll(
            args.n_target, branch_frac=args.branch_frac,
            theta_tail_start=args.theta_tail_start, tail_len=args.tail_len,
            fork_angle=args.fork_angle, seed=args.seed + 1,
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
