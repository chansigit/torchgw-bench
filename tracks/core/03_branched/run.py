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


# ---- FGW solvers --------------------------------------------------------
#
# Six FGW variants are benchmarked. Every run uses the same geodesic-arclen
# feature cost matrix M; they differ in how the structural distances C1, C2
# and the transport plan are computed:
#
#   torchgw-landmark     —  landmark-approximated geodesic distances on GPU
#   torchgw-dijkstra     —  exact geodesic via Dijkstra on kNN graph, GPU
#   torchgw-precomputed  —  raw dense Euclidean cost matrices (no graph), GPU
#   pot-entropic         —  POT entropic FGW (Sinkhorn-based), CPU
#   pot-exact            —  POT exact FGW (conditional-gradient), CPU
#   pot-bapg             —  POT Bregman alternating projected gradient, CPU
#
# All three POT variants share the pot_too_large memory guard.


class _RSSSampler:
    """Polling sampler that records peak process RSS during a code block.

    Used to report CPU memory peak for each solver, alongside
    torch.cuda.max_memory_allocated for GPU peak. Usage::

        sampler = _RSSSampler()
        with sampler:
            ... solver call ...
        peak_bytes = sampler.peak
    """
    def __init__(self, interval_s: float = 0.05):
        import psutil, threading
        self._psutil = psutil
        self._threading = threading
        self._interval = interval_s
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread = None
        self.peak = 0

    def __enter__(self):
        self.peak = int(self._proc.memory_info().rss)
        self._stop.clear()
        self._thread = self._threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        import time as _time
        while not self._stop.is_set():
            try:
                rss = int(self._proc.memory_info().rss)
            except Exception:
                rss = self.peak
            if rss > self.peak:
                self.peak = rss
            _time.sleep(self._interval)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def pot_too_large(n_source: int, n_target: int, threshold: int = 5_000) -> bool:
    """POT FGW variants all build dense O(N^2)/O(K^2) cost matrices. Above
    max(N, K) = 5000 the memory and wall time blow up; main() emits a
    status='skip' record instead of attempting the run.
    """
    return max(n_source, n_target) > threshold


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


def _reset_tracking(seed: int):
    """Reset CUDA peak stats and seed RNGs. Returns (torch, use_cuda)."""
    import torch
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    torch.manual_seed(seed)
    np.random.seed(seed)
    return torch, use_cuda


def _build_metrics(wall_preprocess: float, wall_solve: float,
                    ram_peak_bytes: int, use_cuda: bool) -> dict:
    """Build a consistent metric block shared by every solver wrapper.

    Every solver reports:
      wall_s_preprocess  — distance-matrix / feature build (0 for solvers
                            that cannot separate preprocessing from the
                            main call, e.g. landmark/dijkstra torchgw).
      wall_s_solve       — the transport-plan optimisation call itself.
      wall_s_total       — preprocess + solve, end-to-end wall clock.
      gpu_peak_gb        — torch.cuda.max_memory_allocated (None on CPU).
      ram_peak_gb        — process RSS peak during the call, captured by
                            the _RSSSampler thread.
    """
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


def _finalize_torchgw(T, log, meta: dict, hyperparams: dict) -> dict:
    log_d: dict = log if isinstance(log, dict) else {}  # type: ignore[arg-type]
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
    # Back-compat: many downstream scripts read "wall_s"
    rec["wall_s"] = meta["wall_s_total"]
    return rec


def run_torchgw_landmark(
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
    min_iter_before_converge: int | None = None,
    tol: float | None = None,
) -> dict:
    """torchgw.sampled_gw in landmark distance mode, FGW with arclen feature.

    Landmark distance computation happens inside sampled_gw, so it cannot
    be separated from solve time; wall_s_preprocess = 0 and the whole
    call falls in wall_s_solve.
    """
    from torchgw import sampled_gw
    torch, use_cuda = _reset_tracking(seed)
    with _RSSSampler() as sampler:
        t_prep_start = time.perf_counter()
        M_feat = _build_feature_cost(src_arclens, tgt_arclens)
        t_prep = time.perf_counter() - t_prep_start  # tiny feature cost
        t0 = time.perf_counter()
        _extra = {}
        if min_iter_before_converge is not None:
            _extra["min_iter_before_converge"] = min_iter_before_converge
        if tol is not None:
            _extra["tol"] = tol
        T, log = sampled_gw(  # type: ignore[misc]
            X, Y,
            distance_mode="landmark", mixed_precision=True,
            M=M_samples, epsilon=epsilon, max_iter=max_iter,
            k=k, n_landmarks=n_landmarks,
            fgw_alpha=fgw_alpha, C_linear=M_feat,
            log=True, verbose=False, **_extra,
        )
        if use_cuda:
            torch.cuda.synchronize()
        t_solve = time.perf_counter() - t0

    meta = _build_metrics(t_prep, t_solve, sampler.peak, use_cuda)
    return _finalize_torchgw(T, log, meta, {
        "M_samples": M_samples, "epsilon": epsilon, "max_iter": max_iter,
        "k": k, "n_landmarks": n_landmarks, "fgw_alpha": fgw_alpha,
        "distance_mode": "landmark", "mixed_precision": True,
    })


def run_torchgw_dijkstra(
    X: np.ndarray,
    Y: np.ndarray,
    src_arclens: np.ndarray,
    tgt_arclens: np.ndarray,
    seed: int = 0,
    epsilon: float = 5e-3,
    M_samples: int = 80,
    max_iter: int = 300,
    k: int = 5,
    fgw_alpha: float = 0.5,
    min_iter_before_converge: int | None = None,
    tol: float | None = None,
) -> dict:
    """torchgw.sampled_gw in dijkstra distance mode, FGW with arclen feature.

    Structural distances are exact kNN-graph geodesics computed inside
    sampled_gw; preprocessing cannot be separated out.
    """
    from torchgw import sampled_gw
    torch, use_cuda = _reset_tracking(seed)
    with _RSSSampler() as sampler:
        t_prep_start = time.perf_counter()
        M_feat = _build_feature_cost(src_arclens, tgt_arclens)
        t_prep = time.perf_counter() - t_prep_start
        t0 = time.perf_counter()
        _extra = {}
        if min_iter_before_converge is not None:
            _extra["min_iter_before_converge"] = min_iter_before_converge
        if tol is not None:
            _extra["tol"] = tol
        T, log = sampled_gw(  # type: ignore[misc]
            X, Y,
            distance_mode="dijkstra", mixed_precision=True,
            M=M_samples, epsilon=epsilon, max_iter=max_iter,
            k=k,
            fgw_alpha=fgw_alpha, C_linear=M_feat,
            log=True, verbose=False, **_extra,
        )
        if use_cuda:
            torch.cuda.synchronize()
        t_solve = time.perf_counter() - t0

    meta = _build_metrics(t_prep, t_solve, sampler.peak, use_cuda)
    return _finalize_torchgw(T, log, meta, {
        "M_samples": M_samples, "epsilon": epsilon, "max_iter": max_iter,
        "k": k, "fgw_alpha": fgw_alpha,
        "distance_mode": "dijkstra", "mixed_precision": True,
    })


def run_torchgw_precomputed(
    X: np.ndarray,
    Y: np.ndarray,
    src_arclens: np.ndarray,
    tgt_arclens: np.ndarray,
    seed: int = 0,
    epsilon: float = 5e-3,
    M_samples: int = 80,
    max_iter: int = 300,
    fgw_alpha: float = 0.5,
    min_iter_before_converge: int | None = None,
    tol: float | None = None,
) -> dict:
    """torchgw.sampled_gw with precomputed dense Euclidean distance matrices.

    Preprocessing (the dense cdist) is timed separately so the caller can
    see how much of the wall goes into one-off distance construction vs.
    the repeatable solve call.
    """
    from torchgw import sampled_gw
    from scipy.spatial.distance import cdist
    torch, use_cuda = _reset_tracking(seed)

    with _RSSSampler() as sampler:
        t_prep_start = time.perf_counter()
        M_feat = _build_feature_cost(src_arclens, tgt_arclens)
        dist_source = cdist(X, X, metric="euclidean").astype(np.float32)
        dist_target = cdist(Y, Y, metric="euclidean").astype(np.float32)
        dist_source /= (dist_source.max() + 1e-12)
        dist_target /= (dist_target.max() + 1e-12)
        n_src, n_tgt = X.shape[0], Y.shape[0]
        p = np.full(n_src, 1.0 / n_src, dtype=np.float64)
        q = np.full(n_tgt, 1.0 / n_tgt, dtype=np.float64)
        t_prep = time.perf_counter() - t_prep_start

        t0 = time.perf_counter()
        _extra = {}
        if min_iter_before_converge is not None:
            _extra["min_iter_before_converge"] = min_iter_before_converge
        if tol is not None:
            _extra["tol"] = tol
        T, log = sampled_gw(  # type: ignore[misc]
            X_source=X, X_target=Y, p=p, q=q,
            distance_mode="precomputed",
            dist_source=dist_source, dist_target=dist_target,
            mixed_precision=True,
            M=M_samples, epsilon=epsilon, max_iter=max_iter,
            fgw_alpha=fgw_alpha, C_linear=M_feat,
            log=True, verbose=False, **_extra,
        )
        if use_cuda:
            torch.cuda.synchronize()
        t_solve = time.perf_counter() - t0

    meta = _build_metrics(t_prep, t_solve, sampler.peak, use_cuda)
    return _finalize_torchgw(T, log, meta, {
        "M_samples": M_samples, "epsilon": epsilon, "max_iter": max_iter,
        "fgw_alpha": fgw_alpha,
        "distance_mode": "precomputed", "mixed_precision": True,
    })


def _pot_common_setup_cpu(X: np.ndarray, Y: np.ndarray,
                            src_arclens: np.ndarray, tgt_arclens: np.ndarray,
                            seed: int):
    """Build (C1, C2, M_feat, p, q) as float64 numpy arrays for POT CPU."""
    import ot
    C1 = np.asarray(ot.dist(X, X, metric="sqeuclidean"), dtype=np.float64)
    C2 = np.asarray(ot.dist(Y, Y, metric="sqeuclidean"), dtype=np.float64)
    C1 /= (C1.max() + 1e-12)
    C2 /= (C2.max() + 1e-12)
    M_feat = _build_feature_cost(src_arclens, tgt_arclens)
    p = np.full(X.shape[0], 1.0 / X.shape[0], dtype=np.float64)
    q = np.full(Y.shape[0], 1.0 / Y.shape[0], dtype=np.float64)
    np.random.seed(seed)
    return C1, C2, M_feat, p, q


def _pot_common_setup_gpu(X: np.ndarray, Y: np.ndarray,
                            src_arclens: np.ndarray, tgt_arclens: np.ndarray,
                            seed: int, dtype: str = "float32"):
    """Build (C1, C2, M_feat, p, q) as torch tensors on GPU for POT's
    torch backend. POT auto-dispatches to GPU when tensors are on CUDA.

    dtype: "float32" (default) or "float64". BAPG's Bregman projections
    can blow up numerically with small epsilon at float32; float64 matches
    the CPU (numpy) code path and keeps the solver stable at extra memory
    and compute cost.
    """
    import torch
    device = torch.device("cuda")
    tdtype = torch.float64 if dtype == "float64" else torch.float32
    X_t = torch.as_tensor(X, device=device, dtype=tdtype)
    Y_t = torch.as_tensor(Y, device=device, dtype=tdtype)
    C1 = torch.cdist(X_t, X_t) ** 2
    C2 = torch.cdist(Y_t, Y_t) ** 2
    C1 /= (C1.max() + 1e-12)
    C2 /= (C2.max() + 1e-12)
    F_src = torch.as_tensor(src_arclens, device=device, dtype=tdtype).unsqueeze(1)
    F_tgt = torch.as_tensor(tgt_arclens, device=device, dtype=tdtype).unsqueeze(1)
    M_feat = torch.cdist(F_src, F_tgt) ** 2
    M_feat /= (M_feat.max() + 1e-12)
    p = torch.full((X.shape[0],), 1.0 / X.shape[0], device=device, dtype=tdtype)
    q = torch.full((Y.shape[0],), 1.0 / Y.shape[0], device=device, dtype=tdtype)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return C1, C2, M_feat, p, q


def _finalize_pot(T_and_log, meta: dict, hyperparams: dict) -> dict:
    import torch
    from typing import Any
    import ot
    pot_log: dict[str, Any] = {}
    if isinstance(T_and_log, tuple):
        T, _raw_log = T_and_log
        if isinstance(_raw_log, dict):
            pot_log = _raw_log
    else:
        T = T_and_log
    if hasattr(T, "detach"):
        T_np = T.detach().cpu().numpy().astype(np.float64)
    else:
        T_np = np.asarray(T, dtype=np.float64)
    p_uniform = 1.0 / T_np.shape[0]
    marginal_error = float(np.max(np.abs(T_np.sum(axis=1) - p_uniform)))

    def _as_float(v):
        if hasattr(v, "detach"):
            return float(v.detach().cpu().item())
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    gw_cost = _as_float(
        pot_log.get("fgw_dist", pot_log.get("gw_dist", float("nan")))
    )
    err_list: list = pot_log.get("err") or []
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
    rec["wall_s"] = meta["wall_s_total"]  # back-compat
    return rec


def _run_pot_variant(
    X, Y, src_arclens, tgt_arclens, seed, backend: str,
    algo_fn, algo_kwargs, hyperparams, gpu_dtype: str = "float32",
):
    """Shared driver for the six POT variants. `backend` is 'cpu' or 'gpu'.
    `algo_fn` is the POT FGW function; `algo_kwargs` goes through.
    `gpu_dtype` controls tensor precision for the GPU backend (default
    float32; BAPG-GPU uses float64 for numerical stability).
    """
    assert backend in ("cpu", "gpu")
    import torch
    use_cuda = (backend == "gpu") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    with _RSSSampler() as sampler:
        t_prep_start = time.perf_counter()
        if backend == "gpu":
            C1, C2, M_feat, p, q = _pot_common_setup_gpu(
                X, Y, src_arclens, tgt_arclens, seed, dtype=gpu_dtype)
        else:
            C1, C2, M_feat, p, q = _pot_common_setup_cpu(
                X, Y, src_arclens, tgt_arclens, seed)
        if use_cuda:
            torch.cuda.synchronize()
        t_prep = time.perf_counter() - t_prep_start

        t0 = time.perf_counter()
        T_and_log = algo_fn(M_feat, C1, C2, p, q, **algo_kwargs)
        if use_cuda:
            torch.cuda.synchronize()
        t_solve = time.perf_counter() - t0

    meta = _build_metrics(t_prep, t_solve, sampler.peak, use_cuda)
    return _finalize_pot(T_and_log, meta,
                          {**hyperparams, "backend": backend})


def _make_pot_entropic(backend: str):
    import ot.gromov as otgw
    def fn(X, Y, src_arclens, tgt_arclens, seed=0,
            epsilon=5e-3, max_iter=50, alpha=0.5, tol=1e-9):
        return _run_pot_variant(
            X, Y, src_arclens, tgt_arclens, seed, backend,
            otgw.entropic_fused_gromov_wasserstein,
            dict(loss_fun="square_loss", epsilon=epsilon, alpha=alpha,
                  max_iter=max_iter, tol=tol, log=True, verbose=False),
            {"epsilon": epsilon, "max_iter": max_iter, "alpha": alpha,
             "tol": tol, "loss_fun": "square_loss", "algorithm": "entropic"},
        )
    return fn


def _make_pot_exact(backend: str):
    import ot.gromov as otgw
    def fn(X, Y, src_arclens, tgt_arclens, seed=0,
            max_iter=500, alpha=0.5, tol=1e-6):
        return _run_pot_variant(
            X, Y, src_arclens, tgt_arclens, seed, backend,
            otgw.fused_gromov_wasserstein,
            dict(loss_fun="square_loss", alpha=alpha,
                  max_iter=max_iter, tol_rel=tol, tol_abs=tol, log=True),
            {"max_iter": max_iter, "alpha": alpha, "tol": tol,
             "loss_fun": "square_loss", "algorithm": "exact-CG"},
        )
    return fn


def _make_pot_bapg(backend: str):
    import ot.gromov as otgw
    def fn(X, Y, src_arclens, tgt_arclens, seed=0,
            epsilon=5e-3, max_iter=50, alpha=0.5, tol=1e-6):
        # BAPG's Bregman projection is numerically fragile at small
        # epsilon; use float64 on GPU to match the CPU code path.
        gpu_dtype = "float64" if backend == "gpu" else "float32"
        return _run_pot_variant(
            X, Y, src_arclens, tgt_arclens, seed, backend,
            otgw.BAPG_fused_gromov_wasserstein,
            dict(loss_fun="square_loss", epsilon=epsilon, alpha=alpha,
                  max_iter=max_iter, tol=tol, log=True, verbose=False),
            {"epsilon": epsilon, "max_iter": max_iter, "alpha": alpha,
             "tol": tol, "loss_fun": "square_loss", "algorithm": "BAPG",
             "gpu_dtype": gpu_dtype},
            gpu_dtype=gpu_dtype,
        )
    return fn


# Six POT variants: CPU and GPU for each of entropic / exact / bapg.
run_pot_entropic     = _make_pot_entropic("cpu")
run_pot_exact        = _make_pot_exact("cpu")
run_pot_bapg         = _make_pot_bapg("cpu")
run_pot_entropic_gpu = _make_pot_entropic("gpu")
run_pot_exact_gpu    = _make_pot_exact("gpu")
run_pot_bapg_gpu     = _make_pot_bapg("gpu")


# (BAPG default max_iter=50 is documented in _make_pot_bapg.)


# ---- main ---------------------------------------------------------------

def main() -> None:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="C3 Branched track: branched spiral → branched swiss roll")
    ap.add_argument("--solver", required=True, choices=[
        "torchgw-landmark", "torchgw-dijkstra", "torchgw-precomputed",
        "pot-entropic", "pot-exact", "pot-bapg",
        "pot-entropic-gpu", "pot-exact-gpu", "pot-bapg-gpu",
    ])
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
    ap.add_argument("--max-iter", type=int, default=None,
                    help="Override solver max_iter (anytime Pareto sweep).")
    ap.add_argument("--force-full", action="store_true",
                    help="Disable early stop: run exactly max_iter iterations "
                         "(torchgw: min_iter_before_converge=max_iter, tol=0; "
                         "POT: tol=0).")
    ap.add_argument("--tag", type=str, default=None,
                    help="Optional tag appended to output filename "
                         "(e.g. 'iter50' to separate anytime sweep runs).")
    args = ap.parse_args()

    rec = build_record(
        track="core/03_branched",
        solver=args.solver,
        seed=args.seed,
        subset=args.subset,
    )
    rec["dataset"] = {
        "name": f"yfork_swissroll_{args.n_source}_yfork_spiral_{args.n_target}",
        "n_source": args.n_source,
        "n_target": args.n_target,
        "source_dim": 3,  # swiss roll (with Y-fork) is now the source
        "target_dim": 2,  # spiral (with Y-fork) is the target
        "branch_frac": args.branch_frac,
        "theta_tail_start": args.theta_tail_start,
        "tail1_len": args.tail1_len,
        "tail2_len": args.tail2_len,
        "tail2_angle": args.tail2_angle,
    }

    _tag_part = f"__{args.tag}" if args.tag else ""
    out_path = args.out / (
        f"core_03_branched__{args.solver}"
        f"__n{args.n_source}k{args.n_target}"
        f"__seed{args.seed}{_tag_part}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # POT CPU memory guard — only the CPU POT variants hit host-RAM limits.
    # POT GPU variants are capped instead by GPU memory (~80GB on H100)
    # which is big enough for our scales.
    is_pot_cpu = args.solver in ("pot-entropic", "pot-exact", "pot-bapg")
    if is_pot_cpu and pot_too_large(args.n_source, args.n_target):
        rec["status"] = "skip"
        rec["error"] = (
            f"skipped: POT O(N^2) memory guard "
            f"(max(N,K)={max(args.n_source, args.n_target)} > 5000)"
        )
        out_path.write_text(json.dumps(rec, indent=2))
        print(f"[C3] skipped (POT memory guard) → {out_path}")
        return

    try:
        # Source is the 3D Swiss roll with Y-fork; target is the 2D spiral
        # with Y-fork. Flipping the dim direction (dimensionality reduction
        # framing) is narratively tighter and preserves the asymmetric
        # geometry needed to break GW's orientation ambiguity.
        X, src_arclens, src_labels = sample_branched_swiss_roll(
            args.n_source, branch_frac=args.branch_frac,
            theta_tail_start=args.theta_tail_start,
            tail1_len=args.tail1_len, tail2_len=args.tail2_len,
            tail2_angle=args.tail2_angle, seed=args.seed,
        )
        Y, tgt_arclens, tgt_labels = sample_branched_spiral(
            args.n_target, branch_frac=args.branch_frac,
            theta_tail_start=args.theta_tail_start,
            tail1_len=args.tail1_len, tail2_len=args.tail2_len,
            tail2_angle=args.tail2_angle, seed=args.seed + 1,
        )

        solver_fns = {
            "torchgw-landmark":     run_torchgw_landmark,
            "torchgw-dijkstra":     run_torchgw_dijkstra,
            "torchgw-precomputed":  run_torchgw_precomputed,
            "pot-entropic":         run_pot_entropic,
            "pot-exact":            run_pot_exact,
            "pot-bapg":             run_pot_bapg,
            "pot-entropic-gpu":     run_pot_entropic_gpu,
            "pot-exact-gpu":        run_pot_exact_gpu,
            "pot-bapg-gpu":         run_pot_bapg_gpu,
        }
        fn = solver_fns.get(args.solver)
        if fn is None:
            raise ValueError(f"unknown solver: {args.solver}")

        # Anytime-sweep kwargs: --max-iter overrides per-solver default;
        # --force-full disables early-stop paths.
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
        result = fn(X, Y, src_arclens, tgt_arclens,
                     seed=args.seed, **extra_kwargs)

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
            "wall_s":            result["wall_s_total"],  # back-compat alias
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
        print(f"[C3] wrote {out_path}")


if __name__ == "__main__":
    main()
