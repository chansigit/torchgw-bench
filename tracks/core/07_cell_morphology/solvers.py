"""Single-pair GW dispatch + CAJAL parallel batch path for the four C7 solvers.

The single-pair functions accept two pre-built intracell distance matrices
D1, D2 and return one scalar GW distance. The batch CAJAL path bypasses the
single-pair loop because CAJAL's value-add is its multiprocessing layer.
"""
from __future__ import annotations
import time
import numpy as np


def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=np.float64)


# ---- single-pair backends -----------------------------------------------

def _gw_pot_entropic(D1, D2, epsilon, seed):
    import ot, torch
    a, b = _uniform(D1.shape[0]), _uniform(D2.shape[0])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C1 = torch.as_tensor(D1, dtype=torch.float32, device=dev)
    C2 = torch.as_tensor(D2, dtype=torch.float32, device=dev)
    pa = torch.as_tensor(a, dtype=torch.float32, device=dev)
    pb = torch.as_tensor(b, dtype=torch.float32, device=dev)
    _T, log = ot.gromov.entropic_gromov_wasserstein(
        C1, C2, pa, pb, "square_loss",
        epsilon=epsilon, log=True, max_iter=500,
    )
    return float(log.get("gw_dist", log.get("loss", float("nan"))))


def _gw_pot_exact(D1, D2, seed):
    import ot, torch
    a, b = _uniform(D1.shape[0]), _uniform(D2.shape[0])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    C1 = torch.as_tensor(D1, dtype=torch.float32, device=dev)
    C2 = torch.as_tensor(D2, dtype=torch.float32, device=dev)
    pa = torch.as_tensor(a, dtype=torch.float32, device=dev)
    pb = torch.as_tensor(b, dtype=torch.float32, device=dev)
    _T, log = ot.gromov.gromov_wasserstein(
        C1, C2, pa, pb, "square_loss", log=True, max_iter=500,
    )
    return float(log.get("gw_dist", log.get("loss", float("nan"))))


def _gw_torchgw_precomputed(D1, D2, epsilon, M_samples, seed):
    import torch
    from torchgw import sampled_gw
    torch.manual_seed(seed)
    n, m = D1.shape[0], D2.shape[0]
    M = M_samples if M_samples is not None else max(min(n, 1000), 3 * n // 4)
    p = np.full(n, 1.0 / n, dtype=np.float64)
    q = np.full(m, 1.0 / m, dtype=np.float64)
    # X_source/X_target are required by the API but ignored in precomputed mode
    _T, log = sampled_gw(  # type: ignore[misc]
        X_source=D1, X_target=D2, p=p, q=q,
        distance_mode="precomputed",
        dist_source=D1.astype(np.float32), dist_target=D2.astype(np.float32),
        mixed_precision=True,
        M=M, epsilon=epsilon, max_iter=200,
        log=True, verbose=False,
    )
    return float(log.get("gw_cost", log.get("gw_dist", float("nan"))))


def _gw_cajal_native_pair(D1, D2, seed):
    """Single-pair via cajal.run_gw.gw — exact CG cython core, CPU."""
    from cajal.run_gw import gw
    a, b = _uniform(D1.shape[0]), _uniform(D2.shape[0])
    _T, dist = gw(D1.astype(np.float64), a, D2.astype(np.float64), b)
    return float(dist)


_DISPATCH = {
    "pot-entropic-gpu":    lambda D1, D2, epsilon, M, seed:
                                _gw_pot_entropic(D1, D2, epsilon, seed),
    "pot-exact-gpu":       lambda D1, D2, epsilon, M, seed:
                                _gw_pot_exact(D1, D2, seed),
    "torchgw-precomputed": _gw_torchgw_precomputed,
    "cajal-native":        lambda D1, D2, epsilon, M, seed:
                                _gw_cajal_native_pair(D1, D2, seed),
}


def gw_pair(solver: str, D1: np.ndarray, D2: np.ndarray, *,
            epsilon: float, M_samples: int | None, seed: int) -> dict:
    if solver not in _DISPATCH:
        raise ValueError(f"unknown solver {solver!r}")
    t0 = time.perf_counter()
    val = _DISPATCH[solver](D1, D2, epsilon, M_samples, seed)
    return {"gw": val, "wall_s": time.perf_counter() - t0}


# ---- CAJAL parallel batch path ------------------------------------------

def gw_full_matrix_cajal(D_list: list[np.ndarray], num_processes: int = 0) -> np.ndarray:
    """Hand the full list of D_i to CAJAL's multiprocessing pairwise routine
    so we measure CAJAL's native end-to-end speed (CPU pool across pairs)
    rather than the single-pair POT-via-cython cost."""
    from cajal.run_gw import gw_pairwise_parallel
    if num_processes <= 0:
        import os
        num_processes = max(1, (os.cpu_count() or 1) - 1)
    cells = [(D.astype(np.float64), _uniform(D.shape[0])) for D in D_list]
    M, _coupling = gw_pairwise_parallel(cells, num_processes=num_processes)
    return np.asarray(M, dtype=np.float64)
