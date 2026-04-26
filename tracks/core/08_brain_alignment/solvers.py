"""Single-pair FGW dispatch for the four C8 solvers.

All solvers consume the same (C_a, C_b, C_lin) triple. Differences are in
the solver implementation only.

Probe findings (Step 0):
- FUGW model.pi   : torch.Tensor, shape (n, m)
- FUGW model.loss : dict with key 'total' holding a list of float losses
- FUGW model.loss_steps : list of iteration indices
- torchgw log    : dict with key 'gw_cost' (both balanced and unbalanced)
"""
from __future__ import annotations
import time
import numpy as np


def _uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=np.float64)


# ── POT entropic FGW ─────────────────────────────────────────────────

def _fgw_pot(C_a, C_b, C_lin, epsilon, fgw_alpha, seed):
    import ot
    import torch
    a = _uniform(C_a.shape[0])
    b = _uniform(C_b.shape[0])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Ca = torch.as_tensor(C_a, dtype=torch.float32, device=dev)
    Cb = torch.as_tensor(C_b, dtype=torch.float32, device=dev)
    Cl = torch.as_tensor(C_lin, dtype=torch.float32, device=dev)
    pa = torch.as_tensor(a, dtype=torch.float32, device=dev)
    pb = torch.as_tensor(b, dtype=torch.float32, device=dev)
    T, log = ot.gromov.entropic_fused_gromov_wasserstein(
        Cl, Ca, Cb, pa, pb,
        loss_fun="square_loss", alpha=fgw_alpha, epsilon=epsilon,
        log=True, max_iter=500,
    )
    obj = log.get("fgw_dist", log.get("loss", float("nan")))
    if hasattr(obj, "detach"):
        obj = obj.detach().cpu().item()
    return T.detach().cpu().numpy(), float(obj)


# ── torchgw (balanced or unbalanced) ─────────────────────────────────

def _fgw_torchgw(C_a, C_b, C_lin, epsilon, fgw_alpha, seed,
                 rho_a=None, rho_b=None):
    """torchgw-balanced if rho_{a,b} is None; torchgw-unbalanced otherwise."""
    import torch
    from torchgw import sampled_gw
    torch.manual_seed(seed)
    n, m = C_a.shape[0], C_b.shape[0]
    M_samples = max(min(n, 1000), 3 * n // 4)
    p = _uniform(n).astype(np.float32)
    q = _uniform(m).astype(np.float32)
    kwargs = dict(
        X_source=C_a, X_target=C_b, p=p, q=q,
        distance_mode="precomputed",
        dist_source=C_a.astype(np.float32),
        dist_target=C_b.astype(np.float32),
        fgw_alpha=fgw_alpha,
        C_linear=C_lin.astype(np.float32),
        mixed_precision=True,
        M=M_samples, epsilon=epsilon, max_iter=200,
        log=True, verbose=False,
    )
    if rho_a is not None:
        kwargs.update(semi_relaxed=True, rho_a=rho_a, rho_b=rho_b)
    T, log = sampled_gw(**kwargs)  # type: ignore[misc]
    # Both balanced and unbalanced return 'gw_cost' in the log dict
    obj = log.get("gw_cost", log.get("fgw_cost", float("nan")))
    return (T.detach().cpu().numpy() if hasattr(T, "detach") else np.asarray(T),
            float(obj))


# ── FUGW native ──────────────────────────────────────────────────────

def _fgw_fugw(C_a, C_b, C_lin, epsilon, fgw_alpha, seed,
              rho_a=1.0, rho_b=1.0):
    """FUGW package call.

    model.pi    : torch.Tensor (n, m) — the transport plan
    model.loss  : dict with 'total' key containing list of float losses
    model.loss_steps : list of iteration indices

    Note: FUGW only supports symmetric rho (single scalar). Raise if asymmetric.
    """
    import torch
    from fugw.mappings import FUGW
    if rho_a != rho_b:
        raise ValueError(
            f"FUGW package only supports symmetric rho; got rho_a={rho_a}, rho_b={rho_b}")
    torch.manual_seed(seed)
    n, m = C_a.shape[0], C_b.shape[0]
    p = _uniform(n).astype(np.float32)
    q = _uniform(m).astype(np.float32)
    # FUGW wants features as (n_features x n_vertices). Without explicit
    # features we can't fully encode C_lin into the FUGW formulation; for
    # this benchmark we use C_lin as a literal feature matrix by treating
    # its rows as per-target-vertex feature vectors.
    # source_features: (n_features=m, n_vertices=n) — use C_lin.T
    # target_features: (n_features=m, n_vertices=m) — use identity(m)
    # This is a coarse approximation; real downstream calls should pass
    # the actual contrast feature matrix directly.
    if C_lin.size > 0:
        F_src = C_lin.T.astype(np.float32)   # shape: (m, n)
        F_tgt = np.eye(C_lin.shape[1], dtype=np.float32)  # shape: (m, m)
    else:
        F_src = np.zeros((1, n), dtype=np.float32)
        F_tgt = np.zeros((1, m), dtype=np.float32)
    # Ensure feature dimensions match between source and target
    if F_src.shape[0] != F_tgt.shape[0]:
        d = max(F_src.shape[0], F_tgt.shape[0])
        if F_src.shape[0] < d:
            F_src = np.vstack([F_src,
                               np.zeros((d - F_src.shape[0], n), dtype=np.float32)])
        if F_tgt.shape[0] < d:
            F_tgt = np.vstack([F_tgt,
                               np.zeros((d - F_tgt.shape[0], m), dtype=np.float32)])
    model = FUGW(alpha=fgw_alpha, rho=float(rho_a), eps=epsilon,
                 reg_mode="joint", divergence="kl")
    model.fit(
        source_features=F_src, target_features=F_tgt,
        source_geometry=C_a.astype(np.float32),
        target_geometry=C_b.astype(np.float32),
        source_weights=p, target_weights=q,
        solver="mm", device="auto", verbose=False,
    )
    T = model.pi
    if hasattr(T, "detach"):
        T = T.detach().cpu().numpy()
    else:
        T = np.asarray(T)
    # model.loss is a dict: {'total': [...], 'wasserstein': [...], ...}
    loss_dict = getattr(model, "loss", None)
    if isinstance(loss_dict, dict) and "total" in loss_dict and loss_dict["total"]:
        obj = float(loss_dict["total"][-1])
    else:
        loss_steps = getattr(model, "loss_steps", None)
        obj = float(loss_steps[-1]) if loss_steps else float("nan")
    return T, obj


# ── Dispatch ─────────────────────────────────────────────────────────

_DISPATCH = {
    "pot-entropic-fgw":    lambda Ca, Cb, Cl, eps, alpha, seed, **kw:
                                _fgw_pot(Ca, Cb, Cl, eps, alpha, seed),
    "torchgw-balanced":    lambda Ca, Cb, Cl, eps, alpha, seed, **kw:
                                _fgw_torchgw(Ca, Cb, Cl, eps, alpha, seed),
    "torchgw-unbalanced":  lambda Ca, Cb, Cl, eps, alpha, seed,
                                  rho_a=1.0, rho_b=1.0, **kw:
                                _fgw_torchgw(Ca, Cb, Cl, eps, alpha, seed,
                                             rho_a=rho_a, rho_b=rho_b),
    "fugw-native":         lambda Ca, Cb, Cl, eps, alpha, seed,
                                  rho_a=1.0, rho_b=1.0, **kw:
                                _fgw_fugw(Ca, Cb, Cl, eps, alpha, seed,
                                          rho_a, rho_b),
}


def fgw_pair(solver: str, C_a: np.ndarray, C_b: np.ndarray, C_lin: np.ndarray,
             *, epsilon: float, fgw_alpha: float, seed: int,
             rho_a: float = 1.0, rho_b: float = 1.0) -> dict:
    """Compute FGW transport between a pair of cost matrices.

    Parameters
    ----------
    solver : str
        One of 'pot-entropic-fgw', 'torchgw-balanced',
        'torchgw-unbalanced', 'fugw-native'.
    C_a : np.ndarray, shape (n, n)
        Source intra-domain cost matrix (geodesic distances).
    C_b : np.ndarray, shape (m, m)
        Target intra-domain cost matrix.
    C_lin : np.ndarray, shape (n, m)
        Cross-domain linear cost matrix (e.g. feature dissimilarity).
    epsilon : float
        Entropic regularization strength.
    fgw_alpha : float
        FGW interpolation parameter (0 = pure Wasserstein, 1 = pure GW).
    seed : int
        Random seed for reproducibility.
    rho_a, rho_b : float
        Unbalanced marginal relaxation strengths (used by torchgw-unbalanced
        and fugw-native; ignored by pot-entropic-fgw and torchgw-balanced).

    Returns
    -------
    dict with keys:
        'T'             : np.ndarray (n, m) transport plan
        'fgw_objective' : float FGW objective value
        'wall_s'        : float wall-clock time in seconds
    """
    if solver not in _DISPATCH:
        raise ValueError(
            f"unknown solver {solver!r}; choose from {list(_DISPATCH)}")
    t0 = time.perf_counter()
    T, fgw_obj = _DISPATCH[solver](C_a, C_b, C_lin,
                                   epsilon, fgw_alpha, seed,
                                   rho_a=rho_a, rho_b=rho_b)
    return {"T": T, "fgw_objective": fgw_obj,
            "wall_s": time.perf_counter() - t0}
