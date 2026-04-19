from __future__ import annotations

"""C1 point-cloud scale track — correspondence and geometry evaluation metrics."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Transport-plan metrics
# ---------------------------------------------------------------------------

def correspondence_accuracy(T: np.ndarray) -> float:
    """Fraction of rows where argmax equals the row index.

    GT correspondence is identity: ``source[i] <-> target[i]``.

    Parameters
    ----------
    T:
        Cost / transport plan matrix, shape ``(N, N)``.

    Returns
    -------
    float in [0, 1].
    """
    n = T.shape[0]
    return float((np.argmax(T, axis=1) == np.arange(n)).mean())


def correspondence_recall_at_k(T: np.ndarray, k: int = 5) -> float:
    """Fraction of rows where the row index appears in the top-k of T[i, :].

    Parameters
    ----------
    T:
        Cost / transport plan matrix, shape ``(N, N)``.
    k:
        Number of top entries to consider per row.

    Returns
    -------
    float in [0, 1].
    """
    n = T.shape[0]
    # argsort descending: take the last k indices of ascending sort
    top_k = np.argsort(T, axis=1)[:, -k:]          # (N, k)
    gt = np.arange(n)[:, None]                       # (N, 1)
    hits = (top_k == gt).any(axis=1)
    return float(hits.mean())


# ---------------------------------------------------------------------------
# Geometry metrics
# ---------------------------------------------------------------------------

def barycentric_project(T: np.ndarray, V_tgt: np.ndarray) -> np.ndarray:
    """Barycentric projection of source points onto target via plan T.

    ``proj[i] = (sum_j T[i,j] * V_tgt[j]) / (sum_j T[i,j])``

    Parameters
    ----------
    T:
        Transport plan or soft-assignment matrix, shape ``(N_src, N_tgt)``.
    V_tgt:
        Target point cloud, shape ``(N_tgt, 3)``.

    Returns
    -------
    proj : np.ndarray, shape (N_src, 3)
    """
    row_sum = T.sum(axis=1, keepdims=True).clip(min=1e-30)
    return (T / row_sum) @ V_tgt


def chamfer_distance(projected: np.ndarray, target: np.ndarray) -> float:
    """Symmetric Chamfer distance between two point clouds.

    ``CD = mean_i(min_j ||projected[i] - target[j]||²)``
    ``   + mean_j(min_i ||target[j] - projected[i]||²)``

    Uses ``sklearn.neighbors.NearestNeighbors`` (KD-tree) for O(N log N)
    queries per direction.

    Parameters
    ----------
    projected:
        Shape ``(N, 3)``.
    target:
        Shape ``(M, 3)``.

    Returns
    -------
    float >= 0.
    """
    nn_tgt = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(target)
    dist_p2t, _ = nn_tgt.kneighbors(projected)   # (N, 1)
    d_forward = float((dist_p2t ** 2).mean())

    nn_src = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(projected)
    dist_t2p, _ = nn_src.kneighbors(target)       # (M, 1)
    d_backward = float((dist_t2p ** 2).mean())

    return d_forward + d_backward
