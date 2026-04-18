"""Evaluation utilities for the word-embedding alignment track.

Metrics:
  - barycentric_project  : project source queries into target space via T
  - precision_at_k       : P@k using cosine nearest-neighbour retrieval
  - precision_at_k_csls  : P@k using CSLS-smoothed cosine scores
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Barycentric projection
# ---------------------------------------------------------------------------

def barycentric_project(T: np.ndarray, V_tgt: np.ndarray) -> np.ndarray:
    """Project source embeddings into target space via transport plan T.

    Parameters
    ----------
    T     : (N_src, N_tgt) transport plan (rows need not sum to 1).
    V_tgt : (N_tgt, dim)   target embeddings.

    Returns
    -------
    proj  : (N_src, dim)
    """
    row_sums = T.sum(axis=1, keepdims=True).clip(min=1e-30)
    return (T / row_sums) @ V_tgt


# ---------------------------------------------------------------------------
# Cosine helpers
# ---------------------------------------------------------------------------

def _l2_norm(X: np.ndarray) -> np.ndarray:
    """Return row-wise L2 norms, shape (N,)."""
    return np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-30)


def _cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return (M, N) cosine similarity matrix between A (M,d) and B (N,d)."""
    A_n = A / _l2_norm(A)
    B_n = B / _l2_norm(B)
    return A_n @ B_n.T


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return (M, k) indices of top-k scores per row (sorted desc)."""
    k = min(k, scores.shape[1])
    # argpartition gives an unordered top-k; sort within the partition
    part = np.argpartition(scores, -k, axis=1)[:, -k:]
    # gather those scores and sort each row descending
    row_idx = np.arange(scores.shape[0])[:, None]
    top_scores = scores[row_idx, part]
    order = np.argsort(-top_scores, axis=1)
    return part[row_idx, order]


# ---------------------------------------------------------------------------
# P@k — cosine NN
# ---------------------------------------------------------------------------

def precision_at_k(
    proj: np.ndarray,
    V_tgt: np.ndarray,
    words_src: list[str],
    words_tgt: list[str],
    gold_dict: dict[str, set[str]],
    ks: tuple[int, ...] = (1, 5),
) -> dict[int, float]:
    """Compute P@k via cosine nearest-neighbour retrieval.

    Parameters
    ----------
    proj      : (N_src, dim) barycentric projection of source words.
    V_tgt     : (N_tgt, dim) target embeddings.
    words_src : list of N_src source word strings.
    words_tgt : list of N_tgt target word strings.
    gold_dict : mapping  src_word -> set of acceptable target words.
    ks        : tuple of k values to evaluate.

    Returns
    -------
    {k: mean hit rate}
    """
    src_idx: dict[str, int] = {w: i for i, w in enumerate(words_src)}

    # Collect evaluable queries
    query_rows: list[int] = []
    query_golds: list[set[str]] = []
    for w_src, gold_tgts in gold_dict.items():
        if w_src in src_idx:
            query_rows.append(src_idx[w_src])
            query_golds.append(gold_tgts)

    if not query_rows:
        return {k: 0.0 for k in ks}

    Q = proj[query_rows]  # (M, dim)
    cos_sim = _cosine_matrix(Q, V_tgt)  # (M, N_tgt)

    max_k = max(ks)
    top_indices = _topk_indices(cos_sim, max_k)  # (M, max_k)

    results: dict[int, float] = {}
    for k in ks:
        hits = 0
        for m, gold_tgts in enumerate(query_golds):
            retrieved = {words_tgt[j] for j in top_indices[m, :k]}
            if retrieved & gold_tgts:
                hits += 1
        results[k] = hits / len(query_golds)
    return results


# ---------------------------------------------------------------------------
# P@k — CSLS
# ---------------------------------------------------------------------------

def precision_at_k_csls(
    proj: np.ndarray,
    V_tgt: np.ndarray,
    words_src: list[str],
    words_tgt: list[str],
    gold_dict: dict[str, set[str]],
    ks: tuple[int, ...] = (1, 5),
    k_csls: int = 10,
) -> dict[int, float]:
    """Compute P@k using CSLS-smoothed scores.

    CSLS(x, y) = 2·cos(x,y) − r(x) − r(y)

    where r(x) = mean cosine of x to its k_csls nearest neighbours in V_tgt,
    and   r(y) = mean cosine of y to its k_csls nearest neighbours in proj
                 (approximated by the full query matrix for efficiency).

    Parameters mirror :func:`precision_at_k`; `k_csls` controls neighbourhood
    size for the CSLS correction.
    """
    src_idx: dict[str, int] = {w: i for i, w in enumerate(words_src)}

    # Collect evaluable queries
    query_rows: list[int] = []
    query_golds: list[set[str]] = []
    for w_src, gold_tgts in gold_dict.items():
        if w_src in src_idx:
            query_rows.append(src_idx[w_src])
            query_golds.append(gold_tgts)

    if not query_rows:
        return {k: 0.0 for k in ks}

    Q = proj[query_rows]  # (M, dim)
    cos_sim = _cosine_matrix(Q, V_tgt)  # (M, N_tgt)

    # r(x): mean of top-k_csls cosines per query row
    k_x = min(k_csls, V_tgt.shape[0])
    top_x = np.partition(cos_sim, -k_x, axis=1)[:, -k_x:]  # (M, k_x)
    r_x = top_x.mean(axis=1, keepdims=True)                 # (M, 1)

    # r(y): mean of top-k_csls cosines per target word (over all queries)
    # Shape: (N_tgt, M) then partition over M dimension
    cos_sim_T = cos_sim.T  # (N_tgt, M)
    k_y = min(k_csls, Q.shape[0])
    top_y = np.partition(cos_sim_T, -k_y, axis=1)[:, -k_y:]  # (N_tgt, k_y)
    r_y = top_y.mean(axis=1)                                   # (N_tgt,)

    csls_scores = 2 * cos_sim - r_x - r_y[np.newaxis, :]  # (M, N_tgt)

    max_k = max(ks)
    top_indices = _topk_indices(csls_scores, max_k)  # (M, max_k)

    results: dict[int, float] = {}
    for k in ks:
        hits = 0
        for m, gold_tgts in enumerate(query_golds):
            retrieved = {words_tgt[j] for j in top_indices[m, :k]}
            if retrieved & gold_tgts:
                hits += 1
        results[k] = hits / len(query_golds)
    return results
