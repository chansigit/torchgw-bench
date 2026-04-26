"""Held-out functional correlation + retrieval evaluation.

Given an alignment plan T (n_a × n_b) and held-out test contrasts on each
subject, predict B's contrasts from A's via T and measure (a) vertex-wise
correlation, (b) contrast-level retrieval accuracy.

Naming: file is `eval_brain.py` (not `eval.py`) to avoid shadowing Python's
builtin `eval` function — lesson from C7.
"""
from __future__ import annotations
import numpy as np


def _row_normalize(T: np.ndarray) -> np.ndarray:
    s = T.sum(axis=1, keepdims=True); s[s == 0] = 1.0
    return T / s


def eval_alignment(T: np.ndarray, F_test_a: np.ndarray, F_test_b: np.ndarray
                   ) -> dict:
    """T: (n_a, n_b) alignment plan; F_test_*: (n_v, n_test_contrasts)."""
    n_a, n_b = T.shape
    assert F_test_a.shape[0] == n_a, "F_test_a vertex count mismatch"
    assert F_test_b.shape[0] == n_b, "F_test_b vertex count mismatch"
    n_contrasts = F_test_a.shape[1]
    Tn = _row_normalize(T.T)
    F_pred_b = Tn @ F_test_a

    def _corr(x, y):
        xc = x - x.mean(); yc = y - y.mean()
        return float(np.dot(xc, yc) / (np.linalg.norm(xc) * np.linalg.norm(yc) + 1e-12))
    per_contrast_r = [_corr(F_pred_b[:, c], F_test_b[:, c]) for c in range(n_contrasts)]
    func_corr = float(np.mean(per_contrast_r))

    Fp = F_pred_b / (np.linalg.norm(F_pred_b, axis=0, keepdims=True) + 1e-12)
    Fb = F_test_b / (np.linalg.norm(F_test_b, axis=0, keepdims=True) + 1e-12)
    sim = Fp.T @ Fb
    ranks = np.argsort(-sim, axis=1)
    top1 = float(np.mean(ranks[:, 0] == np.arange(n_contrasts)))
    top5 = float(np.mean(np.any(ranks[:, :5] == np.arange(n_contrasts)[:, None],
                                axis=1)))
    return {
        "func_corr_holdout_mean": func_corr,
        "func_corr_holdout_std":  float(np.std(per_contrast_r)),
        "retrieval_top1":         top1,
        "retrieval_top5":         top5,
    }
