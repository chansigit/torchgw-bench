"""Held-out functional correlation + retrieval evaluation.

Given an alignment plan T (n_a × n_b) and held-out test contrasts on each
subject, predict B's contrasts from A's via T and measure (a) vertex-wise
correlation, (b) contrast-level retrieval accuracy.

NaN handling: vol_to_surf produces NaN at cortical vertices that fall outside
the MNI152 volume (empty slice). We use nanmean-based Pearson correlation and
replace NaN vertices with 0 before matrix operations. This preserves signal
at valid vertices while avoiding NaN propagation.

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
    """T: (n_a, n_b) alignment plan; F_test_*: (n_v, n_test_contrasts).

    NaN values in F_test_* (from vol_to_surf missing-vertex warnings) are
    replaced with 0 before prediction, and Pearson r is computed over
    finite-valued vertices only.
    """
    n_a, n_b = T.shape
    assert F_test_a.shape[0] == n_a, "F_test_a vertex count mismatch"
    assert F_test_b.shape[0] == n_b, "F_test_b vertex count mismatch"
    n_contrasts = F_test_a.shape[1]
    Tn = _row_normalize(T.T)
    # Replace NaN with 0 before matrix multiply (missing vol_to_surf vertices)
    F_a_clean = np.nan_to_num(F_test_a, nan=0.0)
    F_b_clean = np.nan_to_num(F_test_b, nan=0.0)
    F_pred_b = Tn @ F_a_clean

    def _corr(x, y):
        """Pearson r; uses only vertices where both x and y are finite.

        Returns 0.0 when the prediction has zero variance (constant across
        vertices — happens with uniform transport plan), which is the correct
        interpretation: a constant prediction carries no information and is
        equivalent to chance-level alignment.
        """
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return float("nan")
        xm, ym = x[mask], y[mask]
        xc = xm - xm.mean(); yc = ym - ym.mean()
        denom = np.linalg.norm(xc) * np.linalg.norm(yc)
        if denom < 1e-12:
            return 0.0  # constant prediction → zero correlation (chance-level)
        return float(np.dot(xc, yc) / denom)

    per_contrast_r = [_corr(F_pred_b[:, c], F_b_clean[:, c])
                      for c in range(n_contrasts)]
    finite_r = [r for r in per_contrast_r if np.isfinite(r)]
    func_corr = float(np.mean(finite_r)) if finite_r else float("nan")

    norms_p = np.linalg.norm(F_pred_b, axis=0, keepdims=True)
    norms_b = np.linalg.norm(F_b_clean, axis=0, keepdims=True)
    Fp = F_pred_b / (norms_p + 1e-12)
    Fb = F_b_clean / (norms_b + 1e-12)
    sim = Fp.T @ Fb
    ranks = np.argsort(-sim, axis=1)
    top1 = float(np.mean(ranks[:, 0] == np.arange(n_contrasts)))
    top5 = float(np.mean(np.any(ranks[:, :5] == np.arange(n_contrasts)[:, None],
                                axis=1)))
    return {
        "func_corr_holdout_mean": func_corr,
        "func_corr_holdout_std":  float(np.std(finite_r)) if finite_r else float("nan"),
        "retrieval_top1":         top1,
        "retrieval_top5":         top5,
    }
