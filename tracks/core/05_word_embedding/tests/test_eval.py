"""Unit tests for eval.py — barycentric projection + P@k retrieval."""
from __future__ import annotations

import numpy as np
import pytest

import word_eval  # type: ignore[import-not-found]  # noqa: E402 — sys.path + alias set by conftest.py


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

WORDS_SRC = ["a", "b", "c", "d"]
WORDS_TGT = ["A", "B", "C", "D"]
GOLD_DICT: dict[str, set[str]] = {
    "a": {"A"},
    "b": {"B"},
    "c": {"C"},
    "d": {"D"},
}

# Identity embeddings: each word is a one-hot vector.
V_EYE = np.eye(4, dtype=np.float64)


def _make_T_identity() -> np.ndarray:
    """Uniform diagonal transport plan: T[i,i] = 0.25, else 0."""
    return np.eye(4, dtype=np.float64) / 4.0


# ---------------------------------------------------------------------------
# Test 1: identity transport → P@1 = 1.0
# ---------------------------------------------------------------------------

def test_precision_at_k_identity():
    T = _make_T_identity()
    proj = word_eval.barycentric_project(T, V_EYE)

    # After barycentric projection with identity T each row reproduces V_EYE[i]
    np.testing.assert_allclose(proj, V_EYE, atol=1e-12)

    scores = word_eval.precision_at_k(
        proj, V_EYE, WORDS_SRC, WORDS_TGT, GOLD_DICT, ks=(1, 5)
    )
    assert scores[1] == pytest.approx(1.0), f"P@1 expected 1.0, got {scores[1]}"
    assert scores[5] == pytest.approx(1.0), f"P@5 expected 1.0, got {scores[5]}"


# ---------------------------------------------------------------------------
# Test 2: swap rows 0↔1 → P@1 = 0.5
# ---------------------------------------------------------------------------

def test_precision_at_k_swapped():
    T = _make_T_identity()
    T[[0, 1]] = T[[1, 0]]  # swap: query 'a' now maps to V_tgt[1]='B'

    proj = word_eval.barycentric_project(T, V_EYE)

    scores = word_eval.precision_at_k(
        proj, V_EYE, WORDS_SRC, WORDS_TGT, GOLD_DICT, ks=(1, 5)
    )
    # Queries a→B (miss), b→A (miss), c→C (hit), d→D (hit) → 2/4 = 0.5
    assert scores[1] == pytest.approx(0.5), f"P@1 expected 0.5, got {scores[1]}"
