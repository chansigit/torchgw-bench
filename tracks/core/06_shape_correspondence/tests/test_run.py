"""Unit tests for tracks/core/06_shape_correspondence/run.py."""
from __future__ import annotations

import numpy as np
import pytest

import run  # type: ignore[import-not-found]  # noqa: E402


def test_load_off_roundtrip(tmp_path):
    """Write a tiny OFF file and load it."""
    off = tmp_path / "tiny.off"
    off.write_text("OFF\n4 2 0\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 1 2\n3 0 1 3\n")
    V, F = run.load_off(off)
    assert V.shape == (4, 3)
    assert F.shape == (2, 3)
    assert F[0].tolist() == [0, 1, 2]


def test_subsample_pair_gt_consistency():
    """Subsampled GT indices must stay within [0, n_tgt) and nearest-
    neighbour fallback must produce valid indices."""
    rng = np.random.default_rng(0)
    V_src = rng.standard_normal((500, 3)).astype(np.float32)
    V_tgt = rng.standard_normal((600, 3)).astype(np.float32)
    gt_full = rng.integers(0, 600, size=500)
    V_src_sub, V_tgt_sub, gt_sub = run.subsample_pair(
        V_src, V_tgt, gt_full, n_src=100, n_tgt=120, seed=0,
    )
    assert V_src_sub.shape == (100, 3)
    assert V_tgt_sub.shape == (120, 3)
    assert gt_sub.shape == (100,)
    assert gt_sub.min() >= 0 and gt_sub.max() < 120


def test_knn_geodesic_matrix_symmetric_and_zero_diag():
    rng = np.random.default_rng(0)
    V = rng.standard_normal((50, 3)).astype(np.float32)
    D = run.knn_geodesic_matrix(V, k=6)
    assert D.shape == (50, 50)
    # Diagonal should be 0
    assert np.allclose(np.diag(D), 0.0, atol=1e-6)
    # Symmetric
    assert np.allclose(D, D.T, atol=1e-6)
    # All finite (kNN with k=6 on 50 random points is very connected)
    assert np.all(np.isfinite(D))


def test_geodesic_error_perfect_prediction():
    """Identity permutation + identity T gives zero error."""
    n = 30
    # Build a geodesic matrix with known structure
    V = np.linspace(0, 1, n).reshape(-1, 1).astype(np.float32)
    V = np.hstack([V, np.zeros((n, 2), dtype=np.float32)])
    D = run.knn_geodesic_matrix(V, k=4)
    T = np.eye(n) / n
    gt = np.arange(n)
    e = run.geodesic_error(T, D, gt)
    assert e["mean_err_absolute"] == 0.0
    assert e["mean_err_normalised"] == 0.0


def test_match_accuracy_curve_perfect():
    n = 20
    V = np.linspace(0, 1, n).reshape(-1, 1).astype(np.float32)
    V = np.hstack([V, np.zeros((n, 2), dtype=np.float32)])
    D = run.knn_geodesic_matrix(V, k=4)
    T = np.eye(n) / n
    gt = np.arange(n)
    curve = run.match_accuracy_curve(T, D, gt, thresholds=(0.01, 0.1, 0.5))
    for t, frac in curve:
        assert frac == 1.0


def test_match_accuracy_curve_random_below_identity():
    n = 50
    V = np.linspace(0, 1, n).reshape(-1, 1).astype(np.float32)
    V = np.hstack([V, np.zeros((n, 2), dtype=np.float32)])
    D = run.knn_geodesic_matrix(V, k=6)
    # Random T (uniform rows) => random argmax
    rng = np.random.default_rng(0)
    T = rng.random((n, n))
    gt = np.arange(n)
    curve = run.match_accuracy_curve(T, D, gt, thresholds=(0.01, 0.25))
    # Very unlikely that a random T gives 100% accuracy at tau=0.01
    assert curve[0][1] < 0.5


EXPECTED_KEYS = {"T", "gw_cost", "marginal_error", "wall_s",
                  "gpu_peak_gb", "iterations", "hyperparams", "solver_version"}


@pytest.fixture(scope="module")
def small_pair():
    """Synthetic small 3D cloud pair for solver smoke tests."""
    rng = np.random.default_rng(0)
    V_src = rng.standard_normal((60, 3)).astype(np.float32)
    V_tgt = rng.standard_normal((80, 3)).astype(np.float32)
    return V_src, V_tgt


def test_run_torchgw_landmark_smoke(small_pair):
    V_src, V_tgt = small_pair
    out = run.run_torchgw_landmark(V_src, V_tgt, seed=0,
                                     max_iter=30, M_samples=30, n_landmarks=20)
    assert set(out.keys()) >= EXPECTED_KEYS
    assert out["T"].shape == (60, 80)
    assert out["hyperparams"]["distance_mode"] == "landmark"
    assert out["hyperparams"]["fgw_alpha"] == 1.0


def test_run_torchgw_precomputed_smoke(small_pair):
    V_src, V_tgt = small_pair
    out = run.run_torchgw_precomputed(V_src, V_tgt, seed=0,
                                        max_iter=30, M_samples=30)
    assert set(out.keys()) >= EXPECTED_KEYS
    assert out["T"].shape == (60, 80)
    assert out["hyperparams"]["distance_mode"] == "precomputed"


def test_run_pot_entropic_gpu_smoke(small_pair):
    V_src, V_tgt = small_pair
    out = run.run_pot_entropic_gpu(V_src, V_tgt, seed=0, max_iter=50)
    assert set(out.keys()) >= EXPECTED_KEYS
    assert out["T"].shape == (60, 80)
    assert out["hyperparams"]["algorithm"] == "entropic"
