"""Unit tests for tracks/core/03_branched/run.py."""
from __future__ import annotations

import numpy as np
import pytest

import run  # type: ignore[import-not-found]  # noqa: E402 — sys.path set by conftest.py


# ---- data generators ----------------------------------------------------

def test_sample_branched_spiral_shape_and_labels():
    pts, angles, labels = run.sample_branched_spiral(n=100, branch_frac=0.3, seed=0)
    assert pts.shape == (100, 2)
    assert angles.shape == (100,)
    assert labels.shape == (100,)
    assert set(np.unique(labels).tolist()) == {0, 1}
    # Only tail 2 carries label 1. With branch_frac=0.3 and tail1_len=1.2,
    # tail2_len=0.6 (default), tail 2 gets 0.6/(1.2+0.6) = 1/3 of 30 points ≈ 10.
    assert 5 <= int(labels.sum()) <= 15


def test_sample_branched_swiss_roll_shape_and_labels():
    pts, angles, labels = run.sample_branched_swiss_roll(n=100, branch_frac=0.3, seed=1)
    assert pts.shape == (100, 3)
    assert angles.shape == (100,)
    assert labels.shape == (100,)
    assert set(np.unique(labels).tolist()) == {0, 1}
    assert 5 <= int(labels.sum()) <= 15


def test_asymmetric_yfork_tail1_longer_than_tail2():
    """Tail 1 (along tangent, label 0) spatially reaches further than tail 2."""
    pts, _angles, labels = run.sample_branched_spiral(
        n=400, branch_frac=0.3,
        tail1_len=1.2, tail2_len=0.6, tail2_angle=np.pi / 6,
        noise=0.0, seed=0,
    )
    fork_base = np.array([np.cos(9.0), np.sin(9.0)], dtype=np.float32)
    # Distance from fork base to tail-2 points (label==1)
    d2 = np.linalg.norm(pts[labels == 1] - fork_base, axis=1).max()
    # Tail 1 lives in label==0 (mixed with the main spiral); its farthest
    # point from the fork base should exceed tail1_len-ε.
    d1 = np.linalg.norm(pts[labels == 0] - fork_base, axis=1).max()
    assert d1 > d2 + 0.4  # tail1_len - tail2_len = 0.6, allow slack
    # Tail 2 reach should be ~ tail2_len
    assert 0.4 < d2 < 0.8


def test_main_and_tail_label_counts_sum_to_n():
    pts, _a, labels = run.sample_branched_spiral(n=500, branch_frac=0.2, seed=0)
    assert (labels == 0).sum() + (labels == 1).sum() == 500


# ---- metrics ------------------------------------------------------------

def test_branch_accuracy_perfect_identity():
    """When source and target have matching labels in same order, identity T gives 1.0."""
    n = 50
    src_labels = np.array([0] * 35 + [1] * 15)
    tgt_labels = np.array([0] * 35 + [1] * 15)
    T = np.eye(n) / n
    assert pytest.approx(run.branch_accuracy(T, src_labels, tgt_labels), abs=1e-9) == 1.0


def test_tail_arclen_spearman_perfect_identity():
    """Identity T on tail-2 points should give +1."""
    n = 30
    src_angles = np.concatenate((np.linspace(0, 9, 20), np.linspace(0.1, 0.6, 10)))
    tgt_angles = np.concatenate((np.linspace(0, 9, 20), np.linspace(0.1, 0.6, 10)))
    src_labels = np.array([0] * 20 + [1] * 10)
    tgt_labels = np.array([0] * 20 + [1] * 10)
    T = np.eye(n) / n
    rho = run.tail_arclen_spearman(T, src_angles, src_labels, tgt_angles, tgt_labels)
    assert pytest.approx(rho, abs=1e-6) == 1.0


def test_tail_arclen_spearman_reverse_gives_negative():
    """Reversing the tail-to-tail mapping should give signed -1."""
    n = 30
    src_angles = np.concatenate((np.linspace(0, 9, 20), np.linspace(0.1, 0.6, 10)))
    tgt_angles = np.concatenate((np.linspace(0, 9, 20), np.linspace(0.1, 0.6, 10)))
    src_labels = np.array([0] * 20 + [1] * 10)
    tgt_labels = np.array([0] * 20 + [1] * 10)
    T = np.zeros((n, n))
    # Main identity
    for i in range(20):
        T[i, i] = 1.0 / n
    # Tail reverse: src row 20+i → tgt col (29 - i) = 29 - i (mirror within tail)
    for i in range(10):
        T[20 + i, 29 - i] = 1.0 / n
    rho = run.tail_arclen_spearman(T, src_angles, src_labels, tgt_angles, tgt_labels)
    assert pytest.approx(rho, abs=1e-6) == -1.0


def test_branch_accuracy_all_mismatched():
    """If labels are completely flipped and T is identity, accuracy is 0.0."""
    src_labels = np.array([0, 0, 1, 1])
    tgt_labels = np.array([1, 1, 0, 0])
    T = np.eye(4) / 4
    assert run.branch_accuracy(T, src_labels, tgt_labels) == 0.0


def test_main_arclen_spearman_perfect():
    """Identity T on main branch points gives +1.0 (signed)."""
    n = 50
    src_angles = np.concatenate((np.linspace(0, 9, 35), np.array([6.1] * 15)))
    tgt_angles = np.concatenate((np.linspace(0, 9, 35), np.array([6.1] * 15)))
    src_labels = np.array([0] * 35 + [1] * 15)
    tgt_labels = np.array([0] * 35 + [1] * 15)
    T = np.eye(n) / n
    rho = run.main_arclen_spearman(T, src_angles, src_labels, tgt_angles, tgt_labels)
    assert pytest.approx(rho, abs=1e-6) == 1.0


def test_main_arclen_spearman_reverse_gives_negative():
    """Reverse mapping on main points gives signed -1.0 (no abs)."""
    n = 50
    # Both sides: 35 main + 15 branch, main angles are a monotone sequence
    src_angles = np.concatenate((np.linspace(0, 9, 35), np.full(15, 6.1)))
    tgt_angles = np.concatenate((np.linspace(0, 9, 35), np.full(15, 6.1)))
    src_labels = np.array([0] * 35 + [1] * 15)
    tgt_labels = np.array([0] * 35 + [1] * 15)
    # Reverse-identity: source row i matches target col (35-1-i) for i in main
    T = np.zeros((n, n))
    for i in range(35):
        T[i, 34 - i] = 1.0 / n
    for i in range(35, n):
        T[i, i] = 1.0 / n
    rho = run.main_arclen_spearman(T, src_angles, src_labels, tgt_angles, tgt_labels)
    assert pytest.approx(rho, abs=1e-6) == -1.0


# ---- geodesic arclen ----------------------------------------------------

def test_spiral_arclen_monotone_and_positive():
    thetas = np.linspace(0, 9, 20)
    s = run.spiral_arclen(thetas)
    assert np.all(np.diff(s) > 0)
    assert s[0] == 0.0
    # Total spiral arc length should be around 5.9 for r_min=0.3, r_max=1.0
    assert 5.5 < s[-1] < 6.2


def test_arclens_on_branched_manifold_are_geodesic():
    """Tail 1 and tail 2 points share a common fork-base arclen offset;
    tail 2 max arclen is strictly less than tail 1 max arclen."""
    _, arclens, labels = run.sample_branched_spiral(
        n=400, branch_frac=0.3, tail1_len=1.2, tail2_len=0.6,
        noise=0.0, seed=0,
    )
    fork_s = float(run.spiral_arclen(9.0).item())
    tail1_arclens = arclens[(labels == 0) & (arclens > fork_s - 1e-6)]
    tail2_arclens = arclens[labels == 1]
    assert tail1_arclens.max() > tail2_arclens.max() + 0.4
    # Both tail regions start at the fork
    assert abs(tail1_arclens.min() - fork_s) < 0.1
    assert abs(tail2_arclens.min() - fork_s) < 0.1


# ---- solver wrappers (6 FGW variants) -----------------------------------

EXPECTED_KEYS = {
    "T", "gw_cost", "marginal_error", "wall_s",
    "gpu_peak_gb", "iterations", "hyperparams", "solver_version",
}


def _make_small_pair():
    X, src_arclens, _ = run.sample_branched_swiss_roll(n=60, seed=0)
    Y, tgt_arclens, _ = run.sample_branched_spiral(n=80, seed=1)
    return X, Y, src_arclens, tgt_arclens


def test_run_torchgw_landmark_returns_expected_fields():
    X, Y, src_a, tgt_a = _make_small_pair()
    out = run.run_torchgw_landmark(X, Y, src_a, tgt_a, seed=0,
                                     max_iter=50, M_samples=40, n_landmarks=30)
    assert set(out.keys()) >= EXPECTED_KEYS
    assert out["T"].shape == (60, 80)
    assert out["hyperparams"]["distance_mode"] == "landmark"
    assert out["hyperparams"]["fgw_alpha"] == 0.5


def test_run_torchgw_dijkstra_returns_expected_fields():
    X, Y, src_a, tgt_a = _make_small_pair()
    out = run.run_torchgw_dijkstra(X, Y, src_a, tgt_a, seed=0,
                                     max_iter=50, M_samples=40)
    assert set(out.keys()) >= EXPECTED_KEYS
    assert out["T"].shape == (60, 80)
    assert out["hyperparams"]["distance_mode"] == "dijkstra"


def test_run_torchgw_precomputed_returns_expected_fields():
    X, Y, src_a, tgt_a = _make_small_pair()
    out = run.run_torchgw_precomputed(X, Y, src_a, tgt_a, seed=0,
                                        max_iter=50, M_samples=40)
    assert set(out.keys()) >= EXPECTED_KEYS
    assert out["T"].shape == (60, 80)
    assert out["hyperparams"]["distance_mode"] == "precomputed"


def test_run_pot_entropic_returns_expected_fields():
    X, Y, src_a, tgt_a = _make_small_pair()
    out = run.run_pot_entropic(X, Y, src_a, tgt_a, seed=0, max_iter=80)
    assert set(out.keys()) >= EXPECTED_KEYS
    assert out["T"].shape == (60, 80)
    assert out["gpu_peak_gb"] is None
    assert out["hyperparams"]["algorithm"] == "entropic"


def test_run_pot_exact_returns_expected_fields():
    X, Y, src_a, tgt_a = _make_small_pair()
    out = run.run_pot_exact(X, Y, src_a, tgt_a, seed=0, max_iter=200)
    assert set(out.keys()) >= EXPECTED_KEYS
    assert out["T"].shape == (60, 80)
    assert out["hyperparams"]["algorithm"] == "exact-CG"


def test_run_pot_bapg_returns_expected_fields():
    X, Y, src_a, tgt_a = _make_small_pair()
    out = run.run_pot_bapg(X, Y, src_a, tgt_a, seed=0, max_iter=200)
    assert set(out.keys()) >= EXPECTED_KEYS
    assert out["T"].shape == (60, 80)
    assert out["gpu_peak_gb"] is None  # POT runs on CPU
    assert out["hyperparams"]["alpha"] == 0.5


def test_pot_too_large_threshold():
    assert run.pot_too_large(6000, 7000) is True
    assert run.pot_too_large(400, 500) is False
    assert run.pot_too_large(5000, 5000) is False  # threshold is strict-greater
