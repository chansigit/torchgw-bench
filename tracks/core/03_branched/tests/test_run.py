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
    # Branch fraction should be approximately 0.3
    assert abs(labels.sum() - 30) <= 1


def test_sample_branched_swiss_roll_shape_and_labels():
    pts, angles, labels = run.sample_branched_swiss_roll(n=100, branch_frac=0.3, seed=1)
    assert pts.shape == (100, 3)
    assert angles.shape == (100,)
    assert labels.shape == (100,)
    assert set(np.unique(labels).tolist()) == {0, 1}
    assert abs(labels.sum() - 30) <= 1


def test_branched_spiral_branch_points_near_theta_branch():
    """Branch points should start near the spiral position at theta_branch."""
    pts, angles, labels = run.sample_branched_spiral(n=200, branch_frac=0.3,
                                                      theta_branch=6.0, seed=0)
    # Branch angles should all be >= theta_branch (== theta_branch + s, s >= 0.02)
    branch_angles = angles[labels == 1]
    assert branch_angles.min() >= 6.0
    assert branch_angles.max() <= 6.0 + 0.4 + 0.01  # branch_len=0.4 + small tolerance


# ---- metrics ------------------------------------------------------------

def test_branch_accuracy_perfect_identity():
    """When source and target have matching labels in same order, identity T gives 1.0."""
    n = 50
    src_labels = np.array([0] * 35 + [1] * 15)
    tgt_labels = np.array([0] * 35 + [1] * 15)
    T = np.eye(n) / n
    assert pytest.approx(run.branch_accuracy(T, src_labels, tgt_labels), abs=1e-9) == 1.0


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


# ---- solver wrapper -----------------------------------------------------

def test_run_torchgw_landmark_returns_expected_fields():
    X, _, _ = run.sample_branched_spiral(n=60, seed=0)
    Y, _, _ = run.sample_branched_swiss_roll(n=80, seed=1)
    out = run.run_torchgw_landmark(X, Y, seed=0, max_iter=50, M=40, n_landmarks=30)
    assert set(out.keys()) >= {
        "T", "gw_cost", "marginal_error", "wall_s",
        "gpu_peak_gb", "iterations", "hyperparams", "solver_version",
    }
    assert out["T"].shape == (60, 80)
    assert out["wall_s"] > 0
