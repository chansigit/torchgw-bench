"""Unit tests for tracks/core/02_foundation_fused/run.py."""
from __future__ import annotations

import numpy as np
import pytest

import run  # type: ignore[import-not-found]  # noqa: E402 — sys.path set by conftest.py


# ---- data generators (sanity) -------------------------------------------

def test_sample_spiral_shape():
    pts, angles = run.sample_spiral(n=50, seed=0)
    assert pts.shape == (50, 2)
    assert angles.shape == (50,)


def test_sample_swiss_roll_shape():
    pts, angles = run.sample_swiss_roll(n=50, seed=1)
    assert pts.shape == (50, 3)
    assert angles.shape == (50,)


# ---- arclen_spearman returns signed rho ---------------------------------

def test_arclen_spearman_returns_signed_rho():
    """For C2 the metric is signed (no abs). Identity → +1; reverse → -1."""
    n = 50
    src_angles = np.linspace(0, 9, n)
    tgt_angles = np.linspace(0, 9, n)
    T_fwd = np.eye(n) / n
    assert pytest.approx(run.arclen_spearman(T_fwd, src_angles, tgt_angles), abs=1e-6) == 1.0
    T_rev = np.fliplr(np.eye(n)) / n
    assert pytest.approx(run.arclen_spearman(T_rev, src_angles, tgt_angles), abs=1e-6) == -1.0


# ---- feature cost matrix ------------------------------------------------

def test_build_feature_cost_shape_and_normalisation():
    src_angles = np.linspace(0, 9, 20)
    tgt_angles = np.linspace(0, 9, 30)
    M = run._build_feature_cost(src_angles, tgt_angles)
    assert M.shape == (20, 30)
    assert M.max() == pytest.approx(1.0, abs=1e-6)
    assert M.min() >= 0.0


# ---- pot_too_large ------------------------------------------------------

def test_pot_too_large():
    assert run.pot_too_large(6000, 7000, threshold=5000) is True
    assert run.pot_too_large(400, 500, threshold=5000) is False


# ---- solver wrappers ----------------------------------------------------

def test_run_torchgw_fused_returns_expected_fields():
    X, src_angles = run.sample_spiral(n=60, seed=0)
    Y, tgt_angles = run.sample_swiss_roll(n=80, seed=1)
    out = run.run_torchgw_fused(X, Y, src_angles, tgt_angles, seed=0, max_iter=50, M_samples=40, n_landmarks=30)
    assert set(out.keys()) >= {
        "T", "gw_cost", "marginal_error", "wall_s",
        "gpu_peak_gb", "iterations", "hyperparams", "solver_version",
    }
    assert out["T"].shape == (60, 80)
    assert out["wall_s"] > 0
    assert np.isfinite(out["gw_cost"])
    assert out["hyperparams"]["fgw_alpha"] == 0.5


def test_run_pot_fused_returns_expected_fields():
    X, src_angles = run.sample_spiral(n=60, seed=0)
    Y, tgt_angles = run.sample_swiss_roll(n=80, seed=1)
    out = run.run_pot_fused(X, Y, src_angles, tgt_angles, seed=0, max_iter=80)
    assert set(out.keys()) >= {
        "T", "gw_cost", "marginal_error", "wall_s",
        "gpu_peak_gb", "iterations", "hyperparams", "solver_version",
    }
    assert out["T"].shape == (60, 80)
    assert out["gpu_peak_gb"] is None
    assert np.isfinite(out["gw_cost"])
    assert out["hyperparams"]["alpha"] == 0.5
