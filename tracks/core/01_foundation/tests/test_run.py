"""Unit tests for helpers defined at module level in run.py.

These tests exercise pure helper functions only. main() is covered by the
integration smoke test, not here.
"""
from __future__ import annotations

import numpy as np
import pytest

import run  # noqa: E402  — sys.path set by conftest.py


# ---- data generators ----------------------------------------------------

def test_sample_spiral_shape_and_determinism():
    X, angles = run.sample_spiral(n=400, seed=0)
    assert X.shape == (400, 2)
    assert angles.shape == (400,)
    assert np.isfinite(X).all()
    # Re-seeding reproduces the same array
    X2, angles2 = run.sample_spiral(n=400, seed=0)
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(angles, angles2)


def test_sample_spiral_different_seeds_differ():
    X0, _ = run.sample_spiral(n=400, seed=0)
    X1, _ = run.sample_spiral(n=400, seed=1)
    assert not np.array_equal(X0, X1)


def test_sample_swiss_roll_shape():
    Y, angles = run.sample_swiss_roll(n=500, seed=1)
    assert Y.shape == (500, 3)
    assert angles.shape == (500,)
    assert np.isfinite(Y).all()


# ---- metric: Spearman on arclength --------------------------------------

def test_arclen_spearman_perfect_identity():
    """Diagonal transport plan + equal angles -> Spearman = 1."""
    n = 50
    src_angles = np.linspace(0, 9, n)
    tgt_angles = np.linspace(0, 9, n)
    T = np.eye(n) / n
    rho = run.arclen_spearman(T, src_angles, tgt_angles)
    assert pytest.approx(rho, abs=1e-6) == 1.0


def test_arclen_spearman_reverse_permutation():
    """A reverse-diagonal transport plan -> Spearman = -1."""
    n = 50
    src_angles = np.linspace(0, 9, n)
    tgt_angles = np.linspace(0, 9, n)
    T = np.fliplr(np.eye(n)) / n
    rho = run.arclen_spearman(T, src_angles, tgt_angles)
    assert pytest.approx(rho, abs=1e-6) == -1.0


# ---- host info ----------------------------------------------------------

def test_get_host_info_has_required_keys():
    host = run.get_host_info()
    assert "gpu" in host
    assert "torch" in host
    assert "cuda" in host
    # torch must be a version string like "2.x.y"
    assert isinstance(host["torch"], str)
    assert host["torch"][0].isdigit()


# ---- record builder -----------------------------------------------------

def test_build_record_minimum_fields():
    rec = run.build_record(
        track="core/01_foundation",
        solver="torchgw-landmark",
        seed=0,
        subset="full",
    )
    for key in ("track", "solver", "seed", "subset", "timestamp", "host",
                "status", "error", "dataset", "hyperparams", "metrics", "artifacts"):
        assert key in rec, f"missing key: {key}"
    assert rec["status"] == "ok"
    assert rec["error"] is None
    assert rec["metrics"] == {"correctness": {}, "task": {}, "efficiency": {}, "stability": {}}


def test_build_record_timestamp_is_utc_iso8601():
    import re
    rec = run.build_record(track="t", solver="s", seed=0, subset="full")
    # e.g. 2026-04-10T14:32:00Z
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", rec["timestamp"])
