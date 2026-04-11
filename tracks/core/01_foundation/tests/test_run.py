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
