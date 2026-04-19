"""Unit tests for C1 pair.py — rotation-based pair generation."""
from __future__ import annotations

import numpy as np

from conftest import c1_pair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_cloud(n: int = 100, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=(n, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMakePair:
    def test_output_shapes(self):
        P = _random_cloud(100)
        src, tgt, R = c1_pair.make_pair(P, n=50, seed=0)
        assert src.shape == (50, 3), f"source shape {src.shape}"
        assert tgt.shape == (50, 3), f"target shape {tgt.shape}"
        assert R.shape == (3, 3),    f"R shape {R.shape}"

    def test_rotation_correctness(self):
        """target must equal source @ R.T within float32 tolerance."""
        P = _random_cloud(100)
        src, tgt, R = c1_pair.make_pair(P, n=50, seed=0)
        reconstructed = src @ R.T
        np.testing.assert_allclose(tgt, reconstructed, atol=1e-5,
                                   err_msg="target != source @ R.T")

    def test_determinism(self):
        """Two calls with the same seed must produce identical output."""
        P = _random_cloud(100)
        src1, tgt1, R1 = c1_pair.make_pair(P, n=50, seed=42)
        src2, tgt2, R2 = c1_pair.make_pair(P, n=50, seed=42)
        np.testing.assert_array_equal(src1, src2)
        np.testing.assert_array_equal(tgt1, tgt2)
        np.testing.assert_array_equal(R1, R2)

    def test_proper_rotation_determinant(self):
        """det(R) must be ≈ 1.0 (no reflections)."""
        P = _random_cloud(100)
        _, _, R = c1_pair.make_pair(P, n=50, seed=0)
        det = float(np.linalg.det(R))
        assert abs(det - 1.0) < 1e-5, f"det(R) = {det:.6f}, expected ~1.0"

    def test_different_seeds_differ(self):
        """Different seeds should produce different rotations (with high probability)."""
        P = _random_cloud(100)
        _, _, R1 = c1_pair.make_pair(P, n=50, seed=0)
        _, _, R2 = c1_pair.make_pair(P, n=50, seed=1)
        assert not np.allclose(R1, R2), "Different seeds produced identical rotations"
