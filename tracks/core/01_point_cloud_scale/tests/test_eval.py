"""Unit tests for C1 eval.py — correspondence accuracy, Chamfer distance."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import c1_eval


# ---------------------------------------------------------------------------
# correspondence_accuracy
# ---------------------------------------------------------------------------

class TestCorrespondenceAccuracy:
    def test_identity_plan(self):
        """Identity transport plan → perfect accuracy = 1.0."""
        N = 20
        T = np.eye(N)
        assert c1_eval.correspondence_accuracy(T) == 1.0

    def test_flipped_plan(self):
        """Swapping rows 0 and 1 → accuracy = (N-2)/N."""
        N = 20
        T = np.eye(N).copy()
        T[[0, 1]] = T[[1, 0]]   # swap rows → argmax(row 0) = 1, argmax(row 1) = 0
        expected = (N - 2) / N
        result = c1_eval.correspondence_accuracy(T)
        assert abs(result - expected) < 1e-9, f"got {result}, expected {expected}"

    def test_all_wrong(self):
        """All rows point to wrong column → 0.0."""
        N = 5
        T = np.roll(np.eye(N), shift=1, axis=1)   # every argmax is off by 1
        assert c1_eval.correspondence_accuracy(T) == 0.0


# ---------------------------------------------------------------------------
# correspondence_recall_at_k
# ---------------------------------------------------------------------------

class TestCorrespondenceRecallAtK:
    def test_identity_k1(self):
        """Identity plan, k=1 → recall = 1.0."""
        N = 10
        T = np.eye(N)
        assert c1_eval.correspondence_recall_at_k(T, k=1) == 1.0

    def test_offby2_k5(self):
        """Off-by-2 cyclic shift: correct index is at rank 3 → k=5 finds it, k=1 misses."""
        N = 20
        # Build plan: row i has peak at (i+2)%N, but correct is i
        T = np.zeros((N, N))
        for i in range(N):
            T[i, (i + 2) % N] = 1.0   # wrong peak
            T[i, i] = 0.5              # correct is second-largest
        # k=5 should recall all rows (correct col is second-highest → in top 5)
        assert c1_eval.correspondence_recall_at_k(T, k=5) == 1.0
        # k=1 should miss all rows (peak is at wrong col)
        assert c1_eval.correspondence_recall_at_k(T, k=1) == 0.0


# ---------------------------------------------------------------------------
# chamfer_distance
# ---------------------------------------------------------------------------

class TestChamferDistance:
    def test_identical_clouds(self):
        """Identical point clouds → Chamfer distance = 0."""
        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 1, (50, 3)).astype(np.float32)
        cd = c1_eval.chamfer_distance(pts, pts)
        assert abs(cd) < 1e-10, f"CD of identical clouds = {cd}"

    def test_disjoint_shifted_clouds(self):
        """Clouds shifted by 10 units → large positive Chamfer distance."""
        rng = np.random.default_rng(1)
        pts = rng.uniform(0, 1, (50, 3)).astype(np.float32)
        shifted = pts + 10.0
        cd = c1_eval.chamfer_distance(pts, shifted)
        assert cd > 50.0, f"Expected large CD for shifted clouds, got {cd}"

    def test_non_negative(self):
        rng = np.random.default_rng(2)
        A = rng.uniform(0, 5, (30, 3)).astype(np.float32)
        B = rng.uniform(0, 5, (30, 3)).astype(np.float32)
        assert c1_eval.chamfer_distance(A, B) >= 0.0


# ---------------------------------------------------------------------------
# barycentric_project
# ---------------------------------------------------------------------------

class TestBarycentricProject:
    def test_identity_plan_returns_target(self):
        """Identity transport plan → projected points == target."""
        rng = np.random.default_rng(3)
        N = 15
        V_tgt = rng.uniform(0, 1, (N, 3)).astype(np.float32)
        T = np.eye(N)
        proj = c1_eval.barycentric_project(T, V_tgt)
        np.testing.assert_allclose(proj, V_tgt, atol=1e-6,
                                   err_msg="Identity plan should return target unchanged")

    def test_uniform_plan_returns_centroid(self):
        """Uniform plan → each projected point is the centroid of target."""
        rng = np.random.default_rng(4)
        N = 10
        V_tgt = rng.uniform(0, 1, (N, 3)).astype(np.float32)
        T = np.ones((N, N))                    # uniform; will be row-normalised
        proj = c1_eval.barycentric_project(T, V_tgt)
        centroid = V_tgt.mean(axis=0)
        for row in proj:
            np.testing.assert_allclose(row, centroid, atol=1e-5)
