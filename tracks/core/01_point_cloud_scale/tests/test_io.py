"""Unit tests for C1 io.py — OFF parser and FPS downsampling."""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from conftest import c1_io


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STANDARD_OFF = textwrap.dedent("""\
    OFF
    4 2 0
    0.0 0.0 0.0
    1.0 0.0 0.0
    0.0 1.0 0.0
    0.0 0.0 1.0
    3 0 1 2
    3 0 1 3
""")

QUIRKY_OFF = textwrap.dedent("""\
    OFF4 2 0
    0.0 0.0 0.0
    1.0 0.0 0.0
    0.0 1.0 0.0
    0.0 0.0 1.0
    3 0 1 2
    3 0 1 3
""")


@pytest.fixture()
def standard_off_file(tmp_path):
    p = tmp_path / "standard.off"
    p.write_text(STANDARD_OFF)
    return str(p)


@pytest.fixture()
def quirky_off_file(tmp_path):
    p = tmp_path / "quirky.off"
    p.write_text(QUIRKY_OFF)
    return str(p)


# ---------------------------------------------------------------------------
# read_off tests
# ---------------------------------------------------------------------------

class TestReadOff:
    def test_standard_shape_and_dtype(self, standard_off_file):
        V = c1_io.read_off(standard_off_file)
        assert V.shape == (4, 3)
        assert V.dtype == np.float32

    def test_standard_correct_values(self, standard_off_file):
        V = c1_io.read_off(standard_off_file)
        expected = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(V, expected)

    def test_quirky_header_shape_and_dtype(self, quirky_off_file):
        V = c1_io.read_off(quirky_off_file)
        assert V.shape == (4, 3)
        assert V.dtype == np.float32

    def test_quirky_header_correct_values(self, quirky_off_file):
        V = c1_io.read_off(quirky_off_file)
        expected = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(V, expected)


# ---------------------------------------------------------------------------
# fps_downsample tests
# ---------------------------------------------------------------------------

class TestFpsDownsample:
    def test_coverage_random_cloud(self):
        """FPS subset should cover >=90% of the full cloud's diameter."""
        rng = np.random.default_rng(42)
        pts = rng.uniform(0, 1, size=(1000, 3)).astype(np.float32)

        sub = c1_io.fps_downsample(pts, 100, seed=0)
        assert sub.shape == (100, 3)
        assert sub.dtype == np.float32

        # Max pairwise distance in the full cloud (sample to speed up)
        diff_full = pts[:, None, :] - pts[None, :, :]
        max_full = np.sqrt((diff_full ** 2).sum(-1)).max()

        diff_sub = sub[:, None, :] - sub[None, :, :]
        max_sub = np.sqrt((diff_sub ** 2).sum(-1)).max()

        assert max_sub >= 0.9 * max_full, (
            f"FPS coverage too low: max_sub={max_sub:.4f}, max_full={max_full:.4f}"
        )

    def test_n_equals_N_returns_all(self):
        """When n == N, all points are returned."""
        pts = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10],
            [5, 5, 5],
        ], dtype=np.float32)

        result = c1_io.fps_downsample(pts, n=5, seed=7)
        assert result.shape == (5, 3)
        # All original points should be present (order may differ)
        for row in pts:
            assert any(np.allclose(row, r) for r in result), (
                f"Point {row} missing from result"
            )

    def test_n_greater_than_N_returns_all(self):
        """When n > N, no upsampling — return all N points."""
        pts = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0],
        ], dtype=np.float32)
        result = c1_io.fps_downsample(pts, n=10, seed=0)
        assert result.shape == (3, 3)

    def test_deterministic_given_seed(self):
        """Same seed produces same result; different seeds differ."""
        pts = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10],
            [5, 5, 5],
        ], dtype=np.float32)

        r1 = c1_io.fps_downsample(pts, n=3, seed=42)
        r2 = c1_io.fps_downsample(pts, n=3, seed=42)
        r3 = c1_io.fps_downsample(pts, n=3, seed=99)

        np.testing.assert_array_equal(r1, r2)
        assert not np.array_equal(r1, r3), "Different seeds should (likely) differ"
