from __future__ import annotations

"""Unit tests for tracks/core/05_word_embedding/io.py."""

import numpy as np
import pytest

import word_io  # type: ignore[import-not-found]  # noqa: E402 — sys.path + alias set by conftest.py

cosine_cost = word_io.cosine_cost
range_normalize = word_io.range_normalize
read_fasttext = word_io.read_fasttext
read_muse_dict = word_io.read_muse_dict


# ---- cosine_cost ------------------------------------------------------------

def test_cosine_cost_properties():
    """Synthetic 3-row matrix: symmetric, zero diagonal, off-diag in [0, 2]."""
    rng = np.random.default_rng(0)
    V = rng.standard_normal((3, 8)).astype(np.float32)

    C = cosine_cost(V)

    assert C.shape == (3, 3)
    assert C.dtype == np.float32

    # Diagonal must be 0
    np.testing.assert_allclose(np.diag(C), 0.0, atol=1e-6)

    # Symmetric
    np.testing.assert_allclose(C, C.T, atol=1e-6)

    # Off-diagonal in [0, 2]
    mask = ~np.eye(3, dtype=bool)
    assert C[mask].min() >= 0.0
    assert C[mask].max() <= 2.0 + 1e-5


# ---- range_normalize --------------------------------------------------------

def test_range_normalize_min_max():
    """Output min is 0 and max is 1."""
    rng = np.random.default_rng(1)
    C = rng.standard_normal((5, 5)).astype(np.float32)

    C_norm = range_normalize(C)

    assert C_norm.dtype == np.float32
    assert float(C_norm.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(C_norm.max()) == pytest.approx(1.0, abs=1e-6)


# ---- read_muse_dict ---------------------------------------------------------

def test_read_muse_dict_one_to_many(tmp_path):
    """Small fixture: hello maps to two targets, world to one."""
    fixture = tmp_path / "dict.txt"
    fixture.write_text("hello hola\nhello bonjour\nworld mundo\n", encoding="utf-8")

    d = read_muse_dict(str(fixture))

    assert set(d.keys()) == {"hello", "world"}
    assert d["hello"] == {"hola", "bonjour"}
    assert d["world"] == {"mundo"}


# ---- read_fasttext ----------------------------------------------------------

def test_read_fasttext_shape_words_dtype(tmp_path):
    """4-line fixture (header + 3 words): words list, shape, dtype."""
    fixture = tmp_path / "vecs.vec"
    fixture.write_text(
        "3 5\n"
        "foo 0.1 0.2 0.3 0.4 0.5\n"
        "bar 0.5 0.4 0.3 0.2 0.1\n"
        "baz 0.0 0.0 0.0 0.0 1.0\n",
        encoding="utf-8",
    )

    words, V = read_fasttext(str(fixture), N=3)

    assert words == ["foo", "bar", "baz"]
    assert V.shape == (3, 5)
    assert V.dtype == np.float32
