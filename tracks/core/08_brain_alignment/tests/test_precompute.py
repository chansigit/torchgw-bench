import numpy as np
import precompute


def test_geodesic_matrix_small_mesh(tmp_path):
    # 5-vertex linear chain mesh: 0-1-2-3-4
    verts = np.array([[i, 0.0, 0.0] for i in range(5)], dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.int32)
    D = precompute.geodesic_matrix(verts, faces, sparse=False)
    assert D.shape == (5, 5)
    assert np.allclose(D, D.T, atol=1e-6)
    assert (np.diag(D) == 0).all()
    # 0 → 4 should be ~4 along the chain
    assert 3.5 < D[0, 4] < 4.5


def test_feature_cost_matrix_returns_square():
    rng = np.random.default_rng(0)
    F_a = rng.normal(size=(20, 4)).astype(np.float32)
    F_b = rng.normal(size=(25, 4)).astype(np.float32)
    C = precompute.feature_cost_matrix(F_a, F_b)
    assert C.shape == (20, 25)
    assert C.dtype == np.float64
    # cosine cost is in [0, 2]
    assert C.min() >= 0 and C.max() <= 2.0
