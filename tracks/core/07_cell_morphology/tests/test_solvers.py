import numpy as np
import solvers


def _two_blocks(n: int = 30):
    rng = np.random.default_rng(0)
    A = rng.uniform(size=(n, n)); A = (A + A.T) / 2; np.fill_diagonal(A, 0)
    return A, A.copy()  # identical → GW should be ~0


def test_pot_entropic_identical_is_small():
    D1, D2 = _two_blocks()
    out = solvers.gw_pair("pot-entropic-gpu", D1, D2,
                          epsilon=5e-3, M_samples=None, seed=0)
    assert out["gw"] < 1e-2
    assert out["wall_s"] > 0


def test_pot_exact_identical_is_small():
    D1, D2 = _two_blocks()
    out = solvers.gw_pair("pot-exact-gpu", D1, D2,
                          epsilon=5e-3, M_samples=None, seed=0)
    assert out["gw"] < 1e-2


def test_cajal_native_identical_is_small():
    D1, D2 = _two_blocks()
    out = solvers.gw_pair("cajal-native", D1, D2,
                          epsilon=5e-3, M_samples=None, seed=0)
    assert out["gw"] < 1e-2


def test_torchgw_identical_is_small():
    D1, D2 = _two_blocks()
    out = solvers.gw_pair("torchgw-precomputed", D1, D2,
                          epsilon=5e-3, M_samples=20, seed=0)
    # M_samples=20 on N=30 is heavy under-sampling — MC noise floor is real
    # (we deliberately use a small M to keep the test fast)
    assert out["gw"] < 1e-1


def test_cajal_full_matrix_batch():
    D_list = [_two_blocks()[0] for _ in range(4)]
    M = solvers.gw_full_matrix_cajal(D_list, num_processes=2)
    assert M.shape == (4, 4)
    assert np.allclose(M, M.T)
    assert (np.diag(M) < 1e-2).all()
