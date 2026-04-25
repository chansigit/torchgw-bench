import numpy as np
import intracell


def test_compute_intracell_returns_square(tmp_path):
    # 5-node Y-fork: linear 0-1-2 + branch 1-3-4
    swc = tmp_path / "y.swc"
    swc.write_text(
        "1 1 0 0 0 1 -1\n"
        "2 3 1 0 0 1  1\n"
        "3 3 2 0 0 1  2\n"
        "4 3 1 1 0 1  2\n"
        "5 3 1 2 0 1  4\n"
    )
    D = intracell.compute_intracell(swc, n_per_cell=4, cache_dir=tmp_path / "cache")
    assert D.shape == (4, 4)
    assert np.allclose(D, D.T, atol=1e-6)
    assert (np.diag(D) == 0).all()
    # cache hit on second call must return identical bytes
    D2 = intracell.compute_intracell(swc, n_per_cell=4, cache_dir=tmp_path / "cache")
    assert np.array_equal(D, D2)
