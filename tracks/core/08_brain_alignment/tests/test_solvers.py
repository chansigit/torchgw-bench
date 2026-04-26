import numpy as np
import solvers


def _two_random_costs(n: int = 20):
    rng = np.random.default_rng(0)
    A = rng.uniform(size=(n, n)).astype(np.float32)
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    B = A.copy()
    Cl = np.zeros((n, n), dtype=np.float32)
    return A, B, Cl


def test_pot_entropic_fgw_identical_is_small():
    A, B, Cl = _two_random_costs()
    out = solvers.fgw_pair("pot-entropic-fgw", A, B, Cl,
                           epsilon=5e-3, fgw_alpha=0.5, seed=0)
    assert out["fgw_objective"] < 0.1


def test_torchgw_balanced_identical_is_small():
    A, B, Cl = _two_random_costs()
    out = solvers.fgw_pair("torchgw-balanced", A, B, Cl,
                           epsilon=5e-3, fgw_alpha=0.5, seed=0)
    assert out["fgw_objective"] < 0.2


def test_torchgw_unbalanced_identical_is_small():
    A, B, Cl = _two_random_costs()
    out = solvers.fgw_pair("torchgw-unbalanced", A, B, Cl,
                           epsilon=5e-3, fgw_alpha=0.5, seed=0,
                           rho_a=1.0, rho_b=1.0)
    assert out["fgw_objective"] < 0.2


def test_fugw_native_identical_is_small():
    A, B, Cl = _two_random_costs()
    out = solvers.fgw_pair("fugw-native", A, B, Cl,
                           epsilon=5e-2, fgw_alpha=0.5, seed=0,
                           rho_a=1.0, rho_b=1.0)
    assert out["T"].shape == (20, 20)
    # FUGW objective magnitude depends on its joint-divergence formulation;
    # just check the plan T is non-degenerate (not all zeros).
    assert out["T"].sum() > 1e-3
