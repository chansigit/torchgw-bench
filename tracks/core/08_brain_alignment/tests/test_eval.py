import numpy as np
import eval_brain


def test_eval_perfect_alignment_is_perfect():
    rng = np.random.default_rng(0)
    F_test_a = rng.normal(size=(50, 4)).astype(np.float32)
    F_test_b = F_test_a.copy()
    n = 50
    T = np.eye(n) / n  # identity plan, row-normalized to 1/n
    out = eval_brain.eval_alignment(T, F_test_a, F_test_b)
    assert out["func_corr_holdout_mean"] > 0.99
    assert out["retrieval_top1"] == 1.0


def test_eval_random_alignment_is_chance():
    rng = np.random.default_rng(1)
    F_test_a = rng.normal(size=(50, 4)).astype(np.float32)
    F_test_b = rng.normal(size=(50, 4)).astype(np.float32)
    n = 50
    T = np.full((n, n), 1.0 / (n * n))  # uniform plan
    out = eval_brain.eval_alignment(T, F_test_a, F_test_b)
    assert abs(out["func_corr_holdout_mean"]) < 0.2
    assert out["retrieval_top1"] < 0.5
