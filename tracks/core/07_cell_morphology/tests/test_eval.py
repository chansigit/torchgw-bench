import numpy as np
import eval as track_eval


def _block_distance_matrix(n_per_class: int = 10, n_classes: int = 3,
                           sep: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    classes = np.repeat(np.arange(n_classes), n_per_class)
    points = rng.normal(size=(len(classes), 2)) + sep * classes[:, None]
    D = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
    return D, classes


def test_eval_block_recovers_clusters():
    D, y = _block_distance_matrix()
    out = track_eval.eval_distance_matrix(D, y, k_classes=3, knn_k=5)
    assert out["ARI_ward"] > 0.95
    assert out["NMI_ward"] > 0.90
    assert out["knn_acc_k5"] > 0.95


def test_eval_random_is_chance():
    rng = np.random.default_rng(1)
    D = rng.uniform(size=(30, 30)); D = (D + D.T) / 2; np.fill_diagonal(D, 0)
    y = np.repeat(np.arange(3), 10)
    out = track_eval.eval_distance_matrix(D, y, k_classes=3, knn_k=5)
    assert out["ARI_ward"] < 0.30
    assert out["knn_acc_k5"] < 0.55
