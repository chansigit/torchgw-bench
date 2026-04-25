"""Downstream evaluation from an N×N GW distance matrix + ground-truth labels."""
from __future__ import annotations
import numpy as np


def eval_distance_matrix(
    D: np.ndarray,
    labels: np.ndarray,
    k_classes: int,
    knn_k: int = 5,
) -> dict:
    from sklearn.cluster import AgglomerativeClustering, SpectralClustering
    from sklearn.metrics import (
        adjusted_rand_score, normalized_mutual_info_score,
        accuracy_score, f1_score,
    )
    from sklearn.neighbors import KNeighborsClassifier

    # Ward needs vector input or precomputed with linkage='average';
    # use 'average' on precomputed distances (Ward requires Euclidean).
    ward = AgglomerativeClustering(
        n_clusters=k_classes, metric="precomputed", linkage="average"
    ).fit_predict(D)

    # Spectral on similarity = exp(-D / median(D))
    med = float(np.median(D[D > 0])) if np.any(D > 0) else 1.0
    S = np.exp(-D / max(med, 1e-12))
    spec = SpectralClustering(
        n_clusters=k_classes, affinity="precomputed",
        assign_labels="kmeans", random_state=0,
    ).fit_predict(S)

    # Leave-one-out kNN on the precomputed distance matrix
    knn = KNeighborsClassifier(n_neighbors=knn_k, metric="precomputed")
    n = D.shape[0]
    preds = np.empty(n, dtype=labels.dtype)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        D_train = D[np.ix_(mask, mask)]
        knn.fit(D_train, labels[mask])
        preds[i] = knn.predict(D[i:i+1, mask])[0]

    return {
        "ARI_ward":     float(adjusted_rand_score(labels, ward)),
        "NMI_ward":     float(normalized_mutual_info_score(labels, ward)),
        "ARI_spectral": float(adjusted_rand_score(labels, spec)),
        "NMI_spectral": float(normalized_mutual_info_score(labels, spec)),
        "knn_acc_k5":   float(accuracy_score(labels, preds)),
        "knn_macro_f1_k5": float(f1_score(labels, preds, average="macro")),
    }
