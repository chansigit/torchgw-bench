"""Compute the per-cell intracell geodesic distance matrix using CAJAL.

CAJAL's `get_sample_pts_geodesic` uses a deterministic step-size sampling
algorithm — no RNG involvement — so the `(swc, n_per_cell)` pair fully
determines the output D_i. The cache key therefore omits seed; downstream
solver-stage seeds still vary at the GW step.
"""
from __future__ import annotations
import hashlib
import pathlib
import numpy as np


def _cache_key(swc_path: pathlib.Path, n_per_cell: int) -> str:
    h = hashlib.sha256()
    h.update(swc_path.read_bytes())
    h.update(f"|n={n_per_cell}".encode())
    return h.hexdigest()[:16]


def compute_intracell(
    swc_path: str | pathlib.Path,
    n_per_cell: int,
    cache_dir: str | pathlib.Path,
) -> np.ndarray:
    """Return the n×n intracell geodesic distance matrix for one SWC file."""
    swc_path = pathlib.Path(swc_path)
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(swc_path, n_per_cell)
    cache_file = cache_dir / f"{swc_path.stem}__n{n_per_cell}__{key}.npy"
    if cache_file.exists():
        return np.load(cache_file)

    from cajal.sample_swc import read_swc, icdm_geodesic
    from scipy.spatial.distance import squareform

    _forest, trees = read_swc(str(swc_path))
    if not trees:
        raise ValueError(f"no parseable trees in {swc_path}")
    # Pick the largest tree by node count — handles SWCs with detached fragments.
    tree = max(trees.values(), key=lambda t: len(t.subtree_list()) if hasattr(t, 'subtree_list') else 1)

    condensed, _adjacency = icdm_geodesic(tree, n_per_cell)
    D = squareform(condensed).astype(np.float64)
    np.save(cache_file, D)
    return D
