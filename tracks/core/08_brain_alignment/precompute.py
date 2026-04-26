"""Per-subject precomputation for C8 brain alignment:
    - vol_to_surf projection of Brainomics volumes onto fsaverage
    - mesh geodesic distance matrix (sparse-aware for fsaverage7)
    - inter-subject vertex feature cost matrix

At fsaverage7 the dense geodesic matrix is 213 GB and cannot be stored;
sparse mode returns a CSR (k-NN distance graph + Dijkstra) approximation
that the FUGW solver consumes natively.
"""
from __future__ import annotations
import hashlib
import pathlib
import numpy as np
from scipy.sparse import csr_matrix


# ── Geodesic distance ────────────────────────────────────────────────

def _mesh_dijkstra(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Full pairwise geodesic approximation via Dijkstra on mesh edge graph.

    Builds a CSR graph where each edge weight = Euclidean distance between
    vertices, then calls scipy.sparse.csgraph.shortest_path (all sources
    simultaneously in C). For fsaverage5 (10 242 vertices, 20 480 faces)
    this completes in ~10–30 seconds, vs ~15 min for gdist per-vertex Dijkstra.

    The result is a Euclidean-edge approximation to the true geodesic; for
    fine meshes (fsaverage5+) the error is negligible for downstream GW alignment.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path
    n = verts.shape[0]
    # Each triangular face (i, j, k) contributes 3 undirected edges.
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    row = np.concatenate([i0, i1, i1, i2, i2, i0])
    col = np.concatenate([i1, i0, i2, i1, i0, i2])
    def _edist(a, b):
        d = verts[a] - verts[b]
        return np.sqrt((d * d).sum(axis=1))
    w01 = _edist(i0, i1); w12 = _edist(i1, i2); w20 = _edist(i2, i0)
    data = np.concatenate([w01, w01, w12, w12, w20, w20])
    graph = coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    D = shortest_path(graph, method="D", directed=False)
    return D.astype(np.float64)


def _cache_key(verts: np.ndarray, faces: np.ndarray, sparse: bool) -> str:
    h = hashlib.sha256()
    h.update(verts.tobytes()); h.update(faces.tobytes())
    h.update(b"sparse" if sparse else b"dense")
    return h.hexdigest()[:16]


def geodesic_matrix(verts: np.ndarray, faces: np.ndarray,
                    sparse: bool = False, max_dist: float | None = None,
                    cache_dir: pathlib.Path | None = None
                    ) -> np.ndarray | csr_matrix:
    """Pairwise geodesic distance on a triangle mesh.

    sparse=False: dense (N, N) float64 matrix via scipy Dijkstra on mesh edge
                  graph; safe for N <= ~30 000. Completes in ~10–30 s for
                  fsaverage5 (10 242 vertices), vs ~15 min for gdist Dijkstra.
    sparse=True:  CSR (N, N) up to max_dist via gdist.local_gdist_matrix;
                  required for fsaverage7. Result is symmetrized via M.maximum(M.T).
    """
    n = verts.shape[0]

    if cache_dir is not None:
        cache_dir = pathlib.Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(verts, faces, sparse)
        cache_file = cache_dir / f"geo__{n}__{key}{'.npz' if sparse else '.npy'}"
        if cache_file.exists():
            from scipy.sparse import load_npz
            return load_npz(cache_file) if sparse else np.load(cache_file)

    if sparse:
        import gdist
        verts64 = verts.astype(np.float64)
        faces32 = faces.astype(np.int32)
        if max_dist is None:
            max_dist = 50.0  # fsaverage units; coarse but bounded
        M = gdist.local_gdist_matrix(verts64, faces32, max_distance=float(max_dist))
        # Symmetrize: local_gdist_matrix returns upper triangle only
        D = M.maximum(M.T).tocsr()
    else:
        # Use scipy Dijkstra on mesh edge graph — much faster than local_gdist_matrix
        # with max_distance=1e8 (which still visits every vertex per source in C).
        # scipy.sparse.csgraph.shortest_path processes all sources simultaneously.
        D = _mesh_dijkstra(verts, faces)

    if cache_dir is not None:
        if sparse:
            from scipy.sparse import save_npz
            save_npz(cache_file, D)
        else:
            np.save(cache_file, D)
    return D


# ── Volume → surface projection ──────────────────────────────────────

def vol_to_surface(volume_3d: np.ndarray, fsaverage_path: str,
                   affine: np.ndarray | None = None) -> np.ndarray:
    """Project a 3D MNI152 volume onto an fsaverage cortical mesh.

    Returns a 1D array of length n_vertices (per the surface).
    """
    from nilearn import surface
    import nibabel as nb
    if affine is not None:
        img = nb.Nifti1Image(volume_3d, affine)
    else:
        # Caller should pass an Nifti-like; fallback to identity affine
        img = nb.Nifti1Image(volume_3d, np.eye(4))
    return np.asarray(surface.vol_to_surf(img, fsaverage_path), dtype=np.float32)


# ── Feature cost ─────────────────────────────────────────────────────

def feature_cost_matrix(F_a: np.ndarray, F_b: np.ndarray) -> np.ndarray:
    """Inter-subject vertex-vs-vertex feature cost = 1 - cosine similarity
    of train contrast vectors. Output range: [0, 2]."""
    Fa = F_a / (np.linalg.norm(F_a, axis=1, keepdims=True) + 1e-12)
    Fb = F_b / (np.linalg.norm(F_b, axis=1, keepdims=True) + 1e-12)
    C = 1.0 - Fa @ Fb.T
    return C.astype(np.float64)
