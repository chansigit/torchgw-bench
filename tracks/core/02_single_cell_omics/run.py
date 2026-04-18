#!/usr/bin/env python
"""C2 single-cell multi-omics integration — cross-modality GW alignment.

Given a paired multi-omics dataset (10x PBMC 10k Multiome, where each
cell has both RNA and ATAC measurements), we split the modalities,
preprocess each independently, and ask: given only the within-modality
similarity structures (no shared features), can GW recover the
cross-modality correspondence?

Pipeline design choices (informed by SCOT, Demetci 2022 and diagnostic
sweeps on this dataset):

  * RNA: scanpy's HVG → scale → PCA(50).
  * ATAC: variance-top peaks → TF-IDF → truncated SVD(50), drop first
    component (depth-correlated).
  * **L2-normalise** each embedding row — so Euclidean kNN ≈ correlation
    kNN for neighbour selection.
  * Structural cost: **binary kNN connectivity** graph (not weighted
    Euclidean). Dijkstra on binary adjacency yields **hop-count
    geodesic** — empirically ~3× better FOSCTTM than weighted Euclidean
    geodesic (0.27 vs 0.71 at N=1000). Weighted-edge geodesics on
    L2-normalised 50-dim vectors have too little spread to be
    informative.
  * Consequence: torchgw's internal `landmark` / `dijkstra` distance
    modes — which compute *weighted* Euclidean geodesics from the input
    coordinates — produce FOSCTTM > 0.5 (anti-correlated) on this data.
    They remain in the benchmark as a cautionary comparison. The only
    torchgw configuration that works here is `precomputed` mode fed
    with a SCOT-style cost matrix.

Ground truth is the identity permutation (cell i in RNA is the same
cell as cell i in ATAC). This is the standard evaluation protocol
from SCOT (Demetci 2020), UnionCom (Cao 2020), and Pamona (Cao 2022).

Metrics:
  - FOSCTTM (Fraction Of Samples Closer Than True Match): for each
    query cell in modality A, fraction of targets in modality B
    that receive more transport mass than the true partner. Lower
    is better; random is 0.5; perfect is 0.
  - top-k recall: fraction of cells whose true partner is in the
    top-k highest-mass targets.
  - wall time, GPU peak memory.

Solvers are the same five GPU variants as C6 (3 torchgw + 2 POT-GPU).
Pure GW (fgw_alpha=1.0) by default; --fgw-alpha exposes the FGW blend
if the user wants to test with linear cost (e.g. shared PCA anchor).
"""
from __future__ import annotations

import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Any

import numpy as np

TRACK_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRACK_ROOT.parents[2]
DATA_ROOT = REPO_ROOT / "data" / "core_02_sc_omics"


# ---- record helper ------------------------------------------------------

def build_record(track: str, solver: str, seed: int, subset: str) -> dict:
    import socket
    import datetime as _dt
    return {
        "track": track, "solver": solver, "seed": seed,
        "solver_version": None, "subset": subset,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "status": "ok", "error": None,
        "dataset": {}, "hyperparams": {},
        "metrics": {"correctness": {}, "task": {}, "efficiency": {}, "stability": {}},
        "artifacts": {},
    }


# ---- data loading & preprocessing ---------------------------------------

def load_multiome(h5_path: Path):
    """Load 10x Multiome HDF5 and split into two AnnData objects
    (RNA, ATAC) over the same set of cells. Returns (adata_rna, adata_atac)."""
    import scanpy as sc
    import anndata as ad
    adata = sc.read_10x_h5(str(h5_path), gex_only=False)
    adata.var_names_make_unique()
    is_rna = adata.var["feature_types"] == "Gene Expression"
    adata_rna = ad.AnnData(X=adata.X[:, is_rna.values],
                             obs=adata.obs.copy(),
                             var=adata.var.loc[is_rna].copy())
    adata_atac = ad.AnnData(X=adata.X[:, ~is_rna.values],
                              obs=adata.obs.copy(),
                              var=adata.var.loc[~is_rna].copy())
    return adata_rna, adata_atac


def preprocess_rna(adata, n_top_genes: int = 3000, n_comps: int = 50):
    """Standard scanpy: normalize_total → log1p → HVG → scale → PCA.
    Returns (n_cells, n_comps) float32 embedding."""
    import scanpy as sc
    ad = adata.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes)
    ad = ad[:, ad.var["highly_variable"]].copy()
    sc.pp.scale(ad, max_value=10)
    sc.pp.pca(ad, n_comps=n_comps, zero_center=True, svd_solver="arpack")
    return ad.obsm["X_pca"].astype(np.float32)


def _select_top_peaks(adata, n_top_peaks: int):
    """Select n_top_peaks peaks by per-peak variance in raw counts."""
    import scipy.sparse as sp
    X = adata.X
    if sp.issparse(X):
        var_per_peak = np.asarray((X.multiply(X)).mean(axis=0)).ravel() \
                        - np.asarray(X.mean(axis=0)).ravel() ** 2
    else:
        var_per_peak = X.var(axis=0)
    top_idx = np.argsort(-var_per_peak)[:n_top_peaks]
    return adata[:, top_idx].copy()


def preprocess_atac_lsi(adata, n_top_peaks: int = 10000, n_comps: int = 50):
    """LSI: top peaks → TF-IDF → truncated SVD, drop first (depth) component.

    Historical ATAC pipeline (Signac, Seurat). Works but tends to retain
    depth-related structure in later components."""
    import scipy.sparse as sp
    from sklearn.utils.extmath import randomized_svd
    ad = _select_top_peaks(adata, n_top_peaks)
    X = ad.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    tf = X.multiply(1.0 / (np.asarray(X.sum(axis=1)) + 1e-9))
    idf = np.log(1 + X.shape[0] / (np.asarray((X > 0).sum(axis=0)).ravel() + 1e-9))
    tfidf = tf.multiply(idf).tocsr()
    U, s, _ = randomized_svd(tfidf, n_components=n_comps + 1, random_state=0)
    return (U[:, 1:] * s[1:]).astype(np.float32)


def preprocess_atac_lda(adata, n_top_peaks: int = 10000, n_topics: int = 50,
                          max_iter: int = 20):
    """sklearn LDA (online VB): top peaks → binarise → LDA(n_topics).
    Fast but lower-quality topics than cisTopic. Kept for ablation."""
    import scipy.sparse as sp
    from sklearn.decomposition import LatentDirichletAllocation
    ad = _select_top_peaks(adata, n_top_peaks)
    X = ad.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X_bin = (X > 0).astype(np.float32)
    print(f"[C2]   fitting LDA(n_topics={n_topics}, max_iter={max_iter}) on "
           f"{X_bin.shape} binarised matrix...", flush=True)
    lda = LatentDirichletAllocation(
        n_components=n_topics, max_iter=max_iter,
        learning_method="online", batch_size=512,
        random_state=0, n_jobs=-1, verbose=0,
    )
    emb = lda.fit_transform(X_bin)
    return emb.astype(np.float32)


def preprocess_atac_cistopic(adata, n_top_peaks: int = 10000,
                                n_topics: int = 50, n_iter: int = 500,
                                n_cores: int = 4):
    """cisTopic LDA (collapsed Gibbs sampling). Matches SCOT+ exactly.

    Calls an external R subprocess in the `cistopic` micromamba env to
    fit cisTopic, reads the resulting cells × topics probability matrix
    back into numpy. The R side uses cisTopic's `runCGSModels` then
    `modelMatSelection` to pull the cell topic proportions."""
    import subprocess
    import tempfile
    import scipy.sparse as sp
    from scipy.io import mmwrite

    ad = _select_top_peaks(adata, n_top_peaks)
    X = ad.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X_bin = (X > 0).astype(np.int32)             # cells x peaks
    peak_ids = ad.var.index.tolist()             # chr:start-end
    X_pc = X_bin.T.tocsr()                       # peaks x cells for cisTopic

    with tempfile.TemporaryDirectory(prefix="cistopic_") as td:
        mtx_path = Path(td) / "input.mtx"
        peaks_path = Path(td) / "peak_ids.txt"
        csv_path = Path(td) / "output.csv"
        mmwrite(str(mtx_path), X_pc)
        peaks_path.write_text("\n".join(peak_ids))

        script = TRACK_ROOT / "cistopic_lda.R"
        cmd = [
            "micromamba", "run", "-n", "cistopic",
            "Rscript", str(script),
            str(mtx_path), str(peaks_path), str(csv_path),
            str(n_topics), str(n_iter), str(n_cores),
        ]
        print(f"[C2]   invoking cisTopic R subprocess "
               f"(n_topics={n_topics}, n_iter={n_iter}, n_cores={n_cores})...",
               flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        for line in proc.stdout.splitlines():
            if "[cisTopic.R]" in line:
                print(line, flush=True)
        if proc.returncode != 0:
            print("--- stderr ---")
            print(proc.stderr[-2000:])
            raise RuntimeError("cisTopic R subprocess failed")
        emb = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
    return emb


def preprocess_atac(adata, n_top_peaks: int = 10000, n_comps: int = 50,
                      method: str = "cistopic"):
    """Dispatch to the chosen ATAC preprocessing path."""
    if method == "lsi":
        return preprocess_atac_lsi(adata, n_top_peaks=n_top_peaks,
                                      n_comps=n_comps)
    elif method == "lda":
        return preprocess_atac_lda(adata, n_top_peaks=n_top_peaks,
                                      n_topics=n_comps)
    elif method == "cistopic":
        return preprocess_atac_cistopic(adata, n_top_peaks=n_top_peaks,
                                          n_topics=n_comps)
    else:
        raise ValueError(f"unknown atac method: {method}")


def subsample_cells(V_rna: np.ndarray, V_atac: np.ndarray, n_cells: int, seed: int):
    """Uniform random subsample — returns the same indices for both
    modalities so paired ground truth is preserved."""
    rng = np.random.default_rng(seed)
    n = V_rna.shape[0]
    n_cells = min(n_cells, n)
    idx = rng.choice(n, n_cells, replace=False)
    return V_rna[idx], V_atac[idx], idx


# ---- metrics ------------------------------------------------------------

def foscttm(T: np.ndarray, V_src: np.ndarray, V_tgt: np.ndarray) -> float:
    """Fraction Of Samples Closer Than True Match — SCOT/UnionCom
    formulation based on barycentric projection distance.

    Procedure:
      1. Project each source cell into target space via T:
         proj_src = (T * n) @ V_tgt         (barycentric; `* n` keeps
         the scale — T rows sum to 1/n under uniform marginals)
      2. For each i, compute Euclidean distance from proj_src[i] to
         every V_tgt[j].
      3. FOSCTTM[i] = fraction of j != gt_i with dist[i, j] < dist[i, gt_i].
      4. Repeat symmetrically (project V_tgt back via T.T) and average.

    Random FOSCTTM = 0.5. Perfect = 0.
    """
    T = np.asarray(T, dtype=np.float64)
    V_src = np.asarray(V_src, dtype=np.float64)
    V_tgt = np.asarray(V_tgt, dtype=np.float64)
    n = T.shape[0]
    assert T.shape[0] == T.shape[1] == V_src.shape[0] == V_tgt.shape[0]

    # Normalise rows/cols so they sum to 1
    row_norm = T.sum(axis=1, keepdims=True) + 1e-30
    col_norm = T.sum(axis=0, keepdims=True) + 1e-30

    def _avg(proj, ref):
        # Distance of proj[i] to every ref[j]
        d = np.linalg.norm(proj[:, None, :] - ref[None, :, :], axis=2)
        # GT for paired data: identity — so the "true match" distance is d[i,i]
        diag = np.diag(d)
        closer = (d < diag[:, None]).sum(axis=1)   # excludes self via strict <
        return (closer / max(n - 1, 1)).mean()

    # Forward: src → tgt space
    proj_src = (T / row_norm) @ V_tgt
    f_fwd = _avg(proj_src, V_tgt)
    # Backward: tgt → src space
    proj_tgt = (T.T / col_norm.T) @ V_src
    f_bwd = _avg(proj_tgt, V_src)
    return float(0.5 * (f_fwd + f_bwd))


def top_k_recall(T: np.ndarray, ks=(1, 5, 10, 50)) -> dict[int, float]:
    """For each row, check whether the true partner (diagonal index) is
    among the top-k entries (by plan mass)."""
    T = np.asarray(T)
    n = T.shape[0]
    rank_desc = np.argsort(-T, axis=1)
    out: dict[int, float] = {}
    for k in ks:
        hit = (rank_desc[:, :k] == np.arange(n)[:, None]).any(axis=1)
        out[k] = float(hit.mean())
    return out


# ---- memory sampler (verbatim from 03_branched / 06_shape) --------------

class _RSSSampler:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil  # noqa: F401
            self._have_psutil = True
        except ImportError:
            self._have_psutil = False

    def __enter__(self):
        if not self._have_psutil:
            return self
        import psutil
        self._proc = psutil.Process(os.getpid())
        self.peak = self._proc.memory_info().rss
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)

    def _loop(self):
        import psutil
        while not self._stop.is_set():
            try:
                cur = self._proc.memory_info().rss
            except psutil.Error:
                break
            if cur > self.peak:
                self.peak = cur
            self._stop.wait(self.interval)


def _reset_tracking(seed: int):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    return torch, use_cuda


def _build_metrics(wall_preprocess, wall_solve, ram_peak_bytes, use_cuda):
    import torch
    gpu_peak_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3) if use_cuda else None
    )
    return {
        "wall_s_preprocess": float(wall_preprocess),
        "wall_s_solve":      float(wall_solve),
        "wall_s_total":      float(wall_preprocess + wall_solve),
        "gpu_peak_gb":       gpu_peak_gb,
        "ram_peak_gb":       ram_peak_bytes / (1024 ** 3),
    }


# ---- structural cost (SCOT-style by default) ---------------------------

def l2_normalize(V: np.ndarray) -> np.ndarray:
    """L2-normalise rows of V. SCOT's default input preprocessing."""
    n = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    return (V / n).astype(np.float32)


def knn_geodesic_matrix(V: np.ndarray, k: int | None = None,
                          metric: str = "correlation",
                          mode: str = "connectivity") -> np.ndarray:
    """SCOT-style structural cost. Build a kNN graph with the chosen
    metric (default Pearson correlation) in connectivity mode (binary
    0/1 adjacency) or distance mode (weighted edges). Dijkstra on that
    graph gives a geodesic distance matrix (hop-counts for connectivity,
    real distance for distance mode). Final matrix is max-normalised to
    [0, 1].

    Defaults match SCOT.align() (metric='correlation', mode='connectivity',
    k capped at min(0.2 * n, 50)).
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path
    n = V.shape[0]
    if k is None:
        k = min(max(int(0.2 * n), 5), 50)
    G = kneighbors_graph(V, n_neighbors=k, metric=metric, mode=mode,
                          include_self=False)
    G = G.maximum(G.T)                               # symmetrise
    D = shortest_path(G, method="D", directed=False)
    if not np.all(np.isfinite(D)):
        finite_max = D[np.isfinite(D)].max()
        D[~np.isfinite(D)] = 2.0 * finite_max
    return (D / (D.max() + 1e-12)).astype(np.float32)


# ---- torchgw solver wrappers (pure GW) ----------------------------------

def _finalize_torchgw(T, log, meta, hyperparams):
    log_d = log if isinstance(log, dict) else {}
    T_np = T.detach().cpu().numpy() if hasattr(T, "detach") else np.asarray(T)
    marginal_error = float(np.max(np.abs(T_np.sum(axis=1) - 1.0 / T_np.shape[0])))
    import torchgw as _torchgw
    rec = {
        "T": T_np,
        "gw_cost": float(log_d.get("gw_cost", float("nan"))),
        "marginal_error": marginal_error,
        "iterations": int(log_d.get("n_iter", log_d.get("iterations", 0))),
        "hyperparams": hyperparams,
        "solver_version": f"torchgw=={getattr(_torchgw, '__version__', 'unknown')}",
    }
    rec.update(meta)
    rec["wall_s"] = meta["wall_s_total"]
    return rec


def run_torchgw_landmark(V_src, V_tgt, seed=0, epsilon=5e-2,
                           M_samples=80, max_iter=300, k=10, n_landmarks=50):
    from torchgw import sampled_gw
    torch, use_cuda = _reset_tracking(seed)
    with _RSSSampler() as sampler:
        t0 = time.perf_counter()
        T, log = sampled_gw(  # type: ignore[misc]
            V_src, V_tgt,
            distance_mode="landmark", mixed_precision=True,
            M=M_samples, epsilon=epsilon, max_iter=max_iter,
            k=k, n_landmarks=n_landmarks,
            log=True, verbose=False,
        )
        if use_cuda:
            torch.cuda.synchronize()
        t_solve = time.perf_counter() - t0
    meta = _build_metrics(0.0, t_solve, sampler.peak, use_cuda)
    return _finalize_torchgw(T, log, meta, {
        "M_samples": M_samples, "epsilon": epsilon, "max_iter": max_iter,
        "k": k, "n_landmarks": n_landmarks, "fgw_alpha": 1.0,
        "distance_mode": "landmark", "mixed_precision": True,
    })


def run_torchgw_dijkstra(V_src, V_tgt, seed=0, epsilon=5e-2,
                           M_samples=80, max_iter=300, k=10):
    from torchgw import sampled_gw
    torch, use_cuda = _reset_tracking(seed)
    with _RSSSampler() as sampler:
        t0 = time.perf_counter()
        T, log = sampled_gw(  # type: ignore[misc]
            V_src, V_tgt,
            distance_mode="dijkstra", mixed_precision=True,
            M=M_samples, epsilon=epsilon, max_iter=max_iter, k=k,
            log=True, verbose=False,
        )
        if use_cuda:
            torch.cuda.synchronize()
        t_solve = time.perf_counter() - t0
    meta = _build_metrics(0.0, t_solve, sampler.peak, use_cuda)
    return _finalize_torchgw(T, log, meta, {
        "M_samples": M_samples, "epsilon": epsilon, "max_iter": max_iter,
        "k": k, "fgw_alpha": 1.0,
        "distance_mode": "dijkstra", "mixed_precision": True,
    })


def run_torchgw_precomputed(V_src, V_tgt, seed=0, epsilon=5e-2,
                              M_samples=80, max_iter=300):
    from torchgw import sampled_gw
    torch, use_cuda = _reset_tracking(seed)
    with _RSSSampler() as sampler:
        t_prep_start = time.perf_counter()
        dist_source = knn_geodesic_matrix(V_src)
        dist_target = knn_geodesic_matrix(V_tgt)
        dist_source /= (dist_source.max() + 1e-12)
        dist_target /= (dist_target.max() + 1e-12)
        n_src, n_tgt = V_src.shape[0], V_tgt.shape[0]
        p = np.full(n_src, 1.0 / n_src, dtype=np.float64)
        q = np.full(n_tgt, 1.0 / n_tgt, dtype=np.float64)
        t_prep = time.perf_counter() - t_prep_start
        t0 = time.perf_counter()
        T, log = sampled_gw(  # type: ignore[misc]
            X_source=V_src, X_target=V_tgt, p=p, q=q,
            distance_mode="precomputed",
            dist_source=dist_source, dist_target=dist_target,
            mixed_precision=True,
            M=M_samples, epsilon=epsilon, max_iter=max_iter,
            log=True, verbose=False,
        )
        if use_cuda:
            torch.cuda.synchronize()
        t_solve = time.perf_counter() - t0
    meta = _build_metrics(t_prep, t_solve, sampler.peak, use_cuda)
    return _finalize_torchgw(T, log, meta, {
        "M_samples": M_samples, "epsilon": epsilon, "max_iter": max_iter,
        "fgw_alpha": 1.0,
        "distance_mode": "precomputed", "mixed_precision": True,
    })


# ---- POT GPU wrappers (pure GW) -----------------------------------------

def _pot_setup_gpu(V_src, V_tgt, seed, dtype="float32"):
    import torch
    device = torch.device("cuda")
    tdtype = torch.float64 if dtype == "float64" else torch.float32
    C1_np = knn_geodesic_matrix(V_src)
    C2_np = knn_geodesic_matrix(V_tgt)
    C1 = torch.as_tensor(C1_np ** 2, device=device, dtype=tdtype)
    C2 = torch.as_tensor(C2_np ** 2, device=device, dtype=tdtype)
    C1 /= (C1.max() + 1e-12)
    C2 /= (C2.max() + 1e-12)
    p = torch.full((V_src.shape[0],), 1.0 / V_src.shape[0], device=device, dtype=tdtype)
    q = torch.full((V_tgt.shape[0],), 1.0 / V_tgt.shape[0], device=device, dtype=tdtype)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return C1, C2, p, q


def _finalize_pot(T_and_log, meta, hyperparams):
    import ot
    pot_log: dict[str, Any] = {}
    if isinstance(T_and_log, tuple):
        T, raw = T_and_log
        if isinstance(raw, dict):
            pot_log = raw
    else:
        T = T_and_log
    T_np = T.detach().cpu().numpy().astype(np.float64) if hasattr(T, "detach") else np.asarray(T, dtype=np.float64)
    marginal_error = float(np.max(np.abs(T_np.sum(axis=1) - 1.0 / T_np.shape[0])))

    def _as_float(v):
        if hasattr(v, "detach"):
            return float(v.detach().cpu().item())
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    gw_cost = _as_float(pot_log.get("gw_dist", float("nan")))
    err_list = pot_log.get("err") or []
    iterations = len(err_list) if err_list else -1
    rec = {
        "T": T_np,
        "gw_cost": gw_cost,
        "marginal_error": marginal_error,
        "iterations": iterations,
        "hyperparams": hyperparams,
        "solver_version": f"pot=={getattr(ot, '__version__', 'unknown')}",
    }
    rec.update(meta)
    rec["wall_s"] = meta["wall_s_total"]
    return rec


def _run_pot_gw_gpu(V_src, V_tgt, seed, algo_fn, algo_kwargs, hyperparams,
                      dtype="float32"):
    import torch
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    with _RSSSampler() as sampler:
        t_prep_start = time.perf_counter()
        C1, C2, p, q = _pot_setup_gpu(V_src, V_tgt, seed, dtype=dtype)
        if use_cuda:
            torch.cuda.synchronize()
        t_prep = time.perf_counter() - t_prep_start
        t0 = time.perf_counter()
        T_and_log = algo_fn(C1, C2, p, q, **algo_kwargs)
        if use_cuda:
            torch.cuda.synchronize()
        t_solve = time.perf_counter() - t0
    meta = _build_metrics(t_prep, t_solve, sampler.peak, use_cuda)
    return _finalize_pot(T_and_log, meta, hyperparams)


def run_pot_entropic_gpu(V_src, V_tgt, seed=0, epsilon=5e-3, max_iter=100,
                           tol=1e-9):
    import ot.gromov as otgw
    return _run_pot_gw_gpu(
        V_src, V_tgt, seed,
        otgw.entropic_gromov_wasserstein,
        dict(loss_fun="square_loss", epsilon=epsilon,
              max_iter=max_iter, tol=tol, log=True, verbose=False),
        {"epsilon": epsilon, "max_iter": max_iter, "tol": tol,
         "loss_fun": "square_loss", "algorithm": "entropic",
         "backend": "gpu"},
    )


def run_pot_exact_gpu(V_src, V_tgt, seed=0, max_iter=500, tol=1e-6):
    import ot.gromov as otgw
    return _run_pot_gw_gpu(
        V_src, V_tgt, seed,
        otgw.gromov_wasserstein,
        dict(loss_fun="square_loss",
              max_iter=max_iter, tol_rel=tol, tol_abs=tol, log=True),
        {"max_iter": max_iter, "tol": tol, "loss_fun": "square_loss",
         "algorithm": "exact-CG", "backend": "gpu"},
    )


# ---- main ---------------------------------------------------------------

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="C2 single-cell cross-modality GW alignment")
    ap.add_argument("--solver", required=True, choices=[
        "torchgw-landmark", "torchgw-dijkstra", "torchgw-precomputed",
        "pot-entropic-gpu", "pot-exact-gpu",
    ])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-cells", type=int, default=2000,
                    help="Number of cells to subsample per modality.")
    ap.add_argument("--n-comps", type=int, default=50,
                    help="Dimensionality of the per-modality embedding.")
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--atac-method", choices=["lsi", "lda", "cistopic"],
                    default="cistopic",
                    help="ATAC embedding: 'lsi' = TF-IDF + truncated SVD, "
                         "'lda' = sklearn online LDA (fast but lower quality), "
                         "'cistopic' = cisTopic collapsed-Gibbs LDA via R "
                         "subprocess (default, matches SCOT+)")
    ap.add_argument("--M-samples", type=int, default=None,
                    help="torchgw sampled_gw per-iter cost rows. Default "
                         "80 is scalability-tuned; for N<=5000 the gap-"
                         "diagnostic suggests M=N//2 (see c2_msamples_sweep).")
    args = ap.parse_args()

    rec = build_record("core/02_single_cell_omics", args.solver, args.seed, "full")
    rec["dataset"] = {
        "name": f"pbmc_10k_multiome_n{args.n_cells}",
        "n_cells": args.n_cells, "n_comps": args.n_comps,
    }
    _tag = f"__{args.tag}" if args.tag else ""
    out_path = args.out / (
        f"core_02_single_cell_omics__{args.solver}"
        f"__n{args.n_cells}__seed{args.seed}{_tag}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        h5 = DATA_ROOT / "pbmc_10k_multiome.h5"
        if not h5.exists():
            raise FileNotFoundError(
                f"dataset not found at {h5}; run tracks/core/02_single_cell_omics/fetch.sh first"
            )
        cache = DATA_ROOT / f"embeddings_n_comps{args.n_comps}_atac_{args.atac_method}.npz"
        if cache.exists():
            print(f"[C2] loading cached embeddings {cache.name}", flush=True)
            z = np.load(cache)
            V_rna_full, V_atac_full = z["V_rna"], z["V_atac"]
        else:
            print(f"[C2] loading multiome from {h5}", flush=True)
            adata_rna, adata_atac = load_multiome(h5)
            print(f"[C2] RNA shape  {adata_rna.shape}", flush=True)
            print(f"[C2] ATAC shape {adata_atac.shape}", flush=True)
            V_rna_full = preprocess_rna(adata_rna, n_comps=args.n_comps)
            V_atac_full = preprocess_atac(adata_atac, n_comps=args.n_comps,
                                             method=args.atac_method)
            np.savez(cache, V_rna=V_rna_full, V_atac=V_atac_full)
            print(f"[C2] cached embeddings → {cache.name}", flush=True)
        assert V_rna_full.shape[0] == V_atac_full.shape[0]
        V_rna, V_atac, idx = subsample_cells(V_rna_full, V_atac_full,
                                                args.n_cells, args.seed)
        # SCOT-style input: L2-normalise each modality before alignment.
        V_rna = l2_normalize(V_rna)
        V_atac = l2_normalize(V_atac)
        print(f"[C2] subsampled to n={V_rna.shape[0]} (l2-normalised)", flush=True)

        solver_fns = {
            "torchgw-landmark":    run_torchgw_landmark,
            "torchgw-dijkstra":    run_torchgw_dijkstra,
            "torchgw-precomputed": run_torchgw_precomputed,
            "pot-entropic-gpu":    run_pot_entropic_gpu,
            "pot-exact-gpu":       run_pot_exact_gpu,
        }
        fn = solver_fns[args.solver]
        kwargs: dict = {}
        if args.epsilon is not None and args.solver != "pot-exact-gpu":
            kwargs["epsilon"] = args.epsilon
        if args.max_iter is not None:
            kwargs["max_iter"] = args.max_iter
        if args.M_samples is not None and args.solver.startswith("torchgw"):
            kwargs["M_samples"] = args.M_samples
        print(f"[C2] solving with {args.solver}  kwargs={kwargs}", flush=True)
        result = fn(V_rna, V_atac, seed=args.seed, **kwargs)

        T = result["T"]
        foscttm_score = foscttm(T, V_rna, V_atac)
        recalls = top_k_recall(T)
        rec["hyperparams"] = result["hyperparams"]
        rec["solver_version"] = result["solver_version"]
        rec["metrics"]["correctness"] = {
            "gw_cost": result["gw_cost"],
            "marginal_error": result["marginal_error"],
        }
        rec["metrics"]["task"] = {
            "foscttm": foscttm_score,
            "top1":   recalls[1],
            "top5":   recalls[5],
            "top10":  recalls[10],
            "top50":  recalls[50],
        }
        rec["metrics"]["efficiency"] = {
            "wall_s":            result["wall_s_total"],
            "wall_s_preprocess": result["wall_s_preprocess"],
            "wall_s_solve":      result["wall_s_solve"],
            "wall_s_total":      result["wall_s_total"],
            "gpu_peak_gb":       result["gpu_peak_gb"],
            "ram_peak_gb":       result["ram_peak_gb"],
            "iterations":        result["iterations"],
        }
    except Exception as e:
        rec["status"] = "fail"
        rec["error"] = f"{type(e).__name__}: {e}"
        out_path.write_text(json.dumps(rec, indent=2))
        raise
    else:
        out_path.write_text(json.dumps(rec, indent=2))
        print(f"[C2] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
