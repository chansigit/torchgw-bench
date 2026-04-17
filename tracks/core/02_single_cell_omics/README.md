# C2 — Single-cell multi-omics integration

Cross-modality Gromov-Wasserstein alignment on a paired multi-omics
benchmark. Given paired RNA+ATAC measurements from the same cells
(10x PBMC 10k Multiome), split the modalities and ask: **can GW, using
only within-modality similarity structure, recover the cross-modality
correspondence?**

## Dataset

10x PBMC 10k Multiome (`pbmc_granulocyte_sorted_10k`), 11,898 cells ×
(36,601 genes + 143,887 peaks). Paired = same barcodes across modalities.

- Download: `bash fetch.sh` → `data/core_02_sc_omics/pbmc_10k_multiome.h5`
  (~184 MB)

## Preprocessing

- **RNA**: `normalize_total(1e4) → log1p → HVG(3000) → scale(max=10) → PCA(50)`
- **ATAC**: top 10k peaks by variance → TF-IDF → truncated SVD(50) → drop
  first component (depth-correlated). Standard LSI pipeline.

Both modalities end up as (n_cells × 50) float32 embedding matrices;
solvers see only these, not the underlying features.

## Solvers

Same 5 GPU FGW solvers as C3/C6:

- `torchgw-landmark`, `torchgw-dijkstra`, `torchgw-precomputed` —
  all at `ε=5e-2` (C6 finding: pure GW on non-feature-anchored data
  wants 10× the FGW default).
- `pot-entropic-gpu`, `pot-exact-gpu`

## Metrics

Paired ground truth = identity permutation (cell i in RNA is cell i in
ATAC). We report:

- **FOSCTTM** (Fraction Of Samples Closer Than True Match) — per-cell
  fraction of other cells receiving more transport mass than the true
  partner, averaged symmetrically (row + col). Random = 0.5. Perfect =
  0. This is the standard benchmark metric from SCOT/UnionCom.
- **top-k recall** at k ∈ {1, 5, 10, 50}: fraction of cells whose true
  partner is in the top-k highest-mass targets.
- `wall_s_total`, `gpu_peak_gb`, `ram_peak_gb`.

## CLI

```
python run.py --solver torchgw-dijkstra --seed 0 \
    --n-cells 2000 --out results/c2_sc/
```

## Quick reproduce

```bash
source /scratch/users/chensj16/venvs/dl2025/.venv/bin/activate
cd /scratch/users/chensj16/projects/torchgw-bench

bash tracks/core/02_single_cell_omics/fetch.sh   # ~184 MB
bash scripts/run_c2_sc.sh                         # TBD
python scripts/experiments/make_c2_sc_plots.py    # TBD
```
