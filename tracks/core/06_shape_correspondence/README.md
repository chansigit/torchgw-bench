# C6 — TACO Shape Correspondence

Pure GW matching between two meshes of the same TACO class in different
poses. Tests torchgw's geometric-correspondence regime — the setting
where the landmark / dijkstra distance modes were designed to shine.

## Dataset

[TACO](https://zenodo.org/records/14066437) (2024): 9 classes (cat,
centaur, david, dog, gorilla, horse, michael, victoria, wolf), 80 meshes
total, **420 cross-pose pairs with vertex-level ground-truth** stored as
`Pi` permutations in `.mat` files.

- Download: `bash fetch.sh` → `data/core_06_shape/taco/`
- Sizes: ~27k verts (cat, centaur), ~52k (horse), ~7-12k (human
  classes). v1 subsamples uniformly to `n_source = n_target = 2000`.

## Solvers (v1)

All five are pure GW (`fgw_alpha = 1.0`, no linear feature cost). The
structural distance is intrinsic geodesic on a kNN graph over the
subsampled point cloud.

- `torchgw-landmark` — torchgw with landmark-approximated kNN geodesic
- `torchgw-dijkstra` — torchgw with exact kNN-graph Dijkstra
- `torchgw-precomputed` — torchgw fed a precomputed geodesic matrix
- `pot-entropic-gpu` — POT `entropic_gromov_wasserstein` on GPU
- `pot-exact-gpu` — POT `gromov_wasserstein` (conditional gradient) on GPU

CPU POT variants are excluded (O(N²) memory + poor scaling; the
03_branched writeup documents this). POT-BAPG is excluded from v1 (known
fp32 underflow; we proved the fp64 fix on 03_branched but keep the v1
solver list minimal).

## Metrics

- `mean_err_normalised`, `median_err_normalised` — geodesic distance from
  predicted match to GT match on the target mesh, normalised by
  the target's geodesic diameter. Lower is better. 0.0 is perfect,
  1.0 is random.
- `accuracy_curve` — `[(τ, fraction within τ × diameter)]` at
  τ ∈ {0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25}. The standard
  shape-matching benchmark plot is this curve per-solver.
- `wall_s_{preprocess, solve, total}`, `gpu_peak_gb`, `ram_peak_gb`.

## CLI

```
python run.py --solver torchgw-landmark --pair cat0,cat1 --seed 0 \
    --n-source 2000 --n-target 2000 --out results/c6_shape/
```

## Quick reproduce

```bash
source /scratch/users/chensj16/venvs/dl2025/.venv/bin/activate
cd /scratch/users/chensj16/projects/torchgw-bench

bash tracks/core/06_shape_correspondence/fetch.sh   # ~120 MB
bash scripts/run_c6_shape.sh                         # TBD
python scripts/experiments/make_c6_shape_plots.py    # TBD
```

## Scope notes (v1 vs future)

- v1: one subsample size (N=2000), pure GW (no HKS/SHOT features).
- v2 candidates: scale sweep, FGW with spectral descriptors (HKS/WKS),
  ICP baseline, FAUST pairs, full-resolution meshes for torchgw-landmark
  only (POT variants will OOM).
