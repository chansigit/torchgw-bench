# C7 — Cell Morphology vs CAJAL: Design Spec

**Date:** 2026-04-25
**Status:** approved (brainstorming → spec); awaiting user review before plan
**Track slot:** `tracks/core/07_cell_morphology` (C4 left for spec's medical-imaging track)

## 1. One-liner

CAJAL ([readthedocs](https://cajal.readthedocs.io/)) treats cell morphology as
"cell-cell GW distance on intracell geodesic distance matrices." We reuse
CAJAL's preprocessing verbatim and **swap only the pairwise-GW step** to
torchgw, then check (a) at CAJAL's default sample size whether torchgw is in
its small-N death zone, and (b) whether raising sample size lets torchgw
overtake on speed without losing downstream type-recovery quality.

## 2. Why this track

- C2 already ran the same "swap only the solver" pattern against SCOT
  (cisTopic preprocessing + swap GW solver) and produced the strongest
  controlled comparison in the bench. C7 ports that template to a different
  domain (neuron morphology) and a different scaling regime ("many tiny GW"
  instead of "few large GW").
- All five existing core tracks are "few large GW" regimes. C7 is the only
  track that probes torchgw on **N_per_cell = 50** — predicted death zone
  per the C5 SNR argument and C2 M_samples finding. A negative result here
  is itself a deployment-rule contribution.

## 3. Pipeline and swap point

```
SWC files
   │
   ▼  CAJAL: stratified sample N_per_cell points per neuron
   │
   ▼  CAJAL: per-cell intracell geodesic distance matrix D_i  (cached)
   │
   ▼  ── SWAP POINT ──  pairwise GW(D_i, D_j) over all (i, j)
   │
   ▼  N_cells × N_cells GW distance matrix
   │
   ▼  downstream: hierarchical + spectral clustering, leave-one-out kNN, UMAP
```

**Controlled-variable invariant**: every solver consumes the *same* `D_i`
matrices that CAJAL produced. No solver is allowed to rebuild its own cost
matrix (so `torchgw-landmark` and `torchgw-dijkstra` are explicitly excluded
— they would change the cost matrix and break the controlled comparison).

## 4. Datasets (two stages)

### Stage A — sanity (~1 week)
NeuroMorpho.org subset, 200–500 cells across 2–4 morphologically distinct
classes (e.g. cortical pyramidal / cortical basket / cerebellar Purkinje).
Pinned by an explicit cell-id manifest (`stage_a_manifest.txt`) committed to
the repo, **not** "fetch the first N from class X" — reproducibility.

Success criterion: pipeline runs end-to-end; ARI > 0.8 with cajal-native
solver on this small clean dataset. If ARI ≪ 0.8, the pipeline itself is
broken; stop and debug before Stage B.

### Stage B — benchmark (~1 week)
Allen Brain Atlas Cell Types Database (CTDB), ~1000 cells with multimodal
type labels (transcriptomic + electrophysiology + morphology). Use the
publicly released morphology-labelled subset; pin cell IDs by manifest.

Hardware budget: full Stage B should fit in ~24 h on a single H100, dominated
by `cajal-native` at N_per_cell=1000.

**Out of scope (deliberate)**: BBP m-types (declined), EM segmentation
meshes, cross-species transfer. All deferred to potential follow-ups.

## 5. Solvers (4)

| Solver | Role |
|---|---|
| `cajal-native` | literature reference. `cajal.run.compute_gw_distance_matrix` at default settings, CPU + multiprocessing. Internal backend (POT exact vs entropic) probed at install time and documented in writeup. |
| `pot-entropic-gpu` | strips out the CPU↔GPU axis from the POT↔torchgw axis. If `cajal-native` is POT entropic on CPU, this is the clean within-POT GPU control. |
| `pot-exact-gpu` | quality reference. Only viable at N_per_cell ≤ 200; mark fail/skip beyond. |
| `torchgw-precomputed` | the contender. Must run with explicit `M_samples` per the C2 finding (default 80 is the death zone). Use `M = max(min(N_per_cell, 1000), 3·N_per_cell/4)`. |

**Excluded (and why)**: `torchgw-landmark`, `torchgw-dijkstra` — they
construct their own cost matrices and would violate the controlled-variable
invariant. Noted as "future track: torchgw-native cost variants" if asked.

## 6. Evaluation metrics (B + C + D + E)

For each (stage, solver, N_per_cell, seed) cell, produce a JSON record with:

**B — clustering quality**
- `ARI_ward`, `NMI_ward` from Ward hierarchical clustering at K = ground-
  truth class count
- `ARI_spectral`, `NMI_spectral` from spectral clustering on
  exp(-D / median(D))

**C — kNN classification**
- `knn_acc_k5`, `knn_macro_f1_k5` from leave-one-out kNN on the distance
  matrix (k = 5)

**D — efficiency**
- `wall_full_matrix_s` (wall time to build the full N×N matrix)
- `wall_per_pair_ms` (mean per-pair GW solve time)
- `gpu_peak_gb` for GPU solvers; `cpu_peak_gb` for cajal-native
- `n_pairs_failed` and per-failure reason (OOM, NaN, timeout)

**E — qualitative**
- One UMAP figure per (stage, solver, N_per_cell=best) coloured by
  ground-truth type. Saved to `docs/figures/c7_umap_<stage>_<solver>.png`.

## 7. Experiment matrix

```
N_per_cell    ∈ {50, 200, 500, 1000}
solvers       ∈ {cajal-native, pot-entropic-gpu, pot-exact-gpu, torchgw-precomputed}
seeds         ∈ {0, 1, 2}
stages        = {A: ~300 cells, B: ~1000 cells}
```

Skips:
- `pot-exact-gpu` skipped at N_per_cell > 200 (CG memory)
- All solvers at Stage B × N_per_cell=1000 are the long pole — schedule first

GW solves total:
- Stage A: 4 × 4 × 3 × (300² / 2) ≈ 2.2 M
- Stage B: 4 × 4 × 3 × (1000² / 2) ≈ 24 M

Per-pair time scales: ~ms at N_per_cell=50, ~s at N_per_cell=1000.

## 8. File layout

```
tracks/core/07_cell_morphology/
├── README.md
├── fetch.sh                  # NeuroMorpho REST + Allen CTDB → SWC files
├── stage_a_manifest.txt      # pinned NeuroMorpho cell IDs + class labels
├── stage_b_manifest.txt      # pinned Allen CTDB cell IDs + class labels
├── io.py                     # SWC reader (thin wrap on CAJAL's reader)
├── intracell.py              # CAJAL call to compute per-cell D_i, on-disk cache
├── run.py                    # main: --solver --n-per-cell --seed --stage {A,B}
├── eval.py                   # ARI/NMI/kNN/UMAP from N×N distance matrix
└── tests/
    ├── test_io.py
    └── test_eval.py          # synthetic distance matrix → known clustering

scripts/run_c7_bench.sh                    # stage × solver × N × seed sweep
scripts/experiments/make_c7_plots.py       # headline figures
docs/experiments/2026-04-25-c7-cell-morphology.md   # writeup (post-bench)
docs/figures/c7_*.png
```

## 9. Environment

- New env `c7_morph` (or extend `dl2025` if `pip install cajal` is clean):
  CAJAL pulls in `pot`, `numpy`, `networkx`, `pathos` for multiprocessing.
- `torchgw` and `pot-gpu` already in `dl2025`. If CAJAL conflicts on POT
  version, prefer separate env to avoid breaking C2/C3/C5/C6.
- Bootstrap added to `scripts/bootstrap_envs.sh`.

## 10. Engineering caveats (acknowledged, not engineered around)

1. **No batched small-GW API in torchgw.** torchgw on GPU runs per-pair
   serially; CAJAL on CPU parallelizes pairs across cores. At N_per_cell=50,
   GPU launch overhead may dominate the per-pair solve. Document honestly,
   do not build a batching layer.
2. **`cajal-native` backend is unknown until install.** Probe at install
   time; report concretely in the writeup ("CAJAL's default backend is POT-X
   with ε=Y"). If it turns out to be POT exact CPU, we lose one of the GPU
   controls (pot-exact-gpu becomes the within-POT control), which is fine.
3. **NeuroMorpho subset reproducibility.** `fetch.sh` must download by
   explicit ID, not by class query. Class queries return time-varying result
   sets.

## 11. Success criteria (what the writeup will claim)

- **Stage A reproduces the design**: cajal-native achieves ARI > 0.8 on the
  hand-picked clean subset.
- **Sample-size threshold quantified**: a clear N_per_cell value is
  identified above which torchgw-precomputed beats cajal-native on
  full-matrix wall time without dropping ARI / kNN-acc by more than ε
  (target ε = 0.02 absolute).
- **Negative result honestly reported** if torchgw never wins: this becomes
  a deployment red line ("torchgw is not the right tool for many-tiny-GW
  workloads; use POT or stay on CAJAL native"), and the writeup is built
  around that finding instead.
- **Cross-track synthesis updated**: the C7 row added to the table in
  `docs/experiments/README.md`, and the decision rule extended with a
  many-small-GW vs few-large-GW axis.

## 12. Explicit non-goals

- Not a CAJAL replacement. We do not re-implement preprocessing.
- Not an evaluation of CAJAL's preprocessing choices (sampling, intracell
  geodesic algorithm).
- Not cross-species, not EM mesh data, not BBP m-types.
- No batched-small-GW engineering on torchgw.
- No tuning of CAJAL's internal hyperparameters beyond what defaults give.
